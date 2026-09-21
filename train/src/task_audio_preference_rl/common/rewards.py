from __future__ import annotations

import os
from contextlib import redirect_stdout
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import PathsConfig, RewardConfig
from .fad_reward import FADConstraintReward
from .fd_reward import ProjectedFDConstraintReward
from .trajectory import balanced_group_advantage


def _mono(waveform: Tensor) -> Tensor:
    if waveform.ndim == 3:
        waveform = waveform.mean(dim=1)
    if waveform.ndim != 2:
        raise ValueError("waveform must have shape [B,L] or [B,C,L]")
    return waveform.float()


def _peak_normalize(waveform: Tensor) -> Tensor:
    peak = waveform.abs().amax(dim=1, keepdim=True).clamp_min(1.0e-8)
    return waveform * (10.0 ** (-1.0 / 20.0) / peak)


class SpectralRepetitionReward(nn.Module):
    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        min_lag_seconds: float = 0.25,
        max_lag_seconds: float = 2.0,
    ) -> None:
        super().__init__()
        import torchaudio

        self.sample_rate = int(sample_rate)
        self.hop_length = 256
        self.min_lag = max(1, round(min_lag_seconds * sample_rate / self.hop_length))
        self.max_lag = max(
            self.min_lag, round(max_lag_seconds * sample_rate / self.hop_length)
        )
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            win_length=1024,
            hop_length=self.hop_length,
            n_mels=64,
            power=2.0,
        )

    @torch.inference_mode()
    def forward(self, waveform: Tensor) -> Tensor:
        waveform = _mono(waveform)
        self.mel.to(waveform.device)
        features = torch.log1p(self.mel(waveform)).transpose(1, 2)
        features = F.normalize(
            features - features.mean(dim=1, keepdim=True), dim=-1, eps=1.0e-6
        )
        values: list[Tensor] = []
        maximum = min(self.max_lag, features.shape[1] - 1)
        for lag in range(self.min_lag, maximum + 1):
            values.append((features[:, :-lag] * features[:, lag:]).sum(-1).mean(-1))
        if not values:
            return waveform.new_zeros(waveform.shape[0])
        # Ignore negative correlation and penalize the strongest repeated lag.
        return torch.stack(values, dim=1).amax(dim=1).clamp_min(0.0)


class CLAPReward:
    def __init__(self, checkpoint: str | Path, device: torch.device) -> None:
        from task_audio.data.wavcaps_crop10 import _load_clap_model

        self.device = device
        self.model = _load_clap_model(Path(checkpoint), str(device))

    @torch.inference_mode()
    def __call__(self, waveform: Tensor, captions: Sequence[str]) -> Tensor:
        import torchaudio.functional as AF

        audio = _peak_normalize(_mono(waveform)).to(self.device)
        if audio.shape[1] != 160_000:
            audio = F.interpolate(
                audio[:, None], size=160_000, mode="linear", align_corners=False
            )[:, 0]
        audio48 = AF.resample(audio, 16_000, 48_000)
        # Preserve the official 16-bit roundtrip used by stable-audio-metrics.
        audio48 = (audio48.clamp(-1, 1) * 32767.0).to(torch.int16).float() / 32767.0
        audio_embedding = self.model.get_audio_embedding_from_data(
            x=audio48, use_tensor=True
        )
        text_embedding = self.model.get_text_embedding(list(captions), use_tensor=True)
        return F.cosine_similarity(
            audio_embedding.float(), text_embedding.float(), dim=-1, eps=1.0e-8
        )


class HTSATFusedCLAPReward:
    def __init__(self, device: torch.device) -> None:
        from transformers import ClapModel, ClapProcessor

        name = "laion/clap-htsat-fused"
        self.device = device
        self.model = (
            ClapModel.from_pretrained(name, local_files_only=True).to(device).eval()
        )
        self.processor = ClapProcessor.from_pretrained(name, local_files_only=True)

    @torch.inference_mode()
    def __call__(self, waveform: Tensor, captions: Sequence[str]) -> Tensor:
        import torchaudio.functional as AF

        audio = AF.resample(_mono(waveform), 16_000, 48_000).cpu().numpy()
        audio_inputs = self.processor(
            audios=[row for row in audio],
            sampling_rate=48_000,
            return_tensors="pt",
            padding=True,
        )
        text_inputs = self.processor(
            text=list(captions), return_tensors="pt", padding=True, truncation=True
        )
        audio_inputs = {
            key: value.to(self.device) for key, value in audio_inputs.items()
        }
        text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}
        audio_features = F.normalize(
            self.model.get_audio_features(**audio_inputs).float(), dim=-1
        )
        text_features = F.normalize(
            self.model.get_text_features(**text_inputs).float(), dim=-1
        )
        return (audio_features * text_features).sum(dim=-1)


class PaSSTReward:
    def __init__(self, cache_path: str | Path, device: torch.device) -> None:
        from hear21passt.base import get_basic_model

        path = Path(cache_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"missing AudioCaps PaSST cache: {path}; run prepare-passt-cache first"
            )
        try:
            loaded = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            loaded = torch.load(path, map_location="cpu")
        if not isinstance(loaded, dict) or not loaded:
            raise ValueError("PaSST reference cache must be a non-empty mapping")
        self.reference = {
            str(key): torch.as_tensor(value).float().cpu()
            for key, value in loaded.items()
        }
        self.device = device
        with open(os.devnull, "w") as sink, redirect_stdout(sink):
            self.model = get_basic_model(mode="logits").to(device).eval()

    @torch.inference_mode()
    def probabilities(self, waveform: Tensor) -> Tensor:
        import torchaudio.functional as AF
        from evaluation.run_passt_batched import PatchPasstStft

        wave = AF.resample(_peak_normalize(_mono(waveform)), 16_000, 32_000)
        target = 320_000
        if wave.shape[1] < target:
            wave = F.pad(wave, (0, target - wave.shape[1]))
        else:
            wave = wave[:, :target]
        with PatchPasstStft():
            logits = self.model(wave.to(self.device))
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        return F.softmax(torch.as_tensor(logits).float(), dim=-1)

    @torch.inference_mode()
    def __call__(self, waveform: Tensor, item_ids: Sequence[str]) -> Tensor:
        generated = self.probabilities(waveform)
        missing = [value for value in item_ids if str(value) not in self.reference]
        if missing:
            raise KeyError(f"PaSST cache is missing IDs: {missing[:8]}")
        reference = torch.stack(
            [self.reference[str(value)] for value in item_ids], dim=0
        ).to(device=generated.device)
        # Same orientation as the repository's TangoFlux metric.
        return F.kl_div(
            (reference + 1.0e-6).log(), generated, reduction="none", log_target=False
        ).sum(dim=-1)


class BalancedAudioReward:
    def __init__(
        self,
        config: RewardConfig,
        paths: PathsConfig,
        *,
        device: torch.device,
        group_size: int,
    ) -> None:
        self.config = config
        self.group_size = int(group_size)
        if config.clap_backend == "630k":
            self.clap = CLAPReward(paths.clap_checkpoint, device)
        elif config.clap_backend == "htsat_fused":
            self.clap = HTSATFusedCLAPReward(device)
        else:
            self.clap = None
        self.passt = (
            PaSSTReward(paths.passt_reference_cache, device)
            if config.passt_weight > 0
            else None
        )
        self.repetition = (
            SpectralRepetitionReward(
                min_lag_seconds=config.repetition_min_lag_seconds,
                max_lag_seconds=config.repetition_max_lag_seconds,
            ).to(device)
            if config.repetition_weight > 0
            else None
        )
        self.fd = (
            ProjectedFDConstraintReward(
                paths.panns_fd_cache,
                paths.panns_repo,
                paths.panns_checkpoint_dir,
                device,
                projection_dim=config.fd_projection_dim,
                queue_capacity=config.fd_queue_capacity,
                target_scale=config.fd_target_scale,
                dual_lr=config.fd_dual_lr,
                lambda_init=config.fd_lambda_init,
                lambda_max=config.fd_lambda_max,
                exact_every=config.fd_exact_every,
                embedding_batch_size=config.fd_embedding_batch_size,
            )
            if config.fd_enabled
            else None
        )
        self.fad = (
            FADConstraintReward(
                paths.vggish_fad_cache,
                paths.vggish_hub_dir,
                device,
                queue_capacity=config.fad_queue_capacity,
                target_scale=config.fad_target_scale,
                dual_lr=config.fad_dual_lr,
                lambda_init=config.fad_lambda_init,
                lambda_max=config.fad_lambda_max,
                exact_every=config.fad_exact_every,
                embedding_batch_size=config.fad_embedding_batch_size,
            )
            if config.fad_enabled
            else None
        )
        self.calls = 0

    @torch.inference_mode()
    def __call__(
        self,
        waveform: Tensor,
        captions: Sequence[str],
        item_ids: Sequence[str],
    ) -> dict[str, Tensor]:
        device = waveform.device
        self.calls += 1
        clap = (
            waveform.new_zeros(waveform.shape[0])
            if self.clap is None
            else self.clap(waveform, captions).to(device)
        )
        passt = (
            waveform.new_zeros(waveform.shape[0])
            if self.passt is None
            else self.passt(waveform, item_ids).to(device)
        )
        repetition = (
            waveform.new_zeros(waveform.shape[0])
            if self.repetition is None
            else self.repetition(waveform).to(device)
        )
        empty_constraint = {
            "marginal": waveform.new_zeros(waveform.shape[0]),
            "proxy": waveform.new_zeros(waveform.shape[0]),
            "target": waveform.new_zeros(waveform.shape[0]),
            "lambda": waveform.new_zeros(waveform.shape[0]),
            "violation": waveform.new_zeros(waveform.shape[0]),
        }
        fd = empty_constraint if self.fd is None else self.fd(waveform)
        fad_active = self.fad is not None and (
            (self.calls - 1) % self.config.fad_interval == 0
        )
        fad = (
            empty_constraint
            if self.fad is None
            else (
                self.fad(waveform)
                if fad_active
                else self.fad.status(waveform.shape[0], waveform.device)
            )
        )
        advantage = balanced_group_advantage(
            clap,
            passt,
            repetition,
            group_size=self.group_size,
            clap_weight=self.config.clap_weight,
            passt_weight=self.config.passt_weight,
            repetition_weight=self.config.repetition_weight,
            fd_marginal=fd["marginal"],
            fd_weight=float(fd["lambda"][0].item()),
            fad_marginal=fad["marginal"],
            fad_weight=float(fad["lambda"][0].item()),
            eps=self.config.eps,
        )
        return {
            "clap": clap,
            "passt_kl": passt,
            "repetition": repetition,
            "fd_marginal": fd["marginal"],
            "fd_proxy": fd["proxy"],
            "fd_target": fd["target"],
            "fd_lambda": fd["lambda"],
            "fd_violation": fd["violation"],
            "fad_marginal": fad["marginal"],
            "fad_proxy": fad["proxy"],
            "fad_target": fad["target"],
            "fad_lambda": fad["lambda"],
            "fad_violation": fad["violation"],
            "fad_active": waveform.new_full(
                (waveform.shape[0],), 1.0 if fad_active else 0.0
            ),
            "advantage": advantage,
        }

    def state_dict(self) -> dict[str, object]:
        state: dict[str, object] = {"calls": self.calls}
        if self.fd is not None:
            state["fd"] = self.fd.state_dict()
        if self.fad is not None:
            state["fad"] = self.fad.state_dict()
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.calls = int(state.get("calls", 0))
        if self.fd is not None and "fd" in state:
            self.fd.load_state_dict(state["fd"])  # type: ignore[arg-type]
        if self.fad is not None and "fad" in state:
            self.fad.load_state_dict(state["fad"])  # type: ignore[arg-type]


__all__ = [
    "BalancedAudioReward",
    "CLAPReward",
    "HTSATFusedCLAPReward",
    "PaSSTReward",
    "SpectralRepetitionReward",
]
