from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
import math

import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor, nn


class ConvMelSpectrogram(nn.Module):
    """Mel spectrogram using conv1d DFT kernels instead of torch.stft/cuFFT."""

    def __init__(
        self,
        *,
        sample_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        f_min: float,
        f_max: float | None,
    ) -> None:
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        if self.n_fft < 1 or int(win_length) != self.n_fft:
            raise ValueError("ConvMelSpectrogram requires win_length == n_fft > 0")
        n_freqs = self.n_fft // 2 + 1
        time = torch.arange(self.n_fft, dtype=torch.float32)[None, :]
        freq = torch.arange(n_freqs, dtype=torch.float32)[:, None]
        angle = 2.0 * math.pi * freq * time / float(self.n_fft)
        window = torch.hann_window(self.n_fft, periodic=True, dtype=torch.float32)[None, :]
        self.register_buffer("real_kernel", torch.cos(angle).mul(window).unsqueeze(1))
        self.register_buffer("imag_kernel", -torch.sin(angle).mul(window).unsqueeze(1))
        mel_fmax = float(f_max) if f_max is not None else float(sample_rate) / 2.0
        mel_fbanks = torchaudio.functional.melscale_fbanks(
            n_freqs=n_freqs,
            f_min=float(f_min),
            f_max=mel_fmax,
            n_mels=int(n_mels),
            sample_rate=int(sample_rate),
            norm=None,
            mel_scale="htk",
        )
        self.register_buffer("mel_fbanks", mel_fbanks)

    def forward(self, waveform: Tensor) -> Tensor:
        if waveform.ndim != 2:
            raise ValueError("waveform must have shape [B, T]")
        waveform = F.pad(waveform.unsqueeze(1), (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        real = F.conv1d(waveform, self.real_kernel.to(dtype=waveform.dtype), stride=self.hop_length)
        imag = F.conv1d(waveform, self.imag_kernel.to(dtype=waveform.dtype), stride=self.hop_length)
        magnitude = (real.square() + imag.square()).clamp_min(1.0e-12).sqrt()
        return torch.einsum("bft,fm->bmt", magnitude, self.mel_fbanks.to(dtype=magnitude.dtype))


class MelSpectrogramLoss(nn.Module):
    """WavTTS/DAC-style multi-scale log-mel L1 loss."""

    def __init__(
        self,
        *,
        sample_rate: int,
        n_mels: Sequence[int] = (5, 10, 20, 40, 80, 160, 320),
        window_lengths: Sequence[int] = (32, 64, 128, 256, 512, 1024, 2048),
        mel_fmin: Sequence[float] = (0, 0, 0, 0, 0, 0, 0),
        mel_fmax: Sequence[float | None] = (None, None, None, None, None, None, None),
        power: float = 1.0,
        clamp_eps: float = 1.0e-5,
        max_chunk_samples: int = 250_000,
        max_stft_samples: int = 65_536,
        stft_device: str = "cuda",
    ) -> None:
        super().__init__()
        if len(n_mels) != len(window_lengths):
            raise ValueError("n_mels and window_lengths must have the same length")
        self.power = float(power)
        self.clamp_eps = float(clamp_eps)
        self.max_chunk_samples = int(max_chunk_samples)
        self.max_stft_samples = int(max_stft_samples)
        self.stft_device = str(stft_device)
        if self.stft_device not in {"cuda", "cpu", "conv"}:
            raise ValueError("stft_device must be 'cuda', 'cpu', or 'conv'")
        self.min_input_len = max(int(window) // 2 for window in window_lengths) + 1
        self.mel_transforms = nn.ModuleList()
        self.conv_mel_transforms = nn.ModuleList()
        self.cpu_mel_transforms: list[nn.Module] = []
        for idx, (num_mels, window_length) in enumerate(zip(n_mels, window_lengths)):
            fmin = mel_fmin[idx] if idx < len(mel_fmin) else 0.0
            fmax = mel_fmax[idx] if idx < len(mel_fmax) else None
            kwargs = {
                "sample_rate": sample_rate,
                "n_fft": int(window_length),
                "win_length": int(window_length),
                "hop_length": int(window_length) // 4,
                "n_mels": int(num_mels),
                "f_min": fmin,
                "f_max": fmax,
                "power": 1.0,
                "normalized": False,
                "center": True,
                "pad_mode": "reflect",
            }
            self.mel_transforms.append(torchaudio.transforms.MelSpectrogram(**kwargs))
            self.conv_mel_transforms.append(
                ConvMelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=int(window_length),
                    win_length=int(window_length),
                    hop_length=int(window_length) // 4,
                    n_mels=int(num_mels),
                    f_min=fmin,
                    f_max=fmax,
                )
            )
            self.cpu_mel_transforms.append(
                torchaudio.transforms.MelSpectrogram(**kwargs).to("cpu")
            )

    def forward(self, pred: Tensor, target: Tensor, sample_mask: Tensor | None = None) -> Tensor:
        if pred.ndim == 3 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.ndim == 3 and target.shape[1] == 1:
            target = target.squeeze(1)
        if pred.ndim != 2 or target.ndim != 2:
            raise ValueError("pred and target must have shape [B, T] or [B, 1, T]")
        if pred.shape != target.shape:
            raise ValueError("pred and target must have the same shape")

        pred = pred.float()
        target = target.float()
        if sample_mask is not None:
            mask = sample_mask.to(device=pred.device, dtype=pred.dtype)
            if mask.shape != pred.shape:
                raise ValueError("sample_mask must have shape [B, T]")
            pred = pred * mask
            target = target * mask

        if pred.shape[-1] < self.min_input_len:
            pad = self.min_input_len - pred.shape[-1]
            pred = F.pad(pred, (0, pad))
            target = F.pad(target, (0, pad))

        output_device = pred.device
        use_cpu_stft = self.stft_device == "cpu" and pred.device.type == "cuda"
        if use_cpu_stft:
            pred = pred.to("cpu")
            target = target.to("cpu")
            mel_transforms: Sequence[nn.Module] = self.cpu_mel_transforms
            autocast_device = "cpu"
        elif self.stft_device == "conv":
            mel_transforms = self.conv_mel_transforms
            autocast_device = pred.device.type
        else:
            mel_transforms = self.mel_transforms
            autocast_device = pred.device.type
        autocast_ctx = (
            torch.autocast(device_type=autocast_device, enabled=False)
            if autocast_device in {"cuda", "cpu"}
            else nullcontext()
        )
        with autocast_ctx:
            pred = pred.float()
            target = target.float()
            total = pred.new_tensor(0.0)
            for mel_transform in mel_transforms:
                total = total + self._mel_l1_chunked(mel_transform, pred, target)
        return total.to(output_device)

    def _mel_l1_chunked(
        self,
        mel_transform: nn.Module,
        pred: Tensor,
        target: Tensor,
    ) -> Tensor:
        batch = pred.shape[0]
        if batch < 1:
            return pred.new_tensor(0.0)
        if self.max_chunk_samples <= 0:
            chunk_size = batch
        else:
            chunk_size = max(1, min(batch, self.max_chunk_samples // max(int(pred.shape[-1]), 1)))

        total = pred.new_tensor(0.0)
        for start in range(0, batch, chunk_size):
            end = min(batch, start + chunk_size)
            pred_chunk = pred[start:end]
            target_chunk = target[start:end]
            total = total + self._mel_l1_time_chunked(mel_transform, pred_chunk, target_chunk)
        return total / batch

    def _mel_l1_time_chunked(
        self,
        mel_transform: nn.Module,
        pred: Tensor,
        target: Tensor,
        *,
        max_stft_samples: int | None = None,
    ) -> Tensor:
        return self._mel_l1_time_chunked_per_sample(
            mel_transform,
            pred,
            target,
            max_stft_samples=max_stft_samples,
        ).sum()

    def _mel_l1_time_chunked_per_sample(
        self,
        mel_transform: nn.Module,
        pred: Tensor,
        target: Tensor,
        *,
        max_stft_samples: int | None = None,
    ) -> Tensor:
        max_samples = self.max_stft_samples if max_stft_samples is None else int(max_stft_samples)
        total_samples = int(pred.shape[-1])
        if max_samples <= 0 or total_samples <= max_samples:
            try:
                return self._mel_l1_batch(mel_transform, pred, target)
            except RuntimeError as exc:
                if not _is_cufft_error(exc) or pred.device.type != "cuda" or total_samples <= self.min_input_len:
                    raise
                torch.cuda.empty_cache()
                retry_samples = max(self.min_input_len, total_samples // 2)
                if retry_samples >= total_samples:
                    raise
                return self._mel_l1_time_chunked_per_sample(
                    mel_transform,
                    pred,
                    target,
                    max_stft_samples=retry_samples,
                )

        per_sample = pred.new_zeros(pred.shape[0])
        weight_total = 0
        for start in range(0, total_samples, max_samples):
            end = min(total_samples, start + max_samples)
            chunk_len = end - start
            pred_chunk = pred[:, start:end]
            target_chunk = target[:, start:end]
            if chunk_len < self.min_input_len:
                pad = self.min_input_len - chunk_len
                pred_chunk = F.pad(pred_chunk, (0, pad))
                target_chunk = F.pad(target_chunk, (0, pad))
            per_sample = per_sample + self._mel_l1_time_chunked_per_sample(
                mel_transform,
                pred_chunk,
                target_chunk,
                max_stft_samples=max_samples,
            ) * chunk_len
            weight_total += chunk_len
        return per_sample / max(weight_total, 1)

    def _mel_l1_batch(self, mel_transform: nn.Module, pred: Tensor, target: Tensor) -> Tensor:
        with torch.no_grad():
            target_log = self._log_mel(mel_transform, target).detach()
        pred_log = self._log_mel(mel_transform, pred)
        return (pred_log - target_log).abs().mean(dim=(1, 2))

    def _log_mel(self, mel_transform: nn.Module, waveform: Tensor) -> Tensor:
        mel = mel_transform(waveform)
        return mel.clamp(min=self.clamp_eps).pow(self.power).log10()


def _is_cufft_error(exc: RuntimeError) -> bool:
    message = str(exc)
    return "cuFFT error" in message or "CUFFT_" in message
