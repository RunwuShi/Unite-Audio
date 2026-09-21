from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn


THREE_D_SPEAKER_ROOT = Path(__file__).resolve().parent / "_vendor" / "3D-Speaker"
if str(THREE_D_SPEAKER_ROOT) not in sys.path:
    sys.path.insert(0, str(THREE_D_SPEAKER_ROOT))

from speakerlab.models.eres2net.ERes2NetV2 import ERes2NetV2  # noqa: E402
from speakerlab.models.campplus.DTDNN import CAMPPlus  # noqa: E402
from speakerlab.process.processor import FBank  # noqa: E402


class SpeakerERes2NetV2SequenceEncoder(nn.Module):
    """ERes2NetV2 speaker sequence encoder using local TSTP windows.

    The original ERes2NetV2 pools the full utterance over the final time axis
    and maps one global TSTP vector through seg_1. This wrapper keeps the
    backbone up to fuse_out34, applies the same TSTP calculation inside local
    windows, reuses seg_1 per window, and projects the resulting sequence to
    the FM hidden dimension.
    """

    def __init__(
        self,
        *,
        output_dim: int,
        sample_rate: int = 16000,
        num_mel_bins: int = 80,
        window_size: int = 8,
        stride: int = 4,
        embedding_size: int = 192,
        pretrained_path: str | None = None,
        mean_norm: bool = True,
        freeze_backbone: bool = False,
        profile: bool = False,
    ) -> None:
        super().__init__()
        if sample_rate != 16000:
            raise ValueError("SpeakerERes2NetV2SequenceEncoder currently expects 16 kHz reference audio")
        if num_mel_bins != 80:
            raise ValueError("ERes2NetV2 speaker encoder expects 80-bin fbank input")
        if window_size < 1:
            raise ValueError("speaker local TSTP window_size must be positive")
        if stride < 1:
            raise ValueError("speaker local TSTP stride must be positive")
        if output_dim < 1:
            raise ValueError("speaker output_dim must be positive")

        self.sample_rate = int(sample_rate)
        self.num_mel_bins = int(num_mel_bins)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.embedding_size = int(embedding_size)
        self.freeze_backbone = bool(freeze_backbone)
        self.profile = bool(profile)
        self.last_timing: dict[str, float] = {}
        if self.freeze_backbone and not pretrained_path:
            raise ValueError("speaker_freeze_backbone=True requires speaker_pretrained_path; refusing to freeze a random ERes2NetV2")
        self.feature_extractor = FBank(self.num_mel_bins, sample_rate=self.sample_rate, mean_nor=mean_norm)
        self.backbone = ERes2NetV2(feat_dim=self.num_mel_bins, embedding_size=self.embedding_size)
        if pretrained_path:
            self._load_pretrained(pretrained_path)
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        self.out_proj = nn.Linear(self.embedding_size, int(output_dim))

    def train(self, mode: bool = True) -> "SpeakerERes2NetV2SequenceEncoder":
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(
        self,
        ref_wav_clean: Tensor,
        ref_wav_valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        self.last_timing = {}
        total_start = self._timer_start(ref_wav_clean)
        if ref_wav_clean.ndim != 2:
            raise ValueError("ref_wav_clean must have shape [B, S]")
        if ref_wav_valid_mask.shape != ref_wav_clean.shape:
            raise ValueError("ref_wav_valid_mask must have shape [B, S]")
        if not bool(ref_wav_valid_mask.to(dtype=torch.bool).any(dim=1).all().item()):
            raise ValueError("LatentFMFM expects every sample to have reference audio")

        timer_start = self._timer_start(ref_wav_clean)
        fbank, fbank_mask = self._fbank_batch(ref_wav_clean, ref_wav_valid_mask)
        self._timer_stop("speaker_fbank_sec", timer_start, fbank)

        if self.freeze_backbone:
            with torch.no_grad():
                timer_start = self._timer_start(fbank)
                fuse = self._fuse_out34(fbank)
                self._timer_stop("speaker_backbone_sec", timer_start, fuse)

                timer_start = self._timer_start(fuse)
                fuse_mask = self._resize_mask(fbank_mask, fuse.shape[-1])
                emb, token_mask = self._local_tstp_tokens(fuse, fuse_mask)
                self._timer_stop("speaker_pool_sec", timer_start, emb)
        else:
            timer_start = self._timer_start(fbank)
            fuse = self._fuse_out34(fbank)
            self._timer_stop("speaker_backbone_sec", timer_start, fuse)

            timer_start = self._timer_start(fuse)
            fuse_mask = self._resize_mask(fbank_mask, fuse.shape[-1])
            emb, token_mask = self._local_tstp_tokens(fuse, fuse_mask)
            self._timer_stop("speaker_pool_sec", timer_start, emb)

        timer_start = self._timer_start(emb)
        tokens = self.out_proj(emb)
        self._timer_stop("speaker_proj_sec", timer_start, tokens)
        self._timer_stop("speaker_total_sec", total_start, tokens)
        return tokens, token_mask

    def _fbank_batch(self, wav: Tensor, wav_mask: Tensor) -> tuple[Tensor, Tensor]:
        feats: list[Tensor] = []
        lengths: list[int] = []
        wav_mask = wav_mask.to(device=wav.device, dtype=torch.bool)
        for row in range(wav.shape[0]):
            length = int(wav_mask[row].long().sum().item())
            length = max(length, 1)
            feat = self.feature_extractor(wav[row, :length])
            feats.append(feat.to(device=wav.device, dtype=wav.dtype))
            lengths.append(int(feat.shape[0]))
        max_len = max(lengths)
        padded = wav.new_zeros((wav.shape[0], max_len, self.num_mel_bins))
        mask = torch.zeros((wav.shape[0], max_len), device=wav.device, dtype=torch.bool)
        for row, feat in enumerate(feats):
            length = int(feat.shape[0])
            padded[row, :length] = feat
            mask[row, :length] = True
        return padded, mask

    def _fuse_out34(self, fbank: Tensor) -> Tensor:
        x = fbank.permute(0, 2, 1).unsqueeze(1)
        out = F.relu(self.backbone.bn1(self.backbone.conv1(x)))
        out1 = self.backbone.layer1(out)
        out2 = self.backbone.layer2(out1)
        out3 = self.backbone.layer3(out2)
        out4 = self.backbone.layer4(out3)
        out3_ds = self.backbone.layer3_ds(out3)
        return self.backbone.fuse34(out4, out3_ds)

    @staticmethod
    def _resize_mask(mask: Tensor, length: int) -> Tensor:
        if length < 1:
            return mask.new_zeros((mask.shape[0], 0))
        resized = F.interpolate(mask.float().unsqueeze(1), size=length, mode="nearest").squeeze(1)
        return resized > 0.5

    def _local_tstp_tokens(self, fuse: Tensor, fuse_mask: Tensor) -> tuple[Tensor, Tensor]:
        batch, channels, freq, frames = fuse.shape
        if frames < 1:
            raise ValueError("ERes2NetV2 fuse_out34 produced no time frames")
        if frames < self.window_size:
            pad = self.window_size - frames
            fuse = F.pad(fuse, (0, pad))
            fuse_mask = F.pad(fuse_mask, (0, pad))
            frames = self.window_size

        windows = fuse.unfold(dimension=-1, size=self.window_size, step=self.stride)
        window_masks = fuse_mask.unfold(dimension=1, size=self.window_size, step=self.stride)
        if windows.shape[3] < 1:
            windows = fuse[..., : self.window_size].unsqueeze(3)
            window_masks = fuse_mask[:, : self.window_size].unsqueeze(1)

        stats = self._window_stats(windows, window_masks)
        num_tokens = int(stats.shape[1])
        emb = self.backbone.seg_1(stats.reshape(batch * num_tokens, -1)).view(
            batch,
            num_tokens,
            self.embedding_size,
        )
        token_mask = window_masks.all(dim=-1)

        no_full_window = ~token_mask.any(dim=1)
        if bool(no_full_window.any().item()):
            global_stats = self._global_stats(fuse[..., :frames], fuse_mask[:, :frames])
            global_emb = self.backbone.seg_1(global_stats)
            emb = emb.clone()
            token_mask = token_mask.clone()
            emb[no_full_window, 0] = global_emb[no_full_window]
            token_mask[no_full_window, 0] = True
        return emb, token_mask

    @staticmethod
    def _window_stats(windows: Tensor, window_masks: Tensor) -> Tensor:
        # windows: [B, C, F, K, W], window_masks: [B, K, W]
        mask = window_masks[:, None, None, :, :].to(device=windows.device, dtype=windows.dtype)
        count = mask.sum(dim=-1).clamp_min(1.0)
        mean = (windows * mask).sum(dim=-1) / count
        centered = (windows - mean.unsqueeze(-1)) * mask
        var = centered.square().sum(dim=-1) / count
        std = torch.sqrt(var + 1.0e-8)
        mean = mean.permute(0, 3, 1, 2).flatten(start_dim=2)
        std = std.permute(0, 3, 1, 2).flatten(start_dim=2)
        return torch.cat((mean, std), dim=-1)

    @staticmethod
    def _global_stats(fuse: Tensor, fuse_mask: Tensor) -> Tensor:
        mask = fuse_mask[:, None, None, :].to(device=fuse.device, dtype=fuse.dtype)
        count = mask.sum(dim=-1).clamp_min(1.0)
        mean = (fuse * mask).sum(dim=-1) / count
        centered = (fuse - mean.unsqueeze(-1)) * mask
        var = centered.square().sum(dim=-1) / count
        std = torch.sqrt(var + 1.0e-8)
        return torch.cat((mean.flatten(start_dim=1), std.flatten(start_dim=1)), dim=1)

    def _load_pretrained(self, path: str) -> None:
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"speaker pretrained checkpoint not found: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict):
            for key in ("model", "state_dict", "embedding_model"):
                value = state.get(key)
                if isinstance(value, dict):
                    state = value
                    break
        if not isinstance(state, dict):
            raise ValueError(f"unsupported speaker checkpoint format: {checkpoint_path}")
        cleaned = {
            str(key).removeprefix("module."): value
            for key, value in state.items()
            if isinstance(value, Tensor)
        }
        missing, unexpected = self.backbone.load_state_dict(cleaned, strict=False)
        if unexpected:
            raise ValueError(f"unexpected keys in speaker checkpoint: {unexpected[:8]}")
        allowed_missing_prefixes = ("seg_2.", "seg_bn_1.")
        critical_missing = [
            key for key in missing if not key.startswith(allowed_missing_prefixes)
        ]
        if critical_missing:
            raise ValueError(f"missing keys in speaker checkpoint: {critical_missing[:8]}")

    def _timer_start(self, anchor: Tensor) -> float:
        if not self.profile:
            return 0.0
        self._sync_for_profile(anchor)
        return time.perf_counter()

    def _timer_stop(self, name: str, start: float, anchor: Tensor) -> None:
        if not self.profile:
            return
        self._sync_for_profile(anchor)
        self.last_timing[name] = time.perf_counter() - start

    def _sync_for_profile(self, anchor: Tensor) -> None:
        if anchor.device.type == "cuda":
            torch.cuda.synchronize(anchor.device)


class SpeakerCAMPPlusSequenceEncoder(nn.Module):
    """CAM++ speaker sequence encoder using local statistics pooling.

    The original CAM++ path is head -> xvector frame network -> StatsPool ->
    dense. This wrapper stops before the global StatsPool, applies the same
    mean/std statistics in local windows, reuses the pretrained dense layer per
    local window, and projects the sequence to the FM hidden dimension.
    """

    def __init__(
        self,
        *,
        output_dim: int,
        sample_rate: int = 16000,
        num_mel_bins: int = 80,
        window_size: int = 32,
        stride: int = 16,
        embedding_size: int = 192,
        pretrained_path: str | None = None,
        mean_norm: bool = True,
        freeze_backbone: bool = True,
        profile: bool = False,
    ) -> None:
        super().__init__()
        if sample_rate != 16000:
            raise ValueError("SpeakerCAMPPlusSequenceEncoder currently expects 16 kHz reference audio")
        if num_mel_bins != 80:
            raise ValueError("CAM++ speaker encoder expects 80-bin fbank input")
        if window_size < 1:
            raise ValueError("speaker local stats window_size must be positive")
        if stride < 1:
            raise ValueError("speaker local stats stride must be positive")
        if output_dim < 1:
            raise ValueError("speaker output_dim must be positive")

        self.sample_rate = int(sample_rate)
        self.num_mel_bins = int(num_mel_bins)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.embedding_size = int(embedding_size)
        self.freeze_backbone = bool(freeze_backbone)
        self.profile = bool(profile)
        self.last_timing: dict[str, float] = {}
        if self.freeze_backbone and not pretrained_path:
            raise ValueError("speaker_freeze_backbone=True requires speaker_pretrained_path; refusing to freeze a random CAM++")
        self.feature_extractor = FBank(self.num_mel_bins, sample_rate=self.sample_rate, mean_nor=mean_norm)
        self.backbone = CAMPPlus(feat_dim=self.num_mel_bins, embedding_size=self.embedding_size)
        if pretrained_path:
            self._load_pretrained(pretrained_path)
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        self.out_proj = nn.Linear(self.embedding_size, int(output_dim))

    def train(self, mode: bool = True) -> "SpeakerCAMPPlusSequenceEncoder":
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(
        self,
        ref_wav_clean: Tensor,
        ref_wav_valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        self.last_timing = {}
        total_start = self._timer_start(ref_wav_clean)
        if ref_wav_clean.ndim != 2:
            raise ValueError("ref_wav_clean must have shape [B, S]")
        if ref_wav_valid_mask.shape != ref_wav_clean.shape:
            raise ValueError("ref_wav_valid_mask must have shape [B, S]")
        if not bool(ref_wav_valid_mask.to(dtype=torch.bool).any(dim=1).all().item()):
            raise ValueError("LatentFMFM expects every sample to have reference audio")

        timer_start = self._timer_start(ref_wav_clean)
        fbank, fbank_mask = self._fbank_batch(ref_wav_clean, ref_wav_valid_mask)
        self._timer_stop("speaker_fbank_sec", timer_start, fbank)

        if self.freeze_backbone:
            with torch.no_grad():
                timer_start = self._timer_start(fbank)
                frames = self._frame_features(fbank)
                self._timer_stop("speaker_backbone_sec", timer_start, frames)

                timer_start = self._timer_start(frames)
                frame_mask = SpeakerERes2NetV2SequenceEncoder._resize_mask(fbank_mask, frames.shape[-1])
                emb, token_mask = self._local_stats_tokens(frames, frame_mask)
                self._timer_stop("speaker_pool_sec", timer_start, emb)
        else:
            timer_start = self._timer_start(fbank)
            frames = self._frame_features(fbank)
            self._timer_stop("speaker_backbone_sec", timer_start, frames)

            timer_start = self._timer_start(frames)
            frame_mask = SpeakerERes2NetV2SequenceEncoder._resize_mask(fbank_mask, frames.shape[-1])
            emb, token_mask = self._local_stats_tokens(frames, frame_mask)
            self._timer_stop("speaker_pool_sec", timer_start, emb)

        timer_start = self._timer_start(emb)
        tokens = self.out_proj(emb)
        self._timer_stop("speaker_proj_sec", timer_start, tokens)
        self._timer_stop("speaker_total_sec", total_start, tokens)
        return tokens, token_mask

    def _fbank_batch(self, wav: Tensor, wav_mask: Tensor) -> tuple[Tensor, Tensor]:
        feats: list[Tensor] = []
        lengths: list[int] = []
        wav_mask = wav_mask.to(device=wav.device, dtype=torch.bool)
        for row in range(wav.shape[0]):
            length = int(wav_mask[row].long().sum().item())
            length = max(length, 1)
            feat = self.feature_extractor(wav[row, :length])
            feats.append(feat.to(device=wav.device, dtype=wav.dtype))
            lengths.append(int(feat.shape[0]))
        max_len = max(lengths)
        padded = wav.new_zeros((wav.shape[0], max_len, self.num_mel_bins))
        mask = torch.zeros((wav.shape[0], max_len), device=wav.device, dtype=torch.bool)
        for row, feat in enumerate(feats):
            length = int(feat.shape[0])
            padded[row, :length] = feat
            mask[row, :length] = True
        return padded, mask

    def _frame_features(self, fbank: Tensor) -> Tensor:
        x = fbank.permute(0, 2, 1)
        x = self.backbone.head(x)
        for name, module in self.backbone.xvector.named_children():
            if name in {"stats", "dense"}:
                break
            x = module(x)
        return x

    def _local_stats_tokens(self, frames: Tensor, frame_mask: Tensor) -> tuple[Tensor, Tensor]:
        batch, channels, num_frames = frames.shape
        if num_frames < 1:
            raise ValueError("CAM++ frame network produced no time frames")
        if num_frames < self.window_size:
            pad = self.window_size - num_frames
            frames = F.pad(frames, (0, pad))
            frame_mask = F.pad(frame_mask, (0, pad))
            num_frames = self.window_size

        windows = frames.unfold(dimension=-1, size=self.window_size, step=self.stride)
        window_masks = frame_mask.unfold(dimension=1, size=self.window_size, step=self.stride)
        if windows.shape[2] < 1:
            windows = frames[..., : self.window_size].unsqueeze(2)
            window_masks = frame_mask[:, : self.window_size].unsqueeze(1)

        stats = self._window_stats(windows, window_masks)
        emb = self.backbone.xvector._modules["dense"](stats.permute(0, 2, 1)).permute(0, 2, 1)
        token_mask = window_masks.all(dim=-1)

        no_full_window = ~token_mask.any(dim=1)
        if bool(no_full_window.any().item()):
            global_stats = self._global_stats(frames[..., :num_frames], frame_mask[:, :num_frames])
            global_emb = self.backbone.xvector._modules["dense"](global_stats)
            emb = emb.clone()
            token_mask = token_mask.clone()
            emb[no_full_window, 0] = global_emb[no_full_window]
            token_mask[no_full_window, 0] = True
        return emb, token_mask

    @staticmethod
    def _window_stats(windows: Tensor, window_masks: Tensor) -> Tensor:
        # windows: [B, C, K, W], window_masks: [B, K, W]
        mask = window_masks[:, None, :, :].to(device=windows.device, dtype=windows.dtype)
        count = mask.sum(dim=-1).clamp_min(1.0)
        mean = (windows * mask).sum(dim=-1) / count
        centered = (windows - mean.unsqueeze(-1)) * mask
        var = centered.square().sum(dim=-1) / count
        std = torch.sqrt(var + 1.0e-8)
        return torch.cat((mean.permute(0, 2, 1), std.permute(0, 2, 1)), dim=-1)

    @staticmethod
    def _global_stats(frames: Tensor, frame_mask: Tensor) -> Tensor:
        mask = frame_mask[:, None, :].to(device=frames.device, dtype=frames.dtype)
        count = mask.sum(dim=-1).clamp_min(1.0)
        mean = (frames * mask).sum(dim=-1) / count
        centered = (frames - mean.unsqueeze(-1)) * mask
        var = centered.square().sum(dim=-1) / count
        std = torch.sqrt(var + 1.0e-8)
        return torch.cat((mean, std), dim=1)

    def _load_pretrained(self, path: str) -> None:
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"speaker pretrained checkpoint not found: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict):
            for key in ("model", "state_dict", "embedding_model"):
                value = state.get(key)
                if isinstance(value, dict):
                    state = value
                    break
        if not isinstance(state, dict):
            raise ValueError(f"unsupported speaker checkpoint format: {checkpoint_path}")
        cleaned = {
            str(key).removeprefix("module."): value
            for key, value in state.items()
            if isinstance(value, Tensor)
        }
        missing, unexpected = self.backbone.load_state_dict(cleaned, strict=False)
        if missing:
            raise ValueError(f"missing keys in speaker checkpoint: {missing[:8]}")
        if unexpected:
            raise ValueError(f"unexpected keys in speaker checkpoint: {unexpected[:8]}")

    def _timer_start(self, anchor: Tensor) -> float:
        if not self.profile:
            return 0.0
        self._sync_for_profile(anchor)
        return time.perf_counter()

    def _timer_stop(self, name: str, start: float, anchor: Tensor) -> None:
        if not self.profile:
            return
        self._sync_for_profile(anchor)
        self.last_timing[name] = time.perf_counter() - start

    def _sync_for_profile(self, anchor: Tensor) -> None:
        if anchor.device.type == "cuda":
            torch.cuda.synchronize(anchor.device)
