from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SamePadISTFT(nn.Module):
    def __init__(self, *, n_fft: int, hop_length: int, win_length: int | None = None) -> None:
        super().__init__()
        if n_fft < 2 or n_fft % 2 != 0:
            raise ValueError("n_fft must be an even integer >= 2")
        if hop_length < 1:
            raise ValueError("hop_length must be positive")
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(n_fft if win_length is None else win_length)
        if self.win_length < self.hop_length:
            raise ValueError("win_length must be >= hop_length for same-pad ISTFT")
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)

    def forward(self, spec: Tensor) -> Tensor:
        if spec.ndim != 3:
            raise ValueError("spec must have shape [B, F, T]")
        if spec.shape[1] != self.n_fft // 2 + 1:
            raise ValueError(
                f"spec frequency bins must be {self.n_fft // 2 + 1}, got {spec.shape[1]}"
            )

        spec = spec.to(torch.complex64)
        frames = torch.fft.irfft(spec, n=self.n_fft, dim=1, norm="backward")
        frames = frames[:, : self.win_length] * self.window.to(device=frames.device, dtype=frames.dtype)[
            None, :, None
        ]

        batch, _, frame_count = frames.shape
        output_size = (frame_count - 1) * self.hop_length + self.win_length
        audio = F.fold(
            frames,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, :]

        window_sq = self.window.to(device=frames.device, dtype=frames.dtype).square()
        envelope_frames = window_sq[None, :, None].expand(batch, -1, frame_count)
        envelope = F.fold(
            envelope_frames,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, :]

        pad = (self.win_length - self.hop_length) // 2
        if pad > 0:
            audio = audio[:, pad:-pad]
            envelope = envelope[:, pad:-pad]
        return audio / envelope.clamp_min(1.0e-8)


class ISTFTDecoderHead(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        hop_length: int,
        n_fft: int,
        mag_clip: float = 100.0,
    ) -> None:
        super().__init__()
        bins = n_fft // 2 + 1
        self.out = nn.Linear(dim, bins * 2)
        self.istft = SamePadISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
        self.mag_clip = float(mag_clip)
        if self.mag_clip <= 0.0:
            raise ValueError("mag_clip must be positive")

    def forward(self, x: Tensor, original_len: int | None = None) -> Tensor:
        dtype = x.dtype
        coeffs = self.out(x).float().transpose(1, 2)
        log_mag, phase = coeffs.chunk(2, dim=1)
        log_mag = log_mag.clamp(max=math.log(self.mag_clip))
        mag = log_mag.exp()
        spec = mag * torch.complex(torch.cos(phase), torch.sin(phase))
        audio = self.istft(spec)
        if original_len is not None:
            audio = audio[:, :original_len]
        return audio.to(dtype=dtype)
