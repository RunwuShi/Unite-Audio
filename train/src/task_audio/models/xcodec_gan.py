from __future__ import annotations

from contextlib import nullcontext
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm


# Adapted from zhenye234/X-Codec-2.0 (MIT), module/mpd.py and
# module/mstft.py. The public XCodec training code is used as the behavioral
# reference, while the interface follows task_audio's discriminator contract.


def _mono_3d(waveform: Tensor) -> Tensor:
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(1)
    if waveform.ndim != 3 or waveform.shape[1] != 1:
        raise ValueError("waveform must have shape [B, T] or [B, 1, T]")
    return waveform


class XCodecPeriodDiscriminator(nn.Module):
    """One XCodec/HiFi-GAN period discriminator."""

    def __init__(
        self,
        period: int,
        *,
        channels: int = 16,
        channel_increasing_factor: int = 4,
        max_downsample_channels: int = 512,
        downsample_scales: Sequence[int] = (3, 3, 3, 3, 1),
    ) -> None:
        super().__init__()
        if int(period) < 2:
            raise ValueError("period must be >= 2")
        if min(
            int(channels),
            int(channel_increasing_factor),
            int(max_downsample_channels),
        ) < 1:
            raise ValueError("period discriminator channel settings must be positive")
        scales = tuple(int(value) for value in downsample_scales)
        if not scales or any(value < 1 for value in scales):
            raise ValueError("downsample_scales must contain positive integers")

        self.period = int(period)
        self.downsample_scales = scales
        layers: list[nn.Module] = []
        in_channels = 1
        out_channels = int(channels)
        for scale in scales:
            layers.append(
                nn.Sequential(
                    weight_norm(
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size=(5, 1),
                            stride=(scale, 1),
                            padding=(2, 0),
                        )
                    ),
                    nn.LeakyReLU(negative_slope=0.1),
                )
            )
            in_channels = out_channels
            out_channels = min(
                out_channels * int(channel_increasing_factor),
                int(max_downsample_channels),
            )
        self.layers = nn.ModuleList(layers)
        # XCodec uses kernel_sizes[1] - 1 with kernel_sizes=[5, 3].
        self.output = weight_norm(
            nn.Conv2d(
                in_channels,
                1,
                kernel_size=(2, 1),
                stride=1,
                padding=(1, 0),
            )
        )

    def forward(self, waveform: Tensor) -> tuple[Tensor, list[Tensor]]:
        hidden = _mono_3d(waveform)
        batch, channels, samples = hidden.shape
        padding = (-samples) % self.period
        if padding:
            mode = "reflect" if samples > padding else "replicate"
            hidden = F.pad(hidden, (0, padding), mode=mode)
            samples += padding
        hidden = hidden.view(batch, channels, samples // self.period, self.period)
        features: list[Tensor] = []
        for layer in self.layers:
            hidden = layer(hidden)
            features.append(hidden)
        logits = self.output(hidden).flatten(1)
        return logits, features


class XCodecMagnitudeSTFTDiscriminator(nn.Module):
    """One magnitude-STFT discriminator view from XCodec."""

    def __init__(
        self,
        fft_size: int,
        *,
        hop_size: int | None = None,
        channels: int = 32,
        max_downsample_channels: int = 512,
        downsample_scales: Sequence[int] = (2, 2, 2),
    ) -> None:
        super().__init__()
        self.fft_size = int(fft_size)
        self.hop_size = int(hop_size or self.fft_size // 2)
        if self.fft_size < 4 or self.fft_size % 2:
            raise ValueError("fft_size must be an even integer >= 4")
        if self.hop_size < 1 or self.hop_size > self.fft_size:
            raise ValueError("hop_size must be in [1, fft_size]")
        scales = tuple(int(value) for value in downsample_scales)
        if not scales or any(value < 1 for value in scales):
            raise ValueError("downsample_scales must contain positive integers")
        self.register_buffer(
            "window",
            torch.hann_window(self.fft_size, periodic=True, dtype=torch.float32),
            persistent=False,
        )

        layers: list[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(
                    1,
                    int(channels),
                    kernel_size=5,
                    stride=2,
                    padding=2,
                ),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
        ]
        in_channels = int(channels)
        for scale in scales:
            out_channels = min(
                in_channels * scale,
                int(max_downsample_channels),
            )
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=scale * 2 + 1,
                        stride=scale,
                        padding=scale,
                    ),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                )
            )
            in_channels = out_channels
        out_channels = min(in_channels * 2, int(max_downsample_channels))
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
        )
        self.layers = nn.ModuleList(layers)
        self.output = nn.Conv2d(out_channels, 1, kernel_size=3, padding=1)
        self.apply(self._initialize_convolution)
        self.apply(self._apply_weight_norm)

    @staticmethod
    def _initialize_convolution(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _apply_weight_norm(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
            weight_norm(module)

    def _magnitude_spectrogram(self, waveform: Tensor) -> Tensor:
        waveform = _mono_3d(waveform).squeeze(1)
        context = (
            torch.autocast(device_type=waveform.device.type, enabled=False)
            if waveform.device.type in {"cpu", "cuda"}
            else nullcontext()
        )
        with context:
            spectrum = torch.stft(
                waveform.float(),
                n_fft=self.fft_size,
                hop_length=self.hop_size,
                win_length=self.fft_size,
                window=self.window.to(device=waveform.device),
                center=True,
                normalized=False,
                onesided=True,
                return_complex=True,
            )
            magnitude = torch.sqrt(
                torch.clamp(
                    spectrum.real.square() + spectrum.imag.square(),
                    min=1.0e-7,
                    max=1.0e3,
                )
            )
        return magnitude.unsqueeze(1)

    def forward(self, waveform: Tensor) -> tuple[Tensor, list[Tensor]]:
        hidden = self._magnitude_spectrogram(waveform)
        features: list[Tensor] = []
        for layer in self.layers:
            hidden = layer(hidden)
            features.append(hidden)
        logits = self.output(hidden)
        return logits, features


class XCodecMPDSpecDiscriminator(nn.Module):
    """Five-period MPD plus multi-resolution magnitude spectral views."""

    def __init__(
        self,
        *,
        periods: Sequence[int] = (2, 3, 5, 7, 11),
        spec_fft_sizes: Sequence[int] = (
            216,
            348,
            568,
            920,
            1494,
            2414,
            3908,
            6328,
        ),
    ) -> None:
        super().__init__()
        periods = tuple(int(value) for value in periods)
        fft_sizes = tuple(int(value) for value in spec_fft_sizes)
        if not periods or any(value < 2 for value in periods):
            raise ValueError("periods must contain integers >= 2")
        if not fft_sizes or any(value < 4 or value % 2 for value in fft_sizes):
            raise ValueError("spec_fft_sizes must contain even integers >= 4")
        self.periods = periods
        self.spec_fft_sizes = fft_sizes
        self.mpd = nn.ModuleList(
            [XCodecPeriodDiscriminator(period) for period in periods]
        )
        self.spec = nn.ModuleList(
            [
                XCodecMagnitudeSTFTDiscriminator(
                    fft_size,
                    hop_size=fft_size // 2,
                )
                for fft_size in fft_sizes
            ]
        )

    @property
    def num_views(self) -> int:
        return len(self.mpd) + len(self.spec)

    @property
    def num_feature_terms(self) -> int:
        return sum(len(view.layers) for view in (*self.mpd, *self.spec))

    def forward(self, waveform: Tensor) -> tuple[list[Tensor], list[list[Tensor]]]:
        logits: list[Tensor] = []
        features: list[list[Tensor]] = []
        for discriminator in (*self.mpd, *self.spec):
            view_logits, view_features = discriminator(waveform)
            logits.append(view_logits)
            features.append(view_features)
        return logits, features


def lsgan_discriminator_sum(
    real_logits: Sequence[Tensor],
    fake_logits: Sequence[Tensor],
) -> Tensor:
    if len(real_logits) != len(fake_logits) or not real_logits:
        raise ValueError("real/fake logits must contain the same non-zero number of views")
    losses = [
        F.mse_loss(real.float(), torch.ones_like(real, dtype=torch.float32))
        + F.mse_loss(fake.float(), torch.zeros_like(fake, dtype=torch.float32))
        for real, fake in zip(real_logits, fake_logits, strict=True)
    ]
    return torch.stack(losses).sum()


def lsgan_generator_sum(fake_logits: Sequence[Tensor]) -> Tensor:
    if not fake_logits:
        raise ValueError("fake_logits must contain at least one view")
    return torch.stack(
        [
            F.mse_loss(fake, torch.ones_like(fake))
            if fake.dtype == torch.float32
            else F.mse_loss(
                fake.float(),
                torch.ones_like(fake, dtype=torch.float32),
            )
            for fake in fake_logits
        ]
    ).sum()


def feature_matching_sum(
    real_features: Sequence[Sequence[Tensor]],
    fake_features: Sequence[Sequence[Tensor]],
) -> Tensor:
    if len(real_features) != len(fake_features) or not real_features:
        raise ValueError("real/fake features must contain the same non-zero number of views")
    losses: list[Tensor] = []
    for real_view, fake_view in zip(real_features, fake_features, strict=True):
        if len(real_view) != len(fake_view) or not real_view:
            raise ValueError("each discriminator view must expose matching feature layers")
        losses.extend(
            (fake_layer - real_layer.detach()).abs().mean()
            for real_layer, fake_layer in zip(real_view, fake_view, strict=True)
        )
    return torch.stack(losses).sum()


__all__ = [
    "XCodecMPDSpecDiscriminator",
    "XCodecMagnitudeSTFTDiscriminator",
    "XCodecPeriodDiscriminator",
    "feature_matching_sum",
    "lsgan_discriminator_sum",
    "lsgan_generator_sum",
]
