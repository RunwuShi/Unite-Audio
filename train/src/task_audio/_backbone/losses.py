from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.parametrizations import weight_norm

from .mel_losses import ConvMelSpectrogram, MelSpectrogramLoss


@dataclass(frozen=True)
class PairedAudioCrop:
    real: Tensor
    fake: Tensor
    valid_mask: Tensor
    starts: Tensor


def paired_valid_audio_crop(
    real: Tensor,
    fake: Tensor,
    valid_mask: Tensor | None,
    *,
    crop_samples: int,
    generator: torch.Generator | None = None,
) -> PairedAudioCrop:
    """Take the same random valid-region crop from paired real/fake audio.

    Inputs may be ``[B, T]`` or mono ``[B, 1, T]``. Short examples are
    right-padded after both signals have been masked, so the discriminator
    cannot use padding differences to distinguish real from reconstructed
    audio.
    """

    real = _mono_2d(real, name="real")
    fake = _mono_2d(fake, name="fake")
    if real.shape != fake.shape:
        raise ValueError("real and fake waveforms must have the same shape")
    if crop_samples < 1:
        raise ValueError("crop_samples must be positive")

    batch, total_samples = real.shape
    if total_samples == 0:
        raise ValueError("waveforms must contain at least one sample")
    if valid_mask is None:
        valid = torch.ones_like(real, dtype=torch.bool)
    else:
        valid = _mono_2d(valid_mask, name="valid_mask").to(
            device=real.device, dtype=torch.bool
        )
        if valid.shape != real.shape:
            raise ValueError("valid_mask must match the waveform shape")

    positions = torch.arange(total_samples, device=real.device).view(1, -1)
    has_valid = valid.any(dim=1)
    first = torch.where(valid, positions, total_samples).amin(dim=1)
    last = torch.where(valid, positions, -1).amax(dim=1)
    first = torch.where(has_valid, first, torch.zeros_like(first))
    last = torch.where(has_valid, last, torch.full_like(last, -1))
    valid_span = (last - first + 1).clamp_min(0)
    max_offset = (valid_span - int(crop_samples)).clamp_min(0)
    random_value = torch.rand(
        (batch,), device=real.device, generator=generator, dtype=torch.float32
    )
    offsets = torch.floor(random_value * (max_offset + 1).float()).long()
    starts = first + offsets

    relative = torch.arange(crop_samples, device=real.device).view(1, -1)
    indices = starts[:, None] + relative
    in_bounds = indices < total_samples
    safe_indices = indices.clamp(min=0, max=max(total_samples - 1, 0))
    real_crop = real.gather(1, safe_indices)
    fake_crop = fake.gather(1, safe_indices)
    crop_valid = valid.gather(1, safe_indices) & in_bounds
    crop_weight = crop_valid.to(dtype=real_crop.dtype)
    real_crop = real_crop * crop_weight
    fake_crop = fake_crop * crop_weight
    return PairedAudioCrop(
        real=real_crop.unsqueeze(1),
        fake=fake_crop.unsqueeze(1),
        valid_mask=crop_valid.unsqueeze(1),
        starts=starts,
    )


class KWeighting(nn.Module):
    """Differentiable ITU-R BS.1770 K-weighting pre-filter.

    Coefficients use the frequencies/Q values also used by common BS.1770
    implementations. ``torchaudio.functional.lfilter`` supplies the batched,
    differentiable IIR implementation without adding trainable state.
    """

    def __init__(self, sample_rate: int) -> None:
        super().__init__()
        if sample_rate < 8_000:
            raise ValueError("K-weighting requires sample_rate >= 8000")
        shelf_b, shelf_a = _high_shelf_coefficients(
            sample_rate,
            frequency=1681.974450955533,
            gain_db=3.99984385397,
            q=0.7071752369554196,
        )
        highpass_b, highpass_a = _highpass_coefficients(
            sample_rate,
            frequency=38.13547087602444,
            q=0.5003270373238773,
        )
        self.register_buffer("shelf_b", shelf_b, persistent=False)
        self.register_buffer("shelf_a", shelf_a, persistent=False)
        self.register_buffer("highpass_b", highpass_b, persistent=False)
        self.register_buffer("highpass_a", highpass_a, persistent=False)

    def forward(self, waveform: Tensor) -> Tensor:
        try:
            from torchaudio.functional import lfilter
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "K-weighted reconstruction loss requires torchaudio.functional.lfilter"
            ) from exc
        original_shape = waveform.shape
        if waveform.ndim < 2:
            raise ValueError("waveform must have at least batch and time dimensions")
        flat = waveform.float().reshape(-1, waveform.shape[-1])
        flat = lfilter(
            flat,
            self.shelf_a.to(device=flat.device),
            self.shelf_b.to(device=flat.device),
            clamp=False,
        )
        flat = lfilter(
            flat,
            self.highpass_a.to(device=flat.device),
            self.highpass_b.to(device=flat.device),
            clamp=False,
        )
        return flat.reshape(original_shape)


class SAMEPhaseAwareMultiResolutionSTFTLoss(nn.Module):
    """SAME-style K-weighted, phase-aware multi-resolution STFT loss."""

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        fft_sizes: Sequence[int] = (32, 64, 128, 256, 512, 1024, 2048),
        hop_ratio: float = 0.25,
        k_weighting: bool = True,
        eps: float = 1.0e-7,
    ) -> None:
        super().__init__()
        sizes = tuple(int(value) for value in fft_sizes)
        if not sizes or any(value < 4 or value % 2 for value in sizes):
            raise ValueError("fft_sizes must contain positive even integers >= 4")
        if not 0.0 < float(hop_ratio) <= 1.0:
            raise ValueError("hop_ratio must be in (0, 1]")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.fft_sizes = sizes
        self.hop_ratio = float(hop_ratio)
        self.eps = float(eps)
        self.k_weighting = KWeighting(sample_rate) if k_weighting else nn.Identity()
        for fft_size in sizes:
            self.register_buffer(
                f"window_{fft_size}",
                torch.hann_window(fft_size, periodic=True, dtype=torch.float32),
                persistent=False,
            )

    def forward(self, fake: Tensor, real: Tensor) -> dict[str, Tensor]:
        fake = _mono_3d(fake, name="fake")
        real = _mono_3d(real, name="real")
        if fake.shape != real.shape:
            raise ValueError("fake and real waveforms must have the same shape")
        if fake.shape[-1] < 2:
            raise ValueError("waveforms must contain at least two samples")

        autocast_context = (
            torch.autocast(device_type=fake.device.type, enabled=False)
            if fake.device.type in {"cpu", "cuda"}
            else nullcontext()
        )
        with autocast_context:
            fake_filtered = self.k_weighting(fake.float())
            real_filtered = self.k_weighting(real.detach().float())
            totals = {
                "spectral_contrast": fake_filtered.new_tensor(0.0),
                "adaptive_log_magnitude": fake_filtered.new_tensor(0.0),
                "instantaneous_frequency": fake_filtered.new_tensor(0.0),
                "group_delay": fake_filtered.new_tensor(0.0),
                "complex_distance": fake_filtered.new_tensor(0.0),
            }
            for fft_size in self.fft_sizes:
                fake_stft = self._stft(fake_filtered, fft_size)
                real_stft = self._stft(real_filtered, fft_size)
                components = self._resolution_loss(fake_stft, real_stft)
                for name, value in components.items():
                    totals[name] = totals[name] + value

            resolution_count = float(len(self.fft_sizes))
            totals = {name: value / resolution_count for name, value in totals.items()}
            total = sum(totals.values(), fake_filtered.new_tensor(0.0))
        return {"loss": total, **totals}

    def _stft(self, waveform: Tensor, fft_size: int) -> Tensor:
        batch, channels, samples = waveform.shape
        flattened = waveform.reshape(batch * channels, samples)
        window = getattr(self, f"window_{fft_size}").to(device=waveform.device)
        spectrum = torch.stft(
            flattened,
            n_fft=fft_size,
            hop_length=max(1, int(round(fft_size * self.hop_ratio))),
            win_length=fft_size,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        return spectrum.reshape(batch, channels, spectrum.shape[-2], spectrum.shape[-1])

    def _resolution_loss(self, fake: Tensor, real: Tensor) -> dict[str, Tensor]:
        fake_magnitude = fake.abs()
        real_magnitude = real.abs()
        reduce_dims = (-2, -1)

        contrast_numerator = torch.linalg.vector_norm(
            fake_magnitude - real_magnitude, dim=reduce_dims
        )
        contrast_denominator = torch.linalg.vector_norm(
            fake_magnitude + real_magnitude, dim=reduce_dims
        ).clamp_min(self.eps)
        spectral_contrast = (contrast_numerator / contrast_denominator).mean()

        fake_std = fake_magnitude.std(dim=reduce_dims, unbiased=False)
        real_std = real_magnitude.std(dim=reduce_dims, unbiased=False)
        sigma = torch.sqrt(fake_std.square() + real_std.square()).detach()
        sigma = sigma.clamp_min(self.eps).unsqueeze(-1).unsqueeze(-1)
        fake_log = torch.log1p(fake_magnitude / sigma)
        real_log = torch.log1p(real_magnitude / sigma)
        adaptive_log_magnitude = (fake_log - real_log).abs().mean()

        fake_time_product = fake[..., 1:] * fake[..., :-1].conj()
        real_time_product = real[..., 1:] * real[..., :-1].conj()
        fake_time_energy = torch.sqrt(
            fake_magnitude[..., 1:] * fake_magnitude[..., :-1]
        )
        real_time_energy = torch.sqrt(
            real_magnitude[..., 1:] * real_magnitude[..., :-1]
        )
        instantaneous_frequency = self._weighted_phasor_distance(
            fake_time_product,
            real_time_product,
            fake_time_energy,
            real_time_energy,
        )

        fake_frequency_product = fake[..., 1:, :] * fake[..., :-1, :].conj()
        real_frequency_product = real[..., 1:, :] * real[..., :-1, :].conj()
        fake_frequency_energy = torch.sqrt(
            fake_magnitude[..., 1:, :] * fake_magnitude[..., :-1, :]
        )
        real_frequency_energy = torch.sqrt(
            real_magnitude[..., 1:, :] * real_magnitude[..., :-1, :]
        )
        group_delay = self._weighted_phasor_distance(
            fake_frequency_product,
            real_frequency_product,
            fake_frequency_energy,
            real_frequency_energy,
        )

        complex_error = (fake - real).abs().square()
        complex_scale = complex_error.std(dim=reduce_dims, unbiased=False).detach()
        complex_scale = complex_scale.clamp_min(self.eps).unsqueeze(-1).unsqueeze(-1)
        complex_distance = torch.log1p(complex_error / complex_scale).mean()
        return {
            "spectral_contrast": spectral_contrast,
            "adaptive_log_magnitude": adaptive_log_magnitude,
            "instantaneous_frequency": instantaneous_frequency,
            "group_delay": group_delay,
            "complex_distance": complex_distance,
        }

    def _weighted_phasor_distance(
        self,
        fake_product: Tensor,
        real_product: Tensor,
        fake_energy: Tensor,
        real_energy: Tensor,
    ) -> Tensor:
        fake_unit = fake_product / fake_product.abs().clamp_min(self.eps)
        real_unit = real_product / real_product.abs().clamp_min(self.eps)
        cosine_distance = 1.0 - (fake_unit * real_unit.conj()).real
        # ``fake_energy`` and ``real_energy`` are already the geometric mean
        # of each adjacent pair. Their product is therefore the detached,
        # mean-normalised weight from SAME Eq. (5)/(6).
        weight = (fake_energy * real_energy).detach()
        numerator = (weight * cosine_distance).sum(dim=(-2, -1))
        denominator = weight.sum(dim=(-2, -1)).clamp_min(self.eps)
        return (numerator / denominator).mean()


# The discriminator design below is adapted from Stability AI's
# stable-audio-tools EnCodec and HIL discriminators (MIT licensed), including
# the EnCodec complex-STFT stack and HIL FilterBankDiscriminator/PQMF stack:
# https://github.com/Stability-AI/stable-audio-tools/blob/main/stable_audio_tools/models/encodec.py
# https://github.com/Stability-AI/stable-audio-tools/blob/main/stable_audio_tools/models/discriminators.py
# https://github.com/Stability-AI/stable-audio-tools/blob/main/stable_audio_tools/models/transforms.py


class _STFTViewDiscriminator(nn.Module):
    def __init__(self, fft_size: int, *, channels: int = 1, filters: int = 32) -> None:
        super().__init__()
        self.fft_size = int(fft_size)
        self.hop_length = max(1, self.fft_size // 4)
        self.register_buffer(
            "window",
            torch.hann_window(self.fft_size, periodic=True, dtype=torch.float32),
            persistent=False,
        )
        layers: list[nn.Module] = []
        in_channels = int(channels) * 2
        layers.append(
            weight_norm(
                nn.Conv2d(
                    in_channels,
                    filters,
                    kernel_size=(3, 9),
                    padding=(1, 4),
                )
            )
        )
        for dilation in (1, 2, 4):
            layers.append(
                weight_norm(
                    nn.Conv2d(
                        filters,
                        filters,
                        kernel_size=(3, 9),
                        stride=(1, 2),
                        dilation=(dilation, 1),
                        padding=(dilation, 4),
                    )
                )
            )
        layers.append(
            weight_norm(
                nn.Conv2d(filters, filters, kernel_size=(3, 3), padding=(1, 1))
            )
        )
        self.layers = nn.ModuleList(layers)
        self.post = weight_norm(
            nn.Conv2d(filters, 1, kernel_size=(3, 3), padding=(1, 1))
        )

    def forward(self, waveform: Tensor) -> tuple[Tensor, list[Tensor]]:
        spectrum = _complex_spectrogram(
            waveform,
            fft_size=self.fft_size,
            hop_length=self.hop_length,
            window=self.window,
        )
        features: list[Tensor] = []
        hidden = spectrum
        for layer in self.layers:
            hidden = F.leaky_relu(layer(hidden), negative_slope=0.2)
            features.append(hidden)
        logits = self.post(hidden)
        return logits, features


class _PQMFAnalysis(nn.Module):
    """Near-perfect-reconstruction pseudo-QMF analysis filter bank."""

    def __init__(
        self,
        *,
        subbands: int = 4,
        taps: int = 62,
        cutoff_ratio: float = 0.142,
        beta: float = 9.0,
    ) -> None:
        super().__init__()
        if subbands < 2 or taps < 2 or taps % 2:
            raise ValueError("PQMF requires subbands >= 2 and a positive even taps value")
        if not 0.0 < cutoff_ratio < 1.0:
            raise ValueError("PQMF cutoff_ratio must be in (0, 1)")
        self.subbands = int(subbands)
        prototype = _firwin_kaiser(taps + 1, cutoff_ratio, beta)
        time = torch.arange(taps + 1, dtype=torch.float64)
        center = taps / 2.0
        filters = []
        for band in range(subbands):
            phase = (
                (2 * band + 1)
                * math.pi
                / (2.0 * subbands)
                * (time - center)
                + ((-1) ** band) * math.pi / 4.0
            )
            filters.append(2.0 * prototype * torch.cos(phase))
        analysis = (
            torch.stack(filters).float().unsqueeze(1) * math.sqrt(float(subbands))
        )
        self.register_buffer("analysis_filter", analysis, persistent=False)
        self.padding = taps // 2

    def forward(self, waveform: Tensor) -> Tensor:
        waveform = _mono_3d(waveform, name="waveform")
        return F.conv1d(
            waveform.float(),
            self.analysis_filter.to(device=waveform.device),
            stride=self.subbands,
            padding=self.padding,
        )


class _PQMFViewDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.analysis = _PQMFAnalysis()
        # HIL FilterBankDiscriminator defaults, reduced to one four-band PQMF
        # view because TTA deliberately omits the chroma/multi-period branches.
        channels = (32, 128, 512, 1024, 1024)
        strides = (3, 3, 3, 3, 1)
        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels, stride in zip(channels, strides, strict=True):
            layers.append(
                weight_norm(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=(1, 5),
                        stride=(1, stride),
                        padding=(0, 2),
                    )
                )
            )
            in_channels = out_channels
        self.layers = nn.ModuleList(layers)
        self.post = weight_norm(
            nn.Conv2d(in_channels, 1, kernel_size=(1, 3), padding=(0, 1))
        )

    def forward(self, waveform: Tensor) -> tuple[Tensor, list[Tensor]]:
        hidden = self.analysis(waveform).unsqueeze(1)
        features: list[Tensor] = []
        for layer in self.layers:
            hidden = F.leaky_relu(layer(hidden), negative_slope=0.1)
            features.append(hidden)
        logits = self.post(hidden)
        features.append(logits)
        return logits.flatten(1), features


class MultiViewSTFTPQMFAudioDiscriminator(nn.Module):
    """Five complex-STFT views plus one PQMF subband view."""

    def __init__(
        self,
        *,
        stft_fft_sizes: Sequence[int] = (128, 256, 512, 1024, 2048),
        pqmf_enabled: bool = True,
    ) -> None:
        super().__init__()
        sizes = tuple(int(value) for value in stft_fft_sizes)
        if not sizes or any(value < 4 or value % 2 for value in sizes):
            raise ValueError("stft_fft_sizes must contain positive even integers >= 4")
        self.stft_views = nn.ModuleList(
            [_STFTViewDiscriminator(value) for value in sizes]
        )
        self.pqmf_view = _PQMFViewDiscriminator() if pqmf_enabled else None

    @property
    def num_views(self) -> int:
        return len(self.stft_views) + int(self.pqmf_view is not None)

    def forward(self, waveform: Tensor) -> tuple[list[Tensor], list[list[Tensor]]]:
        waveform = _mono_3d(waveform, name="waveform")
        logits: list[Tensor] = []
        features: list[list[Tensor]] = []
        for discriminator in self.stft_views:
            view_logits, view_features = discriminator(waveform)
            logits.append(view_logits)
            features.append(view_features)
        if self.pqmf_view is not None:
            view_logits, view_features = self.pqmf_view(waveform)
            logits.append(view_logits)
            features.append(view_features)
        return logits, features


def relativistic_discriminator_loss(
    real_logits: Sequence[Tensor], fake_logits: Sequence[Tensor]
) -> Tensor:
    if len(real_logits) != len(fake_logits) or not real_logits:
        raise ValueError("real/fake logits must contain the same non-zero number of views")
    losses = [
        F.softplus(-(real_score - fake_score)).mean()
        for real_score, fake_score in zip(real_logits, fake_logits, strict=True)
    ]
    return torch.stack(losses).mean()


def relativistic_generator_loss(
    real_logits: Sequence[Tensor], fake_logits: Sequence[Tensor]
) -> Tensor:
    if len(real_logits) != len(fake_logits) or not real_logits:
        raise ValueError("real/fake logits must contain the same non-zero number of views")
    losses = [
        F.softplus(real_score - fake_score).mean()
        for real_score, fake_score in zip(real_logits, fake_logits, strict=True)
    ]
    return torch.stack(losses).mean()


def discriminator_feature_matching_loss(
    real_features: Sequence[Sequence[Tensor]],
    fake_features: Sequence[Sequence[Tensor]],
) -> Tensor:
    if len(real_features) != len(fake_features) or not real_features:
        raise ValueError("real/fake features must contain the same non-zero number of views")
    view_losses = []
    for real_view, fake_view in zip(real_features, fake_features, strict=True):
        if len(real_view) != len(fake_view) or not real_view:
            raise ValueError("each discriminator view must expose matching feature layers")
        layer_losses = [
            (fake_layer - real_layer.detach()).abs().mean()
            for real_layer, fake_layer in zip(real_view, fake_view, strict=True)
        ]
        view_losses.append(torch.stack(layer_losses).mean())
    return torch.stack(view_losses).mean()


def masked_waveform_l1(fake: Tensor, real: Tensor, valid_mask: Tensor) -> Tensor:
    fake = _mono_3d(fake, name="fake")
    real = _mono_3d(real, name="real")
    valid = _mono_3d(valid_mask, name="valid_mask").to(
        device=fake.device, dtype=fake.dtype
    )
    if fake.shape != real.shape or fake.shape != valid.shape:
        raise ValueError("fake, real, and valid_mask must have matching shapes")
    return ((fake - real).abs() * valid).sum() / valid.sum().clamp_min(1.0)


def _complex_spectrogram(
    waveform: Tensor,
    *,
    fft_size: int,
    hop_length: int,
    window: Tensor,
) -> Tensor:
    batch, channels, samples = waveform.shape
    context = (
        torch.autocast(device_type=waveform.device.type, enabled=False)
        if waveform.device.type in {"cpu", "cuda"}
        else nullcontext()
    )
    with context:
        spectrum = torch.stft(
            waveform.float().reshape(batch * channels, samples),
            n_fft=fft_size,
            hop_length=hop_length,
            win_length=fft_size,
            window=window.to(device=waveform.device),
            center=False,
            normalized=True,
            onesided=True,
            return_complex=True,
        )
        spectrum = spectrum.reshape(
            batch, channels, spectrum.shape[-2], spectrum.shape[-1]
        )
        spectrum = torch.cat((spectrum.real, spectrum.imag), dim=1)
        # EnCodec convolves a [time, frequency] image.
        return spectrum.permute(0, 1, 3, 2).contiguous()


def _firwin_kaiser(length: int, cutoff_ratio: float, beta: float) -> Tensor:
    # ``cutoff_ratio`` is normalized to Nyquist, matching scipy.signal.firwin.
    index = torch.arange(length, dtype=torch.float64)
    centered = index - (length - 1) / 2.0
    prototype = cutoff_ratio * torch.sinc(cutoff_ratio * centered)
    window = torch.kaiser_window(
        length, periodic=False, beta=float(beta), dtype=torch.float64
    )
    prototype = prototype * window
    return prototype / prototype.sum().clamp_min(torch.finfo(torch.float64).eps)


def _high_shelf_coefficients(
    sample_rate: int, *, frequency: float, gain_db: float, q: float
) -> tuple[Tensor, Tensor]:
    k = math.tan(math.pi * frequency / sample_rate)
    vh = 10.0 ** (gain_db / 20.0)
    vb = vh ** 0.4996667741545416
    a0 = 1.0 + k / q + k * k
    b = torch.tensor(
        [
            (vh + vb * k / q + k * k) / a0,
            2.0 * (k * k - vh) / a0,
            (vh - vb * k / q + k * k) / a0,
        ],
        dtype=torch.float32,
    )
    a = torch.tensor(
        [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0],
        dtype=torch.float32,
    )
    return b, a


def _highpass_coefficients(
    sample_rate: int, *, frequency: float, q: float
) -> tuple[Tensor, Tensor]:
    k = math.tan(math.pi * frequency / sample_rate)
    a0 = 1.0 + k / q + k * k
    b = torch.tensor([1.0 / a0, -2.0 / a0, 1.0 / a0], dtype=torch.float32)
    a = torch.tensor(
        [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0],
        dtype=torch.float32,
    )
    return b, a


def _mono_2d(value: Tensor, *, name: str) -> Tensor:
    if value.ndim == 3 and value.shape[1] == 1:
        value = value.squeeze(1)
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [B, T] or [B, 1, T]")
    return value


def _mono_3d(value: Tensor, *, name: str) -> Tensor:
    if value.ndim == 2:
        value = value.unsqueeze(1)
    if value.ndim != 3 or value.shape[1] != 1:
        raise ValueError(f"{name} must have shape [B, T] or mono [B, 1, T]")
    return value


__all__ = [
    "ConvMelSpectrogram",
    "KWeighting",
    "MelSpectrogramLoss",
    "MultiViewSTFTPQMFAudioDiscriminator",
    "PairedAudioCrop",
    "SAMEPhaseAwareMultiResolutionSTFTLoss",
    "discriminator_feature_matching_loss",
    "masked_waveform_l1",
    "paired_valid_audio_crop",
    "relativistic_discriminator_loss",
    "relativistic_generator_loss",
]
