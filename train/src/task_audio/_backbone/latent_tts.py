from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .istft_decoder import ISTFTDecoderHead
from .losses import MelSpectrogramLoss
from .text_encoder import TextEncoder
from .transformer import TransformerBlock
from .wave_fm_udit import LinearUDiTUNetWaveFMDecoder


def masked_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    while mask.ndim < pred.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(device=pred.device, dtype=pred.dtype)
    reduce_dims = tuple(range(1, pred.ndim))
    per_sample_sum = ((pred - target).pow(2) * mask).sum(dim=reduce_dims)
    per_sample_count = mask.expand_as(pred).sum(dim=reduce_dims)
    valid = (per_sample_count > 0).to(dtype=pred.dtype)
    per_sample_loss = per_sample_sum / per_sample_count.clamp_min(1.0)
    return (per_sample_loss * valid).sum() / valid.sum().clamp_min(1.0)


def masked_weighted_mse(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
    weight: Tensor,
    eps: float = 1.0e-8,
) -> Tensor:
    """Weighted x-space MSE with the same per-sample reduction as masked_mse."""
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    while mask.ndim < pred.ndim:
        mask = mask.unsqueeze(-1)
    while weight.ndim < pred.ndim:
        weight = weight.unsqueeze(-1)
    err2 = (pred - target).float().square()
    mask_f = mask.to(device=pred.device, dtype=err2.dtype)
    weight_f = weight.to(device=pred.device, dtype=err2.dtype)
    reduce_dims = tuple(range(1, pred.ndim))
    per_sample_sum = (err2 * mask_f * weight_f).sum(dim=reduce_dims)
    per_sample_count = mask_f.expand_as(err2).sum(dim=reduce_dims)
    valid = (per_sample_count > 0).to(dtype=err2.dtype)
    per_sample_loss = per_sample_sum / per_sample_count.clamp_min(float(eps))
    return (per_sample_loss * valid).sum() / valid.sum().clamp_min(1.0)


def _repeat_steps(x: Tensor, steps: int) -> Tensor:
    if steps == 1:
        return x
    return x.unsqueeze(0).expand(steps, *x.shape).reshape(steps * x.shape[0], *x.shape[1:])


class FixedRMSNorm(nn.Module):
    def __init__(self, dim: int, *, scale: float = 1.0, eps: float = 1.0e-6) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("FixedRMSNorm dim must be positive")
        if scale <= 0.0:
            raise ValueError("FixedRMSNorm scale must be positive")
        if eps <= 0.0:
            raise ValueError("FixedRMSNorm eps must be positive")
        self.dim = int(dim)
        self.scale = float(scale)
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        rms = x.float().square().mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x.float() / rms * self.scale).to(dtype=x.dtype)


def _num_2x_stages(factor: int) -> int:
    if factor < 1:
        raise ValueError("downsample factor must be positive")
    if factor & (factor - 1):
        raise ValueError("downsample_factor must be a power of 2 for progressive AE")
    return factor.bit_length() - 1


def _resolve_stage_depths(
    depths: tuple[int, ...] | list[int] | None,
    *,
    fallback_total: int,
    stages: int,
    name: str,
) -> tuple[int, ...]:
    if depths is not None:
        out = tuple(int(depth) for depth in depths)
        if len(out) != stages:
            raise ValueError(f"{name} must have {stages} entries for downsample_factor=2**{stages}")
    elif stages == 0:
        out = ()
    else:
        fallback_total = int(fallback_total)
        base = fallback_total // stages
        extra = fallback_total % stages
        out = tuple(base + (1 if idx >= stages - extra else 0) for idx in range(stages))
    if any(depth < 0 for depth in out):
        raise ValueError(f"{name} entries must be non-negative")
    return out


def _normalize_fusion_mode(value: str, *, name: str) -> str:
    mode = str(value).lower().replace("-", "_")
    if mode in {"sum", "additive"}:
        mode = "add"
    if mode in {"concat", "project_concat", "projected_fuse"}:
        mode = "projected_concat"
    if mode not in {"add", "projected_concat"}:
        raise ValueError(f"{name} must be 'add' or 'projected_concat'")
    return mode


def _resolve_unet_depths(value: tuple[int, ...] | list[int] | str | None) -> tuple[int, int, int, int, int]:
    if value is None:
        return (4, 4, 8, 4, 4)
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
        depths = tuple(int(part) for part in parts)
    else:
        depths = tuple(int(part) for part in value)
    if len(depths) != 5:
        raise ValueError("wave_fm_unet_depths must have five entries")
    if min(depths) < 1:
        raise ValueError("wave_fm_unet_depths entries must be positive")
    return depths  # type: ignore[return-value]


def sample_span_mask(
    batch_size: int,
    lengths: Tensor | int,
    *,
    min_ratio: float = 0.7,
    max_ratio: float = 1.0,
    device: torch.device | None = None,
    max_len: int | None = None,
) -> Tensor:
    if isinstance(lengths, int):
        lengths = torch.full((batch_size,), lengths, device=device, dtype=torch.long)
    else:
        lengths = lengths.to(device=device, dtype=torch.long)
    if not 0.0 < min_ratio <= max_ratio <= 1.0:
        raise ValueError("mask ratios must satisfy 0 < min <= max <= 1")

    max_len = int(lengths.max().item()) if max_len is None and lengths.numel() else int(max_len or 0)
    if max_len == 0:
        return torch.zeros((batch_size, 0), device=lengths.device, dtype=torch.bool)

    ratios = torch.empty((batch_size,), device=lengths.device).uniform_(min_ratio, max_ratio)
    spans = (lengths.float() * ratios).to(torch.long).clamp_min(1)
    spans = torch.minimum(spans, lengths.clamp_min(0))
    start_max = (lengths - spans).clamp_min(0)
    starts = (torch.rand((batch_size,), device=lengths.device) * (start_max + 1).float()).to(torch.long)
    positions = torch.arange(max_len, device=lengths.device)[None, :]
    mask = (positions >= starts[:, None]) & (positions < (starts + spans)[:, None])
    return mask & (positions < lengths[:, None])


def _prefix_suffix_mask(
    lengths: Tensor,
    valid_mask: Tensor,
    *,
    prefix_fraction_min: float | None = None,
    prefix_fraction_max: float,
) -> Tensor:
    device = valid_mask.device
    lengths = lengths.to(device=device, dtype=torch.long)
    min_fraction = float(prefix_fraction_max if prefix_fraction_min is None else prefix_fraction_min)
    max_fraction = float(prefix_fraction_max)
    if min_fraction == max_fraction:
        ratios = torch.full_like(lengths.float(), max_fraction)
    else:
        ratios = torch.empty_like(lengths.float()).uniform_(min_fraction, max_fraction)
    visible = (lengths.float() * ratios).round().to(torch.long).clamp(min=0)
    visible = torch.minimum(visible, lengths.clamp_min(0))
    positions = torch.arange(valid_mask.shape[1], device=device)[None, :]
    return (positions >= visible[:, None]) & valid_mask.to(dtype=torch.bool)


def _prefix_word_boundary_mask(
    lengths: Tensor,
    valid_mask: Tensor,
    *,
    word_patch_spans: Tensor | None,
    word_lengths: Tensor | None,
    prefix_fraction_min: float | None = None,
    prefix_fraction_max: float,
    downsample_factor: int,
    max_preroll_patches: int | None = None,
    max_prev_tail_patches: int | None = None,
) -> Tensor:
    if word_patch_spans is None or word_lengths is None:
        return valid_mask.to(dtype=torch.bool)
    if word_patch_spans.ndim != 3 or word_patch_spans.shape[-1] != 2:
        return valid_mask.to(dtype=torch.bool)

    device = valid_mask.device
    lengths = lengths.to(device=device, dtype=torch.long)
    word_patch_spans = word_patch_spans.to(device=device, dtype=torch.long)
    word_lengths = word_lengths.to(device=device, dtype=torch.long)
    factor = max(1, int(downsample_factor))
    max_preroll = factor if max_preroll_patches is None else int(max_preroll_patches)
    max_prev_tail = factor if max_prev_tail_patches is None else int(max_prev_tail_patches)
    if max_preroll < 0 or max_prev_tail < 0:
        raise ValueError("word-boundary preroll limits must be non-negative")

    min_fraction = float(prefix_fraction_max if prefix_fraction_min is None else prefix_fraction_min)
    max_fraction = float(prefix_fraction_max)
    if min_fraction == max_fraction:
        desired_fractions = torch.full_like(lengths.float(), max_fraction)
    else:
        desired_fractions = torch.empty_like(lengths.float()).uniform_(min_fraction, max_fraction)

    # With word spans available, do not fall back to a mid-word random prefix
    # for rows that have no safe candidate. Those rows become target-only.
    visible = torch.zeros_like(lengths)
    max_words = word_patch_spans.shape[1]
    for row in range(valid_mask.shape[0]):
        length = int(lengths[row].item())
        words = min(int(word_lengths[row].item()), max_words)
        if length <= 1 or words <= 1:
            continue

        best_visible: int | None = None
        best_distance: float | None = None
        desired = float(desired_fractions[row].item())
        for word_idx in range(1, words):
            target_start = int(word_patch_spans[row, word_idx, 0].item())
            target_end = int(word_patch_spans[row, word_idx, 1].item())
            prev_end = int(word_patch_spans[row, word_idx - 1, 1].item())
            if target_start < 0 or target_end <= target_start:
                continue

            boundary_patch = (target_start // factor) * factor
            boundary_latents = boundary_patch // factor
            if boundary_latents <= 0 or boundary_latents >= length:
                continue

            preroll = target_start - boundary_patch
            prev_tail = max(0, prev_end - boundary_patch) if prev_end >= 0 else 0
            if preroll > max_preroll or prev_tail > max_prev_tail:
                continue

            ratio = boundary_latents / max(length, 1)
            distance = abs(ratio - desired)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_visible = boundary_latents

        if best_visible is not None:
            visible[row] = best_visible

    positions = torch.arange(valid_mask.shape[1], device=device)[None, :]
    return (positions >= visible[:, None]) & valid_mask.to(device=device, dtype=torch.bool)


class WavePatchify(nn.Module):
    def __init__(self, patch_size: int = 160) -> None:
        super().__init__()
        if patch_size < 1:
            raise ValueError("patch_size must be positive")
        self.patch_size = int(patch_size)

    def patchify(self, wav: Tensor) -> tuple[Tensor, int]:
        if wav.ndim != 2:
            raise ValueError("wav must have shape [B, L]")
        batch, length = wav.shape
        patches = (length + self.patch_size - 1) // self.patch_size
        padded_len = patches * self.patch_size
        if padded_len != length:
            wav = F.pad(wav, (0, padded_len - length))
        return wav.reshape(batch, patches, self.patch_size), length

    def unpatchify(self, patches: Tensor, original_len: int | None = None) -> Tensor:
        if patches.ndim != 3:
            raise ValueError("patches must have shape [B, N, P]")
        wav = patches.reshape(patches.shape[0], patches.shape[1] * patches.shape[2])
        if original_len is not None:
            wav = wav[:, :original_len]
        return wav


class TargetEncoder(nn.Module):
    def __init__(
        self,
        *,
        patch_size: int,
        token_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        ffn_mult: int,
        dropout: float,
        rope_base: float,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(patch_size, token_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    token_dim,
                    heads=heads,
                    dim_head=dim_head,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                    rope_base=rope_base,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, x_patch: Tensor) -> Tensor:
        x = self.in_proj(x_patch)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def _use_fm_adaln(idx: int, depth: int, every: int) -> bool:
    return idx == 0 or idx == depth - 1 or idx % every == 0


class FMHead(nn.Module):
    def __init__(self, hidden_dim: int, token_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, token_dim))

    def forward(self, h: Tensor) -> Tensor:
        return self.net(h)


class TokenDownsample(nn.Module):
    """Swin-style 1D patch merging: concat neighboring tokens, then project."""

    def __init__(self, dim: int, factor: int) -> None:
        super().__init__()
        if factor < 1:
            raise ValueError("downsample factor must be positive")
        self.factor = int(factor)
        self.norm = nn.LayerNorm(dim * self.factor)
        self.proj = nn.Linear(dim * self.factor, dim)

    def forward(self, x: Tensor) -> Tensor:
        if self.factor == 1:
            return x
        batch, tokens, dim = x.shape
        pad = (-tokens) % self.factor
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        x = x.view(batch, x.shape[1] // self.factor, self.factor * dim)
        return self.proj(self.norm(x))


class TokenUpsample(nn.Module):
    """Linear token expanding: project one token back to neighboring tokens."""

    def __init__(self, dim: int, factor: int) -> None:
        super().__init__()
        if factor < 1:
            raise ValueError("upsample factor must be positive")
        self.factor = int(factor)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim * self.factor)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, target_tokens: int | None = None) -> Tensor:
        if self.factor == 1:
            out = x
        else:
            batch, tokens, dim = x.shape
            out = self.proj(self.norm(x)).view(batch, tokens * self.factor, dim)
            out = self.out_norm(out)
        if target_tokens is not None:
            out = out[:, :target_tokens]
        return out


class WaveDecoder(nn.Module):
    def __init__(
        self,
        *,
        patch_size: int,
        token_dim: int,
        hidden_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        ffn_mult: int,
        dropout: float,
        rope_base: float,
        head_type: str = "patch",
        istft_n_fft_factor: int = 4,
        istft_mag_clip: float = 100.0,
    ) -> None:
        super().__init__()
        self.patchify = WavePatchify(patch_size)
        self.head_type = str(head_type).lower().replace("-", "_")
        self.in_proj = nn.Linear(token_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim,
                    heads=heads,
                    dim_head=dim_head,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                    rope_base=rope_base,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        if self.head_type in {"patch", "wave_patch", "linear_patch"}:
            self.head_type = "patch"
            self.out_proj = nn.Linear(hidden_dim, patch_size)
        elif self.head_type in {"istft", "stft"}:
            if istft_n_fft_factor < 1:
                raise ValueError("istft_n_fft_factor must be positive")
            self.head_type = "istft"
            self.out_proj = ISTFTDecoderHead(
                dim=hidden_dim,
                hop_length=patch_size,
                n_fft=patch_size * int(istft_n_fft_factor),
                mag_clip=istft_mag_clip,
            )
        else:
            raise ValueError("decoder head must be 'patch' or 'istft'")

    def forward(self, z: Tensor, original_len: int | None = None) -> Tensor:
        h = self.in_proj(z)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        if self.head_type == "istft":
            return self.out_proj(h, original_len=original_len)
        patches = self.out_proj(h)
        return self.patchify.unpatchify(patches, original_len=original_len)


class DeterministicWaveDecoder(nn.Module):
    """Latent-token waveform decoder with selectable DiT or linear U-Net backbone."""

    def __init__(
        self,
        *,
        backbone: str,
        patch_size: int,
        token_dim: int,
        hidden_dim: int,
        depth: int,
        unet_depths: tuple[int, ...] | list[int] | str | None,
        heads: int,
        dim_head: int,
        ffn_mult: int,
        dropout: float,
        rope_base: float,
        input_fusion: str = "add",
        head_type: str = "patch",
        istft_n_fft_factor: int = 4,
        istft_mag_clip: float = 100.0,
    ) -> None:
        super().__init__()
        self.patchify = WavePatchify(patch_size)
        self.backbone = str(backbone).lower().replace("-", "_")
        if self.backbone in {"transformer", "flat", "standard", "dit"}:
            self.backbone = "dit"
        if self.backbone in {"linear_unet", "udit_unet", "linear_udit", "linear_udit_unet", "unet"}:
            self.backbone = "unet"
        if self.backbone not in {"dit", "unet"}:
            raise ValueError("decoder.backbone must be 'dit' or 'unet'")

        if self.backbone == "dit":
            self.decoder = WaveDecoder(
                patch_size=patch_size,
                token_dim=token_dim,
                hidden_dim=hidden_dim,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                ffn_mult=ffn_mult,
                dropout=dropout,
                rope_base=rope_base,
                head_type=head_type,
                istft_n_fft_factor=istft_n_fft_factor,
                istft_mag_clip=istft_mag_clip,
            )
        else:
            normalized_head = str(head_type).lower().replace("-", "_")
            if normalized_head not in {"patch", "wave_patch", "linear_patch"}:
                raise ValueError("unet deterministic decoder currently requires decoder.head='patch'")
            self.decoder = LinearUDiTUNetWaveFMDecoder(
                patch_size=patch_size,
                cond_dim=token_dim,
                hidden_dim=hidden_dim,
                depths=unet_depths,
                adaln_every=2,
                heads=heads,
                dim_head=dim_head,
                ffn_mult=ffn_mult,
                dropout=dropout,
                rope_base=rope_base,
                input_fusion=input_fusion,
            )

    def forward(self, z: Tensor, original_len: int | None = None) -> Tensor:
        if self.backbone == "dit":
            return self.decoder(z, original_len=original_len)
        target_patches = z.shape[1]
        if original_len is not None:
            target_patches = (int(original_len) + self.patchify.patch_size - 1) // self.patchify.patch_size
            z = z[:, :target_patches]
        wave_x = z.new_zeros((z.shape[0], z.shape[1], self.patchify.patch_size))
        t = z.new_zeros((z.shape[0],))
        patches = self.decoder(wave_x, z, t)
        return self.patchify.unpatchify(patches[:, :target_patches], original_len=original_len)


@dataclass
class MaskedFlowWaveTokenTTSConfig:
    vocab_size: int
    architecture: str = "latent_tts"
    pad_id: int = 0
    sample_rate: int = 16000
    latent_hz: float = 25.0
    patch_size: int = 160
    downsample_factor: int = 4
    token_dim: int = 64
    latent_dim: int | None = None
    hidden_dim: int = 128
    text_dim: int = 96
    text_conv_layers: int = 4
    text_conv_mult: int = 2
    encoder_same_depth: int = 1
    encoder_down_depths: tuple[int, ...] | list[int] | None = None
    fm_depth: int = 2
    fm_hidden_dim: int | None = None
    fm_adaln_every: int = 2
    fm_heads: int | None = None
    fm_dim_head: int | None = None
    fm_ffn_mult: int | None = None
    wave_fm_depth: int = 6
    wave_fm_hidden_dim: int | None = None
    wave_fm_adaln_every: int = 2
    wave_fm_heads: int | None = None
    wave_fm_dim_head: int | None = None
    wave_fm_ffn_mult: int | None = None
    prior_fm_input_fusion: str = "add"
    wave_fm_backbone: str = "transformer"
    wave_fm_input_fusion: str = "add"
    wave_fm_unet_depths: tuple[int, ...] | list[int] | str | None = None
    decoder_backbone: str = "dit"
    decoder_input_fusion: str = "add"
    decoder_unet_depths: tuple[int, ...] | list[int] | str | None = None
    decoder_up_depths: tuple[int, ...] | list[int] | None = None
    decoder_same_depth: int = 1
    build_deterministic_decoder: bool = True
    heads: int = 4
    dim_head: int = 32
    ffn_mult: int = 4
    dropout: float = 0.0
    rope_base: float = 10000.0
    latent_norm_mode: str = "layernorm"
    latent_norm_scale: float = 1.0
    latent_norm_eps: float = 1.0e-6
    prediction: str = "v_pred_v_loss"
    fm_input_mode: str = "joint_seq"
    text_condition_mode: str = "sentence"
    megatts_alignment_mode: str = "dense"
    megatts_anchor_mode: str = "center"
    megatts_anchor_ratio: float = 0.25
    sentence_text_mode: str = "prefix"
    word_sparse_width: int = 0
    use_target_token: bool = False
    use_word_audio_boundary: bool = True
    target_id: int | None = None
    eos_id: int | None = None
    mask_source: str = "random_span"
    mask_prefix_word_boundary_max_preroll_patches: int | None = None
    mask_prefix_word_boundary_max_prev_tail_patches: int | None = None
    mask_prefix_fraction_min: float | None = None
    mask_prefix_fraction_max: float = 0.30
    mask_ratio_min: float = 0.7
    mask_ratio_max: float = 1.0
    cfg_dropout: float = 0.0
    prior_cfg_dropout: float | None = None
    prior_cfg_dropout_mode: str = "drop_text"
    wave_cfg_dropout: float = 0.0
    decoder_latent_noise_mode: str = "unite"
    decoder_latent_noise_std: float = 0.0
    decoder_latent_noise_t_start: float = 0.90
    decoder_latent_noise_prob: float = 1.0
    decoder_recon_latent_mode: str = "clean"
    decoder_pred_gt_mix: float = 0.5
    decoder_objective: str = "wave"
    decoder_head: str = "patch"
    decoder_istft_n_fft_factor: int = 4
    decoder_istft_mag_clip: float = 100.0
    target_grad_scale: float = 1.0
    split_target_encoder_ema: bool = False
    split_target_encoder_ema_decay: float = 0.999
    split_target_encoder_ema_start_step: int = 0
    flow_steps_per_recon: int = 4
    flow_xpred_denom_min: float = 2.0e-2
    flow_t_sampling: str = "logit_normal"
    flow_lognorm_mu: float = 0.0
    flow_lognorm_sigma: float = 1.0
    flow_timestep_shift: float = 0.5
    flow_inference_timestep_mapping: str = "power"
    flow_inference_timestep_power: float = 2.0
    flow_inference_timestep_shift: float = 3.0
    flow_inference_sway_sampling_coef: float | None = None
    lambda_flow: float = 1.0
    lambda_wave_fm: float = 1.0
    lambda_recon: float = 1.0
    lambda_mel: float = 0.05
    lambda_latent_reg: float = 0.0
    latent_reg_mode: str = "none"
    latent_reg_target: str = "z_raw"
    latent_reg_sketch_dim: int = 256
    latent_reg_projections: int = 256
    latent_reg_knots: int = 17
    latent_reg_t_max: float = 3.0
    latent_reg_var_floor: float = 0.7
    latent_reg_seq_std_ceiling: float = 1.5
    latent_reg_seq_std_ceiling_weight: float = 1.0
    recon_loss: str = "l1"
    waveform_scale: float = 9.0
    mel_max_chunk_samples: int = 250_000
    mel_max_stft_samples: int = 65_536
    mel_stft_device: str = "cuda"

    def __post_init__(self) -> None:
        self.architecture = str(self.architecture).lower().replace("-", "_")
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.latent_hz <= 0:
            raise ValueError("latent_hz must be positive")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.downsample_factor < 1:
            raise ValueError("downsample_factor must be positive")
        if self.latent_dim is None:
            self.latent_dim = int(self.token_dim)
        self.latent_dim = int(self.latent_dim)
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        self.heads = int(self.heads)
        if self.heads < 1:
            raise ValueError("heads must be positive")
        self.dim_head = int(self.dim_head)
        if self.dim_head < 1 or self.dim_head % 2:
            raise ValueError("dim_head must be a positive even integer")
        self.ffn_mult = int(self.ffn_mult)
        if self.ffn_mult < 1:
            raise ValueError("ffn_mult must be positive")
        if self.fm_hidden_dim is None:
            self.fm_hidden_dim = int(self.hidden_dim)
        self.fm_hidden_dim = int(self.fm_hidden_dim)
        if self.fm_hidden_dim < 1:
            raise ValueError("fm_hidden_dim must be positive")
        if self.fm_heads is None:
            self.fm_heads = int(self.heads)
        self.fm_heads = int(self.fm_heads)
        if self.fm_heads < 1:
            raise ValueError("fm_heads must be positive")
        if self.fm_dim_head is None:
            self.fm_dim_head = int(self.dim_head)
        self.fm_dim_head = int(self.fm_dim_head)
        if self.fm_dim_head < 1 or self.fm_dim_head % 2:
            raise ValueError("fm_dim_head must be a positive even integer")
        if self.fm_ffn_mult is None:
            self.fm_ffn_mult = int(self.ffn_mult)
        self.fm_ffn_mult = int(self.fm_ffn_mult)
        if self.fm_ffn_mult < 1:
            raise ValueError("fm_ffn_mult must be positive")
        if self.wave_fm_hidden_dim is None:
            self.wave_fm_hidden_dim = int(self.fm_hidden_dim)
        self.wave_fm_hidden_dim = int(self.wave_fm_hidden_dim)
        if self.wave_fm_hidden_dim < 1:
            raise ValueError("wave_fm_hidden_dim must be positive")
        if self.wave_fm_heads is None:
            self.wave_fm_heads = int(self.heads)
        self.wave_fm_heads = int(self.wave_fm_heads)
        if self.wave_fm_heads < 1:
            raise ValueError("wave_fm_heads must be positive")
        if self.wave_fm_dim_head is None:
            self.wave_fm_dim_head = int(self.dim_head)
        self.wave_fm_dim_head = int(self.wave_fm_dim_head)
        if self.wave_fm_dim_head < 1 or self.wave_fm_dim_head % 2:
            raise ValueError("wave_fm_dim_head must be a positive even integer")
        if self.wave_fm_ffn_mult is None:
            self.wave_fm_ffn_mult = int(self.ffn_mult)
        self.wave_fm_ffn_mult = int(self.wave_fm_ffn_mult)
        if self.wave_fm_ffn_mult < 1:
            raise ValueError("wave_fm_ffn_mult must be positive")
        self.prior_fm_input_fusion = _normalize_fusion_mode(
            self.prior_fm_input_fusion,
            name="prior_fm_input_fusion",
        )
        self.wave_fm_backbone = str(self.wave_fm_backbone).lower().replace("-", "_")
        if self.wave_fm_backbone in {"flat", "dit", "standard"}:
            self.wave_fm_backbone = "transformer"
        if self.wave_fm_backbone in {"linear_unet", "udit_unet", "linear_udit"}:
            self.wave_fm_backbone = "linear_udit_unet"
        if self.wave_fm_backbone not in {"transformer", "linear_udit_unet"}:
            raise ValueError("wave_fm_backbone must be 'transformer' or 'linear_udit_unet'")
        self.wave_fm_input_fusion = _normalize_fusion_mode(
            self.wave_fm_input_fusion,
            name="wave_fm_input_fusion",
        )
        self.wave_fm_unet_depths = _resolve_unet_depths(self.wave_fm_unet_depths)
        self.decoder_backbone = str(self.decoder_backbone).lower().replace("-", "_")
        if self.decoder_backbone in {"transformer", "flat", "standard"}:
            self.decoder_backbone = "dit"
        if self.decoder_backbone in {"linear_unet", "udit_unet", "linear_udit", "linear_udit_unet"}:
            self.decoder_backbone = "unet"
        if self.decoder_backbone not in {"dit", "unet"}:
            raise ValueError("decoder_backbone must be 'dit' or 'unet'")
        self.decoder_input_fusion = _normalize_fusion_mode(
            self.decoder_input_fusion,
            name="decoder_input_fusion",
        )
        self.decoder_unet_depths = _resolve_unet_depths(
            self.wave_fm_unet_depths if self.decoder_unet_depths is None else self.decoder_unet_depths
        )
        self.latent_norm_mode = str(self.latent_norm_mode).lower().replace("-", "_")
        if self.latent_norm_mode in {"layer_norm", "ln"}:
            self.latent_norm_mode = "layernorm"
        if self.latent_norm_mode in {"rms", "rmsnorm", "fixed_rmsnorm"}:
            self.latent_norm_mode = "fixed_rms"
        if self.latent_norm_mode not in {"layernorm", "fixed_rms", "identity"}:
            raise ValueError("latent_norm_mode must be 'layernorm', 'fixed_rms', or 'identity'")
        self.latent_norm_scale = float(self.latent_norm_scale)
        if self.latent_norm_scale <= 0.0:
            raise ValueError("latent_norm_scale must be positive")
        self.latent_norm_eps = float(self.latent_norm_eps)
        if self.latent_norm_eps <= 0.0:
            raise ValueError("latent_norm_eps must be positive")
        stages = _num_2x_stages(self.downsample_factor)
        self.encoder_down_depths = _resolve_stage_depths(
            self.encoder_down_depths,
            fallback_total=stages,
            stages=stages,
            name="encoder_down_depths",
        )
        if self.build_deterministic_decoder:
            self.decoder_up_depths = _resolve_stage_depths(
                self.decoder_up_depths,
                fallback_total=stages,
                stages=stages,
                name="decoder_up_depths",
            )
        else:
            self.decoder_up_depths = ()
        if min(
            self.text_conv_layers,
            self.encoder_same_depth,
            self.fm_depth,
            self.wave_fm_depth,
            self.decoder_same_depth,
        ) < 0:
            raise ValueError("depths must be non-negative")
        if self.text_conv_mult < 1:
            raise ValueError("text_conv_mult must be positive")
        if self.fm_adaln_every < 1:
            raise ValueError("fm_adaln_every must be positive")
        if self.wave_fm_adaln_every < 1:
            raise ValueError("wave_fm_adaln_every must be positive")
        patch_rate = self.sample_rate / self.patch_size
        expected_latent_hz = patch_rate / self.downsample_factor
        if abs(expected_latent_hz - self.latent_hz) > 1.0e-6:
            raise ValueError(
                "latent_hz must equal sample_rate / patch_size / downsample_factor; "
                f"got latent_hz={self.latent_hz}, sample_rate={self.sample_rate}, "
                f"patch_size={self.patch_size}, downsample_factor={self.downsample_factor}"
            )
        if self.prediction not in {
            "v_pred_v_loss",
            "x_pred_v_loss",
            "x_pred_v_loss_weight",
            "x_pred_x_loss",
        }:
            raise ValueError(
                "prediction must be 'v_pred_v_loss', 'x_pred_v_loss', "
                "'x_pred_v_loss_weight', or 'x_pred_x_loss'"
            )
        self.fm_input_mode = str(self.fm_input_mode).lower().replace("-", "_")
        if self.fm_input_mode in {"joint", "joint_att", "joint_attention"}:
            self.fm_input_mode = "joint_seq"
        if self.fm_input_mode in {"megatts", "megatts_local", "local_add"}:
            self.fm_input_mode = "megatts_add"
        if self.fm_input_mode not in {"joint_seq", "megatts_add", "channel_concat"}:
            raise ValueError("fm_input_mode must be 'joint_seq', 'megatts_add', or 'channel_concat'")
        self.megatts_alignment_mode = str(self.megatts_alignment_mode).lower().replace("-", "_")
        if self.megatts_alignment_mode in {"expand", "full"}:
            self.megatts_alignment_mode = "dense"
        if self.megatts_alignment_mode not in {"dense", "sparse"}:
            raise ValueError("megatts_alignment_mode must be 'dense' or 'sparse'")
        self.megatts_anchor_mode = str(self.megatts_anchor_mode).lower().replace("-", "_")
        if self.megatts_anchor_mode in {"center_window", "near_center", "center_random"}:
            self.megatts_anchor_mode = "center"
        if self.megatts_anchor_mode in {"random", "full", "full_span"}:
            self.megatts_anchor_mode = "uniform"
        if self.megatts_anchor_mode not in {"center", "uniform"}:
            raise ValueError("megatts_anchor_mode must be 'center' or 'uniform'")
        self.megatts_anchor_ratio = float(self.megatts_anchor_ratio)
        if not (0.0 < self.megatts_anchor_ratio <= 1.0):
            raise ValueError("megatts_anchor_ratio must satisfy 0 < ratio <= 1")
        self.text_condition_mode = str(self.text_condition_mode).lower().replace("-", "_")
        if self.text_condition_mode in {"mega_tts_expand", "megatts"}:
            self.text_condition_mode = "megatts_expand"
        if self.text_condition_mode in {"mega_tts_sparse", "megatts_sparse", "sparse_megatts"}:
            self.text_condition_mode = "megatts_expand"
            self.megatts_alignment_mode = "sparse"
        if self.text_condition_mode not in {"sentence", "word", "word_sparse", "megatts_expand"}:
            raise ValueError(
                "text_condition_mode must be 'sentence', 'word', 'word_sparse', "
                "'megatts_expand', or 'megatts_sparse'"
            )
        if self.sentence_text_mode not in {"prefix", "resample"}:
            raise ValueError("sentence_text_mode must be 'prefix' or 'resample'")
        if self.word_sparse_width < 0:
            raise ValueError("word_sparse_width must be non-negative")
        if self.target_id is not None and self.target_id < 0:
            raise ValueError("target_id must be non-negative or None")
        if self.eos_id is not None and self.eos_id < 0:
            raise ValueError("eos_id must be non-negative or None")
        if self.mask_source not in {"batch_target", "prefix", "random_span", "full"}:
            raise ValueError("mask_source must be 'batch_target', 'prefix', 'random_span', or 'full'")
        if (
            self.mask_prefix_word_boundary_max_preroll_patches is not None
            and self.mask_prefix_word_boundary_max_preroll_patches < 0
        ):
            raise ValueError("mask_prefix_word_boundary_max_preroll_patches must be non-negative")
        if (
            self.mask_prefix_word_boundary_max_prev_tail_patches is not None
            and self.mask_prefix_word_boundary_max_prev_tail_patches < 0
        ):
            raise ValueError("mask_prefix_word_boundary_max_prev_tail_patches must be non-negative")
        if self.mask_prefix_fraction_min is None:
            self.mask_prefix_fraction_min = self.mask_prefix_fraction_max
        if not 0.0 <= self.mask_prefix_fraction_min <= self.mask_prefix_fraction_max < 1.0:
            raise ValueError(
                "prefix fractions must satisfy 0 <= mask_prefix_fraction_min "
                "<= mask_prefix_fraction_max < 1"
            )
        if not 0.0 < self.mask_ratio_min <= self.mask_ratio_max <= 1.0:
            raise ValueError("mask ratios must satisfy 0 < min <= max <= 1")
        if not 0.0 <= self.cfg_dropout <= 1.0:
            raise ValueError("cfg_dropout must be in [0, 1]")
        if self.prior_cfg_dropout is None:
            self.prior_cfg_dropout = float(self.cfg_dropout)
        self.prior_cfg_dropout = float(self.prior_cfg_dropout)
        if not 0.0 <= self.prior_cfg_dropout <= 1.0:
            raise ValueError("prior_cfg_dropout must be in [0, 1]")
        self.prior_cfg_dropout_mode = str(self.prior_cfg_dropout_mode).lower().replace("-", "_")
        if self.prior_cfg_dropout_mode in {"text", "text_only"}:
            self.prior_cfg_dropout_mode = "drop_text"
        if self.prior_cfg_dropout_mode in {
            "text_prompt",
            "drop_text_prompt",
            "both",
            "all",
            "text_audio",
            "drop_audio_text",
            "text_speech",
            "drop_text_speech",
            "text_prompt_speech",
        }:
            self.prior_cfg_dropout_mode = "drop_text_audio"
        if self.prior_cfg_dropout_mode not in {"drop_text", "drop_text_audio"}:
            raise ValueError("prior_cfg_dropout_mode must be 'drop_text' or 'drop_text_audio'")
        self.wave_cfg_dropout = float(self.wave_cfg_dropout)
        if not 0.0 <= self.wave_cfg_dropout <= 1.0:
            raise ValueError("wave_cfg_dropout must be in [0, 1]")
        self.decoder_latent_noise_mode = str(self.decoder_latent_noise_mode).lower().replace("-", "_")
        if self.decoder_latent_noise_mode in {"off", "false", "0"}:
            self.decoder_latent_noise_mode = "none"
        elif self.decoder_latent_noise_mode in {"std", "gaussian"}:
            self.decoder_latent_noise_mode = "additive"
        elif self.decoder_latent_noise_mode in {"ul", "ul_fixed", "fixed", "fixed_noise", "fixed_vp"}:
            self.decoder_latent_noise_mode = "ul_fixed"
        elif self.decoder_latent_noise_mode in {"variance_preserving", "vp_noise"}:
            self.decoder_latent_noise_mode = "vp"
        elif self.decoder_latent_noise_mode in {"t", "path", "icplan"}:
            self.decoder_latent_noise_mode = "flow_t"
        elif self.decoder_latent_noise_mode in {"unite_path", "unite_noising"}:
            self.decoder_latent_noise_mode = "unite"
        if self.decoder_latent_noise_mode not in {"none", "additive", "vp", "ul_fixed", "flow_t", "unite"}:
            raise ValueError(
                "decoder_latent_noise_mode must be 'none', 'additive', 'vp', 'ul_fixed', 'flow_t', or 'unite'"
            )
        if self.decoder_latent_noise_std < 0.0:
            raise ValueError("decoder_latent_noise_std must be non-negative")
        if not 0.0 <= self.decoder_latent_noise_t_start <= 1.0:
            raise ValueError("decoder_latent_noise_t_start must be in [0, 1]")
        if not 0.0 <= self.decoder_latent_noise_prob <= 1.0:
            raise ValueError("decoder_latent_noise_prob must be in [0, 1]")
        self.decoder_recon_latent_mode = str(self.decoder_recon_latent_mode).lower().replace("-", "_")
        if self.decoder_recon_latent_mode in {"oracle", "teacher"}:
            self.decoder_recon_latent_mode = "clean"
        elif self.decoder_recon_latent_mode in {"pred_gt", "fm_gt_mix", "fm_pred_gt"}:
            self.decoder_recon_latent_mode = "fm_pred_gt_mix"
        if self.decoder_recon_latent_mode not in {"clean", "fm_pred_gt_mix"}:
            raise ValueError("decoder_recon_latent_mode must be 'clean' or 'fm_pred_gt_mix'")
        if not 0.0 <= self.decoder_pred_gt_mix <= 1.0:
            raise ValueError("decoder_pred_gt_mix must be in [0, 1]")
        self.decoder_objective = str(self.decoder_objective).lower().replace("-", "_")
        if self.decoder_objective in {"standard", "deterministic"}:
            self.decoder_objective = "wave"
        if self.decoder_objective in {"decoder_fm", "fm_decoder"}:
            self.decoder_objective = "wave_fm"
        if self.decoder_objective not in {"wave", "wave_fm"}:
            raise ValueError("decoder_objective must be 'wave' or 'wave_fm'")
        self.decoder_head = str(self.decoder_head).lower().replace("-", "_")
        if self.decoder_head in {"wave_patch", "linear_patch"}:
            self.decoder_head = "patch"
        elif self.decoder_head == "stft":
            self.decoder_head = "istft"
        if self.decoder_head not in {"patch", "istft"}:
            raise ValueError("decoder_head must be 'patch' or 'istft'")
        if self.decoder_istft_n_fft_factor < 1:
            raise ValueError("decoder_istft_n_fft_factor must be positive")
        if self.decoder_istft_mag_clip <= 0.0:
            raise ValueError("decoder_istft_mag_clip must be positive")
        if self.recon_loss not in {"l1", "l2"}:
            raise ValueError("recon_loss must be 'l1' or 'l2'")
        if not 0.0 <= self.target_grad_scale <= 1.0:
            raise ValueError("target_grad_scale must be in [0, 1]")
        self.split_target_encoder_ema = bool(self.split_target_encoder_ema)
        if not 0.0 <= self.split_target_encoder_ema_decay < 1.0:
            raise ValueError("split_target_encoder_ema_decay must be in [0, 1)")
        if self.split_target_encoder_ema_start_step < 0:
            raise ValueError("split_target_encoder_ema_start_step must be non-negative")
        if self.flow_steps_per_recon < 1:
            raise ValueError("flow_steps_per_recon must be positive")
        if self.flow_xpred_denom_min <= 0.0:
            raise ValueError("flow_xpred_denom_min must be positive")
        if self.flow_t_sampling not in {"uniform", "logit_normal"}:
            raise ValueError("flow_t_sampling must be 'uniform' or 'logit_normal'")
        if self.flow_lognorm_sigma <= 0.0:
            raise ValueError("flow_lognorm_sigma must be positive")
        if self.flow_timestep_shift < 0.0:
            raise ValueError("flow_timestep_shift must be non-negative")
        self.flow_inference_timestep_mapping = str(self.flow_inference_timestep_mapping).lower().replace("-", "_")
        if self.flow_inference_timestep_mapping in {"none", "linear"}:
            self.flow_inference_timestep_mapping = "uniform"
        if self.flow_inference_timestep_mapping not in {
            "uniform",
            "power",
            "sway_sampling",
            "legacy_shifted_linear",
        }:
            raise ValueError(
                "flow_inference_timestep_mapping must be one of "
                "uniform, power, sway_sampling, legacy_shifted_linear"
            )
        if self.flow_inference_timestep_power <= 0.0:
            raise ValueError("flow_inference_timestep_power must be positive")
        if self.flow_inference_timestep_shift <= 0.0:
            raise ValueError("flow_inference_timestep_shift must be positive")
        if min(self.lambda_flow, self.lambda_wave_fm, self.lambda_recon, self.lambda_mel, self.lambda_latent_reg) < 0.0:
            raise ValueError("loss weights must be non-negative")
        self.latent_reg_mode = str(self.latent_reg_mode).lower().replace("-", "_")
        if self.latent_reg_mode in {"off", "false", "0"}:
            self.latent_reg_mode = "none"
        if self.latent_reg_mode in {"strong_sigreg", "standard_sigreg"}:
            self.latent_reg_mode = "sigreg"
        if self.latent_reg_mode in {"target_sequence_hinge_sigreg", "seq_hinge_sigreg"}:
            self.latent_reg_mode = "target_seq_hinge_sigreg"
        if self.latent_reg_mode in {"target_sequence_band_hinge_sigreg", "seq_band_hinge_sigreg"}:
            self.latent_reg_mode = "target_seq_band_hinge_sigreg"
        if self.latent_reg_mode in {"vis_reg", "variance_invariance_sketching"}:
            self.latent_reg_mode = "visreg"
        if self.latent_reg_mode not in {
            "none",
            "weak_sigreg",
            "sigreg",
            "hinge_sigreg",
            "target_seq_hinge_sigreg",
            "target_seq_band_hinge_sigreg",
            "visreg",
        }:
            raise ValueError(
                "latent_reg_mode must be 'none', 'weak_sigreg', 'sigreg', "
                "'hinge_sigreg', 'target_seq_hinge_sigreg', "
                "'target_seq_band_hinge_sigreg', or 'visreg'"
            )
        self.latent_reg_target = str(self.latent_reg_target).lower().replace("-", "_")
        if self.latent_reg_target not in {"z_raw", "z_clean", "z_flow_target", "z_pred"}:
            raise ValueError("latent_reg_target must be 'z_raw', 'z_clean', 'z_flow_target', or 'z_pred'")
        if min(self.latent_reg_sketch_dim, self.latent_reg_projections, self.latent_reg_knots) < 1:
            raise ValueError("latent regularization dimensions/counts must be positive")
        if self.latent_reg_knots < 2:
            raise ValueError("latent_reg_knots must be at least 2")
        if self.latent_reg_t_max <= 0.0:
            raise ValueError("latent_reg_t_max must be positive")
        if self.latent_reg_var_floor <= 0.0:
            raise ValueError("latent_reg_var_floor must be positive")
        if self.latent_reg_seq_std_ceiling <= 0.0:
            raise ValueError("latent_reg_seq_std_ceiling must be positive")
        if self.latent_reg_seq_std_ceiling_weight < 0.0:
            raise ValueError("latent_reg_seq_std_ceiling_weight must be non-negative")
        if self.waveform_scale <= 0.0:
            raise ValueError("waveform_scale must be positive")
        if min(self.mel_max_chunk_samples, self.mel_max_stft_samples) < 0:
            raise ValueError("mel chunk limits must be non-negative")
        if self.mel_stft_device not in {"cuda", "cpu", "conv"}:
            raise ValueError("mel_stft_device must be 'cuda', 'cpu', or 'conv'")


class MaskedFlowWaveTokenTTS(nn.Module):
    def __init__(self, config: MaskedFlowWaveTokenTTSConfig) -> None:
        super().__init__()
        self.config = config
        self.patchify = WavePatchify(config.patch_size)
        self.text_encoder = TextEncoder(
            vocab_size=config.vocab_size,
            text_dim=config.text_dim,
            pad_id=config.pad_id,
            conv_layers=config.text_conv_layers,
            conv_mult=config.text_conv_mult,
        )
        self.target_encoder = TargetEncoder(
            patch_size=config.patch_size,
            token_dim=config.token_dim,
            depth=config.encoder_same_depth,
            heads=config.heads,
            dim_head=config.dim_head,
            ffn_mult=config.ffn_mult,
            dropout=config.dropout,
            rope_base=config.rope_base,
        )
        self.token_norm = nn.LayerNorm(config.token_dim)
        self.encoder_downsamples = nn.ModuleList(
            [TokenDownsample(config.token_dim, 2) for _ in config.encoder_down_depths]
        )
        self.encoder_down_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        TransformerBlock(
                            config.token_dim,
                            heads=config.heads,
                            dim_head=config.dim_head,
                            ffn_mult=config.ffn_mult,
                            dropout=config.dropout,
                            rope_base=config.rope_base,
                        )
                        for _ in range(depth)
                    ]
                )
                for depth in config.encoder_down_depths
            ]
        )
        self.encoder_latent_norm = nn.LayerNorm(config.token_dim)
        self.encoder_to_latent = nn.Linear(config.token_dim, config.latent_dim)
        self.latent_norm = self._build_latent_norm(config)
        self.fm_encoder: nn.Module | None = None
        self.fm_head: nn.Module | None = None
        if config.build_deterministic_decoder:
            self.decoder_up_blocks = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            TransformerBlock(
                                config.token_dim,
                                heads=config.heads,
                                dim_head=config.dim_head,
                                ffn_mult=config.ffn_mult,
                                dropout=config.dropout,
                                rope_base=config.rope_base,
                            )
                            for _ in range(depth)
                        ]
                    )
                    for depth in config.decoder_up_depths
                ]
            )
            self.decoder_upsamples = nn.ModuleList(
                [TokenUpsample(config.token_dim, 2) for _ in config.decoder_up_depths]
            )
            self.decoder_latent_norm = nn.LayerNorm(config.latent_dim)
            self.latent_to_decoder = nn.Linear(config.latent_dim, config.token_dim)
            self.decoder_token_norm = nn.LayerNorm(config.token_dim)
            self.decoder = DeterministicWaveDecoder(
                backbone=config.decoder_backbone,
                patch_size=config.patch_size,
                token_dim=config.token_dim,
                hidden_dim=config.hidden_dim,
                depth=config.decoder_same_depth,
                unet_depths=config.decoder_unet_depths,
                heads=config.heads,
                dim_head=config.dim_head,
                ffn_mult=config.ffn_mult,
                dropout=config.dropout,
                rope_base=config.rope_base,
                input_fusion=config.decoder_input_fusion,
                head_type=config.decoder_head,
                istft_n_fft_factor=config.decoder_istft_n_fft_factor,
                istft_mag_clip=config.decoder_istft_mag_clip,
            )
        else:
            self.decoder_up_blocks = nn.ModuleList()
            self.decoder_upsamples = nn.ModuleList()
            self.decoder_latent_norm = nn.Identity()
            self.latent_to_decoder = nn.Identity()
            self.decoder_token_norm = nn.Identity()
            self.decoder = nn.Identity()
        self.mel_loss_fn = MelSpectrogramLoss(
            sample_rate=config.sample_rate,
            max_chunk_samples=config.mel_max_chunk_samples,
            max_stft_samples=config.mel_max_stft_samples,
            stft_device=config.mel_stft_device,
        )

    def _build_latent_norm(self, config: MaskedFlowWaveTokenTTSConfig) -> nn.Module:
        if config.latent_norm_mode == "layernorm":
            return nn.LayerNorm(config.latent_dim)
        if config.latent_norm_mode == "fixed_rms":
            return FixedRMSNorm(
                config.latent_dim,
                scale=config.latent_norm_scale,
                eps=config.latent_norm_eps,
            )
        if config.latent_norm_mode == "identity":
            return nn.Identity()
        raise ValueError(f"unknown latent_norm_mode: {config.latent_norm_mode}")

    def forward(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        raise NotImplementedError("Use LatentTTSSplit for current training.")

    @torch.no_grad()
    def generate(
        self,
        prompt_wav: Tensor,
        text_token_ids: Tensor,
        *,
        target_num_tokens: int,
        num_steps: int = 16,
        text_token_lengths: Tensor | None = None,
        cfg_strength: float = 0.0,
        cfg_rescale: float = 0.0,
        solver: str = "euler",
    ) -> Tensor:
        del prompt_wav, text_token_ids, target_num_tokens, num_steps
        del text_token_lengths, cfg_strength, cfg_rescale, solver
        raise NotImplementedError("Use infer/infer_latentts_split.py for current inference.")

    def encode_wave(self, wav: Tensor) -> Tensor:
        patches, _ = self.patchify.patchify(self._scale_waveform(wav.float()))
        patch_tokens = self.token_norm(self.target_encoder(patches))
        return self._encode_patch_tokens(patch_tokens)

    def decode_tokens(self, z: Tensor, original_len: int | None = None) -> Tensor:
        target_patches = None
        if original_len is not None:
            target_patches = (original_len + self.patchify.patch_size - 1) // self.patchify.patch_size
        z = self.decoder_latent_norm(z)
        z = self.decoder_token_norm(self.latent_to_decoder(z))
        for blocks, upsample in zip(self.decoder_up_blocks, self.decoder_upsamples, strict=True):
            for block in blocks:
                z = block(z)
            z = upsample(z)
        if target_patches is not None:
            z = z[:, :target_patches]
        return self.decoder(z, original_len=original_len)

    def decode_tokens_raw(self, z: Tensor, original_len: int | None = None) -> Tensor:
        return self._unscale_waveform(self.decode_tokens(z, original_len=original_len))

    def _teacher_reconstruction_losses(
        self,
        z_clean: Tensor,
        z_pred: Tensor,
        wav_clean: Tensor,
        wav_clean_raw: Tensor,
        valid_sample_mask: Tensor,
        valid_latent_mask: Tensor | None,
        latent_target_mask: Tensor,
        *,
        original_len: int,
        z_recon_gt: Tensor | None = None,
        z_fm_input: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        zero = wav_clean.new_tensor(0.0)
        if self.config.lambda_recon == 0.0 and self.config.lambda_mel == 0.0:
            empty = wav_clean.new_empty(0)
            return empty, empty, zero, zero, zero, zero, zero, zero, empty
        z_source = self._decoder_recon_latent_source(
            z_clean,
            z_pred,
            latent_target_mask,
            valid_latent_mask=valid_latent_mask,
            z_recon_gt=z_recon_gt,
            z_fm_input=z_fm_input,
        )
        z_decode, decoder_noise_rms, decoder_noise_applied_fraction = self._apply_decoder_latent_noise(
            z_source,
            valid_latent_mask,
        )
        teacher_waveform_model = self.decode_tokens(z_decode, original_len=original_len)
        teacher_waveform = self._unscale_waveform(teacher_waveform_model)
        recon_loss = (
            self._wave_loss(teacher_waveform_model, wav_clean, valid_sample_mask)
            if self.config.lambda_recon != 0.0
            else zero
        )
        mel_loss = (
            self.mel_loss_fn(teacher_waveform_model, wav_clean, sample_mask=valid_sample_mask)
            if self.config.lambda_mel != 0.0
            else zero
        )
        with torch.no_grad():
            recon_loss_raw_metric = (
                self._wave_loss(teacher_waveform.detach(), wav_clean_raw, valid_sample_mask)
                if self.config.lambda_recon != 0.0
                else zero
            )
            mel_loss_raw_metric = mel_loss.detach()
        return (
            teacher_waveform,
            teacher_waveform_model,
            recon_loss,
            mel_loss,
            recon_loss_raw_metric.detach(),
            mel_loss_raw_metric.detach(),
            decoder_noise_rms.detach(),
            decoder_noise_applied_fraction.detach(),
            z_decode.detach(),
        )

    def _decoder_recon_latent_source(
        self,
        z_clean: Tensor,
        z_pred: Tensor,
        latent_target_mask: Tensor,
        *,
        valid_latent_mask: Tensor | None,
        z_recon_gt: Tensor | None,
        z_fm_input: Tensor | None,
    ) -> Tensor:
        mode = str(self.config.decoder_recon_latent_mode)
        if mode == "clean":
            return z_clean
        if mode != "fm_pred_gt_mix":
            raise ValueError(f"unknown decoder_recon_latent_mode: {mode}")
        if z_pred.shape != z_clean.shape:
            raise ValueError("z_pred must have the same shape as z_clean for fm_pred_gt_mix")
        if latent_target_mask.shape != z_clean.shape[:2]:
            raise ValueError("latent_target_mask must have shape [B, T] for fm_pred_gt_mix")
        gt = z_clean if z_recon_gt is None else z_recon_gt
        fm_input = z_clean if z_fm_input is None else z_fm_input
        if gt.shape != z_clean.shape:
            raise ValueError("z_recon_gt must have the same shape as z_clean for fm_pred_gt_mix")
        if fm_input.shape != z_clean.shape:
            raise ValueError("z_fm_input must have the same shape as z_clean for fm_pred_gt_mix")
        target_mask = latent_target_mask.to(device=z_clean.device, dtype=torch.bool)
        if valid_latent_mask is not None:
            target_mask = target_mask & valid_latent_mask.to(device=z_clean.device, dtype=torch.bool)
        mix = float(self.config.decoder_pred_gt_mix)
        z_target = (1.0 - mix) * gt + mix * z_pred
        return torch.where(target_mask.unsqueeze(-1), z_target, fm_input)

    def _apply_decoder_latent_noise(
        self,
        z: Tensor,
        valid_latent_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        zero = z.new_tensor(0.0)
        mode = str(self.config.decoder_latent_noise_mode)
        std = float(self.config.decoder_latent_noise_std)
        t_start = float(self.config.decoder_latent_noise_t_start)
        prob = float(self.config.decoder_latent_noise_prob)
        if not self.training or mode == "none" or prob <= 0.0:
            return z, zero, zero

        if mode == "additive":
            if std <= 0.0:
                return z, zero, zero
            noise = torch.randn_like(z) * std
        elif mode in {"vp", "ul_fixed"}:
            if std <= 0.0:
                return z, zero, zero
            if std >= 1.0:
                raise ValueError("decoder_latent_noise_std must be < 1.0 for vp/ul_fixed mode")
            alpha = float((1.0 - std * std) ** 0.5)
            eps = torch.randn_like(z)
            noise = (alpha - 1.0) * z + std * eps
        elif mode in {"flow_t", "unite"}:
            if t_start >= 1.0:
                return z, zero, zero
            t = torch.empty(z.shape[0], device=z.device, dtype=torch.float32).uniform_(t_start, 1.0)
            t = t.to(dtype=z.dtype).view(z.shape[0], 1, 1)
            eps = torch.randn_like(z)
            noise = (1.0 - t) * (eps - z)
        else:
            raise ValueError(f"unknown decoder_latent_noise_mode: {mode}")

        if valid_latent_mask is not None:
            valid = valid_latent_mask.to(device=z.device, dtype=torch.bool)
            noise = noise * valid.unsqueeze(-1).to(dtype=noise.dtype)
        apply_mask = (torch.rand(z.shape[0], device=z.device) < prob).view(z.shape[0], 1, 1)
        noise = torch.where(apply_mask, noise, torch.zeros_like(noise))
        applied_fraction = apply_mask.detach().float().mean()
        noise_rms = self._latent_noise_rms(noise, valid_latent_mask)
        return z + noise, noise_rms, applied_fraction

    def _latent_noise_rms(self, noise: Tensor, valid_latent_mask: Tensor | None) -> Tensor:
        if noise.numel() == 0:
            return noise.new_tensor(0.0)
        if valid_latent_mask is None:
            return noise.float().pow(2).mean().sqrt()
        valid = valid_latent_mask.to(device=noise.device, dtype=torch.bool)
        if valid.numel() == 0 or not bool(valid.any().item()):
            return noise.new_tensor(0.0)
        selected = noise.float()[valid.unsqueeze(-1).expand_as(noise)]
        if selected.numel() == 0:
            return noise.new_tensor(0.0)
        return selected.pow(2).mean().sqrt()

    def _latent_regularization_loss(
        self,
        z: Tensor,
        valid_latent_mask: Tensor | None,
        target_latent_mask: Tensor | None = None,
    ) -> Tensor:
        mode = str(self.config.latent_reg_mode)
        if mode == "none" or float(self.config.lambda_latent_reg) == 0.0:
            return z.new_tensor(0.0)
        if mode in {"target_seq_hinge_sigreg", "target_seq_band_hinge_sigreg"}:
            return self._target_sequence_hinge_sigreg_loss(
                z,
                target_latent_mask,
                valid_latent_mask,
                use_std_ceiling=(mode == "target_seq_band_hinge_sigreg"),
            )
        x = self._select_valid_latents(z, valid_latent_mask)
        if x.shape[0] < 2:
            return z.new_tensor(0.0)
        if mode == "weak_sigreg":
            return self._weak_sigreg_loss(x)
        if mode == "sigreg":
            return self._sigreg_loss(x)
        if mode == "hinge_sigreg":
            return self._hinge_sigreg_loss(x)
        if mode == "visreg":
            return self._visreg_loss(x)
        raise ValueError(f"unknown latent_reg_mode: {mode}")

    def _select_valid_latents(self, z: Tensor, valid_latent_mask: Tensor | None) -> Tensor:
        if z.ndim != 3:
            raise ValueError("latent regularization expects z with shape [B, T, C]")
        if valid_latent_mask is None:
            return z.reshape(-1, z.shape[-1]).float()
        mask = valid_latent_mask.to(device=z.device, dtype=torch.bool)
        if mask.shape != z.shape[:2]:
            raise ValueError("valid_latent_mask must have shape [B, T]")
        if not bool(mask.any().item()):
            return z.new_zeros((0, z.shape[-1]), dtype=torch.float32)
        return z.float()[mask]

    def _combine_latent_masks(
        self,
        z: Tensor,
        primary_mask: Tensor | None,
        valid_latent_mask: Tensor | None,
    ) -> Tensor | None:
        if z.ndim != 3:
            raise ValueError("latent regularization expects z with shape [B, T, C]")
        mask = None
        if primary_mask is not None:
            mask = primary_mask.to(device=z.device, dtype=torch.bool)
            if mask.shape != z.shape[:2]:
                raise ValueError("primary latent mask must have shape [B, T]")
        if valid_latent_mask is not None:
            valid = valid_latent_mask.to(device=z.device, dtype=torch.bool)
            if valid.shape != z.shape[:2]:
                raise ValueError("valid_latent_mask must have shape [B, T]")
            mask = valid if mask is None else (mask & valid)
        return mask

    def _regularizer_sketch(self, x: Tensor) -> Tensor:
        sketch_dim = min(int(self.config.latent_reg_sketch_dim), x.shape[-1])
        if sketch_dim >= x.shape[-1]:
            return x
        return x[:, :sketch_dim]

    def _normal_quantile_target(self, sample_count: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        q = torch.linspace(1, sample_count, sample_count, device=device, dtype=torch.float32)
        q = q / float(sample_count + 1)
        return torch.erfinv(2.0 * q - 1.0).mul_(math.sqrt(2.0)).to(dtype=dtype)

    def _weak_sigreg_loss(self, x: Tensor) -> Tensor:
        x = self._regularizer_sketch(x)
        x = x - x.mean(dim=0, keepdim=True)
        denom = max(int(x.shape[0]) - 1, 1)
        cov = x.transpose(0, 1).matmul(x) / float(denom)
        eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
        return torch.linalg.matrix_norm(cov - eye, ord="fro")

    def _sigreg_loss(self, x: Tensor) -> Tensor:
        x = self._regularizer_sketch(x)
        x = x - x.mean(dim=0, keepdim=True)
        projections = int(self.config.latent_reg_projections)
        knots_count = int(self.config.latent_reg_knots)
        if projections < 1 or knots_count < 2:
            return x.new_tensor(0.0)
        dim = x.shape[-1]
        sample_count = int(x.shape[0])
        t_max = float(self.config.latent_reg_t_max)
        knots = torch.linspace(0.0, t_max, knots_count, device=x.device, dtype=x.dtype)
        dt = t_max / float(knots_count - 1)
        trapz_weights = torch.full((knots_count,), 2.0 * dt, device=x.device, dtype=x.dtype)
        trapz_weights[0] = dt
        trapz_weights[-1] = dt
        target_real = torch.exp(-0.5 * knots.square())
        weights = trapz_weights * target_real

        projection_chunk = min(projections, 512)
        total = x.new_tensor(0.0)
        processed = 0
        for start in range(0, projections, projection_chunk):
            chunk = min(projection_chunk, projections - start)
            dirs = torch.randn(dim, chunk, device=x.device, dtype=x.dtype)
            dirs = F.normalize(dirs, dim=0)
            projected = x.matmul(dirs)
            values = projected[:, :, None] * knots[None, None, :]
            emp_real = values.cos().mean(dim=0)
            emp_imag = values.sin().mean(dim=0)
            err = (emp_real - target_real[None, :]).square() + emp_imag.square()
            statistic = (err * weights[None, :]).sum(dim=1) * float(sample_count)
            total = total + statistic.sum()
            processed += chunk
        return total / float(max(processed, 1))

    def _hinge_sigreg_loss(self, x: Tensor) -> Tensor:
        x = self._regularizer_sketch(x)
        x = x - x.mean(dim=0, keepdim=True)
        projections = int(self.config.latent_reg_projections)
        knots_count = int(self.config.latent_reg_knots)
        if projections < 1 or knots_count < 2:
            return x.new_tensor(0.0)
        dim = x.shape[-1]
        sample_count = int(x.shape[0])
        t_max = float(self.config.latent_reg_t_max)
        gamma = float(self.config.latent_reg_var_floor)
        knots = torch.linspace(0.0, t_max, knots_count, device=x.device, dtype=x.dtype)
        dt = t_max / float(knots_count - 1)
        trapz_weights = torch.full((knots_count,), 2.0 * dt, device=x.device, dtype=x.dtype)
        trapz_weights[0] = dt
        trapz_weights[-1] = dt

        reference = torch.exp(-0.5 * knots.square())
        max_cf = torch.exp(-0.5 * (gamma * knots).square())
        weights = trapz_weights * reference

        projection_chunk = min(projections, 512)
        total = x.new_tensor(0.0)
        processed = 0
        for start in range(0, projections, projection_chunk):
            chunk = min(projection_chunk, projections - start)
            dirs = torch.randn(dim, chunk, device=x.device, dtype=x.dtype)
            dirs = F.normalize(dirs, dim=0)
            projected = x.matmul(dirs)
            values = projected[:, :, None] * knots[None, None, :]
            emp_real = values.cos().mean(dim=0)
            emp_imag = values.sin().mean(dim=0)
            cf_mag = torch.sqrt(emp_real.square() + emp_imag.square() + 1.0e-12)
            err = F.relu(cf_mag - max_cf[None, :]).square()
            statistic = (err * weights[None, :]).sum(dim=1) * float(sample_count)
            total = total + statistic.sum()
            processed += chunk
        return total / float(max(processed, 1))

    def _visreg_loss(self, x: Tensor) -> Tensor:
        x = self._regularizer_sketch(x)
        sample_count = int(x.shape[0])
        if sample_count < 2:
            return x.new_tensor(0.0)
        mean = x.mean(dim=0, keepdim=True)
        center_loss = mean.square().mean()

        centered = x - mean
        std = centered.norm(dim=0).div(math.sqrt(sample_count)).add(1.0e-6)
        scale_loss = (std - 1.0).square().mean()

        normalized = centered / std.detach().unsqueeze(0)
        projections = int(self.config.latent_reg_projections)
        if projections < 1:
            return scale_loss + center_loss
        dim = normalized.shape[-1]
        target = self._normal_quantile_target(sample_count, normalized.device, normalized.dtype).view(sample_count, 1)

        projection_chunk = min(projections, 512)
        shape_total = normalized.new_tensor(0.0)
        processed = 0
        for start in range(0, projections, projection_chunk):
            chunk = min(projection_chunk, projections - start)
            dirs = torch.randn(dim, chunk, device=normalized.device, dtype=normalized.dtype)
            dirs = F.normalize(dirs, dim=0)
            projected = normalized.matmul(dirs).sort(dim=0).values
            shape_total = shape_total + (projected - target).square().mean() * float(chunk)
            processed += chunk
        shape_loss = shape_total / float(max(processed, 1))
        return center_loss + scale_loss + shape_loss

    def _target_sequence_hinge_sigreg_loss(
        self,
        z: Tensor,
        target_latent_mask: Tensor | None,
        valid_latent_mask: Tensor | None,
        *,
        use_std_ceiling: bool = False,
    ) -> Tensor:
        mask = self._combine_latent_masks(z, target_latent_mask, valid_latent_mask)
        losses = []
        for row in range(z.shape[0]):
            if mask is None:
                x = z[row].float()
            else:
                x = z[row][mask[row]].float()
            if x.shape[0] < 2:
                continue
            loss = self._hinge_sigreg_loss(x)
            if use_std_ceiling:
                loss = loss + self._target_sequence_std_ceiling_loss(x)
            losses.append(loss)
        if not losses:
            return z.new_tensor(0.0)
        return torch.stack(losses).mean()

    @torch.no_grad()
    def _latent_regularization_metrics(
        self,
        z: Tensor,
        target_latent_mask: Tensor | None,
        valid_latent_mask: Tensor | None,
    ) -> dict[str, Tensor]:
        metrics = self._target_sequence_hinge_sigreg_metrics(z, target_latent_mask, valid_latent_mask)
        metrics.update(self._visreg_metrics(z, valid_latent_mask))
        return metrics

    @torch.no_grad()
    def _visreg_metrics(self, z: Tensor, valid_latent_mask: Tensor | None) -> dict[str, Tensor]:
        if str(self.config.latent_reg_mode) != "visreg":
            return {}
        x = self._select_valid_latents(z, valid_latent_mask)
        zero = z.new_tensor(0.0)
        if x.shape[0] < 2:
            return {
                "visreg_tokens": zero,
                "visreg_center_rms": zero,
                "visreg_std_min": zero,
                "visreg_std_mean": zero,
                "visreg_std_max": zero,
            }
        x = self._regularizer_sketch(x)
        mean = x.mean(dim=0, keepdim=True)
        centered = x - mean
        std = centered.norm(dim=0).div(math.sqrt(int(x.shape[0]))).add(1.0e-6)
        return {
            "visreg_tokens": z.new_tensor(float(x.shape[0])),
            "visreg_center_rms": mean.square().mean().sqrt().detach(),
            "visreg_std_min": std.min().detach(),
            "visreg_std_mean": std.mean().detach(),
            "visreg_std_max": std.max().detach(),
        }

    def _target_sequence_std_ceiling_loss(self, x: Tensor) -> Tensor:
        x = self._regularizer_sketch(x)
        x = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt(x.var(dim=0, unbiased=False) + 1.0e-12)
        ceiling = float(self.config.latent_reg_seq_std_ceiling)
        weight = float(self.config.latent_reg_seq_std_ceiling_weight)
        return F.relu(std - ceiling).square().sum() * weight

    @torch.no_grad()
    def _target_sequence_hinge_sigreg_metrics(
        self,
        z: Tensor,
        target_latent_mask: Tensor | None,
        valid_latent_mask: Tensor | None,
    ) -> dict[str, Tensor]:
        if str(self.config.latent_reg_mode) not in {"target_seq_hinge_sigreg", "target_seq_band_hinge_sigreg"}:
            return {}
        zero = z.new_tensor(0.0)
        mask = self._combine_latent_masks(z, target_latent_mask, valid_latent_mask)
        sequence_stds = []
        ceiling_losses = []
        token_counts = []
        for row in range(z.shape[0]):
            if mask is None:
                x = z[row].float()
            else:
                x = z[row][mask[row]].float()
            if x.shape[0] < 2:
                continue
            x = self._regularizer_sketch(x)
            x = x - x.mean(dim=0, keepdim=True)
            std = torch.sqrt(x.var(dim=0, unbiased=False) + 1.0e-12)
            sequence_stds.append(std)
            ceiling = float(self.config.latent_reg_seq_std_ceiling)
            weight = float(self.config.latent_reg_seq_std_ceiling_weight)
            ceiling_losses.append(F.relu(std - ceiling).square().sum() * weight)
            token_counts.append(x.new_tensor(float(x.shape[0])))
        if not sequence_stds:
            return {
                "target_seq_hinge_valid_sequences": zero,
                "target_seq_hinge_tokens_min": zero,
                "target_seq_hinge_tokens_mean": zero,
                "target_seq_hinge_tokens_max": zero,
                "target_seq_hinge_std_min": zero,
                "target_seq_hinge_std_mean": zero,
                "target_seq_hinge_std_max": zero,
                "target_seq_hinge_dead_channel_fraction": zero,
                "target_seq_hinge_gamma": zero,
                "target_seq_hinge_std_ceiling": zero,
                "target_seq_hinge_std_ceiling_fraction": zero,
                "target_seq_hinge_std_ceiling_loss": zero,
            }
        counts = torch.stack(token_counts)
        std = torch.cat(sequence_stds)
        gamma = float(self.config.latent_reg_var_floor)
        ceiling = float(self.config.latent_reg_seq_std_ceiling)
        ceiling_loss = torch.stack(ceiling_losses)
        return {
            "target_seq_hinge_valid_sequences": z.new_tensor(float(len(sequence_stds))),
            "target_seq_hinge_tokens_min": counts.min().detach(),
            "target_seq_hinge_tokens_mean": counts.mean().detach(),
            "target_seq_hinge_tokens_max": counts.max().detach(),
            "target_seq_hinge_std_min": std.min().detach(),
            "target_seq_hinge_std_mean": std.mean().detach(),
            "target_seq_hinge_std_max": std.max().detach(),
            "target_seq_hinge_dead_channel_fraction": (std < gamma).float().mean().detach(),
            "target_seq_hinge_gamma": z.new_tensor(gamma),
            "target_seq_hinge_std_ceiling": z.new_tensor(ceiling),
            "target_seq_hinge_std_ceiling_fraction": (std > ceiling).float().mean().detach(),
            "target_seq_hinge_std_ceiling_loss": ceiling_loss.mean().detach(),
        }

    def _scale_waveform(self, wav: Tensor) -> Tensor:
        return wav * float(self.config.waveform_scale)

    def _unscale_waveform(self, wav: Tensor) -> Tensor:
        return wav / float(self.config.waveform_scale)

    def _normalize_flow_x_pred(self, z_pred: Tensor) -> Tensor:
        return self.latent_norm(z_pred)

    def _flow_xpred_denom(self, t: Tensor) -> Tensor:
        return (1.0 - t).clamp_min(float(self.config.flow_xpred_denom_min))

    def _flow_inference_time_grid(
        self,
        num_steps: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        del dtype
        grid = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=torch.float32)
        mapping = str(
            getattr(self.config, "flow_inference_timestep_mapping", "power")
        ).lower().replace("-", "_")
        if mapping in {"none", "linear"}:
            mapping = "uniform"

        if mapping == "legacy_shifted_linear":
            shift = float(self.config.flow_timestep_shift)
            if shift > 0.0 and shift != 1.0:
                denom = 1.0 + (shift - 1.0) * grid
                grid = shift * grid / denom.clamp_min(1.0e-6)
            return grid

        if mapping == "uniform":
            pass
        elif mapping == "power":
            power = float(getattr(self.config, "flow_inference_timestep_power", 2.0))
            grid = grid.pow(power)
        elif mapping == "sway_sampling":
            coef = getattr(self.config, "flow_inference_sway_sampling_coef", None)
            if coef is not None:
                grid = grid + float(coef) * (torch.cos(torch.pi / 2 * grid) - 1.0 + grid)
        else:
            raise ValueError(f"unknown flow_inference_timestep_mapping: {mapping}")

        shift = float(getattr(self.config, "flow_inference_timestep_shift", 3.0))
        if shift != 1.0:
            grid = grid / (grid + shift * (1.0 - grid)).clamp_min(1.0e-6)
        grid = grid.clamp(0.0, 1.0)
        grid[0] = 0.0
        grid[-1] = 1.0
        return grid

    def _sample_flow_time(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        if self.config.flow_t_sampling == "uniform":
            t = torch.rand(batch_size, device=device, dtype=dtype)
        elif self.config.flow_t_sampling == "logit_normal":
            t = torch.randn(batch_size, device=device, dtype=torch.float32)
            t = t * float(self.config.flow_lognorm_sigma) + float(self.config.flow_lognorm_mu)
            t = t.sigmoid().to(dtype=dtype)
        else:
            raise ValueError(f"unknown flow_t_sampling: {self.config.flow_t_sampling}")
        shift = float(self.config.flow_timestep_shift)
        if shift > 0.0 and shift != 1.0:
            t_float = t.float()
            denom = 1.0 + (shift - 1.0) * t_float
            t = (shift * t_float / denom.clamp_min(1.0e-6)).to(dtype=dtype)
        return t

    def _encode_patch_tokens(
        self,
        patch_tokens: Tensor,
        *,
        return_raw: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        z = patch_tokens
        for downsample, blocks in zip(self.encoder_downsamples, self.encoder_down_blocks, strict=True):
            z = downsample(z)
            for block in blocks:
                z = block(z)
        z = self.encoder_latent_norm(z)
        z_raw = self.encoder_to_latent(z)
        z_norm = self.latent_norm(z_raw)
        if return_raw:
            return z_norm, z_raw
        return z_norm

    def _condition_text_inputs(
        self,
        batch: dict[str, Any],
        *,
        mask: Tensor | None = None,
        valid_latent_mask: Tensor | None = None,
        prompt_patch_ends: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        text_ids = batch["text_token_ids"]
        text_lengths = batch.get("text_token_lengths")
        has_token_patch_spans = batch.get("text_token_patch_spans") is not None
        missing_word_spans = any(
            key not in batch
            for key in ("word_patch_spans", "text_word_spans", "word_lengths")
        )
        if (
            not self.config.use_target_token
            or self.config.target_id is None
            or mask is None
            or valid_latent_mask is None
            or (missing_word_spans and not has_token_patch_spans)
        ):
            return text_ids, text_lengths
        return self._insert_target_text_boundary(
            text_ids,
            text_lengths,
            word_patch_spans=batch.get("word_patch_spans"),
            text_word_spans=batch.get("text_word_spans"),
            word_lengths=batch.get("word_lengths"),
            text_token_patch_spans=batch.get("text_token_patch_spans"),
            mask=mask,
            valid_latent_mask=valid_latent_mask,
            prompt_patch_ends=prompt_patch_ends,
        )

    def _insert_target_text_boundary(
        self,
        text_ids: Tensor,
        text_lengths: Tensor | None,
        *,
        word_patch_spans: Tensor | None,
        text_word_spans: Tensor | None,
        word_lengths: Tensor | None,
        text_token_patch_spans: Tensor | None = None,
        mask: Tensor,
        valid_latent_mask: Tensor,
        prompt_patch_ends: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if text_ids.ndim != 2:
            raise ValueError("text_ids must have shape [B, L]")
        batch, max_text_len = text_ids.shape
        device = text_ids.device
        if text_lengths is None:
            lengths = (text_ids != int(self.config.pad_id)).long().sum(dim=1)
        else:
            lengths = text_lengths.to(device=device, dtype=torch.long).clamp(min=0, max=max_text_len)

        if prompt_patch_ends is None:
            visible_latents = (
                (~mask.to(device=device, dtype=torch.bool))
                & valid_latent_mask.to(device=device, dtype=torch.bool)
            ).long().sum(dim=1)
            prompt_patch_ends = visible_latents * int(self.config.downsample_factor)
        else:
            prompt_patch_ends = prompt_patch_ends.to(device=device, dtype=torch.long)
        target_id = int(self.config.target_id)
        eos_id = self.config.eos_id

        if text_token_patch_spans is not None:
            text_token_patch_spans = text_token_patch_spans.to(device=device, dtype=torch.long)
            if text_token_patch_spans.ndim != 3 or text_token_patch_spans.shape[-1] != 2:
                raise ValueError("text_token_patch_spans must have shape [B, L, 2]")
            if text_token_patch_spans.shape[0] != batch:
                raise ValueError("text_token_patch_spans batch size must match text_ids")
            if max_text_len < 1:
                prompt_token_ends = torch.zeros_like(lengths)
            else:
                span_len = min(max_text_len, int(text_token_patch_spans.shape[1]))
                token_positions = torch.arange(max_text_len, device=device)
                valid_tokens = token_positions[None, :] < lengths[:, None]
                token_prompt = torch.zeros((batch, max_text_len), device=device, dtype=torch.bool)
                if span_len > 0:
                    patch_ends = text_token_patch_spans[:, :span_len, 1]
                    token_prompt[:, :span_len] = (
                        valid_tokens[:, :span_len]
                        & (patch_ends >= 0)
                        & (patch_ends <= prompt_patch_ends[:, None])
                    )
                prompt_token_ends = torch.where(
                    token_prompt,
                    token_positions[None, :] + 1,
                    torch.zeros((batch, max_text_len), device=device, dtype=torch.long),
                ).amax(dim=1)
            normal_rows = lengths > 0
        else:
            if word_patch_spans is None or text_word_spans is None or word_lengths is None:
                raise ValueError("word span tensors are required for target text boundary insertion")
            word_patch_spans = word_patch_spans.to(device=device, dtype=torch.long)
            text_word_spans = text_word_spans.to(device=device, dtype=torch.long)
            word_lengths = word_lengths.to(device=device, dtype=torch.long)
            max_words = min(word_patch_spans.shape[1], text_word_spans.shape[1])
            word_lengths = word_lengths.clamp(min=0, max=max_words)
            normal_rows = (lengths > 0) & (word_lengths > 0)
            if max_words > 0:
                word_positions = torch.arange(max_words, device=device)
                valid_words = word_positions[None, :] < word_lengths[:, None]
                patch_starts = word_patch_spans[:, :max_words, 0]
                target_words = (
                    valid_words
                    & (patch_starts >= 0)
                    & (patch_starts >= prompt_patch_ends[:, None])
                )
                target_indices = torch.where(
                    target_words,
                    word_positions[None, :].expand_as(patch_starts),
                    torch.full_like(patch_starts, max_words),
                )
                first_target_word = target_indices.amin(dim=1)
                boundary_words = torch.where(
                    first_target_word < max_words,
                    first_target_word,
                    word_lengths,
                )

                boundary_indices = (boundary_words - 1).clamp(min=0, max=max_words - 1)
                prompt_token_ends = text_word_spans[:, :max_words, 1].gather(
                    1,
                    boundary_indices[:, None],
                ).squeeze(1)
                prompt_token_ends = torch.where(
                    boundary_words > 0,
                    prompt_token_ends,
                    torch.zeros_like(prompt_token_ends),
                )
            else:
                prompt_token_ends = torch.zeros_like(lengths)
        prompt_token_ends = torch.minimum(prompt_token_ends.clamp_min(0), lengths)
        insert_positions = torch.where(normal_rows, prompt_token_ends, lengths)

        out_len = max_text_len + 2
        positions = torch.arange(out_len, device=device)
        padded = text_ids.new_full((batch, out_len), int(self.config.pad_id))
        target_positions = positions[None, :] == insert_positions[:, None]
        if max_text_len > 0:
            before_target = positions[None, :] < insert_positions[:, None]
            after_target = (
                (positions[None, :] > insert_positions[:, None])
                & (positions[None, :] <= lengths[:, None])
            )
            source_indices = torch.where(
                before_target,
                positions[None, :].expand(batch, -1),
                (positions[None, :] - 1).expand(batch, -1),
            ).clamp(min=0, max=max_text_len - 1)
            source_tokens = text_ids.gather(1, source_indices)
            padded = torch.where(before_target | after_target, source_tokens, padded)
        padded = torch.where(
            target_positions,
            torch.full_like(padded, target_id),
            padded,
        )

        new_lengths = lengths + 1
        if eos_id is not None:
            if max_text_len > 0:
                last_indices = (lengths - 1).clamp(min=0, max=max_text_len - 1)
                original_last = text_ids.gather(1, last_indices[:, None]).squeeze(1)
            else:
                original_last = torch.full_like(lengths, target_id)
            rebuilt_last = torch.where(
                insert_positions < lengths,
                original_last,
                torch.full_like(lengths, target_id),
            )
            append_eos = normal_rows & (rebuilt_last != int(eos_id))
            eos_positions = positions[None, :] == new_lengths[:, None]
            padded = torch.where(
                eos_positions & append_eos[:, None],
                torch.full_like(padded, int(eos_id)),
                padded,
            )
            new_lengths = new_lengths + append_eos.long()
        return padded, new_lengths

    def _text_tokens(self, batch: dict[str, Any]) -> Tensor:
        return self._encode_text_ids(batch["text_token_ids"], text_lengths=batch.get("text_token_lengths"))

    def _encode_text_ids(
        self,
        text_ids: Tensor,
        *,
        text_lengths: Tensor | None = None,
        device: torch.device | None = None,
    ) -> Tensor:
        return self.text_encoder(text_ids, text_lengths=text_lengths, device=device)

    def _latent_text_condition(
        self,
        batch: dict[str, Any],
        text_tokens: Tensor,
        num_latents: int,
        *,
        text_lengths: Tensor | None = None,
    ) -> Tensor:
        sentence_cond = self._sentence_text_condition(
            text_tokens,
            num_latents,
            text_lengths=text_lengths,
        )
        if self.config.text_condition_mode == "sentence":
            return sentence_cond
        return self._word_text_condition(
            batch,
            text_tokens,
            sentence_cond,
            text_lengths=text_lengths,
        )

    def _sentence_text_condition(
        self,
        text_tokens: Tensor,
        num_latents: int,
        *,
        text_lengths: Tensor | None = None,
    ) -> Tensor:
        batch, max_text_tokens, text_dim = text_tokens.shape
        if num_latents < 1:
            return text_tokens.new_zeros((batch, 0, text_dim))
        if max_text_tokens < 1:
            return text_tokens.new_zeros((batch, num_latents, text_dim))

        text_cond = text_tokens.new_zeros((batch, num_latents, text_dim))
        if self.config.sentence_text_mode == "prefix":
            copy_len = min(num_latents, max_text_tokens)
            copied = text_tokens[:, :copy_len]
            if text_lengths is None:
                text_cond[:, :copy_len] = copied
            else:
                lengths = text_lengths.to(device=text_tokens.device, dtype=torch.long)
                valid = torch.arange(copy_len, device=text_tokens.device)[None, :] < lengths[:, None]
                copied = copied * valid.unsqueeze(-1).to(dtype=text_tokens.dtype)
                text_cond[:, :copy_len] = copied
            return text_cond

        if text_lengths is None:
            lengths = torch.full((batch,), max_text_tokens, device=text_tokens.device, dtype=torch.long)
        else:
            lengths = text_lengths.to(device=text_tokens.device, dtype=torch.long).clamp(min=0, max=max_text_tokens)
        for row in range(batch):
            valid_len = int(lengths[row].item())
            if valid_len < 1:
                continue
            valid_text = text_tokens[row, :valid_len]
            if valid_len == 1:
                text_cond[row] = valid_text.expand(num_latents, text_dim)
                continue
            text_cond[row] = F.interpolate(
                valid_text.transpose(0, 1).unsqueeze(0),
                size=num_latents,
                mode="linear",
                align_corners=True,
            ).squeeze(0).transpose(0, 1)
        return text_cond

    def _word_text_condition(
        self,
        batch: dict[str, Any],
        text_tokens: Tensor,
        sentence_cond: Tensor,
        *,
        text_lengths: Tensor | None = None,
    ) -> Tensor:
        required = ("word_patch_spans", "text_word_spans", "word_lengths")
        if any(key not in batch for key in required):
            return sentence_cond

        device = text_tokens.device
        word_patch_spans = batch["word_patch_spans"].to(device=device, dtype=torch.long)
        text_word_spans = batch["text_word_spans"].to(device=device, dtype=torch.long)
        word_lengths = batch["word_lengths"].to(device=device, dtype=torch.long)
        if text_lengths is None:
            token_lengths = torch.full(
                (text_tokens.shape[0],),
                text_tokens.shape[1],
                device=device,
                dtype=torch.long,
            )
        else:
            token_lengths = text_lengths.to(device=device, dtype=torch.long)

        out = sentence_cond.clone()
        filled = torch.zeros(sentence_cond.shape[:2], device=device, dtype=torch.bool)
        factor = self.config.downsample_factor
        max_latents = sentence_cond.shape[1]
        max_text = text_tokens.shape[1]

        for row in range(text_tokens.shape[0]):
            words = min(int(word_lengths[row].item()), word_patch_spans.shape[1])
            text_limit = min(int(token_lengths[row].item()), max_text)
            for word_idx in range(words):
                patch_start = int(word_patch_spans[row, word_idx, 0].item())
                patch_end = int(word_patch_spans[row, word_idx, 1].item())
                text_start = int(text_word_spans[row, word_idx, 0].item())
                text_end = int(text_word_spans[row, word_idx, 1].item())
                if patch_start < 0 or patch_end <= patch_start or text_start < 0:
                    continue

                latent_start = max(0, min(max_latents, patch_start // factor))
                latent_end = max(0, min(max_latents, (patch_end + factor - 1) // factor))
                text_start = max(0, min(text_limit, text_start))
                text_end = max(0, min(text_limit, text_end))
                if latent_end <= latent_start or text_end <= text_start:
                    continue

                word_text = text_tokens[row, text_start:text_end].mean(dim=0)
                if self.config.text_condition_mode == "word_sparse":
                    center = (latent_start + latent_end - 1) // 2
                    width = int(self.config.word_sparse_width)
                    start = max(0, min(max_latents, center - width))
                    end = max(0, min(max_latents, center + width + 1))
                    if end <= start:
                        continue
                    out[row, start:end] = word_text
                    filled[row, start:end] = True
                else:
                    out[row, latent_start:latent_end] = word_text
                    filled[row, latent_start:latent_end] = True

        return torch.where(filled.unsqueeze(-1), out, sentence_cond)

    def _latent_mask(
        self,
        batch: dict[str, Any],
        valid_patch_mask: Tensor,
        valid_latent_mask: Tensor,
    ) -> Tensor:
        if self.config.mask_source == "batch_target" and "wav_target_mask" in batch:
            target_patch_mask = _patch_mask_from_sample_mask(
                batch["wav_target_mask"],
                patch_size=self.config.patch_size,
                num_patches=valid_patch_mask.shape[1],
                fallback_shape=(valid_patch_mask.shape[0], valid_patch_mask.shape[1] * self.config.patch_size),
            )
            return _downsample_bool_mask(
                target_patch_mask & valid_patch_mask,
                factor=self.config.downsample_factor,
            ) & valid_latent_mask

        lengths = valid_latent_mask.long().sum(dim=1)
        if self.config.mask_source == "full":
            return valid_latent_mask
        if self.config.mask_source == "prefix":
            if not self.config.use_word_audio_boundary:
                return _prefix_suffix_mask(
                    lengths,
                    valid_latent_mask,
                    prefix_fraction_min=self.config.mask_prefix_fraction_min,
                    prefix_fraction_max=self.config.mask_prefix_fraction_max,
                )
            return _prefix_word_boundary_mask(
                lengths,
                valid_latent_mask,
                word_patch_spans=batch.get("word_patch_spans"),
                word_lengths=batch.get("word_lengths"),
                prefix_fraction_min=self.config.mask_prefix_fraction_min,
                prefix_fraction_max=self.config.mask_prefix_fraction_max,
                downsample_factor=self.config.downsample_factor,
                max_preroll_patches=self.config.mask_prefix_word_boundary_max_preroll_patches,
                max_prev_tail_patches=self.config.mask_prefix_word_boundary_max_prev_tail_patches,
            )

        mask = sample_span_mask(
            valid_latent_mask.shape[0],
            lengths,
            min_ratio=self.config.mask_ratio_min,
            max_ratio=self.config.mask_ratio_max,
            device=valid_patch_mask.device,
            max_len=valid_latent_mask.shape[1],
        )
        return mask[:, : valid_latent_mask.shape[1]] & valid_latent_mask

    def _wave_loss(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        mask_f = mask.to(device=pred.device, dtype=pred.dtype)
        if self.config.recon_loss == "l1":
            diff = (pred - target).abs()
        else:
            diff = (pred - target).pow(2)
        return (diff * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def _sample_mask(value: Tensor | None, shape: torch.Size | tuple[int, int]) -> Tensor:
    if value is None:
        return torch.ones(shape, dtype=torch.bool)
    return value.to(dtype=torch.bool)


def _patch_mask_from_sample_mask(
    mask: Tensor | None,
    *,
    patch_size: int,
    num_patches: int,
    fallback_shape: torch.Size | tuple[int, int],
) -> Tensor:
    if mask is None:
        return torch.ones((fallback_shape[0], num_patches), dtype=torch.bool)
    mask = mask.to(dtype=torch.bool)
    pad = num_patches * patch_size - mask.shape[1]
    if pad > 0:
        mask = F.pad(mask, (0, pad))
    return mask[:, : num_patches * patch_size].view(mask.shape[0], num_patches, patch_size).any(dim=-1)


def _downsample_bool_mask(mask: Tensor, *, factor: int) -> Tensor:
    if factor == 1:
        return mask
    batch, tokens = mask.shape
    pad = (-tokens) % factor
    if pad:
        mask = F.pad(mask, (0, pad))
    return mask.view(batch, mask.shape[1] // factor, factor).any(dim=-1)


def _upsample_bool_mask(mask: Tensor, *, factor: int, length: int) -> Tensor:
    if factor < 1:
        raise ValueError("factor must be positive")
    out = mask.to(dtype=torch.bool).repeat_interleave(factor, dim=1)
    if out.shape[1] < length:
        out = F.pad(out, (0, length - out.shape[1]))
    return out[:, :length]


def _sample_mask_from_patch_mask(mask: Tensor, *, patch_size: int, original_len: int) -> Tensor:
    if patch_size < 1:
        raise ValueError("patch_size must be positive")
    out = mask.to(dtype=torch.bool).repeat_interleave(patch_size, dim=1)
    if out.shape[1] < original_len:
        out = F.pad(out, (0, original_len - out.shape[1]))
    return out[:, :original_len]
