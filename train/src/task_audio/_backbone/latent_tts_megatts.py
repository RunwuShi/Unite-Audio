from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .latent_tts import _use_fm_adaln
from .transformer import TimeEmbedding, TransformerBlock


def _num_2x_stages(factor: int) -> int:
    if factor < 1:
        raise ValueError("downsample_factor must be positive")
    if factor & (factor - 1):
        raise ValueError("downsample_factor must be a power of 2")
    return factor.bit_length() - 1


class MegaTTSTextConditioner(nn.Module):
    """MegaTTS-style text expansion from token spans to the latent time axis.

    Builds a 1-based patch2token map, gathers text hidden states onto the audio
    patch axis, then downsamples with strided Conv1d blocks. Dense mode fills
    each token span. Sparse mode keeps one anchor per span; the anchor can be
    sampled uniformly over the span or from a small window around its center.
    """

    def __init__(
        self,
        *,
        text_dim: int,
        hidden_dim: int,
        downsample_factor: int,
        alignment_mode: str = "dense",
        anchor_mode: str = "center",
        anchor_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        stages = _num_2x_stages(int(downsample_factor))
        alignment_mode = str(alignment_mode).lower().replace("-", "_")
        if alignment_mode in {"expand", "full"}:
            alignment_mode = "dense"
        if alignment_mode not in {"dense", "sparse"}:
            raise ValueError("alignment_mode must be 'dense' or 'sparse'")
        anchor_mode = str(anchor_mode).lower().replace("-", "_")
        if anchor_mode in {"center_window", "near_center", "center_random"}:
            anchor_mode = "center"
        if anchor_mode in {"random", "full", "full_span"}:
            anchor_mode = "uniform"
        if anchor_mode not in {"center", "uniform"}:
            raise ValueError("anchor_mode must be 'center' or 'uniform'")
        anchor_ratio = float(anchor_ratio)
        if not (0.0 < anchor_ratio <= 1.0):
            raise ValueError("anchor_ratio must satisfy 0 < ratio <= 1")
        self.alignment_mode = alignment_mode
        self.anchor_mode = anchor_mode
        self.anchor_ratio = anchor_ratio
        self.downsample_factor = int(downsample_factor)
        self.input_proj = nn.Linear(text_dim, hidden_dim)
        self.empty_text = (
            nn.Parameter(torch.zeros(1, 1, text_dim))
            if alignment_mode == "sparse"
            else None
        )
        self.down = nn.ModuleList()
        for _ in range(stages):
            self.down.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
                    nn.GELU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                )
            )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        batch: dict[str, Tensor],
        text_tokens: Tensor,
        *,
        num_latents: int,
        text_lengths: Tensor | None = None,
        valid_latent_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if text_tokens.ndim != 3:
            raise ValueError("text_tokens must have shape [B, L, C]")
        batch_size, _, _ = text_tokens.shape
        if num_latents < 1:
            empty = text_tokens.new_zeros((batch_size, 0, self.norm.normalized_shape[0]))
            return empty, self._empty_metrics(text_tokens)

        num_patches = int(num_latents) * self.downsample_factor
        patch2token, valid_patch_mask, metrics = self._patch2token(
            batch,
            text_tokens,
            num_patches=num_patches,
            text_lengths=text_lengths,
            valid_latent_mask=valid_latent_mask,
        )
        text_padded = F.pad(text_tokens, (0, 0, 1, 0))
        gather_index = patch2token.unsqueeze(-1).expand(batch_size, num_patches, text_tokens.shape[-1])
        patch_text = text_padded.gather(1, gather_index)
        if self.empty_text is not None:
            empty = (patch2token == 0) & valid_patch_mask
            empty_text = self.empty_text.to(device=patch_text.device, dtype=patch_text.dtype)
            patch_text = torch.where(empty.unsqueeze(-1), empty_text.expand_as(patch_text), patch_text)
        h = self.input_proj(patch_text).transpose(1, 2)
        for block in self.down:
            h = block(h)
        h = h.transpose(1, 2)
        if h.shape[1] < num_latents:
            h = F.pad(h, (0, 0, 0, num_latents - h.shape[1]))
        elif h.shape[1] > num_latents:
            h = h[:, :num_latents]
        if valid_latent_mask is not None:
            valid = valid_latent_mask.to(device=h.device, dtype=torch.bool)
            if valid.shape != h.shape[:2]:
                raise ValueError("valid_latent_mask must have shape [B, T_latent]")
            h = h * valid.unsqueeze(-1).to(dtype=h.dtype)
        return self.norm(h), metrics

    def _patch2token(
        self,
        batch: dict[str, Tensor],
        text_tokens: Tensor,
        *,
        num_patches: int,
        text_lengths: Tensor | None,
        valid_latent_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        device = text_tokens.device
        batch_size, max_text_len, _ = text_tokens.shape
        patch2token = torch.zeros((batch_size, num_patches), device=device, dtype=torch.long)
        valid_patch_mask = self._valid_patch_mask(
            batch_size,
            num_patches,
            device=device,
            valid_latent_mask=valid_latent_mask,
        )
        spans = batch.get("text_token_patch_spans")
        if spans is None:
            return patch2token, valid_patch_mask, self._empty_metrics(text_tokens)
        spans = spans.to(device=device, dtype=torch.long)
        if spans.ndim != 3 or spans.shape[-1] != 2 or spans.shape[0] != batch_size:
            raise ValueError("text_token_patch_spans must have shape [B, L, 2]")
        span_len = min(int(spans.shape[1]), max_text_len)
        if text_lengths is None:
            lengths = torch.full((batch_size,), max_text_len, device=device, dtype=torch.long)
        else:
            lengths = text_lengths.to(device=device, dtype=torch.long).clamp(min=0, max=max_text_len)

        token_positions = torch.arange(max_text_len, device=device)
        valid_tokens = token_positions[None, :] < lengths[:, None]
        if span_len < max_text_len:
            valid_tokens[:, span_len:] = False

        valid_token_count = valid_tokens[:, :span_len].sum()
        invalid_span_count = text_tokens.new_tensor(0.0)
        collision_count = text_tokens.new_tensor(0.0)
        write_count = text_tokens.new_tensor(0.0)

        for row in range(batch_size):
            row_tokens = int(min(lengths[row].item(), span_len))
            for token_idx in range(row_tokens):
                start = int(spans[row, token_idx, 0].item())
                end = int(spans[row, token_idx, 1].item())
                if start < 0 or end <= start:
                    invalid_span_count = invalid_span_count + 1.0
                    continue
                start = max(0, min(num_patches, start))
                end = max(0, min(num_patches, end))
                if end <= start:
                    invalid_span_count = invalid_span_count + 1.0
                    continue
                if self.alignment_mode == "sparse":
                    anchor = self._span_anchor(
                        start,
                        valid_patch_mask[row, start:end],
                        device=device,
                    )
                    if anchor is None:
                        invalid_span_count = invalid_span_count + 1.0
                        continue
                    current = patch2token[row, anchor : anchor + 1]
                    collision_count = collision_count + (current > 0).to(dtype=text_tokens.dtype).sum()
                    write_count = write_count + 1.0
                    patch2token[row, anchor] = token_idx + 1
                    continue
                current = patch2token[row, start:end]
                collision_count = collision_count + (current > 0).to(dtype=text_tokens.dtype).sum()
                write_count = write_count + float(end - start)
                patch2token[row, start:end] = token_idx + 1

        patch2token = patch2token * valid_patch_mask.to(dtype=patch2token.dtype)

        nonzero = (patch2token > 0) & valid_patch_mask
        valid_patches = valid_patch_mask.to(dtype=text_tokens.dtype).sum().clamp_min(1.0)
        total_patches = text_tokens.new_tensor(float(max(batch_size * num_patches, 1)))
        metrics = {
            "alignment_coverage": nonzero.to(dtype=text_tokens.dtype).sum() / valid_patches,
            "patch2token_nonzero_rate": (patch2token > 0).to(dtype=text_tokens.dtype).sum() / total_patches,
            "missing_token_span_rate": invalid_span_count / valid_token_count.to(dtype=text_tokens.dtype).clamp_min(1.0),
            "patch2token_collision_rate": collision_count / write_count.clamp_min(1.0),
        }
        return patch2token, valid_patch_mask, metrics

    def _valid_patch_mask(
        self,
        batch_size: int,
        num_patches: int,
        *,
        device: torch.device,
        valid_latent_mask: Tensor | None,
    ) -> Tensor:
        if valid_latent_mask is None:
            return torch.ones((batch_size, num_patches), device=device, dtype=torch.bool)
        valid_latent = valid_latent_mask.to(device=device, dtype=torch.bool)
        if valid_latent.shape[0] != batch_size:
            raise ValueError("valid_latent_mask must have shape [B, T_latent]")
        valid_patch_mask = valid_latent.repeat_interleave(self.downsample_factor, dim=1)
        if valid_patch_mask.shape[1] < num_patches:
            valid_patch_mask = F.pad(valid_patch_mask, (0, num_patches - valid_patch_mask.shape[1]))
        return valid_patch_mask[:, :num_patches]

    def _span_anchor(
        self,
        start: int,
        valid_span_mask: Tensor,
        *,
        device: torch.device,
    ) -> int | None:
        valid_offsets = valid_span_mask.nonzero(as_tuple=False).flatten()
        if valid_offsets.numel() == 0:
            return None
        if self.training:
            if self.anchor_mode == "uniform":
                candidates = valid_offsets
            else:
                count = int(valid_offsets.numel())
                window = max(1, min(count, int(math.ceil(count * self.anchor_ratio))))
                center_index = (count - 1) // 2
                left = max(0, center_index - (window - 1) // 2)
                right = min(count, left + window)
                left = max(0, right - window)
                candidates = valid_offsets[left:right]
            choice = int(torch.randint(candidates.numel(), (1,), device=device).item())
        else:
            candidates = valid_offsets
            choice = int((candidates.numel() - 1) // 2)
        return start + int(candidates[choice].item())

    @staticmethod
    def _empty_metrics(text_tokens: Tensor) -> dict[str, Tensor]:
        zero = text_tokens.new_tensor(0.0)
        return {
            "alignment_coverage": zero,
            "patch2token_nonzero_rate": zero,
            "missing_token_span_rate": zero,
            "patch2token_collision_rate": zero,
        }


class MegaTTSAddFMEncoder(nn.Module):
    """FM encoder using MegaTTS-style local additive conditioning."""

    def __init__(
        self,
        *,
        token_dim: int,
        hidden_dim: int,
        depth: int,
        adaln_every: int,
        heads: int,
        dim_head: int,
        ffn_mult: int,
        dropout: float,
        rope_base: float,
    ) -> None:
        super().__init__()
        if adaln_every < 1:
            raise ValueError("adaln_every must be positive")
        self.x_proj = nn.Linear(token_dim, hidden_dim)
        self.cond_proj = nn.Linear(token_dim + 1, hidden_dim)
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)
        self.time_embed = TimeEmbedding(hidden_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim,
                    heads=heads,
                    dim_head=dim_head,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                    rope_base=rope_base,
                    cond_dim=hidden_dim if _use_fm_adaln(idx, depth, adaln_every) else None,
                )
                for idx in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        z_x: Tensor,
        z_cond: Tensor,
        aligned_text: Tensor,
        t: Tensor,
        *,
        valid_audio_mask: Tensor | None = None,
        prompt_audio_mask: Tensor | None = None,
        condition_dropout_mask: Tensor | None = None,
    ) -> Tensor:
        if z_x.shape != z_cond.shape:
            raise ValueError("z_x and z_cond must have the same shape")
        if aligned_text.shape[:2] != z_x.shape[:2]:
            raise ValueError("aligned_text must have shape [B, T, hidden_dim]")
        if prompt_audio_mask is None:
            prompt = z_x.new_zeros(z_x.shape[:2])
        else:
            if prompt_audio_mask.shape != z_x.shape[:2]:
                raise ValueError("prompt_audio_mask must have shape [B, T]")
            prompt = prompt_audio_mask.to(device=z_x.device, dtype=z_x.dtype)
        cond = torch.cat([z_cond, prompt.unsqueeze(-1)], dim=-1)
        text = aligned_text
        if condition_dropout_mask is not None:
            if condition_dropout_mask.shape != (z_x.shape[0],):
                raise ValueError("condition_dropout_mask must have shape [B]")
            drop = condition_dropout_mask.to(device=z_x.device, dtype=torch.bool)
            cond = torch.where(drop[:, None, None], torch.zeros_like(cond), cond)
            text = torch.where(drop[:, None, None], torch.zeros_like(text), text)

        h = self.x_proj(z_x) + self.cond_proj(cond) + self.text_proj(text)
        if valid_audio_mask is not None:
            if valid_audio_mask.shape != z_x.shape[:2]:
                raise ValueError("valid_audio_mask must have shape [B, T]")
            mask = valid_audio_mask.to(device=z_x.device, dtype=torch.bool)
            h = h * mask.unsqueeze(-1).to(dtype=h.dtype)
        else:
            mask = None

        time_cond = self.time_embed(t)
        for block in self.blocks:
            h = block(h, time_cond if block.uses_cond else None, mask=mask)
            if mask is not None:
                h = h * mask.unsqueeze(-1).to(dtype=h.dtype)
        return self.norm(h)


class ChannelConcatFMEncoder(nn.Module):
    """Latent-axis FM encoder with channel concat text/audio conditioning."""

    def __init__(
        self,
        *,
        token_dim: int,
        hidden_dim: int,
        depth: int,
        adaln_every: int,
        heads: int,
        dim_head: int,
        ffn_mult: int,
        dropout: float,
        rope_base: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(token_dim * 2 + hidden_dim, hidden_dim)
        self.time_embed = TimeEmbedding(hidden_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim,
                    heads=heads,
                    dim_head=dim_head,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                    rope_base=rope_base,
                    cond_dim=hidden_dim if _use_fm_adaln(idx, depth, adaln_every) else None,
                )
                for idx in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        z_x: Tensor,
        z_cond: Tensor,
        aligned_text: Tensor,
        t: Tensor,
        *,
        valid_audio_mask: Tensor | None = None,
        condition_dropout_mask: Tensor | None = None,
    ) -> Tensor:
        cond = z_cond
        text = aligned_text
        if condition_dropout_mask is not None:
            drop = condition_dropout_mask.to(device=z_x.device, dtype=torch.bool)
            cond = torch.where(drop[:, None, None], torch.zeros_like(cond), cond)
            text = torch.where(drop[:, None, None], torch.zeros_like(text), text)
        h = self.input_proj(torch.cat([z_x, cond, text], dim=-1))
        if valid_audio_mask is not None:
            mask = valid_audio_mask.to(device=z_x.device, dtype=torch.bool)
            h = h * mask.unsqueeze(-1).to(dtype=h.dtype)
        else:
            mask = None
        time_cond = self.time_embed(t)
        for block in self.blocks:
            h = block(h, time_cond if block.uses_cond else None, mask=mask)
            if mask is not None:
                h = h * mask.unsqueeze(-1).to(dtype=h.dtype)
        return self.norm(h)
