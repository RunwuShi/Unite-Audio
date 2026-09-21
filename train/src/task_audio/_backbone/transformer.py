from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RoPEAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        heads: int = 4,
        dim_head: int = 64,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if dim_head % 2:
            raise ValueError("dim_head must be even for RoPE")
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.rope_base = float(rope_base)
        inner_dim = self.heads * self.dim_head
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout_p = float(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch, seq_len, _ = x.shape
        attn_mask = None
        if mask is not None:
            if mask.shape != (batch, seq_len):
                raise ValueError("attention mask must have shape [B, T]")
            attn_mask = mask.to(device=x.device, dtype=torch.bool)[:, None, None, :]
        qkv = self.to_qkv(x).view(batch, seq_len, 3, self.heads, self.dim_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = _apply_rope(q, k, base=self.rope_base)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(batch, seq_len, self.heads * self.dim_head)
        out = self.to_out(out)
        if mask is not None:
            out = out * mask.to(device=x.device, dtype=out.dtype)[:, :, None]
        return out


def _token_view(value: Tensor, x: Tensor, *, name: str) -> Tensor:
    if value.ndim == 2:
        if value.shape != (x.shape[0], x.shape[-1]):
            raise ValueError(f"{name} must have shape [B,D] or [B,T,D]")
        return value[:, None, :]
    if value.ndim == 3:
        if value.shape != x.shape:
            raise ValueError(f"{name} must have shape [B,D] or [B,T,D]")
        return value
    raise ValueError(f"{name} must have shape [B,D] or [B,T,D]")


def _modulate_norm(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1.0 + _token_view(scale, x, name="scale")) + _token_view(
        shift, x, name="shift"
    )


def _combine_modulation(
    base: tuple[Tensor, ...],
    extra: tuple[Tensor, ...] | None,
    *,
    expected: int,
) -> tuple[Tensor, ...]:
    if len(base) != expected:
        raise ValueError(f"base modulation must contain {expected} tensors")
    if extra is None:
        return base
    if len(extra) != expected:
        raise ValueError(f"extra modulation must contain {expected} tensors")
    combined: list[Tensor] = []
    for base_value, extra_value in zip(base, extra):
        if base_value.ndim == 2 and extra_value.ndim == 3:
            base_value = base_value[:, None, :]
        combined.append(base_value + extra_value)
    return tuple(combined)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        heads: int,
        dim_head: int,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        cond_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = RoPEAttention(
            dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            rope_base=rope_base,
        )
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
        )
        self.ada_norm = None
        self.uses_cond = cond_dim is not None
        if cond_dim is not None:
            self.ada_norm = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, dim * 6))
            nn.init.zeros_(self.ada_norm[-1].weight)
            nn.init.zeros_(self.ada_norm[-1].bias)

    def forward(
        self,
        x: Tensor,
        cond: Tensor | None = None,
        mask: Tensor | None = None,
        *,
        extra_modulation: tuple[Tensor, ...] | None = None,
    ) -> Tensor:
        if self.ada_norm is not None:
            if cond is None:
                raise ValueError("cond is required for adaptive TransformerBlock")
            modulation = _combine_modulation(
                self.ada_norm(cond).chunk(6, dim=-1),
                extra_modulation,
                expected=6,
            )
            shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff = modulation
            x = x + _token_view(gate_attn, x, name="gate_attn") * self.attn(
                _modulate_norm(self.attn_norm(x), shift_attn, scale_attn),
                mask=mask,
            )
            return x + _token_view(gate_ff, x, name="gate_ff") * self.ff(
                _modulate_norm(self.ff_norm(x), shift_ff, scale_ff)
            )
        if extra_modulation is not None:
            raise ValueError("extra modulation requires an adaptive TransformerBlock")
        x = x + self.attn(self.attn_norm(x), mask=mask)
        return x + self.ff(self.ff_norm(x))


class GRN(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: Tensor) -> Tensor:
        gx = torch.norm(x, p=2, dim=1, keepdim=True)
        nx = gx / gx.mean(dim=-1, keepdim=True).clamp_min(1.0e-6)
        return self.gamma * (x * nx) + self.beta + x


class TextConvNeXtV2Block(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, dilation: int = 1) -> None:
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation)
        self.norm = nn.LayerNorm(dim, eps=1.0e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        return self.net(_sinusoidal_embedding(t, self.net[0].in_features))


def _apply_rope(q: Tensor, k: Tensor, *, base: float) -> tuple[Tensor, Tensor]:
    seq_len = q.shape[-2]
    dim = q.shape[-1]
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=q.device, dtype=q.dtype) / dim))
    positions = torch.arange(seq_len, device=q.device, dtype=q.dtype)
    freqs = torch.einsum("n,d->nd", positions, inv_freq)
    cos = freqs.cos()[None, None, :, :]
    sin = freqs.sin()[None, None, :, :]
    return _rotate(q, cos, sin), _rotate(k, cos, sin)


def _rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    x_rot = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
    return x_rot.flatten(-2)


def _sinusoidal_embedding(t: Tensor, dim: int) -> Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=t.dtype)
        / max(half - 1, 1)
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb
