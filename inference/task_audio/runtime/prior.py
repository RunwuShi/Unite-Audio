from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .network import TimeEmbedding, TransformerBlock


TextInjection = Literal["cross_attn", "joint", "mmdit"]


def _normalize_injection(value: str) -> TextInjection:
    value = str(value).lower().replace("-", "_")
    if value in {"cross", "cross_attention"}:
        value = "cross_attn"
    if value in {"joint_attn", "joint_attention", "joint_self_attn", "shared_sequence"}:
        value = "joint"
    if value in {"mm_dit", "multimodal_dit", "multi_modal_dit", "flux"}:
        value = "mmdit"
    if value not in {"cross_attn", "joint", "mmdit"}:
        raise ValueError("text injection must be 'cross_attn', 'joint', or 'mmdit'")
    return value  # type: ignore[return-value]


class GlobalConditionProjection(nn.Module):
    """MeanAudio-style linear projection followed by a gated FFN."""

    def __init__(self, input_dim: int, hidden_dim: int, *, ffn_mult: int = 4) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or ffn_mult < 1:
            raise ValueError("global condition dimensions must be positive")
        inner_dim = int(2 * hidden_dim * ffn_mult / 3)
        inner_dim = 256 * ((inner_dim + 255) // 256)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.w1 = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.w2 = nn.Linear(inner_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, inner_dim, bias=False)

    def forward(self, condition: Tensor) -> Tensor:
        hidden = self.input_proj(condition)
        return self.w2(F.silu(self.w1(hidden)) * self.w3(hidden))


class CrossAttention(nn.Module):
    """Audio queries attending to caption tokens.

    This deliberately uses independent query/key positions instead of RoPE: the
    caption and audio axes have no frame-level alignment in TTA.
    """

    def __init__(
        self,
        dim: int,
        *,
        heads: int,
        dim_head: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if heads < 1 or dim_head < 1:
            raise ValueError("cross-attention heads and dim_head must be positive")
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.dropout_p = float(dropout)
        inner = self.heads * self.dim_head
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(dim, inner * 2, bias=False)
        self.to_out = nn.Linear(inner, dim)

    def forward(
        self,
        audio: Tensor,
        text: Tensor,
        *,
        text_mask: Tensor,
        audio_mask: Tensor | None = None,
    ) -> Tensor:
        if audio.ndim != 3 or text.ndim != 3:
            raise ValueError("cross-attention inputs must have shape [B, T, C]")
        if audio.shape[0] != text.shape[0]:
            raise ValueError("audio and text batch sizes must match")
        if text_mask.shape != text.shape[:2]:
            raise ValueError("text_mask must have shape [B, T_text]")
        safe_text_mask = text_mask.to(device=text.device, dtype=torch.bool)
        if not bool(safe_text_mask.any(dim=1).all().item()):
            raise ValueError("every caption row must contain at least one valid token")

        batch, audio_len, _ = audio.shape
        text_len = text.shape[1]
        q = self.to_q(audio).view(batch, audio_len, self.heads, self.dim_head).transpose(1, 2)
        kv = self.to_kv(text).view(batch, text_len, 2, self.heads, self.dim_head)
        k, v = kv.unbind(dim=2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_mask = safe_text_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(batch, audio_len, self.heads * self.dim_head)
        out = self.to_out(out)
        if audio_mask is not None:
            if audio_mask.shape != audio.shape[:2]:
                raise ValueError("audio_mask must have shape [B, T_audio]")
            out = out * audio_mask.to(device=out.device, dtype=out.dtype).unsqueeze(-1)
        return out


def _stream_token_view(value: Tensor, x: Tensor, *, name: str) -> Tensor:
    if value.ndim == 2:
        if value.shape != (x.shape[0], x.shape[-1]):
            raise ValueError(f"{name} must have shape [B,D] or [B,T,D]")
        return value[:, None, :]
    if value.ndim == 3:
        if value.shape != x.shape:
            raise ValueError(f"{name} must have shape [B,D] or [B,T,D]")
        return value
    raise ValueError(f"{name} must have shape [B,D] or [B,T,D]")


def _modulate_stream(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1.0 + _stream_token_view(scale, x, name="scale")) + (
        _stream_token_view(shift, x, name="shift")
    )


def _merge_stream_modulation(
    base: tuple[Tensor, ...],
    extra: tuple[Tensor, ...] | None,
) -> tuple[Tensor, ...]:
    if extra is None:
        return base
    if len(extra) != len(base):
        raise ValueError("extra stream modulation has the wrong number of tensors")
    combined: list[Tensor] = []
    for base_value, extra_value in zip(base, extra):
        if base_value.ndim == 2 and extra_value.ndim == 3:
            base_value = base_value[:, None, :]
        combined.append(base_value + extra_value)
    return tuple(combined)


def _apply_stream_rope(x: Tensor, *, base: float) -> Tensor:
    """Apply independent 1-D RoPE positions to one modality stream."""

    seq_len = x.shape[-2]
    dim = x.shape[-1]
    if dim % 2:
        raise ValueError("MMDiT dim_head must be even for RoPE")
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, device=x.device, dtype=x.dtype)
            / dim
        )
    )
    positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
    freqs = torch.einsum("n,d->nd", positions, inv_freq)
    cos = freqs.cos()[None, None, :, :]
    sin = freqs.sin()[None, None, :, :]
    even = x[..., 0::2]
    odd = x[..., 1::2]
    return torch.stack(
        (even * cos - odd * sin, even * sin + odd * cos),
        dim=-1,
    ).flatten(-2)


class MMDiTStreamBlock(nn.Module):
    """One modality-specific stream participating in MMDiT joint attention."""

    def __init__(
        self,
        dim: int,
        *,
        heads: int,
        dim_head: int,
        ffn_mult: int,
        dropout: float,
        rope_base: float,
        pre_only: bool = False,
    ) -> None:
        super().__init__()
        if heads < 1 or dim_head < 1:
            raise ValueError("MMDiT heads and dim_head must be positive")
        if dim_head % 2:
            raise ValueError("MMDiT dim_head must be even for RoPE")
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.dropout_p = float(dropout)
        self.rope_base = float(rope_base)
        self.pre_only = bool(pre_only)
        inner_dim = self.heads * self.dim_head

        self.attn_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.to_qkv = nn.Linear(dim, inner_dim * 3)
        self.q_norm = nn.RMSNorm(self.dim_head)
        self.k_norm = nn.RMSNorm(self.dim_head)
        modulation_dim = 2 if self.pre_only else 6
        self.ada_norm = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * modulation_dim))
        nn.init.zeros_(self.ada_norm[-1].weight)
        nn.init.zeros_(self.ada_norm[-1].bias)
        if self.pre_only:
            self.to_out = None
            self.ff_norm = None
            self.ff = None
        else:
            self.to_out = nn.Linear(inner_dim, dim)
            self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False)
            self.ff = nn.Sequential(
                nn.Linear(dim, dim * ffn_mult),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * ffn_mult, dim),
            )
            # Keep both residual paths active at initialization, matching the
            # existing TTA prior rather than starting as an exact identity.
            with torch.no_grad():
                self.ada_norm[-1].bias[2 * dim : 3 * dim].fill_(1.0)
                self.ada_norm[-1].bias[5 * dim : 6 * dim].fill_(1.0)

    def pre_attention(
        self,
        x: Tensor,
        cond: Tensor,
        *,
        extra_modulation: tuple[Tensor, ...] | None = None,
    ) -> tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, ...]]:
        modulation = _merge_stream_modulation(
            self.ada_norm(cond).chunk(2 if self.pre_only else 6, dim=-1),
            extra_modulation,
        )
        if self.pre_only:
            shift_attn, scale_attn = modulation
            post_condition: tuple[Tensor, ...] = ()
        else:
            shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff = (
                modulation
            )
            post_condition = (gate_attn, shift_ff, scale_ff, gate_ff)
        h = _modulate_stream(self.attn_norm(x), shift_attn, scale_attn)
        batch, seq_len, _ = h.shape
        qkv = self.to_qkv(h).view(
            batch,
            seq_len,
            3,
            self.heads,
            self.dim_head,
        )
        q, k, v = qkv.unbind(dim=2)
        q = self.q_norm(q.transpose(1, 2))
        k = self.k_norm(k.transpose(1, 2))
        v = v.transpose(1, 2)
        q = _apply_stream_rope(q, base=self.rope_base)
        k = _apply_stream_rope(k, base=self.rope_base)
        return (q, k, v), post_condition

    def post_attention(
        self,
        x: Tensor,
        attention_output: Tensor,
        condition: tuple[Tensor, ...],
    ) -> Tensor:
        if self.pre_only:
            return x
        assert self.to_out is not None
        assert self.ff_norm is not None
        assert self.ff is not None
        gate_attn, shift_ff, scale_ff, gate_ff = condition
        x = x + _stream_token_view(
            gate_attn, x, name="gate_attn"
        ) * self.to_out(attention_output)
        h = _modulate_stream(self.ff_norm(x), shift_ff, scale_ff)
        return x + _stream_token_view(gate_ff, x, name="gate_ff") * self.ff(h)


class MMDiTJointBlock(nn.Module):
    """Separate audio/text streams sharing one joint attention operation."""

    def __init__(
        self,
        dim: int,
        *,
        heads: int,
        dim_head: int,
        ffn_mult: int,
        dropout: float,
        rope_base: float,
        text_pre_only: bool = False,
    ) -> None:
        super().__init__()
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.dropout_p = float(dropout)
        stream_kwargs = {
            "heads": heads,
            "dim_head": dim_head,
            "ffn_mult": ffn_mult,
            "dropout": dropout,
            "rope_base": rope_base,
        }
        self.audio_stream = MMDiTStreamBlock(dim, **stream_kwargs)
        self.text_stream = MMDiTStreamBlock(
            dim,
            **stream_kwargs,
            pre_only=text_pre_only,
        )

    def forward(
        self,
        audio: Tensor,
        text: Tensor,
        cond: Tensor,
        *,
        audio_mask: Tensor,
        text_mask: Tensor,
        audio_modulation: tuple[Tensor, ...] | None = None,
        text_modulation: tuple[Tensor, ...] | None = None,
    ) -> tuple[Tensor, Tensor]:
        audio_qkv, audio_condition = self.audio_stream.pre_attention(
            audio,
            cond,
            extra_modulation=audio_modulation,
        )
        text_qkv, text_condition = self.text_stream.pre_attention(
            text,
            cond,
            extra_modulation=text_modulation,
        )
        joint_qkv = tuple(
            torch.cat((audio_part, text_part), dim=2)
            for audio_part, text_part in zip(audio_qkv, text_qkv, strict=True)
        )
        joint_mask = torch.cat((audio_mask, text_mask), dim=1)
        attention_output = F.scaled_dot_product_attention(
            *joint_qkv,
            attn_mask=joint_mask[:, None, None, :],
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        attention_output = attention_output.transpose(1, 2).reshape(
            audio.shape[0],
            joint_mask.shape[1],
            self.heads * self.dim_head,
        )
        audio_length = audio.shape[1]
        audio = self.audio_stream.post_attention(
            audio,
            attention_output[:, :audio_length],
            audio_condition,
        )
        text = self.text_stream.post_attention(
            text,
            attention_output[:, audio_length:],
            text_condition,
        )
        audio = audio * audio_mask.unsqueeze(-1).to(dtype=audio.dtype)
        text = text * text_mask.unsqueeze(-1).to(dtype=text.dtype)
        return audio, text


class TextConditionedPriorFMEncoder(nn.Module):
    """TTS masked-prior audio inputs with caption-only text conditioning.

    ``cross_attn`` keeps audio as the main sequence and adds caption
    cross-attention after every FM block. ``joint`` concatenates caption and
    audio tokens into one shared self-attention sequence. ``mmdit`` first uses
    modality-specific audio/text streams with joint attention, then discards
    the text stream and finishes with audio-only fused blocks.
    """

    def __init__(
        self,
        *,
        token_dim: int,
        text_dim: int,
        hidden_dim: int,
        depth: int,
        adaln_every: int,
        heads: int,
        dim_head: int,
        ffn_mult: int,
        dropout: float,
        rope_base: float,
        injection: str = "cross_attn",
        input_fusion: str = "add",
        mmdit_fused_depth: int | None = None,
        global_condition_dim: int | None = None,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("prior FM depth must be positive")
        if adaln_every < 1:
            raise ValueError("prior FM adaln_every must be positive")
        self.injection = _normalize_injection(injection)
        self.input_fusion = str(input_fusion).lower().replace("-", "_")
        if self.input_fusion in {"sum", "additive"}:
            self.input_fusion = "add"
        if self.input_fusion in {"concat", "project_concat", "projected_fuse"}:
            self.input_fusion = "projected_concat"
        if self.input_fusion not in {"add", "projected_concat"}:
            raise ValueError("prior input fusion must be 'add' or 'projected_concat'")

        self.x_proj = nn.Linear(token_dim, hidden_dim)
        self.cond_proj = nn.Linear(token_dim + 1, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.input_fuse = (
            nn.Linear(hidden_dim * 2, hidden_dim)
            if self.input_fusion == "projected_concat"
            else None
        )
        self.input_norm = nn.LayerNorm(hidden_dim) if self.input_fuse is not None else None
        self.modality_embedding = nn.Parameter(torch.zeros(2, hidden_dim))
        nn.init.normal_(self.modality_embedding, std=0.02)
        self.time_embed = TimeEmbedding(hidden_dim)
        self.global_condition_proj = (
            GlobalConditionProjection(int(global_condition_dim), hidden_dim)
            if global_condition_dim is not None
            else None
        )
        self.mmdit_joint_blocks = nn.ModuleList()
        self.mmdit_fused_blocks = nn.ModuleList()
        self.mmdit_fused_depth = 0
        if self.injection == "mmdit":
            if depth < 2:
                raise ValueError("MMDiT prior requires depth >= 2")
            fused_depth = (
                max(1, round(depth * 2 / 3))
                if mmdit_fused_depth is None
                else int(mmdit_fused_depth)
            )
            if not 1 <= fused_depth < depth:
                raise ValueError("mmdit_fused_depth must satisfy 1 <= fused_depth < depth")
            self.mmdit_fused_depth = fused_depth
            joint_depth = depth - fused_depth
            self.blocks = nn.ModuleList()
            self.mmdit_joint_blocks = nn.ModuleList(
                [
                    MMDiTJointBlock(
                        hidden_dim,
                        heads=heads,
                        dim_head=dim_head,
                        ffn_mult=ffn_mult,
                        dropout=dropout,
                        rope_base=rope_base,
                        text_pre_only=index == joint_depth - 1,
                    )
                    for index in range(joint_depth)
                ]
            )
            self.mmdit_fused_blocks = nn.ModuleList(
                [
                    TransformerBlock(
                        hidden_dim,
                        heads=heads,
                        dim_head=dim_head,
                        ffn_mult=ffn_mult,
                        dropout=dropout,
                        rope_base=rope_base,
                        cond_dim=hidden_dim,
                    )
                    for _ in range(fused_depth)
                ]
            )
            for block in self.mmdit_fused_blocks:
                self._init_active_adaln_gates(block)
        else:
            self.blocks = nn.ModuleList(
                [
                    TransformerBlock(
                        hidden_dim,
                        heads=heads,
                        dim_head=dim_head,
                        ffn_mult=ffn_mult,
                        dropout=dropout,
                        rope_base=rope_base,
                        cond_dim=(
                            hidden_dim
                            if self._use_adaln(index, depth, adaln_every)
                            else None
                        ),
                    )
                    for index in range(depth)
                ]
            )
            for block in self.blocks:
                self._init_active_adaln_gates(block)
        if self.injection == "cross_attn":
            self.cross_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(depth)])
            self.cross_attns = nn.ModuleList(
                [
                    CrossAttention(
                        hidden_dim,
                        heads=heads,
                        dim_head=dim_head,
                        dropout=dropout,
                    )
                    for _ in range(depth)
                ]
            )
        else:
            self.cross_norms = nn.ModuleList()
            self.cross_attns = nn.ModuleList()
        self.norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _use_adaln(idx: int, depth: int, every: int) -> bool:
        return idx == 0 or idx == depth - 1 or idx % every == 0

    @staticmethod
    def _init_active_adaln_gates(block: TransformerBlock) -> None:
        if not block.uses_cond or block.ada_norm is None:
            return
        linear = block.ada_norm[-1]
        hidden_dim = linear.bias.numel() // 6
        with torch.no_grad():
            linear.bias[2 * hidden_dim : 3 * hidden_dim].fill_(1.0)
            linear.bias[5 * hidden_dim : 6 * hidden_dim].fill_(1.0)

    def forward(
        self,
        z_x: Tensor,
        z_cond: Tensor,
        text_tokens: Tensor,
        t: Tensor,
        *,
        valid_audio_mask: Tensor,
        text_mask: Tensor,
        prompt_audio_mask: Tensor | None = None,
        condition_dropout_mask: Tensor | None = None,
        condition_dropout_mode: str = "drop_text",
        global_text_embedding: Tensor | None = None,
        time_condition_offset: Tensor | None = None,
    ) -> Tensor:
        if z_x.shape != z_cond.shape:
            raise ValueError("z_x and z_cond must have the same shape")
        if valid_audio_mask.shape != z_x.shape[:2]:
            raise ValueError("valid_audio_mask must have shape [B, T_audio]")
        if text_tokens.shape[:2] != text_mask.shape:
            raise ValueError("text_mask must match text_tokens [B, T_text]")
        if text_tokens.shape[0] != z_x.shape[0]:
            raise ValueError("text and audio batch sizes must match")
        if t.shape != (z_x.shape[0],):
            raise ValueError("t must have shape [B]")

        audio_mask = valid_audio_mask.to(device=z_x.device, dtype=torch.bool)
        if prompt_audio_mask is None:
            prompt_bool = torch.zeros_like(audio_mask)
        else:
            if prompt_audio_mask.shape != audio_mask.shape:
                raise ValueError("prompt_audio_mask must have shape [B, T_audio]")
            prompt_bool = prompt_audio_mask.to(device=z_x.device, dtype=torch.bool)

        if condition_dropout_mask is not None:
            if condition_dropout_mask.shape != (z_x.shape[0],):
                raise ValueError("condition_dropout_mask must have shape [B]")
            mode = str(condition_dropout_mode).lower().replace("-", "_")
            if mode in {"text", "text_only"}:
                mode = "drop_text"
            if mode in {"both", "all", "drop_audio_text", "text_audio", "drop_text_prompt"}:
                mode = "drop_text_audio"
            if mode not in {"drop_text", "drop_text_audio"}:
                raise ValueError("condition_dropout_mode must be 'drop_text' or 'drop_text_audio'")
            if mode == "drop_text_audio":
                drop = condition_dropout_mask.to(device=z_x.device, dtype=torch.bool)
                z_x = torch.where(
                    drop[:, None, None] & prompt_bool[:, :, None],
                    torch.zeros_like(z_x),
                    z_x,
                )
                z_cond = torch.where(drop[:, None, None], torch.zeros_like(z_cond), z_cond)
                prompt_bool = torch.where(drop[:, None], torch.zeros_like(prompt_bool), prompt_bool)

        prompt = prompt_bool.to(dtype=z_x.dtype)
        x_h = self.x_proj(z_x)
        cond_h = self.cond_proj(torch.cat((z_cond, prompt.unsqueeze(-1)), dim=-1))
        if self.input_fuse is None:
            audio_h = x_h + cond_h
        else:
            assert self.input_norm is not None
            audio_h = self.input_norm(self.input_fuse(torch.cat((x_h, cond_h), dim=-1)))
        audio_h = audio_h + self.modality_embedding[1]
        audio_h = audio_h * audio_mask.unsqueeze(-1).to(dtype=audio_h.dtype)

        text_h = self.text_proj(text_tokens.to(dtype=self.text_proj.weight.dtype))
        text_h = text_h.to(dtype=audio_h.dtype) + self.modality_embedding[0].to(dtype=audio_h.dtype)
        text_mask = text_mask.to(device=text_h.device, dtype=torch.bool)
        text_h = text_h * text_mask.unsqueeze(-1).to(dtype=text_h.dtype)
        time_cond = self.time_embed(t)
        if time_condition_offset is not None:
            if time_condition_offset.shape != time_cond.shape:
                raise ValueError(
                    "time_condition_offset must have shape "
                    f"{tuple(time_cond.shape)}"
                )
            time_cond = time_cond + time_condition_offset.to(
                device=time_cond.device,
                dtype=time_cond.dtype,
            )
        if global_text_embedding is not None:
            if self.global_condition_proj is None:
                raise ValueError(
                    "global_text_embedding was provided but global conditioning is disabled"
                )
            if global_text_embedding.ndim != 2 or global_text_embedding.shape[0] != z_x.shape[0]:
                raise ValueError("global_text_embedding must have shape [B,C]")
            global_condition = self.global_condition_proj(
                global_text_embedding.to(
                    device=time_cond.device,
                    dtype=self.global_condition_proj.input_proj.weight.dtype,
                )
            )
            time_cond = time_cond + global_condition.to(dtype=time_cond.dtype)
        elif self.global_condition_proj is not None:
            raise ValueError(
                "global conditioning is enabled but global_text_embedding was not provided"
            )

        if self.injection == "mmdit":
            for block in self.mmdit_joint_blocks:
                audio_h, text_h = block(
                    audio_h,
                    text_h,
                    time_cond,
                    audio_mask=audio_mask,
                    text_mask=text_mask,
                )
            for block in self.mmdit_fused_blocks:
                audio_h = block(audio_h, time_cond, mask=audio_mask)
                audio_h = audio_h * audio_mask.unsqueeze(-1).to(dtype=audio_h.dtype)
            return self.norm(audio_h)

        if self.injection == "joint":
            text_len = text_h.shape[1]
            h = torch.cat((text_h, audio_h), dim=1)
            joint_mask = torch.cat((text_mask, audio_mask), dim=1)
            for block in self.blocks:
                h = block(h, time_cond if block.uses_cond else None, mask=joint_mask)
                h = h * joint_mask.unsqueeze(-1).to(dtype=h.dtype)
            return self.norm(h[:, text_len:])

        h = audio_h
        for block, cross_norm, cross_attn in zip(
            self.blocks,
            self.cross_norms,
            self.cross_attns,
            strict=True,
        ):
            h = block(h, time_cond if block.uses_cond else None, mask=audio_mask)
            h = h + cross_attn(
                cross_norm(h),
                text_h,
                text_mask=text_mask,
                audio_mask=audio_mask,
            )
            h = h * audio_mask.unsqueeze(-1).to(dtype=h.dtype)
        return self.norm(h)


__all__ = [
    "CrossAttention",
    "MMDiTJointBlock",
    "MMDiTStreamBlock",
    "TextConditionedPriorFMEncoder",
    "TextInjection",
]
