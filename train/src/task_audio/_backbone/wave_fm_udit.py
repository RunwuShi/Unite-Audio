from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .transformer import TimeEmbedding, TransformerBlock


def _use_adaln(idx: int, depth: int, every: int) -> bool:
    return idx == 0 or idx == depth - 1 or idx % every == 0


def _parse_depths(value: str | tuple[int, ...] | list[int] | None) -> tuple[int, int, int, int, int]:
    if value is None:
        return (4, 4, 8, 4, 4)
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
        depths = tuple(int(part) for part in parts)
    else:
        depths = tuple(int(part) for part in value)
    if len(depths) != 5:
        raise ValueError("wave_fm_unet_depths must contain five integers")
    if min(depths) < 1:
        raise ValueError("all wave_fm_unet_depths values must be positive")
    return depths  # type: ignore[return-value]


def _downsample_mask(mask: Tensor) -> Tensor:
    pad = (-mask.shape[1]) % 2
    if pad:
        mask = F.pad(mask, (0, pad), value=False)
    return mask.view(mask.shape[0], mask.shape[1] // 2, 2).any(dim=-1)


class LinearTimeDownsample(nn.Module):
    """Time-only 1D patch merge without convolution."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim * 2)
        self.proj = nn.Linear(dim * 2, dim)

    def forward(self, x: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor | None]:
        batch, tokens, dim = x.shape
        pad = (-tokens) % 2
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
            if mask is not None:
                mask = F.pad(mask, (0, pad), value=False)
        x = x.view(batch, x.shape[1] // 2, dim * 2)
        out = self.proj(self.norm(x))
        out_mask = _downsample_mask(mask) if mask is not None else None
        if out_mask is not None:
            out = out * out_mask.unsqueeze(-1).to(dtype=out.dtype)
        return out, out_mask


class LinearTimeUpsample(nn.Module):
    """Time-only 1D token expand without convolution."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim * 2)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, target_tokens: int) -> Tensor:
        batch, tokens, dim = x.shape
        out = self.proj(self.norm(x)).view(batch, tokens * 2, dim)
        out = self.out_norm(out)
        return out[:, :target_tokens]


class LinearUDiTUNetWaveFMDecoder(nn.Module):
    """Linear time-U-Net wave FM decoder with WavTTS-style raw patch output."""

    def __init__(
        self,
        *,
        patch_size: int,
        cond_dim: int,
        hidden_dim: int,
        depths: str | tuple[int, ...] | list[int] | None,
        adaln_every: int,
        heads: int,
        dim_head: int,
        ffn_mult: int,
        dropout: float,
        rope_base: float,
        input_fusion: str = "add",
        speaker_dim: int = 192,
    ) -> None:
        super().__init__()
        if patch_size < 1:
            raise ValueError("patch_size must be positive")
        if cond_dim < 1:
            raise ValueError("cond_dim must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if adaln_every < 1:
            raise ValueError("adaln_every must be positive")
        if speaker_dim < 1:
            raise ValueError("speaker_dim must be positive")
        self.patch_size = int(patch_size)
        self.cond_dim = int(cond_dim)
        self.hidden_dim = int(hidden_dim)
        self.speaker_dim = int(speaker_dim)
        self.depths = _parse_depths(depths)
        self.input_fusion = str(input_fusion).lower().replace("-", "_")
        if self.input_fusion in {"sum", "additive"}:
            self.input_fusion = "add"
        if self.input_fusion in {"concat", "project_concat", "projected_fuse"}:
            self.input_fusion = "projected_concat"
        if self.input_fusion not in {"add", "projected_concat"}:
            raise ValueError("wave_fm_input_fusion must be 'add' or 'projected_concat'")

        self.wave_proj = nn.Linear(self.patch_size, self.hidden_dim)
        self.cond_proj = nn.Linear(self.cond_dim, self.hidden_dim)
        self.input_fuse = (
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
            if self.input_fusion == "projected_concat"
            else None
        )
        self.input_norm = nn.LayerNorm(self.hidden_dim)
        self.time_embed = TimeEmbedding(self.hidden_dim)
        self.speaker_proj = nn.Linear(self.speaker_dim, self.hidden_dim)

        self.encoder_level_0 = self._make_blocks(self.depths[0], adaln_every, heads, dim_head, ffn_mult, dropout, rope_base)
        self.down0_1 = LinearTimeDownsample(self.hidden_dim)
        self.encoder_level_1 = self._make_blocks(self.depths[1], adaln_every, heads, dim_head, ffn_mult, dropout, rope_base)
        self.down1_2 = LinearTimeDownsample(self.hidden_dim)
        self.latent = self._make_blocks(self.depths[2], adaln_every, heads, dim_head, ffn_mult, dropout, rope_base)
        self.up2_1 = LinearTimeUpsample(self.hidden_dim)
        self.reduce_level_1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.decoder_level_1 = self._make_blocks(self.depths[3], adaln_every, heads, dim_head, ffn_mult, dropout, rope_base)
        self.up1_0 = LinearTimeUpsample(self.hidden_dim)
        self.reduce_level_0 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.decoder_level_0 = self._make_blocks(self.depths[4], adaln_every, heads, dim_head, ffn_mult, dropout, rope_base)
        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.patch_size)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _make_blocks(
        self,
        depth: int,
        adaln_every: int,
        heads: int,
        dim_head: int,
        ffn_mult: int,
        dropout: float,
        rope_base: float,
    ) -> nn.ModuleList:
        blocks = nn.ModuleList(
            [
                TransformerBlock(
                    self.hidden_dim,
                    heads=heads,
                    dim_head=dim_head,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                    rope_base=rope_base,
                    cond_dim=self.hidden_dim if _use_adaln(idx, depth, adaln_every) else None,
                )
                for idx in range(depth)
            ]
        )
        for block in blocks:
            self._init_active_adaln_gates(block)
        return blocks

    @staticmethod
    def _init_active_adaln_gates(block: TransformerBlock) -> None:
        if not block.uses_cond or block.ada_norm is None:
            return
        linear = block.ada_norm[-1]
        hidden_dim = linear.bias.numel() // 6
        with torch.no_grad():
            linear.bias[2 * hidden_dim : 3 * hidden_dim].fill_(1.0)
            linear.bias[5 * hidden_dim : 6 * hidden_dim].fill_(1.0)

    def _input_embed(self, wave_x: Tensor, cond: Tensor) -> Tensor:
        wave_h = self.wave_proj(wave_x)
        cond_h = self.cond_proj(cond)
        if self.input_fusion == "projected_concat":
            return self.input_norm(self.input_fuse(torch.cat((wave_h, cond_h), dim=-1)))
        return self.input_norm(wave_h + cond_h)

    def _run_blocks(self, x: Tensor, blocks: nn.ModuleList, ada_cond: Tensor, mask: Tensor | None) -> Tensor:
        for block in blocks:
            x = block(x, ada_cond if block.uses_cond else None, mask=mask)
            if mask is not None:
                x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        return x

    def forward(
        self,
        wave_x: Tensor,
        cond: Tensor,
        t: Tensor,
        *,
        valid_wave_mask: Tensor | None = None,
        speaker_emb: Tensor | None = None,
    ) -> Tensor:
        if wave_x.ndim != 3 or wave_x.shape[-1] != self.patch_size:
            raise ValueError("wave_x must have shape [B, T_patch, patch_size]")
        if cond.shape[:2] != wave_x.shape[:2] or cond.shape[-1] != self.cond_dim:
            raise ValueError("cond must have shape [B, T_patch, cond_dim]")
        if t.shape != (wave_x.shape[0],):
            raise ValueError("t must have shape [B]")
        if speaker_emb is not None and speaker_emb.shape != (wave_x.shape[0], self.speaker_dim):
            raise ValueError("speaker_emb must have shape [B, speaker_dim]")

        if valid_wave_mask is not None:
            if valid_wave_mask.shape != wave_x.shape[:2]:
                raise ValueError("valid_wave_mask must have shape [B, T_patch]")
            mask0 = valid_wave_mask.to(device=wave_x.device, dtype=torch.bool)
        else:
            mask0 = None

        h0 = self._input_embed(wave_x, cond)
        if mask0 is not None:
            h0 = h0 * mask0.unsqueeze(-1).to(dtype=h0.dtype)

        speaker_cond = (
            self.speaker_proj(speaker_emb.to(device=wave_x.device, dtype=wave_x.dtype))
            if speaker_emb is not None
            else h0.new_zeros((wave_x.shape[0], self.hidden_dim))
        )
        ada_cond = self.time_embed(t) + speaker_cond

        enc0 = self._run_blocks(h0, self.encoder_level_0, ada_cond, mask0)
        h1, mask1 = self.down0_1(enc0, mask0)
        enc1 = self._run_blocks(h1, self.encoder_level_1, ada_cond, mask1)
        h2, mask2 = self.down1_2(enc1, mask1)
        h2 = self._run_blocks(h2, self.latent, ada_cond, mask2)

        up1 = self.up2_1(h2, enc1.shape[1])
        h1 = self.reduce_level_1(torch.cat((up1, enc1), dim=-1))
        if mask1 is not None:
            h1 = h1 * mask1.unsqueeze(-1).to(dtype=h1.dtype)
        h1 = self._run_blocks(h1, self.decoder_level_1, ada_cond, mask1)

        up0 = self.up1_0(h1, enc0.shape[1])
        h0 = self.reduce_level_0(torch.cat((up0, enc0), dim=-1))
        if mask0 is not None:
            h0 = h0 * mask0.unsqueeze(-1).to(dtype=h0.dtype)
        h0 = self._run_blocks(h0, self.decoder_level_0, ada_cond, mask0)

        out = self.out_proj(self.out_norm(h0))
        if mask0 is not None:
            out = out * mask0.unsqueeze(-1).to(dtype=out.dtype)
        return out
