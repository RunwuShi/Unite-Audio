from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .network import TransformerBlock

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

class FMHead(nn.Module):
    def __init__(self, hidden_dim: int, token_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, token_dim))

    def forward(self, h: Tensor) -> Tensor:
        return self.net(h)


class TokenUpsample(nn.Module):
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
        if str(head_type).lower().replace("-", "_") not in {"patch", "wave_patch", "linear_patch"}:
            raise ValueError("this release requires decoder.head='patch'")
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
        self.out_proj = nn.Linear(hidden_dim, patch_size)

    def forward(self, z: Tensor, original_len: int | None = None) -> Tensor:
        h = self.in_proj(z)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        patches = self.out_proj(h)
        return self.patchify.unpatchify(patches, original_len=original_len)


class DeterministicWaveDecoder(nn.Module):

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
        if self.backbone != "dit":
            raise ValueError("this release requires decoder.backbone='dit'")
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

    def forward(self, z: Tensor, original_len: int | None = None) -> Tensor:
        return self.decoder(z, original_len=original_len)
