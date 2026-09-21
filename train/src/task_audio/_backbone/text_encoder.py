from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .transformer import TextConvNeXtV2Block


class TextEncoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        text_dim: int,
        pad_id: int,
        conv_layers: int,
        conv_mult: int,
    ) -> None:
        super().__init__()
        self.pad_id = int(pad_id)
        self.embed = nn.Embedding(vocab_size, text_dim, padding_idx=pad_id)
        self.blocks = nn.ModuleList(
            [
                TextConvNeXtV2Block(text_dim, text_dim * conv_mult)
                for _ in range(conv_layers)
            ]
        )

    def forward(
        self,
        text_ids: Tensor,
        *,
        text_lengths: Tensor | None = None,
        device: torch.device | None = None,
    ) -> Tensor:
        if device is None:
            device = self.embed.weight.device
        text_ids = text_ids.to(device=device, dtype=torch.long)
        text_tokens = self.embed(text_ids)
        valid_mask = self.valid_mask(text_ids, text_lengths)
        if self.blocks:
            text_tokens = text_tokens.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
            for block in self.blocks:
                text_tokens = block(text_tokens)
                text_tokens = text_tokens.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        return text_tokens

    def valid_mask(self, text_ids: Tensor, text_lengths: Tensor | None = None) -> Tensor:
        if text_lengths is None:
            return text_ids != self.pad_id
        lengths = text_lengths.to(device=text_ids.device, dtype=torch.long)
        positions = torch.arange(text_ids.shape[1], device=text_ids.device)[None, :]
        return positions < lengths[:, None].clamp_min(0)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        key = prefix + "embed.weight"
        saved = state_dict.get(key)
        current = self.embed.weight
        if saved is not None and saved.ndim == 2 and current.ndim == 2:
            if saved.shape[1] == current.shape[1] and saved.shape[0] < current.shape[0]:
                migrated = current.detach().clone()
                migrated[: saved.shape[0]] = saved
                state_dict[key] = migrated
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
