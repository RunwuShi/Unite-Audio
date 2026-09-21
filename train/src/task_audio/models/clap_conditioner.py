from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ClapTextConditionerConfig:
    """Configuration for the frozen LAION-CLAP caption encoder."""

    checkpoint: str
    embedding_dim: int = 512
    amodel: str = "HTSAT-tiny"
    enable_fusion: bool = False
    text_only: bool = True

    def __post_init__(self) -> None:
        if not self.checkpoint:
            raise ValueError("CLAP checkpoint is required when CLAP conditioning is enabled")
        if int(self.embedding_dim) < 1:
            raise ValueError("CLAP embedding_dim must be positive")
        if not self.amodel:
            raise ValueError("CLAP amodel must be non-empty")


class ClapTextConditioner(nn.Module):
    """Frozen LAION-CLAP text encoder returning one global vector per caption.

    The encoder is injectable for unit tests. The production path deliberately
    requires a local checkpoint instead of allowing ``laion_clap`` to download
    weights implicitly. When ``text_only`` is enabled, the unused audio branch
    is released after loading the complete CLAP checkpoint.
    """

    def __init__(
        self,
        config: ClapTextConditionerConfig | Mapping[str, Any],
        *,
        encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(config, ClapTextConditionerConfig):
            config = ClapTextConditionerConfig(**dict(config))
        self.config = config

        if encoder is None:
            checkpoint = Path(config.checkpoint).expanduser()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"CLAP checkpoint does not exist: {checkpoint}")
            encoder = _build_offline_laion_clap_encoder(config).eval()
            encoder.load_ckpt(str(checkpoint), verbose=False)
            if config.text_only:
                _discard_audio_branch(encoder)

        self.encoder = encoder
        self.output_dim = int(config.embedding_dim)
        self.encoder.eval()
        self.encoder.requires_grad_(False)
        self.register_buffer("_null_embedding", torch.empty(0), persistent=False)

    def train(self, mode: bool = True) -> ClapTextConditioner:
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(
        self,
        captions: Sequence[str],
        *,
        device: torch.device | str | None = None,
    ) -> Tensor:
        captions = [str(caption) for caption in captions]
        if not captions:
            target_device = torch.device(device) if device is not None else _module_device(self.encoder)
            return torch.empty((0, self.output_dim), device=target_device)
        embedding = self.encoder.get_text_embedding(captions, use_tensor=True)
        if not isinstance(embedding, Tensor):
            embedding = torch.as_tensor(embedding)
        if embedding.ndim != 2 or embedding.shape != (len(captions), self.output_dim):
            raise ValueError(
                "CLAP text encoder must return "
                f"[B,{self.output_dim}], got {tuple(embedding.shape)}"
            )
        embedding = embedding.detach()
        return embedding if device is None else embedding.to(device=device)

    @torch.no_grad()
    def null_context(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Return the real CLAP embedding of the empty caption for CFG."""

        batch_size = int(batch_size)
        if batch_size < 0:
            raise ValueError("batch_size must be non-negative")
        target_device = (
            torch.device(device) if device is not None else _module_device(self.encoder)
        )
        if self._null_embedding.numel() != self.output_dim:
            self._null_embedding = self([""], device=target_device).reshape(1, self.output_dim)
        null = self._null_embedding
        if null.device != target_device:
            null = null.to(device=target_device)
        if dtype is not None:
            null = null.to(dtype=dtype)
        return null.expand(batch_size, -1)

    def checkpoint_state_dict(self, *, keep_vars: bool = False) -> dict[str, Tensor]:
        """The frozen CLAP body and non-persistent null cache are reconstructable."""

        del keep_vars
        return {}


def _load_laion_clap_module() -> type[nn.Module]:
    try:
        from laion_clap import CLAP_Module
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "CLAP conditioning requires a working laion_clap installation; "
            "disable model.clap.enabled or install its runtime dependencies"
        ) from exc
    return CLAP_Module


def _build_offline_laion_clap_encoder(
    config: ClapTextConditionerConfig,
) -> nn.Module:
    """Construct LAION-CLAP without requiring redundant RoBERTa weights.

    LAION-CLAP hard-codes ``RobertaModel.from_pretrained('roberta-base')``
    before loading the complete CLAP checkpoint.  The CLAP checkpoint already
    contains the full text tower, so only the locally cached RoBERTa config and
    tokenizer are required here.
    """

    try:
        from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "CLAP conditioning requires local transformers RoBERTa support"
        ) from exc
    try:
        roberta_config = RobertaConfig.from_pretrained(
            "roberta-base", local_files_only=True
        )
        tokenizer = RobertaTokenizer.from_pretrained(
            "roberta-base", local_files_only=True
        )
    except OSError as exc:
        raise RuntimeError(
            "CLAP conditioning requires a locally cached roberta-base config "
            "and tokenizer; model weights are supplied by the CLAP checkpoint"
        ) from exc

    empty_text_tower = RobertaModel(roberta_config)
    clap_module = _load_laion_clap_module()
    with (
        patch.object(
            RobertaModel,
            "from_pretrained",
            return_value=empty_text_tower,
        ),
        patch.object(
            RobertaTokenizer,
            "from_pretrained",
            return_value=tokenizer,
        ),
    ):
        return clap_module(
            enable_fusion=bool(config.enable_fusion),
            device="cpu",
            amodel=str(config.amodel),
        )


def _discard_audio_branch(encoder: nn.Module) -> None:
    model = getattr(encoder, "model", None)
    if model is None:
        return
    # Text embedding uses only text_branch/text_projection. Releasing these
    # modules saves the HTSAT audio tower without changing caption features.
    for name in ("audio_branch", "audio_projection", "audio_transform"):
        if hasattr(model, name):
            setattr(model, name, None)


def _module_device(module: nn.Module) -> torch.device:
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(module.buffers(), None)
    return buffer.device if buffer is not None else torch.device("cpu")


__all__ = ["ClapTextConditioner", "ClapTextConditionerConfig"]
