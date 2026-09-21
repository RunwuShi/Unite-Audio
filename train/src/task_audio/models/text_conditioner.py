from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class FlanT5ConditionerConfig:
    """Configuration for the frozen FLAN-T5 caption encoder.

    ``name_or_path`` may point either at a normal Hugging Face model directory or
    at a hub cache directory containing ``refs/main`` and ``snapshots``.
    """

    name_or_path: str
    text_dim: int
    max_length: int = 128
    cfg_dropout: float = 0.1
    freeze: bool = True
    local_files_only: bool = True
    padding_mode: str = "zero"
    tokenizer_padding: str = "max_length"

    def __post_init__(self) -> None:
        if not self.name_or_path:
            # An empty path is useful only when lightweight test doubles are
            # injected into FlanT5Conditioner.
            pass
        if int(self.text_dim) < 1:
            raise ValueError("text_dim must be positive")
        if int(self.max_length) < 1:
            raise ValueError("max_length must be positive")
        if not 0.0 <= float(self.cfg_dropout) <= 1.0:
            raise ValueError("cfg_dropout must be in [0, 1]")
        if self.padding_mode not in {"none", "zero", "learned"}:
            raise ValueError("padding_mode must be one of: none, zero, learned")
        if self.tokenizer_padding not in {"longest", "max_length"}:
            raise ValueError("tokenizer_padding must be 'longest' or 'max_length'")


class FlanT5Conditioner(nn.Module):
    """Frozen FLAN-T5 encoder followed by a trainable text projection.

    The tokenizer and encoder are injectable so CPU unit tests do not need the
    ``transformers`` package or a FLAN-T5 checkpoint. Caption dropout is applied
    per example, never per token. Dropped and explicitly empty captions are
    represented by one valid, learnable null token so attention implementations
    never receive an entirely masked context.
    """

    def __init__(
        self,
        config: FlanT5ConditionerConfig | Mapping[str, Any],
        *,
        tokenizer: Any | None = None,
        encoder: nn.Module | None = None,
        encoder_hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(config, FlanT5ConditionerConfig):
            config = FlanT5ConditionerConfig(**dict(config))
        self.config = config

        if (tokenizer is None) != (encoder is None):
            raise ValueError("tokenizer and encoder must either both be supplied or both be omitted")
        if tokenizer is None:
            if not config.name_or_path:
                raise ValueError("name_or_path is required when tokenizer/encoder are not injected")
            tokenizer_cls, encoder_cls = _load_transformers()
            model_path = _resolve_hf_path(config.name_or_path)
            tokenizer = tokenizer_cls.from_pretrained(
                model_path,
                local_files_only=bool(config.local_files_only),
            )
            encoder = encoder_cls.from_pretrained(
                model_path,
                local_files_only=bool(config.local_files_only),
            )

        assert encoder is not None
        self.tokenizer = tokenizer
        self.encoder = encoder
        hidden_size = _encoder_hidden_size(encoder, encoder_hidden_size)
        self.output_dim = int(config.text_dim)
        self.projection = nn.Linear(hidden_size, int(config.text_dim))
        self.null_token = nn.Parameter(torch.empty(int(config.text_dim)))
        nn.init.normal_(self.null_token, std=0.02)

        self.padding_embedding: nn.Parameter | None = None
        if config.padding_mode == "learned":
            self.padding_embedding = nn.Parameter(torch.empty(int(config.text_dim)))
            nn.init.normal_(self.padding_embedding, std=0.02)

        if config.freeze:
            self.encoder.eval()
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)

    @property
    def proj(self) -> nn.Linear:
        """Compatibility alias used by the earlier latent-TTA prototype."""

        return self.projection

    def train(self, mode: bool = True) -> FlanT5Conditioner:
        super().train(mode)
        # Calling model.train() must not re-enable dropout in the frozen T5 body.
        if self.config.freeze:
            self.encoder.eval()
        return self

    def forward(
        self,
        captions: Sequence[str],
        *,
        device: torch.device | str | None = None,
        force_unconditional: bool | Tensor = False,
        dropout_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Encode captions and return ``(tokens, valid_token_mask)``.

        ``dropout_mask`` is mainly useful for deterministic tests. When omitted,
        CFG dropout is sampled only while this module is in training mode.
        Empty strings are always treated as unconditional captions.
        """

        captions = [str(caption) for caption in captions]
        batch_size = len(captions)
        if device is None:
            device = self.projection.weight.device
        device = torch.device(device)
        if batch_size == 0:
            empty_tokens = self.projection.weight.new_zeros((0, 0, self.config.text_dim)).to(device=device)
            empty_mask = torch.zeros((0, 0), dtype=torch.bool, device=device)
            return empty_tokens, empty_mask

        encoded = self.tokenizer(
            captions,
            padding="max_length" if self.config.tokenizer_padding == "max_length" else True,
            truncation=True,
            max_length=int(self.config.max_length),
            return_tensors="pt",
        )
        input_ids = _mapping_value(encoded, "input_ids").to(device=device)
        attention_mask_value = encoded.get("attention_mask") if isinstance(encoded, Mapping) else None
        if attention_mask_value is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        else:
            attention_mask = attention_mask_value.to(device=device)

        if self.config.freeze:
            with torch.no_grad():
                output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = _last_hidden_state(output).detach()
        else:
            output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = _last_hidden_state(output)
        text = self.projection(hidden)
        valid_mask = attention_mask.to(dtype=torch.bool)
        text = self._apply_padding(text, valid_mask)

        unconditional = torch.tensor(
            [not caption.strip() for caption in captions],
            device=device,
            dtype=torch.bool,
        )
        unconditional |= _as_batch_mask(force_unconditional, batch_size, device, "force_unconditional")
        if dropout_mask is not None:
            unconditional |= _as_batch_mask(dropout_mask, batch_size, device, "dropout_mask")
        elif self.training and float(self.config.cfg_dropout) > 0.0:
            unconditional |= torch.rand(
                (batch_size,),
                device=device,
                generator=generator,
            ) < float(self.config.cfg_dropout)

        if bool(unconditional.any().item()):
            null_text, null_mask = self.null_condition(
                batch_size,
                sequence_length=text.shape[1],
                device=device,
                dtype=text.dtype,
            )
            text = torch.where(unconditional[:, None, None], null_text, text)
            valid_mask = torch.where(unconditional[:, None], null_mask, valid_mask)
        return text, valid_mask

    def null_condition(
        self,
        batch_size: int,
        *,
        sequence_length: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return a valid unconditional context containing one null token."""

        batch_size = int(batch_size)
        sequence_length = int(sequence_length)
        if batch_size < 0 or sequence_length < 1:
            raise ValueError("batch_size must be non-negative and sequence_length must be positive")
        if device is None:
            device = self.null_token.device
        if dtype is None:
            dtype = self.null_token.dtype
        tokens = torch.zeros(
            (batch_size, sequence_length, int(self.config.text_dim)),
            device=device,
            dtype=dtype,
        )
        # Avoid an in-place assignment from the parameter so gradients to the
        # learnable null token remain explicit and reliable.
        first = self.null_token.to(device=device, dtype=dtype).view(1, 1, -1).expand(batch_size, 1, -1)
        if sequence_length == 1:
            tokens = first
        else:
            tokens = torch.cat((first, tokens[:, 1:]), dim=1)
        mask = torch.zeros((batch_size, sequence_length), device=device, dtype=torch.bool)
        mask[:, 0] = True
        return tokens, mask

    def null_context(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return the stable one-token unconditional prior context API."""

        return self.null_condition(
            batch_size,
            sequence_length=1,
            device=device,
            dtype=dtype,
        )

    def _apply_padding(self, text: Tensor, mask: Tensor) -> Tensor:
        if self.config.padding_mode == "none":
            return text
        if self.config.padding_mode == "zero":
            return text * mask.unsqueeze(-1).to(device=text.device, dtype=text.dtype)
        if self.padding_embedding is None:
            raise RuntimeError("padding_embedding was not initialized")
        padding = self.padding_embedding.to(device=text.device, dtype=text.dtype).view(1, 1, -1)
        return torch.where(mask.unsqueeze(-1), text, padding.expand_as(text))

    def checkpoint_state_dict(self, *, keep_vars: bool = False) -> dict[str, Tensor]:
        """Return conditioner state without the reconstructable frozen T5 body."""

        state = self.state_dict(keep_vars=keep_vars)
        return {key: value for key, value in state.items() if not key.startswith("encoder.")}


def filter_frozen_text_encoder_state_dict(
    state_dict: Mapping[str, Tensor],
    *,
    conditioner_names: Sequence[str] = (
        "conditioner",
        "text_conditioner",
        "clap_conditioner",
    ),
) -> dict[str, Tensor]:
    """Remove frozen T5/CLAP weights from a complete TTA model state dictionary."""

    excluded = tuple(f"{name.rstrip('.')}.encoder." for name in conditioner_names)
    return {key: value for key, value in state_dict.items() if not key.startswith(excluded)}


def checkpoint_model_state(
    model: nn.Module,
    *,
    conditioner_names: Sequence[str] = (
        "conditioner",
        "text_conditioner",
        "clap_conditioner",
    ),
    to_cpu: bool = True,
) -> dict[str, Tensor]:
    """Create a checkpoint state while omitting frozen text encoder weights."""

    state = filter_frozen_text_encoder_state_dict(
        model.state_dict(),
        conditioner_names=conditioner_names,
    )
    if not to_cpu:
        return state
    return {key: value.detach().cpu() for key, value in state.items()}


def _as_batch_mask(value: bool | Tensor, batch_size: int, device: torch.device, name: str) -> Tensor:
    if isinstance(value, bool):
        return torch.full((batch_size,), value, device=device, dtype=torch.bool)
    mask = value.to(device=device, dtype=torch.bool)
    if mask.ndim == 0:
        return mask.expand(batch_size)
    if mask.shape != (batch_size,):
        raise ValueError(f"{name} must be a scalar or have shape [B]")
    return mask


def _mapping_value(encoded: Any, name: str) -> Tensor:
    try:
        value = encoded[name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"tokenizer output is missing {name!r}") from exc
    if not isinstance(value, Tensor):
        value = torch.as_tensor(value)
    return value


def _last_hidden_state(output: Any) -> Tensor:
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None and isinstance(output, Mapping):
        hidden = output.get("last_hidden_state")
    if hidden is None and isinstance(output, (tuple, list)) and output:
        hidden = output[0]
    if not isinstance(hidden, Tensor):
        raise TypeError("text encoder output must expose a Tensor last_hidden_state")
    return hidden


def _encoder_hidden_size(encoder: nn.Module, explicit: int | None) -> int:
    if explicit is not None:
        hidden_size = int(explicit)
    else:
        config = getattr(encoder, "config", None)
        hidden_size = 0
        for name in ("d_model", "hidden_size"):
            value = getattr(config, name, None)
            if value is not None:
                hidden_size = int(value)
                break
    if hidden_size < 1:
        raise ValueError("could not infer encoder hidden size; pass encoder_hidden_size for test doubles")
    return hidden_size


def _load_transformers() -> tuple[type[Any], type[nn.Module]]:
    try:
        from transformers import AutoTokenizer, T5EncoderModel
    except ImportError as exc:
        raise RuntimeError(
            "FlanT5Conditioner requires transformers unless tokenizer and encoder test doubles are injected"
        ) from exc
    return AutoTokenizer, T5EncoderModel


def _resolve_hf_path(path: str) -> str:
    candidate = Path(path).expanduser()
    if (candidate / "snapshots").exists():
        ref = candidate / "refs" / "main"
        if ref.exists():
            revision = ref.read_text(encoding="utf-8").strip()
            snapshot = candidate / "snapshots" / revision
            if snapshot.exists():
                return str(snapshot)
    return str(candidate)


__all__ = [
    "FlanT5Conditioner",
    "FlanT5ConditionerConfig",
    "checkpoint_model_state",
    "filter_frozen_text_encoder_state_dict",
]
