from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from torch import Tensor, nn

from .checkpointing import load_tensor_state, resolve_model_state_path


_ENCODER_ROOTS = {
    "target_encoder",
    "token_norm",
    "encoder_downsamples",
    "encoder_down_blocks",
    "encoder_latent_norm",
    "encoder_to_latent",
    "latent_norm",
    "target_encoder_ema",
    "token_norm_ema",
    "encoder_downsamples_ema",
    "encoder_down_blocks_ema",
    "encoder_latent_norm_ema",
    "encoder_to_latent_ema",
    "latent_norm_ema",
}
_DETERMINISTIC_DECODER_ROOTS = {
    "decoder_up_blocks",
    "decoder_upsamples",
    "decoder_latent_norm",
    "latent_to_decoder",
    "decoder_token_norm",
    "decoder",
}
_FM_DECODER_ROOTS = {"wave_fm_decoder"}
_ALLOWED_ROOTS = _ENCODER_ROOTS | _DETERMINISTIC_DECODER_ROOTS | _FM_DECODER_ROOTS
_FORBIDDEN_SEGMENTS = {
    "text_encoder",
    "conditioner",
    "text_conditioner",
    "prior",
    "prior_fm",
    "fm_encoder",
    "fm_head",
}
_EMA_TO_ONLINE_ROOT = {
    "target_encoder_ema": "target_encoder",
    "token_norm_ema": "token_norm",
    "encoder_downsamples_ema": "encoder_downsamples",
    "encoder_down_blocks_ema": "encoder_down_blocks",
    "encoder_latent_norm_ema": "encoder_latent_norm",
    "encoder_to_latent_ema": "encoder_to_latent",
    "latent_norm_ema": "latent_norm",
}


@dataclass
class TTSInitializationReport:
    checkpoint: str
    loaded: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    mismatched: list[str] = field(default_factory=list)
    unused_source: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"TTS module initialization: checkpoint={self.checkpoint} "
            f"loaded={len(self.loaded)} missing={len(self.missing)} "
            f"mismatch={len(self.mismatched)} unused_source={len(self.unused_source)}"
        )

    def details(self, *, limit: int = 8) -> list[str]:
        lines = [self.summary()]
        for label, values in (
            ("loaded", self.loaded),
            ("missing", self.missing),
            ("mismatch", self.mismatched),
            ("unused_source", self.unused_source),
        ):
            if values:
                lines.append(f"  {label}[:{limit}]={values[:limit]}")
        return lines


def initialize_from_tts_checkpoint(
    model: nn.Module,
    checkpoint: str | Path,
) -> TTSInitializationReport:
    """Initialize only reusable audio encoder/decoder tensors from a TTS checkpoint.

    Prior-FM and all text-related keys are excluded by construction.  Exact
    state-dict names are preferred; a canonical suffix match also supports a
    TTA model that wraps the reused modules in ``audio_encoder`` or
    ``audio_decoder`` containers.
    """

    source_path = resolve_model_state_path(checkpoint)
    source = load_tensor_state(source_path)
    target = model.state_dict()
    source_allowed = {key: value for key, value in source.items() if _is_reusable_key(key)}
    source_by_canonical: dict[str, list[tuple[str, Tensor]]] = {}
    for key, value in source_allowed.items():
        canonical = _canonical_component_key(key)
        if canonical is not None:
            source_by_canonical.setdefault(canonical, []).append((key, value))

    report = TTSInitializationReport(checkpoint=str(source_path))
    selected: dict[str, Tensor] = {}
    used_source: set[str] = set()
    for target_key, target_value in target.items():
        if not _is_reusable_key(target_key):
            continue
        candidates: list[tuple[str, Tensor]] = []
        if target_key in source_allowed:
            candidates.append((target_key, source_allowed[target_key]))
        canonical = _canonical_component_key(target_key)
        if canonical is not None:
            candidates.extend(source_by_canonical.get(canonical, ()))
        candidates = _unique_candidates(candidates)
        shape_matches = [item for item in candidates if tuple(item[1].shape) == tuple(target_value.shape)]
        if len(shape_matches) == 1:
            source_key, source_value = shape_matches[0]
            selected[target_key] = source_value
            used_source.add(source_key)
            report.loaded.append(f"{target_key} <- {source_key}")
        elif len(shape_matches) > 1:
            exact = [item for item in shape_matches if item[0] == target_key]
            if len(exact) == 1:
                source_key, source_value = exact[0]
                selected[target_key] = source_value
                used_source.add(source_key)
                report.loaded.append(f"{target_key} <- {source_key}")
            else:
                report.mismatched.append(
                    f"{target_key}: ambiguous sources={[key for key, _ in shape_matches]}"
                )
        elif candidates:
            shapes = [(key, tuple(value.shape)) for key, value in candidates]
            report.mismatched.append(
                f"{target_key}: target={tuple(target_value.shape)} sources={shapes}"
            )
        else:
            report.missing.append(target_key)

    if selected:
        model.load_state_dict(selected, strict=False)
        # Some reusable TTS checkpoints predate the split target EMA. When the
        # online encoder was initialized but its EMA counterpart was absent,
        # start EMA from the loaded online value instead of the model's random
        # pre-load copy.
        current = model.state_dict()
        ema_fallback: dict[str, Tensor] = {}
        for target_key, target_value in current.items():
            root, separator, suffix = target_key.partition(".")
            online_root = _EMA_TO_ONLINE_ROOT.get(root)
            if online_root is None or target_key in selected:
                continue
            online_key = online_root + (separator + suffix if separator else "")
            online_value = current.get(online_key)
            if online_key not in selected or online_value is None:
                continue
            if tuple(online_value.shape) != tuple(target_value.shape):
                continue
            ema_fallback[target_key] = online_value.detach().clone()
            report.loaded.append(f"{target_key} <- {online_key} (EMA fallback)")
        if ema_fallback:
            model.load_state_dict(ema_fallback, strict=False)
            fallback_keys = set(ema_fallback)
            report.missing = [key for key in report.missing if key not in fallback_keys]
    report.unused_source = sorted(set(source_allowed).difference(used_source))
    return report


def _is_reusable_key(key: str) -> bool:
    parts = set(key.split("."))
    if parts & _FORBIDDEN_SEGMENTS:
        return False
    return bool(parts & _ALLOWED_ROOTS)


def _canonical_component_key(key: str) -> str | None:
    parts = key.split(".")
    while parts and parts[0] in {"module", "model"}:
        parts.pop(0)
    if "wave_fm_decoder" in parts:
        index = parts.index("wave_fm_decoder")
        return ".".join(parts[index:])
    indexes = [index for index, part in enumerate(parts) if part in _ALLOWED_ROOTS]
    if not indexes:
        return None
    # Repeated ``decoder.decoder`` means the first item is a TTA wrapper.
    index = indexes[-1] if parts[indexes[0]] == "decoder" else indexes[0]
    return ".".join(parts[index:])


def _unique_candidates(candidates: Iterable[tuple[str, Tensor]]) -> list[tuple[str, Tensor]]:
    result: list[tuple[str, Tensor]] = []
    seen: set[str] = set()
    for key, value in candidates:
        if key not in seen:
            seen.add(key)
            result.append((key, value))
    return result
