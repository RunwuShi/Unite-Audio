from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn


MODEL_FILENAMES = ("model.safetensors", "pytorch_model.bin", "model.bin", "model.pt")


def resolve_checkpoint_dir(path: str | Path) -> Path:
    """Resolve a checkpoint directory from a state file, checkpoint, or run dir."""

    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        return candidate.parent
    if not candidate.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {candidate}")
    if _has_model_state(candidate):
        return candidate

    preferred = (
        candidate / "checkpoint-last",
        candidate / "checkpoint" / "checkpoint-last",
    )
    for item in preferred:
        if _has_model_state(item):
            return item

    roots = (candidate / "checkpoint", candidate)
    numbered: list[Path] = []
    for root in roots:
        if root.is_dir():
            numbered.extend(item for item in root.glob("checkpoint-*") if _has_model_state(item))
    if numbered:
        return sorted(set(numbered), key=_checkpoint_sort_key)[-1]
    raise FileNotFoundError(f"could not find model state below checkpoint/run directory: {candidate}")


def resolve_model_state_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        return candidate
    directory = resolve_checkpoint_dir(candidate)
    for filename in MODEL_FILENAMES:
        state_path = directory / filename
        if state_path.is_file():
            return state_path
    raise FileNotFoundError(f"checkpoint has no supported model state: {directory}")


def resolve_inference_state_path(
    path: str | Path,
    *,
    prefer_ema: bool = True,
) -> tuple[Path, bool]:
    """Select EMA weights for inference when the resolved checkpoint has them."""

    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        ema_path = candidate.parent / "ema_model.safetensors"
        if prefer_ema and candidate != ema_path and ema_path.is_file():
            return ema_path, True
        return candidate, candidate.name == "ema_model.safetensors"
    checkpoint_dir = resolve_checkpoint_dir(candidate)
    ema_path = checkpoint_dir / "ema_model.safetensors"
    if prefer_ema and ema_path.is_file():
        return ema_path, True
    return resolve_model_state_path(checkpoint_dir), False


def load_tensor_state(path: str | Path) -> dict[str, Tensor]:
    state_path = resolve_model_state_path(path)
    if state_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "loading .safetensors checkpoints requires safetensors"
            ) from exc
        state: Any = load_file(str(state_path), device="cpu")
    else:
        try:
            state = torch.load(str(state_path), map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(str(state_path), map_location="cpu")
    if isinstance(state, Mapping) and isinstance(state.get("state_dict"), Mapping):
        state = state["state_dict"]
    if not isinstance(state, Mapping):
        raise TypeError(f"model state must be a mapping: {state_path}")
    result = {
        _strip_ddp_prefix(str(key)): value
        for key, value in state.items()
        if isinstance(value, Tensor)
    }
    if not result:
        raise ValueError(f"model checkpoint contains no tensors: {state_path}")
    return result


def load_tta_model_state(
    model: nn.Module,
    path: str | Path,
    *,
    require_all_checkpoint_keys: bool = True,
) -> tuple[list[str], list[str]]:
    """Load a filtered state while allowing omitted frozen T5/CLAP keys."""

    state = load_tensor_state(path)
    expected = checkpoint_model_state(model, to_cpu=False)
    current = model.state_dict()
    shape_mismatch = [
        key
        for key, value in state.items()
        if key in current and tuple(value.shape) != tuple(current[key].shape)
    ]
    if shape_mismatch:
        details = ", ".join(
            f"{key}: saved={tuple(state[key].shape)} current={tuple(current[key].shape)}"
            for key in shape_mismatch[:8]
        )
        raise RuntimeError(f"checkpoint shape mismatch ({len(shape_mismatch)}): {details}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing_saved = sorted(set(expected).difference(state))
    if require_all_checkpoint_keys and (missing_saved or unexpected):
        raise RuntimeError(
            "checkpoint is incompatible with the current TTA model: "
            f"missing_saved={missing_saved[:8]} unexpected={list(unexpected)[:8]}"
        )
    # ``missing`` normally contains frozen T5/CLAP conditioner encoder parameters.
    return list(missing), list(unexpected)


def initialize_from_full_tta_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    allowed_missing_prefixes: tuple[str, ...] = (
        "fm_encoder.global_condition_proj.",
    ),
) -> list[str]:
    """Warm-start a compatible expanded TTA model from a complete checkpoint.

    Unlike ``resume_from``, this intentionally restores model weights only.
    Optimizer/scheduler state is left fresh so newly-added trainable modules do
    not inherit misaligned Adam slots.  Every saved key must still exist and
    match shape, and only explicitly-listed new keys may be absent.
    """

    state = load_tensor_state(path)
    expected = checkpoint_model_state(model, to_cpu=False)
    current = model.state_dict()
    unexpected = sorted(set(state).difference(current))
    mismatched = sorted(
        key
        for key, value in state.items()
        if key in current and tuple(value.shape) != tuple(current[key].shape)
    )
    missing = sorted(set(expected).difference(state))
    disallowed_missing = [
        key
        for key in missing
        if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    if unexpected or mismatched or disallowed_missing:
        raise RuntimeError(
            "full TTA warm start is incompatible: "
            f"missing={disallowed_missing[:8]} mismatched={mismatched[:8]} "
            f"unexpected={unexpected[:8]}"
        )
    model.load_state_dict(state, strict=False)

    # The newly-added CLAP branch must begin as an exact no-op when branching
    # from a text-only prior.  A zero output projection preserves the old
    # function while still receiving gradients on the first update.
    if any(key.startswith("fm_encoder.global_condition_proj.") for key in missing):
        fm_encoder = getattr(model, "fm_encoder", None)
        projection = getattr(fm_encoder, "global_condition_proj", None)
        output = getattr(projection, "w2", None)
        if not isinstance(output, nn.Linear):
            raise RuntimeError(
                "CLAP warm start expected fm_encoder.global_condition_proj.w2"
            )
        nn.init.zeros_(output.weight)
    return missing


def checkpoint_model_state(
    model: nn.Module,
    *,
    to_cpu: bool | None = None,
) -> dict[str, Tensor]:
    method = getattr(model, "checkpoint_model_state", None)
    if not callable(method):
        raise AttributeError(
            "TTA models must implement checkpoint_model_state() so frozen text-encoder "
            "weights are not serialized"
        )
    parameters = inspect.signature(method).parameters
    supports_to_cpu = "to_cpu" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    state = method(to_cpu=to_cpu) if to_cpu is not None and supports_to_cpu else method()
    if not isinstance(state, Mapping):
        raise TypeError("model.checkpoint_model_state() must return a mapping")
    result = {str(key): value for key, value in state.items() if isinstance(value, Tensor)}
    if len(result) != len(state):
        raise TypeError("model.checkpoint_model_state() values must all be tensors")
    if to_cpu is True and not supports_to_cpu:
        return {key: value.detach().cpu() for key, value in result.items()}
    return result


def _has_model_state(path: Path) -> bool:
    return path.is_dir() and any((path / name).is_file() for name in MODEL_FILENAMES)


def _strip_ddp_prefix(key: str) -> str:
    while key.startswith("module."):
        key = key[len("module.") :]
    return key


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.name.removeprefix("checkpoint-")
    return (int(suffix) if suffix.isdigit() else -1, path.name)
