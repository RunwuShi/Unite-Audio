from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from torch import Tensor, nn


MODEL_FILENAMES = ("model.safetensors", "pytorch_model.bin", "model.bin", "model.pt")
EMA_MODEL_FILENAME = "ema_model.safetensors"


def resolve_checkpoint_dir(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        return candidate.parent
    if candidate.is_dir() and any((candidate / name).is_file() for name in MODEL_FILENAMES):
        return candidate
    raise FileNotFoundError(f"no model state found in checkpoint directory: {candidate}")


def resolve_model_state_path(path: str | Path) -> Path:
    directory = resolve_checkpoint_dir(path)
    for name in MODEL_FILENAMES:
        state_path = directory / name
        if state_path.is_file():
            return state_path
    raise FileNotFoundError(f"no supported state file in {directory}")


def resolve_inference_state_path(
    path: str | Path,
    *,
    prefer_ema: bool,
) -> tuple[Path, bool]:
    """Return the selected state file and whether it is an EMA state."""

    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        directory = candidate.parent
        ema_path = directory / EMA_MODEL_FILENAME
        if prefer_ema and candidate != ema_path and ema_path.is_file():
            return ema_path, True
        return candidate, candidate.name == EMA_MODEL_FILENAME

    directory = resolve_checkpoint_dir(candidate)
    ema_path = directory / EMA_MODEL_FILENAME
    if prefer_ema and ema_path.is_file():
        return ema_path, True
    return resolve_model_state_path(directory), False


def load_model_state(model: nn.Module, path: str | Path) -> tuple[list[str], list[str]]:
    state_path = Path(path).expanduser().resolve()
    if state_path.suffix != ".safetensors":
        raise ValueError("this bundle requires safetensors model states")
    try:
        from safetensors.torch import load_file
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("install safetensors to load the bundled checkpoints") from exc
    state: Mapping[str, Any] = load_file(str(state_path), device="cpu")
    filter_state = getattr(model, "filter_checkpoint_state", None)
    if not callable(filter_state):
        raise TypeError("runtime model does not expose checkpoint filtering")
    state = filter_state(state)
    expected = model.checkpoint_model_state(to_cpu=False)
    current = model.state_dict()
    mismatched = [
        key for key, value in state.items()
        if key in current and tuple(value.shape) != tuple(current[key].shape)
    ]
    unexpected = sorted(set(state).difference(current))
    missing_saved = sorted(set(expected).difference(state))
    if mismatched or unexpected or missing_saved:
        raise RuntimeError(
            "checkpoint is incompatible: "
            f"missing={missing_saved[:8]} unexpected={unexpected[:8]} mismatched={mismatched[:8]}"
        )
    missing, unexpected_load = model.load_state_dict(state, strict=False)
    return list(missing), list(unexpected_load)
