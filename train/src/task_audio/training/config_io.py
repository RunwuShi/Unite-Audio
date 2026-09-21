from __future__ import annotations

import importlib.util
import json
import shutil
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping


def load_config(path: str | Path | None = None, *, preset: str | None = None) -> dict[str, Any]:
    """Load a TTA config without importing training-only dependencies.

    Python configs may expose either ``build_config`` or ``CONFIG``.  When no
    path is supplied the public ``task_audio.configs.build_config`` entry point is
    used.
    """

    if path is None:
        from task_audio.configs import build_config

        config = build_config(preset=preset) if preset is not None else build_config()
        return _as_config_dict(config)

    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"TTA config does not exist: {config_path}")
    if config_path.suffix.lower() == ".json":
        return _as_config_dict(json.loads(config_path.read_text(encoding="utf-8")))
    if config_path.suffix.lower() != ".py":
        raise ValueError(f"TTA config must be a .py or .json file: {config_path}")

    spec = importlib.util.spec_from_file_location("latentaudio_tta_user_config", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import TTA config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "build_config"):
        builder = module.build_config
        config = builder(preset=preset) if preset is not None else builder()
    elif hasattr(module, "CONFIG"):
        if preset is not None:
            raise ValueError(f"config {config_path} exposes CONFIG, so --preset is unsupported")
        config = module.CONFIG
    else:
        raise ValueError(f"config must expose build_config() or CONFIG: {config_path}")
    return _as_config_dict(config)


def save_config_snapshot(
    config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    source_path: str | Path | None = None,
) -> Path:
    config_dir = Path(output_dir) / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = config_dir / "resolved_config.json"
    resolved_path.write_text(
        json.dumps(to_jsonable(config), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if source_path is not None:
        source = Path(source_path).expanduser().resolve()
        if source.exists():
            destination = config_dir / source.name
            if source != destination:
                shutil.copy2(source, destination)
    return resolved_path


def find_resolved_config(checkpoint: str | Path) -> Path:
    path = Path(checkpoint).expanduser().resolve()
    start = path if path.is_dir() else path.parent
    for directory in (start, *start.parents):
        candidates = (
            directory / "resolved_config.json",
            directory / "config" / "resolved_config.json",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f"could not find config/resolved_config.json above checkpoint: {path}; "
        "pass --config explicitly"
    )


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _as_config_dict(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        config = asdict(config)
    if not isinstance(config, Mapping):
        raise TypeError(f"TTA config must be a mapping, got {type(config).__name__}")
    result = deepcopy(dict(config))
    required = ("model", "data", "training")
    missing = [key for key in required if not isinstance(result.get(key), Mapping)]
    if missing:
        raise ValueError(f"TTA config is missing mapping sections: {', '.join(missing)}")
    result.setdefault("sampling", {})
    result.setdefault("initialization", {})
    result.setdefault("logging", {})
    result.setdefault("paths", {})
    result.setdefault("experiment", {})
    return result
