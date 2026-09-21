from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a bundled resolved JSON configuration."""

    config_path = Path(path).expanduser().resolve()
    if config_path.suffix.lower() != ".json":
        raise ValueError("this release accepts resolved JSON configurations only")
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration does not exist: {config_path}")
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or not isinstance(value.get("model"), Mapping):
        raise ValueError("configuration is missing a model section")
    result = deepcopy(dict(value))
    result.setdefault("sampling", {})
    return result
