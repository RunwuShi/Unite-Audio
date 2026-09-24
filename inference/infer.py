#!/usr/bin/env python3
"""Generate audio with UNITE-AUDIO."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate audio with UNITE-AUDIO.")
    parser.add_argument("caption", help="text description to synthesize")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--device", help="override the configured device")
    parser.add_argument("--flow", help="override the configured flow model")
    parser.add_argument("--decoder", help="override the configured decoder")
    parser.add_argument("--seconds", type=float, help="audio duration")
    parser.add_argument("--steps", type=int, help="number of flow steps")
    parser.add_argument("--wave-steps", type=int, help="number of decoder steps")
    parser.add_argument("--cfg-strength", type=float, help="classifier-free guidance strength")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--text-encoder")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.expanduser().open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_device(value: str) -> str:
    if value != "auto":
        return value
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runtime = dict(config["runtime"])
    models = dict(config["models"])
    defaults = dict(config["default"])

    device = resolve_device(str(args.device or runtime["device"]))
    if device not in runtime["available_devices"]:
        raise ValueError(f"unsupported device: {device}")

    flow_key = str(args.flow or defaults["flow_model"])
    decoder_key = str(args.decoder or defaults["decoder_model"])
    if flow_key not in models or decoder_key not in models:
        raise ValueError("flow and decoder must be defined in config.json")

    command = [
        sys.executable,
        str(ROOT.parent / "src" / "runner.py"),
        "--caption",
        args.caption,
        "--flow-model",
        str(models[flow_key]["runtime_id"]),
        "--decoder-model",
        str(models[decoder_key]["runtime_id"]),
        "--device",
        device,
        "--seconds",
        str(args.seconds if args.seconds is not None else defaults["seconds"]),
        "--prior-steps",
        str(args.steps if args.steps is not None else defaults["steps"]),
        "--wave-steps",
        str(args.wave_steps if args.wave_steps is not None else defaults["wave_steps"]),
        "--cfg-strength",
        str(args.cfg_strength if args.cfg_strength is not None else defaults["cfg_strength"]),
        "--seed",
        str(args.seed if args.seed is not None else defaults["seed"]),
    ]
    if args.output_dir:
        command.extend(("--output-dir", str(args.output_dir.expanduser())))
    if args.text_encoder:
        command.extend(("--text-encoder", args.text_encoder))
    if args.local_files_only:
        command.append("--local-files-only")
    if args.dry_run:
        command.append("--dry-run")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
