#!/usr/bin/env python3
"""Short command for local Unite-Audio generation."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        usage='python run.py "text caption" [steps]',
        description="Generate a 10-second audio sample with Unite-Audio on MPS.",
    )
    parser.add_argument("text", help="text caption")
    parser.add_argument("steps", nargs="?", type=int, help="override prior steps from config")
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error("steps must be at least 1")

    config = json.loads((ROOT / "inference_config.json").read_text(encoding="utf-8"))
    steps = args.steps if args.steps is not None else int(config.get("steps", 32))
    flow_model = str(config["flow_model"])
    decoder_model = str(config["decoder_model"])
    if flow_model not in config["available_flow_models"]:
        raise ValueError(f"unknown flow_model in config: {flow_model}")
    if decoder_model not in config["available_decoder_models"]:
        raise ValueError(f"unknown decoder_model in config: {decoder_model}")
    command = [
        str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / "generate_mps.py"),
        "--caption",
        args.text,
        "--prior-steps",
        str(steps),
        "--flow-model",
        flow_model,
        "--decoder-model",
        decoder_model,
        "--seconds",
        str(config.get("seconds", 10.0)),
        "--wave-steps",
        str(config.get("wave_steps", 1)),
        "--cfg-strength",
        str(config.get("cfg_strength", 4.0)),
        "--seed",
        str(config.get("seed", 1234)),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
