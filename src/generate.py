#!/usr/bin/env python3
"""Generate audio from one of the checkpoints bundled with Unite-Audio."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch


INFERENCE_ROOT = Path(__file__).resolve().parent
BUNDLE_ROOT = INFERENCE_ROOT.parent
if str(INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(INFERENCE_ROOT))

from task_audio.config import load_config  # noqa: E402
from task_audio.checkpoints import (  # noqa: E402
    load_model_state,
    resolve_checkpoint_dir,
    resolve_inference_state_path,
)


MODELS: dict[str, dict[str, Any]] = {
    "s2_noisy_30ep": {"cfg_strength": 4.0, "prefer_ema": True},
    "s3_rl14": {"cfg_strength": 4.0, "prefer_ema": True},
    "decoder_spectral": {"cfg_strength": 5.0, "prefer_ema": False},
    "decoder_mid3_gan": {"cfg_strength": 5.0, "prefer_ema": False},
}


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    model_id, checkpoint = resolve_model(args)
    checkpoint_dir = resolve_checkpoint_dir(checkpoint)
    config_path = args.config or find_bundle_config(checkpoint_dir)
    config = load_config(config_path)
    config = override_text_encoder(
        config,
        name_or_path=args.text_encoder,
        local_files_only=args.local_files_only,
    )

    prefer_ema = resolve_weight_preference(args, model_id)
    state_path, using_ema = resolve_inference_state_path(
        checkpoint_dir, prefer_ema=prefer_ema
    )

    device = resolve_device(args.device)
    from task_audio.runtime import build_model

    model = build_model(model_config(config))
    missing, unexpected = load_model_state(model, state_path)
    model.to(device).eval()

    sampling = config.get("sampling", {})
    if not isinstance(sampling, Mapping):
        sampling = {}
    captions = load_captions(args.caption, args.captions_file)
    output_dir = args.output_dir or default_output_dir(model_id)
    cfg_default = MODELS.get(model_id, {}).get("cfg_strength", sampling.get("cfg_strength", 2.0))
    records = generate_to_directory(
        model,
        captions,
        output_dir,
        seconds=resolve(args.seconds, sampling.get("seconds"), 10.0, float),
        prior_steps=resolve(args.prior_steps, sampling.get("prior_steps"), 32, int),
        wave_steps=resolve(args.wave_steps, sampling.get("wave_steps"), 1, int),
        solver=resolve(args.solver, sampling.get("solver"), "euler", str),
        cfg_strength=resolve(args.cfg_strength, cfg_default, 2.0, float),
        cfg_rescale=resolve(args.cfg_rescale, sampling.get("cfg_rescale"), 0.0, float),
        seed=resolve(args.seed, sampling.get("seed"), 1234, int),
        sample_rate=sample_rate(config),
    )
    run_info = {
        "model": model_id,
        "checkpoint": str(checkpoint_dir),
        "config": str(config_path),
        "state": str(state_path),
        "weights": "ema" if using_ema else "online",
        "text_encoder": args.text_encoder,
        "local_files_only": bool(args.local_files_only),
        "device": str(device),
        "generated": len(records),
        "missing_frozen_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }
    output_dir = Path(output_dir).expanduser().resolve()
    (output_dir / "run.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_info, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unite-Audio text-to-audio inference")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model", choices=tuple(MODELS), default="s3_rl14")
    source.add_argument("--checkpoint", type=Path, help="custom checkpoint directory or state file")
    parser.add_argument("--config", type=Path, help="custom resolved JSON config")
    parser.add_argument(
        "--weights",
        choices=("default", "ema", "online"),
        default="default",
        help="select EMA or online model weights (default is model-specific)",
    )
    parser.add_argument(
        "--text-encoder",
        default="google/flan-t5-large",
        help="Hugging Face name or local FLAN-T5-large directory",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--caption", action="append", default=[], help="repeat for multiple prompts")
    parser.add_argument("--captions-file", type=Path, help="UTF-8 file with one prompt per line")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--prior-steps", type=int)
    parser.add_argument("--wave-steps", type=int)
    parser.add_argument("--solver", choices=("euler", "heun", "rk4"))
    parser.add_argument("--cfg-strength", type=float)
    parser.add_argument("--cfg-rescale", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_model(args: argparse.Namespace) -> tuple[str, Path]:
    if args.checkpoint is not None:
        return "custom", args.checkpoint.expanduser()
    return str(args.model), BUNDLE_ROOT / "checkpoints" / str(args.model)


def resolve_weight_preference(args: argparse.Namespace, model_id: str) -> bool:
    if args.weights == "ema":
        return True
    if args.weights == "online":
        return False
    return bool(MODELS.get(model_id, {}).get("prefer_ema", True))


def find_bundle_config(checkpoint_dir: Path) -> Path:
    config_dir = checkpoint_dir / "config"
    for name in ("release_config.json",):
        candidate = config_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no bundled resolved config in {config_dir}")


def override_text_encoder(
    config: Mapping[str, Any], *, name_or_path: str, local_files_only: bool
) -> dict[str, Any]:
    result = deepcopy(dict(config))
    model = result.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("resolved config is missing a model section")
    model = dict(model)
    text = dict(model.get("text", {}))
    text["name_or_path"] = str(name_or_path)
    text["local_files_only"] = bool(local_files_only)
    model["text"] = text
    model["flan_t5_name_or_path"] = str(name_or_path)
    result["model"] = model
    return result


def model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("resolved config is missing a model section")
    result = dict(model)
    return result


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {value}")
    return device


def default_output_dir(model_id: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return BUNDLE_ROOT / "outputs" / f"{model_id}_{timestamp}"


def sample_rate(config: Mapping[str, Any]) -> int:
    for section in (config.get("data"), config.get("model")):
        if isinstance(section, Mapping) and section.get("sample_rate") is not None:
            return int(section["sample_rate"])
    return 16_000


def resolve(cli_value: Any, config_value: Any, fallback: Any, cast: Any) -> Any:
    return cast(cli_value if cli_value is not None else config_value if config_value is not None else fallback)


if __name__ == "__main__":
    main()
