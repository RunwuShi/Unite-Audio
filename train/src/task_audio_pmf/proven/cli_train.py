#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_PYARROW_SITE = os.environ.get("PYARROW_SITE")
if _PYARROW_SITE:
    sys.path.insert(0, str(Path(_PYARROW_SITE).expanduser().resolve()))
    import pyarrow as _pinned_pyarrow

    if _pinned_pyarrow.__version__ != "18.1.0":
        raise RuntimeError("PYARROW_SITE must provide pyarrow==18.1.0")

import numpy as np
import torch

from task_audio.mask_contract import attach_mask_contract
from task_audio.training import TTATrainerConfig, load_config

from .data import (
    build_probe_dataloader,
    build_stage1_dataloader,
    build_stage2_dataloader,
)
from .model import build_model
from .trainer import PMFTrainer


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the proven TaskAudio pMF extension")
    parser.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--reset-data-cursor", action="store_true")
    parser.add_argument("--no-auto-resume", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--probe-group",
        choices=(
            "stage1_mixed",
            "audioset_10",
            "wavcaps_0_5",
            "wavcaps_5_10",
            "wavcaps_10_15",
            "wavcaps_15_20",
            "audiocaps_10_5",
        ),
    )
    parser.add_argument("--gan-start-step", type=int)
    return parser.parse_args()


def _default_config(stage: str) -> Path:
    return Path(__file__).resolve().parent / f"{stage}.py"


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    training = config["training"]
    if args.output_dir is not None:
        config.setdefault("paths", {})["output_dir"] = str(
            args.output_dir.expanduser().resolve()
        )
    if args.resume_from is not None:
        training["resume_from"] = str(args.resume_from.expanduser().resolve())
    if args.reset_data_cursor:
        training["reset_data_cursor"] = True
    if args.no_auto_resume:
        training["auto_resume"] = False
    if args.max_steps is not None:
        training["max_steps"] = int(args.max_steps)
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        training["batch_size"] = int(args.batch_size)
        if args.probe_group and args.stage == "stage1":
            config["data"]["duration_batch_sizes"][args.probe_group] = int(
                args.batch_size
            )
    if args.gan_start_step is not None:
        value = int(args.gan_start_step)
        config["model"]["adversarial_start_step"] = value
        config["model"]["loss"]["adversarial"]["start_step"] = value
    if args.probe_group is not None:
        if args.stage == "stage1":
            if args.probe_group == "audiocaps_10_5":
                raise ValueError("audiocaps_10_5 is a Stage 2 group")
            config["data"]["probe_group"] = args.probe_group
        elif args.probe_group != "audiocaps_10_5":
            raise ValueError("Stage 2 probe group must be audiocaps_10_5")
        config["audiocaps_eval"]["enabled"] = False
        config["encoder_eval"]["every"] = 0
        config["sampling"]["sample_every"] = 0
        config["logging"].update(
            {
                "save_every": 0,
                "milestone_every": 0,
                "save_final_checkpoint": False,
                "log_every": 1,
            }
        )


def _configure_trainable_parameters(
    model: torch.nn.Module,
    config: dict[str, Any],
) -> tuple[str, ...]:
    """Optionally restrict optimization to explicitly named model modules."""

    raw_prefixes = config.get("training", {}).get("trainable_module_prefixes")
    if raw_prefixes is None:
        return ()
    if isinstance(raw_prefixes, str):
        raw_prefixes = [raw_prefixes]
    prefixes = tuple(
        str(value).strip().rstrip(".") for value in raw_prefixes if str(value).strip()
    )
    if not prefixes:
        raise ValueError("training.trainable_module_prefixes cannot be empty")

    matched = {prefix: 0 for prefix in prefixes}
    for name, parameter in model.named_parameters():
        trainable = False
        for prefix in prefixes:
            if name == prefix or name.startswith(prefix + "."):
                matched[prefix] += parameter.numel()
                trainable = True
        parameter.requires_grad_(trainable)
    missing = [prefix for prefix, count in matched.items() if count == 0]
    if missing:
        raise ValueError(
            "training.trainable_module_prefixes matched no parameters: "
            + ", ".join(missing)
        )
    return prefixes


def main() -> None:
    args = parse_args()
    config_path = args.config or _default_config(args.stage)
    config = load_config(config_path)
    _apply_overrides(config, args)
    output = config.setdefault("paths", {}).get("output_dir")
    if not output:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = (
            ROOT
            / "experiments"
            / str(config["experiment"]["name"])
            / stamp
        )
        config["paths"]["output_dir"] = str(output)
    attach_mask_contract(config)
    seed = int(config.get("experiment", {}).get("seed", 1234))
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)

    if args.probe_group is not None:
        loader = build_probe_dataloader(
            config,
            group=args.probe_group,
            batch_size=int(config["training"]["batch_size"]),
        )
    else:
        loader = (
            build_stage1_dataloader(config)
            if args.stage == "stage1"
            else build_stage2_dataloader(config)
        )
    model_config = dict(config["model"])
    model_config["mask"] = dict(config["mask"])
    model = build_model(model_config)
    trainable_prefixes = _configure_trainable_parameters(model, config)
    trainer_config = TTATrainerConfig.from_config(config)
    trainer = PMFTrainer(
        model,
        trainer_config,
        resolved_config=config,
        config_source=config_path,
    )
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    objective_label = {
        "latent_tta_xpred": "XPred",
        "latent_tta_vpred": "VPred",
    }.get(str(model_config.get("architecture")), "PMF")
    target_label = (
        "full_wave->ema(stopgrad)"
        if bool(model_config.get("split_target_encoder_ema", True))
        else "full_wave->online(stopgrad)"
    )
    condition_label = (
        "none(full_generation)"
        if float(config.get("mask", {}).get("full_generation_probability", 0.0))
        == 1.0
        else "masked_wave->online"
    )
    trainer.accelerator.print(
        f"[{objective_label}] "
        f"stage={args.stage} steps={trainer_config.max_steps} "
        f"trainable={trainable:,} total={total:,} "
        f"trainable_modules={','.join(trainable_prefixes) if trainable_prefixes else 'all'} "
        f"target={target_label} "
        f"condition={condition_label} reconstruction=full_wave->online "
        f"output={trainer_config.output_dir}",
        flush=True,
    )
    trainer.train(loader)


if __name__ == "__main__":
    main()
