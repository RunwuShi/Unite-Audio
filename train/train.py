#!/usr/bin/env python3
"""Run one preserved Unite-Audio training stage from the internal bundle."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parent
BUNDLE_ROOT = ROOT.parent
CHECKPOINTS = BUNDLE_ROOT / "checkpoints"
CONFIGS = ROOT / "configs"
STAGES = {
    "s1_noisy_240k": {"engine": "pmf", "pmf_stage": "stage1"},
    "s2_noisy_30ep": {"engine": "pmf", "pmf_stage": "stage2"},
    "s3_rl14": {"engine": "rl"},
    "decoder_spectral": {"engine": "pmf", "pmf_stage": "stage2"},
    "decoder_mid3_gan": {"engine": "pmf", "pmf_stage": "stage2"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--data-paths", type=Path, default=ROOT / "data_paths.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gan-start-step", type=int)
    parser.add_argument("--prepare-passt-cache", action="store_true")
    parser.add_argument("--prepare-fad-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_data_paths(args.data_paths)
    output_dir = (args.output_dir or default_output_dir(args.stage)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = resolve_stage_config(args.stage, paths, output_dir, args.resume_from)
    config_path = output_dir / "config" / "resolved_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    command = build_command(args, config_path, output_dir)
    environment = dict(os.environ)
    source_root = str(ROOT / "src")
    environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")
    prepare_reward_caches(args, config_path, environment)
    print("[train] " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=environment)


def load_data_paths(path: Path) -> dict[str, str]:
    if not path.is_file():
        example = ROOT / "data_paths.example.yaml"
        raise FileNotFoundError(f"missing {path}; copy and edit {example}")
    values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(values, Mapping):
        raise TypeError("data paths YAML must be a mapping")
    return {str(key): str(value) for key, value in values.items() if value is not None}


def resolve_stage_config(
    stage: str,
    paths: Mapping[str, str],
    output_dir: Path,
    resume_from: Path | None,
) -> dict[str, Any]:
    config = json.loads((CONFIGS / f"{stage}.json").read_text(encoding="utf-8"))
    config = deepcopy(config)
    if STAGES[stage]["engine"] == "rl":
        _resolve_rl_paths(config, paths, output_dir, resume_from)
    else:
        _resolve_pmf_paths(config, stage, paths, output_dir, resume_from)
    return config


def _resolve_pmf_paths(
    config: dict[str, Any],
    stage: str,
    values: Mapping[str, str],
    output_dir: Path,
    resume_from: Path | None,
) -> None:
    paths = config.setdefault("paths", {})
    _set_paths(paths, values, ("audioset_root", "audiosetcaps_captions", "audiocaps_root", "wavcaps_root"))
    paths["project_root"] = str(BUNDLE_ROOT)
    paths["experiments_root"] = str(ROOT / "outputs")
    paths["output_dir"] = str(output_dir)
    model = config.setdefault("model", {})
    if values.get("flan_t5"):
        model["flan_t5_name_or_path"] = values["flan_t5"]
        text = model.get("text")
        if isinstance(text, Mapping):
            text = dict(text)
            text["name_or_path"] = values["flan_t5"]
            model["text"] = text
    training = config.setdefault("training", {})
    training["resume_from"] = None if resume_from is None else str(resume_from.expanduser().resolve())
    initialization = config.setdefault("initialization", {})
    if stage == "s1_noisy_240k":
        # The archived checkpoint resumes exactly; a fresh S1 run requires an
        # external base checkpoint, so do not retain an unavailable path.
        initialization.pop("full_tta_checkpoint", None)
    elif stage == "s2_noisy_30ep":
        initialization["full_tta_checkpoint"] = str(CHECKPOINTS / "s1_noisy_240k")
    else:
        initialization["full_tta_checkpoint"] = str(CHECKPOINTS / "s3_rl14")
    _disable_periodic_evaluation(config)


def _resolve_rl_paths(
    config: dict[str, Any],
    values: Mapping[str, str],
    output_dir: Path,
    resume_from: Path | None,
) -> None:
    paths = config.setdefault("paths", {})
    paths["source_checkpoint"] = str(CHECKPOINTS / "s2_noisy_30ep")
    paths["source_resolved_config"] = str(
        CHECKPOINTS / "s2_noisy_30ep" / "config" / "resolved_config.json"
    )
    paths["output_root"] = str(output_dir.parent)
    _set_paths(paths, values, (
        "audiocaps_manifest", "audiocaps_root", "clap_checkpoint",
        "passt_reference_cache", "vggish_hub_dir", "vggish_fad_cache",
        "vggish_reference_cache", "panns_repo", "panns_checkpoint_dir",
        "panns_fd_reference_cache", "panns_fd_cache",
    ))
    if resume_from is not None:
        paths["resume_from"] = str(resume_from.expanduser().resolve())


def _set_paths(target: dict[str, Any], values: Mapping[str, str], keys: tuple[str, ...]) -> None:
    for key in keys:
        if values.get(key):
            target[key] = values[key]


def _disable_periodic_evaluation(config: dict[str, Any]) -> None:
    config.setdefault("audiocaps_eval", {})["enabled"] = False
    config.setdefault("encoder_eval", {})["every"] = 0
    config.setdefault("sampling", {})["sample_every"] = 0


def build_command(args: argparse.Namespace, config_path: Path, output_dir: Path) -> list[str]:
    stage = STAGES[args.stage]
    if stage["engine"] == "rl":
        command = [sys.executable, "-m", "task_audio_preference_rl.cli", "train", "--config", str(config_path), "--output-dir", str(output_dir)]
        if args.resume_from is not None:
            command.extend(("--resume-from", str(args.resume_from.expanduser().resolve())))
        return command

    command = [
        sys.executable, "-m", "task_audio_pmf.proven.cli_train",
        "--stage", str(stage["pmf_stage"]), "--config", str(config_path),
        "--output-dir", str(output_dir), "--no-auto-resume",
    ]
    if args.resume_from is not None:
        command.extend(("--resume-from", str(args.resume_from.expanduser().resolve())))
    if args.max_steps is not None:
        command.extend(("--max-steps", str(args.max_steps)))
    if args.batch_size is not None:
        command.extend(("--batch-size", str(args.batch_size)))
    if args.gan_start_step is not None:
        command.extend(("--gan-start-step", str(args.gan_start_step)))
    return command


def prepare_reward_caches(
    args: argparse.Namespace,
    config_path: Path,
    environment: Mapping[str, str],
) -> None:
    for enabled, command in (
        (args.prepare_passt_cache, "prepare-passt-cache"),
        (args.prepare_fad_cache, "prepare-fad-cache"),
    ):
        if not enabled:
            continue
        cache_command = [
            sys.executable,
            "-m",
            "task_audio_preference_rl.cli",
            command,
            "--config",
            str(config_path),
        ]
        print("[train] " + " ".join(cache_command), flush=True)
        subprocess.run(cache_command, check=True, env=dict(environment))


def default_output_dir(stage: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "outputs" / f"{stamp}_{stage}"


if __name__ == "__main__":
    main()
