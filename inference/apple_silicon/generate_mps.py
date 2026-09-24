#!/usr/bin/env python3
"""M4 Pro launcher for the Unite-Audio release inference runtime."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parents[1]
DEFAULT_TEXT_ENCODER = "google/flan-t5-large"
TEXT_ENCODER_CACHE = (
    Path.home() / ".cache" / "huggingface" / "hub" / "models--google--flan-t5-large"
)
MODELS = ("s3_rl14", "s2_noisy_30ep", "decoder_spectral", "decoder_mid3_gan")
FLOW_MODELS = {
    "s2_noisy_30ep": "s2_noisy_30ep",
    "s3_rl14_ema": "s3_rl14",
    "s3_rl14_online": "decoder_spectral",
}
DECODER_MODELS = {
    "base": "s3_rl14",
    "spectral": "decoder_spectral",
    "mid3_gan": "decoder_mid3_gan",
}
PACKAGED_MODELS = {
    "s2_noisy_30ep": ("s2_noisy_30ep", "base"),
    "s3_rl14": ("s3_rl14_ema", "base"),
    "decoder_spectral": ("s3_rl14_online", "spectral"),
    "decoder_mid3_gan": ("s3_rl14_online", "mid3_gan"),
}
MODEL_STATES = {
    "s3_rl14": "ema_model.safetensors",
    "s2_noisy_30ep": "ema_model.safetensors",
    "decoder_spectral": "model.safetensors",
    "decoder_mid3_gan": "model.safetensors",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Unite-Audio samples locally on Apple Silicon"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--caption", help="text condition to synthesize")
    source.add_argument(
        "--tangoflux-index",
        type=int,
        help="use caption N from the bundled 886-item TangoFlux evaluation set",
    )
    parser.add_argument(
        "--model",
        choices=MODELS,
        help="compatibility shortcut selecting an original packaged pairing",
    )
    parser.add_argument("--flow-model", choices=tuple(FLOW_MODELS))
    parser.add_argument("--decoder-model", choices=tuple(DECODER_MODELS))
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--prior-steps", type=int, default=32)
    parser.add_argument("--wave-steps", type=int, default=1)
    parser.add_argument("--cfg-strength", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--device",
        choices=("mps", "cpu"),
        default="mps",
        help="MPS uses the Apple GPU; CPU is a slow compatibility fallback",
    )
    parser.add_argument(
        "--text-encoder",
        help="Hugging Face model name or local directory (default: cached FLAN-T5-large)",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_tangoflux_item(repo: Path, index: int) -> dict[str, Any]:
    path = repo / "evaluation" / "audiocaps_tangoflux_886.json"
    if not path.is_file():
        raise FileNotFoundError(f"TangoFlux metadata was not found: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not 0 <= index < len(rows):
        raise IndexError(f"TangoFlux index must be in [0, {len(rows) - 1}], got {index}")
    return dict(rows[index])


def prepare_checkpoint_alias(repo: Path, model: str) -> Path:
    """Expose EMA-only releases through the upstream directory resolver."""

    source_dir = repo / "checkpoints" / model
    source_state = source_dir / MODEL_STATES[model]
    if not source_state.is_file():
        raise FileNotFoundError(f"Unite-Audio checkpoint was not found: {source_state}")

    alias_dir = HERE / ".runtime" / "checkpoints" / model
    alias_dir.mkdir(parents=True, exist_ok=True)
    model_alias = alias_dir / "model.safetensors"
    if not model_alias.exists():
        model_alias.symlink_to(source_state)
    if source_state.name == "ema_model.safetensors":
        ema_alias = alias_dir / "ema_model.safetensors"
        if not ema_alias.exists():
            ema_alias.symlink_to(source_state)
    config_alias = alias_dir / "config"
    if not config_alias.exists():
        config_alias.symlink_to(source_dir / "config", target_is_directory=True)
    return alias_dir


def resolve_alias_state(checkpoint_dir: Path) -> Path:
    ema = checkpoint_dir / "ema_model.safetensors"
    return ema.resolve() if ema.is_file() else (checkpoint_dir / "model.safetensors").resolve()


def resolve_text_encoder(value: str | None) -> tuple[str, bool]:
    if value:
        path = Path(value).expanduser()
        return (str(path.resolve()), True) if path.is_dir() else (value, False)

    ref = TEXT_ENCODER_CACHE / "refs" / "main"
    if ref.is_file():
        revision = ref.read_text(encoding="utf-8").strip()
        snapshot = TEXT_ENCODER_CACHE / "snapshots" / revision
        required = ("config.json", "model.safetensors", "spiece.model")
        if all((snapshot / name).is_file() for name in required):
            return str(snapshot.resolve()), True
    return DEFAULT_TEXT_ENCODER, False


def main() -> None:
    args = parse_args()
    repo = args.repo.expanduser().resolve()
    entrypoint = repo / "inference" / "generate.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"Unite-Audio inference entrypoint was not found: {entrypoint}")
    packaged_flow, packaged_decoder = PACKAGED_MODELS.get(
        args.model or "s3_rl14", PACKAGED_MODELS["s3_rl14"]
    )
    flow_model = args.flow_model or packaged_flow
    decoder_model = args.decoder_model or packaged_decoder
    flow_checkpoint_id = FLOW_MODELS[flow_model]
    decoder_checkpoint_id = DECODER_MODELS[decoder_model]
    flow_checkpoint = prepare_checkpoint_alias(repo, flow_checkpoint_id)
    decoder_checkpoint = prepare_checkpoint_alias(repo, decoder_checkpoint_id)
    flow_state = resolve_alias_state(flow_checkpoint)
    decoder_state = resolve_alias_state(decoder_checkpoint)

    evaluation_item = None
    caption = args.caption
    if args.tangoflux_index is not None:
        evaluation_item = load_tangoflux_item(repo, args.tangoflux_index)
        caption = str(evaluation_item["caption"])
    assert caption is not None
    text_encoder, cached_text_encoder = resolve_text_encoder(args.text_encoder)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else HERE / "outputs" / f"{flow_model}__{decoder_model}_{timestamp}"
    )
    python = HERE / ".venv" / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(f"environment not found; run {HERE / 'setup.sh'} first")

    command = [
        str(python),
        str(HERE / "inference_shim.py"),
        "--checkpoint",
        str(flow_checkpoint),
        "--caption",
        caption,
        "--output-dir",
        str(output_dir),
        "--seconds",
        str(args.seconds),
        "--prior-steps",
        str(args.prior_steps),
        "--wave-steps",
        str(args.wave_steps),
        "--cfg-strength",
        str(args.cfg_strength),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--text-encoder",
        text_encoder,
    ]
    if args.local_files_only or cached_text_encoder:
        command.append("--local-files-only")

    print(f"Caption: {caption}")
    print(f"Device: {args.device}")
    print(f"Flow model: {flow_model}")
    print(f"Decoder model: {decoder_model}")
    print(f"Text encoder: {text_encoder}")
    print(f"Output: {output_dir}")
    if args.dry_run:
        print("Command:", " ".join(command))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if evaluation_item is not None:
        (output_dir / "evaluation_item.json").write_text(
            json.dumps(evaluation_item, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    env = os.environ.copy()
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env["UNITE_AUDIO_ENTRYPOINT"] = str(entrypoint)
    env["UNITE_AUDIO_FLOW_STATE"] = str(flow_state)
    env["UNITE_AUDIO_DECODER_STATE"] = str(decoder_state)
    subprocess.run(command, env=env, check=True)

    run_path = output_dir / "run.json"
    if run_path.is_file():
        run_info = json.loads(run_path.read_text(encoding="utf-8"))
        run_info.update(
            {
                "flow_model": flow_model,
                "flow_checkpoint_id": flow_checkpoint_id,
                "flow_state": str(flow_state),
                "decoder_model": decoder_model,
                "decoder_checkpoint_id": decoder_checkpoint_id,
                "decoder_state": str(decoder_state),
            }
        )
        run_path.write_text(
            json.dumps(run_info, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
