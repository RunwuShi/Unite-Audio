from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from safetensors.torch import save_file

from task_audio.training.checkpointing import (
    checkpoint_model_state,
    load_tta_model_state,
    resolve_checkpoint_dir,
    resolve_model_state_path,
)


def timestamped_run_dir(root: str | Path, algorithm: str) -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    path = Path(root).expanduser().resolve() / f"{stamp}_{algorithm}_from240k"
    path.mkdir(parents=True, exist_ok=False)
    return path


def distributed_run_dir(accelerator: Any, root: str | Path, algorithm: str) -> Path:
    value = (
        str(timestamped_run_dir(root, algorithm)) if accelerator.is_main_process else ""
    )
    if accelerator.num_processes > 1:
        values = [value]
        torch.distributed.broadcast_object_list(values, src=0)
        value = str(values[0])
    return Path(value)


def seed_everything(seed: int, rank: int) -> torch.Generator:
    value = int(seed) + 100_003 * int(rank)
    random.seed(value)
    np.random.seed(value % (2**32 - 1))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    generator.manual_seed(value)
    return generator


@dataclass
class TrainerState:
    update: int = 0
    epoch: int = 0
    prompts_consumed_on_rank: int = 0


class PolicyEMA:
    def __init__(self, module: torch.nn.Module, decay: float = 0.9999) -> None:
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().float().clone()
            for key, value in module.state_dict().items()
            if torch.is_floating_point(value)
        }

    @torch.no_grad()
    def update(self, module: torch.nn.Module) -> None:
        current = module.state_dict()
        for key, target in self.shadow.items():
            target.lerp_(current[key].detach().float(), 1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {key: value.cpu() for key, value in self.shadow.items()}

    def load_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        missing = set(self.shadow).difference(state)
        if missing:
            raise RuntimeError(f"EMA state is missing keys: {sorted(missing)[:8]}")
        for key in self.shadow:
            self.shadow[key].copy_(state[key].to(self.shadow[key].device).float())


def save_preference_checkpoint(
    *,
    run_dir: Path,
    label: str,
    accelerator: Any,
    base_model: torch.nn.Module,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: PolicyEMA,
    trainer_state: TrainerState,
    config: Mapping[str, Any],
    generator: torch.Generator | None = None,
    reward: Any | None = None,
) -> Path:
    checkpoint_dir = run_dir / "checkpoint" / label
    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    torch.save(
        {
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
            "rollout_generator": None if generator is None else generator.get_state(),
        },
        checkpoint_dir / f"rank{accelerator.process_index:02d}_rng.pt",
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        online = checkpoint_model_state(base_model, to_cpu=True)
        save_file(online, str(checkpoint_dir / "model.safetensors"))

        ema_full = {key: value.clone() for key, value in online.items()}
        for policy_key, value in ema.state_dict().items():
            if policy_key in ema_full:
                ema_full[policy_key] = value.to(dtype=ema_full[policy_key].dtype)
        save_file(ema_full, str(checkpoint_dir / "ema_model.safetensors"))
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "policy_ema": ema.state_dict(),
                "reward": None if reward is None else reward.state_dict(),
            },
            checkpoint_dir / "training_state.pt",
        )
        (checkpoint_dir / "trainer_state.json").write_text(
            json.dumps(asdict(trainer_state), indent=2) + "\n", encoding="utf-8"
        )
        (checkpoint_dir / "preference_config.json").write_text(
            json.dumps(dict(config), indent=2) + "\n", encoding="utf-8"
        )
        last = run_dir / "checkpoint" / "checkpoint-last"
        if last.is_symlink() or last.exists():
            if last.is_dir() and not last.is_symlink():
                shutil.rmtree(last)
            else:
                last.unlink()
        last.symlink_to(checkpoint_dir.name, target_is_directory=True)
    accelerator.wait_for_everyone()
    return checkpoint_dir


def load_preference_checkpoint(
    checkpoint: str | Path,
    *,
    accelerator: Any,
    base_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: PolicyEMA,
    generator: torch.Generator | None = None,
    reward: Any | None = None,
) -> tuple[Path, TrainerState]:
    directory = resolve_checkpoint_dir(checkpoint)
    # Resume optimization from online weights; inference continues to prefer EMA.
    load_tta_model_state(base_model, resolve_model_state_path(directory))
    try:
        training = torch.load(
            directory / "training_state.pt", map_location="cpu", weights_only=True
        )
    except TypeError:
        training = torch.load(directory / "training_state.pt", map_location="cpu")
    optimizer.load_state_dict(training["optimizer"])
    ema.load_state_dict(training["policy_ema"])
    if reward is not None and training.get("reward") is not None:
        reward.load_state_dict(training["reward"])
    values = json.loads((directory / "trainer_state.json").read_text(encoding="utf-8"))
    state = TrainerState(**values)
    rank_path = directory / f"rank{accelerator.process_index:02d}_rng.pt"
    try:
        rng = torch.load(rank_path, map_location="cpu", weights_only=True)
    except TypeError:
        rng = torch.load(rank_path, map_location="cpu")
    torch.set_rng_state(rng["torch_rng"])
    if torch.cuda.is_available() and rng["cuda_rng"]:
        torch.cuda.set_rng_state_all(rng["cuda_rng"])
    if generator is not None and rng.get("rollout_generator") is not None:
        generator.set_state(rng["rollout_generator"])
    accelerator.wait_for_everyone()
    return directory, state


__all__ = [
    "PolicyEMA",
    "TrainerState",
    "distributed_run_dir",
    "load_preference_checkpoint",
    "save_preference_checkpoint",
    "seed_everything",
    "timestamped_run_dir",
]
