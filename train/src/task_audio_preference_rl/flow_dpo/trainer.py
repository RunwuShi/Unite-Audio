from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..common.policy import load_policy_bundle
from ..common.runtime import (
    PolicyEMA,
    TrainerState,
    distributed_run_dir,
    save_preference_checkpoint,
    seed_everything,
)
from ..config import ExperimentConfig
from .pairs import PreferencePairDataset, collate_pairs


def _x0_loss(
    policy: torch.nn.Module,
    target: torch.Tensor,
    time_value: torch.Tensor,
    noise: torch.Tensor,
    conditioning: object,
) -> torch.Tensor:
    batch = target.shape[0]
    time_shape = (batch,) + (1,) * (target.ndim - 1)
    noisy = time_value.view(time_shape) * target + (
        1.0 - time_value.view(time_shape)
    ) * noise
    velocity = policy(noisy, time_value, conditioning)
    prediction = noisy + (1.0 - time_value).view(time_shape) * velocity
    return (prediction.float() - target.float()).square().mean(dim=(1, 2))


def train_flow_dpo(
    config: ExperimentConfig,
    preference_dir: str,
    output_dir: str | None = None,
) -> Path:
    cfg = config.dpo
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    if accelerator.num_processes != 2:
        raise RuntimeError("formal Flow-DPO configuration requires exactly 2 GPUs")
    seed_everything(config.seed + 131, accelerator.process_index)
    run_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else distributed_run_dir(accelerator, config.paths.output_root, "flow_dpo")
    )
    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(
            json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
    accelerator.wait_for_everyone()

    dataset = PreferencePairDataset(preference_dir)
    sampler = DistributedSampler(
        dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        seed=config.seed + 131,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.pairs_per_rank_batch,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_pairs,
        drop_last=False,
    )
    bundle = load_policy_bundle(
        config.paths.source_resolved_config,
        config.paths.source_checkpoint,
        device=accelerator.device,
    )
    optimizer = torch.optim.AdamW(
        bundle.policy.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    policy, optimizer = accelerator.prepare(bundle.policy, optimizer)
    ema = PolicyEMA(accelerator.unwrap_model(policy))
    state = TrainerState()
    total_updates = len(loader) * cfg.prompt_epochs
    started = time.monotonic()
    for epoch in range(cfg.prompt_epochs):
        sampler.set_epoch(epoch)
        state.epoch = epoch
        for batch in loader:
            captions = batch["captions"]
            chosen = batch["chosen"].to(accelerator.device)
            rejected = batch["rejected"].to(accelerator.device)
            # Concatenating keeps one DDP forward graph per policy evaluation.
            targets = torch.cat([chosen, rejected], dim=0)
            doubled_captions = captions + captions
            from ..common.policy import XPredRolloutAdapter

            adapter = XPredRolloutAdapter(bundle.model, policy)
            conditioning, _ = adapter.prepare_conditioning(
                doubled_captions, seconds=config.rollout.seconds
            )
            time_value = torch.rand(
                targets.shape[0], device=targets.device, dtype=torch.float32
            )
            noise = torch.randn_like(targets)
            online = _x0_loss(
                policy, targets, time_value, noise, conditioning
            )
            with torch.no_grad():
                reference = _x0_loss(
                    bundle.reference, targets, time_value, noise, conditioning
                )
            chosen_loss, rejected_loss = online.chunk(2)
            reference_chosen, reference_rejected = reference.chunk(2)
            inside = -0.5 * cfg.beta * (
                (chosen_loss - rejected_loss)
                - (reference_chosen - reference_rejected)
            )
            preference_loss = -F.logsigmoid(inside).mean()
            anchor_loss = chosen_loss.mean()
            loss = preference_loss + cfg.chosen_anchor_weight * anchor_loss
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            ema.update(accelerator.unwrap_model(policy))
            state.update += 1
            if state.update % cfg.log_every_updates == 0:
                reduced = accelerator.reduce(
                    torch.stack(
                        [loss.detach(), preference_loss.detach(), anchor_loss.detach()]
                    ),
                    reduction="mean",
                )
                if accelerator.is_main_process:
                    eta = (time.monotonic() - started) / state.update * (
                        total_updates - state.update
                    )
                    print(
                        f"[DPO] epoch={epoch + 1}/{cfg.prompt_epochs} update={state.update}/{total_updates} "
                        f"loss={reduced[0].item():.6f} pref={reduced[1].item():.6f} "
                        f"anchor={reduced[2].item():.6f} eta={eta:.0f}s",
                        flush=True,
                    )
            if state.update % cfg.save_every_updates == 0:
                save_preference_checkpoint(
                    run_dir=run_dir,
                    label=f"checkpoint-{state.update:08d}",
                    accelerator=accelerator,
                    base_model=bundle.model,
                    policy=policy,
                    optimizer=optimizer,
                    ema=ema,
                    trainer_state=state,
                    config=config.to_dict(),
                )
        save_preference_checkpoint(
            run_dir=run_dir,
            label=f"checkpoint-epoch-{epoch + 1:02d}",
            accelerator=accelerator,
            base_model=bundle.model,
            policy=policy,
            optimizer=optimizer,
            ema=ema,
            trainer_state=state,
            config=config.to_dict(),
        )
    return run_dir


__all__ = ["train_flow_dpo"]
