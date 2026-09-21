from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from ..common.data import (
    AudioCapsPromptDataset,
    ExactDistributedEpochSampler,
    collate_prompt_records,
    prompt_updates_per_epoch,
)
from ..common.policy import XPredRolloutAdapter, load_policy_bundle
from ..common.rewards import BalancedAudioReward
from ..common.trajectory import cps_step
from ..common.runtime import (
    PolicyEMA,
    TrainerState,
    distributed_run_dir,
    load_preference_checkpoint,
    save_preference_checkpoint,
    seed_everything,
)
from ..config import ExperimentConfig


def train_flow_grpo(
    config: ExperimentConfig,
    output_dir: str | None = None,
    resume_from: str | None = None,
) -> Path:
    cfg = config.grpo
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    if accelerator.num_processes != 2:
        raise RuntimeError(
            f"formal Flow-GRPO configuration requires exactly 2 GPUs, got {accelerator.num_processes}"
        )
    if config.rollout.samples_per_rank != 16:
        raise RuntimeError(
            "formal rollout batch must be 16 samples/rank (global batch 32)"
        )
    device = accelerator.device
    generator = seed_everything(config.seed, accelerator.process_index)
    if output_dir:
        run_dir = Path(output_dir).expanduser().resolve()
    elif resume_from:
        from task_audio.training.checkpointing import resolve_checkpoint_dir

        run_dir = resolve_checkpoint_dir(resume_from).parent.parent
    else:
        run_dir = distributed_run_dir(
            accelerator, config.paths.output_root, "flow_grpo"
        )
    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(
            json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
    accelerator.wait_for_everyone()

    dataset = AudioCapsPromptDataset(
        config.paths.audiocaps_manifest, config.paths.audiocaps_root
    )
    sampler = ExactDistributedEpochSampler(
        dataset,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        seed=config.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.rollout.prompts_per_rank,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_prompt_records,
    )
    bundle = load_policy_bundle(
        config.paths.source_resolved_config,
        config.paths.source_checkpoint,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        bundle.policy.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    policy, optimizer = accelerator.prepare(bundle.policy, optimizer)
    adapter = XPredRolloutAdapter(bundle.model, policy)
    reward = BalancedAudioReward(
        config.reward,
        config.paths,
        device=device,
        group_size=config.rollout.group_size,
    )
    ema = PolicyEMA(accelerator.unwrap_model(policy))
    state = TrainerState()
    if resume_from:
        resumed, state = load_preference_checkpoint(
            resume_from,
            accelerator=accelerator,
            base_model=bundle.model,
            optimizer=optimizer,
            ema=ema,
            generator=generator,
            reward=reward,
        )
        if accelerator.is_main_process:
            print(
                f"[GRPO] resumed from {resumed}: epoch={state.epoch} "
                f"update={state.update} rank_prompt_offset={state.prompts_consumed_on_rank}",
                flush=True,
            )
    updates_per_epoch = prompt_updates_per_epoch(
        len(dataset),
        config.rollout.prompts_per_rank * accelerator.num_processes,
    )
    total_updates = updates_per_epoch * cfg.prompt_epochs
    started = time.monotonic()

    start_epoch = state.epoch
    for epoch in range(start_epoch, cfg.prompt_epochs):
        sampler.set_epoch(epoch)
        sampler.set_start_index(
            state.prompts_consumed_on_rank if epoch == start_epoch else 0
        )
        state.epoch = epoch
        if epoch != start_epoch:
            state.prompts_consumed_on_rank = 0
        for batch in loader:
            update_started = time.monotonic()
            unique_captions = batch["captions"]
            repeated_captions = [
                caption
                for caption in unique_captions
                for _ in range(config.rollout.group_size)
            ]
            repeated_ids = [
                item
                for item in batch["item_ids"]
                for _ in range(config.rollout.group_size)
            ]
            with torch.no_grad():
                trajectory, waveform, conditioning = adapter.rollout(
                    unique_captions, config.rollout, generator=generator
                )
                rollout_finished = time.monotonic()
                scores = reward(waveform, repeated_captions, repeated_ids)
                advantage = scores["advantage"].clamp(
                    -cfg.advantage_clip, cfg.advantage_clip
                )
                reward_finished = time.monotonic()

            optimizer.zero_grad(set_to_none=True)
            # Every rollout contributes exactly ``window_size`` stochastic
            # transitions. Flatten them into one batch so variable per-prompt
            # window starts do not create up to seven serial transformer passes.
            coordinates = trajectory.stochastic_mask.nonzero(as_tuple=False)
            rows, steps = coordinates[:, 0], coordinates[:, 1]
            flat_state = trajectory.states[rows, steps]
            flat_next = trajectory.next_states[rows, steps]
            flat_time = trajectory.times[rows, steps]
            flat_next_time = trajectory.next_times[rows, steps]
            flat_conditioning = conditioning.index_select(rows)
            velocity = policy(flat_state, flat_time, flat_conditioning)
            transition = cps_step(
                flat_state,
                velocity,
                flat_time,
                flat_next_time,
                noise_level=config.rollout.noise_level,
                next_state=flat_next,
            )
            with torch.no_grad():
                reference_velocity = bundle.reference(
                    flat_state, flat_time, flat_conditioning
                )
                reference_transition = cps_step(
                    flat_state,
                    reference_velocity,
                    flat_time,
                    flat_next_time,
                    noise_level=config.rollout.noise_level,
                    next_state=flat_next,
                )
            # One PPO epoch: the old policy is exactly the pre-update policy.
            # Detaching this vectorized evaluation also avoids harmless GEMM
            # roundoff from comparing rollout batch 16 with transition batch 64.
            old_log_prob = transition.log_prob.detach()
            log_ratio = (transition.log_prob - old_log_prob).clamp(-10.0, 10.0)
            ratio = log_ratio.exp()
            ratio_error = (ratio - 1.0).abs().max()
            flat_advantage = advantage.index_select(0, rows)
            unclipped = -flat_advantage * ratio
            clipped = -flat_advantage * ratio.clamp(
                1.0 - cfg.ppo_clip_range, 1.0 + cfg.ppo_clip_range
            )
            policy_loss = torch.maximum(unclipped, clipped).mean()
            kl_loss = (
                (transition.mean.float() - reference_transition.mean.float()).square()
                / (2.0 * transition.std.float().square().clamp_min(1.0e-8))
            ).mean()
            loss = policy_loss + cfg.reference_kl_coefficient * kl_loss
            accelerator.backward(loss)

            if state.update == 0 and float(ratio_error.item()) > 1.0e-4:
                raise RuntimeError(
                    f"on-policy ratio sanity check failed: max |ratio-1|={ratio_error.item():.6g}"
                )
            accelerator.clip_grad_norm_(policy.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            ema.update(accelerator.unwrap_model(policy))
            update_finished = time.monotonic()
            state.update += 1
            state.prompts_consumed_on_rank += len(unique_captions)

            if state.update <= 3 or state.update % cfg.log_every_updates == 0:
                elapsed = time.monotonic() - started
                eta = (
                    elapsed / state.update * (total_updates - state.update)
                    if state.update >= cfg.eta_warmup_updates
                    else float("nan")
                )
                reduced = {
                    key: accelerator.reduce(value.float(), reduction="mean").item()
                    for key, value in {
                        "loss": loss.detach(),
                        "policy": policy_loss.detach(),
                        "kl": kl_loss.detach(),
                        "clap": scores["clap"].mean(),
                        "passt_kl": scores["passt_kl"].mean(),
                        "repetition": scores["repetition"].mean(),
                        "fd_marginal": scores["fd_marginal"].mean(),
                        "fd_proxy": scores["fd_proxy"].mean(),
                        "fd_target": scores["fd_target"].mean(),
                        "fd_lambda": scores["fd_lambda"].mean(),
                        "fd_violation": scores["fd_violation"].mean(),
                        "fad_marginal": scores["fad_marginal"].mean(),
                        "fad_proxy": scores["fad_proxy"].mean(),
                        "fad_target": scores["fad_target"].mean(),
                        "fad_lambda": scores["fad_lambda"].mean(),
                        "fad_violation": scores["fad_violation"].mean(),
                        "fad_active": scores["fad_active"].mean(),
                    }.items()
                }
                if accelerator.is_main_process:
                    print(
                        "[GRPO] "
                        f"epoch={epoch + 1}/{cfg.prompt_epochs} update={state.update}/{total_updates} "
                        + " ".join(
                            f"{key}={value:.6f}" for key, value in reduced.items()
                        )
                        + f" rollout_s={rollout_finished - update_started:.2f}"
                        + f" reward_s={reward_finished - rollout_finished:.2f}"
                        + f" train_s={update_finished - reward_finished:.2f}"
                        + f" eta={eta:.0f}s",
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
                    generator=generator,
                    reward=reward,
                )

        state.epoch = epoch + 1
        state.prompts_consumed_on_rank = 0
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
            generator=generator,
            reward=reward,
        )
    return run_dir


__all__ = ["train_flow_grpo"]
