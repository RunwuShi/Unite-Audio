from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from ..common.data import (
    AudioCapsPromptDataset,
    ExactDistributedEpochSampler,
    collate_prompt_records,
)
from ..common.policy import XPredRolloutAdapter, load_policy_bundle
from ..common.rewards import BalancedAudioReward
from ..common.runtime import distributed_run_dir, seed_everything
from ..config import ExperimentConfig
from .pairs import PairShardWriter


def collect_preferences(
    config: ExperimentConfig, output_dir: str | None = None
) -> Path:
    accelerator = Accelerator(mixed_precision=config.dpo.mixed_precision)
    if accelerator.num_processes != 2:
        raise RuntimeError("formal DPO collection requires exactly 2 GPUs")
    device = accelerator.device
    generator = seed_everything(config.seed + 71, accelerator.process_index)
    root = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else distributed_run_dir(
            accelerator, config.paths.output_root, "flow_dpo_pairs"
        )
    )
    if accelerator.is_main_process:
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text(
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
        seed=config.seed + 71,
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
    adapter = XPredRolloutAdapter(bundle.model, bundle.policy)
    reward = BalancedAudioReward(
        config.reward,
        config.paths,
        device=device,
        group_size=config.rollout.group_size,
    )
    writer = PairShardWriter(root, accelerator.process_index)
    # DPO gets diversity from independent initial noise; the trajectory itself
    # remains deterministic so pair labels do not include CPS transition noise.
    rollout = replace(config.rollout, stochastic_type="ode")
    kept = 0
    seen = 0
    for epoch in range(config.dpo.collection_prompt_epochs):
        sampler.set_epoch(epoch)
        for batch in loader:
            captions = batch["captions"]
            repeated_captions = [
                caption for caption in captions for _ in range(rollout.group_size)
            ]
            repeated_ids = [
                item for item in batch["item_ids"] for _ in range(rollout.group_size)
            ]
            with torch.inference_mode():
                trajectory, waveform, _ = adapter.rollout(
                    captions, rollout, generator=generator
                )
                scores = reward(waveform, repeated_captions, repeated_ids)
            for prompt_offset, (item_id, caption) in enumerate(
                zip(batch["item_ids"], captions)
            ):
                begin = prompt_offset * rollout.group_size
                end = begin + rollout.group_size
                local_advantage = scores["advantage"][begin:end]
                order = local_advantage.argsort(descending=True)
                chosen_index = begin + int(order[0].item())
                rejected_index = begin + int(order[-1].item())
                gap = float(
                    (scores["advantage"][chosen_index] - scores["advantage"][rejected_index]).item()
                )
                seen += 1
                if gap < config.dpo.minimum_preference_gap:
                    continue
                writer.add(
                    {
                        "item_id": str(item_id),
                        "caption": str(caption),
                        "chosen": trajectory.final_latents[chosen_index].half().cpu(),
                        "rejected": trajectory.final_latents[rejected_index].half().cpu(),
                        "gap": gap,
                        "chosen_clap": float(scores["clap"][chosen_index].item()),
                        "rejected_clap": float(scores["clap"][rejected_index].item()),
                        "chosen_passt_kl": float(scores["passt_kl"][chosen_index].item()),
                        "rejected_passt_kl": float(scores["passt_kl"][rejected_index].item()),
                        "chosen_repetition": float(scores["repetition"][chosen_index].item()),
                        "rejected_repetition": float(scores["repetition"][rejected_index].item()),
                    }
                )
                kept += 1
            if seen % 400 == 0:
                print(
                    f"[DPO-COLLECT rank={accelerator.process_index}] epoch={epoch + 1} seen={seen} kept={kept}",
                    flush=True,
                )
    writer.flush()
    accelerator.wait_for_everyone()
    counts = accelerator.gather(
        torch.tensor([seen, kept], device=device, dtype=torch.long)
    ).view(-1, 2)
    if accelerator.is_main_process:
        summary = {
            "seen": int(counts[:, 0].sum().item()),
            "kept": int(counts[:, 1].sum().item()),
            "collection_prompt_epochs": config.dpo.collection_prompt_epochs,
        }
        (root / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[DPO-COLLECT] complete: {summary} path={root}", flush=True)
    return root


__all__ = ["collect_preferences"]
