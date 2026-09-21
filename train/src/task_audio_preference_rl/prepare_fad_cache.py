from __future__ import annotations

import json
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .common.data import (
    AudioCapsPromptDataset,
    ExactDistributedEpochSampler,
    collate_prompt_records,
)
from .common.fad_reward import (
    FAD_CACHE_VERSION,
    VGGishEmbeddingExtractor,
    frechet_distance,
    statistics_from_sums,
)
from .common.policy import XPredRolloutAdapter, load_policy_bundle
from .common.runtime import seed_everything
from .config import ExperimentConfig


def _load_audio_batch(paths: list[str], device: torch.device) -> torch.Tensor:
    import torchaudio
    import torchaudio.functional as AF

    rows: list[torch.Tensor] = []
    for value in paths:
        wave, sample_rate = torchaudio.load(value)
        wave = wave[0].float()
        if sample_rate != 16_000:
            wave = AF.resample(wave, sample_rate, 16_000)
        if wave.numel() < 160_000:
            wave = F.pad(wave, (0, 160_000 - wave.numel()))
        rows.append(wave[:160_000])
    return torch.stack(rows).to(device, non_blocking=True)


def _accumulate(
    embeddings: torch.Tensor,
    count: torch.Tensor,
    feature_sum: torch.Tensor,
    feature_outer: torch.Tensor,
) -> None:
    values = embeddings.reshape(-1, embeddings.shape[-1]).double()
    count += values.shape[0]
    feature_sum += values.sum(dim=0)
    feature_outer += values.T @ values


def prepare_fad_cache(config: ExperimentConfig, batch_size: int = 16) -> Path:
    """Build train-only VGGish statistics and a fixed 240k baseline queue."""

    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    generator = seed_everything(config.seed + 71_003, accelerator.process_index)
    dataset = AudioCapsPromptDataset(
        config.paths.audiocaps_manifest, config.paths.audiocaps_root
    )
    sampler = ExactDistributedEpochSampler(
        dataset,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        seed=config.seed + 81_019,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_prompt_records,
    )
    extractor = VGGishEmbeddingExtractor(
        config.paths.vggish_hub_dir,
        device,
        batch_size=config.reward.fad_embedding_batch_size,
    )
    target = Path(config.paths.vggish_fad_cache).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    real_shard = target.parent / (
        f".{target.name}.real-rank{accelerator.process_index:02d}.pt"
    )
    count = torch.zeros((), dtype=torch.float64, device=device)
    feature_sum = torch.zeros(128, dtype=torch.float64, device=device)
    feature_outer = torch.zeros(128, 128, dtype=torch.float64, device=device)
    if real_shard.is_file():
        try:
            cached_real = torch.load(real_shard, map_location="cpu", weights_only=True)
        except TypeError:
            cached_real = torch.load(real_shard, map_location="cpu")
        count.copy_(torch.as_tensor(cached_real["count"], device=device))
        feature_sum.copy_(torch.as_tensor(cached_real["sum"], device=device))
        feature_outer.copy_(torch.as_tensor(cached_real["outer"], device=device))
        print(
            f"[FAD-CACHE rank={accelerator.process_index}] reused real shard "
            f"with patches={int(count.item())}",
            flush=True,
        )
    else:
        for step, batch in enumerate(loader, start=1):
            embeddings = extractor(_load_audio_batch(batch["audio_paths"], device))
            _accumulate(embeddings, count, feature_sum, feature_outer)
            if step % 100 == 0:
                print(
                    f"[FAD-CACHE rank={accelerator.process_index}] "
                    f"real_batches={step}/{len(loader)} patches={int(count.item())}",
                    flush=True,
                )
        torch.save(
            {
                "count": count.cpu(),
                "sum": feature_sum.cpu(),
                "outer": feature_outer.cpu(),
            },
            real_shard,
        )
    if accelerator.num_processes > 1:
        torch.distributed.all_reduce(count)
        torch.distributed.all_reduce(feature_sum)
        torch.distributed.all_reduce(feature_outer)

    # Calibrate against exactly the distribution used by GRPO rollouts, using
    # only fixed AudioCaps-train prompts and fixed seeds.  This avoids using any
    # official AudioCaps test audio or metric in the training objective.
    bundle = load_policy_bundle(
        config.paths.source_resolved_config,
        config.paths.source_checkpoint,
        device=device,
    )
    policy = accelerator.prepare(bundle.policy)
    adapter = XPredRolloutAdapter(bundle.model, policy)
    prompt_loader = DataLoader(
        dataset,
        batch_size=config.rollout.prompts_per_rank,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_prompt_records,
    )
    local_audio_target = (
        config.reward.fad_queue_capacity + accelerator.num_processes - 1
    ) // accelerator.num_processes
    local_rows: list[torch.Tensor] = []
    local_count = 0
    for step, batch in enumerate(prompt_loader, start=1):
        # Match the formal trainer exactly: the prepared policy owns its
        # autocast wrapper, while decoder/conditioning stay outside autocast.
        with torch.inference_mode():
            trajectory, waveform, conditioning = adapter.rollout(
                batch["captions"], config.rollout, generator=generator
            )
            if not bool(torch.isfinite(waveform).all().item()):
                raise FloatingPointError(
                    "source baseline rollout produced non-finite waveform"
                )
            embeddings = extractor(waveform).cpu()
        needed = local_audio_target - local_count
        local_rows.append(embeddings[:needed])
        local_count += min(int(embeddings.shape[0]), needed)
        del trajectory, waveform, conditioning, embeddings
        if step % 20 == 0:
            print(
                f"[FAD-CACHE rank={accelerator.process_index}] "
                f"baseline_audios={local_count}/{local_audio_target}",
                flush=True,
            )
        if local_count >= local_audio_target:
            break
    local_baseline = torch.cat(local_rows, dim=0)[:local_audio_target]

    shard = target.parent / f".{target.name}.rank{accelerator.process_index:02d}.pt"
    torch.save(local_baseline, shard)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        shards = [
            target.parent / f".{target.name}.rank{rank:02d}.pt"
            for rank in range(accelerator.num_processes)
        ]
        parts = []
        for path in shards:
            try:
                parts.append(torch.load(path, map_location="cpu", weights_only=True))
            except TypeError:
                parts.append(torch.load(path, map_location="cpu"))
        baseline = torch.cat(parts, dim=0)[: config.reward.fad_queue_capacity]
        flat = baseline.reshape(-1, 128).double()
        baseline_mean = flat.mean(dim=0)
        centered = flat - baseline_mean
        baseline_covariance = centered.T @ centered / (flat.shape[0] - 1)
        reference_mean, reference_covariance = statistics_from_sums(
            int(count.item()), feature_sum.cpu(), feature_outer.cpu()
        )
        baseline_fad = float(
            frechet_distance(
                baseline_mean,
                baseline_covariance,
                reference_mean,
                reference_covariance,
            )
            .clamp_min(0)
            .item()
        )
        payload = {
            "version": FAD_CACHE_VERSION,
            "protocol": "AudioLDM VGGish 16kHz 128d no-PCA no-activation",
            "split": "AudioCaps train after official-test exclusion",
            "reference_count": int(count.item()),
            "reference_sum": feature_sum.cpu(),
            "reference_outer": feature_outer.cpu(),
            "baseline_audio_embeddings": baseline,
            "baseline_fad": baseline_fad,
            "provenance": {
                "source_checkpoint": str(
                    Path(config.paths.source_checkpoint).expanduser().resolve()
                ),
                "source_resolved_config": str(
                    Path(config.paths.source_resolved_config).expanduser().resolve()
                ),
                "audiocaps_manifest": str(
                    Path(config.paths.audiocaps_manifest).expanduser().resolve()
                ),
                "dataset_rows": len(dataset),
                "seed": config.seed,
                "queue_capacity": config.reward.fad_queue_capacity,
                "rollout": config.to_dict()["rollout"],
            },
        }
        torch.save(payload, target)
        metadata = dict(payload["provenance"])
        metadata.update(
            {
                "cache": str(target),
                "reference_patches": int(count.item()),
                "baseline_audios": int(baseline.shape[0]),
                "patches_per_audio": int(baseline.shape[1]),
                "baseline_train_proxy_fad": baseline_fad,
                "protocol": payload["protocol"],
                "split": payload["split"],
            }
        )
        target.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        for path in shards:
            path.unlink()
        for rank in range(accelerator.num_processes):
            path = target.parent / f".{target.name}.real-rank{rank:02d}.pt"
            if path.exists():
                path.unlink()
        print(
            f"[FAD-CACHE] saved {target}; train baseline FAD={baseline_fad:.6f}",
            flush=True,
        )
    accelerator.wait_for_everyone()
    return target


__all__ = ["prepare_fad_cache"]
