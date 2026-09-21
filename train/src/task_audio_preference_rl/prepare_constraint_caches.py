from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
from .common.fd_reward import (
    FD_CACHE_VERSION,
    PANNsEmbeddingExtractor,
    diagonal_frechet_distance,
    diagonal_statistics_from_sums,
)
from .common.policy import XPredRolloutAdapter, load_policy_bundle
from .common.runtime import seed_everything
from .config import ExperimentConfig


def _load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_audio_batch(paths: list[str], device: torch.device) -> torch.Tensor:
    import torchaudio
    import torchaudio.functional as AF

    rows: list[torch.Tensor] = []
    for value in paths:
        wave, sample_rate = torchaudio.load(value)
        wave = wave.mean(dim=0).float()
        if sample_rate != 16_000:
            wave = AF.resample(wave, sample_rate, 16_000)
        if wave.numel() < 160_000:
            wave = F.pad(wave, (0, 160_000 - wave.numel()))
        rows.append(wave[:160_000])
    return torch.stack(rows).to(device, non_blocking=True)


def _prepare_panns_reference(
    config: ExperimentConfig,
    accelerator: Accelerator,
    dataset: AudioCapsPromptDataset,
    extractor: PANNsEmbeddingExtractor,
    *,
    batch_size: int,
) -> dict[str, Any]:
    target = Path(config.paths.panns_fd_reference_cache).expanduser().resolve()
    if target.is_file():
        cache = _load(target)
        if int(cache.get("version", -1)) != FD_CACHE_VERSION:
            raise ValueError(f"unsupported PANNs reference cache: {target}")
        return cache

    target.parent.mkdir(parents=True, exist_ok=True)
    sampler = ExactDistributedEpochSampler(
        dataset,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        seed=config.seed + 91_013,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_prompt_records,
    )
    shard = target.parent / f".{target.name}.real-rank{accelerator.process_index:02d}.pt"
    if shard.is_file():
        local = torch.as_tensor(_load(shard)["embeddings"]).float()
        print(
            f"[FD-CACHE rank={accelerator.process_index}] reused "
            f"reference embeddings={local.shape[0]}",
            flush=True,
        )
    else:
        rows: list[torch.Tensor] = []
        for step, batch in enumerate(loader, start=1):
            rows.append(extractor(_load_audio_batch(batch["audio_paths"], accelerator.device)).cpu())
            if step % 50 == 0:
                print(
                    f"[FD-CACHE rank={accelerator.process_index}] "
                    f"reference_batches={step}/{len(loader)}",
                    flush=True,
                )
        local = torch.cat(rows, dim=0)
        torch.save({"embeddings": local}, shard)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        shards = [
            target.parent / f".{target.name}.real-rank{rank:02d}.pt"
            for rank in range(accelerator.num_processes)
        ]
        real = torch.cat([torch.as_tensor(_load(path)["embeddings"]).float() for path in shards])
        if real.shape != (len(dataset), 2048):
            raise RuntimeError(
                f"PANNs reference shape {tuple(real.shape)} does not match "
                f"dataset size {len(dataset)}"
            )
        values = real.to(accelerator.device)
        center = values.mean(dim=0)
        torch.manual_seed(config.seed + 101_009)
        _, _, vectors = torch.pca_lowrank(
            values,
            q=config.reward.fd_projection_dim,
            center=True,
            niter=2,
        )
        components = vectors.T.contiguous()
        projected = (values - center) @ components.T
        reference_mean = projected.double().mean(dim=0)
        reference_variance = projected.double().var(dim=0, unbiased=True)
        payload = {
            "version": FD_CACHE_VERSION,
            "protocol": "AudioLDM PANNs CNN14 2048d, train-only PCA-diagonal FD",
            "split": "AudioCaps train after official-test exclusion",
            "reference_count": int(projected.shape[0]),
            "center": center.cpu(),
            "components": components.cpu(),
            "reference_mean": reference_mean.cpu(),
            "reference_variance": reference_variance.cpu(),
            "projection_dim": int(config.reward.fd_projection_dim),
            "audiocaps_manifest": str(
                Path(config.paths.audiocaps_manifest).expanduser().resolve()
            ),
        }
        torch.save(payload, target)
        target.with_suffix(".json").write_text(
            json.dumps(
                {
                    "cache": str(target),
                    "protocol": payload["protocol"],
                    "split": payload["split"],
                    "reference_audios": payload["reference_count"],
                    "projection_dim": payload["projection_dim"],
                    "audiocaps_manifest": payload["audiocaps_manifest"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        del values, projected, real
        for path in shards:
            path.unlink(missing_ok=True)
    accelerator.wait_for_everyone()
    return _load(target)


def prepare_constraint_caches(
    config: ExperimentConfig, batch_size: int = 32
) -> tuple[Path, Path]:
    """Build train-only PANNs-FD and source-specific VGGish-FAD caches."""

    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    generator = seed_everything(config.seed + 111_017, accelerator.process_index)
    dataset = AudioCapsPromptDataset(
        config.paths.audiocaps_manifest, config.paths.audiocaps_root
    )
    fd_target = Path(config.paths.panns_fd_cache).expanduser().resolve()
    fad_target = Path(config.paths.vggish_fad_cache).expanduser().resolve()
    if fd_target.is_file() and fad_target.is_file():
        if accelerator.is_main_process:
            print(f"[CONSTRAINT-CACHE] reusing {fd_target} and {fad_target}", flush=True)
        return fd_target, fad_target

    panns = PANNsEmbeddingExtractor(
        config.paths.panns_repo,
        config.paths.panns_checkpoint_dir,
        device,
        batch_size=config.reward.fd_embedding_batch_size,
    )
    reference = _prepare_panns_reference(
        config, accelerator, dataset, panns, batch_size=batch_size
    )
    center = torch.as_tensor(reference["center"]).float().to(device)
    components = torch.as_tensor(reference["components"]).float().to(device)
    vggish = VGGishEmbeddingExtractor(
        config.paths.vggish_hub_dir,
        device,
        batch_size=config.reward.fad_embedding_batch_size,
    )

    bundle = load_policy_bundle(
        config.paths.source_resolved_config,
        config.paths.source_checkpoint,
        device=device,
    )
    policy = accelerator.prepare(bundle.policy)
    adapter = XPredRolloutAdapter(bundle.model, policy)
    sampler = ExactDistributedEpochSampler(
        dataset,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        seed=config.seed + 121_021,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.rollout.prompts_per_rank,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_prompt_records,
    )
    queue_capacity = max(
        int(config.reward.fd_queue_capacity), int(config.reward.fad_queue_capacity)
    )
    local_target = (queue_capacity + accelerator.num_processes - 1) // accelerator.num_processes
    fd_rows: list[torch.Tensor] = []
    fad_rows: list[torch.Tensor] = []
    local_count = 0
    for step, batch in enumerate(loader, start=1):
        with torch.inference_mode():
            trajectory, waveform, conditioning = adapter.rollout(
                batch["captions"], config.rollout, generator=generator
            )
            raw_fd = panns(waveform)
            projected_fd = (raw_fd - center) @ components.T
            fad_embeddings = vggish(waveform)
        needed = local_target - local_count
        fd_rows.append(projected_fd[:needed].cpu())
        fad_rows.append(fad_embeddings[:needed].cpu())
        local_count += min(int(waveform.shape[0]), needed)
        del trajectory, waveform, conditioning, raw_fd, projected_fd, fad_embeddings
        if step % 20 == 0:
            print(
                f"[CONSTRAINT-CACHE rank={accelerator.process_index}] "
                f"baseline_audios={local_count}/{local_target}",
                flush=True,
            )
        if local_count >= local_target:
            break
    shard = fd_target.parent / f".constraint-baseline-rank{accelerator.process_index:02d}.pt"
    shard.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"fd": torch.cat(fd_rows)[:local_target], "fad": torch.cat(fad_rows)[:local_target]},
        shard,
    )
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        shards = [
            fd_target.parent / f".constraint-baseline-rank{rank:02d}.pt"
            for rank in range(accelerator.num_processes)
        ]
        loaded = [_load(path) for path in shards]
        baseline_fd = torch.cat([torch.as_tensor(row["fd"]).float() for row in loaded])[
            : config.reward.fd_queue_capacity
        ]
        baseline_fad = torch.cat(
            [torch.as_tensor(row["fad"]).float() for row in loaded]
        )[: config.reward.fad_queue_capacity]
        fd_values = baseline_fd.double()
        fd_mean, fd_variance = diagonal_statistics_from_sums(
            len(fd_values), fd_values.sum(0), fd_values.square().sum(0)
        )
        baseline_fd_value = float(
            diagonal_frechet_distance(
                fd_mean,
                fd_variance,
                torch.as_tensor(reference["reference_mean"]),
                torch.as_tensor(reference["reference_variance"]),
            ).item()
        )
        fd_payload = dict(reference)
        fd_payload.update(
            {
                "baseline_embeddings": baseline_fd,
                "baseline_fd": baseline_fd_value,
                "provenance": {
                    "source_checkpoint": str(
                        Path(config.paths.source_checkpoint).expanduser().resolve()
                    ),
                    "source_resolved_config": str(
                        Path(config.paths.source_resolved_config).expanduser().resolve()
                    ),
                    "seed": config.seed,
                    "queue_capacity": config.reward.fd_queue_capacity,
                    "rollout": config.to_dict()["rollout"],
                },
            }
        )
        fd_target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(fd_payload, fd_target)

        fad_reference = _load(
            Path(config.paths.vggish_reference_cache).expanduser().resolve()
        )
        reference_mean, reference_covariance = statistics_from_sums(
            int(fad_reference["reference_count"]),
            torch.as_tensor(fad_reference["reference_sum"]),
            torch.as_tensor(fad_reference["reference_outer"]),
        )
        flat_fad = baseline_fad.reshape(-1, 128).double()
        fad_mean = flat_fad.mean(dim=0)
        centered_fad = flat_fad - fad_mean
        fad_covariance = centered_fad.T @ centered_fad / (flat_fad.shape[0] - 1)
        baseline_fad_value = float(
            frechet_distance(
                fad_mean, fad_covariance, reference_mean, reference_covariance
            ).clamp_min(0).item()
        )
        fad_payload = {
            "version": FAD_CACHE_VERSION,
            "protocol": fad_reference["protocol"],
            "split": fad_reference["split"],
            "reference_count": int(fad_reference["reference_count"]),
            "reference_sum": torch.as_tensor(fad_reference["reference_sum"]),
            "reference_outer": torch.as_tensor(fad_reference["reference_outer"]),
            "baseline_audio_embeddings": baseline_fad,
            "baseline_fad": baseline_fad_value,
            "provenance": fd_payload["provenance"],
        }
        fad_target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(fad_payload, fad_target)
        metadata = {
            "source_checkpoint": fd_payload["provenance"]["source_checkpoint"],
            "fd_cache": str(fd_target),
            "fad_cache": str(fad_target),
            "reference_audios": int(reference["reference_count"]),
            "baseline_audios": int(baseline_fd.shape[0]),
            "baseline_fd_proxy": baseline_fd_value,
            "baseline_fad_proxy": baseline_fad_value,
            "fd_target_scale": config.reward.fd_target_scale,
            "fad_target_scale": config.reward.fad_target_scale,
        }
        fd_target.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        for path in shards:
            path.unlink(missing_ok=True)
        print(json.dumps(metadata, indent=2), flush=True)
    accelerator.wait_for_everyone()
    return fd_target, fad_target


__all__ = ["prepare_constraint_caches"]
