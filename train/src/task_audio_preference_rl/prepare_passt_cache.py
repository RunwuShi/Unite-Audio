from __future__ import annotations

import os
from contextlib import redirect_stdout
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
from .common.rewards import _peak_normalize
from .config import ExperimentConfig


def _load_audio_batch(paths: list[str], device: torch.device) -> torch.Tensor:
    import torchaudio
    import torchaudio.functional as AF

    rows: list[torch.Tensor] = []
    for value in paths:
        wave, sample_rate = torchaudio.load(value)
        wave = wave.mean(dim=0)
        if sample_rate != 16_000:
            wave = AF.resample(wave, sample_rate, 16_000)
        if wave.numel() < 160_000:
            wave = F.pad(wave, (0, 160_000 - wave.numel()))
        rows.append(wave[:160_000])
    return torch.stack(rows).to(device)


def prepare_passt_cache(config: ExperimentConfig, batch_size: int = 32) -> Path:
    import torchaudio.functional as AF
    from evaluation.run_passt_batched import PatchPasstStft
    from hear21passt.base import get_basic_model

    accelerator = Accelerator(mixed_precision="fp16")
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
        batch_size=int(batch_size),
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_prompt_records,
    )
    with open(os.devnull, "w") as sink, redirect_stdout(sink):
        model = get_basic_model(mode="logits").to(accelerator.device).eval()
    local: dict[str, torch.Tensor] = {}
    for step, batch in enumerate(loader, start=1):
        wave16 = _load_audio_batch(batch["audio_paths"], accelerator.device)
        wave32 = AF.resample(_peak_normalize(wave16), 16_000, 32_000)
        with torch.inference_mode(), PatchPasstStft():
            logits = model(wave32)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        probabilities = torch.softmax(torch.as_tensor(logits).float(), dim=-1).cpu()
        local.update(zip(batch["item_ids"], probabilities.unbind(0)))
        if step % 100 == 0:
            print(
                f"[PASST-CACHE rank={accelerator.process_index}] batches={step}/{len(loader)} items={len(local)}",
                flush=True,
            )

    target = Path(config.paths.passt_reference_cache).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    shard = target.parent / f".{target.name}.rank{accelerator.process_index:02d}.pt"
    torch.save(local, shard)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        merged: dict[str, torch.Tensor] = {}
        shards = [
            target.parent / f".{target.name}.rank{rank:02d}.pt"
            for rank in range(accelerator.num_processes)
        ]
        for path in shards:
            try:
                values = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                values = torch.load(path, map_location="cpu")
            overlap = set(merged).intersection(values)
            if overlap:
                raise RuntimeError(f"duplicate AudioCaps cache IDs: {sorted(overlap)[:8]}")
            merged.update(values)
        if len(merged) != len(dataset):
            raise RuntimeError(
                f"PaSST cache incomplete: cached={len(merged)} expected={len(dataset)}"
            )
        torch.save(merged, target)
        for path in shards:
            path.unlink()
        print(f"[PASST-CACHE] saved {len(merged)} rows to {target}", flush=True)
    accelerator.wait_for_everyone()
    return target


__all__ = ["prepare_passt_cache"]
