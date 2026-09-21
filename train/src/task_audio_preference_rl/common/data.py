from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset, Sampler


@dataclass(frozen=True)
class PromptRecord:
    index: int
    item_id: str
    caption: str
    audio_path: str


class AudioCapsPromptDataset(Dataset[PromptRecord]):
    def __init__(self, manifest: str | Path, root: str | Path) -> None:
        manifest_path = Path(manifest).expanduser().resolve()
        root_path = Path(root).expanduser().resolve()
        forbidden = _forbidden_video_ids()
        records: list[PromptRecord] = []
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            for source_index, row in enumerate(csv.DictReader(handle)):
                youtube_id = str(row.get("youtube_id") or "").strip()
                caption = str(row.get("caption") or "").strip()
                relative = str(row.get("audio_path") or "").strip()
                if not youtube_id or not caption or not relative:
                    raise ValueError(f"invalid AudioCaps row {source_index}")
                if youtube_id in forbidden:
                    continue
                path = (root_path / relative).resolve()
                records.append(
                    PromptRecord(
                        index=len(records),
                        item_id=f"train_{source_index:06d}",
                        caption=caption,
                        audio_path=str(path),
                    )
                )
        if not records:
            raise ValueError("AudioCaps preference dataset is empty")
        self.records = tuple(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> PromptRecord:
        return self.records[int(index)]


class ExactDistributedEpochSampler(Sampler[int]):
    """Deterministic disjoint shards; no repeated padding samples."""

    def __init__(
        self,
        data_source: Dataset[Any],
        *,
        rank: int,
        world_size: int,
        seed: int,
    ) -> None:
        self.data_source = data_source
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.epoch = 0
        self.start_index = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_start_index(self, index: int) -> None:
        self.start_index = int(index)

    def __len__(self) -> int:
        size = len(self.data_source)
        return max(0, (size - self.rank + self.world_size - 1) // self.world_size)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        permutation = torch.randperm(len(self.data_source), generator=generator)
        shard = permutation[self.rank :: self.world_size].tolist()
        yield from shard[self.start_index :]


def collate_prompt_records(rows: list[PromptRecord]) -> dict[str, Any]:
    return {
        "indices": torch.tensor([row.index for row in rows], dtype=torch.long),
        "item_ids": [row.item_id for row in rows],
        "captions": [row.caption for row in rows],
        "audio_paths": [row.audio_path for row in rows],
    }


def prompt_updates_per_epoch(dataset_size: int, global_prompts_per_update: int) -> int:
    return math.ceil(int(dataset_size) / int(global_prompts_per_update))


def _forbidden_video_ids() -> frozenset[str]:
    from task_audio_pmf.proven.data import load_forbidden_eval_video_ids

    values = set(load_forbidden_eval_video_ids())
    # AudioDEAR uses largely the same AudioCaps test split, but keep its local
    # protocol isolated if present.
    root = Path(__file__).resolve().parents[2]
    for candidate in (
        root / "evaluation" / "audiocaps_audiodear_964" / "protocol.jsonl",
        root / "evaluation" / "audiocaps_audiodear" / "protocol.jsonl",
    ):
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                value = str(row.get("youtube_id") or "").strip()
                if value:
                    values.add(value)
    return frozenset(values)


__all__ = [
    "AudioCapsPromptDataset",
    "ExactDistributedEpochSampler",
    "PromptRecord",
    "collate_prompt_records",
    "prompt_updates_per_epoch",
]
