from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class PairShardWriter:
    def __init__(self, root: Path, rank: int, shard_size: int = 256) -> None:
        self.root = root
        self.rank = int(rank)
        self.shard_size = int(shard_size)
        self.buffer: list[dict[str, Any]] = []
        self.shard_index = 0
        self.root.mkdir(parents=True, exist_ok=True)

    def add(self, row: dict[str, Any]) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        path = self.root / f"rank{self.rank:02d}_shard{self.shard_index:05d}.pt"
        torch.save(self.buffer, path)
        self.buffer = []
        self.shard_index += 1


class PreferencePairDataset(Dataset[dict[str, Any]]):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.index: list[tuple[Path, int]] = []
        for path in sorted(self.root.glob("rank*_shard*.pt")):
            rows = _load(path)
            self.index.extend((path, offset) for offset in range(len(rows)))
        if not self.index:
            raise ValueError(f"no preference pair shards found in {self.root}")
        self._cached_path: Path | None = None
        self._cached_rows: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, offset = self.index[int(index)]
        if path != self._cached_path:
            self._cached_rows = _load(path)
            self._cached_path = path
        return self._cached_rows[offset]


def _load(path: Path) -> list[dict[str, Any]]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, list):
        raise TypeError(f"invalid pair shard: {path}")
    return value


def collate_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "item_ids": [str(row["item_id"]) for row in rows],
        "captions": [str(row["caption"]) for row in rows],
        "chosen": torch.stack([row["chosen"].float() for row in rows]),
        "rejected": torch.stack([row["rejected"].float() for row in rows]),
    }


__all__ = ["PairShardWriter", "PreferencePairDataset", "collate_pairs"]
