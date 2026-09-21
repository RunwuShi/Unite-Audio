from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch


class ReplayMixtureDataset:
    """Map-style primary/replay mixture with deterministic epoch rotation."""

    def __init__(
        self,
        primary: Any,
        replay: Any,
        *,
        replay_fraction: float,
        seed: int = 1234,
    ) -> None:
        self.primary = primary
        self.replay = replay
        self.replay_fraction = float(replay_fraction)
        self.seed = int(seed)
        self.primary_count = int(len(primary))
        self.replay_pool_count = int(len(replay))
        if self.primary_count < 1 or self.replay_pool_count < 1:
            raise ValueError("mixed dataset components must be non-empty")
        if not 0.0 < self.replay_fraction < 1.0:
            raise ValueError("data.mix.replay_fraction must be in (0, 1)")
        self.replay_count = max(
            1,
            int(
                round(
                    self.primary_count
                    * self.replay_fraction
                    / (1.0 - self.replay_fraction)
                )
            ),
        )
        if self.replay_count > self.replay_pool_count:
            raise ValueError(
                "one mixed epoch requests more replay samples than the replay pool: "
                f"{self.replay_count} > {self.replay_pool_count}"
            )
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self._replay_stride = _coprime_stride(self.replay_pool_count, self.seed)
        self.stats = {
            "primary_samples_per_epoch": self.primary_count,
            "replay_samples_per_epoch": self.replay_count,
            "replay_pool_samples": self.replay_pool_count,
            "replay_fraction_requested": self.replay_fraction,
            "replay_fraction_actual": self.replay_count / len(self),
        }

    def __len__(self) -> int:
        return self.primary_count + self.replay_count

    @property
    def epoch(self) -> int:
        return int(self._epoch.item())

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("mixed dataset epoch must be non-negative")
        self._epoch.fill_(epoch)

    def replay_index(self, slot: int, *, epoch: int | None = None) -> int:
        slot = int(slot)
        if not 0 <= slot < self.replay_count:
            raise IndexError(f"replay slot out of range: {slot}")
        active_epoch = self.epoch if epoch is None else int(epoch)
        sequence_index = active_epoch * self.replay_count + slot
        return (
            self.seed + sequence_index * self._replay_stride
        ) % self.replay_pool_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index < self.primary_count:
            role = "primary"
            sample = self.primary[index]
        else:
            role = "replay"
            sample = self.replay[self.replay_index(index - self.primary_count)]
        if not isinstance(sample, Mapping):
            raise TypeError("mixed dataset components must return mappings")
        output = dict(sample)
        metadata = dict(output.get("metadata") or {})
        metadata["mixture_role"] = role
        metadata["mixture_epoch"] = self.epoch
        output["metadata"] = metadata
        output["mixture_role"] = role
        return output


def _coprime_stride(size: int, seed: int) -> int:
    if size < 2:
        return 1
    candidate = 2 * (abs(int(seed)) % max(size // 2, 1)) + 1
    while math.gcd(candidate, size) != 1:
        candidate += 2
        if candidate >= size:
            candidate = 1
    return candidate


__all__ = ["ReplayMixtureDataset"]
