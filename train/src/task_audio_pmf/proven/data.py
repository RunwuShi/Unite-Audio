from __future__ import annotations

import math
import os
import random
import json
from collections.abc import Iterator, Mapping, Sized
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler

from task_audio.data import build_dataset
from task_audio.data.common import collate_audio_text


MEANAUDIO_957_PROTOCOL = (
    Path(__file__).resolve().parents[2]
    / "evaluation"
    / "audiocaps_meanaudio_957"
    / "protocol.jsonl"
)
TANGO_886_PROTOCOL = (
    Path(__file__).resolve().parents[2]
    / "evaluation"
    / "audiocaps_tango"
    / "protocol.jsonl"
)
WAVCAPS_DURATION_GROUPS = (
    ("wavcaps_0_5", 0.0, 5.0),
    ("wavcaps_5_10", 5.0, 10.0),
    ("wavcaps_10_15", 10.0, 15.0),
    ("wavcaps_15_20", 15.0, 20.0),
)
PROBE_GROUP_SECONDS = {
    "stage1_mixed": 20.0,
    "audioset_10": 10.0,
    "wavcaps_0_5": 5.0,
    "wavcaps_5_10": 10.0,
    "wavcaps_10_15": 15.0,
    "wavcaps_15_20": 20.0,
    "audiocaps_10_5": 10.5,
}


def load_meanaudio_video_ids() -> frozenset[str]:
    values: set[str] = set()
    with MEANAUDIO_957_PROTOCOL.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            value = str(row.get("youtube_id") or "").strip()
            if not value:
                raise ValueError("MeanAudio protocol row is missing youtube_id")
            values.add(value)
    if len(values) != 957:
        raise ValueError(f"expected 957 protocol IDs, found {len(values)}")
    return frozenset(values)


def load_forbidden_eval_video_ids() -> frozenset[str]:
    from task_audio.data.audioset import load_forbidden_video_ids

    return frozenset(
        load_meanaudio_video_ids().union(
            load_forbidden_video_ids(TANGO_886_PROTOCOL)
        )
    )


class _ItemSubset(Dataset):
    def __init__(self, dataset: Any, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = tuple(int(value) for value in indices)
        if not self.indices:
            raise ValueError("duration group cannot be empty")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[self.indices[int(index)]]


class RankShardedEpochSampler(Sampler[int]):
    """Deterministic map sampler that works before Accelerator is constructed."""

    def __init__(
        self,
        data_source: Sized,
        *,
        seed: int,
        pad_to_world_size: bool = True,
    ) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.start_index = 0
        self.pad_to_world_size = bool(pad_to_world_size)
        if not 0 <= self.rank < self.world_size:
            raise ValueError("invalid RANK/WORLD_SIZE")
        if not self.pad_to_world_size and len(self.data_source) % self.world_size:
            raise ValueError(
                "strict sample coverage cannot pad a distributed permutation: "
                f"samples={len(self.data_source)} world_size={self.world_size}"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(0, int(epoch))

    def set_start_index(self, index: int) -> None:
        index = max(0, int(index))
        if index > len(self):
            raise ValueError(
                f"sampler start index exceeds rank shard: {index} > {len(self)}"
            )
        self.start_index = index

    def __len__(self) -> int:
        return math.ceil(len(self.data_source) / self.world_size)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_source), generator=generator).tolist()
        padding = (-len(indices)) % self.world_size
        if padding:
            if not self.pad_to_world_size:
                raise RuntimeError("strict sampler unexpectedly requires padding")
            indices.extend(indices[:padding])
        rank_indices = indices[self.rank :: self.world_size]
        yield from rank_indices[self.start_index :]


def _tagged_collate(
    samples: list[Mapping[str, Any]],
    *,
    group: str,
    source: str,
) -> dict[str, Any]:
    batch = collate_audio_text(samples)
    # Keep one CUDA shape per duration bucket. Padding only to the longest
    # sample in each batch creates dozens of allocation sizes and caused the
    # CUDA cache to grow from 71 GiB in the fixed-shape probe to 76 GiB after
    # 50 real steps. The valid mask and lengths still preserve real duration.
    target_samples = int(
        round(PROBE_GROUP_SECONDS[group] * int(batch["wav_sample_rate"][0]))
    )
    current_samples = int(batch["wav_clean"].shape[-1])
    if current_samples < target_samples:
        padding = target_samples - current_samples
        batch["wav_clean"] = torch.nn.functional.pad(
            batch["wav_clean"], (0, padding)
        )
        batch["wav_valid_mask"] = torch.nn.functional.pad(
            batch["wav_valid_mask"], (0, padding), value=False
        )
    elif current_samples > target_samples:
        batch["wav_clean"] = batch["wav_clean"][:, :target_samples]
        batch["wav_valid_mask"] = batch["wav_valid_mask"][:, :target_samples]
        batch["wav_lengths"] = batch["wav_lengths"].clamp_max(target_samples)
    batch["mixture_group"] = group
    batch["mixture_source"] = source
    batch["mixture_local_batch_size"] = torch.tensor(len(samples))
    return batch


class HomogeneousDurationBatchDataset(IterableDataset):
    """Infinite weighted source stream whose individual batches are homogeneous."""

    handles_distributed_sharding = True

    def __init__(
        self,
        *,
        audioset: IterableDataset,
        wavcaps_groups: dict[str, Dataset],
        batch_sizes: Mapping[str, int],
        seed: int,
        num_workers: int,
        nominal_updates_per_epoch: int,
        probe_group: str | None = None,
        source_update_weights: Mapping[str, float] | None = None,
        strict_sample_coverage: bool = False,
    ) -> None:
        super().__init__()
        self.audioset = audioset
        self.wavcaps_groups = dict(wavcaps_groups)
        self.batch_sizes = {str(k): int(v) for k, v in batch_sizes.items()}
        raw_source_weights = {
            str(key): float(value)
            for key, value in (
                source_update_weights
                or {"audiosetcaps": 0.5, "wavcaps": 0.5}
            ).items()
        }
        expected_sources = {"audiosetcaps", "wavcaps"}
        if set(raw_source_weights) != expected_sources:
            raise ValueError(
                "source_update_weights must contain exactly "
                "'audiosetcaps' and 'wavcaps'"
            )
        if any(
            not math.isfinite(value) or value < 0.0
            for value in raw_source_weights.values()
        ):
            raise ValueError("source_update_weights must be finite and non-negative")
        source_weight_total = sum(raw_source_weights.values())
        if source_weight_total <= 0.0:
            raise ValueError("source_update_weights must have a positive sum")
        self.source_update_weights = {
            key: value / source_weight_total
            for key, value in raw_source_weights.items()
        }
        self.seed = int(seed)
        self.num_workers = int(num_workers)
        self.nominal_updates_per_epoch = int(nominal_updates_per_epoch)
        self.probe_group = None if probe_group is None else str(probe_group)
        self.strict_sample_coverage = bool(strict_sample_coverage)
        self.epoch = 0
        self._start_offset = 0
        required = {"audioset_10", *self.wavcaps_groups}
        missing = sorted(required.difference(self.batch_sizes))
        if missing:
            raise ValueError(f"missing duration batch sizes: {missing}")
        if min(self.batch_sizes.values()) < 1:
            raise ValueError("duration batch sizes must be positive")
        if self.num_workers < 0 or self.nominal_updates_per_epoch < 1:
            raise ValueError("invalid workers/nominal epoch")
        if self.probe_group is not None and self.probe_group not in required:
            raise ValueError(f"unknown probe duration group: {self.probe_group}")
        total = sum(len(value) for value in self.wavcaps_groups.values())
        self.wavcaps_group_weights = tuple(
            len(self.wavcaps_groups[name]) / total
            for name in self.wavcaps_groups
        )
        self.wavcaps_batches_per_pass: dict[str, int] = {}
        if self.strict_sample_coverage:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            for name, group in self.wavcaps_groups.items():
                if len(group) % world_size:
                    raise ValueError(
                        "strict WavCaps coverage requires a padding-free rank shard: "
                        f"group={name} samples={len(group)} world_size={world_size}"
                    )
                rank_samples = len(group) // world_size
                self.wavcaps_batches_per_pass[name] = math.ceil(
                    rank_samples / self.batch_sizes[name]
                )
            if not self.wavcaps_batches_per_pass:
                raise ValueError("strict WavCaps coverage has no duration groups")

    def __len__(self) -> int:
        # TTATrainer divides self-sharded loaders by world size for reporting.
        return self.nominal_updates_per_epoch * int(os.environ.get("WORLD_SIZE", "1"))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(0, int(epoch))
        if not self.strict_sample_coverage:
            setter = getattr(self.audioset, "set_epoch", None)
            if callable(setter):
                setter(self.epoch)

    def set_start_offset(self, offset: int) -> None:
        offset = int(offset)
        if not 0 <= offset <= self.nominal_updates_per_epoch:
            raise ValueError(
                "start offset is outside the nominal epoch: "
                f"{offset} not in [0, {self.nominal_updates_per_epoch}]"
            )
        self._start_offset = offset

    def _source_schedule_and_rng(
        self,
    ) -> tuple[tuple[str, ...], random.Random]:
        """Return one exact, deterministic per-update source schedule."""

        rng = random.Random(self.seed + self.epoch * 1_000_003)
        wavcaps_updates = int(
            round(
                self.nominal_updates_per_epoch
                * self.source_update_weights["wavcaps"]
            )
        )
        wavcaps_updates = min(
            max(wavcaps_updates, 0),
            self.nominal_updates_per_epoch,
        )
        audioset_updates = self.nominal_updates_per_epoch - wavcaps_updates

        # Preserve the original alternating stream exactly for legacy 50/50
        # runs. Other ratios use an exact-count deterministic shuffle.
        if audioset_updates == wavcaps_updates:
            phase = rng.randrange(2)
            schedule = tuple(
                "audioset" if (index + phase) % 2 == 0 else "wavcaps"
                for index in range(self.nominal_updates_per_epoch)
            )
        else:
            values = ["audioset"] * audioset_updates
            values.extend(["wavcaps"] * wavcaps_updates)
            rng.shuffle(values)
            schedule = tuple(values)
        return schedule, rng

    def _source_updates_per_epoch(self) -> tuple[int, int]:
        wavcaps = int(
            round(
                self.nominal_updates_per_epoch
                * self.source_update_weights["wavcaps"]
            )
        )
        wavcaps = min(max(wavcaps, 0), self.nominal_updates_per_epoch)
        return self.nominal_updates_per_epoch - wavcaps, wavcaps

    def _strict_wavcaps_schedule(self, pass_id: int) -> tuple[str, ...]:
        if not self.strict_sample_coverage:
            raise RuntimeError("strict WavCaps schedule requested in legacy mode")
        values: list[str] = []
        for name in self.wavcaps_groups:
            values.extend([name] * self.wavcaps_batches_per_pass[name])
        random.Random(
            self.seed + 7_919 + int(pass_id) * 1_000_003
        ).shuffle(values)
        return tuple(values)

    def _strict_wavcaps_group_at(self, absolute_update: int) -> str:
        batches_per_pass = sum(self.wavcaps_batches_per_pass.values())
        pass_id, offset = divmod(max(0, int(absolute_update)), batches_per_pass)
        return self._strict_wavcaps_schedule(pass_id)[offset]

    def _strict_wavcaps_counts_before(self, absolute_update: int) -> dict[str, int]:
        batches_per_pass = sum(self.wavcaps_batches_per_pass.values())
        completed, offset = divmod(max(0, int(absolute_update)), batches_per_pass)
        counts = {
            name: completed * batches
            for name, batches in self.wavcaps_batches_per_pass.items()
        }
        for name in self._strict_wavcaps_schedule(completed)[:offset]:
            counts[name] += 1
        return counts

    def state_for_offset(self, offset: int) -> dict[str, Any]:
        offset = max(0, int(offset))
        if offset > self.nominal_updates_per_epoch:
            raise ValueError(
                "update offset exceeds nominal_updates_per_epoch: "
                f"{offset} > {self.nominal_updates_per_epoch}"
            )
        schedule, _ = self._source_schedule_and_rng()
        audioset_updates = schedule[:offset].count("audioset")
        state = {
            "epoch": self.epoch,
            "update_offset": offset,
            "audioset_updates": audioset_updates,
            "wavcaps_updates": offset - audioset_updates,
            "source_update_weights": dict(self.source_update_weights),
            "schedule_seed": self.seed,
        }
        if self.strict_sample_coverage:
            audioset_per_epoch, wavcaps_per_epoch = self._source_updates_per_epoch()
            absolute_audioset = self.epoch * audioset_per_epoch + audioset_updates
            absolute_wavcaps = (
                self.epoch * wavcaps_per_epoch + offset - audioset_updates
            )
            audioset_batches_per_pass = self.audioset.rank_batches_per_pass(
                batch_size=self.batch_sizes["audioset_10"],
                num_workers=self.num_workers,
            )
            wavcaps_batches_per_pass = sum(
                self.wavcaps_batches_per_pass.values()
            )
            audioset_pass, audioset_pass_offset = divmod(
                absolute_audioset, audioset_batches_per_pass
            )
            wavcaps_pass, wavcaps_pass_offset = divmod(
                absolute_wavcaps, wavcaps_batches_per_pass
            )
            state.update(
                {
                    "data_stream_version": 2,
                    "coverage_semantics": "exhaustive_before_repeat",
                    "absolute_audioset_batches": absolute_audioset,
                    "audioset_batches_per_pass": audioset_batches_per_pass,
                    "audioset_pass": audioset_pass,
                    "audioset_pass_offset": audioset_pass_offset,
                    "absolute_wavcaps_batches": absolute_wavcaps,
                    "wavcaps_batches_per_pass": dict(
                        self.wavcaps_batches_per_pass
                    ),
                    "wavcaps_superpass_batches": wavcaps_batches_per_pass,
                    "wavcaps_pass": wavcaps_pass,
                    "wavcaps_pass_offset": wavcaps_pass_offset,
                    "wavcaps_group_batches_consumed": (
                        self._strict_wavcaps_counts_before(absolute_wavcaps)
                    ),
                }
            )
        return state

    @staticmethod
    def _cycle(
        loader: DataLoader,
        sampler: RankShardedEpochSampler | None = None,
        *,
        start_batches: int = 0,
    ):
        start_batches = max(0, int(start_batches))
        if sampler is not None:
            batches_per_cycle = len(loader)
            if batches_per_cycle < 1:
                raise RuntimeError("component duration loader has no full batch")
            cycle, batch_offset = divmod(start_batches, batches_per_cycle)
        else:
            cycle = 0
            batch_offset = 0
        while True:
            if sampler is not None:
                sampler.set_epoch(cycle)
                sampler.set_start_index(
                    batch_offset * int(loader.batch_size or 1)
                )
            yielded = False
            for batch in loader:
                yielded = True
                yield batch
            if not yielded:
                raise RuntimeError("component duration loader produced no batch")
            if sampler is not None:
                sampler.set_start_index(0)
            batch_offset = 0
            cycle += 1

    def __iter__(self) -> Iterator[dict[str, Any]]:
        # Outer DataLoader must use num_workers=0. Component loaders own decode
        # workers, avoiding worker-dependent source/range RNG.
        start_offset = self._start_offset
        self._start_offset = 0
        source_schedule, rng = self._source_schedule_and_rng()
        wavcaps_names = tuple(self.wavcaps_groups)
        groups: list[str] = []
        _, wavcaps_per_epoch = self._source_updates_per_epoch()
        absolute_wavcaps_update = self.epoch * wavcaps_per_epoch
        for source in source_schedule:
            if self.probe_group is not None:
                group = self.probe_group
            elif source == "audioset":
                group = "audioset_10"
            elif self.strict_sample_coverage:
                group = self._strict_wavcaps_group_at(absolute_wavcaps_update)
                absolute_wavcaps_update += 1
            else:
                group = rng.choices(
                    wavcaps_names, weights=self.wavcaps_group_weights, k=1
                )[0]
            groups.append(group)
        if self.strict_sample_coverage:
            audioset_per_epoch, wavcaps_per_epoch = self._source_updates_per_epoch()
            absolute_audioset_batches = (
                self.epoch * audioset_per_epoch
                + groups[:start_offset].count("audioset_10")
            )
            absolute_wavcaps_batches = (
                self.epoch * wavcaps_per_epoch
                + sum(
                    group != "audioset_10"
                    for group in groups[:start_offset]
                )
            )
            consumed_by_group = self._strict_wavcaps_counts_before(
                absolute_wavcaps_batches
            )
            audioset_batches_per_pass = self.audioset.rank_batches_per_pass(
                batch_size=self.batch_sizes["audioset_10"],
                num_workers=self.num_workers,
            )
            audioset_pass, audioset_batch_offset = divmod(
                absolute_audioset_batches, audioset_batches_per_pass
            )
            self.audioset.set_epoch(audioset_pass)
            consumed_by_group["audioset_10"] = audioset_batch_offset
        else:
            consumed_by_group = {
                group: groups[:start_offset].count(group)
                for group in {"audioset_10", *self.wavcaps_groups}
            }
        audioset_resume = getattr(
            self.audioset, "set_worker_resume_batches", None
        )
        if callable(audioset_resume):
            audioset_resume(
                consumed_by_group["audioset_10"],
                batch_size=self.batch_sizes["audioset_10"],
                num_workers=self.num_workers,
            )
        audioset_loader = DataLoader(
            self.audioset,
            batch_size=self.batch_sizes["audioset_10"],
            shuffle=False,
            drop_last=not self.strict_sample_coverage,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=partial(
                _tagged_collate, group="audioset_10", source="audioset"
            ),
        )
        component_iterators: dict[str, Iterator[dict[str, Any]]] = {
            "audioset_10": self._cycle(audioset_loader)
        }
        if consumed_by_group["audioset_10"] and not callable(audioset_resume):
            for _ in range(consumed_by_group["audioset_10"]):
                next(component_iterators["audioset_10"])
        for group, dataset in self.wavcaps_groups.items():
            sampler = RankShardedEpochSampler(
                dataset,
                seed=self.seed + 97 * (len(component_iterators) + 1),
                pad_to_world_size=not self.strict_sample_coverage,
            )
            loader = DataLoader(
                dataset,
                batch_size=self.batch_sizes[group],
                sampler=sampler,
                drop_last=not self.strict_sample_coverage,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=self.num_workers > 0,
                collate_fn=partial(
                    _tagged_collate, group=group, source="wavcaps"
                ),
            )
            component_iterators[group] = self._cycle(
                loader,
                sampler,
                start_batches=consumed_by_group[group],
            )

        for group in groups[start_offset:]:
            yield next(component_iterators[group])


class SyntheticProbeBatchDataset(IterableDataset):
    """Shape-faithful batches for GPU capacity probing.

    Dataset indexing and audio decoding do not affect CUDA capacity, and
    rebuilding the 1.7M-row AudioSet index in every rank for every candidate
    would dominate the search.  Probe mode therefore uses the upper duration
    bound of each real group while retaining the complete model, pMF/JVP,
    decoder, STFT, GAN and four-rank backward/optimizer path.
    """

    handles_distributed_sharding = True

    def __init__(
        self,
        *,
        group: str,
        batch_size: int,
        sample_rate: int,
        batches: int = 32,
    ) -> None:
        super().__init__()
        if group not in PROBE_GROUP_SECONDS:
            raise ValueError(f"unknown probe group: {group}")
        self.group = str(group)
        self.batch_size = int(batch_size)
        self.sample_rate = int(sample_rate)
        self.batches = int(batches)
        if self.batch_size < 1 or self.sample_rate < 1 or self.batches < 1:
            raise ValueError("invalid synthetic probe batch configuration")

    def __len__(self) -> int:
        return self.batches * int(os.environ.get("WORLD_SIZE", "1"))

    def __iter__(self) -> Iterator[dict[str, Any]]:
        samples = max(
            1,
            int(round(PROBE_GROUP_SECONDS[self.group] * self.sample_rate)),
        )
        dataset = (
            "audiocaps"
            if self.group == "audiocaps_10_5"
            else ("audioset" if self.group == "audioset_10" else "wavcaps")
        )
        # A low-amplitude deterministic signal keeps spectral losses finite
        # without paying real audio decode/index costs.
        time = torch.arange(samples, dtype=torch.float32) / self.sample_rate
        waveform = (
            0.01 * torch.sin(2.0 * math.pi * 440.0 * time)
        ).unsqueeze(0).expand(self.batch_size, -1).contiguous()
        valid = torch.ones(
            (self.batch_size, samples), dtype=torch.bool
        )
        lengths = torch.full(
            (self.batch_size,), samples, dtype=torch.long
        )
        sample_rates = torch.full(
            (self.batch_size,), self.sample_rate, dtype=torch.long
        )
        captions = ["synthetic audio capacity probe"] * self.batch_size
        identifiers = [
            f"probe-{self.group}-{index}" for index in range(self.batch_size)
        ]
        metadata = [
            {"dataset": dataset, "utt_id": identifier, "probe": True}
            for identifier in identifiers
        ]
        batch = {
            "wav_clean": waveform,
            "wav_valid_mask": valid,
            "wav_lengths": lengths,
            "wav_sample_rate": sample_rates,
            "caption": captions,
            "metadata": metadata,
            "utt_id": identifiers,
            "dataset": [dataset] * self.batch_size,
            "split": ["probe"] * self.batch_size,
            "audio_ref": [None] * self.batch_size,
            "mixture_group": self.group,
            "mixture_source": dataset,
            "mixture_local_batch_size": torch.tensor(self.batch_size),
        }
        for _ in range(self.batches):
            yield batch


class SyntheticMixedProbeBatchDataset(IterableDataset):
    """Alternate every Stage-1 shape to expose allocator fragmentation."""

    handles_distributed_sharding = True
    GROUP_SEQUENCE = (
        "wavcaps_0_5",
        "audioset_10",
        "wavcaps_5_10",
        "audioset_10",
        "wavcaps_10_15",
        "audioset_10",
        "wavcaps_15_20",
        "audioset_10",
    )

    def __init__(
        self,
        *,
        batch_sizes: Mapping[str, int],
        sample_rate: int,
        batches: int = 32,
    ) -> None:
        super().__init__()
        self.batches = int(batches)
        self.components = {
            group: SyntheticProbeBatchDataset(
                group=group,
                batch_size=int(batch_sizes[group]),
                sample_rate=sample_rate,
                batches=1,
            )
            for group in dict.fromkeys(self.GROUP_SEQUENCE)
        }

    def __len__(self) -> int:
        return self.batches * int(os.environ.get("WORLD_SIZE", "1"))

    def __iter__(self) -> Iterator[dict[str, Any]]:
        component_batches = {
            group: next(iter(dataset))
            for group, dataset in self.components.items()
        }
        for index in range(self.batches):
            yield component_batches[
                self.GROUP_SEQUENCE[index % len(self.GROUP_SEQUENCE)]
            ]


def build_probe_dataloader(
    config: Mapping[str, Any],
    *,
    group: str,
    batch_size: int,
) -> DataLoader:
    if group == "stage1_mixed":
        dataset = SyntheticMixedProbeBatchDataset(
            batch_sizes=dict(config["data"]["duration_batch_sizes"]),
            sample_rate=int(config["data"].get("sample_rate", 16_000)),
        )
        return DataLoader(dataset, batch_size=None, num_workers=0)
    dataset = SyntheticProbeBatchDataset(
        group=group,
        batch_size=batch_size,
        sample_rate=int(config["data"].get("sample_rate", 16_000)),
    )
    return DataLoader(dataset, batch_size=None, num_workers=0)


def build_stage1_dataloader(config: Mapping[str, Any]) -> DataLoader:
    data = dict(config["data"])
    training = dict(config["training"])
    seed = int(config.get("experiment", {}).get("seed", 1234))
    sample_rate = int(data.get("sample_rate", 16_000))
    resampling = dict(data.get("resampling") or {"method": "linear_legacy"})

    audioset_data = dict(data)
    audioset_data["dataset"] = "audioset"
    audioset_data["max_audio_seconds"] = 10.0
    audioset_settings = dict(audioset_data.get("audioset") or {})
    caption = dict(audioset_settings.get("caption") or {})
    caption.update(
        {
            "mode": "audiosetcaps",
            "fallback": "labels",
            "require_match": True,
        }
    )
    audioset_settings.update(
        {
            "caption": caption,
            # AudioCaps train is intentionally allowed. Only evaluation IDs
            # are forbidden in Stage 1.
            "exclude_audiocaps_overlaps": False,
            "forbidden_id_files": {
                "audiocaps_meanaudio_957": str(MEANAUDIO_957_PROTOCOL),
                "audiocaps_tango_886": str(TANGO_886_PROTOCOL),
            },
            "strict_sample_coverage": bool(
                data.get("strict_sample_coverage", False)
            ),
        }
    )
    audioset_data["audioset"] = audioset_settings
    audioset = build_dataset(
        {"data": audioset_data, "paths": config.get("paths", {})}, split="train"
    )
    expected_audioset = data.get("expected_audiosetcaps_rows")
    if (
        bool(data.get("strict_sample_coverage", False))
        and expected_audioset is not None
        and len(audioset) != int(expected_audioset)
    ):
        raise RuntimeError(
            "AudioSetCaps count mismatch under strict coverage: "
            f"expected {expected_audioset}, got {len(audioset)}"
        )

    wavcaps_data = dict(data)
    wavcaps_data["dataset"] = "wavcaps"
    wavcaps_data["max_audio_seconds"] = 20.0
    wavcaps_data["resampling"] = resampling
    wavcaps = build_dataset(
        {"data": wavcaps_data, "paths": config.get("paths", {})}, split="train"
    )
    expected_wavcaps = data.get("expected_wavcaps_rows")
    if expected_wavcaps is not None and len(wavcaps) != int(expected_wavcaps):
        raise RuntimeError(
            f"WavCaps count mismatch: expected {expected_wavcaps}, got {len(wavcaps)}"
        )
    forbidden_ids = load_forbidden_eval_video_ids()
    wavcaps_overlaps = []
    for item in wavcaps.items:
        if str(item.source) != "AudioSet_SL":
            continue
        item_id = Path(str(item.item_id)).stem
        video_id = item_id[1:] if item_id.startswith("Y") else item_id
        if video_id in forbidden_ids:
            wavcaps_overlaps.append(video_id)
    if wavcaps_overlaps:
        raise RuntimeError(
            "WavCaps contains forbidden AudioCaps-957 IDs: "
            f"{sorted(set(wavcaps_overlaps))[:8]}"
        )
    grouped_indices: dict[str, list[int]] = {
        name: [] for name, _, _ in WAVCAPS_DURATION_GROUPS
    }
    for index, item in enumerate(wavcaps.items):
        duration = float(item.duration)
        for name, low, high in WAVCAPS_DURATION_GROUPS:
            if low < duration <= high:
                grouped_indices[name].append(index)
                break
        else:
            raise RuntimeError(f"WavCaps duration outside (0,20]: {duration}")
    groups = {
        name: _ItemSubset(wavcaps, indices)
        for name, indices in grouped_indices.items()
    }
    expected_duration_rows = dict(
        data.get("expected_wavcaps_duration_rows") or {}
    )
    if expected_duration_rows:
        actual_duration_rows = {
            name: len(group) for name, group in groups.items()
        }
        if actual_duration_rows != {
            str(name): int(count)
            for name, count in expected_duration_rows.items()
        }:
            raise RuntimeError(
                "WavCaps duration-group count mismatch under strict coverage: "
                f"expected={expected_duration_rows} actual={actual_duration_rows}"
            )
    dataset = HomogeneousDurationBatchDataset(
        audioset=audioset,
        wavcaps_groups=groups,
        batch_sizes=dict(data["duration_batch_sizes"]),
        source_update_weights=dict(data["source_update_weights"]),
        seed=seed,
        num_workers=int(training.get("num_workers", 2)),
        nominal_updates_per_epoch=int(data.get("nominal_updates_per_epoch", 10_000)),
        probe_group=data.get("probe_group"),
        strict_sample_coverage=bool(data.get("strict_sample_coverage", False)),
    )
    # Samples are already collated by the selected component loader.
    return DataLoader(dataset, batch_size=None, num_workers=0)


def build_stage2_dataloader(config: Mapping[str, Any]) -> DataLoader:
    from task_audio.data import build_dataloader

    return build_dataloader(config, split="train")


__all__ = [
    "HomogeneousDurationBatchDataset",
    "MEANAUDIO_957_PROTOCOL",
    "TANGO_886_PROTOCOL",
    "RankShardedEpochSampler",
    "PROBE_GROUP_SECONDS",
    "SyntheticProbeBatchDataset",
    "SyntheticMixedProbeBatchDataset",
    "WAVCAPS_DURATION_GROUPS",
    "build_probe_dataloader",
    "build_stage1_dataloader",
    "build_stage2_dataloader",
    "load_meanaudio_video_ids",
    "load_forbidden_eval_video_ids",
]
