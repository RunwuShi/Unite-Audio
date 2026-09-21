from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import IterableDataset, get_worker_info

from .common import (
    collate_audio_text,
    normalize_resampling_config,
    resample_waveform,
    trim_waveform,
)


AUDIOSET_DEFAULT_ROOT = Path("/nativemm2/share/cpfs/audioset/data")
AUDIOSET_TRAIN_SPLITS: tuple[str, ...] = ("bal_train", "unbal_train")
AUDIOSET_EVAL_SPLITS: tuple[str, ...] = ("eval",)
AUDIOSET_REQUIRED_COLUMNS: tuple[str, ...] = ("video_id", "audio", "labels")
AUDIOSET_OPTIONAL_COLUMNS: tuple[str, ...] = ("human_labels",)
AudioSetCaptionMode = Literal[
    "labels", "template", "random_template", "audiosetcaps"
]


class AudioSetCapsCaptionIndex:
    """In-memory lookup for the official AudioSetCaps caption CSV.

    The index is constructed before DataLoader workers are forked so Linux
    workers can share its pages copy-on-write. Loading one index per DDP rank
    is intentional and avoids a filesystem lookup for every training sample.
    """

    def __init__(self, path: str | Path, *, id_prefix: str = "Y") -> None:
        self.path = Path(path).expanduser().resolve()
        self.id_prefix = str(id_prefix)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"missing AudioSetCaps caption metadata: {self.path}"
            )

        captions: dict[str, str] = {}
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            missing = sorted({"id", "caption"}.difference(fields))
            if missing:
                raise ValueError(
                    f"AudioSetCaps CSV {self.path} is missing column(s): {missing}"
                )
            for row_number, row in enumerate(reader, start=2):
                external_id = str(row.get("id") or "").strip()
                caption = str(row.get("caption") or "").strip()
                if not external_id or not caption:
                    raise ValueError(
                        f"AudioSetCaps CSV has an empty id/caption at row {row_number}"
                    )
                if external_id in captions:
                    raise ValueError(
                        f"AudioSetCaps CSV has duplicate id {external_id!r} "
                        f"at row {row_number}"
                    )
                captions[external_id] = caption
        if not captions:
            raise ValueError(f"AudioSetCaps CSV contains no captions: {self.path}")
        self._captions = captions

    def external_id(self, video_id: str) -> str:
        return f"{self.id_prefix}{video_id}"

    def get(self, video_id: str) -> str | None:
        return self._captions.get(self.external_id(str(video_id)))

    def __len__(self) -> int:
        return len(self._captions)


@dataclass(frozen=True)
class AudioSetCaptionFormatter:
    mode: AudioSetCaptionMode = "labels"
    template: str = "{labels}"
    templates: tuple[str, ...] = ()
    label_joiner: str = ", "
    lowercase: bool = False

    def __call__(
        self,
        *,
        human_labels: Sequence[str],
        label_ids: Sequence[str],
        dataset: str,
        utt_id: str,
        rng: random.Random | None = None,
    ) -> str:
        human = _normalize_labels(human_labels)
        ids = _normalize_labels(label_ids)
        selected = human or ids
        labels_text = self.label_joiner.join(selected)
        label_ids_text = self.label_joiner.join(ids)
        if self.lowercase:
            labels_text = labels_text.lower()
            label_ids_text = label_ids_text.lower()

        if self.mode == "labels":
            return labels_text
        if self.mode == "audiosetcaps":
            raise RuntimeError(
                "AudioSetCaps captions must be resolved by AudioSetDataset"
            )
        if self.mode == "template":
            template = self.template
        elif self.mode == "random_template":
            choices = self.templates or (self.template,)
            template = (rng or random).choice(tuple(choices))
        else:
            raise ValueError(f"unknown AudioSet caption mode: {self.mode!r}")
        return template.format(
            labels=labels_text,
            human_labels=self.label_joiner.join(human),
            label=selected[0] if selected else "",
            label_ids=label_ids_text,
            dataset=dataset,
            utt_id=utt_id,
        ).strip()


class AudioSetDataset(IterableDataset):
    """Stream HuggingFace-style AudioSet parquet shards.

    Files are deterministically partitioned over distributed rank and DataLoader
    worker.  This prevents duplicate examples without materializing an index for
    the multi-million-row unbalanced training set.
    """

    # The dataset partitions parquet files over distributed ranks itself.
    # TTATrainer uses this marker to avoid Accelerate wrapping it in a second
    # IterableDatasetShard (or dispatching rank-0 batches containing strings).
    handles_distributed_sharding = True

    def __init__(
        self,
        root: str | Path = AUDIOSET_DEFAULT_ROOT,
        *,
        splits: str | Sequence[str] | None = None,
        sample_rate: int = 16_000,
        max_audio_seconds: float | None = 10.0,
        caption_formatter: AudioSetCaptionFormatter | None = None,
        audiosetcaps_path: str | Path | None = None,
        audiosetcaps_id_prefix: str = "Y",
        audiosetcaps_fallback: str = "labels",
        audiosetcaps_require_match: bool = False,
        seed: int = 1234,
        shuffle_files: bool = True,
        read_batch_size: int = 16,
        required_pyarrow_version: str | None = "18.1.0",
        forbidden_id_files: Mapping[str, str | Path] | None = None,
        forbidden_protocol: str | Path | None = None,
        resampling: Mapping[str, Any] | None = None,
        strict_sample_coverage: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.splits = normalize_audioset_splits(splits)
        self.sample_rate = int(sample_rate)
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.max_audio_seconds = max_audio_seconds
        self.resampling = normalize_resampling_config(resampling)
        if max_audio_seconds is not None and float(max_audio_seconds) <= 0.0:
            raise ValueError("max_audio_seconds must be positive or None")
        self.caption_formatter = caption_formatter or AudioSetCaptionFormatter()
        self.audiosetcaps_fallback = str(audiosetcaps_fallback).strip().lower()
        if self.audiosetcaps_fallback != "labels":
            raise ValueError("AudioSetCaps fallback must currently be 'labels'")
        self.audiosetcaps_index: AudioSetCapsCaptionIndex | None = None
        self.audiosetcaps_require_match = bool(audiosetcaps_require_match)
        if self.audiosetcaps_require_match and self.caption_formatter.mode != "audiosetcaps":
            raise ValueError(
                "AudioSetCaps caption.require_match requires caption.mode='audiosetcaps'"
            )
        if self.caption_formatter.mode == "audiosetcaps":
            if audiosetcaps_path is None:
                raise ValueError(
                    "AudioSetCaps caption mode requires caption.metadata_path"
                )
            self.audiosetcaps_index = AudioSetCapsCaptionIndex(
                audiosetcaps_path,
                id_prefix=audiosetcaps_id_prefix,
            )
        self.seed = int(seed)
        self.shuffle_files = bool(shuffle_files)
        self.read_batch_size = max(1, int(read_batch_size))
        self.required_pyarrow_version = required_pyarrow_version
        self.strict_sample_coverage = bool(strict_sample_coverage)
        source_files = {
            str(name): Path(path).expanduser().resolve()
            for name, path in dict(forbidden_id_files or {}).items()
        }
        # Keep the singular JSONL argument as a backwards-compatible alias.
        self.forbidden_protocol = None
        if forbidden_protocol is not None:
            self.forbidden_protocol = Path(forbidden_protocol).expanduser().resolve()
            source_files.setdefault("forbidden_protocol", self.forbidden_protocol)
        self.forbidden_id_files = source_files
        self.forbidden_video_ids_by_source = {
            name: load_forbidden_video_ids(path)
            for name, path in self.forbidden_id_files.items()
        }
        self.forbidden_video_ids = frozenset().union(
            *self.forbidden_video_ids_by_source.values()
        )
        self._epoch = 0
        self._num_rows: int | None = None
        self._forbidden_matches: list[dict[str, Any]] | None = None
        self._caption_missing_matches: list[dict[str, Any]] | None = None
        self._valid_rows_per_file: dict[Path, int] | None = None
        # Resume offsets are written by the parent process before DataLoader
        # workers start iterating.  One slot per worker avoids replaying and
        # decoding already-consumed audio on exact checkpoint continuation.
        self._resume_samples_by_worker = torch.zeros(
            256, dtype=torch.int64
        ).share_memory_()
        require_pyarrow_parquet(self.required_pyarrow_version)
        self.parquet_files = self._discover_parquet_files()

    def __len__(self) -> int:
        if self.strict_sample_coverage:
            return sum(self.valid_rows_per_file.values())
        if self._num_rows is None:
            parquet = require_pyarrow_parquet(self.required_pyarrow_version)
            self._num_rows = sum(
                int(parquet.ParquetFile(path).metadata.num_rows)
                for path in self.parquet_files
            )
        return (
            self._num_rows
            - len(self.forbidden_matches)
            - len(self.caption_missing_matches)
        )

    @property
    def valid_rows_per_file(self) -> dict[Path, int]:
        """Exact valid-row counts used to make strict shards lossless.

        Counts are computed from the lightweight ``video_id`` column only;
        embedded audio is never decoded during this audit.
        """

        if self._valid_rows_per_file is not None:
            return self._valid_rows_per_file
        parquet = require_pyarrow_parquet(self.required_pyarrow_version)
        counts: dict[Path, int] = {}
        for path in self.parquet_files:
            count = 0
            parquet_file = parquet.ParquetFile(path)
            for batch in parquet_file.iter_batches(
                batch_size=max(self.read_batch_size, 65_536),
                columns=["video_id"],
                use_threads=False,
            ):
                for value in batch.column(0).to_pylist():
                    if self._is_valid_video_id(str(value)):
                        count += 1
            counts[path] = count
        self._valid_rows_per_file = counts
        return counts

    def _is_valid_video_id(self, video_id: str) -> bool:
        if video_id in self.forbidden_video_ids:
            return False
        return not (
            self.audiosetcaps_require_match
            and self.audiosetcaps_index is not None
            and self.audiosetcaps_index.get(video_id) is None
        )

    @staticmethod
    def _balanced_shard_bounds(
        total: int, shard_id: int, shard_count: int
    ) -> tuple[int, int]:
        if shard_count < 1 or not 0 <= shard_id < shard_count:
            raise ValueError("invalid strict AudioSet shard geometry")
        return (
            total * shard_id // shard_count,
            total * (shard_id + 1) // shard_count,
        )

    def rank_batches_per_pass(
        self,
        *,
        batch_size: int,
        num_workers: int,
    ) -> int:
        """Return an exact, equal DDP batch count for one strict pass.

        Strict continuation relies on DataLoader's round-robin worker order.
        We therefore fail closed if the requested geometry would give workers
        different numbers of batches, rather than silently making rank cursors
        ambiguous or allowing one worker to repeat early.
        """

        if not self.strict_sample_coverage:
            raise RuntimeError("rank_batches_per_pass requires strict coverage")
        batch_size = int(batch_size)
        workers = max(1, int(num_workers))
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        total = len(self)
        shard_count = world_size * workers
        all_batch_counts: list[int] = []
        for shard_id in range(shard_count):
            start, end = self._balanced_shard_bounds(total, shard_id, shard_count)
            all_batch_counts.append(math.ceil((end - start) / batch_size))
        if len(set(all_batch_counts)) != 1:
            raise RuntimeError(
                "strict AudioSet coverage requires equal worker batch counts; "
                f"total={total} world_size={world_size} workers={workers} "
                f"batch_size={batch_size} counts={all_batch_counts}"
            )
        if not 0 <= rank < world_size:
            raise ValueError("invalid RANK/WORLD_SIZE")
        first = rank * workers
        return sum(all_batch_counts[first : first + workers])

    @property
    def raw_num_rows(self) -> int:
        if self._num_rows is None:
            parquet = require_pyarrow_parquet(self.required_pyarrow_version)
            self._num_rows = sum(
                int(parquet.ParquetFile(path).metadata.num_rows)
                for path in self.parquet_files
            )
        assert self._num_rows is not None
        return self._num_rows

    @property
    def forbidden_matches(self) -> list[dict[str, Any]]:
        """Physical rows removed by the configured forbidden-ID sources."""

        if self._forbidden_matches is None:
            self._forbidden_matches = self.scan_video_id_matches(
                self.forbidden_video_ids
            )
        return self._forbidden_matches

    @property
    def caption_missing_matches(self) -> list[dict[str, Any]]:
        """Rows dropped because strict AudioSetCaps mode found no caption."""

        if not self.audiosetcaps_require_match:
            return []
        if self.audiosetcaps_index is None:
            raise RuntimeError("strict AudioSetCaps mode has no caption index")
        if self._caption_missing_matches is None:
            parquet = require_pyarrow_parquet(self.required_pyarrow_version)
            missing: list[dict[str, Any]] = []
            for path in self.parquet_files:
                parquet_file = parquet.ParquetFile(path)
                row_offset = 0
                for batch in parquet_file.iter_batches(
                    batch_size=max(self.read_batch_size, 65_536),
                    columns=["video_id"],
                    use_threads=False,
                ):
                    for index, value in enumerate(batch.column(0).to_pylist()):
                        video_id = str(value)
                        if (
                            video_id not in self.forbidden_video_ids
                            and self.audiosetcaps_index.get(video_id) is None
                        ):
                            missing.append(
                                {
                                    "video_id": video_id,
                                    "split": path.parent.name,
                                    "parquet_path": str(path),
                                    "row_index": row_offset + index,
                                }
                            )
                    row_offset += batch.num_rows
            self._caption_missing_matches = missing
        return self._caption_missing_matches

    def scan_video_id_matches(
        self,
        video_ids: Sequence[str] | set[str] | frozenset[str],
    ) -> list[dict[str, Any]]:
        """Find source rows by ID without reading or decoding embedded audio."""

        wanted = {str(value) for value in video_ids}
        if not wanted:
            return []
        parquet = require_pyarrow_parquet(self.required_pyarrow_version)
        matches: list[dict[str, Any]] = []
        for path in self.parquet_files:
            parquet_file = parquet.ParquetFile(path)
            row_offset = 0
            for batch in parquet_file.iter_batches(
                batch_size=max(self.read_batch_size, 65_536),
                columns=["video_id"],
                use_threads=False,
            ):
                values = batch.column(0).to_pylist()
                for row_index, value in enumerate(values):
                    video_id = str(value)
                    if video_id in wanted:
                        matches.append(
                            {
                                "video_id": video_id,
                                "split": path.parent.name,
                                "parquet_path": str(path),
                                "row_index": row_offset + row_index,
                            }
                        )
                row_offset += len(values)
        return matches

    def set_epoch(self, epoch: int) -> None:
        self._epoch = max(0, int(epoch))

    def set_worker_resume_batches(
        self,
        batches: int,
        *,
        batch_size: int,
        num_workers: int,
    ) -> None:
        """Position each iterable worker at an already-consumed batch cursor."""

        batches = max(0, int(batches))
        batch_size = int(batch_size)
        workers = max(1, int(num_workers))
        if batch_size < 1:
            raise ValueError("resume batch_size must be positive")
        if workers > int(self._resume_samples_by_worker.numel()):
            raise ValueError(
                f"resume num_workers exceeds supported slots: {workers} > "
                f"{self._resume_samples_by_worker.numel()}"
            )
        self._resume_samples_by_worker.zero_()
        complete_rounds, remainder = divmod(batches, workers)
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        total = len(self) if self.strict_sample_coverage else 0
        for worker_id in range(workers):
            worker_batches = complete_rounds + int(worker_id < remainder)
            resume_samples = worker_batches * batch_size
            if self.strict_sample_coverage:
                shard_id = rank * workers + worker_id
                start, end = self._balanced_shard_bounds(
                    total, shard_id, world_size * workers
                )
                resume_samples = min(resume_samples, end - start)
            self._resume_samples_by_worker[worker_id] = resume_samples

    def __iter__(self) -> Iterator[dict[str, Any]]:
        parquet = require_pyarrow_parquet(self.required_pyarrow_version)
        epoch = self._epoch
        self._epoch += 1
        files = list(self.parquet_files)
        if self.shuffle_files:
            random.Random(self.seed + epoch).shuffle(files)

        shard_id, shard_count = distributed_worker_shard_info()
        if self.strict_sample_coverage:
            yield from self._iter_strict_shard(
                parquet,
                files,
                epoch=epoch,
                shard_id=shard_id,
                shard_count=shard_count,
            )
            return
        files = [path for index, path in enumerate(files) if index % shard_count == shard_id]
        rng = random.Random(self.seed + epoch * 1_000_003 + shard_id)
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        resume_samples = [int(self._resume_samples_by_worker[worker_id].item())]
        self._resume_samples_by_worker[worker_id] = 0
        for path in files:
            yield from self._iter_parquet_file(
                parquet,
                path,
                rng,
                resume_samples=resume_samples,
            )
        if resume_samples[0] > 0:
            raise RuntimeError(
                "AudioSet resume cursor exceeds this rank/worker stream: "
                f"rank_worker_shard={shard_id}/{shard_count} "
                f"remaining_samples={resume_samples[0]}"
            )

    def _iter_strict_shard(
        self,
        parquet: Any,
        files: Sequence[Path],
        *,
        epoch: int,
        shard_id: int,
        shard_count: int,
    ) -> Iterator[dict[str, Any]]:
        """Yield one balanced, exhaustive shard of the shuffled valid rows."""

        total = len(self)
        shard_start, shard_end = self._balanced_shard_bounds(
            total, shard_id, shard_count
        )
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        resume = int(self._resume_samples_by_worker[worker_id].item())
        self._resume_samples_by_worker[worker_id] = 0
        shard_size = shard_end - shard_start
        if resume > shard_size:
            raise RuntimeError(
                "AudioSet strict resume exceeds worker shard: "
                f"resume={resume} shard_size={shard_size} shard={shard_id}/{shard_count}"
            )
        wanted_start = shard_start + resume
        wanted_end = shard_end
        logical_offset = 0
        yielded = 0
        rng = random.Random(self.seed + epoch * 1_000_003 + shard_id)
        counts = self.valid_rows_per_file
        for path in files:
            file_count = counts[path]
            file_start = logical_offset
            file_end = file_start + file_count
            logical_offset = file_end
            overlap_start = max(wanted_start, file_start)
            overlap_end = min(wanted_end, file_end)
            if overlap_start >= overlap_end:
                continue
            skip = overlap_start - file_start
            take = overlap_end - overlap_start
            rows = self._iter_parquet_file(
                parquet,
                path,
                rng,
                resume_samples=[skip],
            )
            for row in islice(rows, take):
                yielded += 1
                yield row
        expected = wanted_end - wanted_start
        if yielded != expected:
            raise RuntimeError(
                "AudioSet strict shard produced an incomplete pass: "
                f"yielded={yielded} expected={expected} "
                f"shard={shard_id}/{shard_count} epoch={epoch}"
            )

    def _discover_parquet_files(self) -> list[Path]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"missing AudioSet root: {self.root}")
        files: list[Path] = []
        for split in self.splits:
            directory = self.root / split
            if not directory.is_dir():
                raise FileNotFoundError(f"missing AudioSet split directory: {directory}")
            files.extend(sorted(directory.glob("*.parquet")))
        if not files:
            raise FileNotFoundError(
                f"no AudioSet parquet files found under {self.root} for splits {self.splits}"
            )
        return files

    def _iter_parquet_file(
        self,
        parquet: Any,
        path: Path,
        rng: random.Random,
        *,
        resume_samples: list[int] | None = None,
    ):
        parquet_file = parquet.ParquetFile(path)
        available = set(parquet_file.schema_arrow.names)
        missing = sorted(set(AUDIOSET_REQUIRED_COLUMNS).difference(available))
        if missing:
            raise ValueError(f"AudioSet parquet {path} is missing column(s): {missing}")
        columns_to_read = [
            name
            for name in AUDIOSET_REQUIRED_COLUMNS + AUDIOSET_OPTIONAL_COLUMNS
            if name in available
        ]
        resume_samples = resume_samples if resume_samples is not None else [0]
        start_row_group = 0
        start_row_in_group = 0
        if resume_samples[0] > 0:
            (
                start_row_group,
                start_row_in_group,
            ) = self._locate_resume_row(
                parquet_file,
                rng,
                resume_samples=resume_samples,
            )
            if resume_samples[0] > 0:
                return
        split = path.parent.name
        row_group_base = sum(
            int(parquet_file.metadata.row_group(index).num_rows)
            for index in range(start_row_group)
        )
        for row_group in range(start_row_group, parquet_file.num_row_groups):
            row_in_group = 0
            for record_batch in parquet_file.iter_batches(
                batch_size=self.read_batch_size,
                row_groups=[row_group],
                columns=columns_to_read,
                use_threads=False,
            ):
                columns = record_batch.to_pydict()
                count = len(columns["video_id"])
                for row_index in range(count):
                    source_row_index = row_group_base + row_in_group
                    row_in_group += 1
                    if (
                        row_group == start_row_group
                        and source_row_index
                        < row_group_base + start_row_in_group
                    ):
                        continue
                    video_id = str(columns["video_id"][row_index])
                    # Filter before touching labels or embedded audio.  The mounted
                    # AudioSet shards do not retain segment start times, so a whole
                    # YouTube-ID exclusion is the conservative leakage boundary.
                    if video_id in self.forbidden_video_ids:
                        continue
                    external_caption: str | None = None
                    if self.audiosetcaps_index is not None:
                        external_caption = self.audiosetcaps_index.get(video_id)
                        if (
                            self.audiosetcaps_require_match
                            and external_caption is None
                        ):
                            continue
                    label_ids = _normalize_labels(columns["labels"][row_index])
                    human_labels = _normalize_labels(
                        columns.get("human_labels", [None] * count)[row_index]
                    )
                    audio = columns["audio"][row_index] or {}
                    audio_bytes = (
                        audio.get("bytes") if isinstance(audio, dict) else None
                    )
                    if audio_bytes is None:
                        raise ValueError(
                            "AudioSet row has no embedded audio bytes: "
                            f"{path}:{source_row_index}"
                        )
                    wav, source_rate = load_audio_bytes(bytes(audio_bytes))
                    if source_rate != self.sample_rate:
                        wav = resample_waveform(
                            wav,
                            source_rate,
                            self.sample_rate,
                            **self.resampling,
                        )
                    wav = trim_waveform(
                        wav,
                        sample_rate=self.sample_rate,
                        max_audio_seconds=self.max_audio_seconds,
                    )
                    wav = torch.clamp(wav.contiguous(), -1.0, 1.0)

                    utt_id = f"audioset/{split}/{video_id}"
                    audiosetcaps_id: str | None = None
                    audiosetcaps_matched = False
                    if self.audiosetcaps_index is not None:
                        audiosetcaps_id = self.audiosetcaps_index.external_id(
                            video_id
                        )
                        if external_caption is not None:
                            caption = external_caption
                            caption_source = "audiosetcaps"
                            audiosetcaps_matched = True
                        else:
                            caption = self._fallback_caption(
                                human_labels=human_labels,
                                label_ids=label_ids,
                                utt_id=utt_id,
                            )
                            caption_source = "labels_fallback"
                    else:
                        caption = self.caption_formatter(
                            human_labels=human_labels,
                            label_ids=label_ids,
                            dataset="audioset",
                            utt_id=utt_id,
                            rng=rng,
                        )
                        caption_source = self.caption_formatter.mode
                    metadata = {
                        "dataset": "audioset",
                        "utt_id": utt_id,
                        "split": split,
                        "video_id": video_id,
                        "labels": label_ids,
                        "human_labels": human_labels,
                        "caption_mode": self.caption_formatter.mode,
                        "caption_source": caption_source,
                        "audiosetcaps_id": audiosetcaps_id,
                        "audiosetcaps_matched": audiosetcaps_matched,
                        "row_index": source_row_index,
                        "audio_ref": {
                            "parquet_path": str(path),
                            "audio_path": str(audio.get("path") or ""),
                            "source_sample_rate": source_rate,
                            "row_index": source_row_index,
                        },
                    }
                    yield {
                        "dataset": "audioset",
                        "utt_id": utt_id,
                        "caption": caption,
                        "split": split,
                        "wav": wav,
                        "wav_len": int(wav.numel()),
                        "wav_sample_rate": self.sample_rate,
                        "metadata": metadata,
                    }
            row_group_base += int(
                parquet_file.metadata.row_group(row_group).num_rows
            )

    def _locate_resume_row(
        self,
        parquet_file: Any,
        rng: random.Random,
        *,
        resume_samples: list[int],
    ) -> tuple[int, int]:
        """Scan metadata only and return the first physical row after a cursor."""

        for row_group in range(parquet_file.num_row_groups):
            video_ids = parquet_file.read_row_group(
                row_group,
                columns=["video_id"],
                use_threads=False,
            ).column("video_id").to_pylist()
            for row_in_group, value in enumerate(video_ids):
                video_id = str(value)
                if video_id in self.forbidden_video_ids:
                    continue
                if (
                    self.audiosetcaps_require_match
                    and self.audiosetcaps_index is not None
                    and self.audiosetcaps_index.get(video_id) is None
                ):
                    continue
                if resume_samples[0] <= 0:
                    return row_group, row_in_group
                if self.caption_formatter.mode == "random_template":
                    choices = (
                        self.caption_formatter.templates
                        or (self.caption_formatter.template,)
                    )
                    rng.choice(tuple(choices))
                resume_samples[0] -= 1
                if resume_samples[0] == 0:
                    return row_group, row_in_group + 1
        return parquet_file.num_row_groups, 0

    def _fallback_caption(
        self,
        *,
        human_labels: Sequence[str],
        label_ids: Sequence[str],
        utt_id: str,
    ) -> str:
        return AudioSetCaptionFormatter(
            mode="labels",
            label_joiner=self.caption_formatter.label_joiner,
            lowercase=self.caption_formatter.lowercase,
        )(
            human_labels=human_labels,
            label_ids=label_ids,
            dataset="audioset",
            utt_id=utt_id,
        )


def normalize_audioset_splits(splits: str | Sequence[str] | None) -> tuple[str, ...]:
    if splits is None:
        return AUDIOSET_TRAIN_SPLITS
    if isinstance(splits, str):
        normalized = splits.strip().lower()
        if normalized in {"", "train"}:
            return AUDIOSET_TRAIN_SPLITS
        if normalized in {"eval", "test", "valid", "validation"}:
            return AUDIOSET_EVAL_SPLITS
        values = tuple(normalized.replace(",", " ").split())
    else:
        values = tuple(str(value).strip().lower() for value in splits)
    allowed = set(AUDIOSET_TRAIN_SPLITS + AUDIOSET_EVAL_SPLITS)
    unknown = sorted(set(values).difference(allowed))
    if unknown:
        raise ValueError(f"unknown AudioSet split(s): {unknown}; expected one of {sorted(allowed)}")
    if not values:
        raise ValueError("AudioSet splits cannot be empty")
    return values


def require_pyarrow_parquet(required_version: str | None = "18.1.0"):
    try:
        import pyarrow
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("AudioSet parquet reading requires pyarrow==18.1.0") from exc
    if required_version and pyarrow.__version__ != required_version:
        raise RuntimeError(
            f"AudioSet parquet reading requires pyarrow=={required_version}; "
            f"found pyarrow=={pyarrow.__version__}. Other versions are known to fail on these shards."
        )
    return parquet


def _forbidden_id_cache_path(protocol: Path) -> Path | None:
    cache_root = os.environ.get("TASK_AUDIO_FORBIDDEN_ID_CACHE_DIR", "").strip()
    if not cache_root or protocol.suffix.lower() != ".parquet":
        return None
    stat = protocol.stat()
    identity = f"{protocol}\0{stat.st_size}\0{stat.st_mtime_ns}".encode()
    digest = hashlib.sha256(identity).hexdigest()
    return Path(cache_root).expanduser().resolve() / f"{digest}.json"


def _read_forbidden_id_cache(cache_path: Path, protocol: Path) -> frozenset[str]:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    stat = protocol.stat()
    if (
        payload.get("version") != 1
        or Path(payload.get("source", "")).resolve() != protocol
        or int(payload.get("size", -1)) != stat.st_size
        or int(payload.get("mtime_ns", -1)) != stat.st_mtime_ns
    ):
        raise ValueError(f"stale forbidden-ID cache: {cache_path}")
    video_ids = frozenset(str(value).strip() for value in payload.get("video_ids", ()))
    if not video_ids or "" in video_ids:
        raise ValueError(f"invalid forbidden-ID cache: {cache_path}")
    return video_ids


def _write_forbidden_id_cache(
    cache_path: Path, protocol: Path, video_ids: set[str]
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    stat = protocol.stat()
    payload = {
        "version": 1,
        "source": str(protocol),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "video_ids": sorted(video_ids),
    }
    temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, cache_path)


def load_forbidden_video_ids(path: str | Path | None) -> frozenset[str]:
    """Load YouTube IDs from a CSV manifest or JSONL protocol."""

    if path is None:
        return frozenset()
    protocol = Path(path).expanduser().resolve()
    if not protocol.is_file():
        raise FileNotFoundError(f"missing forbidden evaluation protocol: {protocol}")
    cache_path = _forbidden_id_cache_path(protocol)
    if cache_path is not None and cache_path.is_file():
        return _read_forbidden_id_cache(cache_path, protocol)
    video_ids: set[str] = set()
    if protocol.suffix.lower() == ".csv":
        with protocol.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            id_column = "youtube_id" if "youtube_id" in fields else "video_id"
            if id_column not in fields:
                raise ValueError(
                    f"forbidden CSV has no youtube_id/video_id column: {protocol}"
                )
            for line_number, record in enumerate(reader, start=2):
                video_id = str(record.get(id_column) or "").strip()
                if not video_id:
                    raise ValueError(
                        f"forbidden CSV has an empty {id_column} at "
                        f"{protocol}:{line_number}"
                    )
                video_ids.add(video_id)
    elif protocol.suffix.lower() == ".jsonl":
        with protocol.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                video_id = str(
                    record.get("youtube_id") or record.get("video_id") or ""
                ).strip()
                if not video_id:
                    raise ValueError(
                        f"forbidden protocol has no youtube_id/video_id at "
                        f"{protocol}:{line_number}"
                    )
                video_ids.add(video_id)
    elif protocol.suffix.lower() == ".parquet":
        parquet = require_pyarrow_parquet(None)
        parquet_file = parquet.ParquetFile(protocol)
        fields = set(parquet_file.schema_arrow.names)
        id_column = "youtube_id" if "youtube_id" in fields else "video_id"
        if id_column not in fields:
            raise ValueError(
                f"forbidden Parquet has no youtube_id/video_id column: {protocol}"
            )
        for batch in parquet_file.iter_batches(
            batch_size=65_536,
            columns=[id_column],
            use_threads=False,
        ):
            for value in batch.column(0).to_pylist():
                video_id = str(value or "").strip()
                if not video_id:
                    raise ValueError(
                        f"forbidden Parquet has an empty {id_column}: {protocol}"
                    )
                video_ids.add(video_id)
    else:
        raise ValueError(
            f"forbidden ID source must be CSV, JSONL, or Parquet: {protocol}"
        )
    if not video_ids:
        raise ValueError(f"forbidden protocol contains no video IDs: {protocol}")
    if cache_path is not None:
        _write_forbidden_id_cache(cache_path, protocol, video_ids)
    return frozenset(video_ids)


def distributed_worker_shard_info() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = int(torch.distributed.get_rank())
        world_size = int(torch.distributed.get_world_size())
    else:
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    worker = get_worker_info()
    worker_id = int(worker.id) if worker is not None else 0
    worker_count = int(worker.num_workers) if worker is not None else 1
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(f"invalid distributed rank/world size: rank={rank}, world_size={world_size}")
    return rank * worker_count + worker_id, world_size * worker_count


def load_audio_bytes(audio_bytes: bytes) -> tuple[torch.Tensor, int]:
    import numpy as np
    import soundfile as sf

    data, sample_rate = sf.read(BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return torch.from_numpy(np.ascontiguousarray(data, dtype="float32")), int(sample_rate)


def _normalize_labels(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    else:
        try:
            values = list(value)
        except TypeError:
            values = [value]
    return [str(item).strip() for item in values if str(item).strip()]


collate_audioset = collate_audio_text


__all__ = [
    "AUDIOSET_DEFAULT_ROOT",
    "AUDIOSET_EVAL_SPLITS",
    "AUDIOSET_TRAIN_SPLITS",
    "AudioSetCapsCaptionIndex",
    "AudioSetCaptionFormatter",
    "AudioSetDataset",
    "collate_audioset",
    "distributed_worker_shard_info",
    "load_forbidden_video_ids",
    "load_audio_bytes",
    "normalize_audioset_splits",
    "require_pyarrow_parquet",
]
