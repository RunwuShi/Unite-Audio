from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from .common import (
    collate_audio_text,
    normalize_resampling_config,
    resample_waveform,
    trim_waveform,
)


AUDIOCAPS_SPLIT_COUNTS: dict[str, int] = {
    "train": 45_178,
    "validation": 2_223,
    "test": 4_411,
}
AudioCapsSplit = Literal["train", "validation", "valid", "test", "all"]


@dataclass(frozen=True)
class AudioCapsItem:
    audiocap_id: str
    youtube_id: str
    start_time: float
    audio_length: int
    wav_path: str
    caption: str
    split: str
    row_index: int

    @property
    def utt_id(self) -> str:
        return f"audiocaps/{self.split}/{self.audiocap_id}"


def load_audiocaps_rows(
    root: str | Path,
    *,
    validate_counts: bool = True,
) -> list[AudioCapsItem]:
    root = Path(root)
    csv_path = root / "captions.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing AudioCaps captions file: {csv_path}")

    required_columns = {
        "audiocap_id",
        "youtube_id",
        "start_time",
        "audio_length",
        "wav_path",
        "caption",
    }
    rows: list[AudioCapsItem] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = sorted(required_columns.difference(reader.fieldnames or ()))
        if missing_columns:
            raise ValueError(f"AudioCaps captions.csv is missing column(s): {missing_columns}")
        for row_index, row in enumerate(reader):
            caption = str(row["caption"] or "").strip()
            if not caption:
                raise ValueError(f"AudioCaps row {row_index} has an empty caption")
            rows.append(
                AudioCapsItem(
                    audiocap_id=str(row["audiocap_id"]),
                    youtube_id=str(row["youtube_id"]),
                    start_time=float(row["start_time"]),
                    audio_length=int(row["audio_length"]),
                    wav_path=str(row["wav_path"]),
                    caption=caption,
                    split="all",
                    row_index=row_index,
                )
            )

    expected = sum(AUDIOCAPS_SPLIT_COUNTS.values())
    if validate_counts and len(rows) != expected:
        raise ValueError(f"expected {expected} AudioCaps rows, found {len(rows)}")
    return rows


def load_audiocaps_manifest(
    manifest_path: str | Path,
    *,
    audio_root: str | Path,
    expected_count: int | None = None,
    forbidden_protocol: str | Path | None = None,
) -> list[AudioCapsItem]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    audio_root = Path(audio_root).expanduser().resolve()
    rows: list[AudioCapsItem] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for row_index, row in enumerate(csv.DictReader(handle)):
            caption = str(row.get("caption") or "").strip()
            if not caption:
                raise ValueError(f"AudioCaps manifest row {row_index} has an empty caption")
            raw_path = Path(str(row.get("audio_path") or ""))
            path = raw_path if raw_path.is_absolute() else manifest_path.parent / raw_path
            path = path.resolve()
            if path.parent != audio_root:
                raise ValueError(f"AudioCaps manifest path escapes the required train directory: {path}")
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append(AudioCapsItem(
                audiocap_id=path.stem,
                youtube_id=str(row["youtube_id"]),
                start_time=float(row["start_time"]),
                audio_length=0,
                wav_path=str(path),
                caption=caption,
                split="train",
                row_index=row_index,
            ))
    if expected_count is not None and len(rows) != int(expected_count):
        raise ValueError(f"expected {expected_count} AudioCaps manifest rows, found {len(rows)}")
    if forbidden_protocol is not None:
        import json
        forbidden = set()
        forbidden_youtube_ids = set()
        with Path(forbidden_protocol).open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                youtube_id = str(record["youtube_id"])
                if record.get("start_time") is None:
                    # Some official evaluation exports identify clips only by
                    # YouTube id. Exclude every train segment from that video.
                    forbidden_youtube_ids.add(youtube_id)
                else:
                    forbidden.add((youtube_id, float(record["start_time"])))
        overlap = {(item.youtube_id, item.start_time) for item in rows}.intersection(forbidden)
        youtube_overlap = {
            item.youtube_id for item in rows if item.youtube_id in forbidden_youtube_ids
        }
        if overlap or youtube_overlap:
            raise ValueError(
                "AudioCaps train leaks forbidden evaluation clips: "
                f"exact={len(overlap)} youtube_id={len(youtube_overlap)}"
            )
    return rows


def split_audiocaps_rows(
    rows: list[AudioCapsItem],
    split: AudioCapsSplit,
    *,
    validate_counts: bool = True,
) -> list[AudioCapsItem]:
    normalized = "validation" if split == "valid" else str(split)
    if normalized == "all":
        return [replace(item, split="all") for item in rows]
    if normalized not in AUDIOCAPS_SPLIT_COUNTS:
        raise ValueError(f"unknown AudioCaps split: {split!r}")

    expected = sum(AUDIOCAPS_SPLIT_COUNTS.values())
    if validate_counts and len(rows) != expected:
        raise ValueError(f"cannot apply canonical AudioCaps splits to {len(rows)} rows; expected {expected}")

    start = 0
    for name, count in AUDIOCAPS_SPLIT_COUNTS.items():
        if name == normalized:
            break
        start += count
    stop = start + AUDIOCAPS_SPLIT_COUNTS[normalized]
    if not validate_counts and len(rows) < stop:
        # This branch is useful for tiny fixtures.  Production data keeps the
        # canonical positional split above.
        selected = rows if normalized == "train" else []
    else:
        selected = rows[start:stop]
    return [replace(item, split=normalized) for item in selected]


class AudioCapsDataset:
    """Map-style AudioCaps dataset using the canonical positional splits."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: AudioCapsSplit = "train",
        sample_rate: int = 16_000,
        max_audio_seconds: float | None = 10.0,
        validate_counts: bool = True,
        manifest_path: str | Path | None = None,
        audio_root: str | Path | None = None,
        expected_count: int | None = None,
        forbidden_protocol: str | Path | None = None,
        resampling: dict[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.sample_rate = int(sample_rate)
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.max_audio_seconds = max_audio_seconds
        self.resampling = normalize_resampling_config(resampling)
        self.validate_counts = bool(validate_counts)
        if manifest_path is not None:
            if str(split) != "train":
                raise ValueError("AudioCaps manifest mode supports only the train split")
            if audio_root is None:
                raise ValueError("AudioCaps manifest mode requires audio_root")
            self.rows = load_audiocaps_manifest(
                manifest_path, audio_root=audio_root, expected_count=expected_count,
                forbidden_protocol=forbidden_protocol,
            )
        else:
            rows = load_audiocaps_rows(self.root, validate_counts=self.validate_counts)
            self.rows = split_audiocaps_rows(rows, split, validate_counts=self.validate_counts)
        self.split = "validation" if split == "valid" else str(split)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        item = self.rows[index]
        relative_path = Path(item.wav_path)
        path = relative_path if relative_path.is_absolute() else self.root / "audio" / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"missing AudioCaps audio: {path}")

        wav, source_rate = _load_audio_file(path)
        if source_rate != self.sample_rate:
            wav = resample_waveform(
                wav, source_rate, self.sample_rate, **self.resampling
            )
        wav = trim_waveform(
            wav,
            sample_rate=self.sample_rate,
            max_audio_seconds=self.max_audio_seconds,
        )
        wav = torch.clamp(wav.contiguous(), -1.0, 1.0)
        metadata = {
            "dataset": "audiocaps",
            "utt_id": item.utt_id,
            "split": item.split,
            "audiocap_id": item.audiocap_id,
            "youtube_id": item.youtube_id,
            "start_time": item.start_time,
            "declared_audio_length": item.audio_length,
            "row_index": item.row_index,
            "audio_ref": {
                "path": str(path),
                "wav_path": item.wav_path,
                "source_sample_rate": source_rate,
            },
        }
        return {
            "dataset": "audiocaps",
            "utt_id": item.utt_id,
            "caption": item.caption,
            "split": item.split,
            "wav": wav,
            "wav_len": int(wav.numel()),
            "wav_sample_rate": self.sample_rate,
            "metadata": metadata,
        }


def _load_audio_file(path: Path):
    import numpy as np
    import soundfile as sf
    import torch

    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    data = np.ascontiguousarray(data, dtype="float32")
    return torch.from_numpy(data), int(sample_rate)


collate_audiocaps = collate_audio_text


__all__ = [
    "AUDIOCAPS_SPLIT_COUNTS",
    "AudioCapsDataset",
    "AudioCapsItem",
    "collate_audiocaps",
    "load_audiocaps_rows",
    "split_audiocaps_rows",
]
