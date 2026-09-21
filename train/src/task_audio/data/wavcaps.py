from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import collate_audio_text, normalize_resampling_config, resample_waveform


WAVCAPS_DEFAULT_ROOT = Path("/nativemm2/share/cpfs/wangyujin.wyj/space_edit/WavCaps")
_SOURCES = {
    "AudioSet_SL": ("AudioSet_SL/as_final.json", ".flac", "AudioSet"),
    "BBC_Sound_Effects": ("BBC_Sound_Effects/bbc_final.json", ".flac", None),
    "FreeSound": ("FreeSound/fsd_final.json", ".flac", "FreeSound"),
    "SoundBible": ("SoundBible/sb_final.json", ".flac", None),
}
_MAX_DURATION_ROUNDING_TOLERANCE_SECONDS = 1.0e-3


@dataclass(frozen=True)
class WavCapsItem:
    source: str
    item_id: str
    caption: str
    duration: float
    path: Path
    start_frame: int = 0
    num_frames: int | None = None
    source_sample_rate: int | None = None
    clap_score: float | None = None
    selection_reason: str = "legacy_metadata"
    manifest_utt_id: str | None = None

    @property
    def utt_id(self) -> str:
        return self.manifest_utt_id or f"wavcaps/{self.source}/{self.item_id}"


def _resolve_audio_path(root: Path, source: str, item_id: str, extension: str) -> Path:
    if source == "AudioSet_SL":
        filename = f"{Path(item_id).stem}{extension}"
    else:
        filename = item_id if Path(item_id).suffix else f"{item_id}{extension}"
    return root / "audio" / source / filename


def load_wavcaps_clean_manifest(
    manifest_path: str | Path,
    *,
    root: str | Path,
    require_files: bool = True,
    expected_count: int | None = None,
) -> list[WavCapsItem]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    root = Path(root).expanduser().resolve()
    audio_root = Path(os.path.abspath(root / "audio"))
    items: list[WavCapsItem] = []
    seen: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            utt_id = str(row.get("utt_id") or "").strip()
            caption = str(row.get("caption") or "").strip()
            source = str(row.get("source") or "").strip()
            item_id = str(row.get("item_id") or "").strip()
            if not utt_id or not caption or not source or not item_id:
                raise ValueError(
                    f"WavCaps clean manifest row {row_index} has empty identity/caption fields"
                )
            if utt_id in seen:
                raise ValueError(f"duplicate WavCaps clean utt_id: {utt_id}")
            seen.add(utt_id)
            raw_path = Path(str(row.get("audio_path") or ""))
            # Manifest generation already resolved and audited every source
            # path.  realpath()/Path.resolve() performs lstat calls for every
            # component and makes a 380k-row CPFS manifest take minutes to
            # load, so enforce containment lexically here without filesystem
            # I/O.  __getitem__ still opens the exact file.
            candidate = raw_path if raw_path.is_absolute() else root / raw_path
            path = Path(os.path.abspath(candidate))
            try:
                path.relative_to(audio_root)
            except ValueError as exc:
                raise ValueError(
                    f"WavCaps clean manifest path escapes source audio root: {path}"
                ) from exc
            start_frame = int(row.get("start_frame", 0))
            num_frames = int(row.get("num_frames", 0))
            source_rate = int(row.get("source_sample_rate", 0))
            if start_frame < 0 or num_frames <= 0 or source_rate <= 0:
                raise ValueError(
                    f"WavCaps clean manifest row {row_index} has invalid frame metadata"
                )
            if require_files and not path.is_file():
                raise FileNotFoundError(path)
            score = row.get("clap_score")
            items.append(
                WavCapsItem(
                    source=source,
                    item_id=item_id,
                    caption=caption,
                    duration=float(row.get("original_duration_seconds") or 0.0),
                    path=path,
                    start_frame=start_frame,
                    num_frames=num_frames,
                    source_sample_rate=source_rate,
                    clap_score=None if score is None else float(score),
                    selection_reason=str(row.get("selection_reason") or "clean_manifest"),
                    manifest_utt_id=utt_id,
                )
            )
    if expected_count is not None and len(items) != int(expected_count):
        raise ValueError(
            f"expected {int(expected_count)} WavCaps clean rows, found {len(items)}"
        )
    if not items:
        raise RuntimeError("WavCaps clean manifest contains no usable samples")
    return items


class WavCapsDataset:
    def __init__(
        self,
        root: str | Path = WAVCAPS_DEFAULT_ROOT,
        *,
        sample_rate: int = 16_000,
        max_audio_seconds: float = 20.0,
        blacklist_path: str | Path | None = None,
        require_files: bool = True,
        manifest_path: str | Path | None = None,
        expected_count: int | None = None,
        resampling: dict[str, Any] | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.sample_rate = int(sample_rate)
        self.max_audio_seconds = float(max_audio_seconds)
        self.resampling = normalize_resampling_config(resampling)
        if self.sample_rate <= 0 or self.max_audio_seconds <= 0:
            raise ValueError("sample_rate and max_audio_seconds must be positive")
        if manifest_path is not None:
            self.items = load_wavcaps_clean_manifest(
                manifest_path,
                root=self.root,
                require_files=require_files,
                expected_count=expected_count,
            )
            self.stats = {
                "manifest": {
                    "metadata": len(self.items),
                    "blacklisted": 0,
                    "invalid": 0,
                    "over_duration": 0,
                    "missing": 0,
                    "usable": len(self.items),
                }
            }
            self.blacklist_path = None
            self.manifest_path = Path(manifest_path).expanduser().resolve()
            return
        blacklist_file = Path(blacklist_path) if blacklist_path else (
            self.root / "json_files/blacklist/blacklist_exclude_test_ac.json"
        )
        blacklist_raw = json.loads(blacklist_file.read_text(encoding="utf-8"))
        blacklist = {key: set(map(str, values)) for key, values in blacklist_raw.items()}
        items: list[WavCapsItem] = []
        stats: dict[str, dict[str, int]] = {}
        for source, (metadata_name, extension, blacklist_key) in _SOURCES.items():
            records = json.loads(
                (self.root / "json_files" / metadata_name).read_text(encoding="utf-8")
            )["data"]
            source_stats = {"metadata": 0, "blacklisted": 0, "invalid": 0, "over_duration": 0, "missing": 0, "usable": 0}
            blocked = blacklist.get(blacklist_key or "", set())
            for record in records:
                source_stats["metadata"] += 1
                item_id = str(record["id"])
                if item_id in blocked or f"{item_id}.wav" in blocked:
                    source_stats["blacklisted"] += 1
                    continue
                caption = str(record.get("caption") or "").strip()
                duration = float(record.get("duration") or 0.0)
                if not caption or duration <= 0.0:
                    source_stats["invalid"] += 1
                    continue
                if duration > self.max_audio_seconds:
                    source_stats["over_duration"] += 1
                    continue
                path = _resolve_audio_path(self.root, source, item_id, extension)
                if require_files and not path.is_file():
                    source_stats["missing"] += 1
                    continue
                items.append(WavCapsItem(source, item_id, caption, duration, path))
                source_stats["usable"] += 1
            stats[source] = source_stats
        if not items:
            raise RuntimeError("WavCaps filtering produced no usable samples")
        self.items = items
        self.stats = stats
        self.blacklist_path = blacklist_file.resolve()
        self.manifest_path = None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import numpy as np
        import soundfile as sf
        import torch

        item = self.items[index]
        with sf.SoundFile(str(item.path), mode="r") as handle:
            source_rate = int(handle.samplerate)
            if item.source_sample_rate is not None and source_rate != item.source_sample_rate:
                raise ValueError(
                    f"WavCaps sample-rate changed since manifest creation: {item.path} "
                    f"({source_rate} != {item.source_sample_rate})"
                )
            handle.seek(item.start_frame)
            data = handle.read(
                frames=-1 if item.num_frames is None else item.num_frames,
                dtype="float32",
                always_2d=False,
            )
        if item.num_frames is not None and len(data) != item.num_frames:
            raise ValueError(
                f"WavCaps clean segment short read: {item.path} "
                f"({len(data)} != {item.num_frames})"
            )
        if data.ndim == 2:
            data = data.mean(axis=1)
        wav = torch.from_numpy(np.ascontiguousarray(data, dtype="float32"))
        if source_rate != self.sample_rate:
            wav = resample_waveform(
                wav, source_rate, self.sample_rate, **self.resampling
            )
        actual_samples = int(wav.numel())
        max_samples = int(round(self.max_audio_seconds * self.sample_rate))
        excess_samples = max(0, actual_samples - max_samples)
        tolerance_samples = max(
            1,
            int(
                math.ceil(
                    _MAX_DURATION_ROUNDING_TOLERANCE_SECONDS * self.sample_rate
                )
            ),
        )
        if excess_samples > tolerance_samples:
            actual_seconds = actual_samples / self.sample_rate
            raise ValueError(
                f"WavCaps audio exceeds {self.max_audio_seconds}s: "
                f"{item.path} ({actual_seconds:.6f}s)"
            )
        if excess_samples:
            # Clean manifests derive frame boundaries at the source sample
            # rate. One source-frame rounding tail can become a few samples
            # after resampling (for example 640001@32k -> 882002@44.1k).
            # Trim only this sub-millisecond boundary tail; material duration
            # violations above remain hard errors.
            wav = wav[:max_samples]
        if wav.numel() == 0:
            raise ValueError(f"empty WavCaps audio: {item.path}")
        wav = torch.clamp(wav.contiguous(), -1.0, 1.0)
        metadata = {
            "dataset": "wavcaps", "source": item.source, "item_id": item.item_id,
            "duration": item.duration,
            "clap_score": item.clap_score,
            "selection_reason": item.selection_reason,
            "duration_rounding_trimmed_samples": excess_samples,
            "audio_ref": {
                "path": str(item.path),
                "source_sample_rate": int(source_rate),
                "start_frame": item.start_frame,
                "num_frames": item.num_frames,
            },
        }
        return {
            "dataset": "wavcaps", "utt_id": item.utt_id, "caption": item.caption,
            "split": "train", "wav": wav, "wav_len": int(wav.numel()),
            "wav_sample_rate": self.sample_rate, "metadata": metadata,
        }


collate_wavcaps = collate_audio_text

__all__ = [
    "WAVCAPS_DEFAULT_ROOT",
    "WavCapsDataset",
    "WavCapsItem",
    "collate_wavcaps",
    "load_wavcaps_clean_manifest",
]
