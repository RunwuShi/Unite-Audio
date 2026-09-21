from __future__ import annotations

import csv
import hashlib
import json
import math
import multiprocessing
import os
import random
from collections import Counter, defaultdict
from datetime import datetime
from itertools import groupby
from pathlib import Path
from typing import Any, Iterable, Iterator

from .wavcaps import _SOURCES, _resolve_audio_path


PROTOCOL = "wavcaps_clean_20_v1"
DEFAULT_WINDOW_SECONDS = 20.0
DEFAULT_STRIDE_SECONDS = 20.0
DEFAULT_TAIL_MIN_NEW_SECONDS = 10.0
DEFAULT_MAX_SOURCE_SECONDS = 120.0
DEFAULT_ALIGNMENT_TARGET = 0.80
SILENCE_FILTER_PROTOCOL = "wavcaps_clean_20_v2_full_silence_rms50ms_m60dbfs"
REPLAY_FILTER_PROTOCOL = "wavcaps_clean_20_v2_full_silence_replay_le10s"
DEFAULT_SILENCE_FRAME_MS = 50.0
DEFAULT_SILENCE_THRESHOLD_DBFS = -60.0


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    os.replace(temporary, path)
    return count


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _silence_metrics(
    audio: Any,
    sample_rate: int,
    *,
    frame_ms: float = DEFAULT_SILENCE_FRAME_MS,
    threshold_dbfs: float = DEFAULT_SILENCE_THRESHOLD_DBFS,
) -> dict[str, float | int]:
    """Measure silence and diagnostic levels without changing waveform scale."""
    import numpy as np

    sample_rate = int(sample_rate)
    frame_ms = float(frame_ms)
    threshold_dbfs = float(threshold_dbfs)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if frame_ms <= 0.0:
        raise ValueError("frame_ms must be positive")
    if not math.isfinite(threshold_dbfs) or threshold_dbfs >= 0.0:
        raise ValueError("threshold_dbfs must be finite and negative")
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim == 2:
        values = values.mean(axis=1, dtype=np.float32)
    values = values.reshape(-1)
    if values.size == 0:
        raise ValueError("cannot analyze empty audio")
    if not np.isfinite(values).all():
        raise ValueError("audio contains non-finite values")

    frame_samples = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    starts = np.arange(0, values.size, frame_samples, dtype=np.int64)
    lengths = np.minimum(frame_samples, values.size - starts).astype(np.float64)
    squared = np.square(values.astype(np.float64))
    frame_power = np.add.reduceat(squared, starts) / lengths
    silence_power = 10.0 ** (threshold_dbfs / 10.0)
    silent_frames = int(np.count_nonzero(frame_power < silence_power))
    frame_count = int(starts.size)
    absolute = np.abs(values.astype(np.float64))
    return {
        "silence_ratio": silent_frames / frame_count,
        "silent_frames": silent_frames,
        "frame_count": frame_count,
        "peak": float(absolute.max()),
        "rms": float(np.sqrt(squared.mean())),
        "clipping_ratio": float(np.mean(absolute >= 0.999)),
        "dc_offset": float(values.astype(np.float64).mean()),
    }


def _analyze_silence_file(
    payload: tuple[str, list[dict[str, Any]], float, float],
) -> list[tuple[dict[str, Any], dict[str, float | int]]]:
    """Analyze all manifest windows from one source file with one file open."""
    import soundfile as sf

    audio_path, rows, frame_ms, threshold_dbfs = payload
    output: list[tuple[dict[str, Any], dict[str, float | int]]] = []
    with sf.SoundFile(audio_path, mode="r") as handle:
        source_rate = int(handle.samplerate)
        for row in rows:
            expected_rate = int(row["source_sample_rate"])
            if source_rate != expected_rate:
                raise ValueError(
                    f"sample-rate changed for {audio_path}: "
                    f"{source_rate} != {expected_rate}"
                )
            start_frame = int(row["start_frame"])
            num_frames = int(row["num_frames"])
            handle.seek(start_frame)
            audio = handle.read(
                frames=num_frames,
                dtype="float32",
                always_2d=False,
            )
            if len(audio) != num_frames:
                raise ValueError(
                    f"short read for {row['utt_id']}: {len(audio)} != {num_frames}"
                )
            output.append(
                (
                    row,
                    _silence_metrics(
                        audio,
                        source_rate,
                        frame_ms=frame_ms,
                        threshold_dbfs=threshold_dbfs,
                    ),
                )
            )
    return output


def filter_manifest_silence(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    frame_ms: float = DEFAULT_SILENCE_FRAME_MS,
    threshold_dbfs: float = DEFAULT_SILENCE_THRESHOLD_DBFS,
    workers: int = 16,
    expected_count: int | None = None,
    log_every_files: int = 1_000,
) -> dict[str, Any]:
    """Derive an immutable silence-filtered manifest without altering audio."""
    import numpy as np

    manifest_path = Path(manifest_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_manifest = output_dir / "manifest.jsonl"
    rejected_path = output_dir / "rejected_silence.jsonl"
    if manifest_path == output_manifest:
        raise ValueError("silence filtering must not overwrite its input manifest")
    frame_ms = float(frame_ms)
    threshold_dbfs = float(threshold_dbfs)
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be positive")

    def grouped_payloads() -> Iterator[tuple[str, list[dict[str, Any]], float, float]]:
        rows = iter_jsonl(manifest_path)
        for audio_path, grouped_rows in groupby(rows, key=lambda row: str(row["audio_path"])):
            yield audio_path, list(grouped_rows), frame_ms, threshold_dbfs

    payloads = grouped_payloads()
    pool = None
    if workers == 1:
        analyzed = map(_analyze_silence_file, payloads)
    else:
        pool = multiprocessing.get_context("spawn").Pool(processes=workers)
        analyzed = pool.imap(_analyze_silence_file, payloads, chunksize=8)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_tmp = output_manifest.with_suffix(".jsonl.tmp")
    rejected_tmp = rejected_path.with_suffix(".jsonl.tmp")
    counts: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    diagnostics: dict[str, list[float]] = defaultdict(list)
    seen_ids: set[str] = set()
    completed = False
    try:
        with output_tmp.open("w", encoding="utf-8") as retained_handle, rejected_tmp.open(
            "w", encoding="utf-8"
        ) as rejected_handle:
            for file_index, file_results in enumerate(analyzed, start=1):
                counts["source_files"] += 1
                for row, metrics in file_results:
                    utt_id = str(row["utt_id"])
                    if utt_id in seen_ids:
                        raise ValueError(f"duplicate manifest utt_id: {utt_id}")
                    seen_ids.add(utt_id)
                    source = str(row["source"])
                    counts["input_rows"] += 1
                    source_counts[source]["input_rows"] += 1
                    for key in ("silence_ratio", "peak", "rms", "clipping_ratio", "dc_offset"):
                        diagnostics[key].append(float(metrics[key]))
                    silence_ratio = float(metrics["silence_ratio"])
                    all_frames_silent = int(metrics["silent_frames"]) == int(
                        metrics["frame_count"]
                    )
                    if all_frames_silent:
                        rejected = dict(row)
                        rejected.update(
                            {
                                "silence_filter_protocol": SILENCE_FILTER_PROTOCOL,
                                "silence_ratio": silence_ratio,
                                "selection_reason_before_silence_filter": row.get(
                                    "selection_reason"
                                ),
                                "selection_reason": "silence_rejected",
                            }
                        )
                        rejected_handle.write(
                            json.dumps(
                                rejected,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                        counts["rejected_silence"] += 1
                        source_counts[source]["rejected_silence"] += 1
                        continue
                    retained = dict(row)
                    retained["protocol"] = SILENCE_FILTER_PROTOCOL
                    retained["silence_filter_protocol"] = SILENCE_FILTER_PROTOCOL
                    retained["silence_ratio"] = silence_ratio
                    retained_handle.write(
                        json.dumps(
                            retained,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    counts["retained"] += 1
                    source_counts[source]["retained"] += 1
                if file_index % max(1, log_every_files) == 0:
                    print(
                        "[WavCaps silence] "
                        f"files={file_index} rows={counts['input_rows']} "
                        f"retained={counts['retained']} "
                        f"rejected={counts['rejected_silence']}",
                        flush=True,
                    )
        if expected_count is not None and counts["input_rows"] != int(expected_count):
            raise RuntimeError(
                f"input manifest count mismatch: {counts['input_rows']} != {expected_count}"
            )
        if counts["retained"] + counts["rejected_silence"] != counts["input_rows"]:
            raise RuntimeError("silence-filter count mismatch")
        os.replace(output_tmp, output_manifest)
        os.replace(rejected_tmp, rejected_path)
        completed = True
    except BaseException:
        output_tmp.unlink(missing_ok=True)
        rejected_tmp.unlink(missing_ok=True)
        raise
    finally:
        if pool is not None:
            if completed:
                pool.close()
            else:
                pool.terminate()
            pool.join()

    quantiles = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    diagnostic_summary: dict[str, Any] = {}
    for name, values in diagnostics.items():
        array = np.asarray(values, dtype=np.float64)
        diagnostic_summary[name] = {
            "mean": float(array.mean()),
            "quantiles": {
                f"{quantile:.2f}": float(np.quantile(array, quantile))
                for quantile in quantiles
            },
        }
    version = {
        "protocol": SILENCE_FILTER_PROTOCOL,
        "created_at": now_iso(),
        "parent_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "manifest_path": str(output_manifest),
        "manifest_sha256": sha256_file(output_manifest),
        "rejected_path": str(rejected_path),
        "rejected_sha256": sha256_file(rejected_path),
        "silence_filter": {
            "frame_ms": frame_ms,
            "threshold_dbfs": threshold_dbfs,
            "comparison": "frame_rms_strictly_below_threshold",
            "rejection_comparison": "all_real_audio_frames_are_silent",
            "normalization": "none",
            "padding_included": False,
        },
        "counts": dict(counts),
        "source_counts": {key: dict(value) for key, value in source_counts.items()},
        "diagnostics": diagnostic_summary,
    }
    _atomic_json(output_dir / "VERSION.json", version)
    readme = "# WavCaps Clean_20 v2 Full Silence\n\n"
    readme += "Derived from the immutable Clean_20 v1 manifest without modifying source audio.\n\n"
    readme += "## Silence rule\n\n"
    readme += f"- Frame length: {frame_ms:g} ms, non-overlapping, no padding.\n"
    readme += f"- A frame is silent when raw waveform RMS is below {threshold_dbfs:g} dBFS.\n"
    readme += "- A window is rejected only when every real-audio frame is silent.\n"
    readme += "- Any window with at least one non-silent frame is retained, including short events.\n"
    readme += "- No peak, RMS, LUFS, or gain normalization is applied.\n\n"
    readme += "## Counts\n\n"
    readme += f"- Input rows: {counts['input_rows']:,}\n"
    readme += f"- Retained rows: {counts['retained']:,}\n"
    readme += f"- Silence-rejected rows: {counts['rejected_silence']:,}\n"
    readme += "\nExact hashes, source counts, and diagnostics are recorded in `VERSION.json`.\n"
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return version


def filter_manifest_max_duration(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    max_duration_seconds: float = 10.0,
    expected_count: int | None = None,
) -> dict[str, Any]:
    """Derive an immutable replay manifest without altering source audio."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_manifest = output_dir / "manifest.jsonl"
    if manifest_path == output_manifest:
        raise ValueError("duration filtering must not overwrite its input manifest")
    maximum = float(max_duration_seconds)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("max_duration_seconds must be finite and positive")

    counts: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    seen_ids: set[str] = set()

    def retained_rows() -> Iterator[dict[str, Any]]:
        for row in iter_jsonl(manifest_path):
            counts["input_rows"] += 1
            source = str(row["source"])
            source_counts[source]["input_rows"] += 1
            utt_id = str(row["utt_id"])
            if utt_id in seen_ids:
                raise ValueError(f"duplicate manifest utt_id: {utt_id}")
            seen_ids.add(utt_id)
            raw_duration = row.get("segment_duration_seconds", row.get("duration"))
            if raw_duration is None:
                raise ValueError(f"manifest row has no duration: {utt_id}")
            duration = float(raw_duration)
            if not math.isfinite(duration) or duration <= 0.0:
                raise ValueError(f"invalid duration for {utt_id}: {duration}")
            if duration > maximum:
                counts["excluded_over_duration"] += 1
                source_counts[source]["excluded_over_duration"] += 1
                continue
            retained = dict(row)
            retained["parent_protocol"] = row.get("protocol")
            retained["protocol"] = REPLAY_FILTER_PROTOCOL
            retained["replay_filter_protocol"] = REPLAY_FILTER_PROTOCOL
            retained["replay_max_duration_seconds"] = maximum
            counts["retained"] += 1
            source_counts[source]["retained"] += 1
            yield retained

    output_dir.mkdir(parents=True, exist_ok=True)
    written = _atomic_jsonl(output_manifest, retained_rows())
    if expected_count is not None and counts["input_rows"] != int(expected_count):
        output_manifest.unlink(missing_ok=True)
        raise RuntimeError(
            f"input manifest count mismatch: {counts['input_rows']} != {expected_count}"
        )
    if written != counts["retained"]:
        output_manifest.unlink(missing_ok=True)
        raise RuntimeError("duration-filter count mismatch")
    version = {
        "protocol": REPLAY_FILTER_PROTOCOL,
        "created_at": now_iso(),
        "parent_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "manifest_path": str(output_manifest),
        "manifest_sha256": sha256_file(output_manifest),
        "duration_filter": {
            "field": "segment_duration_seconds",
            "max_duration_seconds": maximum,
            "comparison": "less_than_or_equal",
            "audio_modified": False,
        },
        "counts": dict(counts),
        "source_counts": {key: dict(value) for key, value in source_counts.items()},
    }
    _atomic_json(output_dir / "VERSION.json", version)
    readme = "# WavCaps Clean_20 v2 Stage2 Replay <=10s\n\n"
    readme += "Derived from the full-silence-filtered v2 manifest.\n\n"
    readme += f"- Keep segments with duration <= {maximum:g} seconds.\n"
    readme += "- Source audio is not copied, cropped, normalized, or re-encoded.\n"
    readme += f"- Input rows: {counts['input_rows']:,}\n"
    readme += f"- Retained rows: {counts['retained']:,}\n"
    readme += f"- Excluded over-duration rows: {counts['excluded_over_duration']:,}\n"
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return version


def window_start_frames(
    total_frames: int,
    sample_rate: int,
    *,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    stride_seconds: float = DEFAULT_STRIDE_SECONDS,
    tail_min_new_seconds: float = DEFAULT_TAIL_MIN_NEW_SECONDS,
) -> list[int]:
    if total_frames <= 0 or sample_rate <= 0:
        return []
    window = int(round(window_seconds * sample_rate))
    stride = int(round(stride_seconds * sample_rate))
    tail_min_new = int(round(tail_min_new_seconds * sample_rate))
    if min(window, stride) <= 0 or tail_min_new < 0:
        raise ValueError("window, stride, and tail policy must be positive")
    if total_frames <= window:
        return [0]
    starts = list(range(0, total_frames - window + 1, stride))
    last_end = starts[-1] + window
    if total_frames - last_end >= tail_min_new:
        end_aligned = total_frames - window
        if end_aligned != starts[-1]:
            starts.append(end_aligned)
    return starts


def load_blacklist_union(paths: Iterable[str | Path]) -> tuple[dict[str, set[str]], list[dict[str, str]]]:
    union: dict[str, set[str]] = defaultdict(set)
    provenance: list[dict[str, str]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key, values in payload.items():
            union[str(key)].update(map(str, values))
        provenance.append({"path": str(path), "sha256": sha256_file(path)})
    return dict(union), provenance


def _is_blocked(item_id: str, blocked: set[str]) -> bool:
    return item_id in blocked or f"{item_id}.wav" in blocked


def build_candidates(
    *,
    root: str | Path,
    output_dir: str | Path,
    blacklist_paths: Iterable[str | Path],
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    stride_seconds: float = DEFAULT_STRIDE_SECONDS,
    tail_min_new_seconds: float = DEFAULT_TAIL_MIN_NEW_SECONDS,
    max_source_seconds: float = DEFAULT_MAX_SOURCE_SECONDS,
) -> dict[str, Any]:
    import soundfile as sf

    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    candidates_path = output_dir / "candidates.jsonl"
    blacklist, blacklist_provenance = load_blacklist_union(blacklist_paths)
    stats: Counter[str] = Counter()
    source_stats: dict[str, Counter[str]] = defaultdict(Counter)
    metadata_provenance: list[dict[str, str]] = []

    def rows() -> Iterator[dict[str, Any]]:
        for source, (metadata_name, extension, blacklist_key) in _SOURCES.items():
            metadata_path = root / "json_files" / metadata_name
            metadata_provenance.append(
                {"path": str(metadata_path), "sha256": sha256_file(metadata_path)}
            )
            records = json.loads(metadata_path.read_text(encoding="utf-8"))["data"]
            blocked = blacklist.get(blacklist_key or "", set())
            for record in records:
                stats["metadata"] += 1
                source_stats[source]["metadata"] += 1
                item_id = str(record.get("id") or "").strip()
                caption = str(record.get("caption") or "").strip()
                if not item_id or not caption:
                    stats["invalid_metadata"] += 1
                    source_stats[source]["invalid_metadata"] += 1
                    continue
                if _is_blocked(item_id, blocked):
                    stats["blacklisted"] += 1
                    source_stats[source]["blacklisted"] += 1
                    continue
                path = _resolve_audio_path(root, source, item_id, extension).resolve()
                if not path.is_file():
                    stats["missing"] += 1
                    source_stats[source]["missing"] += 1
                    continue
                try:
                    info = sf.info(str(path))
                except Exception:
                    stats["unreadable"] += 1
                    source_stats[source]["unreadable"] += 1
                    continue
                total_frames = int(info.frames)
                sample_rate = int(info.samplerate)
                if total_frames <= 0 or sample_rate <= 0:
                    stats["empty"] += 1
                    source_stats[source]["empty"] += 1
                    continue
                duration = total_frames / sample_rate
                if duration > max_source_seconds + (1.0 / sample_rate):
                    stats["over_120s"] += 1
                    source_stats[source]["over_120s"] += 1
                    continue
                is_long = duration > window_seconds + (1.0 / sample_rate)
                starts = window_start_frames(
                    total_frames,
                    sample_rate,
                    window_seconds=window_seconds,
                    stride_seconds=stride_seconds,
                    tail_min_new_seconds=tail_min_new_seconds,
                )
                window_frames = int(round(window_seconds * sample_rate))
                stats["usable_originals"] += 1
                source_stats[source]["usable_originals"] += 1
                for start_frame in starts:
                    num_frames = window_frames if is_long else total_frames
                    reason = "long_pending_clap" if is_long else "short_passthrough"
                    utt_id = (
                        f"wavcaps_clean20/{source}/{item_id}/"
                        f"{start_frame:012d}-{num_frames:012d}"
                    )
                    stats["candidates"] += 1
                    stats[reason] += 1
                    source_stats[source]["candidates"] += 1
                    source_stats[source][reason] += 1
                    yield {
                        "protocol": PROTOCOL,
                        "utt_id": utt_id,
                        "source": source,
                        "item_id": item_id,
                        "caption": caption,
                        "audio_path": str(path),
                        "start_frame": start_frame,
                        "num_frames": num_frames,
                        "source_sample_rate": sample_rate,
                        "original_num_frames": total_frames,
                        "original_duration_seconds": duration,
                        "segment_duration_seconds": num_frames / sample_rate,
                        "metadata_duration_seconds": float(record.get("duration") or 0.0),
                        "clap_score": None,
                        "selection_reason": reason,
                    }

    count = _atomic_jsonl(candidates_path, rows())
    if count != stats["candidates"]:
        raise RuntimeError("candidate writer count mismatch")
    summary = {
        "protocol": PROTOCOL,
        "created_at": now_iso(),
        "source_root": str(root),
        "candidates_path": str(candidates_path),
        "window_seconds": window_seconds,
        "stride_seconds": stride_seconds,
        "tail_min_new_seconds": tail_min_new_seconds,
        "max_source_seconds": max_source_seconds,
        "blacklists": blacklist_provenance,
        "metadata": metadata_provenance,
        "counts": dict(stats),
        "source_counts": {key: dict(value) for key, value in source_stats.items()},
    }
    _atomic_json(output_dir / "candidate_summary.json", summary)
    return summary


def _load_clap_segment(row: dict[str, Any]):
    import numpy as np
    import soundfile as sf

    with sf.SoundFile(row["audio_path"], mode="r") as handle:
        handle.seek(int(row["start_frame"]))
        data = handle.read(
            frames=int(row["num_frames"]), dtype="float32", always_2d=False
        )
        source_rate = int(handle.samplerate)
    if len(data) != int(row["num_frames"]):
        raise ValueError("short read")
    if data.ndim == 2:
        data = data.mean(axis=1)
    data = np.asarray(data, dtype=np.float32)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if not math.isfinite(peak) or peak <= 1e-8:
        return None
    return np.ascontiguousarray(data, dtype=np.float32), source_rate


def _prepare_clap_audio_batch(native_audio, device: str, target_rate: int = 48_000):
    import numpy as np
    import torch
    import torch.nn.functional as torch_functional
    import torchaudio.functional as audio_functional

    if not native_audio:
        raise ValueError("cannot prepare an empty CLAP audio batch")
    target_frames = int(round(DEFAULT_WINDOW_SECONDS * target_rate))
    output = torch.empty(
        (len(native_audio), target_frames), dtype=torch.float32, device=device
    )
    grouped: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for index, (audio, source_rate) in enumerate(native_audio):
        grouped[int(source_rate)].append((index, audio))
    for source_rate, entries in grouped.items():
        max_frames = max(len(audio) for _, audio in entries)
        padded = np.zeros((len(entries), max_frames), dtype=np.float32)
        for group_index, (_, audio) in enumerate(entries):
            padded[group_index, : len(audio)] = audio
        tensor = torch.from_numpy(padded).to(device, non_blocking=True)
        if source_rate != target_rate:
            tensor = audio_functional.resample(tensor, source_rate, target_rate)
        if tensor.shape[1] < target_frames:
            tensor = torch_functional.pad(tensor, (0, target_frames - tensor.shape[1]))
        else:
            tensor = tensor[:, :target_frames]
        peak = tensor.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
        tensor = tensor * (10.0 ** (-1.0 / 20.0) / peak)
        tensor = (tensor.clamp(-1.0, 1.0) * 32767.0).to(torch.int16).float()
        tensor = tensor / 32767.0
        for group_index, (output_index, _) in enumerate(entries):
            output[output_index].copy_(tensor[group_index])
    return output


def _load_clap_model(checkpoint: Path, device: str):
    import laion_clap
    from clap_module.factory import load_state_dict

    model = laion_clap.CLAP_Module(enable_fusion=True, device=device)
    state = load_state_dict(str(checkpoint))
    state.pop("text_branch.embeddings.position_ids", None)
    model.model.load_state_dict(state)
    model.eval()
    return model


def _stable_feature_seed(utt_id: str) -> int:
    digest = hashlib.sha256(utt_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _get_audio_embeddings_with_gpu_features(model, audio_tensor, rows, device: str):
    """Vectorize official CLAP fusion preprocessing on the model device."""
    import numpy as np
    import torch
    import torchaudio
    import torchvision

    audio_tensor = audio_tensor.to(device, non_blocking=True)
    audio_cfg = model.model_cfg["audio_cfg"]
    cache = getattr(model, "_clean20_feature_transforms", None)
    if cache is None:
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=audio_cfg["sample_rate"],
            n_fft=audio_cfg["window_size"],
            win_length=audio_cfg["window_size"],
            hop_length=audio_cfg["hop_size"],
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm=None,
            onesided=True,
            n_mels=audio_cfg["mel_bins"],
            f_min=audio_cfg["fmin"],
            f_max=audio_cfg["fmax"],
        ).to(device)
        amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=None).to(device)
        chunk_frames = 480_000 // audio_cfg["hop_size"] + 1
        resize = torchvision.transforms.Resize(
            size=[chunk_frames, audio_cfg["mel_bins"]]
        )
        cache = (mel_transform, amplitude_to_db, resize, chunk_frames)
        model._clean20_feature_transforms = cache
    mel_transform, amplitude_to_db, resize, chunk_frames = cache
    # get_mel() in LAION-CLAP returns (time, mel); its transforms accept
    # leading batch dimensions, so compute all 64 windows with one STFT call.
    mel = amplitude_to_db(mel_transform(audio_tensor)).transpose(1, 2)
    total_frames = int(mel.shape[1])
    possible = np.arange(0, total_frames - chunk_frames + 1)
    ranges = np.array_split(possible, 3)
    if any(len(values) == 0 for values in ranges):
        raise RuntimeError("CLAP fusion ranges are empty for a 20-second window")
    mel_shrink = resize(mel.unsqueeze(1)).squeeze(1)
    audio_input = []
    overflow = int(audio_tensor.shape[1]) - 480_000
    if overflow < 0:
        raise RuntimeError("CLAP batch contains audio shorter than 10 seconds")
    for index, row in enumerate(rows):
        # RandomState uses the same MT19937 sequence as np.random.seed plus
        # np.random.choice/randint in the upstream implementation.
        rng = np.random.RandomState(_stable_feature_seed(str(row["utt_id"])))
        front = int(rng.choice(ranges[0]))
        middle = int(rng.choice(ranges[1]))
        back = int(rng.choice(ranges[2]))
        crop = int(rng.randint(0, overflow + 1))
        mel_fusion = torch.stack(
            [
                mel_shrink[index],
                mel[index, front : front + chunk_frames],
                mel[index, middle : middle + chunk_frames],
                mel[index, back : back + chunk_frames],
            ],
            dim=0,
        )
        audio_input.append(
            {
                "mel_fusion": mel_fusion,
                "longer": torch.tensor([True], device=device),
                "waveform": audio_tensor[index, crop : crop + 480_000],
            }
        )
    return model.model.get_audio_embedding(audio_input)


def score_candidates(
    *,
    candidates_path: str | Path,
    output_path: str | Path,
    checkpoint: str | Path,
    shard_index: int = 0,
    num_shards: int = 1,
    batch_size: int = 16,
    device: str = "cuda:0",
    log_every: int = 100,
) -> dict[str, int]:
    import numpy as np
    import torch

    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    seen_existing: set[str] = set()
    successful_rows: list[dict[str, Any]] = []
    discarded_failures = 0
    if output_path.is_file():
        for row in iter_jsonl(output_path):
            utt_id = str(row["utt_id"])
            if utt_id in seen_existing:
                raise ValueError(f"duplicate resumed score: {utt_id}")
            seen_existing.add(utt_id)
            if row.get("clap_score") is None or row.get("status") == "error":
                discarded_failures += 1
                continue
            completed.add(utt_id)
            successful_rows.append(row)
        if discarded_failures:
            _atomic_jsonl(output_path, successful_rows)
    selected: list[dict[str, Any]] = []
    long_index = 0
    for row in iter_jsonl(candidates_path):
        if row["selection_reason"] != "long_pending_clap":
            continue
        if long_index % num_shards == shard_index and row["utt_id"] not in completed:
            selected.append(row)
        long_index += 1
    model = _load_clap_model(Path(checkpoint).expanduser().resolve(), device)
    counters: Counter[str] = Counter(
        resumed=len(completed),
        pending=len(selected),
        retried_failures=discarded_failures,
    )
    with output_path.open("a", encoding="utf-8") as handle:
        for batch_start in range(0, len(selected), batch_size):
            raw_batch = selected[batch_start : batch_start + batch_size]
            active_rows: list[dict[str, Any]] = []
            audio_values: list[tuple[np.ndarray, int]] = []
            for row in raw_batch:
                try:
                    audio = _load_clap_segment(row)
                except Exception as exc:
                    result = {
                        "utt_id": row["utt_id"],
                        "clap_score": None,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    counters["error"] += 1
                    continue
                if audio is None:
                    handle.write(
                        json.dumps(
                            {
                                "utt_id": row["utt_id"],
                                "clap_score": -1.0,
                                "status": "silent",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    counters["silent"] += 1
                    continue
                active_rows.append(row)
                audio_values.append(audio)
            if active_rows:
                audio_tensor = _prepare_clap_audio_batch(audio_values, device)
                texts = [str(row["caption"]) for row in active_rows]
                with torch.inference_mode():
                    text_embeddings = model.get_text_embedding(texts, use_tensor=True)
                    audio_embeddings = _get_audio_embeddings_with_gpu_features(
                        model, audio_tensor, active_rows, device
                    )
                    similarities = torch.nn.functional.cosine_similarity(
                        audio_embeddings, text_embeddings, dim=1, eps=1e-8
                    )
                for row, score in zip(active_rows, similarities.detach().cpu().tolist()):
                    handle.write(
                        json.dumps(
                            {
                                "utt_id": row["utt_id"],
                                "clap_score": float(score),
                                "status": "ok",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    counters["ok"] += 1
            handle.flush()
            processed = min(batch_start + batch_size, len(selected))
            if processed % max(1, log_every) < batch_size:
                print(
                    f"[WavCaps Clean20] shard={shard_index}/{num_shards} "
                    f"processed={processed}/{len(selected)} ok={counters['ok']} "
                    f"silent={counters['silent']} error={counters['error']}",
                    flush=True,
                )
    return dict(counters)


def load_scores(scores_dir: str | Path) -> dict[str, dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}
    paths = sorted(Path(scores_dir).glob("part-*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no score shards found under {scores_dir}")
    for path in paths:
        for row in iter_jsonl(path):
            utt_id = str(row["utt_id"])
            if utt_id in scores:
                raise ValueError(f"duplicate score across shards: {utt_id}")
            scores[utt_id] = row
    return scores


def summarize_scores(
    *,
    candidates_path: str | Path,
    scores_dir: str | Path,
    output_dir: str | Path,
    audit_seed: int = 1234,
    audit_bins: int = 20,
    audit_per_bin: int = 20,
) -> dict[str, Any]:
    import numpy as np

    output_dir = Path(output_dir).expanduser().resolve()
    scores = load_scores(scores_dir)
    candidate_by_id: dict[str, dict[str, Any]] = {}
    long_ids: set[str] = set()
    grouped: dict[str, list[float]] = defaultdict(list)
    failures: list[str] = []
    for row in iter_jsonl(candidates_path):
        if row["selection_reason"] != "long_pending_clap":
            continue
        utt_id = str(row["utt_id"])
        long_ids.add(utt_id)
        candidate_by_id[utt_id] = row
        score_row = scores.get(utt_id)
        if score_row is None or score_row.get("clap_score") is None:
            failures.append(utt_id)
            continue
        score = float(score_row["clap_score"])
        grouped["global"].append(score)
        grouped[f"source:{row['source']}"] .append(score)
        duration = float(row["original_duration_seconds"])
        duration_band = "20-40" if duration <= 40 else "40-60" if duration <= 60 else "60-120"
        grouped[f"duration:{duration_band}"].append(score)
    extras = set(scores).difference(long_ids)
    if failures or extras or len(scores) != len(long_ids):
        raise RuntimeError(
            "score set is incomplete: "
            f"long={len(long_ids)} scores={len(scores)} missing_or_error={len(failures)} "
            f"extras={len(extras)}"
        )
    quantiles = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    distributions: dict[str, Any] = {}
    for name, values in grouped.items():
        array = np.asarray(values, dtype=np.float64)
        distributions[name] = {
            "count": int(array.size),
            "mean": float(array.mean()),
            "std": float(array.std()),
            "quantiles": {
                f"{q:.2f}": float(np.quantile(array, q)) for q in quantiles
            },
        }
    global_scores = np.asarray(grouped["global"], dtype=np.float64)
    threshold_grid = np.round(np.arange(-0.10, 0.501, 0.01), 2)
    retention = [
        {
            "threshold": float(value),
            "count": int((global_scores >= value).sum()),
            "ratio": float((global_scores >= value).mean()),
        }
        for value in threshold_grid
    ]
    ordered_ids = sorted(long_ids, key=lambda item: float(scores[item]["clap_score"]))
    bins = [list(part) for part in np.array_split(np.asarray(ordered_ids, dtype=object), audit_bins)]
    rng = random.Random(audit_seed)
    audit_rows: list[dict[str, Any]] = []
    for bin_index, values in enumerate(bins):
        population = len(values)
        chosen = rng.sample(values, min(audit_per_bin, population))
        weight = population / max(1, len(chosen))
        for utt_id in chosen:
            row = candidate_by_id[str(utt_id)]
            audit_rows.append(
                {
                    "utt_id": utt_id,
                    "score": float(scores[str(utt_id)]["clap_score"]),
                    "score_bin": bin_index,
                    "weight": weight,
                    "source": row["source"],
                    "item_id": row["item_id"],
                    "caption": row["caption"],
                    "audio_path": row["audio_path"],
                    "start_seconds": int(row["start_frame"]) / int(row["source_sample_rate"]),
                    "duration_seconds": row["segment_duration_seconds"],
                    "label": "",
                    "notes": "",
                }
            )
    audit_path = output_dir / "audit_400.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = audit_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(sorted(audit_rows, key=lambda row: float(row["score"])))
    os.replace(temporary, audit_path)
    summary = {
        "protocol": PROTOCOL,
        "created_at": now_iso(),
        "long_candidate_count": len(long_ids),
        "distributions": distributions,
        "retention_curve": retention,
        "audit": {
            "path": str(audit_path),
            "seed": audit_seed,
            "bins": audit_bins,
            "per_bin": audit_per_bin,
        },
    }
    _atomic_json(output_dir / "score_stats.json", summary)
    return summary


def calibrated_threshold(
    labels_path: str | Path,
    *,
    alignment_target: float = DEFAULT_ALIGNMENT_TARGET,
    min_labeled_above: int | None = None,
) -> tuple[float, dict[str, Any]]:
    labels: list[tuple[float, float, int]] = []
    with Path(labels_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_label = str(row.get("label") or "").strip().lower()
            if raw_label not in {"0", "1", "false", "true", "no", "yes"}:
                continue
            label = int(raw_label in {"1", "true", "yes"})
            labels.append((float(row["score"]), float(row.get("weight") or 1.0), label))
    if min_labeled_above is None:
        min_labeled_above = min(40, max(10, len(labels) // 4))
    if len(labels) < min_labeled_above:
        raise ValueError(
            f"need at least {min_labeled_above} audit labels, found {len(labels)}"
        )
    low = math.floor(min(value[0] for value in labels) * 100) / 100
    high = math.ceil(max(value[0] for value in labels) * 100) / 100
    candidates: list[tuple[float, float, int]] = []
    threshold = low
    while threshold <= high + 1e-9:
        above = [value for value in labels if value[0] >= threshold]
        if len(above) >= min_labeled_above:
            total_weight = sum(value[1] for value in above)
            precision = sum(value[1] * value[2] for value in above) / total_weight
            if precision >= alignment_target:
                candidates.append((threshold, precision, len(above)))
        threshold = round(threshold + 0.01, 10)
    if not candidates:
        raise RuntimeError(
            f"no audited threshold reaches alignment target {alignment_target:.1%}"
        )
    threshold, precision, count = min(candidates, key=lambda value: value[0])
    return threshold, {
        "alignment_target": alignment_target,
        "estimated_alignment": precision,
        "labeled_above": count,
        "total_labels": len(labels),
    }


def finalize_manifest(
    *,
    candidates_path: str | Path,
    scores_dir: str | Path,
    output_dir: str | Path,
    threshold: float | None = None,
    labels_path: str | Path | None = None,
    checkpoint: str | Path | None = None,
    alignment_target: float = DEFAULT_ALIGNMENT_TARGET,
    calibration_mode: str = "explicit",
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    scores = load_scores(scores_dir)
    calibration: dict[str, Any] = {"mode": str(calibration_mode)}
    if threshold is None:
        if labels_path is None:
            raise ValueError("finalize requires either threshold or labels_path")
        threshold, calibration = calibrated_threshold(
            labels_path, alignment_target=alignment_target
        )
        calibration["mode"] = "audit_calibrated"
        calibration["labels_path"] = str(Path(labels_path).resolve())
    threshold = float(threshold)
    counts: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    seen_scores: set[str] = set()

    def retained_rows() -> Iterator[dict[str, Any]]:
        for row in iter_jsonl(candidates_path):
            source = str(row["source"])
            counts["candidates"] += 1
            source_counts[source]["candidates"] += 1
            if row["selection_reason"] == "short_passthrough":
                counts["short_passthrough"] += 1
                counts["retained"] += 1
                source_counts[source]["short_passthrough"] += 1
                source_counts[source]["retained"] += 1
                yield row
                continue
            utt_id = str(row["utt_id"])
            score_row = scores.get(utt_id)
            if score_row is None or score_row.get("clap_score") is None:
                raise RuntimeError(f"missing valid CLAP score for {utt_id}")
            seen_scores.add(utt_id)
            score = float(score_row["clap_score"])
            counts["long_scored"] += 1
            source_counts[source]["long_scored"] += 1
            if score < threshold:
                counts["long_rejected"] += 1
                source_counts[source]["long_rejected"] += 1
                continue
            kept = dict(row)
            kept["clap_score"] = score
            kept["selection_reason"] = "clap_pass"
            counts["long_retained"] += 1
            counts["retained"] += 1
            source_counts[source]["long_retained"] += 1
            source_counts[source]["retained"] += 1
            yield kept

    manifest_path = output_dir / "manifest.jsonl"
    written = _atomic_jsonl(manifest_path, retained_rows())
    extras = set(scores).difference(seen_scores)
    if extras:
        manifest_path.unlink(missing_ok=True)
        raise RuntimeError(f"score shards contain {len(extras)} non-candidate rows")
    if written != counts["retained"]:
        raise RuntimeError("final manifest count mismatch")
    version = {
        "protocol": PROTOCOL,
        "created_at": now_iso(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "threshold": threshold,
        "calibration": calibration,
        "checkpoint": None
        if checkpoint is None
        else {
            "path": str(Path(checkpoint).expanduser().resolve()),
            "sha256": sha256_file(checkpoint),
        },
        "counts": dict(counts),
        "source_counts": {key: dict(value) for key, value in source_counts.items()},
    }
    _atomic_json(output_dir / "VERSION.json", version)
    return version


__all__ = [
    "DEFAULT_ALIGNMENT_TARGET",
    "DEFAULT_SILENCE_FRAME_MS",
    "DEFAULT_SILENCE_THRESHOLD_DBFS",
    "PROTOCOL",
    "REPLAY_FILTER_PROTOCOL",
    "SILENCE_FILTER_PROTOCOL",
    "build_candidates",
    "calibrated_threshold",
    "filter_manifest_silence",
    "filter_manifest_max_duration",
    "finalize_manifest",
    "iter_jsonl",
    "load_scores",
    "score_candidates",
    "summarize_scores",
    "window_start_frames",
]
