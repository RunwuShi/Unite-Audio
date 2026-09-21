from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator


PROTOCOL = "wavcaps_crop10_clap_audiobox_v1"
DEFAULT_WINDOW_SECONDS = 10.0
DEFAULT_HOP_SECONDS = 5.0
DEFAULT_CLAP_THRESHOLD = 0.40
DEFAULT_AUDIOBOX_PQ_THRESHOLD = 5.0


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
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            count += 1
    os.replace(temporary, path)
    return count


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def crop_start_frames(
    total_frames: int,
    sample_rate: int,
    *,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
) -> list[int]:
    """Return hop-aligned starts plus an exact tail-aligned crop."""
    total_frames = int(total_frames)
    sample_rate = int(sample_rate)
    window = int(round(float(window_seconds) * sample_rate))
    hop = int(round(float(hop_seconds) * sample_rate))
    if total_frames <= 0 or sample_rate <= 0:
        return []
    if window <= 0 or hop <= 0:
        raise ValueError("window and hop must be positive")
    if total_frames <= window:
        return [0]
    starts = list(range(0, total_frames - window + 1, hop))
    tail = total_frames - window
    if starts[-1] != tail:
        starts.append(tail)
    return starts


def build_candidates(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
    expected_count: int | None = None,
) -> dict[str, Any]:
    """Build deterministic 10-second crops from a cleaned offset manifest."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    candidates_path = output_dir / "candidates.jsonl"
    window_seconds = float(window_seconds)
    hop_seconds = float(hop_seconds)
    if window_seconds <= 0.0 or hop_seconds <= 0.0:
        raise ValueError("window_seconds and hop_seconds must be positive")

    counts: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    seen_parent_ids: set[str] = set()
    seen_crops: set[tuple[str, int, int]] = set()

    def rows() -> Iterator[dict[str, Any]]:
        for parent in iter_jsonl(manifest_path):
            counts["input_rows"] += 1
            source = str(parent["source"])
            source_counts[source]["input_rows"] += 1
            parent_id = str(parent["utt_id"])
            if parent_id in seen_parent_ids:
                raise ValueError(f"duplicate parent utt_id: {parent_id}")
            seen_parent_ids.add(parent_id)
            sample_rate = int(parent["source_sample_rate"])
            start_frame = int(parent["start_frame"])
            num_frames = int(parent["num_frames"])
            window_frames = int(round(window_seconds * sample_rate))
            if min(sample_rate, num_frames) <= 0:
                raise ValueError(f"invalid audio geometry for {parent_id}")

            if num_frames <= window_frames:
                kept = dict(parent)
                kept["parent_protocol"] = parent.get("protocol")
                kept["protocol"] = PROTOCOL
                kept["crop10_parent_utt_id"] = parent_id
                kept["crop10_window_seconds"] = window_seconds
                kept["crop10_hop_seconds"] = hop_seconds
                kept["selection_reason"] = "short_passthrough"
                counts["short_passthrough"] += 1
                source_counts[source]["short_passthrough"] += 1
                yield kept
                continue

            counts["long_parent_rows"] += 1
            source_counts[source]["long_parent_rows"] += 1
            for relative_start in crop_start_frames(
                num_frames,
                sample_rate,
                window_seconds=window_seconds,
                hop_seconds=hop_seconds,
            ):
                absolute_start = start_frame + relative_start
                key = (str(parent["audio_path"]), absolute_start, window_frames)
                counts["generated_long_candidates"] += 1
                source_counts[source]["generated_long_candidates"] += 1
                if key in seen_crops:
                    counts["duplicate_long_candidates"] += 1
                    source_counts[source]["duplicate_long_candidates"] += 1
                    continue
                seen_crops.add(key)
                utt_id = (
                    f"wavcaps_crop10/{source}/{parent['item_id']}/"
                    f"{absolute_start:012d}-{window_frames:012d}"
                )
                counts["unique_long_candidates"] += 1
                source_counts[source]["unique_long_candidates"] += 1
                yield {
                    **parent,
                    "parent_protocol": parent.get("protocol"),
                    "protocol": PROTOCOL,
                    "utt_id": utt_id,
                    "start_frame": absolute_start,
                    "num_frames": window_frames,
                    "segment_duration_seconds": window_frames / sample_rate,
                    "crop10_parent_utt_id": parent_id,
                    "crop10_parent_start_frame": start_frame,
                    "crop10_parent_num_frames": num_frames,
                    "crop10_window_seconds": window_seconds,
                    "crop10_hop_seconds": hop_seconds,
                    "crop10_relative_start_seconds": relative_start / sample_rate,
                    "crop10_parent_clap_score": parent.get("clap_score"),
                    "clap_score": None,
                    "audiobox_ce": None,
                    "audiobox_cu": None,
                    "audiobox_pc": None,
                    "audiobox_pq": None,
                    "selection_reason": "long_crop_pending_scores",
                }

    written = _atomic_jsonl(candidates_path, rows())
    if expected_count is not None and counts["input_rows"] != int(expected_count):
        candidates_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"input manifest count mismatch: {counts['input_rows']} != {expected_count}"
        )
    expected_written = (
        counts["short_passthrough"] + counts["unique_long_candidates"]
    )
    if written != expected_written:
        candidates_path.unlink(missing_ok=True)
        raise RuntimeError(f"candidate count mismatch: {written} != {expected_written}")
    summary = {
        "protocol": PROTOCOL,
        "created_at": now_iso(),
        "parent_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "candidates_path": str(candidates_path),
        "candidates_sha256": sha256_file(candidates_path),
        "window_seconds": window_seconds,
        "hop_seconds": hop_seconds,
        "tail_policy": "append_exact_end_aligned_window",
        "deduplication_key": ["audio_path", "start_frame", "num_frames"],
        "counts": dict(counts),
        "source_counts": {key: dict(value) for key, value in source_counts.items()},
    }
    _atomic_json(output_dir / "candidate_summary.json", summary)
    return summary


@lru_cache(maxsize=16)
def _load_source_audio(audio_path: str):
    import numpy as np
    import soundfile as sf

    audio, source_rate = sf.read(
        audio_path, dtype="float32", always_2d=False
    )
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("empty or non-finite audio")
    return np.ascontiguousarray(audio), int(source_rate)


def _load_segment(row: dict[str, Any]):
    import numpy as np

    source_audio, source_rate = _load_source_audio(str(row["audio_path"]))
    expected_rate = int(row["source_sample_rate"])
    if source_rate != expected_rate:
        raise ValueError(f"sample rate changed: {source_rate} != {expected_rate}")
    start = int(row["start_frame"])
    end = start + int(row["num_frames"])
    audio = source_audio[start:end]
    if len(audio) != int(row["num_frames"]):
        raise ValueError(f"short read: {len(audio)} != {row['num_frames']}")
    if float(np.max(np.abs(audio))) <= 1e-8:
        return None
    return np.ascontiguousarray(audio), source_rate


def _prepare_clap_batch(native_audio, device: str, target_rate: int = 48_000):
    import numpy as np
    import torch
    import torch.nn.functional as torch_functional
    import torchaudio.functional as audio_functional

    target_frames = int(round(DEFAULT_WINDOW_SECONDS * target_rate))
    output = torch.empty(
        (len(native_audio), target_frames), dtype=torch.float32, device=device
    )
    grouped: dict[int, list[tuple[int, Any]]] = defaultdict(list)
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


def _prepare_audiobox_batch(
    native_audio, device: str, target_rate: int = 16_000
):
    """Vectorized equivalent of AudioBox's per-item mono/resample/collate path."""
    import numpy as np
    import torch
    import torch.nn.functional as torch_functional
    import torchaudio.functional as audio_functional

    target_frames = int(round(DEFAULT_WINDOW_SECONDS * target_rate))
    output = torch.empty(
        (len(native_audio), target_frames), dtype=torch.float32, device=device
    )
    grouped: dict[int, list[tuple[int, Any]]] = defaultdict(list)
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
        for group_index, (output_index, _) in enumerate(entries):
            output[output_index].copy_(tensor[group_index])
    return output.unsqueeze(1)


def _load_clap_model(checkpoint: Path, device: str):
    import laion_clap
    from clap_module.factory import load_state_dict

    model = laion_clap.CLAP_Module(enable_fusion=True, device=device)
    state = load_state_dict(str(checkpoint))
    state.pop("text_branch.embeddings.position_ids", None)
    model.model.load_state_dict(state)
    model.eval()
    return model


def _get_short_clap_audio_embeddings(model, audio_tensor, device: str):
    """Vectorize LAION-CLAP fusion features for exact 10-second crops."""
    import torch
    import torchaudio

    audio_tensor = audio_tensor.to(device, non_blocking=True)
    audio_cfg = model.model_cfg["audio_cfg"]
    cache = getattr(model, "_crop10_feature_transforms", None)
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
        cache = (mel_transform, amplitude_to_db)
        model._crop10_feature_transforms = cache
    mel_transform, amplitude_to_db = cache
    mel = amplitude_to_db(mel_transform(audio_tensor)).transpose(1, 2)
    mel_fusion = mel.unsqueeze(1).expand(-1, 4, -1, -1)
    audio_input = [
        {
            "mel_fusion": mel_fusion[index],
            "longer": torch.tensor([False], device=device),
            "waveform": audio_tensor[index],
        }
        for index in range(audio_tensor.shape[0])
    ]
    return model.model.get_audio_embedding(audio_input)


def _audio_path_shard(audio_path: str, num_shards: int) -> int:
    """Keep all crops from one source file on one scorer for decode reuse."""
    digest = hashlib.sha256(audio_path.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little") % int(num_shards)


def score_candidates(
    *,
    candidates_path: str | Path,
    output_path: str | Path,
    clap_checkpoint: str | Path,
    shard_index: int = 0,
    num_shards: int = 1,
    batch_size: int = 32,
    device: str = "cuda:0",
    log_every: int = 1_000,
    clap_threshold: float = DEFAULT_CLAP_THRESHOLD,
) -> dict[str, int]:
    """CLAP-score a shard, then AudioBox-score only CLAP-passing crops."""
    import torch
    from audiobox_aesthetics.infer import initialize_predictor

    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    successful_rows: list[dict[str, Any]] = []
    discarded_failures = 0
    if output_path.is_file():
        seen_existing: set[str] = set()
        for row in iter_jsonl(output_path):
            utt_id = str(row["utt_id"])
            if utt_id in seen_existing:
                raise ValueError(f"duplicate resumed score: {utt_id}")
            seen_existing.add(utt_id)
            clap_score = row.get("clap_score")
            aes_valid = all(
                row.get(key) is not None
                for key in (
                    "audiobox_ce",
                    "audiobox_cu",
                    "audiobox_pc",
                    "audiobox_pq",
                )
            )
            valid = clap_score is not None and (
                float(clap_score) < float(clap_threshold) or aes_valid
            )
            recorded_gate = row.get("score_gate_clap_threshold")
            if (
                recorded_gate is not None
                and float(recorded_gate) != float(clap_threshold)
            ):
                raise ValueError(
                    f"score gate changed for {utt_id}: "
                    f"{recorded_gate} != {clap_threshold}"
                )
            if not valid or row.get("status") == "error":
                discarded_failures += 1
                continue
            completed.add(utt_id)
            successful_rows.append(row)
        if discarded_failures:
            _atomic_jsonl(output_path, successful_rows)

    selected: list[dict[str, Any]] = []
    for row in iter_jsonl(candidates_path):
        if row["selection_reason"] != "long_crop_pending_scores":
            continue
        row_shard = _audio_path_shard(str(row["audio_path"]), num_shards)
        if row_shard == shard_index and row["utt_id"] not in completed:
            selected.append(row)

    clap_model = _load_clap_model(
        Path(clap_checkpoint).expanduser().resolve(), device
    )
    audiobox = initialize_predictor()
    if str(audiobox.device) != str(torch.device(device)):
        audiobox.device = torch.device(device)
        audiobox.model.to(audiobox.device)
    counters: Counter[str] = Counter(
        resumed=len(completed),
        pending=len(selected),
        retried_failures=discarded_failures,
    )
    with output_path.open("a", encoding="utf-8") as handle:
        for batch_start in range(0, len(selected), batch_size):
            raw_batch = selected[batch_start : batch_start + batch_size]
            active_rows: list[dict[str, Any]] = []
            audio_values: list[tuple[Any, int]] = []
            for row in raw_batch:
                try:
                    audio = _load_segment(row)
                except Exception as exc:
                    handle.write(
                        json.dumps(
                            {
                                "utt_id": row["utt_id"],
                                "status": "error",
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    counters["error"] += 1
                    continue
                if audio is None:
                    handle.write(
                        json.dumps(
                            {
                                "utt_id": row["utt_id"],
                                "clap_score": -1.0,
                                "audiobox_ce": 0.0,
                                "audiobox_cu": 0.0,
                                "audiobox_pc": 0.0,
                                "audiobox_pq": 0.0,
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
                clap_audio = _prepare_clap_batch(audio_values, device)
                texts = [str(row["caption"]) for row in active_rows]
                with torch.inference_mode():
                    unique_texts = list(dict.fromkeys(texts))
                    unique_text_embeddings = clap_model.get_text_embedding(
                        unique_texts, use_tensor=True
                    )
                    text_to_index = {
                        value: index for index, value in enumerate(unique_texts)
                    }
                    text_indices = torch.tensor(
                        [text_to_index[value] for value in texts],
                        dtype=torch.long,
                        device=device,
                    )
                    text_embeddings = unique_text_embeddings.index_select(
                        0, text_indices
                    )
                    audio_embeddings = _get_short_clap_audio_embeddings(
                        clap_model, clap_audio, device
                    )
                    similarities = torch.nn.functional.cosine_similarity(
                        audio_embeddings, text_embeddings, dim=1, eps=1e-8
                    )
                clap_values = similarities.detach().cpu().tolist()
                passing_indices = [
                    index
                    for index, value in enumerate(clap_values)
                    if float(value) >= float(clap_threshold)
                ]
                aesthetic_by_index: dict[int, dict[str, float]] = {}
                if passing_indices:
                    passing_audio = [
                        audio_values[index] for index in passing_indices
                    ]
                    audiobox_audio = _prepare_audiobox_batch(
                        passing_audio, device
                    )
                    with torch.inference_mode():
                        aes_raw = audiobox.model(
                            {
                                "wav": audiobox_audio,
                                "mask": torch.ones_like(
                                    audiobox_audio, dtype=torch.bool
                                ),
                            }
                        )
                        aes_values = {
                            axis: audiobox.target_transform[axis]
                            .inverse(aes_raw[axis])
                            .detach()
                            .cpu()
                            .tolist()
                            for axis in ("CE", "CU", "PC", "PQ")
                        }
                    for aes_index, active_index in enumerate(passing_indices):
                        aesthetic_by_index[active_index] = {
                            axis: float(aes_values[axis][aes_index])
                            for axis in ("CE", "CU", "PC", "PQ")
                        }
                for index, (row, clap_score) in enumerate(
                    zip(active_rows, clap_values)
                ):
                    aes = aesthetic_by_index.get(index)
                    result = {
                        "utt_id": row["utt_id"],
                        "clap_score": float(clap_score),
                        "audiobox_ce": None if aes is None else aes["CE"],
                        "audiobox_cu": None if aes is None else aes["CU"],
                        "audiobox_pc": None if aes is None else aes["PC"],
                        "audiobox_pq": None if aes is None else aes["PQ"],
                        "score_gate_clap_threshold": float(clap_threshold),
                        "status": "clap_rejected" if aes is None else "ok",
                    }
                    finite_values = [float(result["clap_score"])]
                    if aes is not None:
                        finite_values.extend(
                            float(result[key])
                            for key in (
                                "audiobox_ce",
                                "audiobox_cu",
                                "audiobox_pc",
                                "audiobox_pq",
                            )
                        )
                    if not all(math.isfinite(value) for value in finite_values):
                        raise RuntimeError(f"non-finite score for {row['utt_id']}")
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    counters["ok"] += 1
                    counters["clap_pass"] += int(aes is not None)
                    counters["clap_rejected"] += int(aes is None)
            handle.flush()
            processed = min(batch_start + batch_size, len(selected))
            if processed % max(1, log_every) < batch_size:
                print(
                    f"[WavCaps Crop10] shard={shard_index}/{num_shards} "
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


def _distribution(values: list[float]) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    quantiles = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "quantiles": {
            f"{quantile:.2f}": float(np.quantile(array, quantile))
            for quantile in quantiles
        },
    }


def summarize_scores(
    *,
    candidates_path: str | Path,
    scores_dir: str | Path,
    output_dir: str | Path,
    clap_threshold: float = DEFAULT_CLAP_THRESHOLD,
    audiobox_pq_threshold: float = DEFAULT_AUDIOBOX_PQ_THRESHOLD,
) -> dict[str, Any]:
    import numpy as np

    scores = load_scores(scores_dir)
    expected: set[str] = set()
    values: dict[str, list[float]] = defaultdict(list)
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    counters: Counter[str] = Counter()
    for row in iter_jsonl(candidates_path):
        if row["selection_reason"] != "long_crop_pending_scores":
            continue
        utt_id = str(row["utt_id"])
        expected.add(utt_id)
        score = scores.get(utt_id)
        if score is None or score.get("clap_score") is None:
            counters["missing_or_error"] += 1
            continue
        source = str(row["source"])
        source_counts[source]["scored"] += 1
        clap_pass = float(score["clap_score"]) >= float(clap_threshold)
        values["clap_score"].append(float(score["clap_score"]))
        aes_keys = (
            "audiobox_ce",
            "audiobox_cu",
            "audiobox_pc",
            "audiobox_pq",
        )
        if clap_pass and any(score.get(key) is None for key in aes_keys):
            counters["missing_or_error"] += 1
            continue
        if clap_pass:
            for key in aes_keys:
                values[key].append(float(score[key]))
        pq_pass = clap_pass and (
            float(score["audiobox_pq"]) >= float(audiobox_pq_threshold)
        )
        source_counts[source]["clap_pass"] += int(clap_pass)
        source_counts[source]["pq_pass"] += int(pq_pass)
        source_counts[source]["both_pass"] += int(clap_pass and pq_pass)
        counters["scored"] += 1
        counters["clap_pass"] += int(clap_pass)
        counters["pq_pass"] += int(pq_pass)
        counters["both_pass"] += int(clap_pass and pq_pass)

    extras = set(scores).difference(expected)
    if counters["missing_or_error"] or extras or len(scores) != len(expected):
        raise RuntimeError(
            "score set is incomplete: "
            f"expected={len(expected)} scores={len(scores)} "
            f"missing_or_error={counters['missing_or_error']} extras={len(extras)}"
        )
    passing_clap = [
        float(score["clap_score"])
        for score in scores.values()
        if float(score["clap_score"]) >= float(clap_threshold)
    ]
    correlation = np.corrcoef(
        np.asarray(passing_clap), np.asarray(values["audiobox_pq"])
    )
    summary = {
        "protocol": PROTOCOL,
        "created_at": now_iso(),
        "thresholds": {
            "clap_score": float(clap_threshold),
            "audiobox_pq": float(audiobox_pq_threshold),
            "logic": "clap_score >= threshold AND audiobox_pq >= threshold",
        },
        "counts": dict(counters),
        "source_counts": {key: dict(value) for key, value in source_counts.items()},
        "distributions": {
            key: _distribution(metric_values)
            for key, metric_values in values.items()
        },
        "clap_pq_pearson": float(correlation[0, 1]),
    }
    output_dir = Path(output_dir).expanduser().resolve()
    _atomic_json(output_dir / "score_stats.json", summary)
    return summary


def finalize_manifest(
    *,
    candidates_path: str | Path,
    scores_dir: str | Path,
    output_dir: str | Path,
    clap_threshold: float = DEFAULT_CLAP_THRESHOLD,
    audiobox_pq_threshold: float = DEFAULT_AUDIOBOX_PQ_THRESHOLD,
    clap_checkpoint: str | Path | None = None,
    audiobox_model: str = "facebook/audiobox-aesthetics",
) -> dict[str, Any]:
    candidates_path = Path(candidates_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    scores = load_scores(scores_dir)
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
            score = scores.get(utt_id)
            if score is None:
                raise RuntimeError(f"missing score for {utt_id}")
            seen_scores.add(utt_id)
            clap_pass = float(score["clap_score"]) >= float(clap_threshold)
            aes_keys = (
                "audiobox_ce",
                "audiobox_cu",
                "audiobox_pc",
                "audiobox_pq",
            )
            if clap_pass and any(score.get(key) is None for key in aes_keys):
                raise RuntimeError(f"missing AudioBox score for CLAP-pass {utt_id}")
            pq_pass = clap_pass and (
                float(score["audiobox_pq"]) >= float(audiobox_pq_threshold)
            )
            counts["long_scored"] += 1
            counts["clap_pass"] += int(clap_pass)
            counts["pq_pass"] += int(pq_pass)
            source_counts[source]["long_scored"] += 1
            source_counts[source]["clap_pass"] += int(clap_pass)
            source_counts[source]["pq_pass"] += int(pq_pass)
            if not (clap_pass and pq_pass):
                counts["long_rejected"] += 1
                source_counts[source]["long_rejected"] += 1
                continue
            kept = dict(row)
            kept["clap_score"] = float(score["clap_score"])
            for key in aes_keys:
                kept[key] = float(score[key])
            kept["selection_reason"] = "clap_audiobox_pass"
            kept["crop10_clap_threshold"] = float(clap_threshold)
            kept["crop10_audiobox_pq_threshold"] = float(
                audiobox_pq_threshold
            )
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
        manifest_path.unlink(missing_ok=True)
        raise RuntimeError("final manifest count mismatch")
    version = {
        "protocol": PROTOCOL,
        "created_at": now_iso(),
        "parent_candidates": {
            "path": str(candidates_path),
            "sha256": sha256_file(candidates_path),
        },
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "selection": {
            "clap_threshold": float(clap_threshold),
            "audiobox_pq_threshold": float(audiobox_pq_threshold),
            "logic": "both_thresholds_must_pass",
            "multiple_crops_per_source_allowed": True,
        },
        "models": {
            "clap": None
            if clap_checkpoint is None
            else {
                "path": str(Path(clap_checkpoint).expanduser().resolve()),
                "sha256": sha256_file(clap_checkpoint),
            },
            "audiobox": audiobox_model,
            "audiobox_axis_for_filtering": "PQ",
            "audiobox_axes_recorded": ["CE", "CU", "PC", "PQ"],
        },
        "counts": dict(counts),
        "source_counts": {key: dict(value) for key, value in source_counts.items()},
    }
    _atomic_json(output_dir / "VERSION.json", version)
    readme = "# WavCaps 10-second CLAP + AudioBox crops\n\n"
    readme += "- Parent: cleaned, full-silence-filtered WavCaps offset manifest.\n"
    readme += "- Short segments (<=10s) pass through unchanged.\n"
    readme += "- Long segments use 10s windows at 5s hop plus an end-aligned window.\n"
    readme += "- Exact duplicate source offsets are removed before scoring.\n"
    readme += "- Multiple windows from one source recording may be retained.\n"
    readme += (
        f"- Long-window filter: CLAP >= {float(clap_threshold):g} AND "
        f"AudioBox PQ >= {float(audiobox_pq_threshold):g}.\n"
    )
    readme += "- Audio remains in source files; the manifest stores fixed frame offsets.\n\n"
    readme += f"Final rows: {counts['retained']:,}\n"
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return version


__all__ = [
    "DEFAULT_AUDIOBOX_PQ_THRESHOLD",
    "DEFAULT_CLAP_THRESHOLD",
    "DEFAULT_HOP_SECONDS",
    "DEFAULT_WINDOW_SECONDS",
    "PROTOCOL",
    "build_candidates",
    "crop_start_frames",
    "finalize_manifest",
    "iter_jsonl",
    "load_scores",
    "score_candidates",
    "summarize_scores",
]
