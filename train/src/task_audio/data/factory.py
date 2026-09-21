from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import DataLoader, IterableDataset

from .audiocaps import AudioCapsDataset
from .audioset import (
    AUDIOSET_DEFAULT_ROOT,
    AUDIOSET_EVAL_SPLITS,
    AUDIOSET_TRAIN_SPLITS,
    AudioSetCaptionFormatter,
    AudioSetDataset,
)
from .common import collate_audio_text
from .mixed import ReplayMixtureDataset
from .wavcaps import WAVCAPS_DEFAULT_ROOT, WavCapsDataset


SUPPORTED_DATASETS = ("audiocaps", "audioset", "wavcaps", "mixed")


def build_dataset(config: Mapping[str, Any], split: str = "train"):
    """Build one configured TTA dataset or a controlled primary/replay mix."""

    data, paths = _resolve_sections(config)
    dataset_name = str(data.get("dataset", "audiocaps")).strip().lower()
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"data.dataset must select exactly one of {SUPPORTED_DATASETS}; got {dataset_name!r}"
        )

    sample_rate = int(data.get("sample_rate", 16_000))
    max_seconds = data.get("max_audio_seconds", 10.0)
    resampling = dict(data.get("resampling") or {})
    normalized_split = _normalize_user_split(split)

    if dataset_name == "mixed":
        if normalized_split != "train":
            raise ValueError("mixed datasets currently support only the train split")
        mix = dict(data.get("mix") or {})
        primary_name = str(mix.get("primary_dataset", "")).strip().lower()
        replay_name = str(mix.get("replay_dataset", "")).strip().lower()
        if (primary_name, replay_name) != ("audiocaps", "wavcaps"):
            raise ValueError(
                "data.mix currently requires primary_dataset='audiocaps' and "
                "replay_dataset='wavcaps'"
            )
        component_data = dict(data)
        component_data.pop("mix", None)
        primary_data = dict(component_data)
        primary_data["dataset"] = primary_name
        replay_data = dict(component_data)
        replay_data["dataset"] = replay_name
        primary = build_dataset(
            {"data": primary_data, "paths": paths}, split=normalized_split
        )
        replay = build_dataset(
            {"data": replay_data, "paths": paths}, split=normalized_split
        )
        return ReplayMixtureDataset(
            primary,
            replay,
            replay_fraction=float(mix.get("replay_fraction", 0.10)),
            seed=int(data.get("seed", _experiment_seed(config))),
        )

    if dataset_name == "audiocaps":
        settings = dict(data.get("audiocaps") or {})
        root = settings.get("root") or paths.get("audiocaps_root")
        if not root:
            raise ValueError(
                "AudioCaps requires an explicit data.audiocaps.root or "
                "paths.audiocaps_root; no filesystem default is used"
            )
        default_split = "validation" if normalized_split == "eval" else normalized_split
        dataset_split = settings.get(f"{normalized_split}_split", default_split)
        return AudioCapsDataset(
            root=root,
            split=dataset_split,
            sample_rate=sample_rate,
            max_audio_seconds=max_seconds,
            validate_counts=bool(settings.get("validate_counts", True)),
            manifest_path=settings.get("manifest_path"),
            audio_root=settings.get("audio_root"),
            expected_count=settings.get("expected_count"),
            forbidden_protocol=settings.get("forbidden_protocol"),
            resampling=resampling,
        )

    if dataset_name == "wavcaps":
        if normalized_split != "train":
            raise ValueError("WavCaps currently supports only the train split")
        settings = dict(data.get("wavcaps") or {})
        return WavCapsDataset(
            root=settings.get("root") or paths.get("wavcaps_root") or WAVCAPS_DEFAULT_ROOT,
            sample_rate=sample_rate,
            max_audio_seconds=float(max_seconds),
            blacklist_path=settings.get("blacklist_path"),
            require_files=bool(settings.get("require_files", True)),
            manifest_path=settings.get("manifest_path"),
            expected_count=settings.get("expected_count"),
            resampling=resampling,
        )

    settings = dict(data.get("audioset") or {})
    root = settings.get("root") or paths.get("audioset_root") or AUDIOSET_DEFAULT_ROOT
    if normalized_split == "train":
        dataset_splits = settings.get("train_splits", AUDIOSET_TRAIN_SPLITS)
    else:
        dataset_splits = settings.get("eval_splits", AUDIOSET_EVAL_SPLITS)
    caption_config = dict(settings.get("caption") or data.get("caption") or {})
    caption_mode = (
        str(caption_config.get("mode", "labels"))
        .strip()
        .lower()
        .replace("-", "_")
    )
    if caption_mode in {"audio_set_caps", "audio_setcaps"}:
        caption_mode = "audiosetcaps"
    allowed_caption_modes = {
        "labels",
        "template",
        "random_template",
        "audiosetcaps",
    }
    if caption_mode not in allowed_caption_modes:
        raise ValueError(
            "AudioSet caption.mode must be one of "
            f"{sorted(allowed_caption_modes)}; got {caption_mode!r}"
        )
    caption_fallback = (
        str(caption_config.get("fallback", "labels")).strip().lower()
    )
    if caption_fallback != "labels":
        raise ValueError("AudioSetCaps caption.fallback must currently be 'labels'")
    use_external_captions = normalized_split == "train" or bool(
        caption_config.get("use_for_eval", False)
    )
    if caption_mode == "audiosetcaps" and not use_external_captions:
        caption_mode = caption_fallback
    formatter = AudioSetCaptionFormatter(
        mode=caption_mode,
        template=str(caption_config.get("template", "{labels}")),
        templates=tuple(str(value) for value in caption_config.get("templates", ())),
        label_joiner=str(caption_config.get("label_joiner", ", ")),
        lowercase=bool(caption_config.get("lowercase", False)),
    )
    forbidden_id_files: dict[str, Any] = {}
    if bool(settings.get("exclude_audiocaps_overlaps", True)):
        train_manifest = settings.get("audiocaps_train_manifest")
        eval_protocol = settings.get("audiocaps_eval_protocol")
        if not train_manifest or not eval_protocol:
            raise ValueError(
                "AudioSet overlap exclusion requires audiocaps_train_manifest "
                "and audiocaps_eval_protocol"
            )
        forbidden_id_files = {
            "audiocaps_train": train_manifest,
            "tango_audiocaps_test": eval_protocol,
        }
    forbidden_id_files.update(dict(settings.get("forbidden_id_files") or {}))
    return AudioSetDataset(
        root=root,
        splits=dataset_splits,
        sample_rate=sample_rate,
        max_audio_seconds=max_seconds,
        caption_formatter=formatter,
        audiosetcaps_path=(
            caption_config.get("metadata_path")
            or paths.get("audiosetcaps_captions")
        ),
        audiosetcaps_id_prefix=str(caption_config.get("id_prefix", "Y")),
        audiosetcaps_fallback=caption_fallback,
        audiosetcaps_require_match=bool(caption_config.get("require_match", False)),
        seed=int(data.get("seed", _experiment_seed(config))),
        shuffle_files=(
            normalized_split == "train" and bool(settings.get("shuffle_files", True))
        ),
        read_batch_size=int(settings.get("read_batch_size", 16)),
        required_pyarrow_version=settings.get("required_pyarrow_version", "18.1.0"),
        forbidden_id_files=forbidden_id_files,
        forbidden_protocol=settings.get("forbidden_protocol"),
        resampling=resampling,
        strict_sample_coverage=bool(settings.get("strict_sample_coverage", False)),
    )


def build_dataloader(
    config: Mapping[str, Any],
    split: str = "train",
    *,
    batch_size: int | None = None,
    shuffle: bool | None = None,
    drop_last: bool | None = None,
    **loader_overrides: Any,
) -> DataLoader:
    """Build a DataLoader that emits the canonical TTA batch contract."""

    data, _ = _resolve_sections(config)
    full_config = "data" in config
    training = dict(config.get("training") or {}) if full_config else {}
    if full_config and "batch_size" in data:
        raise ValueError(
            "data.batch_size was removed; set the single canonical "
            "training.batch_size instead"
        )
    dataset = build_dataset(config, split=split)
    normalized_split = _normalize_user_split(split)

    if batch_size is None:
        # Full experiment configs have one canonical batch-size setting under
        # ``training``. The data-section fallback only keeps the low-level
        # ``build_dataloader(config["data"])`` API usable.
        batch_size = int(
            training.get("batch_size", 4) if full_config else data.get("batch_size", 4)
        )
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if drop_last is None:
        drop_last = (
            bool(data.get("drop_last", True)) if normalized_split == "train" else False
        )

    iterable = isinstance(dataset, IterableDataset)
    if shuffle is None:
        shuffle = normalized_split == "train" and not iterable
    if iterable and shuffle:
        raise ValueError(
            "AudioSet is already file-shuffled; DataLoader shuffle must be False"
        )

    num_workers = int(training.get("num_workers", data.get("num_workers", 2)))
    pin_memory = bool(training.get("pin_memory", data.get("pin_memory", True)))
    persistent_workers = bool(
        training.get(
            "persistent_workers", data.get("persistent_workers", num_workers > 0)
        )
    )
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": bool(shuffle),
        "drop_last": bool(drop_last),
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers if num_workers > 0 else False,
        "collate_fn": collate_audio_text,
    }
    if not iterable:
        kwargs["generator"] = torch.Generator().manual_seed(_experiment_seed(config))
    prefetch_factor = training.get("prefetch_factor", data.get("prefetch_factor"))
    if num_workers > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    kwargs.update(loader_overrides)
    return DataLoader(dataset, **kwargs)


def _resolve_sections(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if "data" in config:
        data = dict(config.get("data") or {})
        paths = dict(config.get("paths") or {})
    else:
        data = dict(config)
        paths = dict(data.pop("paths", {}) or {})
    return data, paths


def _normalize_user_split(split: str) -> str:
    normalized = str(split).strip().lower()
    if normalized == "train":
        return "train"
    if normalized in {"valid", "validation", "eval", "test"}:
        return "validation" if normalized in {"valid", "validation"} else normalized
    raise ValueError("split must be train, validation/valid, eval, or test")


def _experiment_seed(config: Mapping[str, Any]) -> int:
    experiment = config.get("experiment") if isinstance(config, Mapping) else None
    if isinstance(experiment, Mapping):
        return int(experiment.get("seed", 1234))
    return 1234


__all__ = ["SUPPORTED_DATASETS", "build_dataloader", "build_dataset"]
