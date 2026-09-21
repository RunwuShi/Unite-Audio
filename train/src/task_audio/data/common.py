from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any


DEFAULT_RESAMPLING: dict[str, Any] = {
    "method": "sinc_kaiser",
    "lowpass_filter_width": 16,
    "rolloff": 0.9475937167,
    "beta": 14.769656459,
}


def normalize_resampling_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the validated task_audio resampling configuration."""

    resolved = dict(DEFAULT_RESAMPLING)
    resolved.update(dict(config or {}))
    method = str(resolved["method"]).strip().lower().replace("-", "_")
    if method in {"linear", "legacy_linear"}:
        method = "linear_legacy"
    if method not in {"sinc_kaiser", "linear_legacy"}:
        raise ValueError(
            "resampling.method must be 'sinc_kaiser' or 'linear_legacy'"
        )
    width = int(resolved["lowpass_filter_width"])
    rolloff = float(resolved["rolloff"])
    beta = float(resolved["beta"])
    if width < 1:
        raise ValueError("resampling.lowpass_filter_width must be positive")
    if not 0.0 < rolloff <= 1.0:
        raise ValueError("resampling.rolloff must be in (0, 1]")
    if beta <= 0.0:
        raise ValueError("resampling.beta must be positive")
    return {
        "method": method,
        "lowpass_filter_width": width,
        "rolloff": rolloff,
        "beta": beta,
    }


def collate_audio_text(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pad audio-text samples into the canonical TTA batch contract.

    Dataset-specific information is kept in ``metadata``.  A few frequently
    used metadata fields are also exposed at the top level for logging and for
    compatibility with the TTS trainer utilities.
    """

    import torch

    if not samples:
        raise ValueError("cannot collate an empty audio-text batch")

    required = ("wav", "caption", "utt_id", "dataset", "wav_sample_rate")
    for index, sample in enumerate(samples):
        missing = [key for key in required if key not in sample]
        if missing:
            raise KeyError(f"sample {index} is missing required field(s): {missing}")

    waveforms = []
    for index, sample in enumerate(samples):
        wav = sample["wav"]
        if not isinstance(wav, torch.Tensor):
            wav = torch.as_tensor(wav)
        wav = wav.float().reshape(-1).contiguous()
        if wav.numel() == 0:
            raise ValueError(f"sample {index} ({sample['utt_id']!r}) has empty audio")
        waveforms.append(wav)

    lengths = torch.tensor([wav.numel() for wav in waveforms], dtype=torch.long)
    max_length = int(lengths.max().item())
    wav_clean = torch.zeros((len(waveforms), max_length), dtype=torch.float32)
    for row, wav in enumerate(waveforms):
        wav_clean[row, : wav.numel()] = wav
    wav_valid_mask = torch.arange(max_length, dtype=torch.long).unsqueeze(0) < lengths.unsqueeze(1)

    captions = [str(sample["caption"]).strip() for sample in samples]
    metadata = []
    for sample in samples:
        item_metadata = dict(sample.get("metadata") or {})
        item_metadata.setdefault("dataset", str(sample["dataset"]))
        item_metadata.setdefault("utt_id", str(sample["utt_id"]))
        if "split" in sample:
            item_metadata.setdefault("split", str(sample["split"]))
        metadata.append(item_metadata)

    return {
        "wav_clean": wav_clean,
        "wav_valid_mask": wav_valid_mask,
        "wav_lengths": lengths,
        "wav_sample_rate": torch.tensor(
            [int(sample["wav_sample_rate"]) for sample in samples],
            dtype=torch.long,
        ),
        "caption": captions,
        "metadata": metadata,
        "utt_id": [str(sample["utt_id"]) for sample in samples],
        "dataset": [str(sample["dataset"]) for sample in samples],
        "split": [str(sample.get("split", "")) for sample in samples],
        "audio_ref": [item.get("audio_ref") for item in metadata],
    }


@lru_cache(maxsize=32)
def _cpu_sinc_kaiser_resampler(
    source_rate: int,
    target_rate: int,
    lowpass_filter_width: int,
    rolloff: float,
    beta: float,
):
    import torchaudio

    return torchaudio.transforms.Resample(
        orig_freq=source_rate,
        new_freq=target_rate,
        resampling_method="sinc_interp_kaiser",
        lowpass_filter_width=lowpass_filter_width,
        rolloff=rolloff,
        beta=beta,
        dtype=None,
    )


def resample_waveform(
    wav: Any,
    source_rate: int,
    target_rate: int,
    *,
    method: str = "sinc_kaiser",
    lowpass_filter_width: int = 16,
    rolloff: float = 0.9475937167,
    beta: float = 14.769656459,
):
    """Resample mono audio with anti-aliasing by default.

    ``linear_legacy`` exactly preserves the historical interpolation path for
    old experiment reproduction.
    """

    import torch

    source_rate = int(source_rate)
    target_rate = int(target_rate)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    wav = torch.as_tensor(wav).float().reshape(-1)
    if source_rate == target_rate:
        return wav.contiguous()
    if wav.numel() == 0:
        return wav.contiguous()
    settings = normalize_resampling_config(
        {
            "method": method,
            "lowpass_filter_width": lowpass_filter_width,
            "rolloff": rolloff,
            "beta": beta,
        }
    )
    if settings["method"] == "sinc_kaiser":
        try:
            if wav.device.type == "cpu":
                transform = _cpu_sinc_kaiser_resampler(
                    source_rate,
                    target_rate,
                    settings["lowpass_filter_width"],
                    settings["rolloff"],
                    settings["beta"],
                )
                return transform(wav).reshape(-1).contiguous()
            import torchaudio

            return torchaudio.functional.resample(
                wav,
                source_rate,
                target_rate,
                lowpass_filter_width=settings["lowpass_filter_width"],
                rolloff=settings["rolloff"],
                resampling_method="sinc_interp_kaiser",
                beta=settings["beta"],
            ).reshape(-1).contiguous()
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "task_audio sinc-kaiser resampling requires a working torchaudio install; "
                "set data.resampling.method='linear_legacy' only to reproduce old runs"
            ) from exc

    import torch.nn.functional as F

    target_length = max(1, int(round(wav.numel() * target_rate / source_rate)))
    return F.interpolate(
        wav.view(1, 1, -1),
        size=target_length,
        mode="linear",
        align_corners=False,
    ).view(-1)


def trim_waveform(wav: Any, *, sample_rate: int, max_audio_seconds: float | None):
    import torch

    wav = torch.as_tensor(wav).float().reshape(-1)
    if max_audio_seconds is None:
        return wav.contiguous()
    max_audio_seconds = float(max_audio_seconds)
    if max_audio_seconds <= 0.0:
        raise ValueError("max_audio_seconds must be positive or None")
    max_samples = max(1, int(round(max_audio_seconds * int(sample_rate))))
    return wav[:max_samples].contiguous()


__all__ = [
    "DEFAULT_RESAMPLING",
    "collate_audio_text",
    "normalize_resampling_config",
    "resample_waveform",
    "trim_waveform",
]
