from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@torch.inference_mode()
def generate_waveforms(
    model: nn.Module,
    captions: Sequence[str],
    *,
    seconds: float = 10.0,
    prior_steps: int | None = None,
    wave_steps: int | None = None,
    solver: str = "euler",
    cfg_strength: float = 2.0,
    wave_cfg_strength: float | None = None,
    cfg_rescale: float = 0.0,
    seed: int = 1234,
    sample_rate: int | None = None,
) -> Tensor:
    """Generate a batch through the release model API."""

    captions = [str(caption).strip() for caption in captions]
    if not captions or any(not caption for caption in captions):
        raise ValueError("captions must contain at least one non-empty string")
    if seconds <= 0.0:
        raise ValueError("seconds must be positive")
    if solver not in {"euler", "heun", "rk4", "dopri5"}:
        raise ValueError("solver must be euler, heun, rk4, or dopri5")

    device = _model_device(model)
    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    kwargs: dict[str, Any] = {
        "seconds": float(seconds),
        "solver": str(solver),
        "cfg_strength": float(cfg_strength),
        "cfg_rescale": float(cfg_rescale),
    }
    if wave_cfg_strength is not None:
        kwargs["wave_cfg_strength"] = float(wave_cfg_strength)
    if prior_steps is not None:
        kwargs["prior_steps"] = int(prior_steps)
    if wave_steps is not None:
        kwargs["wave_steps"] = int(wave_steps)

    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        generated = model.generate(captions, **kwargs)
    waveform = _extract_waveform(generated).detach().float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim == 3 and waveform.shape[1] == 1:
        waveform = waveform[:, 0]
    if waveform.ndim != 2:
        raise ValueError(
            f"model.generate must return waveform [B,T] or [B,1,T], got {tuple(waveform.shape)}"
        )
    if waveform.shape[0] != len(captions):
        raise ValueError(
            f"generated batch size {waveform.shape[0]} does not match captions {len(captions)}"
        )
    resolved_sample_rate = int(sample_rate or _model_sample_rate(model))
    target_samples = max(1, int(round(float(seconds) * resolved_sample_rate)))
    if waveform.shape[-1] < target_samples:
        waveform = F.pad(waveform, (0, target_samples - waveform.shape[-1]))
    return waveform[..., :target_samples].cpu().contiguous()


def generate_to_directory(
    model: nn.Module,
    captions: Sequence[str],
    output_dir: str | Path,
    *,
    seconds: float = 10.0,
    prior_steps: int | None = None,
    wave_steps: int | None = None,
    solver: str = "euler",
    cfg_strength: float = 2.0,
    wave_cfg_strength: float | None = None,
    cfg_rescale: float = 0.0,
    seed: int = 1234,
    sample_rate: int | None = None,
) -> list[dict[str, Any]]:
    captions = [str(caption).strip() for caption in captions]
    resolved_sample_rate = int(sample_rate or _model_sample_rate(model))
    waveform = generate_waveforms(
        model,
        captions,
        seconds=seconds,
        prior_steps=prior_steps,
        wave_steps=wave_steps,
        solver=solver,
        cfg_strength=cfg_strength,
        wave_cfg_strength=wave_cfg_strength,
        cfg_rescale=cfg_rescale,
        seed=seed,
        sample_rate=resolved_sample_rate,
    )
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, (caption, audio) in enumerate(zip(captions, waveform)):
        filename = f"{index:03d}_{_caption_slug(caption)}.wav"
        path = root / filename
        _save_wave(path, audio, resolved_sample_rate)
        records.append(
            {
                "index": index,
                "caption": caption,
                "path": filename,
                "sample_rate": resolved_sample_rate,
                "num_samples": int(audio.numel()),
                "seconds": float(audio.numel() / resolved_sample_rate),
                "seed": int(seed),
                "solver": solver,
                "prior_steps": prior_steps,
                "wave_steps": wave_steps,
                "cfg_strength": float(cfg_strength),
                "wave_cfg_strength": (
                    None
                    if wave_cfg_strength is None
                    else float(wave_cfg_strength)
                ),
                "cfg_rescale": float(cfg_rescale),
            }
        )
    (root / "metadata.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return records


def load_captions(
    captions: Iterable[str] = (),
    captions_file: str | Path | None = None,
) -> list[str]:
    result = [str(caption).strip() for caption in captions if str(caption).strip()]
    if captions_file is not None:
        path = Path(captions_file).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"captions file does not exist: {path}")
        result.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if not result:
        raise ValueError("provide at least one --caption or --captions-file")
    return result


def _extract_waveform(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, Mapping):
        for key in ("waveform", "wav", "audio", "generated_waveform"):
            value = output.get(key)
            if isinstance(value, Tensor):
                return value
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        return output[0]
    raise TypeError("model.generate returned no waveform tensor")


def _save_wave(path: Path, waveform: Tensor, sample_rate: int) -> None:
    try:
        import soundfile as sf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "saving waveform samples requires soundfile"
        ) from exc
    sf.write(
        str(path),
        waveform.detach().float().cpu().numpy(),
        int(sample_rate),
        subtype="FLOAT",
    )


def _model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    return buffer.device if buffer is not None else torch.device("cpu")


def _model_sample_rate(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    if isinstance(config, Mapping):
        return int(config.get("sample_rate", 16_000))
    return int(getattr(config, "sample_rate", 16_000))


def _caption_slug(caption: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "-", caption).strip("-").lower()
    return (slug or "sample")[:48]
