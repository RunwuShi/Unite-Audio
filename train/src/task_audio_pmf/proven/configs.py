from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


LATENTAUDIO_ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_RUN = (
    LATENTAUDIO_ROOT
    / "experiments"
    / "tta_wavcaps_audiocaps"
    / "20260716_224349"
)
HISTORICAL_STAGE1 = (
    HISTORICAL_RUN / "stage1_wavcaps" / "config" / "resolved_config.json"
)
HISTORICAL_STAGE2 = (
    HISTORICAL_RUN / "stage2_audiocaps" / "config" / "resolved_config.json"
)


def _deep_update(target: dict[str, Any], values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)


def _load_frozen(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen baseline config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _common(config: dict[str, Any], *, stage: str) -> dict[str, Any]:
    model = config["model"]
    _deep_update(
        model,
        {
            "architecture": "latent_tta_pmf",
            # Keep the successful historical codec bottleneck exactly:
            # 25 latent tokens/s, 64 channels/token.  Define these here
            # explicitly instead of relying on the inherited frozen config.
            "latent_dim": 64,
            "latent_hz": 25.0,
            "prediction": "pmf_x_v",
            "student_condition_view": "masked_input",
            "target_ema_view": "full_audio",
            "split_target_encoder_ema": True,
            "reconstruction_latent_source": "full_online",
            # The proven baseline is 512 x 16. Increase capacity modestly by
            # two blocks while preserving its efficient 8 x 64 attention
            # width. The earlier 640 x 20 trial made pMF/JVP too expensive.
            "fm_hidden_dim": 512,
            "fm_depth": 18,
            "fm_mmdit_fused_depth": 11,
            "fm_heads": 8,
            "fm_dim_head": 64,
            "fm_ffn_mult": 4,
            "prior_fm": {
                "hidden_dim": 512,
                "depth": 18,
                "fused_depth": 11,
                "heads": 8,
                "dim_head": 64,
                "ffn_mult": 4,
                "adaln_every": 2,
                "input_fusion": "add",
                "text_injection": "mmdit",
            },
            "pmf_fm_proportion": 0.5,
            "pmf_norm_p": 1.0,
            "pmf_norm_eps": 0.01,
            "pmf_main_weight": 1.0,
            "pmf_auxiliary_v_weight": 1.0,
            "pmf_min_time": 0.02,
            "global_stft_weight": 1.0,
            "global_stft_n_fft": 1024,
            "global_stft_win_length": 1024,
            "global_stft_hop_length": 256,
            "global_stft_eps": 1.0e-4,
            "same_mrstft_crop_seconds": 2.0,
            "adversarial_crop_seconds": 2.0,
            "loss": {
                "prior_fm": 1.0,
                "decoder": 0.0,
                "recon": 0.0,
                "mel": 0.0,
                "reconstruction": {
                    "type": "same_phase_mrstft",
                    "enabled": True,
                    "weight": 1.0,
                    "crop_seconds": 2.0,
                    "fft_sizes": [32, 64, 128, 256, 512, 1024, 2048],
                    "hop_ratio": 0.25,
                    "k_weighting": True,
                    "eps": 1.0e-7,
                    "stability_eps": 1.0e-4,
                },
                "adversarial": {
                    "enabled": True,
                    "start_step": 10_000,
                    "crop_seconds": 2.0,
                    "objective": "relativistic_paired",
                    "generator_weight": 0.1,
                    "feature_matching_weight": 5.0,
                    "stft_fft_sizes": [128, 256, 512, 1024, 2048],
                    "pqmf_enabled": True,
                    "chroma_enabled": False,
                },
            },
        },
    )
    config["experiment"] = {
        "name": f"task_audio_pmf_prior512d18_{stage}",
        "seed": 1234,
        "stage": stage,
    }
    config["ema"] = {"decay": 0.999, "update_every": 1, "start_step": 0}
    _deep_update(
        config["training"],
        {
            "learning_rate": 1.0e-4,
            "weight_decay": 0.01,
            "num_warmup_steps": 1_000,
            "scheduler": "constant_with_warmup",
            "grad_accumulation_steps": 1,
            "grad_clip_norm": 1.0,
            "mixed_precision": "bf16",
            "auto_resume": True,
            "discriminator": {
                "learning_rate": 1.0e-4,
                "betas": [0.8, 0.99],
                "weight_decay": 0.0,
                "scheduler": "constant",
                "grad_clip_norm": 1.0,
                "updates_per_generator": 1,
                "update_every": 1,
                "reset_on_resume": False,
            },
        },
    )
    config["logging"] = {
        "log_every": 50,
        "save_every": 1_000,
        "milestone_every": 10_000,
        "keep_last_n_checkpoints": 2,
    }
    config["sampling"] = {
        "seconds": 10.0,
        "prior_steps": 32,
        "wave_steps": 1,
        "solver": "euler",
        "cfg_strength": 2.0,
        "cfg_rescale": 0.0,
        "sample_every": 0,
        "seed": 1234,
        "captions": [],
    }
    config["encoder_eval"] = {
        "every": 2_000,
        "dataset": "esc50",
        "root": (
            "/nativemm2/share/cpfs/ljl542081/latenTTA/"
            "latent_tta_shi/report/esc50/ESC-50-master"
        ),
        "auto_download": False,
        "fold": 1,
        "epochs": 100,
        "max_seconds": 5.0,
        "encoder": "target_ema",
        "pool": "mean_time",
        "pools": ["mean_time", "mean_std_time"],
        "feature_batch_size": 24,
        "probe_batch_size": 256,
        "learning_rate": 3.0e-3,
        "weight_decay": 1.0e-4,
        "scheduler": "cosine",
        "feature_standardize": True,
        "standardize_eps": 1.0e-6,
        "num_workers": 2,
    }
    config["audiocaps_eval"] = {
        "enabled": True,
        "every": 10_000,
        "name": "audiocaps_meanaudio_957",
        "protocol_family": "meanaudio_957",
        "prior_steps": [32],
        "wave_steps": 1,
        "cfg_strength": 2.0,
        "milestone_cfg_strengths": [3.0, 4.0, 5.0],
        "milestone_cfg_steps": [100_000, 200_000, 210_000, 220_000],
        "generation_batch_sizes": [64, 32, 16, 8, 4, 2, 1],
        "num_shards": 4,
        "expected_count": 957,
        "timeout_seconds": 21_600,
        "failure_policy": "raise",
    }
    config["initialization"] = {"tts_checkpoint": None}
    return config


def build_stage1_config() -> dict[str, Any]:
    config = _common(_load_frozen(HISTORICAL_STAGE1), stage="stage1")
    data = config["data"]
    data.update(
        {
            "dataset": "audiosetcaps_wavcaps_duration_mixed",
            "sample_rate": 16_000,
            "max_audio_seconds": 20.0,
            "drop_last": True,
            "resampling": {"method": "linear_legacy"},
            # Strict external-caption matches after excluding both evaluation
            # protocols. AudioCaps train IDs remain allowed.
            "expected_audiosetcaps_rows": 1_729_107,
            "expected_wavcaps_rows": 246_324,
            "nominal_updates_per_epoch": 10_000,
            "source_update_weights": {
                "audiosetcaps": 0.5,
                "wavcaps": 0.5,
            },
            "duration_batch_sizes": {
                "audioset_10": int(os.environ.get("PMF_BS_AUDIOSET10", "2")),
                "wavcaps_0_5": int(os.environ.get("PMF_BS_WAVCAPS_0_5", "2")),
                "wavcaps_5_10": int(os.environ.get("PMF_BS_WAVCAPS_5_10", "2")),
                "wavcaps_10_15": int(os.environ.get("PMF_BS_WAVCAPS_10_15", "2")),
                "wavcaps_15_20": int(os.environ.get("PMF_BS_WAVCAPS_15_20", "2")),
            },
        }
    )
    audioset = dict(data.get("audioset") or {})
    caption = dict(audioset.get("caption") or {})
    caption.update(
        {
            "mode": "audiosetcaps",
            "fallback": "labels",
            "require_match": True,
        }
    )
    audioset["caption"] = caption
    data["audioset"] = audioset
    config["training"].update(
        {
            "batch_size": int(data["duration_batch_sizes"]["audioset_10"]),
            "num_workers": int(os.environ.get("PMF_DATA_WORKERS", "2")),
            "max_steps": 200_000,
            "resume_from": None,
        }
    )
    return config


def build_stage2_config() -> dict[str, Any]:
    config = _common(_load_frozen(HISTORICAL_STAGE2), stage="stage2")
    config["data"].update(
        {
            "dataset": "audiocaps",
            "sample_rate": 16_000,
            "max_audio_seconds": 10.5,
            "drop_last": True,
            "resampling": {"method": "linear_legacy"},
        }
    )
    config["training"].update(
        {
            "batch_size": int(os.environ.get("PMF_BS_AUDIOCAPS10_5", "2")),
            "num_workers": int(os.environ.get("PMF_DATA_WORKERS", "2")),
            "max_steps": 220_000,
            "resume_from": None,
        }
    )
    return config


def _as_xpred(config: dict[str, Any], *, stage: str) -> dict[str, Any]:
    """Replace only the pMF objective with the historical TaskAudio x-v loss."""

    model = config["model"]
    model["architecture"] = "latent_tta_xpred"
    model["prediction"] = "x_pred_v_loss"
    for key in (
        "pmf_fm_proportion",
        "pmf_norm_p",
        "pmf_norm_eps",
        "pmf_main_weight",
        "pmf_auxiliary_v_weight",
        "pmf_min_time",
    ):
        model.pop(key, None)
    config["experiment"]["name"] = f"task_audio_xpred_prior512d18_{stage}"
    config["audiocaps_eval"]["model_name"] = "TaskAudio x-pred"
    return config


def build_xpred_stage1_config() -> dict[str, Any]:
    return _as_xpred(build_stage1_config(), stage="stage1")


def build_xpred_stage2_config() -> dict[str, Any]:
    return _as_xpred(build_stage2_config(), stage="stage2")


__all__ = [
    "HISTORICAL_STAGE1",
    "HISTORICAL_STAGE2",
    "build_stage1_config",
    "build_stage2_config",
    "build_xpred_stage1_config",
    "build_xpred_stage2_config",
]
