"""Model-size and decoder architecture definitions for text-to-audio.

The base experiment config selects an architecture; it does not describe its
layers. Deterministic decoding mirrors the encoder, while FM decoding selects
either a DiT or U-Net waveform backbone.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


# Production training exposes exactly ``small`` and ``medium``. ``debug`` is a
# lightweight fixture kept for CPU/unit tests and is rejected by production
# launchers.
PRODUCTION_MODEL_SIZES: tuple[str, ...] = ("small", "medium")


MODEL_SIZE_PRESETS: dict[str, dict[str, Any]] = {
    "debug": {
        "token_dim": 64,
        "latent_dim": 32,
        "hidden_dim": 128,
        "text_dim": 128,
        "encoder_same_depth": 1,
        "encoder_down_depths": [1, 1],
        "encoder": {"heads": 4, "dim_head": 32, "ffn_mult": 4},
        "prior_fm": {
            "depth": 2,
            "fused_depth": 1,
            "hidden_dim": 128,
            "adaln_every": 2,
            "heads": 4,
            "dim_head": 32,
            "ffn_mult": 4,
        },
    },
    "small": {
        "token_dim": 512,
        "latent_dim": 64,
        "hidden_dim": 512,
        "text_dim": 512,
        "encoder_same_depth": 12,
        "encoder_down_depths": [1, 1],
        "encoder": {"heads": 8, "dim_head": 64, "ffn_mult": 4},
        "prior_fm": {
            "depth": 16,
            "fused_depth": 10,
            "hidden_dim": 512,
            "adaln_every": 2,
            "heads": 8,
            "dim_head": 64,
            "ffn_mult": 4,
        },
    },
    "medium": {
        "token_dim": 768,
        "latent_dim": 128,
        "hidden_dim": 768,
        "text_dim": 768,
        "encoder_same_depth": 16,
        "encoder_down_depths": [2, 2],
        "encoder": {"heads": 12, "dim_head": 64, "ffn_mult": 2},
        "prior_fm": {
            "depth": 18,
            "fused_depth": 12,
            "hidden_dim": 768,
            "adaln_every": 2,
            "heads": 12,
            "dim_head": 64,
            "ffn_mult": 2,
        },
    },
}


# Deterministic decoding has no backbone choice: its depth and progressive
# upsample stages are derived from the selected encoder in
# ``resolve_decoder_config``. Its output head may be "patch" or "istft".
DETERMINISTIC_DECODER: dict[str, Any] = {
    "head": "patch",
}

# FM decoding chooses one of these two waveform backbones. FM width/attention
# (and DiT depth) follow the selected prior size. DiT head options are "patch"
# and "istft"; U-Net only supports "patch". input_fusion may be "add" or
# "projected_concat".
FM_DIT_DECODER: dict[str, Any] = {
    "head": "patch",
    "input_fusion": "add",
}

FM_UNET_DECODER: dict[str, Any] = {
    "head": "patch",
    "input_fusion": "add",
    "depths": [4, 4, 8, 4, 4],
}

DECODER_ARCHITECTURES: dict[str, dict[str, Any]] = {
    "deterministic": DETERMINISTIC_DECODER,
    "fm_dit": FM_DIT_DECODER,
    "fm_unet": FM_UNET_DECODER,
}


def resolve_decoder_latent_noise_config(value: Any) -> dict[str, Any]:
    """Normalize the shared decoder-latent training-noise configuration."""

    if value is None:
        raw: dict[str, Any] = {}
    elif isinstance(value, dict):
        raw = deepcopy(value)
    else:
        raise TypeError("model.decoder.latent_noise must be a mapping")
    unknown = sorted(set(raw).difference({"mode", "std", "t_start", "probability"}))
    if unknown:
        raise ValueError(
            "unknown model.decoder.latent_noise fields: " + ", ".join(unknown)
        )

    mode = str(raw.get("mode", "none")).strip().lower().replace("-", "_")
    aliases = {
        "off": "none",
        "false": "none",
        "0": "none",
        "std": "additive",
        "gaussian": "additive",
        "variance_preserving": "vp",
        "vp_noise": "vp",
        "t": "flow_t",
        "path": "flow_t",
        "icplan": "flow_t",
        "unite_path": "unite",
        "unite_noising": "unite",
    }
    mode = aliases.get(mode, mode)
    allowed_modes = {"none", "additive", "vp", "ul_fixed", "flow_t", "unite"}
    if mode not in allowed_modes:
        raise ValueError(
            "model.decoder.latent_noise.mode must be one of: "
            + ", ".join(sorted(allowed_modes))
        )

    std = float(raw.get("std", 0.0))
    t_start = float(raw.get("t_start", 1.0 if mode == "none" else 0.85))
    probability = float(raw.get("probability", 0.0 if mode == "none" else 1.0))
    if std < 0.0:
        raise ValueError("model.decoder.latent_noise.std must be non-negative")
    if not 0.0 <= t_start <= 1.0:
        raise ValueError("model.decoder.latent_noise.t_start must be in [0, 1]")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("model.decoder.latent_noise.probability must be in [0, 1]")
    if mode in {"vp", "ul_fixed"} and std >= 1.0:
        raise ValueError("model.decoder.latent_noise.std must be < 1 for VP noise")

    return {
        "mode": mode,
        "std": std,
        "t_start": t_start,
        "probability": probability,
    }


def resolve_decoder_config(model: dict[str, Any]) -> dict[str, Any]:
    """Resolve deterministic, FM-DiT, or FM-U-Net decoder configuration."""

    raw_value = model.get("decoder", {})
    if isinstance(raw_value, str):
        name = raw_value.strip().lower().replace("-", "_")
        if name == "deterministic":
            raw: dict[str, Any] = {"type": "deterministic"}
        elif name in {"fm_dit", "fm_unet"}:
            raw = {"type": "fm", "backbone": name.removeprefix("fm_")}
        else:
            raise ValueError(
                "model.decoder string must be one of: deterministic, fm_dit, fm_unet"
            )
    elif isinstance(raw_value, dict):
        raw = deepcopy(raw_value)
    else:
        raise TypeError("model.decoder must be a mapping or decoder-name string")

    latent_noise = resolve_decoder_latent_noise_config(raw.pop("latent_noise", None))
    reconstruction_latent_source = (
        str(raw.pop("reconstruction_latent_source", "full_online"))
        .strip()
        .lower()
        .replace("-", "_")
    )
    if reconstruction_latent_source not in {"full_online", "mask_hybrid"}:
        raise ValueError(
            "model.decoder.reconstruction_latent_source must be "
            "'full_online' or 'mask_hybrid'"
        )

    decoder_type = (
        str(raw.get("type", "deterministic")).strip().lower().replace("-", "_")
    )
    if decoder_type not in {"deterministic", "fm"}:
        raise ValueError("model.decoder.type must be 'fm' or 'deterministic'")

    if decoder_type == "deterministic":
        forbidden = {
            "backbone",
            "depth",
            "depths",
            "input_fusion",
            "upsample_depths",
            "unet_depths",
            "up_depths",
            "same_depth",
        }
        configured = sorted(forbidden.intersection(raw))
        if configured:
            raise ValueError(
                "deterministic decoder mirrors the encoder and does not accept "
                f"architecture fields: {', '.join(configured)}"
            )
        resolved = {
            "type": "deterministic",
            **deepcopy(DETERMINISTIC_DECODER),
        }
        if "head" in raw:
            resolved["head"] = deepcopy(raw["head"])
        unknown = sorted(set(raw).difference({"type", "head"}))
        if unknown:
            raise ValueError(
                f"unknown deterministic decoder fields: {', '.join(unknown)}"
            )

        encoder_same_depth = int(model.get("encoder_same_depth", 0))
        encoder_down_depths = [
            int(depth) for depth in model.get("encoder_down_depths", ())
        ]
        if encoder_same_depth < 1:
            raise ValueError(
                "model.encoder_same_depth must be positive for deterministic decoding"
            )
        if not encoder_down_depths or min(encoder_down_depths) < 0:
            raise ValueError(
                "model.encoder_down_depths must contain non-negative integers "
                "for deterministic decoding"
            )
        resolved["depth"] = encoder_same_depth
        resolved["upsample_depths"] = list(reversed(encoder_down_depths))

        head = str(resolved["head"]).strip().lower().replace("-", "_")
        if head not in {"patch", "istft"}:
            raise ValueError("model.decoder.head must be 'patch' or 'istft'")
        resolved["head"] = head
        resolved["latent_noise"] = latent_noise
        resolved["reconstruction_latent_source"] = reconstruction_latent_source
        return resolved

    backbone = str(raw.get("backbone", "dit")).strip().lower().replace("-", "_")
    if backbone not in {"dit", "unet"}:
        raise ValueError("FM model.decoder.backbone must be 'dit' or 'unet'")

    aliases = {"unet_depths": "depths"}
    for old_name, new_name in aliases.items():
        if old_name in raw:
            if new_name in raw:
                raise ValueError(
                    f"model.decoder cannot set both {old_name!r} and {new_name!r}"
                )
            raw[new_name] = raw.pop(old_name)

    allowed = {
        "type",
        "backbone",
        "head",
        "input_fusion",
        "depth",
        "depths",
        "hidden_dim",
        "adaln_every",
        "heads",
        "dim_head",
        "ffn_mult",
    }
    unknown = sorted(set(raw).difference(allowed))
    if unknown:
        raise ValueError(f"unknown FM decoder fields: {', '.join(unknown)}")
    if backbone == "dit" and "depths" in raw:
        raise ValueError("FM DiT decoder does not accept model.decoder.depths")
    if backbone == "unet" and "depth" in raw:
        raise ValueError("FM U-Net decoder does not accept model.decoder.depth")

    resolved: dict[str, Any] = {
        "type": "fm",
        "backbone": backbone,
        **deepcopy(DECODER_ARCHITECTURES[f"fm_{backbone}"]),
    }
    prior = model.get("prior_fm", {})
    if not isinstance(prior, dict):
        raise TypeError("model.prior_fm must be a mapping")
    for name in ("hidden_dim", "adaln_every", "heads", "dim_head", "ffn_mult"):
        if name not in prior:
            raise ValueError(f"model.prior_fm.{name} is required for the FM decoder")
        resolved[name] = deepcopy(prior[name])
    if backbone == "dit":
        if "depth" not in prior:
            raise ValueError("model.prior_fm.depth is required for the FM DiT decoder")
        resolved["depth"] = int(prior["depth"])

    for name, value in raw.items():
        if name not in {"type", "backbone"}:
            resolved[name] = deepcopy(value)

    head = str(resolved.get("head", "patch")).strip().lower().replace("-", "_")
    if head not in {"patch", "istft"}:
        raise ValueError("model.decoder.head must be 'patch' or 'istft'")
    if backbone == "unet" and head != "patch":
        raise ValueError(
            "model.decoder.head must be 'patch' when decoder.backbone is 'unet'"
        )
    resolved["head"] = head

    if backbone == "unet":
        depths = [int(depth) for depth in resolved.get("depths", ())]
        if len(depths) != 5 or min(depths, default=0) < 1:
            raise ValueError(
                "model.decoder.depths must contain five positive integers for UNet"
            )
        resolved["depths"] = depths
    else:
        depth = int(resolved.get("depth", 0))
        if depth < 1:
            raise ValueError("model.decoder.depth must be positive for DiT")
        resolved["depth"] = depth

    resolved["latent_noise"] = latent_noise
    resolved["reconstruction_latent_source"] = reconstruction_latent_source

    return resolved


__all__ = [
    "DECODER_ARCHITECTURES",
    "DETERMINISTIC_DECODER",
    "FM_DIT_DECODER",
    "FM_UNET_DECODER",
    "MODEL_SIZE_PRESETS",
    "PRODUCTION_MODEL_SIZES",
    "resolve_decoder_config",
    "resolve_decoder_latent_noise_config",
]
