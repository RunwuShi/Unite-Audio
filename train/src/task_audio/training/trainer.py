from __future__ import annotations

import json
import gc
import hashlib
import inspect
import math
import os
import shutil
import time
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from task_audio.mask_contract import validate_mask_transition_request
from task_audio.models.losses import (
    MultiViewSTFTPQMFAudioDiscriminator,
    discriminator_feature_matching_loss,
    masked_waveform_l1,
    relativistic_discriminator_loss,
    relativistic_generator_loss,
)
from task_audio.models.xcodec_gan import (
    XCodecMPDSpecDiscriminator,
    feature_matching_sum,
    lsgan_discriminator_sum,
    lsgan_generator_sum,
)

from .checkpointing import (
    checkpoint_model_state,
    load_tta_model_state,
    resolve_checkpoint_dir,
)
from .config_io import save_config_snapshot, to_jsonable


def _distillation_mono_3d(value: Tensor, *, name: str) -> Tensor:
    if value.ndim == 2:
        value = value.unsqueeze(1)
    if value.ndim != 3 or value.shape[1] != 1:
        raise ValueError(f"{name} must have shape [batch, time] or [batch, 1, time]")
    return value


def selective_decoder_distillation_loss(
    current: Tensor,
    reference: Tensor,
    valid_mask: Tensor,
    *,
    sample_rate: int,
    lowpass_hz: float,
    envelope_ms: tuple[float, ...],
    lowpass_weight: float,
    envelope_weight: float,
    transient_weight: float,
) -> dict[str, Tensor]:
    """Preserve teacher structure without copying its full-band waveform.

    The low-pass term anchors coarse waveform structure below ``lowpass_hz``.
    Multi-scale absolute-amplitude envelopes preserve loudness and event timing,
    while envelope differences preserve transient placement.  Frequencies above
    the low-pass transition remain free to follow real-audio reconstruction and
    adversarial objectives instead of being copied from the teacher.
    """

    current = _distillation_mono_3d(current, name="current")
    reference = _distillation_mono_3d(reference, name="reference")
    valid = _distillation_mono_3d(valid_mask, name="valid_mask").to(
        device=current.device, dtype=current.dtype
    )
    if current.shape != reference.shape or current.shape != valid.shape:
        raise ValueError("current, reference, and valid_mask must have matching shapes")
    if current.shape[-1] < 2:
        raise ValueError("distillation waveforms must contain at least two samples")
    if sample_rate < 2 or not 0.0 < lowpass_hz < 0.5 * float(sample_rate):
        raise ValueError("lowpass_hz must be within (0, Nyquist)")
    if not envelope_ms or any(value <= 0.0 for value in envelope_ms):
        raise ValueError("envelope_ms must contain positive values")
    weights = (float(lowpass_weight), float(envelope_weight), float(transient_weight))
    if min(weights) < 0.0 or sum(weights) <= 0.0:
        raise ValueError("selective distillation weights must be non-negative")

    context = (
        torch.autocast(device_type=current.device.type, enabled=False)
        if current.device.type in {"cpu", "cuda"}
        else nullcontext()
    )
    with context:
        current_float = current.float() * valid.float()
        reference_float = reference.detach().float() * valid.float()
        samples = current_float.shape[-1]
        frequencies = torch.fft.rfftfreq(
            samples,
            d=1.0 / float(sample_rate),
            device=current.device,
        )
        transition_hz = max(100.0, 0.25 * float(lowpass_hz))
        transition_end = min(
            0.5 * float(sample_rate), float(lowpass_hz) + transition_hz
        )
        response = torch.ones_like(frequencies)
        above = frequencies > float(lowpass_hz)
        if transition_end > float(lowpass_hz):
            phase = (
                (frequencies - float(lowpass_hz))
                / (transition_end - float(lowpass_hz))
            ).clamp(0.0, 1.0)
            response = torch.where(
                above,
                0.5 * (1.0 + torch.cos(math.pi * phase)),
                response,
            )
        response = torch.where(frequencies >= transition_end, 0.0, response)
        current_low = torch.fft.irfft(
            torch.fft.rfft(current_float, dim=-1) * response,
            n=samples,
            dim=-1,
        )
        reference_low = torch.fft.irfft(
            torch.fft.rfft(reference_float, dim=-1) * response,
            n=samples,
            dim=-1,
        )
        lowpass = masked_waveform_l1(current_low, reference_low, valid)

        envelope_losses: list[Tensor] = []
        transient_losses: list[Tensor] = []
        for milliseconds in envelope_ms:
            kernel = min(
                samples,
                max(2, int(round(float(milliseconds) * float(sample_rate) / 1_000.0))),
            )
            stride = max(1, kernel // 2)
            pooled_valid = F.avg_pool1d(valid.float(), kernel, stride=stride)
            denominator = pooled_valid.clamp_min(1.0e-4)
            current_envelope = (
                F.avg_pool1d(current_float.abs(), kernel, stride=stride)
                / denominator
            )
            reference_envelope = (
                F.avg_pool1d(reference_float.abs(), kernel, stride=stride)
                / denominator
            )
            envelope_valid = pooled_valid >= 0.5
            envelope_losses.append(
                masked_waveform_l1(
                    current_envelope,
                    reference_envelope,
                    envelope_valid,
                )
            )
            if current_envelope.shape[-1] > 1:
                transient_valid = envelope_valid[..., 1:] & envelope_valid[..., :-1]
                transient_losses.append(
                    masked_waveform_l1(
                        current_envelope[..., 1:] - current_envelope[..., :-1],
                        reference_envelope[..., 1:] - reference_envelope[..., :-1],
                        transient_valid,
                    )
                )

        envelope = torch.stack(envelope_losses).mean()
        transient = (
            torch.stack(transient_losses).mean()
            if transient_losses
            else lowpass.new_tensor(0.0)
        )
        normalizer = sum(weights)
        loss = (
            weights[0] * lowpass
            + weights[1] * envelope
            + weights[2] * transient
        ) / normalizer
    return {
        "loss": loss,
        "lowpass": lowpass,
        "envelope": envelope,
        "transient": transient,
    }


@dataclass
class TTATrainerConfig:
    output_dir: str
    max_steps: int = 100_000
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1_000
    scheduler: str = "constant_with_warmup"
    grad_accumulation_steps: int = 1
    grad_clip_norm: float = 1.0
    decoder_distillation_weight: float = 0.0
    decoder_distillation_mode: str = "waveform_l1"
    decoder_distillation_sample_rate: int = 16_000
    decoder_distillation_lowpass_hz: float = 1_000.0
    decoder_distillation_envelope_ms: tuple[float, ...] = (10.0, 40.0)
    decoder_distillation_lowpass_weight: float = 1.0
    decoder_distillation_envelope_weight: float = 0.5
    decoder_distillation_transient_weight: float = 0.25
    parameter_anchor_weight: float = 0.0
    max_consecutive_non_finite_gradients: int = 8
    mixed_precision: str = "no"
    find_unused_parameters: bool = False
    seed: int = 1234
    log_every: int = 50
    save_every: int = 5_000
    save_final_checkpoint: bool = True
    milestone_every: int = 0
    keep_last_n_checkpoints: int = 2
    auto_resume: bool = True
    resume_from: str | None = None
    warm_start_from: str | None = None
    auto_continue_data_cursor: bool = True
    reset_data_cursor: bool = False
    allow_batch_size_cursor_migration: bool = False
    data_batch_size: int = 4
    data_num_workers: int = 2
    data_config_fingerprint: str = ""
    mask_contract_id: str | None = None
    mask_contract_fingerprint: str | None = None
    mask_transition: dict[str, Any] = field(default_factory=dict)
    ema_decay: float = 0.0
    ema_update_every: int = 1
    ema_start_step: int = 0
    ema_trainable_only: bool = False
    update_internal_ema: bool = True
    sample_every: int = 0
    sample_captions: tuple[str, ...] = field(default_factory=tuple)
    sample_seconds: float = 10.0
    sample_prior_steps: int | None = None
    sample_wave_steps: int | None = None
    sample_solver: str = "euler"
    sample_cfg_strength: float = 2.0
    sample_wave_cfg_strength: float | None = None
    sample_cfg_rescale: float = 0.0
    sample_seed: int = 1234
    encoder_eval_every: int = 0
    encoder_eval_config: dict[str, Any] = field(default_factory=dict)
    audiocaps_eval_every: int = 0
    audiocaps_eval_config: dict[str, Any] = field(default_factory=dict)
    model_diagnostics_every: int = 0
    model_diagnostics_config: dict[str, Any] = field(default_factory=dict)
    adversarial_enabled: bool = False
    adversarial_backend: str = "multiview_stft_pqmf_v1"
    adversarial_mixed_precision: bool = False
    adversarial_objective: str = "relativistic_paired"
    adversarial_start_step: int = 10_000
    adversarial_weight_warmup_steps: int = 0
    adversarial_discriminator_weight: float = 1.0
    adversarial_generator_weight: float = 0.1
    adversarial_feature_matching_weight: float = 2.0
    adversarial_weight_transition: dict[str, Any] = field(default_factory=dict)
    adversarial_stft_fft_sizes: tuple[int, ...] = (128, 256, 512, 1024, 2048)
    adversarial_pqmf_enabled: bool = True
    adversarial_mpd_enabled: bool = False
    adversarial_mpd_periods: tuple[int, ...] = (2, 3, 5, 7, 11)
    adversarial_msd_enabled: bool = False
    adversarial_msd_scales: int = 3
    adversarial_gan_batch_size: int = 0
    artifact_discriminator_enabled: bool = False
    artifact_discriminator_mixed_precision: bool = True
    artifact_discriminator_stft_fft_sizes: tuple[int, ...] = (4096,)
    artifact_discriminator_mpd_periods: tuple[int, ...] = (2, 3, 5, 7, 11)
    artifact_discriminator_learning_rate: float = 1.0e-4
    artifact_discriminator_betas: tuple[float, float] = (0.8, 0.99)
    artifact_discriminator_weight_decay: float = 0.0
    artifact_family_target_weight: float = 0.5
    artifact_family_warmup_steps: int = 2_000
    artifact_family_combination: str = "convex"
    artifact_family_beta: float = 1.0
    artifact_loss_transition: dict[str, Any] = field(default_factory=dict)
    artifact_bootstrap_source_fingerprint: str = ""
    discriminator_learning_rate: float = 1.0e-4
    discriminator_betas: tuple[float, float] = (0.8, 0.99)
    discriminator_weight_decay: float = 0.0
    discriminator_scheduler: str = "constant"
    discriminator_grad_clip_norm: float = 1.0
    discriminator_updates_per_generator: int = 1
    discriminator_update_every: int = 2
    discriminator_reset_on_resume: bool = True

    def __post_init__(self) -> None:
        if not self.output_dir:
            raise ValueError("TTATrainerConfig.output_dir is required")
        if self.max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.grad_accumulation_steps < 1:
            raise ValueError("grad_accumulation_steps must be >= 1")
        if self.decoder_distillation_weight < 0.0:
            raise ValueError("decoder_distillation_weight must be non-negative")
        if self.decoder_distillation_mode not in {"waveform_l1", "selective"}:
            raise ValueError(
                "decoder_distillation_mode must be 'waveform_l1' or 'selective'"
            )
        if self.decoder_distillation_sample_rate < 2:
            raise ValueError("decoder_distillation_sample_rate must be at least 2")
        if not (
            0.0
            < self.decoder_distillation_lowpass_hz
            < 0.5 * float(self.decoder_distillation_sample_rate)
        ):
            raise ValueError(
                "decoder_distillation_lowpass_hz must be within (0, Nyquist)"
            )
        if not self.decoder_distillation_envelope_ms or any(
            value <= 0.0 for value in self.decoder_distillation_envelope_ms
        ):
            raise ValueError(
                "decoder_distillation_envelope_ms must contain positive values"
            )
        selective_weights = (
            self.decoder_distillation_lowpass_weight,
            self.decoder_distillation_envelope_weight,
            self.decoder_distillation_transient_weight,
        )
        if min(selective_weights) < 0.0 or sum(selective_weights) <= 0.0:
            raise ValueError(
                "selective decoder distillation weights must be non-negative "
                "with a positive sum"
            )
        if self.parameter_anchor_weight < 0.0:
            raise ValueError("parameter_anchor_weight must be non-negative")
        if self.max_consecutive_non_finite_gradients < 1:
            raise ValueError(
                "max_consecutive_non_finite_gradients must be positive"
            )
        if self.log_every < 1:
            raise ValueError("log_every must be >= 1")
        if self.save_every < 0 or self.milestone_every < 0:
            raise ValueError("save_every and milestone_every must be non-negative")
        if self.keep_last_n_checkpoints < -1:
            raise ValueError("keep_last_n_checkpoints must be >= -1")
        if self.resume_from and self.warm_start_from:
            raise ValueError("resume_from and warm_start_from are mutually exclusive")
        if self.data_batch_size < 1:
            raise ValueError("data_batch_size must be positive")
        if self.data_num_workers < 0:
            raise ValueError("data_num_workers must be non-negative")
        if self.ema_decay != 0.0 and not 0.0 < self.ema_decay < 1.0:
            raise ValueError("ema_decay must be 0 (disabled) or in (0, 1)")
        if self.ema_update_every < 1:
            raise ValueError("ema_update_every must be >= 1")
        if self.ema_start_step < 0:
            raise ValueError("ema_start_step must be non-negative")
        if self.encoder_eval_every < 0:
            raise ValueError("encoder_eval_every must be non-negative")
        if self.audiocaps_eval_every < 0:
            raise ValueError("audiocaps_eval_every must be non-negative")
        if self.model_diagnostics_every < 0:
            raise ValueError("model_diagnostics_every must be non-negative")
        self.adversarial_enabled = bool(self.adversarial_enabled)
        self.adversarial_backend = (
            str(self.adversarial_backend).strip().lower().replace("-", "_")
        )
        if self.adversarial_backend not in {
            "multiview_stft_pqmf_v1",
            "xcodec_mpd_spec_v1",
        }:
            raise ValueError(
                "adversarial_backend must be multiview_stft_pqmf_v1 or "
                "xcodec_mpd_spec_v1"
            )
        self.adversarial_objective = (
            str(self.adversarial_objective).strip().lower().replace("-", "_")
        )
        if self.adversarial_objective not in {
            "relativistic_paired",
            "lsgan_sum",
        }:
            raise ValueError(
                "adversarial_objective must be relativistic_paired or lsgan_sum"
            )
        expected_objective = (
            "lsgan_sum"
            if self.adversarial_backend == "xcodec_mpd_spec_v1"
            else "relativistic_paired"
        )
        if self.adversarial_objective != expected_objective:
            raise ValueError(
                f"{self.adversarial_backend} requires objective={expected_objective}"
            )
        if self.adversarial_start_step < 0:
            raise ValueError("adversarial_start_step must be non-negative")
        if self.adversarial_weight_warmup_steps < 0:
            raise ValueError("adversarial_weight_warmup_steps must be non-negative")
        if (
            min(
                self.adversarial_discriminator_weight,
                self.adversarial_generator_weight,
                self.adversarial_feature_matching_weight,
            )
            < 0.0
        ):
            raise ValueError("adversarial weights must be non-negative")
        self.adversarial_weight_transition = dict(
            self.adversarial_weight_transition
        )
        if bool(self.adversarial_weight_transition.get("enabled", False)):
            transition = self.adversarial_weight_transition
            source_fingerprint = str(
                transition.get("source_fingerprint", "")
            ).strip()
            if len(source_fingerprint) != 64:
                raise ValueError(
                    "adversarial weight transition requires a 64-character "
                    "source_fingerprint"
                )
            source_step = int(transition.get("source_step", -1))
            if source_step < self.adversarial_start_step:
                raise ValueError(
                    "adversarial weight transition source_step must be at or "
                    "after adversarial_start_step"
                )
            transition_steps = int(transition.get("transition_steps", 0))
            if transition_steps < 0:
                raise ValueError(
                    "adversarial weight transition transition_steps must be "
                    "non-negative"
                )
            source_generator_weight = float(
                transition.get("source_generator_weight", -1.0)
            )
            source_feature_matching_weight = float(
                transition.get("source_feature_matching_weight", -1.0)
            )
            if min(source_generator_weight, source_feature_matching_weight) < 0.0:
                raise ValueError(
                    "adversarial weight transition source weights must be "
                    "non-negative"
                )
            transition.update(
                {
                    "enabled": True,
                    "source_fingerprint": source_fingerprint,
                    "source_step": source_step,
                    "transition_steps": transition_steps,
                    "source_generator_weight": source_generator_weight,
                    "source_feature_matching_weight": (
                        source_feature_matching_weight
                    ),
                    "reason": str(transition.get("reason", "")),
                }
            )
        self.adversarial_stft_fft_sizes = tuple(
            int(value) for value in self.adversarial_stft_fft_sizes
        )
        if not self.adversarial_stft_fft_sizes or any(
            value < 4 or value % 2 for value in self.adversarial_stft_fft_sizes
        ):
            raise ValueError(
                "adversarial_stft_fft_sizes must contain positive even integers >= 4"
            )
        self.adversarial_mpd_periods = tuple(
            int(value) for value in self.adversarial_mpd_periods
        )
        if self.adversarial_mpd_enabled and (
            not self.adversarial_mpd_periods
            or any(value < 2 for value in self.adversarial_mpd_periods)
        ):
            raise ValueError("adversarial_mpd_periods must contain integers >= 2")
        if self.adversarial_msd_enabled and self.adversarial_msd_scales < 1:
            raise ValueError("adversarial_msd_scales must be positive")
        if self.adversarial_gan_batch_size < 0:
            raise ValueError("adversarial_gan_batch_size must be non-negative")
        self.artifact_discriminator_enabled = bool(
            self.artifact_discriminator_enabled
        )
        self.artifact_discriminator_stft_fft_sizes = tuple(
            int(value) for value in self.artifact_discriminator_stft_fft_sizes
        )
        if self.artifact_discriminator_enabled and (
            not self.artifact_discriminator_stft_fft_sizes
            or any(
                value < 4 or value % 2
                for value in self.artifact_discriminator_stft_fft_sizes
            )
        ):
            raise ValueError(
                "artifact_discriminator_stft_fft_sizes must contain positive "
                "even integers >= 4"
            )
        self.artifact_discriminator_mpd_periods = tuple(
            int(value) for value in self.artifact_discriminator_mpd_periods
        )
        if self.artifact_discriminator_enabled and (
            not self.artifact_discriminator_mpd_periods
            or any(
                value < 2 for value in self.artifact_discriminator_mpd_periods
            )
        ):
            raise ValueError(
                "artifact_discriminator_mpd_periods must contain integers >= 2"
            )
        self.artifact_discriminator_betas = tuple(
            float(value) for value in self.artifact_discriminator_betas
        )
        if len(self.artifact_discriminator_betas) != 2 or not all(
            0.0 <= value < 1.0
            for value in self.artifact_discriminator_betas
        ):
            raise ValueError(
                "artifact_discriminator_betas must contain two values in [0, 1)"
            )
        if self.artifact_discriminator_learning_rate <= 0.0:
            raise ValueError(
                "artifact_discriminator_learning_rate must be positive"
            )
        if self.artifact_discriminator_weight_decay < 0.0:
            raise ValueError(
                "artifact_discriminator_weight_decay must be non-negative"
            )
        if not 0.0 <= self.artifact_family_target_weight <= 1.0:
            raise ValueError("artifact_family_target_weight must be in [0, 1]")
        if self.artifact_family_warmup_steps < 0:
            raise ValueError(
                "artifact_family_warmup_steps must be non-negative"
            )
        self.artifact_family_combination = (
            str(self.artifact_family_combination)
            .strip()
            .lower()
            .replace("-", "_")
        )
        if self.artifact_family_combination not in {"convex", "additive"}:
            raise ValueError(
                "artifact_family_combination must be 'convex' or 'additive'"
            )
        if self.artifact_family_beta < 0.0:
            raise ValueError("artifact_family_beta must be non-negative")
        self.artifact_loss_transition = dict(
            self.artifact_loss_transition
        )
        if bool(self.artifact_loss_transition.get("enabled", False)):
            transition = self.artifact_loss_transition
            source_fingerprint = str(
                transition.get(
                    "source_artifact_discriminator_fingerprint", ""
                )
            ).strip()
            if len(source_fingerprint) != 64:
                raise ValueError(
                    "artifact loss transition requires a 64-character "
                    "source_artifact_discriminator_fingerprint"
                )
            source_step = int(transition.get("source_step", -1))
            if source_step < 0:
                raise ValueError(
                    "artifact loss transition source_step must be non-negative"
                )
            transition.update(
                {
                    "enabled": True,
                    "source_artifact_discriminator_fingerprint": (
                        source_fingerprint
                    ),
                    "source_step": source_step,
                    "reason": str(transition.get("reason", "")),
                }
            )
        self.artifact_bootstrap_source_fingerprint = str(
            self.artifact_bootstrap_source_fingerprint
        ).strip()
        if (
            self.artifact_discriminator_enabled
            and self.artifact_bootstrap_source_fingerprint
            and len(self.artifact_bootstrap_source_fingerprint) != 64
        ):
            raise ValueError(
                "artifact_bootstrap_source_fingerprint must be empty or a "
                "64-character fingerprint"
            )
        if self.artifact_discriminator_enabled and not self.adversarial_enabled:
            raise ValueError(
                "artifact discriminator requires adversarial training"
            )
        self.discriminator_betas = tuple(
            float(value) for value in self.discriminator_betas
        )
        if len(self.discriminator_betas) != 2 or not all(
            0.0 <= value < 1.0 for value in self.discriminator_betas
        ):
            raise ValueError("discriminator_betas must contain two values in [0, 1)")
        if self.discriminator_learning_rate <= 0.0:
            raise ValueError("discriminator_learning_rate must be positive")
        if self.discriminator_weight_decay < 0.0:
            raise ValueError("discriminator_weight_decay must be non-negative")
        self.discriminator_scheduler = (
            str(self.discriminator_scheduler).lower().replace("-", "_")
        )
        if self.discriminator_scheduler != "constant":
            raise ValueError("discriminator_scheduler must be 'constant'")
        if self.discriminator_grad_clip_norm < 0.0:
            raise ValueError("discriminator_grad_clip_norm must be non-negative")
        if self.discriminator_updates_per_generator != 1:
            raise ValueError(
                "TTA v1 requires one discriminator update per generator update"
            )
        if self.discriminator_update_every < 1:
            raise ValueError("discriminator_update_every must be positive")
        self.discriminator_reset_on_resume = bool(
            self.discriminator_reset_on_resume
        )
        normalized = self.scheduler.lower().replace("-", "_")
        if normalized not in {
            "constant",
            "constant_with_warmup",
            "cosine",
            "cosine_with_warmup",
        }:
            raise ValueError(
                "scheduler must be constant, constant_with_warmup, cosine, or cosine_with_warmup"
            )
        self.scheduler = normalized
        self.sample_captions = tuple(
            str(item) for item in self.sample_captions if str(item).strip()
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "TTATrainerConfig":
        training = dict(config.get("training", {}))
        initialization = dict(config.get("initialization", {}))
        logging = dict(config.get("logging", {}))
        sampling = dict(config.get("sampling", {}))
        encoder_eval = dict(config.get("encoder_eval", {}))
        audiocaps_eval = dict(config.get("audiocaps_eval", {}))
        model_diagnostics = dict(config.get("model_diagnostics", {}))
        ema = dict(config.get("ema", {}))
        paths = dict(config.get("paths", {}))
        experiment = dict(config.get("experiment", {}))
        model = dict(config.get("model", {}))
        data = dict(config.get("data", {}))
        mask_contract = dict(config.get("mask_contract", {}))
        loss = dict(model.get("loss", {}))
        adversarial = dict(loss.get("adversarial", {}))
        artifact_discriminator = dict(
            adversarial.get("artifact_discriminator", {})
        )
        discriminator = dict(training.get("discriminator", {}))
        preservation = dict(training.get("decoder_preservation", {}))
        captions = sampling.get("fixed_captions", sampling.get("captions", ()))
        if isinstance(captions, str):
            captions = (captions,)
        return cls(
            output_dir=str(paths.get("output_dir") or training.get("output_dir") or ""),
            max_steps=int(training.get("max_steps", 100_000)),
            learning_rate=float(training.get("learning_rate", 1.0e-4)),
            weight_decay=float(training.get("weight_decay", 0.01)),
            warmup_steps=int(
                training.get("num_warmup_steps", training.get("warmup_steps", 1_000))
            ),
            scheduler=str(
                training.get(
                    "scheduler", training.get("lr_scheduler", "constant_with_warmup")
                )
            ),
            grad_accumulation_steps=int(training.get("grad_accumulation_steps", 1)),
            grad_clip_norm=float(training.get("grad_clip_norm", 1.0)),
            decoder_distillation_weight=float(
                preservation.get("output_distillation_weight", 0.0)
            ),
            decoder_distillation_mode=str(
                preservation.get("output_distillation_mode", "waveform_l1")
            ),
            decoder_distillation_sample_rate=int(
                model.get("sample_rate", data.get("sample_rate", 16_000))
            ),
            decoder_distillation_lowpass_hz=float(
                preservation.get("lowpass_hz", 1_000.0)
            ),
            decoder_distillation_envelope_ms=tuple(
                float(value)
                for value in preservation.get("envelope_ms", (10.0, 40.0))
            ),
            decoder_distillation_lowpass_weight=float(
                preservation.get("lowpass_weight", 1.0)
            ),
            decoder_distillation_envelope_weight=float(
                preservation.get("envelope_weight", 0.5)
            ),
            decoder_distillation_transient_weight=float(
                preservation.get("transient_weight", 0.25)
            ),
            parameter_anchor_weight=float(
                preservation.get("parameter_anchor_weight", 0.0)
            ),
            max_consecutive_non_finite_gradients=int(
                training.get("max_consecutive_non_finite_gradients", 8)
            ),
            mixed_precision=str(training.get("mixed_precision", "no")),
            find_unused_parameters=bool(training.get("find_unused_parameters", False)),
            seed=int(experiment.get("seed", training.get("seed", 1234))),
            log_every=int(logging.get("log_every", training.get("log_every", 50))),
            save_every=int(
                logging.get("save_every", training.get("save_every", 5_000))
            ),
            save_final_checkpoint=bool(
                logging.get(
                    "save_final_checkpoint",
                    training.get("save_final_checkpoint", True),
                )
            ),
            milestone_every=int(
                logging.get("milestone_every", training.get("milestone_every", 0))
            ),
            keep_last_n_checkpoints=int(
                logging.get(
                    "keep_last_n_checkpoints",
                    training.get("keep_last_n_checkpoints", 2),
                )
            ),
            auto_resume=bool(training.get("auto_resume", True)),
            resume_from=training.get("resume_from"),
            warm_start_from=initialization.get("full_tta_checkpoint"),
            auto_continue_data_cursor=bool(
                training.get("auto_continue_data_cursor", True)
            ),
            reset_data_cursor=bool(training.get("reset_data_cursor", False)),
            allow_batch_size_cursor_migration=bool(
                training.get("allow_batch_size_cursor_migration", False)
            ),
            data_batch_size=int(training.get("batch_size", 4)),
            data_num_workers=int(training.get("num_workers", 2)),
            data_config_fingerprint=_data_config_fingerprint(config),
            mask_contract_id=mask_contract.get("id"),
            mask_contract_fingerprint=mask_contract.get("fingerprint"),
            mask_transition=dict(training.get("mask_transition", {})),
            ema_decay=float(training.get("ema_decay", ema.get("decay", 0.0))),
            ema_update_every=int(
                training.get("ema_update_every", ema.get("update_every", 1))
            ),
            ema_start_step=int(
                training.get("ema_start_step", ema.get("start_step", 0))
            ),
            ema_trainable_only=bool(training.get("ema_trainable_only", False)),
            update_internal_ema=bool(training.get("update_internal_ema", True)),
            sample_every=int(
                training.get(
                    "infer_every",
                    sampling.get("sample_every", sampling.get("every", 0)),
                )
            ),
            sample_captions=tuple(captions or ()),
            sample_seconds=float(sampling.get("seconds", 10.0)),
            sample_prior_steps=_optional_int(sampling.get("prior_steps")),
            sample_wave_steps=_optional_int(sampling.get("wave_steps")),
            sample_solver=str(sampling.get("solver", "euler")),
            sample_cfg_strength=float(sampling.get("cfg_strength", 2.0)),
            sample_wave_cfg_strength=(
                None
                if sampling.get("wave_cfg_strength") is None
                else float(sampling["wave_cfg_strength"])
            ),
            sample_cfg_rescale=float(sampling.get("cfg_rescale", 0.0)),
            sample_seed=int(sampling.get("seed", experiment.get("seed", 1234))),
            encoder_eval_every=int(encoder_eval.get("every", 0) or 0),
            encoder_eval_config=encoder_eval,
            audiocaps_eval_every=(
                int(audiocaps_eval.get("every", 0) or 0)
                if bool(audiocaps_eval.get("enabled", False))
                else 0
            ),
            audiocaps_eval_config=audiocaps_eval,
            model_diagnostics_every=int(
                model_diagnostics.get("every", 0) or 0
            ),
            model_diagnostics_config=model_diagnostics,
            adversarial_enabled=bool(adversarial.get("enabled", False)),
            adversarial_backend=str(
                adversarial.get("backend", "multiview_stft_pqmf_v1")
            ),
            adversarial_mixed_precision=bool(
                adversarial.get("mixed_precision", False)
            ),
            adversarial_objective=str(
                adversarial.get("objective", "relativistic_paired")
            ),
            adversarial_start_step=int(adversarial.get("start_step", 10_000)),
            adversarial_weight_warmup_steps=int(
                adversarial.get("weight_warmup_steps", 0)
            ),
            adversarial_discriminator_weight=float(
                adversarial.get("discriminator_weight", 1.0)
            ),
            adversarial_generator_weight=float(
                adversarial.get("generator_weight", 0.1)
            ),
            adversarial_feature_matching_weight=float(
                adversarial.get("feature_matching_weight", 2.0)
            ),
            adversarial_weight_transition=dict(
                adversarial.get("weight_transition", {})
            ),
            adversarial_stft_fft_sizes=tuple(
                int(value)
                for value in adversarial.get(
                    "stft_fft_sizes", (128, 256, 512, 1024, 2048)
                )
            ),
            adversarial_pqmf_enabled=bool(adversarial.get("pqmf_enabled", True)),
            adversarial_mpd_enabled=bool(adversarial.get("mpd_enabled", False)),
            adversarial_mpd_periods=tuple(
                int(value) for value in adversarial.get("mpd_periods", (2, 3, 5, 7, 11))
            ),
            adversarial_msd_enabled=bool(adversarial.get("msd_enabled", False)),
            adversarial_msd_scales=int(adversarial.get("msd_scales", 3)),
            adversarial_gan_batch_size=int(
                adversarial.get("gan_batch_size", 0)
            ),
            artifact_discriminator_enabled=bool(
                artifact_discriminator.get("enabled", False)
            ),
            artifact_discriminator_mixed_precision=bool(
                artifact_discriminator.get("mixed_precision", True)
            ),
            artifact_discriminator_stft_fft_sizes=tuple(
                int(value)
                for value in artifact_discriminator.get(
                    "stft_fft_sizes", (4096,)
                )
            ),
            artifact_discriminator_mpd_periods=tuple(
                int(value)
                for value in artifact_discriminator.get(
                    "mpd_periods", (2, 3, 5, 7, 11)
                )
            ),
            artifact_discriminator_learning_rate=float(
                artifact_discriminator.get("learning_rate", 1.0e-4)
            ),
            artifact_discriminator_betas=tuple(
                float(value)
                for value in artifact_discriminator.get(
                    "betas", (0.8, 0.99)
                )
            ),
            artifact_discriminator_weight_decay=float(
                artifact_discriminator.get("weight_decay", 0.0)
            ),
            artifact_family_target_weight=float(
                artifact_discriminator.get("family_target_weight", 0.5)
            ),
            artifact_family_warmup_steps=int(
                artifact_discriminator.get("family_warmup_steps", 2_000)
            ),
            artifact_family_combination=str(
                artifact_discriminator.get("family_combination", "convex")
            ),
            artifact_family_beta=float(
                artifact_discriminator.get("family_beta", 1.0)
            ),
            artifact_loss_transition=dict(
                artifact_discriminator.get("loss_transition", {})
            ),
            artifact_bootstrap_source_fingerprint=str(
                artifact_discriminator.get(
                    "bootstrap_source_fingerprint", ""
                )
            ),
            discriminator_learning_rate=float(
                discriminator.get("learning_rate", 1.0e-4)
            ),
            discriminator_betas=tuple(
                float(value) for value in discriminator.get("betas", (0.8, 0.99))
            ),
            discriminator_weight_decay=float(discriminator.get("weight_decay", 0.0)),
            discriminator_scheduler=str(discriminator.get("scheduler", "constant")),
            discriminator_grad_clip_norm=float(
                discriminator.get("grad_clip_norm", 1.0)
            ),
            discriminator_updates_per_generator=int(
                discriminator.get("updates_per_generator", 1)
            ),
            discriminator_update_every=int(discriminator.get("update_every", 2)),
            discriminator_reset_on_resume=bool(
                discriminator.get("reset_on_resume", True)
            ),
        )


class TTATrainer:
    """Accelerate/DDP trainer for the public TTA model and batch contracts."""

    log_label = "TTA"

    def __init__(
        self,
        model: nn.Module,
        config: TTATrainerConfig,
        *,
        resolved_config: Mapping[str, Any] | None = None,
        config_source: str | Path | None = None,
    ) -> None:
        (
            Accelerator,
            DataLoaderConfiguration,
            DistributedDataParallelKwargs,
            set_seed,
        ) = _load_accelerate()
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=bool(config.find_unused_parameters)
        )
        dataloader_kwargs: dict[str, Any] = {"use_seedable_sampler": True}
        # Accelerate <=0.32 seeds SeedableRandomSampler through set_seed and
        # has no data_seed field; newer releases expose an explicit override.
        # Keep training deterministic on both environments.
        if "data_seed" in inspect.signature(DataLoaderConfiguration).parameters:
            dataloader_kwargs["data_seed"] = int(config.seed)
        try:
            self.accelerator = Accelerator(
                mixed_precision=config.mixed_precision,
                gradient_accumulation_steps=config.grad_accumulation_steps,
                kwargs_handlers=[ddp_kwargs],
                dataloader_config=DataLoaderConfiguration(**dataloader_kwargs),
                # TTA counts max_steps in global DDP optimizer updates. Step
                # the prepared scheduler explicitly once per completed update
                # instead of Accelerate multiplying it by world size.
                step_scheduler_with_optimizer=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "failed to initialize Accelerate for TTA training; verify the CUDA/runtime "
                "configuration and launch with `accelerate launch` for distributed training"
            ) from exc
        distributed_type = str(self.accelerator.distributed_type).upper()
        if (
            "FSDP" in distributed_type
            or "DEEPSPEED" in distributed_type
            or "MEGATRON" in distributed_type
        ):
            raise ValueError(
                "TTA filtered checkpoints currently support Accelerate single-process or DDP, "
                f"not {self.accelerator.distributed_type}"
            )
        set_seed(config.seed, device_specific=True)

        self.config = config
        self.model = model
        self.output_dir = Path(config.output_dir).expanduser().resolve()
        self.checkpoint_root = self.output_dir / "checkpoint"
        self.log_root = self.output_dir / "log"
        if self.accelerator.is_main_process:
            self.checkpoint_root.mkdir(parents=True, exist_ok=True)
            self.log_root.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "config").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "config" / "trainer_config.json").write_text(
                json.dumps(to_jsonable(asdict(config)), indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            if resolved_config is not None:
                save_config_snapshot(
                    resolved_config, self.output_dir, source_path=config_source
                )
        self.accelerator.wait_for_everyone()

        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        if not parameters:
            raise ValueError("TTA model has no trainable parameters")
        self.optimizer = AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self._lr_lambda)
        self.discriminator: nn.Module | None = None
        self.discriminator_optimizer: Any | None = None
        self.discriminator_scheduler: Any | None = None
        # The artifact discriminator is deliberately prepared only after the
        # base checkpoint has loaded. Older checkpoints contain exactly the
        # generator and the original spectral discriminator; registering a
        # third model/optimizer with Accelerate before load would shift or
        # reject those established states.
        self.artifact_discriminator: nn.Module | None = None
        self.artifact_discriminator_optimizer: Any | None = None
        self.artifact_discriminator_scheduler: Any | None = None
        self.artifact_start_step: int | None = None
        if config.adversarial_enabled:
            if config.adversarial_backend == "xcodec_mpd_spec_v1":
                self.discriminator = XCodecMPDSpecDiscriminator(
                    periods=config.adversarial_mpd_periods,
                    spec_fft_sizes=config.adversarial_stft_fft_sizes,
                )
            else:
                self.discriminator = MultiViewSTFTPQMFAudioDiscriminator(
                    stft_fft_sizes=config.adversarial_stft_fft_sizes,
                    pqmf_enabled=config.adversarial_pqmf_enabled,
                    mpd_enabled=config.adversarial_mpd_enabled,
                    mpd_periods=config.adversarial_mpd_periods,
                    msd_enabled=config.adversarial_msd_enabled,
                    msd_scales=config.adversarial_msd_scales,
                )
            self.discriminator_optimizer = AdamW(
                self.discriminator.parameters(),
                lr=config.discriminator_learning_rate,
                betas=config.discriminator_betas,
                weight_decay=config.discriminator_weight_decay,
            )
            self.discriminator_scheduler = LambdaLR(
                self.discriminator_optimizer, lr_lambda=lambda _: 1.0
            )
            (
                self.model,
                self.discriminator,
                self.optimizer,
                self.discriminator_optimizer,
                self.scheduler,
                self.discriminator_scheduler,
            ) = self.accelerator.prepare(
                self.model,
                self.discriminator,
                self.optimizer,
                self.discriminator_optimizer,
                self.scheduler,
                self.discriminator_scheduler,
            )
            if (
                config.adversarial_backend == "xcodec_mpd_spec_v1"
                or config.adversarial_mixed_precision
            ):
                # Accelerate normally converts every forward output to FP32.
                # Audio discriminators return many large FM feature maps, so
                # that conversion creates a second full-precision copy and can
                # consume tens of GiB. Keep features in the autocast dtype and
                # cast only scalar GAN/FM reductions to FP32.
                self.accelerator.unwrap_model(
                    self.discriminator,
                    keep_fp32_wrapper=False,
                )
        else:
            self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
                self.model,
                self.optimizer,
                self.scheduler,
            )
        self.accelerator.register_save_state_pre_hook(self._save_model_state_hook)
        self.accelerator.register_load_state_pre_hook(self._load_model_state_hook)
        # Keep only the model's filtered checkpoint keys. In particular this
        # avoids a second copy of the frozen FLAN-T5 body.
        self.ema_state = self._build_ema_state()
        self.global_step = 0
        self.data_epoch = 0
        self.data_microbatch_offset = 0
        self.data_microbatches_per_epoch = 0
        self._resume_trainer_state: dict[str, Any] = {}
        self._resume_checkpoint_path: Path | None = None
        self._warm_started = False
        self._run_start_step = 0
        self._run_start_time = time.monotonic()
        self._model_diagnostics_batch: Mapping[str, Any] | None = None
        self._consecutive_non_finite_gradients: dict[str, int] = {}
        self._reference_decoder: nn.Module | None = None
        self._parameter_anchor: tuple[tuple[str, Tensor, Tensor], ...] = ()

    def _initialize_decoder_preservation(self) -> None:
        """Snapshot the warm-start decoder before the first optimizer update."""

        if (
            self.config.decoder_distillation_weight <= 0.0
            and self.config.parameter_anchor_weight <= 0.0
        ):
            return
        if not self._warm_started:
            raise ValueError(
                "decoder preservation requires initialization.full_tta_checkpoint "
                "so the reference is an explicit warm-start model"
            )
        model = _unwrap_prepared_model(self.model)
        if self.config.decoder_distillation_weight > 0.0:
            decoder = getattr(model, "decoder", None)
            if not isinstance(decoder, nn.Module):
                raise TypeError("decoder output distillation requires model.decoder")
            self._reference_decoder = deepcopy(decoder).eval()
            _set_requires_grad(self._reference_decoder, False)
        if self.config.parameter_anchor_weight > 0.0:
            anchor = []
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    anchor.append(
                        (name, parameter, parameter.detach().float().clone())
                    )
            if not anchor:
                raise ValueError("parameter anchoring found no trainable parameters")
            self._parameter_anchor = tuple(anchor)
        self.accelerator.print(
            f"[{self.log_label}] decoder preservation initialized: "
            f"output_distillation_weight={self.config.decoder_distillation_weight:g} "
            f"output_distillation_mode={self.config.decoder_distillation_mode} "
            f"parameter_anchor_weight={self.config.parameter_anchor_weight:g} "
            f"anchored_tensors={len(self._parameter_anchor)}",
            flush=True,
        )

    def _decoder_preservation_losses(
        self,
        output: Mapping[str, Any],
        batch: Mapping[str, Any],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return reference-output L1, summed L2-SP, and mean parameter drift."""

        loss = output["loss"]
        zero = loss.new_tensor(0.0)
        distillation = zero
        if self._reference_decoder is not None:
            z_decode = output.get("z_decode")
            current = output.get("teacher_waveform_model")
            if not isinstance(z_decode, Tensor) or not isinstance(current, Tensor):
                raise RuntimeError(
                    "decoder output distillation requires z_decode and "
                    "teacher_waveform_model"
                )
            model = _unwrap_prepared_model(self.model)
            with torch.no_grad(), self.accelerator.autocast():
                reference_tokens = model.decoder_latent_norm(z_decode.detach())
                reference_tokens = model.decoder_token_norm(
                    model.latent_to_decoder(reference_tokens)
                )
                for blocks, upsample in zip(
                    model.decoder_up_blocks,
                    model.decoder_upsamples,
                    strict=True,
                ):
                    for block in blocks:
                        reference_tokens = block(reference_tokens)
                    reference_tokens = upsample(reference_tokens)
                target_patches = (
                    int(current.shape[-1]) + int(model.patchify.patch_size) - 1
                ) // int(model.patchify.patch_size)
                reference_tokens = reference_tokens[:, :target_patches]
                reference = self._reference_decoder(
                    reference_tokens, original_len=int(current.shape[-1])
                )
            valid = batch.get("wav_valid_mask")
            if not isinstance(valid, Tensor):
                valid = torch.ones_like(current, dtype=torch.bool)
            elif valid.ndim == 2:
                valid = valid.unsqueeze(1)
            valid = valid.to(device=current.device, dtype=torch.bool)
            reference = reference.detach().to(dtype=current.dtype)
            if self.config.decoder_distillation_mode == "selective":
                distillation = selective_decoder_distillation_loss(
                    current,
                    reference,
                    valid,
                    sample_rate=self.config.decoder_distillation_sample_rate,
                    lowpass_hz=self.config.decoder_distillation_lowpass_hz,
                    envelope_ms=self.config.decoder_distillation_envelope_ms,
                    lowpass_weight=self.config.decoder_distillation_lowpass_weight,
                    envelope_weight=self.config.decoder_distillation_envelope_weight,
                    transient_weight=self.config.decoder_distillation_transient_weight,
                )["loss"]
            else:
                distillation = masked_waveform_l1(current, reference, valid)

        anchor_sum = zero
        anchor_count = 0
        for _, parameter, reference in self._parameter_anchor:
            delta = parameter.float() - reference
            anchor_sum = anchor_sum + delta.square().sum()
            anchor_count += parameter.numel()
        anchor_mean = (
            anchor_sum / float(anchor_count) if anchor_count > 0 else zero
        )
        return distillation, anchor_sum, anchor_mean

    def _clip_and_validate_gradients(
        self,
        parameters: Iterable[Tensor],
        *,
        max_norm: float,
        optimizer: Any,
        stage: str,
        batch: Mapping[str, Any] | None = None,
    ) -> tuple[bool, float | None]:
        """Clip gradients and prevent a non-finite optimizer update.

        DDP has already reduced gradients when ``sync_gradients`` is true, so
        every rank makes the same skip/raise decision.  The returned norm from
        ``clip_grad_norm_`` is inspected before ``optimizer.step``; clipping
        alone cannot repair Inf/NaN gradients.
        """

        if not self.accelerator.sync_gradients:
            return True, None
        trainable = tuple(
            parameter for parameter in parameters if parameter.requires_grad
        )
        clip_limit = float(max_norm) if max_norm > 0.0 else math.inf
        total_norm = self.accelerator.clip_grad_norm_(trainable, clip_limit)
        norm_tensor = torch.as_tensor(total_norm).detach()
        finite = bool(torch.isfinite(norm_tensor).all().item())
        norm_value = float(norm_tensor.float().item())
        if finite:
            self._consecutive_non_finite_gradients[stage] = 0
            return True, norm_value

        # Never let AdamW consume a bad gradient: one poisoned optimizer state
        # is enough to make all following forward passes non-finite.
        optimizer.zero_grad(set_to_none=True)
        consecutive = self._consecutive_non_finite_gradients.get(stage, 0) + 1
        self._consecutive_non_finite_gradients[stage] = consecutive
        self._log_non_finite_gradient_event(
            stage=stage,
            grad_norm=norm_value,
            consecutive=consecutive,
            batch=batch,
        )
        if consecutive >= self.config.max_consecutive_non_finite_gradients:
            raise FloatingPointError(
                f"{stage} gradients were non-finite for {consecutive} consecutive "
                f"updates at optimizer step {self.global_step}; parameters were "
                "not updated"
            )
        return False, norm_value

    def _log_non_finite_gradient_event(
        self,
        *,
        stage: str,
        grad_norm: float,
        consecutive: int,
        batch: Mapping[str, Any] | None,
    ) -> None:
        if not self.accelerator.is_main_process:
            return
        event = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "event": "non_finite_gradient_skipped",
            "stage": str(stage),
            "step": int(self.global_step),
            "grad_norm": (
                grad_norm if math.isfinite(grad_norm) else str(grad_norm)
            ),
            "consecutive": int(consecutive),
            # This is the main rank's local batch. A bad gradient from any rank
            # is intentionally propagated by DDP before this safety check.
            "local_utt_id": None if batch is None else batch.get("utt_id"),
        }
        print(
            f"[{event['timestamp']}] [TTA][SAFETY] skipped {stage} optimizer "
            f"update at step={self.global_step}: grad_norm={grad_norm}, "
            f"consecutive={consecutive}",
            flush=True,
        )
        with (self.log_root / "safety_events.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(
                json.dumps(to_jsonable(event), ensure_ascii=False) + "\n"
            )

    def train(self, train_dataloader: Iterable[Mapping[str, Any]]) -> int:
        if _handles_distributed_sharding(train_dataloader):
            # AudioSet already shards parquet files over rank x worker. Letting
            # Accelerate wrap this iterable would either shard twice or
            # dispatch rank-0 batches whose caption/metadata strings cannot be
            # tensor-broadcast safely.
            loader = train_dataloader
        else:
            loader = self.accelerator.prepare(train_dataloader)
        self.global_step = self._load_checkpoint_if_needed()
        self._initialize_decoder_preservation()
        self._initialize_artifact_discriminator()
        if self.global_step < self.config.max_steps:
            steps_per_epoch = _optimizer_steps_per_epoch(
                loader,
                world_size=int(self.accelerator.num_processes),
                grad_accumulation_steps=int(self.config.grad_accumulation_steps),
            )
            self.data_microbatches_per_epoch = (
                _self_sharded_microbatches_per_epoch(loader)
                if _handles_distributed_sharding(loader)
                else _loader_microbatches_per_epoch(loader)
            )
            self._restore_data_cursor()
        else:
            steps_per_epoch = 0
        self._run_start_step = self.global_step
        self._run_start_time = time.monotonic()
        self.model.train()
        if self.discriminator is not None:
            self.discriminator.train()
        if self.artifact_discriminator is not None:
            self.artifact_discriminator.train()
        if not self._warm_started and self._should_evaluate_audiocaps(self.global_step):
            self._evaluate_audiocaps(self.global_step)
        while self.global_step < self.config.max_steps:
            _set_loader_epoch(loader, self.data_epoch)
            cursor_applied_by_dataset = _set_loader_start_offset(
                loader, self.data_microbatch_offset
            )
            if self.data_microbatch_offset and cursor_applied_by_dataset:
                self.accelerator.print(
                    f"[{self.log_label}] dataset applied resume cursor without "
                    f"batch replay: epoch={self.data_epoch} "
                    f"microbatch={self.data_microbatch_offset}"
                )
            active_loader = (
                self.accelerator.skip_first_batches(
                    loader, num_batches=self.data_microbatch_offset
                )
                if self.data_microbatch_offset and not cursor_applied_by_dataset
                else loader
            )
            saw_batch = False
            for batch in active_loader:
                saw_batch = True
                self.data_microbatch_offset += 1
                step_start = time.monotonic()
                batch = _move_to_device(batch, self.accelerator.device)
                if (
                    self.config.model_diagnostics_every > 0
                    and self._model_diagnostics_batch is None
                ):
                    self._model_diagnostics_batch = _copy_batch_to_cpu(batch)
                progress = self.global_step / max(self.config.max_steps, 1)
                accumulate_models = tuple(
                    module
                    for module in (
                        self.model,
                        self.discriminator,
                        self.artifact_discriminator,
                    )
                    if module is not None
                )
                with self.accelerator.accumulate(*accumulate_models):
                    generator_gradients_finite = True
                    output = self.model(batch, progress=progress)
                    if not isinstance(output, Mapping) or not isinstance(
                        output.get("loss"), Tensor
                    ):
                        raise TypeError(
                            "TTA model forward must return a mapping containing Tensor `loss`"
                        )
                    output = dict(output)
                    loss = output["loss"]
                    if loss.ndim != 0:
                        loss = loss.mean()
                    (
                        decoder_distillation_loss,
                        parameter_anchor_loss,
                        parameter_anchor_mean_square,
                    ) = self._decoder_preservation_losses(output, batch)
                    weighted_decoder_distillation_loss = (
                        decoder_distillation_loss
                        * float(self.config.decoder_distillation_weight)
                    )
                    weighted_parameter_anchor_loss = (
                        parameter_anchor_loss
                        * float(self.config.parameter_anchor_weight)
                    )
                    loss = (
                        loss
                        + weighted_decoder_distillation_loss
                        + weighted_parameter_anchor_loss
                    )
                    output.update(
                        {
                            "decoder_distillation_loss": decoder_distillation_loss,
                            "weighted_decoder_distillation_loss": (
                                weighted_decoder_distillation_loss
                            ),
                            "parameter_anchor_loss": parameter_anchor_loss,
                            "parameter_anchor_mean_square": (
                                parameter_anchor_mean_square
                            ),
                            "weighted_parameter_anchor_loss": (
                                weighted_parameter_anchor_loss
                            ),
                        }
                    )
                    gan_active = self._gan_active(self.global_step)
                    if gan_active:
                        gan_weight_scale = self._gan_generator_weight_scale(
                            self.global_step
                        )
                        (
                            generator_weight,
                            feature_matching_weight,
                        ) = self._gan_generator_weights(
                            self.global_step,
                            activation_scale=gan_weight_scale,
                        )
                        (
                            discriminator_loss,
                            adversarial_loss,
                            feature_loss,
                            discriminator_updated,
                            artifact_discriminator_updated,
                            gan_view_metrics,
                        ) = (
                            self._adversarial_losses(output)
                        )
                        loss = (
                            loss
                            + generator_weight * adversarial_loss
                            + feature_matching_weight * feature_loss
                        )
                    else:
                        zero = loss.new_tensor(0.0)
                        discriminator_loss = zero
                        adversarial_loss = zero
                        feature_loss = zero
                        discriminator_updated = False
                        artifact_discriminator_updated = False
                        gan_view_metrics = {}
                        gan_weight_scale = 0.0
                        generator_weight = 0.0
                        feature_matching_weight = 0.0
                    output.update(
                        {
                            "loss": loss,
                            "gan_active": loss.new_tensor(float(gan_active)),
                            "gan_discriminator_loss": discriminator_loss.detach(),
                            "weighted_gan_discriminator_loss": (
                                discriminator_loss.detach()
                                * float(
                                    self.config.adversarial_discriminator_weight
                                )
                            ),
                            "gan_generator_loss": adversarial_loss,
                            "gan_feature_matching_loss": feature_loss,
                            "gan_weight_scale": loss.new_tensor(gan_weight_scale),
                            "gan_generator_weight": loss.new_tensor(
                                generator_weight
                            ),
                            "gan_feature_matching_weight": loss.new_tensor(
                                feature_matching_weight
                            ),
                            "weighted_gan_generator_loss": adversarial_loss
                            * float(generator_weight),
                            "weighted_gan_feature_matching_loss": feature_loss
                            * float(feature_matching_weight),
                            "discriminator_lr": loss.new_tensor(
                                self._discriminator_learning_rate()
                            ),
                            "discriminator_updated": loss.new_tensor(
                                float(discriminator_updated)
                            ),
                            "artifact_discriminator_updated": loss.new_tensor(
                                float(artifact_discriminator_updated)
                            ),
                            **gan_view_metrics,
                        }
                    )
                    if not torch.isfinite(loss.detach()):
                        non_finite = {
                            str(name): float(value.detach().float().mean().item())
                            for name, value in output.items()
                            if isinstance(value, Tensor)
                            and value.numel() > 0
                            and not bool(torch.isfinite(value.detach()).all().item())
                        }
                        raise FloatingPointError(
                            f"non-finite TTA loss at optimizer step {self.global_step}: "
                            f"{float(loss.detach().float().item())}; "
                            f"components={non_finite}; "
                            f"utt_id={batch.get('utt_id')}; "
                            f"caption={batch.get('caption')}"
                        )
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        (
                            generator_gradients_finite,
                            generator_grad_norm,
                        ) = self._clip_and_validate_gradients(
                            self.model.parameters(),
                            max_norm=self.config.grad_clip_norm,
                            optimizer=self.optimizer,
                            stage="generator",
                            batch=batch,
                        )
                        output["generator_grad_norm"] = loss.new_tensor(
                            float(generator_grad_norm)
                        )
                    if generator_gradients_finite:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                completed_step = (
                    self.accelerator.sync_gradients
                    and generator_gradients_finite
                    and not bool(self.accelerator.optimizer_step_was_skipped)
                )
                if not completed_step:
                    if (
                        self.accelerator.sync_gradients
                        and not generator_gradients_finite
                        and gan_active
                        and (
                            discriminator_updated
                            or artifact_discriminator_updated
                        )
                    ):
                        self._step_discriminator_schedulers(
                            discriminator_updated=discriminator_updated,
                            artifact_discriminator_updated=(
                                artifact_discriminator_updated
                            ),
                        )
                    continue
                self.scheduler.step()
                if gan_active and (
                    discriminator_updated or artifact_discriminator_updated
                ):
                    self._step_discriminator_schedulers(
                        discriminator_updated=discriminator_updated,
                        artifact_discriminator_updated=(
                            artifact_discriminator_updated
                        ),
                    )
                self.global_step += 1
                self._update_internal_ema(self.global_step)
                self._update_ema(self.global_step)
                step_seconds = time.monotonic() - step_start
                if (
                    self.global_step == 1
                    or self.global_step % self.config.log_every == 0
                ):
                    self._add_detached_reconstruction_metrics(output)
                    self._log_step(
                        output,
                        batch,
                        step_seconds=step_seconds,
                        epoch=self.data_epoch,
                        steps_per_epoch=steps_per_epoch,
                    )
                if (
                    self.config.save_every > 0
                    and self.global_step % self.config.save_every == 0
                ):
                    self.save_checkpoint(self.global_step, last=True)
                if (
                    self.config.milestone_every > 0
                    and self.global_step % self.config.milestone_every == 0
                ):
                    self.save_milestone(self.global_step)
                if self._should_sample(self.global_step):
                    self._save_fixed_samples(self.global_step)
                if self._should_evaluate_model_diagnostics(self.global_step):
                    self._evaluate_model_diagnostics(self.global_step)
                if self._should_evaluate_encoder(self.global_step):
                    self._evaluate_encoder(self.global_step)
                if self._should_evaluate_audiocaps(self.global_step):
                    self._evaluate_audiocaps(self.global_step)
                if self.global_step >= self.config.max_steps:
                    break
            if not saw_batch:
                # A checkpoint may be written on the final batch before the
                # iterator reports StopIteration. On resume, skipping that
                # exact offset legitimately yields no batches; advance to the
                # next deterministic epoch instead of replaying epoch zero.
                if self.data_microbatch_offset > 0:
                    self.data_epoch += 1
                    self.data_microbatch_offset = 0
                    continue
                raise RuntimeError("TTA train dataloader produced no batches")
            if self.global_step >= self.config.max_steps:
                break
            self.data_epoch += 1
            self.data_microbatch_offset = 0

        if self.config.save_final_checkpoint:
            self.save_checkpoint(self.global_step, last=True)
        else:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                print(
                    f"[{self.log_label}] final checkpoint disabled at step "
                    f"{self.global_step}",
                    flush=True,
                )
        self.accelerator.wait_for_everyone()
        return self.global_step

    def _gan_active(self, step: int) -> bool:
        return bool(
            self.config.adversarial_enabled
            and int(step) >= int(self.config.adversarial_start_step)
        )

    def _gan_generator_weight_scale(self, step: int) -> float:
        if not self._gan_active(step):
            return 0.0
        warmup = int(self.config.adversarial_weight_warmup_steps)
        if warmup == 0:
            return 1.0
        elapsed = int(step) - int(self.config.adversarial_start_step) + 1
        return min(1.0, max(0.0, elapsed / warmup))

    def _gan_generator_weights(
        self,
        step: int,
        *,
        activation_scale: float | None = None,
    ) -> tuple[float, float]:
        scale = (
            self._gan_generator_weight_scale(step)
            if activation_scale is None
            else float(activation_scale)
        )
        target_generator = float(
            self.config.adversarial_generator_weight
        )
        target_feature = float(
            self.config.adversarial_feature_matching_weight
        )
        transition = dict(
            getattr(self.config, "adversarial_weight_transition", {}) or {}
        )
        if not bool(transition.get("enabled", False)):
            return target_generator * scale, target_feature * scale

        source_step = int(transition["source_step"])
        transition_steps = int(transition["transition_steps"])
        if int(step) < source_step:
            alpha = 0.0
        elif transition_steps == 0:
            alpha = 1.0
        else:
            alpha = min(
                1.0,
                max(0.0, (int(step) - source_step) / transition_steps),
            )
        source_generator = float(transition["source_generator_weight"])
        source_feature = float(
            transition["source_feature_matching_weight"]
        )
        generator = source_generator + alpha * (
            target_generator - source_generator
        )
        feature = source_feature + alpha * (
            target_feature - source_feature
        )
        return generator * scale, feature * scale

    def _initialize_artifact_discriminator(self) -> None:
        if not self.config.artifact_discriminator_enabled:
            return
        if self.artifact_discriminator is not None:
            raise RuntimeError("artifact discriminator was initialized twice")
        discriminator = MultiViewSTFTPQMFAudioDiscriminator(
            stft_fft_sizes=self.config.artifact_discriminator_stft_fft_sizes,
            pqmf_enabled=False,
            mpd_enabled=True,
            mpd_periods=self.config.artifact_discriminator_mpd_periods,
            msd_enabled=False,
        )
        discriminator.to(self.accelerator.device)
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and int(self.accelerator.num_processes) > 1
        ):
            device = self.accelerator.device
            discriminator = torch.nn.parallel.DistributedDataParallel(
                discriminator,
                device_ids=[device.index] if device.type == "cuda" else None,
                output_device=device.index if device.type == "cuda" else None,
                find_unused_parameters=False,
            )
        optimizer = AdamW(
            discriminator.parameters(),
            lr=self.config.artifact_discriminator_learning_rate,
            betas=self.config.artifact_discriminator_betas,
            weight_decay=self.config.artifact_discriminator_weight_decay,
        )
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        self.artifact_discriminator = discriminator
        self.artifact_discriminator_optimizer = optimizer
        self.artifact_discriminator_scheduler = scheduler

        state = self._resume_trainer_state
        path = self._resume_checkpoint_path
        saved_enabled = bool(state.get("artifact_discriminator_enabled", False))
        reset = bool(
            path is not None
            and self._should_reset_discriminator_for_checkpoint(path)
        )
        if saved_enabled and path is not None and not reset:
            self._load_artifact_checkpoint(path)
            self.artifact_start_step = int(
                state.get("artifact_start_step", self.global_step)
            )
            action = "restored"
        else:
            self.artifact_start_step = int(self.global_step)
            action = "bootstrapped"
        self.artifact_discriminator.train()
        self.accelerator.print(
            f"[{self.log_label}] {action} artifact discriminator at "
            f"step={self.global_step}; family_start={self.artifact_start_step}; "
            f"stft={self.config.artifact_discriminator_stft_fft_sizes}; "
            f"mpd={self.config.artifact_discriminator_mpd_periods}; "
            f"gan_local_batch={self.config.adversarial_gan_batch_size}"
        )

    def _artifact_family_weight(self, step: int) -> float:
        if (
            self.artifact_discriminator is None
            or self.artifact_start_step is None
        ):
            return 0.0
        warmup = int(self.config.artifact_family_warmup_steps)
        if warmup == 0:
            scale = 1.0
        else:
            elapsed = int(step) - int(self.artifact_start_step) + 1
            scale = min(1.0, max(0.0, elapsed / warmup))
        return float(self.config.artifact_family_target_weight) * scale

    def _gan_family_weights(self, step: int) -> tuple[float, float]:
        if self.artifact_discriminator is None:
            return 1.0, 0.0
        if self.config.artifact_family_combination == "additive":
            return 1.0, float(self.config.artifact_family_beta)
        artifact_weight = self._artifact_family_weight(step)
        return 1.0 - artifact_weight, artifact_weight

    def _gan_batch_indices(self, batch_size: int, device: torch.device) -> Tensor:
        requested = int(self.config.adversarial_gan_batch_size)
        selected = batch_size if requested <= 0 else min(requested, batch_size)
        if selected == batch_size:
            return torch.arange(batch_size, device=device)
        generator = torch.Generator(device="cpu")
        seed = (
            int(self.config.seed) * 1_000_003
            + int(self.global_step) * 97_409
            + int(self.accelerator.process_index) * 65_537
        ) % (2**63 - 1)
        generator.manual_seed(seed)
        return torch.randperm(
            batch_size,
            generator=generator,
            device="cpu",
        )[:selected].to(device=device)

    def _step_discriminator_schedulers(
        self,
        *,
        discriminator_updated: bool,
        artifact_discriminator_updated: bool,
    ) -> None:
        if discriminator_updated:
            if self.discriminator_scheduler is None:
                raise RuntimeError(
                    "GAN is active but discriminator scheduler is unavailable"
                )
            self.discriminator_scheduler.step()
        if artifact_discriminator_updated:
            if self.artifact_discriminator_scheduler is None:
                raise RuntimeError(
                    "artifact discriminator scheduler is unavailable"
                )
            self.artifact_discriminator_scheduler.step()

    def _adversarial_losses(
        self, output: Mapping[str, Any]
    ) -> tuple[Tensor, Tensor, Tensor, bool, bool, dict[str, Tensor]]:
        if (
            self.discriminator is None
            or self.discriminator_optimizer is None
            or self.discriminator_scheduler is None
        ):
            raise RuntimeError(
                "GAN is active but discriminator training state is unavailable"
            )
        real = output.get("reconstruction_real_crop")
        fake = output.get("reconstruction_fake_crop")
        if not isinstance(real, Tensor) or not isinstance(fake, Tensor):
            raise RuntimeError(
                "GAN requires reconstruction_real_crop and reconstruction_fake_crop outputs"
            )
        if real.shape[0] != fake.shape[0]:
            raise RuntimeError(
                "GAN real/fake crops must have the same local batch size"
            )
        indices = self._gan_batch_indices(real.shape[0], real.device)
        real = real.index_select(0, indices)
        fake = fake.index_select(0, indices)

        discriminator_updated = self._should_update_discriminator(self.global_step)
        if discriminator_updated:
            with self._discriminator_autocast():
                real_logits_d, _ = self.discriminator(real.detach())
                fake_logits_d, _ = self.discriminator(fake.detach())
            discriminator_loss = self._discriminator_objective(
                real_logits_d,
                fake_logits_d,
            )
        else:
            with torch.no_grad():
                with self._discriminator_autocast():
                    real_logits_d, _ = self.discriminator(real.detach())
                    fake_logits_d, _ = self.discriminator(fake.detach())
                discriminator_loss = self._discriminator_objective(
                    real_logits_d,
                    fake_logits_d,
                )
        if not torch.isfinite(discriminator_loss.detach()):
            raise FloatingPointError(
                "non-finite discriminator loss at optimizer step "
                f"{self.global_step}: {float(discriminator_loss.detach().float().item())}"
            )
        if discriminator_updated:
            self.accelerator.backward(
                discriminator_loss
                * float(self.config.adversarial_discriminator_weight)
            )
            discriminator_gradients_finite = True
            if self.accelerator.sync_gradients:
                (
                    discriminator_gradients_finite,
                    _,
                ) = self._clip_and_validate_gradients(
                    self.discriminator.parameters(),
                    max_norm=self.config.discriminator_grad_clip_norm,
                    optimizer=self.discriminator_optimizer,
                    stage="discriminator",
                )
            if discriminator_gradients_finite:
                self.discriminator_optimizer.step()
            self.discriminator_optimizer.zero_grad(set_to_none=True)
            discriminator_updated = (
                discriminator_updated and discriminator_gradients_finite
            )

        discriminator_module = _unwrap_prepared_model(self.discriminator)
        _set_requires_grad(discriminator_module, False)
        try:
            with self._discriminator_autocast():
                with torch.no_grad():
                    real_logits_g, real_features = discriminator_module(real.detach())
                fake_logits_g, fake_features = discriminator_module(fake)
            if self.config.adversarial_objective == "lsgan_sum":
                adversarial_loss = lsgan_generator_sum(fake_logits_g)
                feature_loss = feature_matching_sum(real_features, fake_features)
            else:
                adversarial_loss = relativistic_generator_loss(
                    real_logits_g,
                    fake_logits_g,
                )
                feature_loss = discriminator_feature_matching_loss(
                    real_features,
                    fake_features,
                )
            view_metrics = self._gan_view_metrics(real_logits_g, fake_logits_g)
        finally:
            _set_requires_grad(discriminator_module, True)

        spectral_adversarial_loss = adversarial_loss
        spectral_feature_loss = feature_loss
        artifact_discriminator_updated = False
        spectral_weight, artifact_weight = self._gan_family_weights(
            self.global_step
        )
        view_metrics.update(
            {
                "gan_subbatch_size": real.new_tensor(float(real.shape[0])),
                "gan_subbatch_fraction": real.new_tensor(
                    float(real.shape[0] / max(output["reconstruction_real_crop"].shape[0], 1))
                ),
                "gan_spectral_family_weight": real.new_tensor(
                    spectral_weight
                ),
                "gan_artifact_family_weight": real.new_tensor(artifact_weight),
                "gan_spectral_generator_loss": spectral_adversarial_loss.detach(),
                "gan_spectral_feature_matching_loss": spectral_feature_loss.detach(),
            }
        )
        if self.artifact_discriminator is not None:
            (
                artifact_discriminator_loss,
                artifact_adversarial_loss,
                artifact_feature_loss,
                artifact_discriminator_updated,
                artifact_metrics,
            ) = self._artifact_adversarial_losses(real=real, fake=fake)
            adversarial_loss = (
                spectral_weight * spectral_adversarial_loss
                + artifact_weight * artifact_adversarial_loss
            )
            feature_loss = (
                spectral_weight * spectral_feature_loss
                + artifact_weight * artifact_feature_loss
            )
            view_metrics.update(
                {
                    "gan_artifact_discriminator_loss": (
                        artifact_discriminator_loss.detach()
                    ),
                    "gan_artifact_generator_loss": (
                        artifact_adversarial_loss.detach()
                    ),
                    "gan_artifact_feature_matching_loss": (
                        artifact_feature_loss.detach()
                    ),
                    "artifact_discriminator_lr": real.new_tensor(
                        self._artifact_discriminator_learning_rate()
                    ),
                    **artifact_metrics,
                }
            )
        return (
            discriminator_loss,
            adversarial_loss,
            feature_loss,
            discriminator_updated,
            artifact_discriminator_updated,
            view_metrics,
        )

    def _artifact_adversarial_losses(
        self,
        *,
        real: Tensor,
        fake: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, bool, dict[str, Tensor]]:
        if (
            self.artifact_discriminator is None
            or self.artifact_discriminator_optimizer is None
            or self.artifact_discriminator_scheduler is None
        ):
            raise RuntimeError("artifact discriminator state is unavailable")
        updated = self._should_update_discriminator(self.global_step)
        if updated:
            with self._artifact_discriminator_autocast():
                real_logits_d, _ = self.artifact_discriminator(real.detach())
                fake_logits_d, _ = self.artifact_discriminator(fake.detach())
            discriminator_loss = relativistic_discriminator_loss(
                real_logits_d,
                fake_logits_d,
            )
        else:
            with torch.no_grad():
                with self._artifact_discriminator_autocast():
                    real_logits_d, _ = self.artifact_discriminator(real.detach())
                    fake_logits_d, _ = self.artifact_discriminator(fake.detach())
                discriminator_loss = relativistic_discriminator_loss(
                    real_logits_d,
                    fake_logits_d,
                )
        if not torch.isfinite(discriminator_loss.detach()):
            raise FloatingPointError(
                "non-finite artifact discriminator loss at optimizer step "
                f"{self.global_step}: "
                f"{float(discriminator_loss.detach().float().item())}"
            )
        if updated:
            self.accelerator.backward(
                discriminator_loss
                * float(self.config.adversarial_discriminator_weight)
            )
            gradients_finite = True
            if self.accelerator.sync_gradients:
                gradients_finite, _ = self._clip_and_validate_gradients(
                    self.artifact_discriminator.parameters(),
                    max_norm=self.config.discriminator_grad_clip_norm,
                    optimizer=self.artifact_discriminator_optimizer,
                    stage="artifact_discriminator",
                )
                if gradients_finite:
                    self.artifact_discriminator_optimizer.step()
                self.artifact_discriminator_optimizer.zero_grad(set_to_none=True)
                updated = updated and gradients_finite
            else:
                # Manual DDP preparation keeps this optimizer out of
                # Accelerate's legacy checkpoint registry. Preserve gradients
                # until the accumulation boundary just like AcceleratedOptimizer.
                updated = False

        module = _unwrap_prepared_model(self.artifact_discriminator)
        _set_requires_grad(module, False)
        try:
            with self._artifact_discriminator_autocast():
                with torch.no_grad():
                    real_logits_g, real_features = module(real.detach())
                fake_logits_g, fake_features = module(fake)
            adversarial_loss = relativistic_generator_loss(
                real_logits_g,
                fake_logits_g,
            )
            feature_loss = discriminator_feature_matching_loss(
                real_features,
                fake_features,
            )
            metrics = self._gan_view_metrics_for_names(
                real_logits_g,
                fake_logits_g,
                names=self._artifact_gan_view_names(),
                prefix="gan_artifact",
                objective="relativistic_paired",
            )
        finally:
            _set_requires_grad(module, True)
        return (
            discriminator_loss,
            adversarial_loss,
            feature_loss,
            updated,
            metrics,
        )

    def _discriminator_autocast(self) -> Any:
        if (
            self.config.adversarial_backend == "xcodec_mpd_spec_v1"
            or self.config.adversarial_mixed_precision
        ):
            return self.accelerator.autocast()
        return nullcontext()

    def _artifact_discriminator_autocast(self) -> Any:
        if self.config.artifact_discriminator_mixed_precision:
            return self.accelerator.autocast()
        return nullcontext()

    def _discriminator_objective(
        self,
        real_logits: Iterable[Tensor],
        fake_logits: Iterable[Tensor],
    ) -> Tensor:
        real = tuple(real_logits)
        fake = tuple(fake_logits)
        if self.config.adversarial_objective == "lsgan_sum":
            return lsgan_discriminator_sum(real, fake)
        return relativistic_discriminator_loss(real, fake)

    def _should_update_discriminator(self, step: int) -> bool:
        return (int(step) + 1) % int(self.config.discriminator_update_every) == 0

    def _gan_view_names(self) -> tuple[str, ...]:
        if (
            getattr(
                self.config,
                "adversarial_backend",
                "multiview_stft_pqmf_v1",
            )
            == "xcodec_mpd_spec_v1"
        ):
            return tuple(
                [
                    *(
                        f"mpd_{period}"
                        for period in self.config.adversarial_mpd_periods
                    ),
                    *(
                        f"spec_{size}"
                        for size in self.config.adversarial_stft_fft_sizes
                    ),
                ]
            )
        names = [f"stft_{size}" for size in self.config.adversarial_stft_fft_sizes]
        if self.config.adversarial_pqmf_enabled:
            names.append("pqmf")
        if self.config.adversarial_mpd_enabled:
            names.extend(f"mpd_{period}" for period in self.config.adversarial_mpd_periods)
        if self.config.adversarial_msd_enabled:
            names.extend(
                f"msd_{index}" for index in range(self.config.adversarial_msd_scales)
            )
        return tuple(names)

    def _gan_view_metrics(
        self,
        real_logits: Iterable[Tensor],
        fake_logits: Iterable[Tensor],
    ) -> dict[str, Tensor]:
        return self._gan_view_metrics_for_names(
            real_logits,
            fake_logits,
            names=self._gan_view_names(),
            prefix="gan",
            objective=getattr(
                self.config,
                "adversarial_objective",
                "relativistic_paired",
            ),
        )

    def _artifact_gan_view_names(self) -> tuple[str, ...]:
        return tuple(
            [
                *(
                    f"stft_{size}"
                    for size in self.config.artifact_discriminator_stft_fft_sizes
                ),
                *(
                    f"mpd_{period}"
                    for period in self.config.artifact_discriminator_mpd_periods
                ),
            ]
        )

    def _gan_view_metrics_for_names(
        self,
        real_logits: Iterable[Tensor],
        fake_logits: Iterable[Tensor],
        *,
        names: tuple[str, ...],
        prefix: str,
        objective: str,
    ) -> dict[str, Tensor]:
        real_views = tuple(real_logits)
        fake_views = tuple(fake_logits)
        if len(real_views) != len(fake_views) or len(real_views) != len(names):
            raise RuntimeError(
                "discriminator view count does not match configured diagnostic names"
            )
        metrics: dict[str, Tensor] = {}
        for name, real_score, fake_score in zip(
            names, real_views, fake_views, strict=True
        ):
            margin = real_score.float() - fake_score.float()
            view_prefix = f"{prefix}_{name}"
            metrics[f"{view_prefix}_real_logit"] = (
                real_score.detach().float().mean()
            )
            metrics[f"{view_prefix}_fake_logit"] = (
                fake_score.detach().float().mean()
            )
            metrics[f"{view_prefix}_margin"] = margin.detach().mean()
            if objective == "lsgan_sum":
                metrics[f"{view_prefix}_discriminator_loss"] = (
                    F.mse_loss(
                        real_score.detach().float(),
                        torch.ones_like(real_score.detach().float()),
                    )
                    + F.mse_loss(
                        fake_score.detach().float(),
                        torch.zeros_like(fake_score.detach().float()),
                    )
                )
                metrics[f"{view_prefix}_generator_loss"] = F.mse_loss(
                    fake_score.detach().float(),
                    torch.ones_like(fake_score.detach().float()),
                )
            else:
                metrics[f"{view_prefix}_discriminator_loss"] = F.softplus(
                    -margin.detach()
                ).mean()
                metrics[f"{view_prefix}_generator_loss"] = F.softplus(
                    margin
                ).mean()
        return metrics

    @torch.no_grad()
    def _add_detached_reconstruction_metrics(self, output: dict[str, Any]) -> None:
        real = output.get("reconstruction_full_real")
        fake = output.get("reconstruction_full_fake")
        valid = output.get("reconstruction_full_valid_mask")
        if not all(isinstance(value, Tensor) for value in (real, fake, valid)):
            return
        output["waveform_l1_metric"] = masked_waveform_l1(fake, real, valid)
        model = _unwrap_prepared_model(self.model)
        mel_loss = getattr(model, "mel_loss_fn", None)
        if isinstance(mel_loss, nn.Module):
            output["log_mel_metric"] = mel_loss(
                fake,
                real,
                sample_mask=valid.squeeze(1),
            ).detach()

    def _discriminator_learning_rate(self) -> float:
        if self.discriminator_optimizer is None:
            return 0.0
        return float(self.discriminator_optimizer.param_groups[0]["lr"])

    def _artifact_discriminator_learning_rate(self) -> float:
        if self.artifact_discriminator_optimizer is None:
            return 0.0
        return float(
            self.artifact_discriminator_optimizer.param_groups[0]["lr"]
        )

    def save_checkpoint(self, step: int, *, last: bool = False) -> Path:
        self.accelerator.wait_for_everyone()
        path = self._last_checkpoint_dir() if last else self._checkpoint_dir(step)
        if self.accelerator.is_main_process and path.exists():
            shutil.rmtree(path)
        self.accelerator.wait_for_everyone()
        try:
            self.accelerator.save_state(str(path), safe_serialization=True)
        except Exception as exc:
            raise RuntimeError(
                "Accelerate failed to save the TTA training state; verify optional "
                "DeepSpeed/TransformerEngine packages are compatible with this PyTorch runtime"
            ) from exc
        self._finalize_discriminator_checkpoint(path)
        self._save_artifact_checkpoint(path)
        self._save_ema_checkpoint(path)
        data_cursors = self._gather_rank_data_cursors()
        if self.accelerator.is_main_process:
            (path / "trainer_state.json").write_text(
                json.dumps(
                    {
                        "global_step": int(step),
                        "adversarial_enabled": bool(self.config.adversarial_enabled),
                        "adversarial_backend": self.config.adversarial_backend,
                        "adversarial_objective": self.config.adversarial_objective,
                        "adversarial_config_fingerprint": (
                            _adversarial_config_fingerprint(self.config)
                        ),
                        "gan_active": bool(self._gan_active(step)),
                        "adversarial_start_step": int(
                            self.config.adversarial_start_step
                        ),
                        "adversarial_gan_batch_size": int(
                            self.config.adversarial_gan_batch_size
                        ),
                        "artifact_discriminator_enabled": bool(
                            self.config.artifact_discriminator_enabled
                        ),
                        "artifact_discriminator_config_fingerprint": (
                            _artifact_discriminator_config_fingerprint(
                                self.config
                            )
                            if self.config.artifact_discriminator_enabled
                            else None
                        ),
                        "artifact_loss_config_fingerprint": (
                            _artifact_loss_config_fingerprint(self.config)
                            if self.config.artifact_discriminator_enabled
                            else None
                        ),
                        "artifact_family_combination": (
                            self.config.artifact_family_combination
                        ),
                        "artifact_family_beta": float(
                            self.config.artifact_family_beta
                        ),
                        "artifact_loss_transition": (
                            {
                                **dict(
                                    self.config.artifact_loss_transition
                                ),
                                "target_fingerprint": (
                                    _artifact_loss_config_fingerprint(
                                        self.config
                                    )
                                ),
                                "target_combination": (
                                    self.config.artifact_family_combination
                                ),
                                "target_beta": float(
                                    self.config.artifact_family_beta
                                ),
                            }
                            if bool(
                                self.config.artifact_loss_transition.get(
                                    "enabled", False
                                )
                            )
                            else None
                        ),
                        "artifact_start_step": self.artifact_start_step,
                        "artifact_family_weight": float(
                            self._artifact_family_weight(step)
                        ),
                        "adversarial_weight_transition": (
                            {
                                **dict(
                                    self.config.adversarial_weight_transition
                                ),
                                "target_fingerprint": (
                                    _adversarial_config_fingerprint(self.config)
                                ),
                                "target_generator_weight": float(
                                    self.config.adversarial_generator_weight
                                ),
                                "target_feature_matching_weight": float(
                                    self.config.adversarial_feature_matching_weight
                                ),
                            }
                            if bool(
                                self.config.adversarial_weight_transition.get(
                                    "enabled", False
                                )
                            )
                            else None
                        ),
                        "mask_contract_id": self.config.mask_contract_id,
                        "mask_contract_fingerprint": (
                            self.config.mask_contract_fingerprint
                        ),
                        "mask_transition": (
                            {
                                "source_fingerprint": str(
                                    self.config.mask_transition.get(
                                        "source_fingerprint", ""
                                    )
                                ),
                                "source_step": int(
                                    self.config.mask_transition.get(
                                        "source_step", -1
                                    )
                                ),
                                "target_fingerprint": str(
                                    self.config.mask_contract_fingerprint or ""
                                ),
                                "reason": str(
                                    self.config.mask_transition.get("reason", "")
                                ),
                            }
                            if bool(self.config.mask_transition.get("enabled", False))
                            else None
                        ),
                        "saved_at": datetime.now()
                        .astimezone()
                        .isoformat(timespec="seconds"),
                        "data_epoch": int(self.data_epoch),
                        "data_microbatch_offset": int(
                            self.data_microbatch_offset
                        ),
                        "data_microbatches_per_epoch": int(
                            self.data_microbatches_per_epoch
                        ),
                        "data_cursors": data_cursors,
                        "data_sampler_seed": int(self.config.seed),
                        "data_world_size": int(self.accelerator.num_processes),
                        "data_grad_accumulation_steps": int(
                            self.config.grad_accumulation_steps
                        ),
                        "data_batch_size": int(self.config.data_batch_size),
                        "data_num_workers": int(self.config.data_num_workers),
                        "data_config_fingerprint": str(
                            self.config.data_config_fingerprint
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"[{self.log_label}] saved checkpoint: {path}", flush=True)
            if not last:
                self._rotate_checkpoints()
        self.accelerator.wait_for_everyone()
        return path

    def save_milestone(self, step: int) -> Path:
        """Persist the verified rolling checkpoint without serializing twice."""
        self.accelerator.wait_for_everyone()
        source = self._last_checkpoint_dir()
        destination = self._checkpoint_dir(step)
        if self.accelerator.is_main_process:
            state = source / "trainer_state.json"
            if not state.is_file():
                raise FileNotFoundError(f"rolling checkpoint is incomplete: {source}")
            if destination.exists():
                shutil.rmtree(destination)
            temporary = destination.with_name(destination.name + ".tmp")
            if temporary.exists():
                shutil.rmtree(temporary)
            try:
                shutil.copytree(source, temporary, copy_function=os.link)
            except OSError:
                if temporary.exists():
                    shutil.rmtree(temporary)
                shutil.copytree(source, temporary, copy_function=shutil.copy2)
            temporary.replace(destination)
            print(f"[{self.log_label}] saved milestone: {destination}", flush=True)
        self.accelerator.wait_for_everyone()
        return destination

    def _save_model_state_hook(
        self,
        models: list[nn.Module],
        weights: list[dict[str, Tensor]],
        output_dir: str,
    ) -> None:
        del output_dir
        expected = 2 if self.discriminator is not None else 1
        if len(models) != expected or len(weights) != expected:
            raise RuntimeError(
                "TTATrainer prepared-model count changed while saving: "
                f"expected={expected} models={len(models)} weights={len(weights)}"
            )
        model = _unwrap_prepared_model(models[0])
        # This call is the checkpoint contract that deliberately excludes the
        # frozen FLAN-T5 and optional CLAP encoders.
        weights[0] = checkpoint_model_state(model)

    def _load_model_state_hook(self, models: list[nn.Module], input_dir: str) -> None:
        expected = 2 if self.discriminator is not None else 1
        if len(models) != expected:
            raise RuntimeError(
                "TTATrainer prepared-model count changed while loading: "
                f"expected={expected} models={len(models)}"
            )
        model = _unwrap_prepared_model(models[0])
        missing, unexpected = load_tta_model_state(model, input_dir)
        reset_discriminator = self._should_reset_discriminator_for_checkpoint(
            Path(input_dir)
        )
        if self.discriminator is not None and not reset_discriminator:
            discriminator_path = Path(input_dir) / "discriminator.safetensors"
            if not discriminator_path.is_file():
                raise FileNotFoundError(
                    "GAN resume checkpoint is missing discriminator.safetensors; "
                    "legacy non-GAN checkpoints may be used only as initialization"
                )
            try:
                from safetensors.torch import load_file
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "loading discriminator checkpoints requires safetensors"
                ) from exc
            discriminator = _unwrap_prepared_model(models[1])
            discriminator.load_state_dict(
                load_file(str(discriminator_path), device="cpu"), strict=True
            )
        # Prevent Accelerate's strict loader from rejecting omitted frozen
        # T5/CLAP keys.
        models.clear()
        self.accelerator.print(
            f"[{self.log_label}] loaded filtered model state: {input_dir}; "
            f"omitted_frozen={len(missing)} unexpected={len(unexpected)}"
        )

    def _load_checkpoint_if_needed(self) -> int:
        resume: str | None = self.config.resume_from
        if resume is None and self.config.auto_resume:
            automatic = self._latest_resumable_checkpoint()
            if automatic is not None:
                resume = str(automatic)
        if resume is None and self.config.warm_start_from:
            return self._load_warm_start(self.config.warm_start_from)
        if resume is None:
            self._resume_trainer_state = {}
            self._resume_checkpoint_path = None
            return 0
        path = resolve_checkpoint_dir(resume)
        self._resume_checkpoint_path = path
        trainer_state_path = path / "trainer_state.json"
        if not trainer_state_path.is_file():
            raise FileNotFoundError(
                f"resume checkpoint has no trainer_state.json (use it only for model initialization): {path}"
            )
        state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
        self._resume_trainer_state = dict(state)
        step = int(state.get("global_step", 0))
        self._validate_resume_checkpoint(path)
        self._validate_resume_metadata(state, path=path, step=step)
        self._validate_artifact_resume_checkpoint(state, path=path)
        fresh_discriminator_optimizer_state = None
        fresh_discriminator_scheduler_state = None
        reset_discriminator = self._should_reset_discriminator_for_checkpoint(path)
        if reset_discriminator:
            if (
                self.discriminator_optimizer is None
                or self.discriminator_scheduler is None
            ):
                raise RuntimeError(
                    "discriminator_reset_on_resume requires GAN optimizer state"
                )
            fresh_discriminator_optimizer_state = deepcopy(
                self.discriminator_optimizer.state_dict()
            )
            fresh_discriminator_scheduler_state = deepcopy(
                self.discriminator_scheduler.state_dict()
            )
        self.accelerator.load_state(str(path))
        if fresh_discriminator_optimizer_state is not None:
            assert self.discriminator_optimizer is not None
            assert self.discriminator_scheduler is not None
            self.discriminator_optimizer.load_state_dict(
                fresh_discriminator_optimizer_state
            )
            self.discriminator_scheduler.load_state_dict(
                fresh_discriminator_scheduler_state
            )
            self.accelerator.print(
                f"[{self.log_label}] reset discriminator model/optimizer/scheduler "
                f"after loading generator state from {path}"
            )
        self._load_ema_checkpoint(path)
        self.accelerator.print(
            f"[{self.log_label}] resumed checkpoint: {path}; step={step}"
        )
        return step

    def _load_warm_start(self, source: str) -> int:
        """Restore counters/EMA/GAN around a model-only compatible warm start."""

        path = resolve_checkpoint_dir(source)
        trainer_state_path = path / "trainer_state.json"
        if not trainer_state_path.is_file():
            raise FileNotFoundError(
                f"warm-start checkpoint has no trainer_state.json: {path}"
            )
        state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
        step = int(state.get("global_step", 0))
        if step < 0:
            raise ValueError(f"warm-start checkpoint has invalid global_step={step}")
        self._resume_checkpoint_path = path
        self._resume_trainer_state = dict(state)
        self._warm_started = True

        model = _unwrap_prepared_model(self.model)
        missing, unexpected = load_tta_model_state(model, path)
        self.accelerator.print(
            f"[{self.log_label}] loaded warm-start model state: {path}; "
            f"omitted_frozen={len(missing)} unexpected={len(unexpected)}"
        )

        if self.discriminator is not None:
            discriminator_path = path / "discriminator.safetensors"
            if not discriminator_path.is_file():
                if not self.config.discriminator_reset_on_resume:
                    raise FileNotFoundError(
                        f"warm-start GAN checkpoint is missing {discriminator_path}; "
                        "set training.discriminator.reset_on_resume=True only when "
                        "intentionally bootstrapping a new discriminator"
                    )
                self.accelerator.print(
                    f"[{self.log_label}] warm-start source has no discriminator; "
                    "using the freshly initialized discriminator as requested"
                )
            else:
                try:
                    from safetensors.torch import load_file
                except ModuleNotFoundError as exc:
                    raise ModuleNotFoundError(
                        "loading warm-start discriminator requires safetensors"
                    ) from exc
                discriminator = _unwrap_prepared_model(self.discriminator)
                discriminator.load_state_dict(
                    load_file(str(discriminator_path), device="cpu"), strict=True
                )
        self._load_compatible_warm_start_ema(path)
        self.accelerator.print(
            f"[{self.log_label}] compatible warm start: {path}; step={step}; "
            "optimizer=reset scheduler=rewarm discriminator_optimizer=reset"
        )
        return step

    def _restore_data_cursor(self) -> None:
        """Restore the deterministic sampler position for exact continuation."""

        state = self._resume_trainer_state
        if not state:
            self.data_epoch = 0
            self.data_microbatch_offset = 0
            return

        local_resume = self._resume_checkpoint_is_local()
        if self.config.reset_data_cursor:
            self.data_epoch = 0
            self.data_microbatch_offset = 0
            self.accelerator.print(
                f"[{self.log_label}] explicitly reset data cursor at model step "
                f"{self.global_step}; subsequent checkpoints will auto-continue"
            )
            return
        if not self.config.auto_continue_data_cursor and not local_resume:
            self.data_epoch = 0
            self.data_microbatch_offset = 0
            self.accelerator.print(
                f"[{self.log_label}] reset external data cursor at model step "
                f"{self.global_step}; subsequent local checkpoints will auto-continue"
            )
            return

        cursors = state.get("data_cursors")
        if not isinstance(cursors, list):
            # Checkpoints written by the immediately preceding trainer stored
            # one shared cursor. Map-style DDP loaders consume the same number
            # of rank-local microbatches, so that cursor can be migrated
            # exactly when the checkpoint is local and its saved geometry is
            # unchanged. The next checkpoint writes the per-rank format.
            legacy_fields = {
                "data_epoch",
                "data_microbatch_offset",
                "data_microbatches_per_epoch",
                "data_sampler_seed",
                "data_world_size",
                "data_grad_accumulation_steps",
            }
            if local_resume and legacy_fields.issubset(state):
                expected_legacy = {
                    "data_microbatches_per_epoch": int(
                        self.data_microbatches_per_epoch
                    ),
                    "data_sampler_seed": int(self.config.seed),
                    "data_world_size": int(self.accelerator.num_processes),
                    "data_grad_accumulation_steps": int(
                        self.config.grad_accumulation_steps
                    ),
                }
                mismatches = {
                    key: (state.get(key), value)
                    for key, value in expected_legacy.items()
                    if int(state.get(key, -1)) != value
                }
                if mismatches:
                    raise ValueError(
                        "legacy resume checkpoint data-loader geometry changed; "
                        f"refusing to replay/skip training data: {mismatches}"
                    )
                self.data_epoch = int(state["data_epoch"])
                self.data_microbatch_offset = int(
                    state["data_microbatch_offset"]
                )
                if self.data_epoch < 0 or not (
                    0
                    <= self.data_microbatch_offset
                    <= self.data_microbatches_per_epoch
                ):
                    raise ValueError(
                        "legacy checkpoint contains an invalid data cursor: "
                        f"epoch={self.data_epoch} "
                        f"offset={self.data_microbatch_offset}"
                    )
                self.accelerator.print(
                    f"[{self.log_label}] migrated local shared data cursor: "
                    f"epoch={self.data_epoch} "
                    f"microbatch={self.data_microbatch_offset}/"
                    f"{self.data_microbatches_per_epoch}"
                )
                return
            raise ValueError(
                "resume checkpoint has no per-rank data cursor; set "
                "training.auto_continue_data_cursor=false (or pass "
                "--reset-data-cursor) only when intentionally branching "
                "with the dataset starting from epoch zero"
            )

        expected = {
            "data_sampler_seed": int(self.config.seed),
            "data_world_size": int(self.accelerator.num_processes),
            "data_grad_accumulation_steps": int(
                self.config.grad_accumulation_steps
            ),
            "data_batch_size": int(self.config.data_batch_size),
            "data_num_workers": int(self.config.data_num_workers),
        }
        mismatches = {
            key: (state.get(key), value)
            for key, value in expected.items()
            if int(state.get(key, -1)) != value
        }
        saved_fingerprint = str(state.get("data_config_fingerprint", ""))
        if saved_fingerprint != str(self.config.data_config_fingerprint):
            raise ValueError(
                "resume checkpoint data config changed; refusing to continue "
                f"cursor: saved={saved_fingerprint!r} "
                f"current={self.config.data_config_fingerprint!r}"
            )
        saved_world_size = int(state.get("data_world_size", -1))
        by_saved_rank = {
            int(cursor.get("rank", -1)): cursor
            for cursor in cursors
            if isinstance(cursor, Mapping)
        }
        saved_ranks_complete = (
            saved_world_size > 0
            and set(by_saved_rank) == set(range(saved_world_size))
        )
        saved_epochs = {
            int(cursor.get("epoch", -1))
            for cursor in by_saved_rank.values()
        }
        saved_offsets = {
            int(cursor.get("microbatch_offset", -1))
            for cursor in by_saved_rank.values()
        }
        at_epoch_boundary = (
            saved_ranks_complete
            and len(saved_epochs) == 1
            and min(saved_epochs) >= 0
            and self.data_microbatches_per_epoch > 0
            and saved_offsets == {int(self.data_microbatches_per_epoch)}
            and int(state.get("data_sampler_seed", -1))
            == int(self.config.seed)
        )
        if at_epoch_boundary:
            self.data_epoch = next(iter(saved_epochs)) + 1
            self.data_microbatch_offset = 0
            self.accelerator.print(
                f"[{self.log_label}] migrated completed data epoch across "
                f"loader geometry: epoch={self.data_epoch} microbatch=0 "
                f"saved_world_size={saved_world_size} "
                f"world_size={self.accelerator.num_processes} "
                f"grad_accumulation={state.get('data_grad_accumulation_steps')}"
                f"->{self.config.grad_accumulation_steps}"
            )
            return
        batch_size_migration = (
            self.config.allow_batch_size_cursor_migration
            and set(mismatches) == {"data_batch_size"}
        )
        if mismatches and not batch_size_migration:
            raise ValueError(
                "resume checkpoint data-loader geometry changed; refusing to "
                f"silently replay/skip training data: {mismatches}"
            )
        rank = int(self.accelerator.process_index)
        by_rank = {
            int(cursor.get("rank", -1)): cursor
            for cursor in cursors
            if isinstance(cursor, Mapping)
        }
        if set(by_rank) != set(range(int(self.accelerator.num_processes))):
            raise ValueError(
                "resume checkpoint per-rank data cursors are incomplete: "
                f"found={sorted(by_rank)} world_size={self.accelerator.num_processes}"
            )
        cursor = by_rank[rank]
        self.data_epoch = int(cursor.get("epoch", -1))
        self.data_microbatch_offset = int(cursor.get("microbatch_offset", -1))
        if batch_size_migration:
            saved_batch_size = int(state["data_batch_size"])
            saved_offset = self.data_microbatch_offset
            consumed_samples = saved_offset * saved_batch_size
            self.data_microbatch_offset = (
                consumed_samples // int(self.config.data_batch_size)
            )
            replayed_samples = consumed_samples - (
                self.data_microbatch_offset * int(self.config.data_batch_size)
            )
            self.accelerator.print(
                f"[{self.log_label}] migrated data cursor batch size "
                f"{saved_batch_size}->{self.config.data_batch_size}: rank={rank} "
                f"epoch={self.data_epoch} microbatch={saved_offset}->"
                f"{self.data_microbatch_offset} replayed_samples={replayed_samples}"
            )
        if self.data_epoch < 0 or self.data_microbatch_offset < 0:
            raise ValueError(
                "resume checkpoint contains an invalid data cursor: "
                f"rank={rank} epoch={self.data_epoch} "
                f"offset={self.data_microbatch_offset}"
            )
        self.accelerator.print(
            f"[{self.log_label}] restored data cursor: rank={rank} "
            f"epoch={self.data_epoch} microbatch={self.data_microbatch_offset}"
        )

    def _resume_checkpoint_is_local(self) -> bool:
        path = self._resume_checkpoint_path
        return bool(
            path is not None
            and path.expanduser().resolve().parent == self.checkpoint_root.resolve()
        )

    def _gather_rank_data_cursors(self) -> list[dict[str, int]]:
        local = {
            "rank": int(self.accelerator.process_index),
            "epoch": int(self.data_epoch),
            "microbatch_offset": int(self.data_microbatch_offset),
        }
        world_size = int(self.accelerator.num_processes)
        if world_size == 1:
            return [local]
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError(
                "multi-process checkpointing requires initialized torch.distributed "
                "to save every rank's data cursor"
            )
        gathered: list[dict[str, int] | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered, local)
        result = [item for item in gathered if item is not None]
        if len(result) != world_size:
            raise RuntimeError("failed to gather every rank's data cursor")
        return sorted(result, key=lambda item: int(item["rank"]))

    def _should_reset_discriminator_for_checkpoint(self, path: Path) -> bool:
        """Reset D only when branching from a different experiment directory.

        Segment-to-segment resumes and crash recovery inside ``output_dir``
        keep the newly trained discriminator state.
        """

        if not self.config.discriminator_reset_on_resume:
            return False
        checkpoint = Path(path).expanduser().resolve()
        try:
            checkpoint.relative_to(self.output_dir)
        except ValueError:
            return True
        return False

    def _validate_resume_metadata(
        self,
        state: Mapping[str, Any],
        *,
        path: Path,
        step: int,
    ) -> None:
        saved_enabled = state.get("adversarial_enabled")
        saved_mask = state.get("mask_contract_fingerprint")
        current_mask = self.config.mask_contract_fingerprint
        if saved_mask is not None and current_mask is not None:
            validate_mask_transition_request(
                saved_fingerprint=str(saved_mask),
                saved_step=int(step),
                target_fingerprint=str(current_mask),
                transition=self.config.mask_transition,
                checkpoint=path,
            )
        current_enabled = bool(self.config.adversarial_enabled)
        if saved_enabled is None:
            if current_enabled:
                raise ValueError(
                    "GAN strict resume requires adversarial metadata in "
                    f"trainer_state.json: {path}. Use this legacy checkpoint only "
                    "as initialization."
                )
            return
        if bool(saved_enabled) != current_enabled:
            raise ValueError(
                "resume checkpoint adversarial_enabled does not match the current "
                f"configuration: saved={bool(saved_enabled)} current={current_enabled}"
            )
        if not current_enabled:
            return
        saved_gan_fingerprint = state.get("adversarial_config_fingerprint")
        current_gan_fingerprint = _adversarial_config_fingerprint(self.config)
        if saved_gan_fingerprint is None:
            if self.config.adversarial_backend != "multiview_stft_pqmf_v1":
                raise ValueError(
                    "GAN strict resume requires adversarial_config_fingerprint "
                    f"for backend={self.config.adversarial_backend}: {path}"
                )
        elif str(saved_gan_fingerprint) != current_gan_fingerprint:
            transition = dict(
                self.config.adversarial_weight_transition or {}
            )
            transition_allowed = bool(transition.get("enabled", False))
            transition_allowed = transition_allowed and (
                str(saved_gan_fingerprint)
                == str(transition.get("source_fingerprint", ""))
            )
            transition_allowed = transition_allowed and (
                int(step) == int(transition.get("source_step", -1))
            )
            if not transition_allowed:
                raise ValueError(
                    "resume checkpoint GAN configuration fingerprint does not "
                    "match and no exact weight transition was declared: "
                    f"saved={saved_gan_fingerprint!r} "
                    f"current={current_gan_fingerprint!r}"
                )
        saved_start = int(state.get("adversarial_start_step", -1))
        if saved_start != int(self.config.adversarial_start_step):
            raise ValueError(
                "resume checkpoint adversarial_start_step does not match the current "
                f"configuration: saved={saved_start} "
                f"current={self.config.adversarial_start_step}"
            )
        saved_active = bool(state.get("gan_active", False))
        expected_active = self._gan_active(step)
        if saved_active != expected_active:
            raise ValueError(
                "resume checkpoint GAN activation metadata is inconsistent with its "
                f"global step: step={step} saved_active={saved_active} "
                f"expected_active={expected_active}"
            )
        saved_artifact_enabled = bool(
            state.get("artifact_discriminator_enabled", False)
        )
        current_artifact_enabled = bool(
            getattr(
                self.config,
                "artifact_discriminator_enabled",
                False,
            )
        )
        if saved_artifact_enabled:
            if not current_artifact_enabled:
                raise ValueError(
                    "resume checkpoint contains an artifact discriminator but "
                    "the current configuration disables it"
                )
            saved_artifact_fingerprint = str(
                state.get(
                    "artifact_discriminator_config_fingerprint", ""
                )
            )
            current_artifact_fingerprint = (
                _artifact_discriminator_config_fingerprint(self.config)
            )
            if saved_artifact_fingerprint != current_artifact_fingerprint:
                raise ValueError(
                    "resume checkpoint artifact discriminator configuration "
                    "does not match: "
                    f"saved={saved_artifact_fingerprint!r} "
                    f"current={current_artifact_fingerprint!r}"
                )
            saved_gan_batch = int(
                state.get("adversarial_gan_batch_size", -1)
            )
            if saved_gan_batch != int(
                self.config.adversarial_gan_batch_size
            ):
                raise ValueError(
                    "resume checkpoint GAN subbatch size does not match: "
                    f"saved={saved_gan_batch} "
                    f"current={self.config.adversarial_gan_batch_size}"
                )
            saved_loss_fingerprint = state.get(
                "artifact_loss_config_fingerprint"
            )
            current_loss_fingerprint = (
                _artifact_loss_config_fingerprint(self.config)
            )
            if saved_loss_fingerprint is not None:
                if str(saved_loss_fingerprint) != current_loss_fingerprint:
                    raise ValueError(
                        "resume checkpoint artifact loss configuration does "
                        "not match: "
                        f"saved={saved_loss_fingerprint!r} "
                        f"current={current_loss_fingerprint!r}"
                    )
            elif self.config.artifact_family_combination != "convex":
                transition = dict(
                    self.config.artifact_loss_transition or {}
                )
                transition_allowed = bool(
                    transition.get("enabled", False)
                )
                transition_allowed = transition_allowed and (
                    str(saved_artifact_fingerprint)
                    == str(
                        transition.get(
                            "source_artifact_discriminator_fingerprint",
                            "",
                        )
                    )
                )
                transition_allowed = transition_allowed and (
                    int(step) == int(transition.get("source_step", -1))
                )
                if not transition_allowed:
                    raise ValueError(
                        "changing the legacy convex artifact loss requires "
                        "an exact loss_transition declaration: "
                        f"saved_artifact={saved_artifact_fingerprint!r} "
                        f"step={step}"
                    )
        elif current_artifact_enabled:
            expected_source = str(
                getattr(
                    self.config,
                    "artifact_bootstrap_source_fingerprint",
                    "",
                )
            )
            if (
                not expected_source
                or str(saved_gan_fingerprint) != expected_source
            ):
                raise ValueError(
                    "adding the artifact discriminator requires an explicit "
                    "bootstrap_source_fingerprint matching the source GAN "
                    "checkpoint: "
                    f"saved={saved_gan_fingerprint!r} "
                    f"declared={expected_source!r}"
                )

    def _validate_resume_checkpoint(self, path: Path) -> None:
        if self.discriminator is None:
            return
        required = (
            "discriminator.safetensors",
            "optimizer_1.bin",
            "scheduler_1.bin",
        )
        missing = [name for name in required if not (path / name).is_file()]
        if missing:
            raise FileNotFoundError(
                "GAN strict resume requires discriminator model/optimizer/scheduler state; "
                f"missing={missing} in {path}. Use the legacy checkpoint only as initialization."
            )

    def _validate_artifact_resume_checkpoint(
        self,
        state: Mapping[str, Any],
        *,
        path: Path,
    ) -> None:
        if not bool(state.get("artifact_discriminator_enabled", False)):
            return
        required = (
            "artifact_discriminator.safetensors",
            "artifact_optimizer.bin",
            "artifact_scheduler.bin",
        )
        missing = [name for name in required if not (path / name).is_file()]
        if missing:
            raise FileNotFoundError(
                "artifact discriminator strict resume requires model, optimizer, "
                f"and scheduler state; missing={missing} in {path}"
            )

    def _finalize_discriminator_checkpoint(self, path: Path) -> None:
        if self.discriminator is None:
            return
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            generated = path / "model_1.safetensors"
            destination = path / "discriminator.safetensors"
            if not generated.is_file():
                raise FileNotFoundError(
                    "Accelerate did not write the expected second model state: "
                    f"{generated}"
                )
            generated.replace(destination)
        self.accelerator.wait_for_everyone()

    def _save_artifact_checkpoint(self, path: Path) -> None:
        if self.artifact_discriminator is None:
            return
        if (
            self.artifact_discriminator_optimizer is None
            or self.artifact_discriminator_scheduler is None
        ):
            raise RuntimeError(
                "artifact discriminator optimizer/scheduler is unavailable"
            )
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            try:
                from safetensors.torch import save_file
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "saving artifact discriminator requires safetensors"
                ) from exc
            module = _unwrap_prepared_model(self.artifact_discriminator)
            state = {
                name: value.detach().cpu().contiguous()
                for name, value in module.state_dict().items()
            }
            save_file(
                state,
                str(path / "artifact_discriminator.safetensors"),
            )
            torch.save(
                self.artifact_discriminator_optimizer.state_dict(),
                path / "artifact_optimizer.bin",
            )
            torch.save(
                self.artifact_discriminator_scheduler.state_dict(),
                path / "artifact_scheduler.bin",
            )
        self.accelerator.wait_for_everyone()

    def _load_artifact_checkpoint(self, path: Path) -> None:
        if (
            self.artifact_discriminator is None
            or self.artifact_discriminator_optimizer is None
            or self.artifact_discriminator_scheduler is None
        ):
            raise RuntimeError(
                "artifact discriminator must be prepared before loading"
            )
        try:
            from safetensors.torch import load_file
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "loading artifact discriminator requires safetensors"
            ) from exc
        module = _unwrap_prepared_model(self.artifact_discriminator)
        module.load_state_dict(
            load_file(
                str(path / "artifact_discriminator.safetensors"),
                device="cpu",
            ),
            strict=True,
        )
        self.artifact_discriminator_optimizer.load_state_dict(
            torch.load(
                path / "artifact_optimizer.bin",
                map_location="cpu",
                weights_only=False,
            )
        )
        self.artifact_discriminator_scheduler.load_state_dict(
            torch.load(
                path / "artifact_scheduler.bin",
                map_location="cpu",
                weights_only=False,
            )
        )

    def _latest_resumable_checkpoint(self) -> Path | None:
        candidates = [self._last_checkpoint_dir()]
        candidates.extend(
            path
            for path in self.checkpoint_root.glob("checkpoint-*")
            if path.name != "checkpoint-last"
        )
        resumable: list[tuple[int, int, Path]] = []
        for path in candidates:
            state_path = path / "trainer_state.json"
            if not state_path.is_file():
                continue
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                step = int(state["global_step"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            resumable.append((step, int(path.name == "checkpoint-last"), path))
        return (
            max(resumable, default=None, key=lambda item: item[:2])[2]
            if resumable
            else None
        )

    def _update_internal_ema(self, step: int) -> None:
        if not self.config.update_internal_ema:
            return
        model = _unwrap_prepared_model(self.model)
        update = getattr(model, "update_internal_ema", None)
        if callable(update):
            update(step)

    def _build_ema_state(self) -> dict[str, Tensor] | None:
        if self.config.ema_decay <= 0.0:
            return None
        model = _unwrap_prepared_model(self.model)
        online = checkpoint_model_state(model, to_cpu=False)
        return {key: value.detach().clone() for key, value in online.items()}

    @torch.no_grad()
    def _update_ema(self, step: int) -> None:
        if self.ema_state is None:
            return
        if step < self.config.ema_start_step:
            self._reset_ema_from_online()
            return
        if step % self.config.ema_update_every != 0:
            return
        model = _unwrap_prepared_model(self.model)
        online = checkpoint_model_state(model, to_cpu=False)
        _validate_ema_keys_and_shapes(self.ema_state, online, context="EMA update")
        parameters = dict(model.named_parameters())
        decay = float(self.config.ema_decay)
        for key, ema_value in self.ema_state.items():
            if self.config.ema_trainable_only:
                parameter = parameters.get(key)
                if parameter is None or not parameter.requires_grad:
                    continue
            value = online[key].detach().to(device=ema_value.device)
            if torch.is_floating_point(ema_value) or torch.is_complex(ema_value):
                ema_value.mul_(decay).add_(
                    value.to(dtype=ema_value.dtype), alpha=1.0 - decay
                )
            else:
                ema_value.copy_(value)

    @torch.no_grad()
    def _reset_ema_from_online(self) -> None:
        if self.ema_state is None:
            return
        model = _unwrap_prepared_model(self.model)
        online = checkpoint_model_state(model, to_cpu=False)
        _validate_ema_keys_and_shapes(self.ema_state, online, context="EMA reset")
        for key, ema_value in self.ema_state.items():
            ema_value.copy_(
                online[key].detach().to(device=ema_value.device, dtype=ema_value.dtype)
            )

    def _save_ema_checkpoint(self, path: Path) -> None:
        if self.ema_state is None or not self.accelerator.is_main_process:
            return
        try:
            from safetensors.torch import save_file
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "saving EMA checkpoints requires safetensors"
            ) from exc
        state = {
            key: value.detach().cpu().clone().contiguous()
            for key, value in self.ema_state.items()
        }
        save_file(state, str(path / "ema_model.safetensors"))

    def _load_ema_checkpoint(self, path: Path) -> None:
        if self.ema_state is None:
            return
        ema_path = path / "ema_model.safetensors"
        if not ema_path.is_file():
            self._reset_ema_from_online()
            self.accelerator.print(
                f"[{self.log_label}] EMA checkpoint missing at {ema_path}; "
                "initialized from online filtered state"
            )
            return
        try:
            from safetensors.torch import load_file
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "loading EMA checkpoints requires safetensors"
            ) from exc
        saved = load_file(str(ema_path), device="cpu")
        _validate_ema_keys_and_shapes(
            self.ema_state, saved, context=f"EMA load {ema_path}"
        )
        for key, ema_value in self.ema_state.items():
            ema_value.copy_(
                saved[key].to(device=ema_value.device, dtype=ema_value.dtype)
            )
        self.accelerator.print(f"[{self.log_label}] loaded EMA checkpoint: {ema_path}")

    def _load_compatible_warm_start_ema(self, path: Path) -> None:
        if self.ema_state is None:
            return
        ema_path = path / "ema_model.safetensors"
        if not ema_path.is_file():
            self._reset_ema_from_online()
            return
        try:
            from safetensors.torch import load_file
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "loading warm-start EMA requires safetensors"
            ) from exc
        saved = load_file(str(ema_path), device="cpu")
        missing = sorted(set(self.ema_state).difference(saved))
        unexpected = sorted(set(saved).difference(self.ema_state))
        disallowed = [
            key
            for key in missing
            if not key.startswith("fm_encoder.global_condition_proj.")
        ]
        mismatched = sorted(
            key
            for key in set(saved).intersection(self.ema_state)
            if tuple(saved[key].shape) != tuple(self.ema_state[key].shape)
        )
        if disallowed or unexpected or mismatched:
            raise RuntimeError(
                "warm-start EMA is incompatible: "
                f"missing={disallowed[:8]} mismatched={mismatched[:8]} "
                f"unexpected={unexpected[:8]}"
            )
        online = checkpoint_model_state(
            _unwrap_prepared_model(self.model), to_cpu=False
        )
        for key, ema_value in self.ema_state.items():
            value = saved.get(key, online[key].detach().cpu())
            ema_value.copy_(
                value.to(device=ema_value.device, dtype=ema_value.dtype)
            )
        self.accelerator.print(
            f"[{self.log_label}] loaded compatible warm-start EMA: {ema_path}; "
            f"new_keys={len(missing)}"
        )

    def _lr_lambda(self, scheduler_step: int) -> float:
        step = scheduler_step + 1
        warmup = max(int(self.config.warmup_steps), 0)
        warmup_scale = min(float(step) / float(max(warmup, 1)), 1.0) if warmup else 1.0
        if self.config.scheduler in {"constant", "constant_with_warmup"}:
            return 1.0 if self.config.scheduler == "constant" else warmup_scale
        decay_start = warmup if self.config.scheduler == "cosine_with_warmup" else 0
        decay_progress = (step - decay_start) / max(
            self.config.max_steps - decay_start, 1
        )
        decay_progress = min(max(decay_progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return warmup_scale * cosine

    def _checkpoint_dir(self, step: int) -> Path:
        return self.checkpoint_root / f"checkpoint-{step:08d}"

    def _last_checkpoint_dir(self) -> Path:
        return self.checkpoint_root / "checkpoint-last"

    def _rotate_checkpoints(self) -> None:
        keep = self.config.keep_last_n_checkpoints
        if keep < 0:
            return
        checkpoints = sorted(
            path
            for path in self.checkpoint_root.glob("checkpoint-*")
            if path.is_dir() and path.name != "checkpoint-last"
        )
        while len(checkpoints) > keep:
            shutil.rmtree(checkpoints.pop(0))

    def _log_step(
        self,
        output: Mapping[str, Any],
        batch: Mapping[str, Any],
        *,
        step_seconds: float,
        epoch: int,
        steps_per_epoch: int,
    ) -> None:
        loss = output["loss"].detach().float().mean().reshape(1)
        world_loss = self.accelerator.gather_for_metrics(loss).mean()
        output_metrics = self._mean_output_scalar_metrics(output)
        batch_metrics = _batch_metrics(batch)
        batch_totals = torch.tensor(
            [
                float(batch_metrics.get("batch_size", 0.0)),
                float(batch_metrics.get("audio_seconds", 0.0)),
                float(batch_metrics.get("audiocaps_samples", 0.0)),
                float(batch_metrics.get("wavcaps_samples", 0.0)),
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        )
        batch_totals = self.accelerator.reduce(batch_totals, reduction="sum")
        batch_metrics["global_batch_size"] = int(round(float(batch_totals[0].item())))
        batch_metrics["global_audio_seconds"] = float(batch_totals[1].item())
        batch_metrics["global_audiocaps_samples"] = int(
            round(float(batch_totals[2].item()))
        )
        batch_metrics["global_wavcaps_samples"] = int(
            round(float(batch_totals[3].item()))
        )
        global_batch_size = max(int(batch_metrics["global_batch_size"]), 1)
        batch_metrics["audiocaps_fraction"] = (
            int(batch_metrics["global_audiocaps_samples"]) / global_batch_size
        )
        batch_metrics["wavcaps_replay_fraction"] = (
            int(batch_metrics["global_wavcaps_samples"]) / global_batch_size
        )
        elapsed = max(time.monotonic() - self._run_start_time, 1.0e-6)
        completed = max(self.global_step - self._run_start_step, 1)
        seconds_per_step = elapsed / completed
        if steps_per_epoch > 0:
            epoch_current = (max(self.global_step, 1) - 1) // steps_per_epoch + 1
            epoch_step = (max(self.global_step, 1) - 1) % steps_per_epoch + 1
            total_epochs = int(math.ceil(self.config.max_steps / steps_per_epoch))
            epoch_progress = epoch_step / steps_per_epoch
            epochs_completed = self.global_step / steps_per_epoch
        else:
            # Some user-provided iterable loaders do not expose a length. Keep
            # the loop epoch useful while marking the unknown quantities as 0.
            epoch_current = epoch + 1
            epoch_step = 0
            total_epochs = 0
            epoch_progress = 0.0
            epochs_completed = float(epoch)
        metrics: dict[str, Any] = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "step": self.global_step,
            "max_steps": self.config.max_steps,
            "epoch": epoch,
            "epoch_current": epoch_current,
            "total_epochs": total_epochs,
            "epoch_step": epoch_step,
            "steps_per_epoch": steps_per_epoch,
            "epoch_progress": epoch_progress,
            "epochs_completed": epochs_completed,
            "progress": self.global_step / max(self.config.max_steps, 1),
            "loss": float(world_loss.item()),
            "lr": float(self.scheduler.get_last_lr()[0]),
            "ema_decay": float(self.config.ema_decay),
            "step_seconds": float(step_seconds),
            "seconds_per_step": seconds_per_step,
            "eta_seconds": seconds_per_step
            * max(self.config.max_steps - self.global_step, 0),
            **output_metrics,
            **batch_metrics,
            **_memory_metrics(self.accelerator.device),
        }
        for name in (
            "prior_fm_loss",
            "prior_fm_loss_ssl",
            "prior_fm_loss_full_generation",
            "weighted_same_mrstft_loss",
            "same_mrstft_spectral_contrast",
            "same_mrstft_adaptive_log_magnitude",
            "same_mrstft_instantaneous_frequency",
            "same_mrstft_group_delay",
            "same_mrstft_complex_distance",
            "weighted_highband_detail_loss",
            "highband_detail_log_magnitude",
            "highband_detail_spectral_flux",
            "weighted_stationary_artifact_loss",
            "stationary_artifact_fake_score_db",
            "stationary_artifact_real_score_db",
            "weighted_decoder_distillation_loss",
            "weighted_parameter_anchor_loss",
            "parameter_anchor_mean_square",
            "gan_active",
            "gan_discriminator_loss",
            "gan_generator_loss",
            "gan_feature_matching_loss",
            "mask_ratio",
            "mask_full_generation_fraction",
            "mask_ssl_visible_ratio_mean",
            "target_ema_latent_cosine",
            "target_ema_ssl_cosine",
            "target_ema_fullgen_cosine",
            "target_ema_latent_delta_rms",
            "target_ema_ssl_delta_rms",
            "target_ema_fullgen_delta_rms",
            "target_ema_norm_ratio",
            "target_ema_param_cosine",
            "target_ema_param_delta_rms",
            "waveform_l1_metric",
            "log_mel_metric",
            "global_audio_seconds",
            "global_batch_size",
        ):
            metrics.setdefault(name, 0.0)
        metrics["audio_throughput"] = float(metrics["global_audio_seconds"]) / max(
            float(step_seconds), 1.0e-6
        )
        if self.accelerator.is_main_process:
            self._print_step_metrics(metrics)
            with (self.log_root / "train_log.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(
                    json.dumps(to_jsonable(metrics), ensure_ascii=False) + "\n"
                )

    def _print_step_metrics(self, metrics: Mapping[str, Any]) -> None:
        """Print task-specific progress while keeping JSON logging shared."""

        print(
            "[{timestamp}] [TTA] step={step}/{max_steps} "
            "epoch={epoch_current}/{total_epochs} "
            "epoch_step={epoch_step}/{steps_per_epoch} "
            "total={loss:.5f} "
            "prior_fm={prior_fm_loss:.5f} ssl={prior_fm_loss_ssl:.5f} "
            "fullgen={prior_fm_loss_full_generation:.5f} "
            "mrstft={weighted_same_mrstft_loss:.5f} "
            "highband={weighted_highband_detail_loss:.5f} "
            "stationary={weighted_stationary_artifact_loss:.5f} "
            "distill={weighted_decoder_distillation_loss:.5f} "
            "anchor={weighted_parameter_anchor_loss:.5f} "
            "gan={gan_active:.0f} d={gan_discriminator_loss:.5f} "
            "g_adv={gan_generator_loss:.5f} feat={gan_feature_matching_loss:.5f} "
            "g_w={gan_generator_weight:.5f} "
            "fm_w={gan_feature_matching_weight:.5f} "
            "lr={lr:.3e}".format(**metrics),
            flush=True,
        )
        print(
            "[{timestamp}] [TTA] mrstft contrast={same_mrstft_spectral_contrast:.5f} "
            "logmag={same_mrstft_adaptive_log_magnitude:.5f} "
            "if={same_mrstft_instantaneous_frequency:.5f} "
            "gd={same_mrstft_group_delay:.5f} "
            "complex={same_mrstft_complex_distance:.5f}".format(**metrics),
            flush=True,
        )
        if "wave_x_mse" in metrics or "prior_x_mse" in metrics:
            x_mse_metrics = dict(metrics)
            x_mse_metrics.setdefault("wave_x_mse", 0.0)
            x_mse_metrics.setdefault("prior_x_mse", 0.0)
            print(
                "[{timestamp}] [TTA] x_mse wave={wave_x_mse:.5f} "
                "prior={prior_x_mse:.5f}".format(**x_mse_metrics),
                flush=True,
            )
        if "wave_v_mse" in metrics or "prior_v_mse" in metrics:
            v_mse_metrics = dict(metrics)
            v_mse_metrics.setdefault("wave_v_mse", 0.0)
            v_mse_metrics.setdefault("prior_v_mse", 0.0)
            print(
                "[{timestamp}] [TTA] v_mse wave={wave_v_mse:.5f} "
                "prior={prior_v_mse:.5f}".format(**v_mse_metrics),
                flush=True,
            )
        print(
            "[{timestamp}] [TTA] target_ema cos={target_ema_latent_cosine:.5f} "
            "ssl_cos={target_ema_ssl_cosine:.5f} "
            "full_cos={target_ema_fullgen_cosine:.5f} "
            "delta={target_ema_latent_delta_rms:.5f} "
            "ssl_delta={target_ema_ssl_delta_rms:.5f} "
            "full_delta={target_ema_fullgen_delta_rms:.5f} "
            "norm_ratio={target_ema_norm_ratio:.5f} "
            "param_cos={target_ema_param_cosine:.7f} "
            "param_delta={target_ema_param_delta_rms:.7f}".format(**metrics),
            flush=True,
        )
        print(
            "[{timestamp}] [TTA] batch={global_batch_size:.0f} mask={mask_ratio:.3f} "
            "full={mask_full_generation_fraction:.3f} "
            "ssl_visible={mask_ssl_visible_ratio_mean:.3f} "
            "wav_l1={waveform_l1_metric:.5f} mel={log_mel_metric:.5f} "
            "audio={global_audio_seconds:.1f}s throughput={audio_throughput:.1f}x "
            "step_time={step_seconds:.3f}s eta={eta_seconds:.0f}s".format(**metrics),
            flush=True,
        )
        if float(metrics.get("visible_condition_noise_enabled", 0.0)) > 0.0:
            print(
                "[{timestamp}] [TTA] noisy_visible noise_rms="
                "{visible_condition_noise_rms:.5f} delta_rms="
                "{visible_condition_noisy_delta_rms:.5f} mean_t="
                "{visible_condition_effective_t_mean:.5f}".format(**metrics),
                flush=True,
            )
        if float(metrics.get("prior_loss_all_valid_enabled", 0.0)) > 0.0:
            print(
                "[{timestamp}] [TTA] all_token_fm visible="
                "{prior_fm_loss_visible:.5f} target_ssl="
                "{prior_fm_loss_ssl:.5f} fullgen="
                "{prior_fm_loss_full_generation:.5f}".format(**metrics),
                flush=True,
            )

    def _mean_output_scalar_metrics(
        self,
        output: Mapping[str, Any],
    ) -> dict[str, float]:
        """Average scalar model metrics over ranks before logging them."""

        local = _output_scalar_metrics(output)
        if not local:
            return {}
        names = tuple(sorted(local))
        values = torch.tensor(
            [float(local[name]) for name in names],
            device=self.accelerator.device,
            dtype=torch.float32,
        )
        means = self.accelerator.reduce(values, reduction="mean")
        return {
            name: float(value)
            for name, value in zip(names, means.detach().cpu().tolist(), strict=True)
        }

    def _should_sample(self, step: int) -> bool:
        return (
            self.config.sample_every > 0
            and bool(self.config.sample_captions)
            and step % self.config.sample_every == 0
        )

    def _should_evaluate_encoder(self, step: int) -> bool:
        return (
            self.config.encoder_eval_every > 0
            and step % self.config.encoder_eval_every == 0
        )

    def _should_evaluate_audiocaps(self, step: int) -> bool:
        return (
            step > 0
            and int(getattr(self.config, "audiocaps_eval_every", 0)) > 0
            and step % int(self.config.audiocaps_eval_every) == 0
        )

    def _should_evaluate_model_diagnostics(self, step: int) -> bool:
        return (
            self.config.model_diagnostics_every > 0
            and step % self.config.model_diagnostics_every == 0
        )

    def _evaluate_model_diagnostics(self, step: int) -> None:
        if self._model_diagnostics_batch is None:
            raise RuntimeError("model diagnostics requested before a batch was cached")
        model = _unwrap_prepared_model(self.model)
        method = getattr(model, "training_diagnostics", None)
        if not callable(method):
            raise AttributeError(
                "model_diagnostics.every requires model.training_diagnostics()"
            )
        was_training = model.training
        model.eval()
        output: Any = None
        error: str | None = None
        try:
            batch = _move_to_device(
                self._model_diagnostics_batch,
                self.accelerator.device,
            )
            with torch.inference_mode(), self.accelerator.autocast():
                output = method(
                    batch,
                    seed=int(
                        self.config.model_diagnostics_config.get(
                            "seed",
                            self.config.seed,
                        )
                    ),
                )
        except Exception as exc:
            error = repr(exc)
        finally:
            model.train(was_training)
        failed = torch.tensor(
            float(error is not None),
            device=self.accelerator.device,
            dtype=torch.float32,
        )
        failure_count = int(
            round(
                float(
                    self.accelerator.reduce(failed, reduction="sum")
                    .detach()
                    .cpu()
                    .item()
                )
            )
        )
        if failure_count:
            if self.accelerator.is_main_process:
                metrics = {
                    "timestamp": datetime.now()
                    .astimezone()
                    .isoformat(timespec="seconds"),
                    "step": int(step),
                    "status": "failed",
                    "failed_ranks": failure_count,
                    "rank0_error": error,
                }
                self.accelerator.print(
                    f"[{self.log_label}] model diagnostics failed at step={step}: "
                    f"{metrics}",
                    flush=True,
                )
                with (self.log_root / "model_diagnostics.jsonl").open(
                    "a",
                    encoding="utf-8",
                ) as handle:
                    handle.write(
                        json.dumps(to_jsonable(metrics), ensure_ascii=False) + "\n"
                    )
            self.accelerator.wait_for_everyone()
            return
        if not isinstance(output, Mapping):
            raise TypeError("model.training_diagnostics() must return a mapping")
        local = _output_scalar_metrics(output)
        names = tuple(sorted(local))
        if not names:
            raise ValueError("model.training_diagnostics() returned no scalar metrics")
        values = torch.tensor(
            [float(local[name]) for name in names],
            device=self.accelerator.device,
            dtype=torch.float32,
        )
        means = self.accelerator.reduce(values, reduction="mean")
        metrics = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "step": int(step),
            **{
                name: float(value)
                for name, value in zip(
                    names,
                    means.detach().cpu().tolist(),
                    strict=True,
                )
            },
        }
        if self.accelerator.is_main_process:
            self.accelerator.print(
                f"[{self.log_label}] model diagnostics step={step}: {metrics}",
                flush=True,
            )
            with (self.log_root / "model_diagnostics.jsonl").open(
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(
                    json.dumps(to_jsonable(metrics), ensure_ascii=False) + "\n"
                )
        self.accelerator.wait_for_everyone()

    def _evaluate_audiocaps(self, step: int) -> None:
        """Pause all ranks while rank 0 evaluates a saved milestone."""

        evaluation_name = str(
            self.config.audiocaps_eval_config.get(
                "name",
                "audiocaps_tango",
            )
        ).strip()
        if not evaluation_name or "/" in evaluation_name or "\\" in evaluation_name:
            raise ValueError("audiocaps_eval.name must be a single directory name")
        status_dir = (
            self.output_dir / "evaluation" / evaluation_name / "hook_status"
        )
        status_path = status_dir / f"step-{step:08d}.json"
        if self.accelerator.is_main_process:
            status_dir.mkdir(parents=True, exist_ok=True)
            status_path.unlink(missing_ok=True)
        self.accelerator.wait_for_everyone()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.accelerator.is_main_process:
            _write_json_atomic(
                status_path,
                {
                    "status": "running",
                    "step": int(step),
                    "started_at": datetime.now()
                    .astimezone()
                    .isoformat(timespec="seconds"),
                },
            )
            started = time.monotonic()
            try:
                from task_audio.eval.periodic_audiocaps import (
                    run_periodic_audiocaps_eval,
                )

                results = run_periodic_audiocaps_eval(
                    output_dir=self.output_dir,
                    checkpoint=self._checkpoint_dir(step),
                    step=step,
                    config=self.config.audiocaps_eval_config,
                )
                status = {
                    "status": "completed",
                    "step": int(step),
                    "duration_seconds": float(time.monotonic() - started),
                    "results": results,
                }
            except Exception as exc:
                status = {
                    "status": "failed",
                    "step": int(step),
                    "duration_seconds": float(time.monotonic() - started),
                    "error": repr(exc),
                }
            status["finished_at"] = (
                datetime.now().astimezone().isoformat(timespec="seconds")
            )
            _write_json_atomic(status_path, status)
        else:
            deadline = time.monotonic() + float(
                self.config.audiocaps_eval_config.get("timeout_seconds", 21_600)
            )
            poll_seconds = float(
                self.config.audiocaps_eval_config.get("poll_seconds", 2.0)
            )
            status = {}
            while time.monotonic() < deadline:
                try:
                    status = json.loads(status_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    # CPFS can transiently return ESTALE/EIO while another
                    # rank atomically replaces the status file. Treat that
                    # exactly like a not-yet-visible status and poll again;
                    # aborting one rank would tear down the full DDP job.
                    status = {}
                if status.get("status") in {"completed", "failed"}:
                    break
                time.sleep(max(poll_seconds, 0.1))
            else:
                raise TimeoutError(
                    f"timed out waiting for AudioCaps evaluation: {status_path}"
                )
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if status.get("status") == "completed":
                self.accelerator.print(
                    f"[{self.log_label}] AudioCaps evaluation completed at step {step}: "
                    f"{status.get('results')}",
                    flush=True,
                )
            else:
                self.accelerator.print(
                    f"[{self.log_label}] AudioCaps evaluation failed at step {step}; "
                    f"training continues: {status.get('error')}",
                    flush=True,
                )
        if self.discriminator is not None:
            self.discriminator.train()
        self.model.train()

    def _evaluate_encoder(self, step: int) -> None:
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            from task_audio.eval import (
                ESC50ProbeConfig,
                VisibleConditionEvalConfig,
                format_probe_metrics,
                format_visible_condition_metrics,
                run_esc50_linear_probe,
                run_visible_condition_dependence,
                write_probe_metrics,
                write_visible_condition_metrics,
            )

            model = _unwrap_prepared_model(self.model)
            started = time.monotonic()
            try:
                probe_config = ESC50ProbeConfig.from_mapping(
                    self.config.encoder_eval_config,
                    sample_rate=_model_sample_rate(model),
                    seed=self.config.seed,
                    mixed_precision=self.config.mixed_precision,
                )
                metrics = run_esc50_linear_probe(
                    model,
                    probe_config=probe_config,
                    output_dir=self.output_dir,
                    step=step,
                    device=self.accelerator.device,
                )
            except Exception as exc:
                metrics = {
                    "timestamp": datetime.now()
                    .astimezone()
                    .isoformat(timespec="seconds"),
                    "dataset": "esc50",
                    "step": int(step),
                    "fold": int(self.config.encoder_eval_config.get("fold", 1)),
                    "epochs": int(self.config.encoder_eval_config.get("epochs", 100)),
                    "encoder": str(
                        self.config.encoder_eval_config.get("encoder", "target_ema")
                    ),
                    "pool": str(
                        self.config.encoder_eval_config.get("pool", "mean_time")
                    ),
                    "error": repr(exc),
                    "duration_seconds": float(time.monotonic() - started),
                }
                write_probe_metrics(self.output_dir, metrics)
            message = format_probe_metrics(metrics)
            if self.log_label != "TTA" and message.startswith("[TTA]"):
                message = f"[{self.log_label}]" + message[len("[TTA]") :]
            self.accelerator.print(message, flush=True)
            visible_config = self.config.encoder_eval_config.get(
                "visible_condition", {}
            )
            if isinstance(visible_config, Mapping) and bool(
                visible_config.get("enabled", False)
            ):
                visible_started = time.monotonic()
                try:
                    dependence_config = VisibleConditionEvalConfig.from_mapping(
                        self.config.encoder_eval_config,
                        sample_rate=_model_sample_rate(model),
                        seed=self.config.seed,
                        mixed_precision=self.config.mixed_precision,
                    )
                    visible_metrics = run_visible_condition_dependence(
                        model,
                        eval_config=dependence_config,
                        output_dir=self.output_dir,
                        step=step,
                        device=self.accelerator.device,
                    )
                except Exception as exc:
                    visible_metrics = {
                        "timestamp": datetime.now()
                        .astimezone()
                        .isoformat(timespec="seconds"),
                        "dataset": "esc50_visible_condition",
                        "step": int(step),
                        "error": repr(exc),
                        "duration_seconds": float(time.monotonic() - visible_started),
                    }
                    write_visible_condition_metrics(self.output_dir, visible_metrics)
                visible_message = format_visible_condition_metrics(visible_metrics)
                if self.log_label != "TTA" and visible_message.startswith("[TTA]"):
                    visible_message = (
                        f"[{self.log_label}]" + visible_message[len("[TTA]") :]
                    )
                self.accelerator.print(visible_message, flush=True)
        self.accelerator.wait_for_everyone()

    def _save_fixed_samples(self, step: int) -> None:
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            from task_audio.infer.generation import generate_to_directory

            model = _unwrap_prepared_model(self.model)
            was_training = model.training
            model.eval()
            try:
                generate_to_directory(
                    model,
                    self.config.sample_captions,
                    self.output_dir / "infer" / f"step-{step:08d}",
                    seconds=self.config.sample_seconds,
                    prior_steps=self.config.sample_prior_steps,
                    wave_steps=self.config.sample_wave_steps,
                    solver=self.config.sample_solver,
                    cfg_strength=self.config.sample_cfg_strength,
                    wave_cfg_strength=self.config.sample_wave_cfg_strength,
                    cfg_rescale=self.config.sample_cfg_rescale,
                    seed=self.config.sample_seed,
                    sample_rate=_model_sample_rate(model),
                )
            finally:
                model.train(was_training)
        self.accelerator.wait_for_everyone()


def _load_accelerate() -> tuple[Any, Any, Any, Any]:
    try:
        from accelerate import Accelerator
        from accelerate.utils import (
            DataLoaderConfiguration,
            DistributedDataParallelKwargs,
            set_seed,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TTATrainer requires accelerate; install it in the training environment"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "accelerate could not be imported in the current runtime; verify the PyTorch/CUDA installation"
        ) from exc
    return (
        Accelerator,
        DataLoaderConfiguration,
        DistributedDataParallelKwargs,
        set_seed,
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _copy_batch_to_cpu(value: Any) -> Any:
    """Keep one immutable rank-local batch for repeatable model diagnostics."""

    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _copy_batch_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_batch_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_batch_to_cpu(item) for item in value)
    return deepcopy(value)


def _unwrap_prepared_model(model: nn.Module) -> nn.Module:
    """Unwrap the supported DDP/DataParallel wrappers without importing optional runtimes."""

    seen: set[int] = set()
    while id(model) not in seen:
        seen.add(id(model))
        inner = getattr(model, "module", None)
        if not isinstance(inner, nn.Module):
            break
        model = inner
    return model


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _optimizer_steps_per_epoch(
    loader: Any,
    *,
    world_size: int,
    grad_accumulation_steps: int,
) -> int:
    """Estimate global optimizer updates required to consume one dataset epoch.

    A regular loader has already been sharded by ``Accelerator.prepare``, so
    its length is the number of local microbatches. AudioSet instead partitions
    parquet files inside the iterable dataset; its unwrapped DataLoader length
    describes the full dataset at the per-rank batch size and must additionally
    be divided by the DDP world size.
    """

    try:
        microbatches = int(len(loader))
    except (TypeError, NotImplementedError):
        return 0
    if microbatches <= 0:
        return 0
    if _handles_distributed_sharding(loader):
        microbatches = int(math.ceil(microbatches / max(int(world_size), 1)))
    return int(math.ceil(microbatches / max(int(grad_accumulation_steps), 1)))


def _loader_microbatches_per_epoch(loader: Any) -> int:
    try:
        value = int(len(loader))
    except (TypeError, NotImplementedError) as exc:
        raise RuntimeError(
            "resumable TTA training requires a finite-length dataloader"
        ) from exc
    if value < 1:
        raise RuntimeError("TTA train dataloader produced no batches")
    return value


def _self_sharded_microbatches_per_epoch(loader: Any) -> int:
    dataset = getattr(loader, "dataset", None)
    value = getattr(dataset, "nominal_updates_per_epoch", 0)
    return max(int(value), 0)


def _set_loader_epoch(loader: Any, epoch: int) -> None:
    updated = False
    setter = getattr(loader, "set_epoch", None)
    if callable(setter):
        setter(epoch)
        updated = True
    if not updated:
        sampler = getattr(loader, "sampler", None)
        setter = getattr(sampler, "set_epoch", None)
        if callable(setter):
            setter(epoch)
    dataset = getattr(loader, "dataset", None)
    setter = getattr(dataset, "set_epoch", None)
    if callable(setter):
        setter(epoch)


def _set_loader_start_offset(loader: Any, offset: int) -> bool:
    """Let cursor-aware iterable datasets seek without replaying old batches."""

    current = loader
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        setter = getattr(current, "set_start_offset", None)
        if callable(setter):
            setter(int(offset))
            return True
        dataset = getattr(current, "dataset", None)
        current = dataset if dataset is not current else None
    return False


def _handles_distributed_sharding(loader: Any) -> bool:
    dataset = getattr(loader, "dataset", None)
    return bool(getattr(dataset, "handles_distributed_sharding", False))


def _data_config_fingerprint(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        to_jsonable(config.get("data", {})),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _adversarial_config_fingerprint(config: TTATrainerConfig) -> str:
    payload = {
        "backend": config.adversarial_backend,
        "objective": config.adversarial_objective,
        "start_step": config.adversarial_start_step,
        "weight_warmup_steps": config.adversarial_weight_warmup_steps,
        "discriminator_weight": config.adversarial_discriminator_weight,
        "generator_weight": config.adversarial_generator_weight,
        "feature_matching_weight": config.adversarial_feature_matching_weight,
        "weight_transition": config.adversarial_weight_transition,
        "stft_fft_sizes": config.adversarial_stft_fft_sizes,
        "pqmf_enabled": config.adversarial_pqmf_enabled,
        "mpd_enabled": config.adversarial_mpd_enabled,
        "mpd_periods": config.adversarial_mpd_periods,
        "msd_enabled": config.adversarial_msd_enabled,
        "msd_scales": config.adversarial_msd_scales,
        "discriminator_learning_rate": config.discriminator_learning_rate,
        "discriminator_betas": config.discriminator_betas,
        "discriminator_weight_decay": config.discriminator_weight_decay,
        "discriminator_grad_clip_norm": config.discriminator_grad_clip_norm,
        "discriminator_update_every": config.discriminator_update_every,
    }
    encoded = json.dumps(
        to_jsonable(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_discriminator_config_fingerprint(
    config: TTATrainerConfig,
) -> str:
    payload = {
        "stft_fft_sizes": config.artifact_discriminator_stft_fft_sizes,
        "mpd_periods": config.artifact_discriminator_mpd_periods,
        "mixed_precision": config.artifact_discriminator_mixed_precision,
        "learning_rate": config.artifact_discriminator_learning_rate,
        "betas": config.artifact_discriminator_betas,
        "weight_decay": config.artifact_discriminator_weight_decay,
        "family_target_weight": config.artifact_family_target_weight,
        "family_warmup_steps": config.artifact_family_warmup_steps,
        "gan_batch_size": config.adversarial_gan_batch_size,
        "discriminator_weight": config.adversarial_discriminator_weight,
        "generator_weight": config.adversarial_generator_weight,
        "feature_matching_weight": (
            config.adversarial_feature_matching_weight
        ),
        "discriminator_grad_clip_norm": (
            config.discriminator_grad_clip_norm
        ),
        "discriminator_update_every": (
            config.discriminator_update_every
        ),
    }
    encoded = json.dumps(
        to_jsonable(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_loss_config_fingerprint(
    config: TTATrainerConfig,
) -> str:
    payload = {
        "combination": config.artifact_family_combination,
        "beta": (
            config.artifact_family_beta
            if config.artifact_family_combination == "additive"
            else None
        ),
        "convex_target_weight": (
            config.artifact_family_target_weight
            if config.artifact_family_combination == "convex"
            else None
        ),
        "convex_warmup_steps": (
            config.artifact_family_warmup_steps
            if config.artifact_family_combination == "convex"
            else None
        ),
    }
    encoded = json.dumps(
        to_jsonable(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _output_scalar_metrics(output: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in output.items():
        if key == "loss":
            continue
        if isinstance(value, Tensor) and value.numel() == 1:
            result[str(key)] = float(value.detach().float().item())
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            result[str(key)] = float(value)
    for container_name in ("mask_metrics", "metrics"):
        container = output.get(container_name)
        if isinstance(container, Mapping):
            for key, value in container.items():
                if isinstance(value, Tensor) and value.numel() == 1:
                    result[str(key)] = float(value.detach().float().item())
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    result[str(key)] = float(value)
    if "mask_ratio" not in result:
        mask = output.get("target_mask", output.get("mask"))
        valid = output.get("valid_latent_mask")
        if isinstance(mask, Tensor):
            mask_bool = mask.detach().bool()
            if isinstance(valid, Tensor) and valid.shape == mask.shape:
                valid_bool = valid.detach().bool()
                result["mask_ratio"] = float(
                    (mask_bool & valid_bool).float().sum().item()
                    / max(valid_bool.float().sum().item(), 1.0)
                )
            else:
                result["mask_ratio"] = float(mask_bool.float().mean().item())
    return result


def _batch_metrics(batch: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    wav = batch.get("wav_clean")
    if isinstance(wav, Tensor) and wav.ndim >= 2:
        result["batch_size"] = int(wav.shape[0])
        result["wav_samples"] = int(wav.shape[-1])
    valid = batch.get("wav_valid_mask")
    sample_rate = batch.get("wav_sample_rate", 16_000)
    if isinstance(sample_rate, Tensor):
        sample_rate_value = float(sample_rate.detach().flatten()[0].item())
    elif isinstance(sample_rate, (list, tuple)) and sample_rate:
        sample_rate_value = float(sample_rate[0])
    else:
        sample_rate_value = float(sample_rate)
    if isinstance(valid, Tensor):
        result["audio_seconds"] = float(
            valid.detach().float().sum().item() / max(sample_rate_value, 1.0)
        )
    captions = batch.get("caption")
    if isinstance(captions, str):
        result["caption_example"] = captions[:256]
    elif isinstance(captions, (list, tuple)) and captions:
        result["caption_example"] = str(captions[0])[:256]
    datasets = batch.get("dataset")
    if isinstance(datasets, (list, tuple)):
        normalized = [str(value).strip().lower() for value in datasets]
        result["audiocaps_samples"] = normalized.count("audiocaps")
        result["wavcaps_samples"] = normalized.count("wavcaps")
    for key in ("dataset", "utt_id"):
        value = batch.get(key)
        if isinstance(value, (list, tuple)) and value:
            result[key] = str(value[0])
        elif isinstance(value, str):
            result[key] = value
    return result


def _memory_metrics(device: torch.device) -> dict[str, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}
    index = device.index if device.index is not None else torch.cuda.current_device()
    return {
        "cuda_allocated_gb": float(torch.cuda.memory_allocated(index) / (1024**3)),
        "cuda_reserved_gb": float(torch.cuda.memory_reserved(index) / (1024**3)),
        "cuda_peak_allocated_gb": float(
            torch.cuda.max_memory_allocated(index) / (1024**3)
        ),
    }


def _model_sample_rate(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    if isinstance(config, Mapping):
        return int(config.get("sample_rate", 16_000))
    return int(getattr(config, "sample_rate", 16_000))


def _validate_ema_keys_and_shapes(
    target: Mapping[str, Tensor],
    source: Mapping[str, Tensor],
    *,
    context: str,
) -> None:
    target_keys = set(target)
    source_keys = set(source)
    if target_keys != source_keys:
        missing = sorted(target_keys.difference(source_keys))
        unexpected = sorted(source_keys.difference(target_keys))
        raise RuntimeError(
            f"{context} key mismatch: missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    mismatched = [
        key for key in target if tuple(target[key].shape) != tuple(source[key].shape)
    ]
    if mismatched:
        details = ", ".join(
            f"{key}: target={tuple(target[key].shape)} source={tuple(source[key].shape)}"
            for key in mismatched[:8]
        )
        raise RuntimeError(f"{context} shape mismatch ({len(mismatched)}): {details}")


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
