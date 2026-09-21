from __future__ import annotations

import hashlib
import math
import warnings
from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from task_audio._backbone.latent_fmfm import (
    LatentFMFMMaskedPrior,
    LatentFMFMMaskedPriorConfig,
)
from task_audio._backbone.latent_tts import (
    FMHead,
    _downsample_bool_mask,
    _patch_mask_from_sample_mask,
    _repeat_steps,
    _upsample_bool_mask,
    masked_mse,
    masked_weighted_mse,
)
from task_audio._backbone.wave_fm_udit import LinearUDiTUNetWaveFMDecoder

from .audio import initialize_tts_audio_components
from .clap_conditioner import ClapTextConditioner, ClapTextConditionerConfig
from .losses import (
    HighBandDetailLoss,
    SAMEPhaseAwareMultiResolutionSTFTLoss,
    StationaryArtifactExcessLoss,
    paired_valid_audio_crop,
)
from .masking import MASK_MODE_SSL, MaskPlan, PriorMaskPolicy, build_mask_policy
from .prior import TextConditionedPriorFMEncoder, _normalize_injection
from .text_conditioner import (
    FlanT5Conditioner,
    FlanT5ConditionerConfig,
    checkpoint_model_state,
)


DEFAULT_FLAN_T5_LARGE = (
    "/nativemm2/share/cpfs/wangyujin.wyj/space_edit/hf_cache/hub/"
    "models--google--flan-t5-large"
)


@dataclass
class TTALatentAudioConfig(LatentFMFMMaskedPriorConfig):
    """TTA-specific normalization around the reusable TTS audio stack."""

    vocab_size: int = 1
    architecture: str = "latent_tta"
    fm_input_mode: str = "megatts_add"
    text_condition_mode: str = "megatts_expand"
    prediction: str = "x_pred_v_loss"
    mask_source: str = "random_span"
    use_target_token: bool = False
    split_target_encoder_ema: bool = True
    target_ema_view: str = "full_audio"
    student_condition_view: str = "masked_input"
    reconstruction_latent_source: str = "full_online"
    decoder_objective: str = "wave_fm"
    build_deterministic_decoder: bool = False
    lambda_flow: float = 1.0
    lambda_wave_fm: float = 1.0
    lambda_recon: float = 0.0
    lambda_mel: float = 0.0
    same_mrstft_enabled: bool = False
    lambda_same_mrstft: float = 0.0
    same_mrstft_crop_seconds: float = 2.0
    same_mrstft_fft_sizes: tuple[int, ...] | list[int] = (
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
    )
    same_mrstft_hop_ratio: float = 0.25
    same_mrstft_k_weighting: bool = True
    same_mrstft_eps: float = 1.0e-7
    same_mrstft_stability_eps: float = 1.0e-4
    highband_detail_enabled: bool = False
    lambda_highband_detail: float = 0.0
    highband_detail_fft_sizes: tuple[int, ...] | list[int] = (512, 1024, 2048)
    highband_detail_bands_hz: tuple[tuple[float, float], ...] | list[list[float]] = (
        (4_000.0, 6_000.0),
        (6_000.0, 8_000.0),
    )
    highband_detail_hop_ratio: float = 0.25
    highband_detail_log_magnitude_weight: float = 1.0
    highband_detail_spectral_flux_weight: float = 0.25
    highband_detail_stability_eps: float = 1.0e-4
    stationary_artifact_enabled: bool = False
    lambda_stationary_artifact: float = 0.0
    stationary_artifact_fft_sizes: tuple[int, ...] | list[int] = (1024, 2048, 4096)
    stationary_artifact_min_frequency_hz: float = 200.0
    stationary_artifact_max_frequency_hz: float = 7_800.0
    stationary_artifact_hop_ratio: float = 0.25
    stationary_artifact_local_frequency_bins: int = 31
    stationary_artifact_prominence_threshold_db: float = 6.0
    stationary_artifact_softness_db: float = 2.0
    stationary_artifact_excess_margin_db: float = 0.0
    stationary_artifact_top_frequency_fraction: float = 1.0
    stationary_artifact_stability_eps: float = 1.0e-5
    adversarial_crop_seconds: float = 2.0
    adversarial_enabled: bool = False
    adversarial_start_step: int = 10_000
    adversarial_generator_weight: float = 0.1
    adversarial_feature_matching_weight: float = 5.0
    adversarial_stft_fft_sizes: tuple[int, ...] | list[int] = (
        128,
        256,
        512,
        1024,
        2048,
    )
    adversarial_pqmf_enabled: bool = True
    adversarial_chroma_enabled: bool = False
    speaker_encoder_type: str = "none"
    wave_speaker_encoder_type: str = "none"
    prior_cfg_dropout_mode: str = "drop_text"
    text_injection: str = "mmdit"
    fm_mmdit_fused_depth: int = 0
    clap_enabled: bool = False
    clap_embedding_dim: int = 512

    def __post_init__(self) -> None:
        # These are structural TTA v1 invariants, not public compatibility
        # switches inherited from the TTS dataclass.  The split target encoder
        # remains configurable so controlled no-teacher ablations can use the
        # online full-audio latent as the stop-gradient flow target.
        self.vocab_size = 1
        self.architecture = "latent_tta"
        self.fm_input_mode = "megatts_add"
        self.text_condition_mode = "megatts_expand"
        self.mask_source = "random_span"
        self.use_target_token = False
        requested_split_target_encoder_ema = bool(self.split_target_encoder_ema)
        # The reusable speech parent historically validates the EMA split as
        # mandatory.  TTA keeps that default, but temporarily presents the
        # legacy value during parent validation so this explicit ablation can
        # restore the requested no-teacher setting afterward.
        self.split_target_encoder_ema = True
        self.speaker_encoder_type = "none"
        self.wave_speaker_encoder_type = "none"
        super().__post_init__()
        self.split_target_encoder_ema = requested_split_target_encoder_ema
        self.text_injection = _normalize_injection(self.text_injection)
        self.clap_enabled = bool(self.clap_enabled)
        self.clap_embedding_dim = int(self.clap_embedding_dim)
        if self.clap_embedding_dim < 1:
            raise ValueError("clap_embedding_dim must be positive")
        self.target_ema_view = (
            str(self.target_ema_view).strip().lower().replace("-", "_")
        )
        if self.target_ema_view not in {"target_only", "full_audio"}:
            raise ValueError("target_ema_view must be 'target_only' or 'full_audio'")
        self.student_condition_view = (
            str(self.student_condition_view).strip().lower().replace("-", "_")
        )
        if self.student_condition_view not in {
            "masked_input",
            "full_input_visible_output",
        }:
            raise ValueError(
                "student_condition_view must be 'masked_input' or "
                "'full_input_visible_output'"
            )
        self.reconstruction_latent_source = (
            str(self.reconstruction_latent_source).strip().lower().replace("-", "_")
        )
        if self.reconstruction_latent_source not in {"full_online", "mask_hybrid"}:
            raise ValueError(
                "reconstruction_latent_source must be 'full_online' or 'mask_hybrid'"
            )
        self.same_mrstft_enabled = bool(self.same_mrstft_enabled)
        self.lambda_same_mrstft = float(self.lambda_same_mrstft)
        self.same_mrstft_crop_seconds = float(self.same_mrstft_crop_seconds)
        self.same_mrstft_fft_sizes = tuple(
            int(value) for value in self.same_mrstft_fft_sizes
        )
        self.same_mrstft_hop_ratio = float(self.same_mrstft_hop_ratio)
        self.same_mrstft_k_weighting = bool(self.same_mrstft_k_weighting)
        self.same_mrstft_eps = float(self.same_mrstft_eps)
        self.same_mrstft_stability_eps = float(
            self.same_mrstft_stability_eps
        )
        self.highband_detail_enabled = bool(self.highband_detail_enabled)
        self.lambda_highband_detail = float(self.lambda_highband_detail)
        self.highband_detail_fft_sizes = tuple(
            int(value) for value in self.highband_detail_fft_sizes
        )
        self.highband_detail_bands_hz = tuple(
            (float(value[0]), float(value[1]))
            for value in self.highband_detail_bands_hz
        )
        self.highband_detail_hop_ratio = float(self.highband_detail_hop_ratio)
        self.highband_detail_log_magnitude_weight = float(
            self.highband_detail_log_magnitude_weight
        )
        self.highband_detail_spectral_flux_weight = float(
            self.highband_detail_spectral_flux_weight
        )
        self.highband_detail_stability_eps = float(
            self.highband_detail_stability_eps
        )
        self.stationary_artifact_enabled = bool(self.stationary_artifact_enabled)
        self.lambda_stationary_artifact = float(self.lambda_stationary_artifact)
        self.stationary_artifact_fft_sizes = tuple(
            int(value) for value in self.stationary_artifact_fft_sizes
        )
        self.stationary_artifact_min_frequency_hz = float(
            self.stationary_artifact_min_frequency_hz
        )
        self.stationary_artifact_max_frequency_hz = float(
            self.stationary_artifact_max_frequency_hz
        )
        self.stationary_artifact_hop_ratio = float(
            self.stationary_artifact_hop_ratio
        )
        self.stationary_artifact_local_frequency_bins = int(
            self.stationary_artifact_local_frequency_bins
        )
        self.stationary_artifact_prominence_threshold_db = float(
            self.stationary_artifact_prominence_threshold_db
        )
        self.stationary_artifact_softness_db = float(
            self.stationary_artifact_softness_db
        )
        self.stationary_artifact_excess_margin_db = float(
            self.stationary_artifact_excess_margin_db
        )
        self.stationary_artifact_top_frequency_fraction = float(
            self.stationary_artifact_top_frequency_fraction
        )
        self.stationary_artifact_stability_eps = float(
            self.stationary_artifact_stability_eps
        )
        self.adversarial_enabled = bool(self.adversarial_enabled)
        self.adversarial_crop_seconds = float(self.adversarial_crop_seconds)
        self.adversarial_start_step = int(self.adversarial_start_step)
        self.adversarial_generator_weight = float(self.adversarial_generator_weight)
        self.adversarial_feature_matching_weight = float(
            self.adversarial_feature_matching_weight
        )
        self.adversarial_stft_fft_sizes = tuple(
            int(value) for value in self.adversarial_stft_fft_sizes
        )
        self.adversarial_pqmf_enabled = bool(self.adversarial_pqmf_enabled)
        self.adversarial_chroma_enabled = bool(self.adversarial_chroma_enabled)
        if self.lambda_same_mrstft < 0.0:
            raise ValueError("lambda_same_mrstft must be non-negative")
        if self.same_mrstft_crop_seconds <= 0.0:
            raise ValueError("same_mrstft_crop_seconds must be positive")
        if not self.same_mrstft_fft_sizes or any(
            value < 4 or value % 2 for value in self.same_mrstft_fft_sizes
        ):
            raise ValueError(
                "same_mrstft_fft_sizes must contain positive even integers >= 4"
            )
        if not 0.0 < self.same_mrstft_hop_ratio <= 1.0:
            raise ValueError("same_mrstft_hop_ratio must be in (0, 1]")
        if self.same_mrstft_eps <= 0.0:
            raise ValueError("same_mrstft_eps must be positive")
        if self.same_mrstft_stability_eps <= 0.0:
            raise ValueError("same_mrstft_stability_eps must be positive")
        if self.lambda_highband_detail < 0.0:
            raise ValueError("lambda_highband_detail must be non-negative")
        if not self.highband_detail_fft_sizes or any(
            value < 4 or value % 2 for value in self.highband_detail_fft_sizes
        ):
            raise ValueError(
                "highband_detail_fft_sizes must contain positive even integers >= 4"
            )
        if not self.highband_detail_bands_hz:
            raise ValueError("highband_detail_bands_hz cannot be empty")
        nyquist = float(self.sample_rate) / 2.0
        if any(
            low < 0.0 or high <= low or high > nyquist
            for low, high in self.highband_detail_bands_hz
        ):
            raise ValueError(
                "highband_detail_bands_hz must be ordered intervals within Nyquist"
            )
        if not 0.0 < self.highband_detail_hop_ratio <= 1.0:
            raise ValueError("highband_detail_hop_ratio must be in (0, 1]")
        if min(
            self.highband_detail_log_magnitude_weight,
            self.highband_detail_spectral_flux_weight,
        ) < 0.0:
            raise ValueError("high-band component weights must be non-negative")
        if self.highband_detail_stability_eps <= 0.0:
            raise ValueError("highband_detail_stability_eps must be positive")
        if self.lambda_stationary_artifact < 0.0:
            raise ValueError("lambda_stationary_artifact must be non-negative")
        if not self.stationary_artifact_fft_sizes or any(
            value < 4 or value % 2 for value in self.stationary_artifact_fft_sizes
        ):
            raise ValueError(
                "stationary_artifact_fft_sizes must contain positive even integers >= 4"
            )
        if not (
            0.0
            <= self.stationary_artifact_min_frequency_hz
            < self.stationary_artifact_max_frequency_hz
            <= nyquist
        ):
            raise ValueError(
                "stationary artifact frequency range must be ordered within Nyquist"
            )
        if not 0.0 < self.stationary_artifact_hop_ratio <= 1.0:
            raise ValueError("stationary_artifact_hop_ratio must be in (0, 1]")
        if (
            self.stationary_artifact_local_frequency_bins < 3
            or self.stationary_artifact_local_frequency_bins % 2 == 0
        ):
            raise ValueError(
                "stationary_artifact_local_frequency_bins must be an odd integer >= 3"
            )
        if self.stationary_artifact_softness_db <= 0.0:
            raise ValueError("stationary_artifact_softness_db must be positive")
        if self.stationary_artifact_excess_margin_db < 0.0:
            raise ValueError("stationary_artifact_excess_margin_db must be non-negative")
        if not 0.0 < self.stationary_artifact_top_frequency_fraction <= 1.0:
            raise ValueError(
                "stationary_artifact_top_frequency_fraction must be in (0, 1]"
            )
        if self.stationary_artifact_stability_eps <= 0.0:
            raise ValueError("stationary_artifact_stability_eps must be positive")
        if self.adversarial_start_step < 0:
            raise ValueError("adversarial_start_step must be non-negative")
        if self.adversarial_crop_seconds <= 0.0:
            raise ValueError("adversarial_crop_seconds must be positive")
        if (
            min(
                self.adversarial_generator_weight,
                self.adversarial_feature_matching_weight,
            )
            < 0.0
        ):
            raise ValueError("adversarial loss weights must be non-negative")
        if not self.adversarial_stft_fft_sizes or any(
            value < 4 or value % 2 for value in self.adversarial_stft_fft_sizes
        ):
            raise ValueError(
                "adversarial_stft_fft_sizes must contain positive even integers >= 4"
            )
        if self.same_mrstft_enabled and self.decoder_objective != "wave":
            raise ValueError("SAME reconstruction requires the deterministic decoder")
        if self.same_mrstft_enabled and self.decoder_recon_latent_mode != "clean":
            raise ValueError(
                "SAME reconstruction/GAN requires clean online encoder latents; "
                "z_pred reconstruction is not supported"
            )
        if self.highband_detail_enabled and not self.same_mrstft_enabled:
            raise ValueError("high-band detail loss requires SAME reconstruction")
        if self.stationary_artifact_enabled and not self.same_mrstft_enabled:
            raise ValueError("stationary artifact loss requires SAME reconstruction")
        if self.adversarial_enabled and not self.same_mrstft_enabled:
            raise ValueError("adversarial training requires SAME reconstruction")
        if self.adversarial_enabled and self.decoder_objective != "wave":
            raise ValueError("adversarial training requires the deterministic decoder")
        if self.adversarial_chroma_enabled:
            raise ValueError("TTA v1 does not enable the chroma discriminator")
        if self.text_injection == "mmdit":
            if int(self.fm_depth) < 2:
                raise ValueError("MMDiT prior requires fm_depth >= 2")
            if int(self.fm_mmdit_fused_depth) <= 0:
                self.fm_mmdit_fused_depth = max(1, round(int(self.fm_depth) * 2 / 3))
            if not 1 <= int(self.fm_mmdit_fused_depth) < int(self.fm_depth):
                raise ValueError(
                    "fm_mmdit_fused_depth must satisfy 1 <= fused depth < fm_depth"
                )


class TTALatentAudio(LatentFMFMMaskedPrior):
    """Caption-to-audio model reusing the TTS audio encoder and decoders.

    A lightweight façade constructs only the audio modules. Inherited methods
    provide the proven split/EMA encoding, wave losses, and normalization
    utilities without ever constructing the TTS text/alignment stack.
    """

    def __init__(
        self,
        config: TTALatentAudioConfig,
        *,
        text_conditioner: nn.Module,
        clap_conditioner: nn.Module | None = None,
        mask_policy: PriorMaskPolicy | None = None,
    ) -> None:
        # Deliberately bypass the TTS model constructor: it would instantiate
        # token text/alignment modules that do not belong in TTA. The façade
        # attaches only the reusable audio modules with the same state keys.
        nn.Module.__init__(self)
        self.config = config
        initialize_tts_audio_components(self, config)
        self.config: TTALatentAudioConfig
        self.text_conditioner = text_conditioner
        if config.clap_enabled:
            if clap_conditioner is None:
                raise ValueError("clap_conditioner is required when clap_enabled=True")
            clap_dim = int(
                getattr(clap_conditioner, "output_dim", config.clap_embedding_dim)
            )
            if clap_dim != int(config.clap_embedding_dim):
                raise ValueError(
                    "CLAP conditioner output_dim must match clap_embedding_dim "
                    f"({clap_dim} != {config.clap_embedding_dim})"
                )
            self.clap_conditioner = clap_conditioner
        elif clap_conditioner is not None:
            raise ValueError(
                "clap_conditioner was supplied while CLAP conditioning is disabled"
            )
        text_dim = int(getattr(text_conditioner, "output_dim", config.text_dim))
        self.fm_encoder = TextConditionedPriorFMEncoder(
            token_dim=config.latent_dim,
            text_dim=text_dim,
            hidden_dim=config.fm_hidden_dim,
            depth=config.fm_depth,
            adaln_every=config.fm_adaln_every,
            heads=config.fm_heads,
            dim_head=config.fm_dim_head,
            ffn_mult=config.fm_ffn_mult,
            dropout=config.dropout,
            rope_base=config.rope_base,
            injection=config.text_injection,
            input_fusion=config.prior_fm_input_fusion,
            mmdit_fused_depth=(
                config.fm_mmdit_fused_depth
                if config.text_injection == "mmdit"
                else None
            ),
            global_condition_dim=(
                config.clap_embedding_dim if config.clap_enabled else None
            ),
        )
        self.fm_head = FMHead(config.fm_hidden_dim, config.latent_dim)
        self.mask_policy = mask_policy or build_mask_policy(
            {
                "policy": "random_span",
                "min_ratio": config.mask_ratio_min,
                "max_ratio": config.mask_ratio_max,
            }
        )
        self._mask_progress: float | None = None
        self._last_mask_plan: MaskPlan | None = None
        self._last_z_full_online: Tensor | None = None
        self._last_z_full_teacher: Tensor | None = None
        self.reconstruction_loss_fn = (
            SAMEPhaseAwareMultiResolutionSTFTLoss(
                sample_rate=int(config.sample_rate),
                fft_sizes=config.same_mrstft_fft_sizes,
                hop_ratio=float(config.same_mrstft_hop_ratio),
                k_weighting=bool(config.same_mrstft_k_weighting),
                eps=float(config.same_mrstft_eps),
                stability_eps=float(config.same_mrstft_stability_eps),
            )
            if config.same_mrstft_enabled and config.lambda_same_mrstft > 0.0
            else None
        )
        self.highband_detail_loss_fn = (
            HighBandDetailLoss(
                sample_rate=int(config.sample_rate),
                fft_sizes=config.highband_detail_fft_sizes,
                bands_hz=config.highband_detail_bands_hz,
                hop_ratio=float(config.highband_detail_hop_ratio),
                log_magnitude_weight=float(
                    config.highband_detail_log_magnitude_weight
                ),
                spectral_flux_weight=float(
                    config.highband_detail_spectral_flux_weight
                ),
                stability_eps=float(config.highband_detail_stability_eps),
            )
            if config.highband_detail_enabled
            and config.lambda_highband_detail > 0.0
            else None
        )
        self.stationary_artifact_loss_fn = (
            StationaryArtifactExcessLoss(
                sample_rate=int(config.sample_rate),
                fft_sizes=config.stationary_artifact_fft_sizes,
                min_frequency_hz=float(config.stationary_artifact_min_frequency_hz),
                max_frequency_hz=float(config.stationary_artifact_max_frequency_hz),
                hop_ratio=float(config.stationary_artifact_hop_ratio),
                local_frequency_bins=int(
                    config.stationary_artifact_local_frequency_bins
                ),
                prominence_threshold_db=float(
                    config.stationary_artifact_prominence_threshold_db
                ),
                softness_db=float(config.stationary_artifact_softness_db),
                excess_margin_db=float(
                    config.stationary_artifact_excess_margin_db
                ),
                top_frequency_fraction=float(
                    config.stationary_artifact_top_frequency_fraction
                ),
                stability_eps=float(config.stationary_artifact_stability_eps),
            )
            if config.stationary_artifact_enabled
            and config.lambda_stationary_artifact > 0.0
            else None
        )

    def forward(
        self,
        batch: dict[str, Any],
        progress: float | None = None,
    ) -> dict[str, Tensor]:
        self._mask_progress = progress
        self._last_mask_plan = None
        self._last_z_full_online = None
        self._last_z_full_teacher = None
        try:
            output = super().forward(batch)
        finally:
            self._mask_progress = None
        plan = self._last_mask_plan
        if self._last_z_full_online is not None:
            output["z_full_online"] = self._last_z_full_online
        if self._last_z_full_teacher is not None:
            output["z_full_teacher"] = self._last_z_full_teacher
        output["student_condition_full_attention"] = output["loss"].new_tensor(
            float(self.config.student_condition_view == "full_input_visible_output")
        )
        if plan is not None:
            anchor = output["loss"]
            for name, value in plan.metrics.items():
                metric = (
                    value
                    if isinstance(value, Tensor)
                    else anchor.new_tensor(float(value))
                )
                output[f"mask_{name}"] = metric.to(device=anchor.device)
        self._add_target_ema_diagnostics(output, plan=plan)
        if self.reconstruction_loss_fn is not None:
            teacher_waveform = output.get("teacher_waveform")
            if (
                not isinstance(teacher_waveform, Tensor)
                or teacher_waveform.numel() == 0
            ):
                raise RuntimeError(
                    "SAME reconstruction is enabled but deterministic decoding returned no waveform"
                )
            full_valid_mask = batch.get("wav_valid_mask")
            same_crop = paired_valid_audio_crop(
                batch["wav_clean"].float(),
                teacher_waveform,
                full_valid_mask,
                crop_samples=max(
                    1,
                    int(
                        round(
                            float(self.config.same_mrstft_crop_seconds)
                            * int(self.config.sample_rate)
                        )
                    ),
                ),
            )
            reconstruction = self.reconstruction_loss_fn(
                same_crop.fake,
                same_crop.real,
            )
            reconstruction_loss = reconstruction["loss"]
            weighted_reconstruction = reconstruction_loss * float(
                self.config.lambda_same_mrstft
            )
            output["loss"] = output["loss"] + weighted_reconstruction
            output["same_mrstft_loss"] = reconstruction_loss
            output["weighted_same_mrstft_loss"] = weighted_reconstruction
            for name in (
                "spectral_contrast",
                "adaptive_log_magnitude",
                "instantaneous_frequency",
                "group_delay",
                "complex_distance",
            ):
                output[f"same_mrstft_{name}"] = reconstruction[name]
            if self.highband_detail_loss_fn is not None:
                highband = self.highband_detail_loss_fn(
                    same_crop.fake,
                    same_crop.real,
                )
                weighted_highband = highband["loss"] * float(
                    self.config.lambda_highband_detail
                )
                output["loss"] = output["loss"] + weighted_highband
                output["highband_detail_loss"] = highband["loss"]
                output["weighted_highband_detail_loss"] = weighted_highband
                output["highband_detail_log_magnitude"] = highband[
                    "log_magnitude"
                ]
                output["highband_detail_spectral_flux"] = highband[
                    "spectral_flux"
                ]
            if self.stationary_artifact_loss_fn is not None:
                stationary_artifact = self.stationary_artifact_loss_fn(
                    same_crop.fake,
                    same_crop.real,
                )
                weighted_stationary_artifact = stationary_artifact["loss"] * float(
                    self.config.lambda_stationary_artifact
                )
                output["loss"] = output["loss"] + weighted_stationary_artifact
                output["stationary_artifact_loss"] = stationary_artifact["loss"]
                output["weighted_stationary_artifact_loss"] = (
                    weighted_stationary_artifact
                )
                output["stationary_artifact_fake_score_db"] = stationary_artifact[
                    "fake_stationary_score_db"
                ]
                output["stationary_artifact_real_score_db"] = stationary_artifact[
                    "real_stationary_score_db"
                ]
            output["reconstruction_full_real"] = batch["wav_clean"].float().unsqueeze(1)
            output["reconstruction_full_fake"] = teacher_waveform
            output["reconstruction_full_valid_mask"] = (
                full_valid_mask.unsqueeze(1)
                if full_valid_mask is not None and full_valid_mask.ndim == 2
                else full_valid_mask
            )
            output["reconstruction_same_real_crop"] = same_crop.real
            output["reconstruction_same_fake_crop"] = same_crop.fake
            output["reconstruction_same_crop_valid_mask"] = same_crop.valid_mask
            output["reconstruction_same_crop_starts"] = same_crop.starts
            output["reconstruction_same_crop_valid_fraction"] = (
                same_crop.valid_mask.detach().float().mean()
            )

            crop = paired_valid_audio_crop(
                batch["wav_clean"].float(),
                teacher_waveform,
                full_valid_mask,
                crop_samples=max(
                    1,
                    int(
                        round(
                            float(self.config.adversarial_crop_seconds)
                            * int(self.config.sample_rate)
                        )
                    ),
                ),
            )
            output["reconstruction_real_crop"] = crop.real
            output["reconstruction_fake_crop"] = crop.fake
            output["reconstruction_crop_valid_mask"] = crop.valid_mask
            output["reconstruction_crop_starts"] = crop.starts
            output["reconstruction_crop_valid_fraction"] = (
                crop.valid_mask.detach().float().mean()
            )
        return output

    @torch.no_grad()
    def _add_target_ema_diagnostics(
        self,
        output: dict[str, Tensor],
        *,
        plan: MaskPlan | None,
    ) -> None:
        """Measure whether the frozen EMA teacher tracks the online encoder.

        Cosine and RMS deltas are restricted to prior target tokens. Visible
        tokens are deliberately excluded because ``z_flow_target`` equals the
        online condition there and would produce a trivial cosine of one.
        """

        anchor = output["loss"].detach()
        zero = anchor.new_zeros((), dtype=torch.float32)
        output["target_ema_enabled"] = anchor.new_tensor(
            float(self.config.split_target_encoder_ema), dtype=torch.float32
        )
        output["target_ema_decay"] = anchor.new_tensor(
            float(self.config.split_target_encoder_ema_decay), dtype=torch.float32
        )
        metric_names = (
            "target_ema_latent_cosine",
            "target_ema_ssl_cosine",
            "target_ema_fullgen_cosine",
            "target_ema_latent_delta_rms",
            "target_ema_ssl_delta_rms",
            "target_ema_fullgen_delta_rms",
            "target_ema_online_rms",
            "target_ema_teacher_rms",
            "target_ema_norm_ratio",
            "target_ema_param_cosine",
            "target_ema_param_delta_rms",
        )
        for name in metric_names:
            output[name] = zero

        online = output.get("z_clean")
        teacher = output.get("z_flow_target")
        target = output.get("mask")
        valid = output.get("valid_latent_mask")
        if not all(
            isinstance(value, Tensor) for value in (online, teacher, target, valid)
        ):
            return
        if online.shape != teacher.shape or target.shape != online.shape[:2]:
            return
        target_mask = target.detach().bool() & valid.detach().bool()
        full_rows = torch.zeros(online.shape[0], device=online.device, dtype=torch.bool)
        if plan is not None and plan.full_generation_mask is not None:
            full_rows = plan.full_generation_mask.to(
                device=online.device, dtype=torch.bool
            )
        ssl_mask = target_mask & ~full_rows[:, None]
        fullgen_mask = target_mask & full_rows[:, None]

        online_f = online.detach().float()
        teacher_f = teacher.detach().float()
        cosine = F.cosine_similarity(online_f, teacher_f, dim=-1, eps=1.0e-8)
        delta_mse = (online_f - teacher_f).square().mean(dim=-1)
        online_power = online_f.square().mean(dim=-1)
        teacher_power = teacher_f.square().mean(dim=-1)

        def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
            weights = mask.to(device=values.device, dtype=values.dtype)
            return (values * weights).sum() / weights.sum().clamp_min(1.0)

        def masked_rms(values: Tensor, mask: Tensor) -> Tensor:
            return masked_mean(values, mask).clamp_min(0.0).sqrt()

        latent_cosine = masked_mean(cosine, target_mask)
        ssl_cosine = masked_mean(cosine, ssl_mask)
        fullgen_cosine = masked_mean(cosine, fullgen_mask)
        latent_delta = masked_rms(delta_mse, target_mask)
        ssl_delta = masked_rms(delta_mse, ssl_mask)
        fullgen_delta = masked_rms(delta_mse, fullgen_mask)
        online_rms = masked_rms(online_power, target_mask)
        teacher_rms = masked_rms(teacher_power, target_mask)
        output.update(
            {
                "target_ema_latent_cosine": latent_cosine,
                "target_ema_ssl_cosine": ssl_cosine,
                "target_ema_fullgen_cosine": fullgen_cosine,
                "target_ema_latent_delta_rms": latent_delta,
                "target_ema_ssl_delta_rms": ssl_delta,
                "target_ema_fullgen_delta_rms": fullgen_delta,
                "target_ema_online_rms": online_rms,
                "target_ema_teacher_rms": teacher_rms,
                "target_ema_norm_ratio": teacher_rms / online_rms.clamp_min(1.0e-8),
            }
        )

        online_projection = getattr(self, "encoder_to_latent", None)
        ema_projection = getattr(self, "encoder_to_latent_ema", None)
        if not isinstance(online_projection, nn.Module) or not isinstance(
            ema_projection, nn.Module
        ):
            return
        online_parameter = next(online_projection.parameters(), None)
        ema_parameter = next(ema_projection.parameters(), None)
        if online_parameter is None or ema_parameter is None:
            return
        online_flat = online_parameter.detach().float().flatten()
        ema_flat = ema_parameter.detach().float().flatten()
        output["target_ema_param_cosine"] = F.cosine_similarity(
            online_flat.unsqueeze(0),
            ema_flat.unsqueeze(0),
            dim=-1,
            eps=1.0e-12,
        ).squeeze(0)
        output["target_ema_param_delta_rms"] = (
            (online_flat - ema_flat).square().mean().sqrt()
        )

    @torch.no_grad()
    def visible_condition_dependence(
        self,
        batch: dict[str, Any],
        *,
        seed: int,
    ) -> dict[str, float | int | str]:
        """Measure prior reliance on the real visible audio condition.

        One policy-specific SSL mask template is repeated across the batch so
        cyclic row permutation replaces visible latents with another clip's
        masked-view latents at exactly the same positions. Target EMA latents,
        caption, flow noise, and flow time remain identical across ablations.
        """

        wav_clean_raw = batch["wav_clean"].float()
        if wav_clean_raw.ndim != 2 or wav_clean_raw.shape[0] < 2:
            raise ValueError(
                "visible-condition evaluation requires wav_clean [B,T] with B >= 2"
            )
        wav_clean = self._scale_waveform(wav_clean_raw)
        patches, _ = self.patchify.patchify(wav_clean)
        valid_patch_mask = _patch_mask_from_sample_mask(
            batch.get("wav_valid_mask"),
            patch_size=int(self.config.patch_size),
            num_patches=int(patches.shape[1]),
            fallback_shape=wav_clean_raw.shape,
        ).to(device=patches.device, dtype=torch.bool)
        valid_seed = _downsample_bool_mask(
            valid_patch_mask,
            factor=int(self.config.downsample_factor),
        )
        if not bool((valid_seed == valid_seed[:1]).all().item()):
            raise ValueError(
                "visible-condition evaluation requires equal valid lengths"
            )

        mask_generator = torch.Generator(device=patches.device).manual_seed(int(seed))
        template_plan: MaskPlan | None = None
        for _ in range(32):
            candidate = self.mask_policy.sample(
                valid_seed[:1],
                batch=None,
                progress=None,
                generator=mask_generator,
            )
            is_full = candidate.full_generation_mask is not None and bool(
                candidate.full_generation_mask[0].item()
            )
            if not is_full and bool((valid_seed[:1] & ~candidate.target_mask).any()):
                template_plan = candidate
                break
        if template_plan is None:
            raise RuntimeError("could not sample an SSL diagnostic mask")

        batch_size = int(patches.shape[0])
        target_mask = template_plan.target_mask.expand(batch_size, -1).clone()
        repeated_plan = MaskPlan(
            target_mask=target_mask,
            policy_name=template_plan.policy_name,
            encoding_mode=template_plan.encoding_mode,
            full_generation_mask=torch.zeros(
                batch_size, device=patches.device, dtype=torch.bool
            ),
            mode_ids=torch.full(
                (batch_size,), MASK_MODE_SSL, device=patches.device, dtype=torch.long
            ),
            metrics={},
        )
        previous_plan = self._last_mask_plan
        previous_full = self._last_z_full_online
        try:
            self._last_mask_plan = repeated_plan
            (
                z_clean,
                target_mask,
                valid_latent_mask,
                z_flow_target,
                _,
            ) = self.encode_for_mask(
                patches,
                valid_patch_mask=valid_patch_mask,
                valid_latent_mask=valid_seed,
                seed_mask=target_mask,
            )
            text_tokens, text_lengths = self._condition_text_inputs(batch)
            steps = int(self.config.flow_steps_per_recon)
            repeated_shape = (steps * batch_size, *z_flow_target.shape[1:])
            noise_generator = torch.Generator(device=z_clean.device).manual_seed(
                int(seed) + 10_000
            )
            flow_noise = torch.randn(
                repeated_shape,
                device=z_clean.device,
                dtype=z_clean.dtype,
                generator=noise_generator,
            )
            cuda_devices = []
            if z_clean.device.type == "cuda":
                cuda_devices = [
                    z_clean.device.index
                    if z_clean.device.index is not None
                    else torch.cuda.current_device()
                ]
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(int(seed) + 20_000)
                if cuda_devices:
                    torch.cuda.manual_seed_all(int(seed) + 20_000)
                flow_time = self._sample_flow_time(
                    steps * batch_size,
                    device=z_clean.device,
                    dtype=z_clean.dtype,
                )

            visible = valid_latent_mask & ~target_mask
            zero_condition = torch.where(
                visible.unsqueeze(-1), torch.zeros_like(z_clean), z_clean
            )
            permuted = z_clean.roll(shifts=1, dims=0)
            permuted_condition = torch.where(visible.unsqueeze(-1), permuted, z_clean)
            dropout = torch.zeros(batch_size, device=z_clean.device, dtype=torch.bool)

            def evaluate(condition: Tensor) -> tuple[Tensor, Tensor]:
                result = self._masked_prior_flow_loss(
                    batch,
                    z_clean,
                    z_flow_target,
                    text_tokens,
                    text_lengths=text_lengths,
                    mask=target_mask,
                    valid_latent_mask=valid_latent_mask,
                    condition_dropout_mask=dropout,
                    flow_noise=flow_noise,
                    flow_time_override=flow_time,
                    visible_condition_override=condition,
                )
                return result[0].float(), result[5].float()

            original_loss, original_pred = evaluate(z_clean)
            zero_loss, zero_pred = evaluate(zero_condition)
            permuted_loss, permuted_pred = evaluate(permuted_condition)
            target_flow = _repeat_steps(target_mask & valid_latent_mask, steps)

            def prediction_delta(candidate: Tensor) -> float:
                values = (candidate - original_pred).square().mean(dim=-1)
                weights = target_flow.to(dtype=values.dtype)
                mse = (values * weights).sum() / weights.sum().clamp_min(1.0)
                return float(mse.clamp_min(0.0).sqrt().item())

            original_value = float(original_loss.item())
            zero_value = float(zero_loss.item())
            permuted_value = float(permuted_loss.item())
            mask_bytes = (
                target_mask[0].detach().to("cpu", torch.uint8).numpy().tobytes()
            )
            return {
                "seed": int(seed),
                "policy": str(repeated_plan.policy_name),
                "batch_size": batch_size,
                "visible_tokens": int(visible[0].sum().item()),
                "target_tokens": int(
                    (target_mask[0] & valid_latent_mask[0]).sum().item()
                ),
                "mask_fingerprint": hashlib.sha256(mask_bytes).hexdigest(),
                "fm_original": original_value,
                "fm_zero_visible": zero_value,
                "fm_permuted_visible": permuted_value,
                "fm_zero_delta": zero_value - original_value,
                "fm_permuted_delta": permuted_value - original_value,
                "fm_zero_ratio": zero_value / max(original_value, 1.0e-12),
                "fm_permuted_ratio": permuted_value / max(original_value, 1.0e-12),
                "prediction_zero_delta_rms": prediction_delta(zero_pred),
                "prediction_permuted_delta_rms": prediction_delta(permuted_pred),
            }
        finally:
            self._last_mask_plan = previous_plan
            self._last_z_full_online = previous_full

    def _teacher_reconstruction_losses(
        self,
        z_clean: Tensor,
        z_pred: Tensor,
        wav_clean: Tensor,
        wav_clean_raw: Tensor,
        valid_sample_mask: Tensor,
        valid_latent_mask: Tensor | None,
        latent_target_mask: Tensor,
        *,
        original_len: int,
        z_recon_gt: Tensor | None = None,
        z_fm_input: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Decode once even when legacy waveform/mel training weights are zero."""

        if not self.config.same_mrstft_enabled:
            return super()._teacher_reconstruction_losses(
                z_clean,
                z_pred,
                wav_clean,
                wav_clean_raw,
                valid_sample_mask,
                valid_latent_mask,
                latent_target_mask,
                original_len=original_len,
                z_recon_gt=z_recon_gt,
                z_fm_input=z_fm_input,
            )
        zero = wav_clean.new_tensor(0.0)
        z_source = self._decoder_recon_latent_source(
            z_clean,
            z_pred,
            latent_target_mask,
            valid_latent_mask=valid_latent_mask,
            z_recon_gt=z_recon_gt,
            z_fm_input=z_fm_input,
        )
        z_decode, decoder_noise_rms, decoder_noise_applied_fraction = (
            self._apply_decoder_latent_noise(z_source, valid_latent_mask)
        )
        teacher_waveform_model = self.decode_tokens(z_decode, original_len=original_len)
        teacher_waveform = self._unscale_waveform(teacher_waveform_model)
        recon_loss = (
            self._wave_loss(teacher_waveform_model, wav_clean, valid_sample_mask)
            if self.config.lambda_recon != 0.0
            else zero
        )
        mel_loss = (
            self.mel_loss_fn(
                teacher_waveform_model, wav_clean, sample_mask=valid_sample_mask
            )
            if self.config.lambda_mel != 0.0
            else zero
        )
        with torch.no_grad():
            recon_loss_raw_metric = (
                self._wave_loss(
                    teacher_waveform.detach(), wav_clean_raw, valid_sample_mask
                )
                if self.config.lambda_recon != 0.0
                else zero
            )
            mel_loss_raw_metric = mel_loss.detach()
        return (
            teacher_waveform,
            teacher_waveform_model,
            recon_loss,
            mel_loss,
            recon_loss_raw_metric.detach(),
            mel_loss_raw_metric.detach(),
            decoder_noise_rms.detach(),
            decoder_noise_applied_fraction.detach(),
            z_decode.detach(),
        )

    def _decoder_recon_latent_source(
        self,
        z_clean: Tensor,
        z_pred: Tensor,
        latent_target_mask: Tensor,
        *,
        valid_latent_mask: Tensor | None,
        z_recon_gt: Tensor | None,
        z_fm_input: Tensor | None,
    ) -> Tensor:
        """Select the deterministic reconstruction view without another encode."""

        if self.config.reconstruction_latent_source == "full_online":
            z_full_online = self._last_z_full_online
            if z_full_online is None:
                raise RuntimeError(
                    "full online reconstruction latent was not produced by encode_for_mask"
                )
            if z_full_online.shape != z_clean.shape:
                raise RuntimeError(
                    "full online reconstruction latent must match the masked latent shape"
                )
            return z_full_online
        return super()._decoder_recon_latent_source(
            z_clean,
            z_pred,
            latent_target_mask,
            valid_latent_mask=valid_latent_mask,
            z_recon_gt=z_recon_gt,
            z_fm_input=z_fm_input,
        )

    @torch.no_grad()
    def encode_audio_features(
        self,
        wav: Tensor,
        *,
        wav_valid_mask: Tensor | None = None,
        encoder: str = "target_ema",
        pool: str = "mean_time",
    ) -> dict[str, Tensor]:
        """Extract frozen full-audio features for downstream evaluation.

        ``target_ema`` uses the complete internal EMA encoder stack. ``online``
        uses the corresponding trainable stack. Neither path applies an SSL
        mask, text conditioning, or decoder computation.
        """

        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.ndim != 2:
            raise ValueError("wav must have shape [B, T]")
        if wav_valid_mask is not None and wav_valid_mask.shape != wav.shape:
            raise ValueError("wav_valid_mask must match wav shape [B, T]")

        wav_model = self._scale_waveform(wav.float())
        patches, _ = self.patchify.patchify(wav_model)
        valid_patch_mask = _patch_mask_from_sample_mask(
            wav_valid_mask,
            patch_size=int(self.config.patch_size),
            num_patches=int(patches.shape[1]),
            fallback_shape=wav.shape,
        ).to(device=patches.device, dtype=torch.bool)

        encoder_name = str(encoder).lower().strip().replace("-", "_")
        if encoder_name in {
            "ema",
            "target",
            "target_ema",
            "target_encoder_ema",
        }:
            use_target_ema = True
            encoder_name = "target_ema"
        elif encoder_name in {"online", "student", "target_online"}:
            use_target_ema = False
            encoder_name = "online"
        else:
            raise ValueError("encoder must be 'target_ema' or 'online'")

        latent, valid_latent_mask = self._encode_patch_segments_batch(
            patches,
            valid_patch_mask,
            use_target_ema=use_target_ema,
        )
        pool_name = str(pool).lower().strip().replace("-", "_")
        if pool_name not in {"mean_time", "mean_std_time"}:
            raise ValueError("pool must be 'mean_time' or 'mean_std_time'")
        weights = valid_latent_mask.to(device=latent.device, dtype=latent.dtype)
        feature = (latent * weights.unsqueeze(-1)).sum(dim=1)
        feature = feature / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        if pool_name == "mean_std_time":
            centered = latent - feature.unsqueeze(1)
            variance = (centered.square() * weights.unsqueeze(-1)).sum(dim=1)
            variance = variance / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            feature = torch.cat((feature, variance.clamp_min(0.0).sqrt()), dim=-1)
        return {
            "feature": feature,
            "latent": latent,
            "valid_latent_mask": valid_latent_mask,
        }

    def _latent_mask(
        self,
        batch: dict[str, Any],
        valid_patch_mask: Tensor,
        valid_latent_mask: Tensor,
    ) -> Tensor:
        del valid_patch_mask
        plan = self.mask_policy.sample(
            valid_latent_mask,
            batch=batch,
            progress=self._mask_progress,
        )
        self._last_mask_plan = plan
        return plan.target_mask

    def _encode_split_latents(
        self,
        patches: Tensor,
        *,
        valid_patch_mask: Tensor,
        valid_latent_mask: Tensor,
        seed_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.encode_for_mask(
            patches,
            valid_patch_mask=valid_patch_mask,
            valid_latent_mask=valid_latent_mask,
            seed_mask=seed_mask,
        )

    def encode_for_mask(
        self,
        patches: Tensor,
        *,
        valid_patch_mask: Tensor,
        valid_latent_mask: Tensor,
        seed_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Mask-dependent audio encoding hook.

        Both contiguous and arbitrary masks use masked/full online views in
        one vectorized call. The EMA view stays on the original timeline and
        excludes visible patches unless legacy ``full_audio`` is selected.
        """

        plan = self._last_mask_plan
        encoding_mode = plan.encoding_mode if plan is not None else "contiguous_span"
        if encoding_mode == "contiguous_span":
            self._validate_contiguous_mask(seed_mask, valid_latent_mask)
            return self._encode_masked_full_latents(
                patches,
                valid_patch_mask=valid_patch_mask,
                valid_latent_mask=valid_latent_mask,
                seed_mask=seed_mask,
            )
        if encoding_mode == "arbitrary_mask":
            return self._encode_masked_full_latents(
                patches,
                valid_patch_mask=valid_patch_mask,
                valid_latent_mask=valid_latent_mask,
                seed_mask=seed_mask,
            )
        raise RuntimeError(f"unsupported mask encoding mode: {encoding_mode}")

    def _encode_masked_full_latents(
        self,
        patches: Tensor,
        *,
        valid_patch_mask: Tensor,
        valid_latent_mask: Tensor,
        seed_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        factor = int(self.config.downsample_factor)
        num_patches = int(patches.shape[1])
        max_latents = int(valid_latent_mask.shape[1])
        valid_seed = valid_latent_mask.to(device=patches.device, dtype=torch.bool)
        target_region = seed_mask.to(device=patches.device, dtype=torch.bool)
        if target_region.shape != valid_seed.shape or bool(
            (target_region & ~valid_seed).any().item()
        ):
            raise ValueError("target mask must match and stay inside valid_latent_mask")

        target_patch_mask = (
            _upsample_bool_mask(target_region, factor=factor, length=num_patches)
            & valid_patch_mask
        )
        visible_rows = ((~target_region) & valid_seed).any(dim=1)
        visible_indices = visible_rows.nonzero(as_tuple=False).flatten()
        z_condition: Tensor | None = None
        condition_mask = torch.zeros_like(valid_seed)
        full_input_condition = (
            self.config.student_condition_view == "full_input_visible_output"
        )
        condition_patches = patches.new_zeros((0, num_patches, patches.shape[-1]))
        condition_patch_mask = valid_patch_mask.new_zeros((0, num_patches))
        if not full_input_condition and visible_indices.numel() > 0:
            condition_patches = patches.index_select(0, visible_indices).masked_fill(
                target_patch_mask.index_select(0, visible_indices).unsqueeze(-1),
                0.0,
            )
            condition_patch_mask = valid_patch_mask.index_select(0, visible_indices)

        online_segments = torch.cat((condition_patches, patches), dim=0)
        online_patch_mask = torch.cat((condition_patch_mask, valid_patch_mask), dim=0)
        z_online, online_mask = self._encode_patch_segments_batch(
            online_segments, online_patch_mask
        )
        condition_count = 0 if full_input_condition else int(visible_indices.numel())
        if condition_count > 0:
            z_condition_subset = z_online[:condition_count]
            condition_mask_subset = online_mask[:condition_count]
            z_condition_subset = z_condition_subset[:, :max_latents]
            condition_mask_subset = condition_mask_subset[:, :max_latents]
            z_condition = z_condition_subset.new_zeros(
                patches.shape[0],
                max_latents,
                self.config.latent_dim,
            ).index_copy(0, visible_indices, z_condition_subset)
            condition_mask.index_copy_(0, visible_indices, condition_mask_subset)

        z_full_online = z_online[condition_count:]
        full_online_mask = online_mask[condition_count:]
        z_full_teacher = z_full_online
        full_teacher_mask = full_online_mask
        if self.config.split_target_encoder_ema:
            if self.config.target_ema_view == "target_only":
                teacher_patch_mask = target_patch_mask
                teacher_patches = patches.masked_fill(
                    ~teacher_patch_mask.unsqueeze(-1), 0.0
                )
            else:
                teacher_patch_mask = valid_patch_mask
                teacher_patches = patches
            with torch.no_grad():
                z_full_teacher, full_teacher_mask = self._encode_patch_segments_batch(
                    teacher_patches,
                    teacher_patch_mask,
                    use_target_ema=True,
                )

        z_full_online = z_full_online[:, :max_latents]
        z_full_teacher = z_full_teacher[:, :max_latents]
        full_online_mask = full_online_mask[:, :max_latents]
        full_teacher_mask = full_teacher_mask[:, :max_latents]
        if full_input_condition:
            # Reuse the reconstruction view: its visible outputs have attended
            # to the complete waveform, while the prior still selects only
            # visible positions through ``prompt_flow``.
            z_condition = z_full_online
            condition_mask = full_online_mask
        if z_condition is None:
            z_condition = z_full_online.new_zeros(
                patches.shape[0],
                max_latents,
                self.config.latent_dim,
            )
        condition_coverage = torch.where(
            visible_rows[:, None],
            condition_mask,
            valid_seed,
        )
        teacher_coverage = full_teacher_mask | ~target_region
        valid = valid_seed & condition_coverage & full_online_mask & teacher_coverage
        target_region = target_region & valid
        visible_region = valid & ~target_region

        z_clean = torch.where(
            visible_region.unsqueeze(-1),
            z_condition,
            z_full_online,
        )
        z_flow_target = torch.where(
            visible_region.unsqueeze(-1),
            z_condition,
            z_full_teacher,
        )
        valid_values = valid.unsqueeze(-1).to(dtype=z_clean.dtype)
        z_clean = z_clean * valid_values
        z_flow_target = z_flow_target * valid_values
        self._last_z_full_online = z_full_online * valid_values
        self._last_z_full_teacher = z_full_teacher.detach() * valid_values

        # TTA text conditioning ignores speech prefix alignment. Returning a
        # zero vector preserves the inherited method contract.
        prompt_patch_ends = torch.zeros(
            patches.shape[0],
            device=patches.device,
            dtype=torch.long,
        )
        return z_clean, target_region, valid, z_flow_target, prompt_patch_ends

    @staticmethod
    def _validate_contiguous_mask(mask: Tensor, valid_mask: Tensor) -> None:
        mask = mask.to(dtype=torch.bool)
        valid = valid_mask.to(device=mask.device, dtype=torch.bool)
        if mask.shape != valid.shape or bool((mask & ~valid).any().item()):
            raise ValueError("target mask must match and stay inside valid_latent_mask")
        for row in mask:
            indices = row.nonzero(as_tuple=False).flatten()
            if indices.numel() > 1 and int(indices[-1] - indices[0] + 1) != int(
                indices.numel()
            ):
                raise ValueError(
                    "TTA v1 encode_for_mask requires one contiguous target span"
                )

    def _condition_text_inputs(
        self,
        batch: dict[str, Any],
        *,
        mask: Tensor | None = None,
        valid_latent_mask: Tensor | None = None,
        prompt_patch_ends: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        del mask, valid_latent_mask, prompt_patch_ends
        captions = batch.get("caption")
        if captions is None:
            raise KeyError("TTA batch is missing required 'caption'")
        if isinstance(captions, str):
            captions = [captions]
        if not isinstance(captions, Sequence):
            raise TypeError("batch['caption'] must be a sequence of strings")
        device = batch["wav_clean"].device
        no_dropout = torch.zeros(len(captions), device=device, dtype=torch.bool)
        text_tokens, text_mask = self.text_conditioner(
            [str(caption) for caption in captions],
            device=device,
            dropout_mask=no_dropout,
        )
        if text_tokens.ndim != 3 or text_mask.shape != text_tokens.shape[:2]:
            raise ValueError(
                "text conditioner must return [B,L,C] tokens and [B,L] mask"
            )
        return text_tokens, text_mask.long().sum(dim=1)

    def _encode_text_ids(
        self,
        text_ids: Tensor,
        *,
        text_lengths: Tensor | None = None,
        device: torch.device | None = None,
    ) -> Tensor:
        del text_lengths
        if not isinstance(text_ids, Tensor) or text_ids.ndim != 3:
            raise ValueError("TTA caption embeddings must have shape [B,L,C]")
        return text_ids if device is None else text_ids.to(device=device)

    def _encode_clap_captions(
        self,
        captions: Sequence[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        dropout_mask: Tensor | None = None,
    ) -> Tensor | None:
        """Encode and synchronously CFG-drop the optional global caption condition."""

        if not self.config.clap_enabled:
            return None
        conditioner = getattr(self, "clap_conditioner", None)
        if conditioner is None:
            raise RuntimeError(
                "CLAP conditioning is enabled but no conditioner is attached"
            )
        captions = [str(caption) for caption in captions]
        embedding = conditioner(captions, device=device)
        if not isinstance(embedding, Tensor):
            raise TypeError("CLAP conditioner must return a Tensor")
        expected = (len(captions), int(self.config.clap_embedding_dim))
        if embedding.shape != expected:
            raise ValueError(
                f"CLAP conditioner must return {expected}, got {tuple(embedding.shape)}"
            )
        embedding = embedding.to(device=device, dtype=dtype)
        if dropout_mask is None:
            return embedding
        if dropout_mask.shape != (len(captions),):
            raise ValueError("caption dropout mask must have shape [B]")
        drop = dropout_mask.to(device=device, dtype=torch.bool)
        if not bool(drop.any().item()):
            return embedding
        null_context = getattr(conditioner, "null_context", None)
        if null_context is None:
            raise AttributeError("CLAP conditioner must expose null_context() for CFG")
        null_embedding = null_context(len(captions), device=device, dtype=dtype)
        if not isinstance(null_embedding, Tensor) or null_embedding.shape != expected:
            raise ValueError(f"CLAP null_context() must return {expected}")
        return torch.where(drop[:, None], null_embedding, embedding)

    def _masked_prior_flow_loss(
        self,
        batch: dict[str, Any],
        z_clean: Tensor,
        z_flow_target: Tensor,
        text_tokens: Tensor,
        *,
        text_lengths: Tensor | None,
        mask: Tensor,
        valid_latent_mask: Tensor,
        condition_dropout_mask: Tensor | None,
        flow_noise: Tensor | None = None,
        flow_time_override: Tensor | None = None,
        visible_condition_override: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        steps = int(self.config.flow_steps_per_recon)
        batch_size = z_clean.shape[0]
        z_stop = z_flow_target.detach()
        z_target = z_stop + float(self.config.target_grad_scale) * (
            z_flow_target - z_stop
        )
        z_target_flow = _repeat_steps(z_target, steps)
        z_visible = (
            z_clean
            if visible_condition_override is None
            else visible_condition_override
        )
        if z_visible.shape != z_clean.shape:
            raise ValueError("visible_condition_override must match z_clean")
        z_visible_flow = _repeat_steps(z_visible, steps)
        mask_flow = _repeat_steps(mask, steps)
        valid_flow = _repeat_steps(valid_latent_mask, steps)

        if text_lengths is None:
            text_mask = torch.ones(
                text_tokens.shape[:2], device=text_tokens.device, dtype=torch.bool
            )
        else:
            positions = torch.arange(text_tokens.shape[1], device=text_tokens.device)[
                None, :
            ]
            text_mask = (
                positions
                < text_lengths.to(device=text_tokens.device, dtype=torch.long)[:, None]
            )
        text_tokens, text_mask = self._replace_dropped_text(
            text_tokens,
            text_mask,
            condition_dropout_mask,
        )
        text_flow = _repeat_steps(text_tokens, steps)
        text_mask_flow = _repeat_steps(text_mask, steps)
        captions = batch.get("caption")
        if isinstance(captions, str):
            captions = [captions]
        if not isinstance(captions, Sequence):
            raise TypeError("batch['caption'] must be a sequence of strings")
        global_text_embedding = self._encode_clap_captions(
            captions,
            device=z_clean.device,
            dtype=z_clean.dtype,
            dropout_mask=condition_dropout_mask,
        )
        global_text_flow = (
            _repeat_steps(global_text_embedding, steps)
            if global_text_embedding is not None
            else None
        )
        if condition_dropout_mask is not None:
            if condition_dropout_mask.shape != (batch_size,):
                raise ValueError("condition_dropout_mask must have shape [B]")
            condition_dropout_mask_flow = _repeat_steps(
                condition_dropout_mask.to(device=z_clean.device, dtype=torch.bool),
                steps,
            )
        else:
            condition_dropout_mask_flow = None

        prompt_flow = (~mask_flow) & valid_flow
        eps = torch.randn_like(z_target_flow) if flow_noise is None else flow_noise
        if eps.shape != z_target_flow.shape:
            raise ValueError("flow_noise must match repeated target latents")
        if flow_time_override is None:
            t = self._sample_flow_time(
                z_target_flow.shape[0],
                device=z_clean.device,
                dtype=z_clean.dtype,
            )
        else:
            t = flow_time_override.to(device=z_clean.device, dtype=z_clean.dtype)
            if t.shape != (z_target_flow.shape[0],):
                raise ValueError("flow_time_override must have shape [steps * batch]")
        t_view = t[:, None, None]
        z_t = (1.0 - t_view) * eps + t_view * z_target_flow
        visible_condition_noise_mode = str(
            getattr(self.config, "visible_condition_noise_mode", "none")
        ).strip().lower().replace("-", "_")
        if visible_condition_noise_mode == "flow_matched":
            visible_for_prior = (1.0 - t_view) * eps + t_view * z_visible_flow
        else:
            visible_for_prior = z_visible_flow
        z_x_all = torch.where(prompt_flow.unsqueeze(-1), visible_for_prior, z_t)
        visible_condition_injection_mode = str(
            getattr(self.config, "visible_condition_injection_mode", "latent_prompt")
        ).strip().lower().replace("-", "_")
        if visible_condition_injection_mode == "none":
            # Strict no-mask alignment: the mask only selects whether the noisy
            # endpoint came from the online or EMA encoder.  It is not exposed
            # to the prior through either an audio condition or a prompt bit.
            z_cond_all = torch.zeros_like(z_visible_flow)
            prior_prompt_flow = torch.zeros_like(prompt_flow)
        else:
            z_cond_all = torch.where(
                prompt_flow.unsqueeze(-1),
                visible_for_prior,
                torch.zeros_like(z_visible_flow),
            )
            prior_prompt_flow = prompt_flow
        z_x_all = z_x_all * valid_flow.unsqueeze(-1).to(dtype=z_x_all.dtype)
        z_cond_all = z_cond_all * valid_flow.unsqueeze(-1).to(dtype=z_cond_all.dtype)

        h = self.fm_encoder(
            z_x_all,
            z_cond_all,
            text_flow,
            t,
            valid_audio_mask=valid_flow,
            text_mask=text_mask_flow,
            prompt_audio_mask=prior_prompt_flow,
            condition_dropout_mask=condition_dropout_mask_flow,
            condition_dropout_mode=self.config.prior_cfg_dropout_mode,
            global_text_embedding=global_text_flow,
        )
        fm_out = self.fm_head(h)
        target_mask = mask_flow & valid_flow
        prior_loss_scope = str(
            getattr(self.config, "prior_loss_scope", "target")
        ).strip().lower().replace("-", "_")
        loss_mask = valid_flow if prior_loss_scope == "all_valid" else target_mask
        # The prior consumes visible_for_prior at prompt positions, not z_t.
        # This distinction is immaterial for the historical target-only loss,
        # but all-token XPred supervision must measure velocity from the state
        # that was actually presented to the model.
        flow_state_for_loss = torch.where(
            prompt_flow.unsqueeze(-1), visible_for_prior, z_t
        )
        if self.config.prediction == "v_pred_v_loss":
            v_pred = fm_out
            v_target = z_target_flow - eps
            z_pred_all = z_t + (1.0 - t_view) * v_pred
            flow_loss = masked_mse(v_pred, v_target, loss_mask) * steps
            diagnostic_pred = v_pred
            diagnostic_target = v_target
            diagnostic_weight = None
        elif self.config.prediction == "x_pred_v_loss":
            z_pred_all = self._normalize_flow_x_pred(fm_out)
            denom = self._flow_xpred_denom(t_view)
            v_pred = (z_pred_all - flow_state_for_loss) / denom
            v_target = (z_target_flow - flow_state_for_loss) / denom
            flow_loss = masked_mse(v_pred, v_target, loss_mask) * steps
            diagnostic_pred = v_pred
            diagnostic_target = v_target
            diagnostic_weight = None
        elif self.config.prediction == "x_pred_v_loss_weight":
            z_pred_all = self._normalize_flow_x_pred(fm_out)
            denom = self._flow_xpred_denom(t_view.float())
            diagnostic_weight = denom.reciprocal().square()
            flow_loss = (
                masked_weighted_mse(
                    z_pred_all,
                    z_target_flow,
                    loss_mask,
                    diagnostic_weight,
                )
                * steps
            )
            diagnostic_pred = z_pred_all
            diagnostic_target = z_target_flow
        else:
            raise ValueError(
                f"unsupported TTA prior prediction mode: {self.config.prediction}"
            )

        # Report the two TTA training modes separately without changing the
        # mixed objective or retaining an extra autograd graph. This makes it
        # possible to verify that bucket-SSL and full generation both learn.
        plan = self._last_mask_plan
        full_generation_rows = torch.zeros(
            batch_size,
            device=target_mask.device,
            dtype=torch.bool,
        )
        if plan is not None and plan.full_generation_mask is not None:
            full_generation_rows = plan.full_generation_mask.to(
                device=target_mask.device,
                dtype=torch.bool,
            )
        full_generation_flow = _repeat_steps(full_generation_rows, steps)
        full_generation_target = target_mask & full_generation_flow[:, None]
        ssl_target = target_mask & ~full_generation_flow[:, None]
        visible_target = prompt_flow
        with torch.no_grad():
            if diagnostic_weight is None:
                ssl_flow_loss = (
                    masked_mse(
                        diagnostic_pred.detach(),
                        diagnostic_target.detach(),
                        ssl_target,
                    )
                    * steps
                )
                full_generation_flow_loss = (
                    masked_mse(
                        diagnostic_pred.detach(),
                        diagnostic_target.detach(),
                        full_generation_target,
                    )
                    * steps
                )
                visible_flow_loss = (
                    masked_mse(
                        diagnostic_pred.detach(),
                        diagnostic_target.detach(),
                        visible_target,
                    )
                    * steps
                )
            else:
                ssl_flow_loss = (
                    masked_weighted_mse(
                        diagnostic_pred.detach(),
                        diagnostic_target.detach(),
                        ssl_target,
                        diagnostic_weight.detach(),
                    )
                    * steps
                )
                full_generation_flow_loss = (
                    masked_weighted_mse(
                        diagnostic_pred.detach(),
                        diagnostic_target.detach(),
                        full_generation_target,
                        diagnostic_weight.detach(),
                    )
                    * steps
                )
                visible_flow_loss = (
                    masked_weighted_mse(
                        diagnostic_pred.detach(),
                        diagnostic_target.detach(),
                        visible_target,
                        diagnostic_weight.detach(),
                    )
                    * steps
                )

        z_x = z_x_all.view(steps, batch_size, *z_clean.shape[1:])[-1]
        z_cond = z_cond_all.view(steps, batch_size, *z_clean.shape[1:])[-1]
        z_pred = z_pred_all.view(steps, batch_size, *z_clean.shape[1:])[-1]
        flow_time = t.view(steps, batch_size)
        metrics = {
            "prior_fm_loss_ssl": ssl_flow_loss.detach(),
            "prior_fm_loss_full_generation": full_generation_flow_loss.detach(),
            "prior_fm_loss_visible": visible_flow_loss.detach(),
            "prior_loss_all_valid_enabled": z_clean.new_tensor(
                float(prior_loss_scope == "all_valid")
            ),
            "visible_condition_injection_enabled": z_clean.new_tensor(
                float(visible_condition_injection_mode != "none")
            ),
            "predicted_velocity_rms": diagnostic_pred.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "target_velocity_rms": diagnostic_target.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "caption_tokens_mean": text_mask.float().sum(dim=1).mean().detach(),
            "text_injection_joint": z_clean.new_tensor(
                float(self.config.text_injection == "joint")
            ),
            "text_injection_mmdit": z_clean.new_tensor(
                float(self.config.text_injection == "mmdit")
            ),
            "clap_conditioning_enabled": z_clean.new_tensor(
                float(self.config.clap_enabled)
            ),
        }
        if visible_condition_noise_mode == "flow_matched":
            with torch.no_grad():
                visible_weights = prompt_flow.unsqueeze(-1).to(
                    dtype=z_clean.dtype
                )
                visible_value_count = (
                    prompt_flow.sum().to(dtype=z_clean.dtype)
                    * z_clean.shape[-1]
                ).clamp_min(1.0)
                noise_component = (1.0 - t_view) * eps
                visible_delta = visible_for_prior - z_visible_flow
                visible_token_count = prompt_flow.sum().to(
                    dtype=z_clean.dtype
                ).clamp_min(1.0)
            metrics.update(
                {
                    "visible_condition_noise_enabled": z_clean.new_tensor(1.0),
                    "visible_condition_noise_rms": (
                        noise_component.float().square()
                        * visible_weights.float()
                    ).sum().div(visible_value_count).sqrt(),
                    "visible_condition_noisy_delta_rms": (
                        visible_delta.float().square()
                        * visible_weights.float()
                    ).sum().div(visible_value_count).sqrt(),
                    "visible_condition_effective_t_mean": (
                        t[:, None] * prompt_flow.to(dtype=t.dtype)
                    ).sum().div(visible_token_count),
                }
            )
        if global_text_embedding is not None:
            metrics["clap_embedding_norm"] = (
                global_text_embedding.float().norm(dim=-1).mean().detach()
            )
        return flow_loss, z_x, z_cond, z_pred, flow_time, z_pred_all, metrics

    def _replace_dropped_text(
        self,
        text_tokens: Tensor,
        text_mask: Tensor,
        dropout_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if dropout_mask is None:
            return text_tokens, text_mask
        if dropout_mask.shape != (text_tokens.shape[0],):
            raise ValueError("caption dropout mask must have shape [B]")
        # Keep the learned null token in the graph even when this batch has no
        # dropped captions. It then receives a zero (rather than missing)
        # gradient, so default DDP does not classify it as an unused parameter.
        null_tokens, null_mask = self._null_text_sequence(
            text_tokens.shape[0],
            text_tokens.shape[1],
            device=text_tokens.device,
            dtype=text_tokens.dtype,
        )
        drop = dropout_mask.to(device=text_tokens.device, dtype=torch.bool)
        return (
            torch.where(drop[:, None, None], null_tokens, text_tokens),
            torch.where(drop[:, None], null_mask, text_mask),
        )

    def _null_text_sequence(
        self,
        batch_size: int,
        sequence_length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        null_context = getattr(self.text_conditioner, "null_context", None)
        if null_context is None:
            raise AttributeError("text conditioner must expose null_context() for CFG")
        first, first_mask = null_context(batch_size, device=device, dtype=dtype)
        if first.shape[1] != 1 or first_mask.shape != (batch_size, 1):
            raise ValueError("null_context() must return one valid token per row")
        if sequence_length == 1:
            return first, first_mask
        tokens = torch.cat(
            (
                first,
                text_tokens_zeros(first, sequence_length - 1),
            ),
            dim=1,
        )
        mask = torch.zeros(
            (batch_size, sequence_length), device=device, dtype=torch.bool
        )
        mask[:, 0] = True
        return tokens, mask

    @torch.no_grad()
    def generate(
        self,
        captions: Sequence[str] | str,
        *,
        seconds: float = 10.0,
        num_steps: int = 16,
        prior_steps: int | None = None,
        wave_steps: int | None = None,
        solver: str = "euler",
        cfg_strength: float = 0.0,
        cfg_rescale: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if isinstance(captions, str):
            captions = [captions]
        captions = [str(caption) for caption in captions]
        if not captions:
            raise ValueError("generate requires at least one caption")
        if float(seconds) <= 0.0:
            raise ValueError("seconds must be positive")
        solver = str(solver).lower()
        if solver not in {"euler", "heun", "rk4"}:
            raise ValueError("solver must be 'euler', 'heun', or 'rk4'")
        prior_step_count = int(num_steps if prior_steps is None else prior_steps)
        wave_step_count = int(num_steps if wave_steps is None else wave_steps)
        if prior_step_count < 1 or wave_step_count < 1:
            raise ValueError("prior_steps and wave_steps must be positive")

        param = next(self.fm_encoder.parameters())
        device, dtype = param.device, param.dtype
        sample_count = max(1, int(round(float(seconds) * int(self.config.sample_rate))))
        required_patches = math.ceil(sample_count / int(self.config.patch_size))
        total_tokens = math.ceil(required_patches / int(self.config.downsample_factor))
        total_patches = total_tokens * int(self.config.downsample_factor)
        batch_size = len(captions)
        valid_latent_mask = torch.ones(
            (batch_size, total_tokens), device=device, dtype=torch.bool
        )

        no_dropout = torch.zeros(batch_size, device=device, dtype=torch.bool)
        text_tokens, text_mask = self.text_conditioner(
            captions,
            device=device,
            dropout_mask=no_dropout,
        )
        text_tokens = text_tokens.to(device=device)
        text_mask = text_mask.to(device=device, dtype=torch.bool)
        global_text_embedding = self._encode_clap_captions(
            captions,
            device=device,
            dtype=dtype,
        )
        z_state = torch.randn(
            (batch_size, total_tokens, int(self.config.latent_dim)),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        z_cond = torch.zeros_like(z_state)
        prompt_mask = torch.zeros_like(valid_latent_mask)

        def prior_prediction(
            state: Tensor,
            time_value: float,
            context: Tensor,
            context_mask: Tensor,
            global_context: Tensor | None,
        ) -> Tensor:
            t = torch.full((state.shape[0],), time_value, device=device, dtype=dtype)
            valid = valid_latent_mask
            cond = z_cond
            prompt = prompt_mask
            if state.shape[0] != batch_size:
                valid = torch.cat((valid, valid), dim=0)
                cond = torch.cat((cond, cond), dim=0)
                prompt = torch.cat((prompt, prompt), dim=0)
            h = self.fm_encoder(
                state,
                cond,
                context,
                t,
                valid_audio_mask=valid,
                text_mask=context_mask,
                prompt_audio_mask=prompt,
                global_text_embedding=global_context,
            )
            prediction = self.fm_head(h)
            if self.config.prediction == "v_pred_v_loss":
                return prediction
            return self._normalize_flow_x_pred(prediction)

        if float(cfg_strength) > 0.0:
            null_tokens, null_mask = self._null_text_sequence(
                batch_size,
                text_tokens.shape[1],
                device=device,
                dtype=text_tokens.dtype,
            )
            cfg_text = torch.cat((text_tokens, null_tokens), dim=0)
            cfg_text_mask = torch.cat((text_mask, null_mask), dim=0)
            if global_text_embedding is not None:
                clap_conditioner = getattr(self, "clap_conditioner", None)
                null_context = getattr(clap_conditioner, "null_context", None)
                if null_context is None:
                    raise AttributeError(
                        "CLAP conditioner must expose null_context() for CFG"
                    )
                null_global = null_context(batch_size, device=device, dtype=dtype)
                cfg_global_text = torch.cat((global_text_embedding, null_global), dim=0)
            else:
                cfg_global_text = None
        else:
            cfg_text = text_tokens
            cfg_text_mask = text_mask
            cfg_global_text = global_text_embedding

        def prior_velocity(state: Tensor, time_value: float) -> Tensor:
            if float(cfg_strength) <= 0.0:
                prediction = prior_prediction(
                    state,
                    time_value,
                    cfg_text,
                    cfg_text_mask,
                    cfg_global_text,
                )
                if self.config.prediction == "v_pred_v_loss":
                    return prediction
                return self._velocity_from_x0(prediction, state, time_value)
            state_cat = torch.cat((state, state), dim=0)
            pred_cond, pred_uncond = prior_prediction(
                state_cat,
                time_value,
                cfg_text,
                cfg_text_mask,
                cfg_global_text,
            ).chunk(2, dim=0)
            if self.config.prediction == "v_pred_v_loss":
                v_cond, v_uncond = pred_cond, pred_uncond
            else:
                v_cond = self._velocity_from_x0(pred_cond, state, time_value)
                v_uncond = self._velocity_from_x0(pred_uncond, state, time_value)
            guided = v_cond + float(cfg_strength) * (v_cond - v_uncond)
            if float(cfg_rescale) > 0.0:
                guided = self._rescale_guided_velocity(
                    guided,
                    v_cond,
                    target_start=0,
                    mix=max(0.0, min(float(cfg_rescale), 1.0)),
                )
            return guided

        prior_grid = self._flow_inference_time_grid(
            prior_step_count, device=device, dtype=dtype
        )
        for idx in range(prior_step_count):
            z_state = self._solver_step(
                z_state, prior_velocity, prior_grid, idx, solver
            )

        if self.config.decoder_objective == "wave":
            return self.decode_tokens_raw(z_state, original_len=sample_count)
        if self.wave_fm_decoder is None:
            raise RuntimeError("wave FM decoder is not initialized")

        valid_wave_mask = torch.ones(
            (batch_size, total_patches), device=device, dtype=torch.bool
        )
        wave_state = torch.randn(
            (batch_size, total_patches, int(self.config.patch_size)),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        wave_cond = z_state.repeat_interleave(
            int(self.config.downsample_factor), dim=1
        )[:, :total_patches]

        def wave_velocity(state: Tensor, time_value: float) -> Tensor:
            t = torch.full((batch_size,), time_value, device=device, dtype=dtype)
            kwargs: dict[str, Tensor | None] = {"valid_wave_mask": valid_wave_mask}
            if isinstance(self.wave_fm_decoder, LinearUDiTUNetWaveFMDecoder):
                kwargs["speaker_emb"] = None
            x0 = self.wave_fm_decoder(state, wave_cond, t, **kwargs)
            return self._velocity_from_x0(x0, state, time_value)

        wave_grid = self._flow_inference_time_grid(
            wave_step_count, device=device, dtype=dtype
        )
        for idx in range(wave_step_count):
            wave_state = self._solver_step(
                wave_state, wave_velocity, wave_grid, idx, solver
            )
        waveform_model = self.patchify.unpatchify(wave_state, original_len=sample_count)
        return self._unscale_waveform(waveform_model)

    def checkpoint_model_state(self, *, to_cpu: bool = True) -> dict[str, Tensor]:
        return checkpoint_model_state(self, to_cpu=to_cpu)

    def _velocity_from_x0(self, x0: Tensor, state: Tensor, time_value: float) -> Tensor:
        denom = max(float(self.config.flow_xpred_denom_min), 1.0 - float(time_value))
        return (x0 - state) / denom

    @staticmethod
    def _solver_step(
        state: Tensor,
        velocity_fn: Any,
        time_grid: Tensor,
        step_idx: int,
        solver: str,
    ) -> Tensor:
        time_value = float(time_grid[step_idx].item())
        next_time = float(time_grid[step_idx + 1].item())
        dt = (time_grid[step_idx + 1] - time_grid[step_idx]).to(dtype=state.dtype)
        if solver == "euler":
            return state + dt * velocity_fn(state, time_value)
        if solver == "heun":
            k1 = velocity_fn(state, time_value)
            k2 = velocity_fn(state + dt * k1, next_time)
            return state + dt * 0.5 * (k1 + k2)
        middle = 0.5 * (time_value + next_time)
        half_dt = dt * 0.5
        k1 = velocity_fn(state, time_value)
        k2 = velocity_fn(state + half_dt * k1, middle)
        k3 = velocity_fn(state + half_dt * k2, middle)
        k4 = velocity_fn(state + dt * k3, next_time)
        return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def text_tokens_zeros(first: Tensor, count: int) -> Tensor:
    return first.new_zeros((first.shape[0], int(count), first.shape[-1]))


def build_model(
    config: Mapping[str, Any],
    *,
    text_conditioner: nn.Module | None = None,
    clap_conditioner: nn.Module | None = None,
    model_class: type[TTALatentAudio] = TTALatentAudio,
    config_class: type[TTALatentAudioConfig] = TTALatentAudioConfig,
) -> TTALatentAudio:
    """Build a TTA model from the public nested ``model`` config section."""

    raw = deepcopy(dict(config))
    public_mask = (
        deepcopy(dict(raw["mask"]))
        if isinstance(raw.get("mask"), Mapping)
        else None
    )
    if "model" in raw and isinstance(raw["model"], Mapping):
        raw = deepcopy(dict(raw["model"]))
        if public_mask is not None:
            raw["mask"] = public_mask
    if "wave_fm" in raw:
        raise ValueError(
            "model.wave_fm was removed; select and configure model.decoder instead"
        )
    from task_audio.configs.model_architectures import (
        MODEL_SIZE_PRESETS,
        resolve_decoder_config,
    )

    size = raw.get("size")
    if size is not None:
        normalized_size = str(size)
        if normalized_size not in MODEL_SIZE_PRESETS:
            raise ValueError(
                f"unknown TTA model size {normalized_size!r}; "
                f"available: {sorted(MODEL_SIZE_PRESETS)}"
            )
        merged = deepcopy(MODEL_SIZE_PRESETS[normalized_size])
        _deep_update(merged, raw)
        raw = merged
    decoder = raw.get("decoder")
    if isinstance(decoder, Mapping):
        decoder_type = str(decoder.get("type", "deterministic")).lower().strip()
        resolved_fields = {"depth", "upsample_depths"}
        if decoder_type == "deterministic" and resolved_fields.issubset(decoder):
            expected_depth = int(raw.get("encoder_same_depth", 0))
            expected_upsample_depths = list(
                reversed([int(value) for value in raw.get("encoder_down_depths", ())])
            )
            if (
                int(decoder["depth"]) != expected_depth
                or list(decoder["upsample_depths"]) != expected_upsample_depths
            ):
                raise ValueError(
                    "resolved deterministic decoder does not match the selected encoder"
                )
            # ``build_config`` already materializes these two derived fields.
            # Reduce it back to the public selector before the model builder's
            # own validation, making the config -> model boundary idempotent.
            decoder = deepcopy(dict(decoder))
            decoder.pop("depth")
            decoder.pop("upsample_depths")
            raw["decoder"] = decoder
    raw["decoder"] = resolve_decoder_config(raw)
    text_config = deepcopy(dict(raw.get("text", {})))
    clap_config = deepcopy(dict(raw.get("clap", {})))
    text_encoder = (
        str(text_config.get("encoder", "flan_t5")).strip().lower().replace("-", "_")
    )
    if text_encoder not in {"flan_t5", "flan_t5_large"}:
        raise ValueError("TTA v1 requires model.text.encoder='flan_t5'")
    if not bool(text_config.get("freeze", True)):
        raise ValueError("TTA v1 requires the FLAN-T5 text encoder to remain frozen")
    mask_config = deepcopy(dict(raw.get("mask", {})))
    encoder_config = raw.get("encoder")
    prior_config = raw.get("prior_fm")
    decoder_config = raw.get("decoder")
    if isinstance(encoder_config, Mapping):
        for name in ("heads", "dim_head", "ffn_mult"):
            if encoder_config.get(name) is not None:
                raw[name] = encoder_config[name]
    if isinstance(prior_config, Mapping):
        prior_mapping = {
            "depth": "fm_depth",
            "fused_depth": "fm_mmdit_fused_depth",
            "mmdit_fused_depth": "fm_mmdit_fused_depth",
            "hidden_dim": "fm_hidden_dim",
            "adaln_every": "fm_adaln_every",
            "heads": "fm_heads",
            "dim_head": "fm_dim_head",
            "ffn_mult": "fm_ffn_mult",
        }
        for source, target in prior_mapping.items():
            if prior_config.get(source) is not None:
                raw.setdefault(target, prior_config[source])
        raw.setdefault("prior_fm_input_fusion", prior_config.get("input_fusion", "add"))
    # The public text section is canonical. Keep the prior-local alias for
    # direct model configs, then fall back to the flat dataclass field.
    text_injection = text_config.get("injection")
    if text_injection is None and isinstance(prior_config, Mapping):
        text_injection = prior_config.get("text_injection")
    if text_injection is not None:
        raw["text_injection"] = text_injection
    else:
        raw.setdefault("text_injection", "mmdit")

    if isinstance(decoder_config, Mapping):
        decoder_type = str(decoder_config["type"])
        head = str(decoder_config["head"])
        latent_noise = decoder_config["latent_noise"]
        raw["decoder_latent_noise_mode"] = str(latent_noise["mode"])
        raw["decoder_latent_noise_std"] = float(latent_noise["std"])
        raw["decoder_latent_noise_t_start"] = float(latent_noise["t_start"])
        raw["decoder_latent_noise_prob"] = float(latent_noise["probability"])
        raw["reconstruction_latent_source"] = str(
            decoder_config.get("reconstruction_latent_source", "full_online")
        )
        raw["decoder_objective"] = "wave_fm" if decoder_type == "fm" else "wave"
        raw["build_deterministic_decoder"] = decoder_type == "deterministic"
        raw["decoder_head"] = head

        if decoder_type == "deterministic":
            # The public deterministic decoder has no backbone choice. The
            # inherited TTS implementation calls its fixed symmetric output
            # stack "dit", so keep that name only at this private boundary.
            raw["decoder_backbone"] = "dit"
            raw["decoder_up_depths"] = decoder_config["upsample_depths"]
            raw["decoder_same_depth"] = int(decoder_config["depth"])
        else:
            backbone = str(decoder_config["backbone"])
            decoder_depth = int(
                decoder_config.get(
                    "depth",
                    prior_config.get("depth", raw.get("fm_depth", 1))
                    if isinstance(prior_config, Mapping)
                    else raw.get("fm_depth", 1),
                )
            )
            decoder_depths = decoder_config.get("depths", [4, 4, 8, 4, 4])
            decoder_input_fusion = decoder_config.get("input_fusion", "add")
            decoder_hidden_dim = decoder_config.get(
                "hidden_dim",
                prior_config.get("hidden_dim", raw.get("fm_hidden_dim"))
                if isinstance(prior_config, Mapping)
                else raw.get("fm_hidden_dim"),
            )
            decoder_adaln_every = decoder_config.get(
                "adaln_every",
                prior_config.get("adaln_every", raw.get("fm_adaln_every", 2))
                if isinstance(prior_config, Mapping)
                else raw.get("fm_adaln_every", 2),
            )
            decoder_heads = decoder_config.get(
                "heads",
                prior_config.get("heads", raw.get("fm_heads"))
                if isinstance(prior_config, Mapping)
                else raw.get("fm_heads"),
            )
            decoder_dim_head = decoder_config.get(
                "dim_head",
                prior_config.get("dim_head", raw.get("fm_dim_head"))
                if isinstance(prior_config, Mapping)
                else raw.get("fm_dim_head"),
            )
            decoder_ffn_mult = decoder_config.get(
                "ffn_mult",
                prior_config.get("ffn_mult", raw.get("fm_ffn_mult"))
                if isinstance(prior_config, Mapping)
                else raw.get("fm_ffn_mult"),
            )
            raw["wave_fm_backbone"] = (
                "linear_udit_unet" if backbone == "unet" else "transformer"
            )
            # Internal inherited-field aliases; no public ``wave_fm`` section.
            raw["wave_fm_depth"] = decoder_depth
            raw["wave_fm_hidden_dim"] = decoder_hidden_dim
            raw["wave_fm_adaln_every"] = decoder_adaln_every
            raw["wave_fm_heads"] = decoder_heads
            raw["wave_fm_dim_head"] = decoder_dim_head
            raw["wave_fm_ffn_mult"] = decoder_ffn_mult
            raw["wave_fm_input_fusion"] = decoder_input_fusion
            raw["wave_fm_unet_depths"] = decoder_depths
            raw["wave_fm_encoder_noise_std"] = float(
                raw.get(
                    "decoder_condition_noise_std",
                    raw.get("wave_fm_encoder_noise_std", 0.05),
                )
            )

    decoder_type = (
        str(decoder_config.get("type", "fm")).lower().replace("-", "_")
        if isinstance(decoder_config, Mapping)
        else (
            "deterministic"
            if str(raw.get("decoder_objective", "wave_fm")) == "wave"
            else "fm"
        )
    )
    loss_config = raw.get("loss")
    loss_defaults = (
        {
            "lambda_flow": 1.0,
            "lambda_wave_fm": 0.0,
            "lambda_recon": 1.0,
            "lambda_mel": 0.05,
        }
        if decoder_type == "deterministic"
        else {
            "lambda_flow": 1.0,
            "lambda_wave_fm": 1.0,
            "lambda_recon": 0.0,
            "lambda_mel": 0.0,
        }
    )
    for name, value in loss_defaults.items():
        raw.setdefault(name, value)
    if isinstance(loss_config, Mapping):
        if "wave_fm" in loss_config:
            raise ValueError("model.loss.wave_fm was renamed to model.loss.decoder")
        loss_mapping = {
            "prior_fm": "lambda_flow",
            "flow": "lambda_flow",
            "decoder": "lambda_wave_fm",
            "recon": "lambda_recon",
            "mel": "lambda_mel",
            "latent_reg": "lambda_latent_reg",
            "recon_loss": "recon_loss",
        }
        for source, target in loss_mapping.items():
            if loss_config.get(source) is not None:
                raw[target] = loss_config[source]
        reconstruction_config = loss_config.get("reconstruction")
        if isinstance(reconstruction_config, Mapping):
            reconstruction_type = (
                str(reconstruction_config.get("type", "same_phase_mrstft"))
                .strip()
                .lower()
                .replace("-", "_")
            )
            reconstruction_weight = float(reconstruction_config.get("weight", 0.0))
            raw["same_mrstft_enabled"] = bool(
                reconstruction_config.get(
                    "enabled",
                    reconstruction_type == "same_phase_mrstft"
                    and reconstruction_weight > 0.0,
                )
            )
            raw["lambda_same_mrstft"] = reconstruction_weight
            raw["same_mrstft_crop_seconds"] = float(
                reconstruction_config.get("crop_seconds", 2.0)
            )
            raw["same_mrstft_fft_sizes"] = reconstruction_config.get(
                "fft_sizes", [32, 64, 128, 256, 512, 1024, 2048]
            )
            raw["same_mrstft_hop_ratio"] = float(
                reconstruction_config.get("hop_ratio", 0.25)
            )
            raw["same_mrstft_k_weighting"] = bool(
                reconstruction_config.get("k_weighting", True)
            )
            raw["same_mrstft_eps"] = float(reconstruction_config.get("eps", 1.0e-7))
            raw["same_mrstft_stability_eps"] = float(
                reconstruction_config.get("stability_eps", 1.0e-4)
            )
        highband_config = loss_config.get("highband_detail")
        if isinstance(highband_config, Mapping):
            highband_weight = float(highband_config.get("weight", 0.0))
            raw["highband_detail_enabled"] = bool(
                highband_config.get("enabled", highband_weight > 0.0)
            )
            raw["lambda_highband_detail"] = highband_weight
            raw["highband_detail_fft_sizes"] = highband_config.get(
                "fft_sizes", [512, 1024, 2048]
            )
            raw["highband_detail_bands_hz"] = highband_config.get(
                "bands_hz", [[4_000.0, 6_000.0], [6_000.0, 8_000.0]]
            )
            raw["highband_detail_hop_ratio"] = float(
                highband_config.get("hop_ratio", 0.25)
            )
            raw["highband_detail_log_magnitude_weight"] = float(
                highband_config.get("log_magnitude_weight", 1.0)
            )
            raw["highband_detail_spectral_flux_weight"] = float(
                highband_config.get("spectral_flux_weight", 0.25)
            )
            raw["highband_detail_stability_eps"] = float(
                highband_config.get("stability_eps", 1.0e-4)
            )
        stationary_artifact_config = loss_config.get("stationary_artifact")
        if isinstance(stationary_artifact_config, Mapping):
            stationary_artifact_weight = float(
                stationary_artifact_config.get("weight", 0.0)
            )
            raw["stationary_artifact_enabled"] = bool(
                stationary_artifact_config.get(
                    "enabled", stationary_artifact_weight > 0.0
                )
            )
            raw["lambda_stationary_artifact"] = stationary_artifact_weight
            raw["stationary_artifact_fft_sizes"] = stationary_artifact_config.get(
                "fft_sizes", [1024, 2048, 4096]
            )
            raw["stationary_artifact_min_frequency_hz"] = float(
                stationary_artifact_config.get("min_frequency_hz", 200.0)
            )
            raw["stationary_artifact_max_frequency_hz"] = float(
                stationary_artifact_config.get("max_frequency_hz", 7_800.0)
            )
            raw["stationary_artifact_hop_ratio"] = float(
                stationary_artifact_config.get("hop_ratio", 0.25)
            )
            raw["stationary_artifact_local_frequency_bins"] = int(
                stationary_artifact_config.get("local_frequency_bins", 31)
            )
            raw["stationary_artifact_prominence_threshold_db"] = float(
                stationary_artifact_config.get("prominence_threshold_db", 6.0)
            )
            raw["stationary_artifact_softness_db"] = float(
                stationary_artifact_config.get("softness_db", 2.0)
            )
            raw["stationary_artifact_excess_margin_db"] = float(
                stationary_artifact_config.get("excess_margin_db", 0.0)
            )
            raw["stationary_artifact_top_frequency_fraction"] = float(
                stationary_artifact_config.get("top_frequency_fraction", 1.0)
            )
            raw["stationary_artifact_stability_eps"] = float(
                stationary_artifact_config.get("stability_eps", 1.0e-5)
            )
        adversarial_config = loss_config.get("adversarial")
        if isinstance(adversarial_config, Mapping):
            raw["adversarial_enabled"] = bool(adversarial_config.get("enabled", False))
            raw["adversarial_start_step"] = int(
                adversarial_config.get("start_step", 10_000)
            )
            raw["adversarial_generator_weight"] = float(
                adversarial_config.get("generator_weight", 0.1)
            )
            raw["adversarial_feature_matching_weight"] = float(
                adversarial_config.get("feature_matching_weight", 5.0)
            )
            raw["adversarial_stft_fft_sizes"] = adversarial_config.get(
                "stft_fft_sizes", [128, 256, 512, 1024, 2048]
            )
            raw["adversarial_pqmf_enabled"] = bool(
                adversarial_config.get("pqmf_enabled", True)
            )
            raw["adversarial_chroma_enabled"] = bool(
                adversarial_config.get("chroma_enabled", False)
            )

    # ``mask.policy`` is the public, canonical selector.  ``model.mask_policy``
    # is a resolved-config compatibility alias and must never override a mask
    # block supplied at the config -> model boundary.  Production recipes may
    # derive from a baseline and replace the top-level mask after the baseline
    # has already materialized the legacy alias; preferring the alias silently
    # ran those recipes with the baseline policy.
    nested_mask_policy = mask_config.get("policy")
    legacy_mask_policy = raw.get("mask_policy")
    if nested_mask_policy is not None and legacy_mask_policy is not None:
        normalized_nested = str(nested_mask_policy).strip().lower().replace("-", "_")
        normalized_legacy = str(legacy_mask_policy).strip().lower().replace("-", "_")
        if normalized_nested != normalized_legacy:
            warnings.warn(
                "conflicting mask selectors: mask.policy="
                f"{normalized_nested!r} overrides deprecated "
                f"model.mask_policy={normalized_legacy!r}",
                UserWarning,
                stacklevel=2,
            )
    mask_source = (
        str(
            nested_mask_policy
            if nested_mask_policy is not None
            else raw.get("mask_policy", raw.get("mask_source", "bucket_ssl"))
        )
        .strip()
        .lower()
        .replace("-", "_")
    )
    if mask_source in {"random_span", "tts_random_span"}:
        mask_config = {
            "policy": "random_span",
            "min_ratio": float(
                mask_config.get("min_ratio", raw.get("mask_ratio_min", 0.70))
            ),
            "max_ratio": float(
                mask_config.get("max_ratio", raw.get("mask_ratio_max", 1.00))
            ),
        }
        raw["mask_ratio_min"] = mask_config["min_ratio"]
        raw["mask_ratio_max"] = mask_config["max_ratio"]
    elif mask_source in {
        "bucket_ssl",
        "tta_bucket_ssl",
        "bucket_ssl_shifted",
        "bucket_ssl_coverage",
        "bucket_ssl_shifted_coverage",
    }:
        if mask_source == "tta_bucket_ssl":
            mask_source = "bucket_ssl"
        mask_config = {
            "policy": mask_source,
            "full_generation_probability": float(
                mask_config.get("full_generation_probability", 0.25)
            ),
            "unit_tokens": int(mask_config.get("unit_tokens", 3)),
            "bucket_tokens": int(mask_config.get("bucket_tokens", 50)),
            "visible_ratio_min": float(mask_config.get("visible_ratio_min", 0.0)),
            "visible_ratio_max": float(mask_config.get("visible_ratio_max", 0.30)),
            "random_bucket_offset": bool(
                mask_config.get(
                    "random_bucket_offset",
                    mask_source
                    in {"bucket_ssl_shifted", "bucket_ssl_shifted_coverage"},
                )
            ),
            "ensure_bucket_coverage": bool(
                mask_config.get(
                    "ensure_bucket_coverage",
                    mask_source
                    in {"bucket_ssl_coverage", "bucket_ssl_shifted_coverage"},
                )
            ),
        }
    elif mask_source in {
        "bucket_scattered",
        "bucket_uniform_scattered",
        "bucket_span",
        "global_scattered",
        "global_variable_block_matched",
        "global_span",
    }:
        mask_config = {
            "policy": mask_source,
            "full_generation_probability": float(
                mask_config.get("full_generation_probability", 0.25)
            ),
            "unit_tokens": int(mask_config.get("unit_tokens", 3)),
            "bucket_tokens": int(mask_config.get("bucket_tokens", 50)),
            "visible_ratio_min": float(mask_config.get("visible_ratio_min", 0.0)),
            "visible_ratio_max": float(mask_config.get("visible_ratio_max", 0.30)),
            "seed": int(mask_config.get("seed", 1234)),
        }
        if mask_source == "global_variable_block_matched":
            mask_config["block_lengths"] = tuple(
                int(value) for value in mask_config.get("block_lengths", (3, 5, 8))
            )
    elif mask_source == "global_dynamic_block_v1":
        mask_config = dict(mask_config)
        mask_config["policy"] = "global_dynamic_block_v1"
    elif mask_source in {"global_variable_block", "global_blocks"}:
        mask_config = dict(mask_config)
        mask_config["policy"] = "global_variable_block"
    elif mask_source in {"multi_anchor", "multi_anchor_context"}:
        mask_config = dict(mask_config)
        mask_config["policy"] = "multi_anchor"
    elif mask_source in {"hierarchical_macro_micro", "macro_micro"}:
        mask_config = dict(mask_config)
        mask_config["policy"] = "hierarchical_macro_micro"
    else:
        raise ValueError(
            "mask policy must be 'random_span', a 'bucket_ssl' variant, "
            "a structural 2x2 policy, 'global_variable_block', 'multi_anchor', or "
            "'hierarchical_macro_micro'"
        )
    # The speech dataclass retains random_span as its structural mask source;
    # TTALatentAudio dispatches the actual TTA policy through MaskPlan.
    raw["mask_source"] = "random_span"
    raw["prior_cfg_dropout"] = float(
        raw.get(
            "prior_cfg_dropout",
            text_config.get("cfg_dropout", raw.get("cfg_dropout", 0.1)),
        )
    )
    clap_enabled = bool(clap_config.get("enabled", raw.get("clap_enabled", False)))
    clap_embedding_dim = int(
        clap_config.get("embedding_dim", raw.get("clap_embedding_dim", 512))
    )
    if clap_embedding_dim < 1:
        raise ValueError("model.clap.embedding_dim must be positive")
    clap_injection = (
        str(clap_config.get("injection", "global_adaln")).lower().replace("-", "_")
    )
    if clap_injection not in {"global_adaln", "adaln", "global"}:
        raise ValueError("model.clap.injection must be 'global_adaln'")
    if not bool(clap_config.get("freeze", True)):
        raise ValueError("TTA requires the CLAP conditioner to remain frozen")
    raw["clap_enabled"] = clap_enabled
    raw["clap_embedding_dim"] = clap_embedding_dim

    allowed = {field.name for field in fields(config_class)}
    model_kwargs = {
        key: value for key, value in raw.items() if key in allowed and value is not None
    }
    model_config = config_class(**model_kwargs)

    if text_conditioner is None:
        text_dim = int(
            text_config.get("text_dim", raw.get("text_dim", model_config.text_dim))
        )
        conditioner_config = FlanT5ConditionerConfig(
            name_or_path=str(
                text_config.get(
                    "name_or_path",
                    raw.get("flan_t5_name_or_path", DEFAULT_FLAN_T5_LARGE),
                )
            ),
            text_dim=text_dim,
            max_length=int(
                text_config.get("max_length", raw.get("flan_t5_max_length", 128))
            ),
            # The model owns CFG sampling so logging and prior audio-drop modes
            # stay synchronized. Conditioner dropout remains disabled here.
            cfg_dropout=0.0,
            freeze=True,
            local_files_only=bool(text_config.get("local_files_only", True)),
            padding_mode=str(text_config.get("padding_mode", "zero")),
            tokenizer_padding=str(text_config.get("tokenizer_padding", "max_length")),
        )
        text_conditioner = FlanT5Conditioner(conditioner_config)

    if clap_enabled and clap_conditioner is None:
        checkpoint = clap_config.get("checkpoint")
        if not checkpoint:
            raise ValueError(
                "model.clap.checkpoint is required when model.clap.enabled=True"
            )
        clap_conditioner = ClapTextConditioner(
            ClapTextConditionerConfig(
                checkpoint=str(checkpoint),
                embedding_dim=clap_embedding_dim,
                amodel=str(clap_config.get("amodel", "HTSAT-tiny")),
                enable_fusion=bool(clap_config.get("enable_fusion", False)),
                text_only=bool(clap_config.get("text_only", True)),
            )
        )
    elif not clap_enabled and clap_conditioner is not None:
        raise ValueError(
            "clap_conditioner was supplied while model.clap.enabled is false"
        )

    return model_class(
        model_config,
        text_conditioner=text_conditioner,
        clap_conditioner=clap_conditioner,
        mask_policy=build_mask_policy(mask_config),
    )


def _deep_update(target: dict[str, Any], updates: Mapping[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)


__all__ = [
    "DEFAULT_FLAN_T5_LARGE",
    "TTALatentAudio",
    "TTALatentAudioConfig",
    "build_model",
]
