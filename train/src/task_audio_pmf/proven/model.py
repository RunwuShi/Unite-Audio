from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from task_audio.models.model import (
    TTALatentAudio,
    TTALatentAudioConfig,
    build_model as build_task_audio_model,
)
from task_audio._backbone.latent_tts import FMHead, _repeat_steps


def _per_row_masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    if mask.shape != prediction.shape[:2]:
        raise ValueError("mask must have shape [B,T]")
    weights = mask.unsqueeze(-1).to(dtype=prediction.dtype)
    numerator = ((prediction.float() - target.float()).square() * weights).sum(
        dim=tuple(range(1, prediction.ndim))
    )
    denominator = weights.expand_as(prediction).sum(
        dim=tuple(range(1, prediction.ndim))
    ).clamp_min(1.0)
    return numerator / denominator


def _row_group_mean(values: Tensor, rows: Tensor) -> Tensor:
    rows = rows.to(device=values.device, dtype=torch.bool)
    return values[rows].mean() if bool(rows.any().item()) else values.sum() * 0.0


class FullDurationSTFTMagnitudeLoss(nn.Module):
    """Padding-invariant, one-resolution full-duration magnitude objective."""

    def __init__(
        self,
        *,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 256,
        eps: float = 1.0e-4,
    ) -> None:
        super().__init__()
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.eps = float(eps)
        if min(self.n_fft, self.win_length, self.hop_length) < 1:
            raise ValueError("STFT sizes must be positive")
        if self.win_length > self.n_fft:
            raise ValueError("win_length cannot exceed n_fft")
        self.register_buffer(
            "window",
            torch.hann_window(self.win_length, periodic=True, dtype=torch.float32),
            persistent=False,
        )

    def _stft(self, waveform: Tensor) -> Tensor:
        # Reflect padding requires more than n_fft/2 samples.
        minimum = self.n_fft // 2 + 1
        if waveform.shape[-1] < minimum:
            waveform = F.pad(waveform, (0, minimum - waveform.shape[-1]))
        return torch.stft(
            waveform.float(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=waveform.device),
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )

    def forward(
        self,
        fake: Tensor,
        real: Tensor,
        valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        if fake.ndim == 3 and fake.shape[1] == 1:
            fake = fake[:, 0]
        if real.ndim == 3 and real.shape[1] == 1:
            real = real[:, 0]
        if fake.shape != real.shape or valid_mask.shape != fake.shape:
            raise ValueError("fake, real and valid_mask must share [B,L] shape")
        lengths = valid_mask.long().sum(dim=1)
        adaptive_rows: list[Tensor] = []
        contrast_rows: list[Tensor] = []
        for row, length_value in enumerate(lengths.tolist()):
            length = max(1, int(length_value))
            fake_mag = self._stft(fake[row : row + 1, :length]).abs()
            real_mag = self._stft(real[row : row + 1, :length].detach()).abs()
            fake_std = fake_mag.std(dim=(-2, -1), unbiased=False)
            real_std = real_mag.std(dim=(-2, -1), unbiased=False)
            sigma = torch.sqrt(
                fake_std.square() + real_std.square() + self.eps**2
            ).detach()
            adaptive_rows.append(
                (
                    torch.log1p(fake_mag / sigma[:, None, None])
                    - torch.log1p(real_mag / sigma[:, None, None])
                )
                .abs()
                .mean()
            )
            numerator = torch.linalg.vector_norm(
                fake_mag - real_mag, dim=(-2, -1)
            )
            denominator = torch.sqrt(
                torch.linalg.vector_norm(
                    fake_mag + real_mag, dim=(-2, -1)
                ).square()
                + self.eps**2
            )
            contrast_rows.append((numerator / denominator).mean())
        adaptive = torch.stack(adaptive_rows).mean()
        contrast = torch.stack(contrast_rows).mean()
        return {
            "loss": 0.5 * adaptive + 0.5 * contrast,
            "adaptive_log_magnitude": adaptive,
            "spectral_contrast": contrast,
        }


@dataclass
class XPredConfig(TTALatentAudioConfig):
    """Historical TaskAudio x-pred objective with the proven extension losses."""

    architecture: str = "latent_tta_xpred"
    prediction: str = "x_pred_v_loss"
    global_stft_weight: float = 1.0
    global_stft_n_fft: int = 1024
    global_stft_win_length: int = 1024
    global_stft_hop_length: int = 256
    global_stft_eps: float = 1.0e-4
    visible_condition_noise_mode: str = "none"
    visible_condition_injection_mode: str = "latent_prompt"
    prior_loss_scope: str = "target"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.architecture = "latent_tta_xpred"
        if self.prediction != "x_pred_v_loss":
            raise ValueError("XPredConfig requires prediction='x_pred_v_loss'")
        if float(self.global_stft_eps) <= 0.0:
            raise ValueError("global_stft_eps must be positive")
        if float(self.global_stft_weight) < 0.0:
            raise ValueError("global_stft_weight must be non-negative")
        self.visible_condition_noise_mode = (
            str(self.visible_condition_noise_mode)
            .strip()
            .lower()
            .replace("-", "_")
        )
        if self.visible_condition_noise_mode not in {"none", "flow_matched"}:
            raise ValueError(
                "visible_condition_noise_mode must be 'none' or 'flow_matched'"
            )
        self.visible_condition_injection_mode = (
            str(self.visible_condition_injection_mode)
            .strip()
            .lower()
            .replace("-", "_")
        )
        if self.visible_condition_injection_mode not in {"latent_prompt", "none"}:
            raise ValueError(
                "visible_condition_injection_mode must be 'latent_prompt' or 'none'"
            )
        self.prior_loss_scope = (
            str(self.prior_loss_scope).strip().lower().replace("-", "_")
        )
        if self.prior_loss_scope not in {"target", "all_valid"}:
            raise ValueError(
                "prior_loss_scope must be 'target' or 'all_valid'"
            )


class XPredModel(TTALatentAudio):
    """The proven codec/prior trained with the historical x-pred v-loss."""

    config: XPredConfig

    def __init__(self, config: XPredConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.global_stft_loss = (
            FullDurationSTFTMagnitudeLoss(
                n_fft=config.global_stft_n_fft,
                win_length=config.global_stft_win_length,
                hop_length=config.global_stft_hop_length,
                eps=config.global_stft_eps,
            )
            if config.global_stft_weight > 0.0
            else None
        )

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
        result = super()._masked_prior_flow_loss(
            batch,
            z_clean,
            z_flow_target,
            text_tokens,
            text_lengths=text_lengths,
            mask=mask,
            valid_latent_mask=valid_latent_mask,
            condition_dropout_mask=condition_dropout_mask,
            flow_noise=flow_noise,
            flow_time_override=flow_time_override,
            visible_condition_override=visible_condition_override,
        )
        (
            flow_loss,
            z_x,
            z_condition,
            z_prediction,
            flow_time,
            predicted_x_all,
            metrics,
        ) = result
        steps = int(self.config.flow_steps_per_recon)
        batch_size = int(z_clean.shape[0])
        target = _repeat_steps(z_flow_target.detach(), steps)
        target_mask = _repeat_steps(mask, steps) & _repeat_steps(
            valid_latent_mask, steps
        )
        x_rows = _per_row_masked_mse(predicted_x_all, target, target_mask)
        time = flow_time.reshape(-1, 1, 1)
        denom = self._flow_xpred_denom(time)
        if self.config.prediction == "v_pred_v_loss":
            velocity_mse = (flow_loss.detach() / steps).float()
        else:
            velocity_error = (predicted_x_all - target) / denom
            velocity_rows = _per_row_masked_mse(
                velocity_error, torch.zeros_like(velocity_error), target_mask
            )
            velocity_mse = velocity_rows.mean().detach()

        plan = self._last_mask_plan
        full_rows = torch.zeros(
            batch_size, device=target.device, dtype=torch.bool
        )
        if plan is not None and plan.full_generation_mask is not None:
            full_rows = plan.full_generation_mask.to(
                device=target.device, dtype=torch.bool
            )
        full_flow_rows = _repeat_steps(full_rows, steps)
        ssl_flow_rows = ~full_flow_rows
        metrics.update(
            {
                "prior_x_mse": x_rows.mean().detach(),
                "prior_x_mse_ssl": _row_group_mean(
                    x_rows.detach(), ssl_flow_rows
                ),
                "prior_x_mse_full_generation": _row_group_mean(
                    x_rows.detach(), full_flow_rows
                ),
                "prior_velocity_mse": velocity_mse,
                "predicted_x_rms": predicted_x_all.float()
                .square()
                .mean()
                .sqrt()
                .detach(),
                "target_latent_rms": target.float()
                .square()
                .mean()
                .sqrt()
                .detach(),
            }
        )
        return (
            flow_loss,
            z_x,
            z_condition,
            z_prediction,
            flow_time,
            predicted_x_all,
            metrics,
        )

    def forward(
        self,
        batch: dict[str, Any],
        progress: float | None = None,
    ) -> dict[str, Tensor]:
        output = super().forward(batch, progress=progress)
        group = batch.get("mixture_group")
        if isinstance(group, str):
            for name in (
                "audioset_10",
                "wavcaps_0_5",
                "wavcaps_5_10",
                "wavcaps_10_15",
                "wavcaps_15_20",
            ):
                output[f"data_group_{name}"] = output["loss"].new_tensor(
                    float(group == name)
                )
        local_batch = batch.get("mixture_local_batch_size")
        if isinstance(local_batch, Tensor) and local_batch.numel() == 1:
            output["duration_group_local_batch_size"] = local_batch.to(
                device=output["loss"].device,
                dtype=output["loss"].dtype,
            )
        if self.global_stft_loss is None:
            return output
        fake = output.get("teacher_waveform")
        valid = batch.get("wav_valid_mask")
        if not isinstance(fake, Tensor) or not isinstance(valid, Tensor):
            raise RuntimeError("global STFT requires decoded waveform and valid mask")
        components = self.global_stft_loss(fake, batch["wav_clean"].float(), valid)
        weighted = components["loss"] * float(self.config.global_stft_weight)
        output["loss"] = output["loss"] + weighted
        output["global_stft_loss"] = components["loss"]
        output["weighted_global_stft_loss"] = weighted
        output["global_stft_adaptive_log_magnitude"] = components[
            "adaptive_log_magnitude"
        ]
        output["global_stft_spectral_contrast"] = components["spectral_contrast"]
        return output


@dataclass
class PMFConfig(TTALatentAudioConfig):
    architecture: str = "latent_tta_pmf"
    prediction: str = "pmf_x_v"
    pmf_fm_proportion: float = 0.5
    pmf_norm_p: float = 1.0
    pmf_norm_eps: float = 0.01
    pmf_main_weight: float = 1.0
    pmf_auxiliary_v_weight: float = 1.0
    pmf_min_time: float = 0.02
    global_stft_weight: float = 1.0
    global_stft_n_fft: int = 1024
    global_stft_win_length: int = 1024
    global_stft_hop_length: int = 256
    global_stft_eps: float = 1.0e-4

    def __post_init__(self) -> None:
        requested_prediction = (
            str(self.prediction).strip().lower().replace("-", "_")
        )
        # The inherited speech dataclass validates only its legacy objectives.
        # Validate that layer with a legal placeholder, then restore the new
        # extension-local objective before any module is constructed.
        self.prediction = "x_pred_v_loss"
        super().__post_init__()
        # The parent pins its public architecture label. This subclass owns a
        # distinct checkpoint/inference contract while retaining the old codec.
        self.architecture = "latent_tta_pmf"
        self.prediction = requested_prediction
        if requested_prediction != "pmf_x_v":
            raise ValueError("PMFConfig requires prediction='pmf_x_v'")
        if not 0.0 <= float(self.pmf_fm_proportion) <= 1.0:
            raise ValueError("pmf_fm_proportion must be in [0,1]")
        if min(
            float(self.pmf_norm_eps),
            float(self.pmf_min_time),
            float(self.global_stft_eps),
        ) <= 0.0:
            raise ValueError("pMF/STFT epsilons and min_time must be positive")
        if min(
            float(self.pmf_main_weight),
            float(self.pmf_auxiliary_v_weight),
            float(self.global_stft_weight),
        ) < 0.0:
            raise ValueError("pMF and global STFT weights must be non-negative")


class PMFModel(TTALatentAudio):
    """The proven historical codec with a larger x+v pMF prior."""

    config: PMFConfig

    def __init__(self, config: PMFConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        # fm_head is deliberately the independent, non-RMS-normalized x head.
        self.velocity_head = FMHead(config.fm_hidden_dim, config.latent_dim)
        self.interval_condition = nn.Sequential(
            nn.Linear(4, config.fm_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.fm_hidden_dim, config.fm_hidden_dim),
        )
        self.global_stft_loss = (
            FullDurationSTFTMagnitudeLoss(
                n_fft=config.global_stft_n_fft,
                win_length=config.global_stft_win_length,
                hop_length=config.global_stft_hop_length,
                eps=config.global_stft_eps,
            )
            if config.global_stft_weight > 0.0
            else None
        )

    def _prior_x_v(
        self,
        z_t: Tensor,
        z_condition: Tensor,
        text_tokens: Tensor,
        text_mask: Tensor,
        t: Tensor,
        r: Tensor,
        *,
        valid_mask: Tensor,
        prompt_mask: Tensor,
        condition_dropout_mask: Tensor | None = None,
        global_text_embedding: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        scalars = torch.stack(
            (
                t - r,
                torch.log1p(torch.ones_like(t)),
                torch.zeros_like(t),
                torch.ones_like(t),
            ),
            dim=-1,
        )
        interval_offset = self.interval_condition(
            scalars.to(device=z_t.device, dtype=z_t.dtype)
        )
        hidden = self.fm_encoder(
            z_t,
            z_condition,
            text_tokens,
            t,
            valid_audio_mask=valid_mask,
            text_mask=text_mask,
            prompt_audio_mask=prompt_mask,
            condition_dropout_mask=condition_dropout_mask,
            condition_dropout_mode=self.config.prior_cfg_dropout_mode,
            global_text_embedding=global_text_embedding,
            time_condition_offset=interval_offset,
        )
        valid = valid_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        # Do not route x through latent_norm: the independent head must learn
        # the audio-derived target rather than receiving an RMS shortcut.
        return self.fm_head(hidden) * valid, self.velocity_head(hidden) * valid

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
        batch_size = int(z_clean.shape[0])
        # EMA target is a hard stop-gradient. Visible online latents retain
        # their graph through z_condition, so the prior can shape the encoder.
        target = _repeat_steps(z_flow_target.detach(), steps)
        visible = _repeat_steps(
            z_clean if visible_condition_override is None else visible_condition_override,
            steps,
        )
        mask_flow = _repeat_steps(mask, steps)
        valid_flow = _repeat_steps(valid_latent_mask, steps)
        prompt_flow = (~mask_flow) & valid_flow

        if text_lengths is None:
            text_mask = torch.ones(
                text_tokens.shape[:2], device=text_tokens.device, dtype=torch.bool
            )
        else:
            positions = torch.arange(text_tokens.shape[1], device=text_tokens.device)
            text_mask = positions[None] < text_lengths.to(
                device=text_tokens.device, dtype=torch.long
            )[:, None]
        text_tokens, text_mask = self._replace_dropped_text(
            text_tokens, text_mask, condition_dropout_mask
        )
        text_flow = _repeat_steps(text_tokens, steps)
        text_mask_flow = _repeat_steps(text_mask, steps)
        dropout_flow = (
            None
            if condition_dropout_mask is None
            else _repeat_steps(
                condition_dropout_mask.to(device=z_clean.device, dtype=torch.bool),
                steps,
            )
        )
        captions = batch.get("caption")
        if isinstance(captions, str):
            captions = [captions]
        if not isinstance(captions, Sequence):
            raise TypeError("batch['caption'] must be a sequence")
        global_text = self._encode_clap_captions(
            captions,
            device=z_clean.device,
            dtype=z_clean.dtype,
            dropout_mask=condition_dropout_mask,
        )
        global_text_flow = (
            None if global_text is None else _repeat_steps(global_text, steps)
        )

        noise = torch.randn_like(target) if flow_noise is None else flow_noise
        if noise.shape != target.shape:
            raise ValueError("flow_noise must match repeated target latents")
        if flow_time_override is None:
            # Preserve the historical shifted-logit-normal noise-state
            # distribution while changing to pMF's t=1 noise -> t=0 data.
            old_time = self._sample_flow_time(
                target.shape[0], device=target.device, dtype=target.dtype
            )
            t = 1.0 - old_time
        else:
            t = flow_time_override.to(device=target.device, dtype=target.dtype)
            if t.shape != (target.shape[0],):
                raise ValueError("flow_time_override must have shape [steps * batch]")
        t = t.clamp(0.0, 1.0)

        fm_count = int(round(t.shape[0] * float(self.config.pmf_fm_proportion)))
        permutation = torch.randperm(t.shape[0], device=t.device)
        fm_rows = torch.zeros_like(t, dtype=torch.bool)
        fm_rows[permutation[:fm_count]] = True
        # For non-FM rows sample a strict earlier endpoint r<t. FM rows use
        # r=t exactly, making the compound target ordinary flow matching.
        r_candidate = t * torch.rand_like(t)
        r = torch.where(fm_rows, t, r_candidate)
        t_view = t[:, None, None]
        z_t = (1.0 - t_view) * target + t_view * noise
        target_velocity = noise - target
        z_input = torch.where(prompt_flow.unsqueeze(-1), visible, z_t)
        z_condition = torch.where(
            prompt_flow.unsqueeze(-1), visible, torch.zeros_like(visible)
        )
        valid_values = valid_flow.unsqueeze(-1).to(dtype=z_input.dtype)
        z_input = z_input * valid_values
        z_condition = z_condition * valid_values

        def prior_fn(state: Tensor, time: Tensor, endpoint: Tensor) -> tuple[Tensor, Tensor]:
            return self._prior_x_v(
                state,
                z_condition,
                text_flow,
                text_mask_flow,
                time,
                endpoint,
                valid_mask=valid_flow,
                prompt_mask=prompt_flow,
                condition_dropout_mask=dropout_flow,
                global_text_embedding=global_text_flow,
            )

        initial_x, initial_v = prior_fn(z_input, t, r)

        def average_velocity(
            state: Tensor, time: Tensor, endpoint: Tensor
        ) -> tuple[Tensor, Tensor]:
            x_prediction, velocity_prediction = prior_fn(state, time, endpoint)
            denom = time[:, None, None].clamp_min(
                float(self.config.pmf_min_time)
            )
            return (state - x_prediction) / denom, velocity_prediction

        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.MATH):
            average, derivative, auxiliary_velocity = torch.func.jvp(
                average_velocity,
                (z_input, t, r),
                (
                    initial_v,
                    torch.ones_like(t),
                    torch.zeros_like(r),
                ),
                has_aux=True,
            )
        interval = (t - r)[:, None, None]
        compound = average + interval * derivative.detach()
        target_mask = mask_flow & valid_flow
        main_rows = _per_row_masked_mse(compound, target_velocity, target_mask)
        auxiliary_rows = _per_row_masked_mse(
            auxiliary_velocity, target_velocity, target_mask
        )
        main_adaptive = main_rows / (
            main_rows.detach() + float(self.config.pmf_norm_eps)
        ).pow(float(self.config.pmf_norm_p))
        auxiliary_adaptive = auxiliary_rows / (
            auxiliary_rows.detach() + float(self.config.pmf_norm_eps)
        ).pow(float(self.config.pmf_norm_p))
        total_rows = (
            float(self.config.pmf_main_weight) * main_adaptive
            + float(self.config.pmf_auxiliary_v_weight) * auxiliary_adaptive
        )
        flow_loss = total_rows.mean() * steps

        plan = self._last_mask_plan
        full_rows = torch.zeros(
            batch_size, device=target.device, dtype=torch.bool
        )
        if plan is not None and plan.full_generation_mask is not None:
            full_rows = plan.full_generation_mask.to(
                device=target.device, dtype=torch.bool
            )
        full_flow_rows = _repeat_steps(full_rows, steps)
        visible_flow_rows = ~full_flow_rows
        predicted_x = z_input - t_view * average
        prior_x_rows = _per_row_masked_mse(predicted_x, target, target_mask)
        prior_v_rows = _per_row_masked_mse(
            auxiliary_velocity, target_velocity, target_mask
        )
        metrics = {
            "prior_fm_loss_ssl": _row_group_mean(
                total_rows.detach(), visible_flow_rows
            )
            * steps,
            "prior_fm_loss_full_generation": _row_group_mean(
                total_rows.detach(), full_flow_rows
            )
            * steps,
            "prior_pmf_main_loss": main_adaptive.mean().detach() * steps,
            "prior_auxiliary_v_loss": auxiliary_adaptive.mean().detach() * steps,
            "prior_x_mse": prior_x_rows.mean().detach(),
            "prior_x_mse_ssl": _row_group_mean(
                prior_x_rows.detach(), visible_flow_rows
            ),
            "prior_x_mse_full_generation": _row_group_mean(
                prior_x_rows.detach(), full_flow_rows
            ),
            "prior_velocity_mse": prior_v_rows.mean().detach(),
            "prior_fm_fraction": fm_rows.float().mean().detach(),
            "prior_time_mean": t.float().mean().detach(),
            "prior_endpoint_mean": r.float().mean().detach(),
            "prior_interval_mean": (t - r).float().mean().detach(),
            "predicted_x_rms": predicted_x.float().square().mean().sqrt().detach(),
            "target_latent_rms": target.float().square().mean().sqrt().detach(),
            "caption_tokens_mean": text_mask.float().sum(dim=1).mean().detach(),
        }
        z_x = z_input.view(steps, batch_size, *z_clean.shape[1:])[-1]
        z_cond = z_condition.view(steps, batch_size, *z_clean.shape[1:])[-1]
        z_pred = predicted_x.view(steps, batch_size, *z_clean.shape[1:])[-1]
        flow_time = t.view(steps, batch_size)
        return (
            flow_loss,
            z_x,
            z_cond,
            z_pred,
            flow_time,
            predicted_x,
            metrics,
        )

    def forward(
        self,
        batch: dict[str, Any],
        progress: float | None = None,
    ) -> dict[str, Tensor]:
        output = super().forward(batch, progress=progress)
        group = batch.get("mixture_group")
        if isinstance(group, str):
            for name in (
                "audioset_10",
                "wavcaps_0_5",
                "wavcaps_5_10",
                "wavcaps_10_15",
                "wavcaps_15_20",
            ):
                output[f"data_group_{name}"] = output["loss"].new_tensor(
                    float(group == name)
                )
        local_batch = batch.get("mixture_local_batch_size")
        if isinstance(local_batch, Tensor) and local_batch.numel() == 1:
            output["duration_group_local_batch_size"] = local_batch.to(
                device=output["loss"].device,
                dtype=output["loss"].dtype,
            )
        if self.global_stft_loss is None:
            return output
        fake = output.get("teacher_waveform")
        valid = batch.get("wav_valid_mask")
        if not isinstance(fake, Tensor) or not isinstance(valid, Tensor):
            raise RuntimeError("global STFT requires decoded waveform and valid mask")
        components = self.global_stft_loss(fake, batch["wav_clean"].float(), valid)
        weighted = components["loss"] * float(self.config.global_stft_weight)
        output["loss"] = output["loss"] + weighted
        output["global_stft_loss"] = components["loss"]
        output["weighted_global_stft_loss"] = weighted
        output["global_stft_adaptive_log_magnitude"] = components[
            "adaptive_log_magnitude"
        ]
        output["global_stft_spectral_contrast"] = components["spectral_contrast"]
        return output

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
        **_: Any,
    ) -> Tensor:
        del wave_steps
        if isinstance(captions, str):
            captions = [captions]
        captions = [str(value) for value in captions]
        if not captions:
            raise ValueError("generate requires captions")
        step_count = int(num_steps if prior_steps is None else prior_steps)
        if step_count < 1:
            raise ValueError("prior_steps must be positive")
        if solver not in {"euler", "heun", "rk4"}:
            raise ValueError("pMF sampling supports euler, heun, or rk4")
        parameter = next(self.fm_encoder.parameters())
        device, dtype = parameter.device, parameter.dtype
        sample_count = max(
            1, round(float(seconds) * int(self.config.sample_rate))
        )
        token_count = math.ceil(
            math.ceil(sample_count / int(self.config.patch_size))
            / int(self.config.downsample_factor)
        )
        batch_size = len(captions)
        valid = torch.ones(
            (batch_size, token_count), device=device, dtype=torch.bool
        )
        prompt = torch.zeros_like(valid)
        z_condition = torch.zeros(
            (batch_size, token_count, int(self.config.latent_dim)),
            device=device,
            dtype=dtype,
        )
        no_dropout = torch.zeros(batch_size, device=device, dtype=torch.bool)
        text, text_mask = self.text_conditioner(
            captions, device=device, dropout_mask=no_dropout
        )
        text_mask = text_mask.to(device=device, dtype=torch.bool)
        global_text = self._encode_clap_captions(
            captions, device=device, dtype=dtype
        )
        if float(cfg_strength) > 0.0:
            null_text, null_mask = self._null_text_sequence(
                batch_size,
                text.shape[1],
                device=device,
                dtype=text.dtype,
            )
            cfg_text = torch.cat((text, null_text), dim=0)
            cfg_mask = torch.cat((text_mask, null_mask), dim=0)
            cfg_valid = torch.cat((valid, valid), dim=0)
            cfg_prompt = torch.cat((prompt, prompt), dim=0)
            cfg_condition = torch.cat((z_condition, z_condition), dim=0)
            if global_text is not None:
                null_global = self.clap_conditioner.null_context(
                    batch_size, device=device, dtype=dtype
                )
                cfg_global = torch.cat((global_text, null_global), dim=0)
            else:
                cfg_global = None
        else:
            cfg_text, cfg_mask = text, text_mask
            cfg_valid, cfg_prompt, cfg_condition = valid, prompt, z_condition
            cfg_global = global_text

        old_grid = self._flow_inference_time_grid(
            step_count, device=device, dtype=dtype
        )
        grid = 1.0 - old_grid
        state = torch.randn(
            (batch_size, token_count, int(self.config.latent_dim)),
            device=device,
            dtype=dtype,
            generator=generator,
        )

        def velocity(
            current: Tensor, time_value: float, endpoint_value: float
        ) -> Tensor:
            current_input = (
                current
                if float(cfg_strength) <= 0.0
                else torch.cat((current, current), dim=0)
            )
            time = torch.full(
                (current_input.shape[0],),
                float(time_value),
                device=device,
                dtype=dtype,
            )
            endpoint = torch.full_like(time, float(endpoint_value))
            x_prediction, _ = self._prior_x_v(
                current_input,
                cfg_condition,
                cfg_text,
                cfg_mask,
                time,
                endpoint,
                valid_mask=cfg_valid,
                prompt_mask=cfg_prompt,
                global_text_embedding=cfg_global,
            )
            denom = max(float(self.config.pmf_min_time), float(time_value))
            if float(cfg_strength) <= 0.0:
                return (current - x_prediction) / denom
            x_cond, x_uncond = x_prediction.chunk(2, dim=0)
            v_cond = (current - x_cond) / denom
            v_uncond = (current - x_uncond) / denom
            guided = v_cond + float(cfg_strength) * (v_cond - v_uncond)
            if float(cfg_rescale) > 0.0:
                guided = self._rescale_guided_velocity(
                    guided,
                    v_cond,
                    target_start=0,
                    mix=max(0.0, min(float(cfg_rescale), 1.0)),
                )
            return guided

        for index in range(step_count):
            current_t = float(grid[index].item())
            next_t = float(grid[index + 1].item())
            dt = grid[index + 1] - grid[index]
            if solver == "euler":
                state = state + dt * velocity(state, current_t, next_t)
            elif solver == "heun":
                first = velocity(state, current_t, next_t)
                second = velocity(state + dt * first, next_t, next_t)
                state = state + dt * 0.5 * (first + second)
            else:
                middle = 0.5 * (current_t + next_t)
                half = 0.5 * dt
                k1 = velocity(state, current_t, next_t)
                k2 = velocity(state + half * k1, middle, next_t)
                k3 = velocity(state + half * k2, middle, next_t)
                k4 = velocity(state + dt * k3, next_t, next_t)
                state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return self.decode_tokens_raw(state, original_len=sample_count)


def build_model(
    config: Mapping[str, Any],
    **kwargs: Any,
) -> TTALatentAudio:
    raw = dict(config)
    if "model" in raw and isinstance(raw["model"], Mapping):
        raw = dict(raw["model"])
    architecture = str(raw.get("architecture", "latent_tta_pmf"))
    if architecture == "latent_tta_xpred":
        return build_task_audio_model(
            config,
            model_class=XPredModel,
            config_class=XPredConfig,
            **kwargs,
        )
    if architecture != "latent_tta_pmf":
        raise ValueError(f"unsupported proven architecture: {architecture}")
    return build_task_audio_model(
        config,
        model_class=PMFModel,
        config_class=PMFConfig,
        **kwargs,
    )


__all__ = [
    "FullDurationSTFTMagnitudeLoss",
    "PMFConfig",
    "PMFModel",
    "XPredConfig",
    "XPredModel",
    "build_model",
]
