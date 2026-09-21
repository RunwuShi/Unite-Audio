from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .istft_decoder import ISTFTDecoderHead
from .latent_tts import (
    FMHead,
    MaskedFlowWaveTokenTTSConfig,
    WavePatchify,
    _patch_mask_from_sample_mask,
    _repeat_steps,
    _sample_mask,
    _sample_mask_from_patch_mask,
    _upsample_bool_mask,
    _use_fm_adaln,
    masked_mse,
    masked_weighted_mse,
)
from .latent_tts_split import LatentTTSSplit
from .transformer import TimeEmbedding, TransformerBlock


LatentTSMegaTTSFMDecoderConfig = MaskedFlowWaveTokenTTSConfig


class WaveFMDecoderHead(nn.Module):
    """Patch-axis flow head conditioned by prior-FM hidden states."""

    def __init__(
        self,
        *,
        patch_size: int,
        cond_dim: int,
        hidden_dim: int,
        depth: int,
        adaln_every: int,
        heads: int,
        dim_head: int,
        ffn_mult: int,
        dropout: float,
        rope_base: float,
        head_type: str = "patch",
        istft_n_fft_factor: int = 4,
        istft_mag_clip: float = 100.0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("wave_fm_depth must be positive")
        if adaln_every < 1:
            raise ValueError("wave_fm_adaln_every must be positive")
        self.patch_size = int(patch_size)
        self.patchify = WavePatchify(self.patch_size)
        self.head_type = str(head_type).lower().replace("-", "_")
        self.input_proj = nn.Linear(self.patch_size + int(cond_dim), int(hidden_dim))
        self.time_embed = TimeEmbedding(int(hidden_dim))
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    int(hidden_dim),
                    heads=heads,
                    dim_head=dim_head,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                    rope_base=rope_base,
                    cond_dim=int(hidden_dim) if _use_fm_adaln(idx, depth, adaln_every) else None,
                )
                for idx in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(int(hidden_dim))
        if self.head_type in {"patch", "wave_patch", "linear_patch"}:
            self.head_type = "patch"
            self.out_proj = nn.Linear(int(hidden_dim), self.patch_size)
        elif self.head_type in {"istft", "stft"}:
            if istft_n_fft_factor < 1:
                raise ValueError("istft_n_fft_factor must be positive")
            self.head_type = "istft"
            self.out_proj = ISTFTDecoderHead(
                dim=int(hidden_dim),
                hop_length=self.patch_size,
                n_fft=self.patch_size * int(istft_n_fft_factor),
                mag_clip=float(istft_mag_clip),
            )
        else:
            raise ValueError("wave FM decoder head must be 'patch' or 'istft'")

    def forward(
        self,
        wave_x: Tensor,
        cond: Tensor,
        t: Tensor,
        *,
        valid_wave_mask: Tensor | None = None,
    ) -> Tensor:
        if wave_x.ndim != 3:
            raise ValueError("wave_x must have shape [B, T_patch, patch_size]")
        if cond.shape[:2] != wave_x.shape[:2]:
            raise ValueError("cond must have shape [B, T_patch, hidden_dim]")
        if t.shape != (wave_x.shape[0],):
            raise ValueError("t must have shape [B]")
        h = self.input_proj(torch.cat([wave_x, cond], dim=-1))
        if valid_wave_mask is not None:
            if valid_wave_mask.shape != wave_x.shape[:2]:
                raise ValueError("valid_wave_mask must have shape [B, T_patch]")
            mask = valid_wave_mask.to(device=wave_x.device, dtype=torch.bool)
            h = h * mask.unsqueeze(-1).to(dtype=h.dtype)
        else:
            mask = None

        time_cond = self.time_embed(t)
        for block in self.blocks:
            h = block(h, time_cond if block.uses_cond else None, mask=mask)
            if mask is not None:
                h = h * mask.unsqueeze(-1).to(dtype=h.dtype)
        h = self.norm(h)
        if self.head_type == "istft":
            wav = self.out_proj(h, original_len=wave_x.shape[1] * self.patch_size)
            out, _ = self.patchify.patchify(wav)
        else:
            out = self.out_proj(h)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(dtype=out.dtype)
        return out


class LatentTSMegaTTSFMDecoder(LatentTTSSplit):
    """MegaTTS split-prior model with a wave-patch FM decoder head."""

    def __init__(self, config: LatentTSMegaTTSFMDecoderConfig) -> None:
        if config.fm_input_mode != "megatts_add":
            raise ValueError("LatentTSMegaTTSFMDecoder requires fm_input_mode='megatts_add'")
        if not config.split_target_encoder_ema:
            raise ValueError("LatentTSMegaTTSFMDecoder requires split_target_encoder_ema=True")
        if float(config.lambda_recon) != 0.0:
            raise ValueError("LatentTSMegaTTSFMDecoder requires lambda_recon=0")

        original_decoder_objective = config.decoder_objective
        if original_decoder_objective == "wave":
            original_decoder_objective = "wave_fm"
        if original_decoder_objective != "wave_fm":
            raise ValueError("LatentTSMegaTTSFMDecoder requires decoder_objective='wave_fm'")

        config.decoder_objective = "wave"
        config.build_deterministic_decoder = False
        super().__init__(config)
        config.decoder_objective = original_decoder_objective
        self.config.decoder_objective = original_decoder_objective
        self.config.build_deterministic_decoder = False
        self.wave_fm_decoder = WaveFMDecoderHead(
            patch_size=config.patch_size,
            cond_dim=config.fm_hidden_dim,
            hidden_dim=config.wave_fm_hidden_dim,
            depth=config.wave_fm_depth,
            adaln_every=config.wave_fm_adaln_every,
            heads=config.wave_fm_heads,
            dim_head=config.wave_fm_dim_head,
            ffn_mult=config.wave_fm_ffn_mult,
            dropout=config.dropout,
            rope_base=config.rope_base,
            head_type=config.decoder_head,
            istft_n_fft_factor=config.decoder_istft_n_fft_factor,
            istft_mag_clip=config.decoder_istft_mag_clip,
        )

    def decode_tokens(self, z: Tensor, original_len: int | None = None) -> Tensor:
        del z, original_len
        raise NotImplementedError("LatentTSMegaTTSFMDecoder has no deterministic decoder; use wave FM sampling")

    def decode_tokens_raw(self, z: Tensor, original_len: int | None = None) -> Tensor:
        del z, original_len
        raise NotImplementedError("LatentTSMegaTTSFMDecoder has no deterministic decoder; use wave FM sampling")

    def forward(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        wav_clean_raw = batch["wav_clean"].float()
        wav_clean = self._scale_waveform(wav_clean_raw)
        patches, original_len = self.patchify.patchify(wav_clean)
        valid_patch_mask = _patch_mask_from_sample_mask(
            batch.get("wav_valid_mask"),
            patch_size=self.config.patch_size,
            num_patches=patches.shape[1],
            fallback_shape=wav_clean_raw.shape,
        ).to(device=patches.device, dtype=torch.bool)
        valid_latent_mask_seed = self._valid_latent_mask_seed(valid_patch_mask)
        split_seed_mask = self._latent_mask(batch, valid_patch_mask, valid_latent_mask_seed)
        z_clean, mask, valid_latent_mask, z_flow_target, prompt_patch_ends = self._encode_split_latents(
            patches,
            valid_patch_mask=valid_patch_mask,
            valid_latent_mask=valid_latent_mask_seed,
            seed_mask=split_seed_mask,
        )

        text_token_ids, text_lengths = self._condition_text_inputs(
            batch,
            mask=mask,
            valid_latent_mask=valid_latent_mask,
            prompt_patch_ends=prompt_patch_ends,
        )
        text_tokens = self._encode_text_ids(text_token_ids, text_lengths=text_lengths)
        if self.training and self.config.cfg_dropout > 0.0:
            condition_dropout_mask = (
                torch.rand(z_clean.shape[0], device=z_clean.device)
                < float(self.config.cfg_dropout)
            )
        else:
            condition_dropout_mask = torch.zeros(z_clean.shape[0], device=z_clean.device, dtype=torch.bool)

        (
            flow_loss,
            z_x,
            z_cond,
            z_pred,
            flow_time,
            h_prior_flow,
            alignment_metrics,
        ) = self._split_flow_loss_with_hidden(
            batch,
            z_flow_target,
            text_tokens,
            text_lengths=text_lengths,
            mask=mask,
            valid_latent_mask=valid_latent_mask,
            condition_dropout_mask=condition_dropout_mask,
        )
        (
            wave_fm_loss,
            wave_pred_patches,
            wave_x,
            wave_target_mask,
            wave_valid_mask,
        ) = self._wave_flow_loss(
            patches,
            h_prior_flow,
            flow_time.reshape(-1),
            latent_target_mask=mask,
            valid_patch_mask=valid_patch_mask,
        )

        pred_patches = torch.where(wave_target_mask.unsqueeze(-1), wave_pred_patches, patches)
        pred_patches = pred_patches * valid_patch_mask.unsqueeze(-1).to(dtype=pred_patches.dtype)
        teacher_waveform_model = self.patchify.unpatchify(pred_patches, original_len=original_len)
        teacher_waveform = self._unscale_waveform(teacher_waveform_model)
        valid_sample_mask = _sample_mask(batch.get("wav_valid_mask"), wav_clean_raw.shape).to(
            device=wav_clean.device,
            dtype=torch.bool,
        )
        target_sample_mask = _sample_mask_from_patch_mask(
            wave_target_mask & wave_valid_mask,
            patch_size=self.config.patch_size,
            original_len=original_len,
        ).to(device=wav_clean.device, dtype=torch.bool)
        zero = wav_clean.new_tensor(0.0)
        mel_loss = (
            self.mel_loss_fn(teacher_waveform_model, wav_clean, sample_mask=target_sample_mask)
            if self.config.lambda_mel != 0.0
            else zero
        )
        total_loss = (
            self.config.lambda_flow * flow_loss
            + self.config.lambda_wave_fm * wave_fm_loss
            + self.config.lambda_mel * mel_loss
        )

        return {
            "loss": total_loss,
            "flow_loss": flow_loss,
            "prior_fm_loss": flow_loss,
            "wave_fm_loss": wave_fm_loss,
            "recon_loss": zero,
            "mel_loss": mel_loss,
            "recon_loss_raw_metric": zero,
            "mel_loss_raw_metric": mel_loss.detach(),
            "decoder_latent_noise_rms": zero,
            "decoder_latent_noise_applied_fraction": zero,
            "z_decode": z_pred,
            "teacher_waveform": teacher_waveform,
            "teacher_waveform_model": teacher_waveform_model,
            "wav_model": wav_clean,
            "waveform_scale": wav_clean.new_tensor(float(self.config.waveform_scale)),
            "wav_input": z_x,
            "wav_cond": z_cond,
            "wave_fm_input": wave_x,
            "wave_fm_pred_patches": wave_pred_patches,
            "wave_target_mask": wave_target_mask,
            "target_sample_mask": target_sample_mask,
            "flow_time": flow_time,
            "condition_dropout_fraction": condition_dropout_mask.detach().float().mean(),
            "mask": mask,
            "valid_latent_mask": valid_latent_mask,
            "z_clean": z_clean,
            "z_flow_target": z_flow_target,
            "split_target_encoder_ema": wav_clean.new_tensor(float(self.config.split_target_encoder_ema)),
            "z_pred": z_pred,
            **alignment_metrics,
        }

    def _valid_latent_mask_seed(self, valid_patch_mask: Tensor) -> Tensor:
        from .latent_tts import _downsample_bool_mask

        return _downsample_bool_mask(
            valid_patch_mask,
            factor=self.config.downsample_factor,
        )

    def _split_flow_loss_with_hidden(
        self,
        batch: dict[str, Any],
        z_clean: Tensor,
        text_tokens: Tensor,
        *,
        text_lengths: Tensor | None,
        mask: Tensor,
        valid_latent_mask: Tensor,
        condition_dropout_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        steps = int(self.config.flow_steps_per_recon)
        batch_size = z_clean.shape[0]
        z_stop = z_clean.detach()
        z_target = z_stop + float(self.config.target_grad_scale) * (z_clean - z_stop)
        z_target_flow = _repeat_steps(z_target, steps)
        mask_flow = _repeat_steps(mask, steps)
        valid_flow = _repeat_steps(valid_latent_mask, steps)
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

        eps = torch.randn_like(z_target_flow)
        t = self._sample_flow_time(
            z_target_flow.shape[0],
            device=z_clean.device,
            dtype=z_clean.dtype,
        )
        t_view = t[:, None, None]
        z_t = (1.0 - t_view) * eps + t_view * z_target_flow
        v_target = z_target_flow - eps

        z_x_all = torch.where(prompt_flow.unsqueeze(-1), z_target_flow, z_t)
        z_cond_all = torch.where(
            prompt_flow.unsqueeze(-1),
            z_target_flow,
            torch.zeros_like(z_target_flow),
        )

        aligned_text, alignment_metrics = self._aligned_text_condition(
            batch,
            text_tokens,
            num_latents=z_clean.shape[1],
            text_lengths=text_lengths,
            valid_latent_mask=valid_latent_mask,
        )
        aligned_text_flow = _repeat_steps(aligned_text, steps)
        h = self.fm_encoder(
            z_x_all,
            z_cond_all,
            aligned_text_flow,
            t,
            valid_audio_mask=valid_flow,
            prompt_audio_mask=prompt_flow,
            condition_dropout_mask=condition_dropout_mask_flow,
        )
        fm_out = self.fm_head(h)
        target_mask = mask_flow & valid_flow
        if self.config.prediction == "x_pred_v_loss":
            z_pred_all = self._normalize_flow_x_pred(fm_out)
            denom = self._flow_xpred_denom(t_view)
            v_pred = (z_pred_all - z_t) / denom
            v_target = (z_target_flow - z_t) / denom
            flow_loss = masked_mse(v_pred, v_target, target_mask) * steps
        elif self.config.prediction == "x_pred_v_loss_weight":
            z_pred_all = self._normalize_flow_x_pred(fm_out)
            denom = self._flow_xpred_denom(t_view.float())
            weight = denom.reciprocal().square()
            flow_loss = masked_weighted_mse(
                z_pred_all,
                z_target_flow,
                target_mask,
                weight,
            ) * steps
        else:
            raise ValueError(f"unsupported prediction mode for split FM decoder: {self.config.prediction}")

        z_x = z_x_all.view(steps, batch_size, *z_clean.shape[1:])[-1]
        z_cond = z_cond_all.view(steps, batch_size, *z_clean.shape[1:])[-1]
        z_pred = z_pred_all.view(steps, batch_size, *z_clean.shape[1:])[-1]
        flow_time = t.view(steps, batch_size)
        return flow_loss, z_x, z_cond, z_pred, flow_time, h, alignment_metrics

    def _wave_flow_loss(
        self,
        patches: Tensor,
        h_prior_flow: Tensor,
        t: Tensor,
        *,
        latent_target_mask: Tensor,
        valid_patch_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size, num_patches, _ = patches.shape
        steps = int(self.config.flow_steps_per_recon)
        if h_prior_flow.shape[0] != batch_size * steps:
            raise ValueError("h_prior_flow batch does not match flow_steps_per_recon")
        if t.shape != (batch_size * steps,):
            raise ValueError("wave FM time tensor must have shape [B * flow_steps]")

        wave_target_mask = _upsample_bool_mask(
            latent_target_mask,
            factor=self.config.downsample_factor,
            length=num_patches,
        ) & valid_patch_mask
        wave_target_mask_flow = _repeat_steps(wave_target_mask, steps)
        valid_patch_flow = _repeat_steps(valid_patch_mask, steps)
        patches_flow = _repeat_steps(patches, steps)
        eps = torch.randn_like(patches_flow)
        t_view = t[:, None, None]
        w_t = (1.0 - t_view) * eps + t_view * patches_flow
        v_target = patches_flow - eps

        prompt_wave_flow = (~wave_target_mask_flow) & valid_patch_flow
        wave_x_all = torch.where(prompt_wave_flow.unsqueeze(-1), patches_flow, w_t)
        wave_x_all = wave_x_all * valid_patch_flow.unsqueeze(-1).to(dtype=wave_x_all.dtype)

        h_wave = h_prior_flow.repeat_interleave(int(self.config.downsample_factor), dim=1)
        if h_wave.shape[1] < num_patches:
            h_wave = F.pad(h_wave, (0, 0, 0, num_patches - h_wave.shape[1]))
        h_wave = h_wave[:, :num_patches]
        wave_pred_all = self.wave_fm_decoder(
            wave_x_all,
            h_wave,
            t,
            valid_wave_mask=valid_patch_flow,
        )
        if self.config.prediction == "x_pred_v_loss":
            denom = self._flow_xpred_denom(t_view)
            v_pred = (wave_pred_all - w_t) / denom
            v_target = (patches_flow - w_t) / denom
            wave_fm_loss = masked_mse(v_pred, v_target, wave_target_mask_flow) * steps
        elif self.config.prediction == "x_pred_v_loss_weight":
            denom = self._flow_xpred_denom(t_view.float())
            weight = denom.reciprocal().square()
            wave_fm_loss = masked_weighted_mse(
                wave_pred_all,
                patches_flow,
                wave_target_mask_flow,
                weight,
            ) * steps
        else:
            raise ValueError(f"unsupported prediction mode for wave FM: {self.config.prediction}")

        wave_pred = wave_pred_all.view(steps, batch_size, num_patches, self.config.patch_size)[-1]
        wave_x = wave_x_all.view(steps, batch_size, num_patches, self.config.patch_size)[-1]
        return wave_fm_loss, wave_pred, wave_x, wave_target_mask, valid_patch_mask

    @torch.no_grad()
    def generate_fm_decoder_prefix(
        self,
        prompt_wav: Tensor,
        text_token_ids: Tensor,
        *,
        target_num_tokens: int,
        num_steps: int,
        sample_batch: dict[str, Any],
        text_token_lengths: Tensor | None = None,
        cfg_strength: float = 0.0,
        cfg_rescale: float = 0.0,
        solver: str = "heun",
    ) -> Tensor:
        solver = str(solver).lower()
        if solver not in {"euler", "heun", "rk4"}:
            raise ValueError("solver must be 'euler', 'heun', or 'rk4'")
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        if target_num_tokens < 1:
            raise ValueError("target_num_tokens must be positive")

        prompt_wav = prompt_wav.to(dtype=torch.float32)
        z_prompt = self.encode_wave(prompt_wav)
        batch_size, prompt_tokens, _ = z_prompt.shape
        total_tokens = prompt_tokens + int(target_num_tokens)
        factor = int(self.config.downsample_factor)
        patch_size = int(self.config.patch_size)
        prompt_patches_count = prompt_tokens * factor
        total_patches = total_tokens * factor
        total_len = total_patches * patch_size

        prompt_patches, _ = self.patchify.patchify(self._scale_waveform(prompt_wav))
        if prompt_patches.shape[1] < prompt_patches_count:
            prompt_patches = F.pad(prompt_patches, (0, 0, 0, prompt_patches_count - prompt_patches.shape[1]))
        prompt_patches = prompt_patches[:, :prompt_patches_count]

        z_state = torch.randn(
            batch_size,
            total_tokens,
            self.config.latent_dim,
            device=z_prompt.device,
            dtype=z_prompt.dtype,
        )
        z_state[:, :prompt_tokens] = z_prompt
        wave_state = torch.randn(
            batch_size,
            total_patches,
            patch_size,
            device=z_prompt.device,
            dtype=z_prompt.dtype,
        )
        wave_state[:, :prompt_patches_count] = prompt_patches.to(device=z_prompt.device, dtype=z_prompt.dtype)

        valid_latent_mask = torch.ones(z_state.shape[:2], device=z_state.device, dtype=torch.bool)
        prompt_latent_mask = torch.zeros_like(valid_latent_mask)
        prompt_latent_mask[:, :prompt_tokens] = True
        z_prompt_full = torch.zeros_like(z_state)
        z_prompt_full[:, :prompt_tokens] = z_prompt

        valid_wave_mask = torch.ones(wave_state.shape[:2], device=wave_state.device, dtype=torch.bool)
        prompt_wave_mask = torch.zeros_like(valid_wave_mask)
        prompt_wave_mask[:, :prompt_patches_count] = True
        wave_prompt_full = torch.zeros_like(wave_state)
        wave_prompt_full[:, :prompt_patches_count] = prompt_patches.to(
            device=wave_state.device,
            dtype=wave_state.dtype,
        )

        text_token_ids = text_token_ids.to(device=z_prompt.device, dtype=torch.long)
        if text_token_lengths is not None:
            text_token_lengths = text_token_lengths.to(device=z_prompt.device, dtype=torch.long)
        sample_batch = self._move_batch_to_device(sample_batch, z_prompt.device)
        text_tokens = self._encode_text_ids(
            text_token_ids,
            text_lengths=text_token_lengths,
            device=z_prompt.device,
        )
        aligned_text, _ = self._aligned_text_condition(
            sample_batch,
            text_tokens,
            num_latents=total_tokens,
            text_lengths=text_token_lengths,
            valid_latent_mask=valid_latent_mask,
        )
        time_grid = self._flow_inference_time_grid(
            num_steps,
            device=z_prompt.device,
            dtype=z_prompt.dtype,
        )

        def with_prompt_z(state: Tensor) -> Tensor:
            state = state.clone()
            state[:, :prompt_tokens] = z_prompt
            return state

        def with_prompt_wave(state: Tensor) -> Tensor:
            state = state.clone()
            state[:, :prompt_patches_count] = wave_prompt_full[:, :prompt_patches_count]
            return state

        def z_velocity_from_x0(z_pred: Tensor, state: Tensor, time_value: float) -> Tensor:
            denom = max(float(self.config.flow_xpred_denom_min), 1.0 - float(time_value))
            return (self._normalize_flow_x_pred(z_pred) - state) / denom

        def wave_velocity_from_x0(wave_pred: Tensor, state: Tensor, time_value: float) -> Tensor:
            denom = max(float(self.config.flow_xpred_denom_min), 1.0 - float(time_value))
            return (wave_pred - state) / denom

        def h_to_wave_cond(h: Tensor) -> Tensor:
            h_wave = h.repeat_interleave(factor, dim=1)
            if h_wave.shape[1] < total_patches:
                h_wave = F.pad(h_wave, (0, 0, 0, total_patches - h_wave.shape[1]))
            return h_wave[:, :total_patches]

        def eval_velocity(z_current: Tensor, wave_current: Tensor, time_value: float) -> tuple[Tensor, Tensor]:
            z_current = with_prompt_z(z_current)
            wave_current = with_prompt_wave(wave_current)
            z_cond = torch.where(prompt_latent_mask.unsqueeze(-1), z_prompt_full, torch.zeros_like(z_current))
            t = torch.full((batch_size,), time_value, device=z_current.device, dtype=z_current.dtype)
            if cfg_strength > 0.0:
                condition_dropout_mask = torch.cat(
                    (
                        torch.zeros(batch_size, device=z_current.device, dtype=torch.bool),
                        torch.ones(batch_size, device=z_current.device, dtype=torch.bool),
                    ),
                    dim=0,
                )
                h = self.fm_encoder(
                    torch.cat((z_current, z_current), dim=0),
                    torch.cat((z_cond, z_cond), dim=0),
                    torch.cat((aligned_text, aligned_text), dim=0),
                    torch.cat((t, t), dim=0),
                    valid_audio_mask=torch.cat((valid_latent_mask, valid_latent_mask), dim=0),
                    prompt_audio_mask=torch.cat((prompt_latent_mask, prompt_latent_mask), dim=0),
                    condition_dropout_mask=condition_dropout_mask,
                )
                z_pred_cond, z_pred_uncond = self.fm_head(h).chunk(2, dim=0)
                z_v_cond = z_velocity_from_x0(z_pred_cond, z_current, time_value)
                z_v_uncond = z_velocity_from_x0(z_pred_uncond, z_current, time_value)
                z_velocity = z_v_cond + float(cfg_strength) * (z_v_cond - z_v_uncond)
                h_cond, h_uncond = h.chunk(2, dim=0)
                wave_pred_cond, wave_pred_uncond = self.wave_fm_decoder(
                    torch.cat((wave_current, wave_current), dim=0),
                    torch.cat((h_to_wave_cond(h_cond), h_to_wave_cond(h_uncond)), dim=0),
                    torch.cat((t, t), dim=0),
                    valid_wave_mask=torch.cat((valid_wave_mask, valid_wave_mask), dim=0),
                ).chunk(2, dim=0)
                wave_v_cond = wave_velocity_from_x0(wave_pred_cond, wave_current, time_value)
                wave_v_uncond = wave_velocity_from_x0(wave_pred_uncond, wave_current, time_value)
                wave_velocity = wave_v_cond + float(cfg_strength) * (wave_v_cond - wave_v_uncond)
                if cfg_rescale > 0.0:
                    mix = max(0.0, min(float(cfg_rescale), 1.0))
                    z_velocity = self._rescale_guided_velocity(
                        z_velocity,
                        z_v_cond,
                        target_start=prompt_tokens,
                        mix=mix,
                    )
                    wave_velocity = self._rescale_guided_velocity(
                        wave_velocity,
                        wave_v_cond,
                        target_start=prompt_patches_count,
                        mix=mix,
                    )
                return z_velocity, wave_velocity

            h = self.fm_encoder(
                z_current,
                z_cond,
                aligned_text,
                t,
                valid_audio_mask=valid_latent_mask,
                prompt_audio_mask=prompt_latent_mask,
                condition_dropout_mask=None,
            )
            z_velocity = z_velocity_from_x0(self.fm_head(h), z_current, time_value)
            wave_pred = self.wave_fm_decoder(
                wave_current,
                h_to_wave_cond(h),
                t,
                valid_wave_mask=valid_wave_mask,
            )
            wave_velocity = wave_velocity_from_x0(wave_pred, wave_current, time_value)
            return z_velocity, wave_velocity

        def apply_velocity(
            z_current: Tensor,
            wave_current: Tensor,
            z_velocity: Tensor,
            wave_velocity: Tensor,
            scale: Tensor,
        ) -> tuple[Tensor, Tensor]:
            z_next = z_current.clone()
            wave_next = wave_current.clone()
            z_next[:, prompt_tokens:] = z_next[:, prompt_tokens:] + scale * z_velocity[:, prompt_tokens:]
            wave_next[:, prompt_patches_count:] = (
                wave_next[:, prompt_patches_count:]
                + scale.to(dtype=wave_next.dtype) * wave_velocity[:, prompt_patches_count:]
            )
            return with_prompt_z(z_next), with_prompt_wave(wave_next)

        for step_idx in range(num_steps):
            time_value = float(time_grid[step_idx].item())
            next_time_value = float(time_grid[step_idx + 1].item())
            dt = (time_grid[step_idx + 1] - time_grid[step_idx]).to(dtype=z_state.dtype)
            if solver == "euler":
                k1_z, k1_w = eval_velocity(z_state, wave_state, time_value)
                z_state, wave_state = apply_velocity(z_state, wave_state, k1_z, k1_w, dt)
            elif solver == "heun":
                k1_z, k1_w = eval_velocity(z_state, wave_state, time_value)
                z_euler, w_euler = apply_velocity(z_state, wave_state, k1_z, k1_w, dt)
                k2_z, k2_w = eval_velocity(z_euler, w_euler, next_time_value)
                z_state, wave_state = apply_velocity(
                    z_state,
                    wave_state,
                    0.5 * (k1_z + k2_z),
                    0.5 * (k1_w + k2_w),
                    dt,
                )
            else:
                mid_time_value = 0.5 * (time_value + next_time_value)
                half_dt = dt * 0.5
                k1_z, k1_w = eval_velocity(z_state, wave_state, time_value)
                z2, w2 = apply_velocity(z_state, wave_state, k1_z, k1_w, half_dt)
                k2_z, k2_w = eval_velocity(z2, w2, mid_time_value)
                z3, w3 = apply_velocity(z_state, wave_state, k2_z, k2_w, half_dt)
                k3_z, k3_w = eval_velocity(z3, w3, mid_time_value)
                z4, w4 = apply_velocity(z_state, wave_state, k3_z, k3_w, dt)
                k4_z, k4_w = eval_velocity(z4, w4, next_time_value)
                z_state, wave_state = apply_velocity(
                    z_state,
                    wave_state,
                    (k1_z + 2.0 * k2_z + 2.0 * k3_z + k4_z) / 6.0,
                    (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w) / 6.0,
                    dt,
                )

        generated_model = self.patchify.unpatchify(with_prompt_wave(wave_state), original_len=total_len)
        return self._unscale_waveform(generated_model)

    @staticmethod
    def _rescale_guided_velocity(guided: Tensor, cond: Tensor, *, target_start: int, mix: float) -> Tensor:
        cond_std = cond[:, target_start:].float().std(unbiased=False).clamp_min(1.0e-6)
        guided_std = guided[:, target_start:].float().std(unbiased=False).clamp_min(1.0e-6)
        rescaled = guided * (cond_std / guided_std).to(dtype=guided.dtype)
        return mix * rescaled + (1.0 - mix) * guided

    @staticmethod
    def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, Tensor):
                out[key] = value.to(device=device)
            else:
                out[key] = value
        return out
