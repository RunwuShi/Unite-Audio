from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .latent_tts import (
    FMHead,
    MaskedFlowWaveTokenTTS,
    MaskedFlowWaveTokenTTSConfig,
    _downsample_bool_mask,
    _patch_mask_from_sample_mask,
    _repeat_steps,
    _sample_mask,
    _use_fm_adaln,
    masked_mse,
    masked_weighted_mse,
)
from .transformer import TimeEmbedding, TransformerBlock


LatentTTSJointAttentionConfig = MaskedFlowWaveTokenTTSConfig


class JointAttentionFMEncoder(nn.Module):
    """FM encoder with audio/text token-axis joint attention.

    Audio tokens are one GibbsTTS-style sequence: prompt positions hold clean
    prompt latents and target positions hold the noisy flow state. Text tokens
    stay at their original text length and are prepended before audio tokens on
    the sequence axis. Only the audio part is returned for flow prediction.
    """

    def __init__(
        self,
        *,
        token_dim: int,
        hidden_dim: int,
        text_dim: int,
        depth: int,
        adaln_every: int,
        heads: int,
        dim_head: int,
        ffn_mult: int,
        dropout: float,
        rope_base: float,
    ) -> None:
        super().__init__()
        if adaln_every < 1:
            raise ValueError("adaln_every must be positive")
        self.token_dim = int(token_dim)
        self.text_dim = int(text_dim)
        self.audio_proj = nn.Linear(token_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.prompt_type = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.prompt_cfg_type = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.text_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.text_cfg_type = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.time_embed = TimeEmbedding(hidden_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim,
                    heads=heads,
                    dim_head=dim_head,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                    rope_base=rope_base,
                    cond_dim=hidden_dim if _use_fm_adaln(idx, depth, adaln_every) else None,
                )
                for idx in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.prompt_type, std=0.02)
        nn.init.normal_(self.prompt_cfg_type, std=0.02)
        nn.init.normal_(self.text_cfg_type, std=0.02)

    def forward(
        self,
        z_x: Tensor,
        text_tokens: Tensor,
        t: Tensor,
        *,
        valid_audio_mask: Tensor | None = None,
        prompt_audio_mask: Tensor | None = None,
        condition_dropout_mask: Tensor | None = None,
        text_lengths: Tensor | None = None,
    ) -> Tensor:
        if text_tokens.ndim != 3 or text_tokens.shape[0] != z_x.shape[0]:
            raise ValueError("text_tokens must have shape [B, L, text_dim]")

        audio_h = self.audio_proj(z_x) + self.audio_type
        if prompt_audio_mask is not None:
            if prompt_audio_mask.shape != z_x.shape[:2]:
                raise ValueError("prompt_audio_mask must have shape [B, T]")
            prompt_mask = prompt_audio_mask.to(device=z_x.device, dtype=audio_h.dtype)
            audio_h = audio_h + prompt_mask.unsqueeze(-1) * self.prompt_type
        if condition_dropout_mask is not None:
            if condition_dropout_mask.shape != (z_x.shape[0],):
                raise ValueError("condition_dropout_mask must have shape [B]")
            drop = condition_dropout_mask.to(device=z_x.device, dtype=torch.bool)
            if prompt_audio_mask is not None:
                prompt_drop = drop[:, None] & prompt_audio_mask.to(device=z_x.device, dtype=torch.bool)
                audio_h = torch.where(prompt_drop.unsqueeze(-1), self.prompt_cfg_type.to(dtype=audio_h.dtype), audio_h)
        text_h = self.text_proj(text_tokens) + self.text_type
        if condition_dropout_mask is not None:
            drop = condition_dropout_mask.to(device=z_x.device, dtype=torch.bool)
            text_h = torch.where(drop[:, None, None], self.text_cfg_type.to(dtype=text_h.dtype), text_h)
        joint = torch.cat([text_h, audio_h], dim=1)

        joint_mask = self._joint_mask(
            audio_shape=z_x.shape[:2],
            text_shape=text_tokens.shape[:2],
            device=joint.device,
            valid_audio_mask=valid_audio_mask,
            text_lengths=text_lengths,
        )
        if joint_mask is not None:
            joint = joint * joint_mask.to(dtype=joint.dtype).unsqueeze(-1)

        time_cond = self.time_embed(t)
        for block in self.blocks:
            joint = block(joint, time_cond if block.uses_cond else None, mask=joint_mask)
            if joint_mask is not None:
                joint = joint * joint_mask.to(dtype=joint.dtype).unsqueeze(-1)
        return self.norm(joint)[:, text_tokens.shape[1] :]

    def _joint_mask(
        self,
        *,
        audio_shape: torch.Size | tuple[int, int],
        text_shape: torch.Size | tuple[int, int],
        device: torch.device,
        valid_audio_mask: Tensor | None,
        text_lengths: Tensor | None,
    ) -> Tensor | None:
        batch, audio_tokens = int(audio_shape[0]), int(audio_shape[1])
        text_batch, text_tokens = int(text_shape[0]), int(text_shape[1])
        if text_batch != batch:
            raise ValueError("audio/text batch size mismatch")
        if valid_audio_mask is None and text_lengths is None:
            return None

        if valid_audio_mask is None:
            audio_mask = torch.ones((batch, audio_tokens), device=device, dtype=torch.bool)
        else:
            if valid_audio_mask.shape != (batch, audio_tokens):
                raise ValueError("valid_audio_mask must have shape [B, M]")
            audio_mask = valid_audio_mask.to(device=device, dtype=torch.bool)

        if text_lengths is None:
            text_mask = torch.ones((batch, text_tokens), device=device, dtype=torch.bool)
        else:
            lengths = text_lengths.to(device=device, dtype=torch.long).clamp(min=0, max=text_tokens)
            positions = torch.arange(text_tokens, device=device)[None, :]
            text_mask = positions < lengths[:, None]
        return torch.cat([text_mask, audio_mask], dim=1)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        old_weight = state_dict.get(prefix + "input_proj.weight")
        old_bias = state_dict.get(prefix + "input_proj.bias")
        audio_weight_key = prefix + "audio_proj.weight"
        audio_weight = state_dict.get(audio_weight_key)
        if audio_weight is not None and tuple(audio_weight.shape) != tuple(self.audio_proj.weight.shape):
            migrated_audio_weight = self._migrate_old_audio_proj_weight(audio_weight)
            if migrated_audio_weight is not None:
                state_dict[audio_weight_key] = migrated_audio_weight
        if old_weight is not None and prefix + "audio_proj.weight" not in state_dict:
            migrated = self._migrate_old_input_proj_weight(old_weight)
            if migrated is not None:
                audio_weight, text_weight = migrated
                state_dict[prefix + "audio_proj.weight"] = audio_weight
                state_dict[prefix + "text_proj.weight"] = text_weight
                if old_bias is not None:
                    state_dict[prefix + "audio_proj.bias"] = old_bias
                    state_dict[prefix + "text_proj.bias"] = old_bias.new_zeros(self.text_proj.bias.shape)
                state_dict.pop(prefix + "input_proj.weight", None)
                state_dict.pop(prefix + "input_proj.bias", None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _migrate_old_input_proj_weight(self, weight: Tensor) -> tuple[Tensor, Tensor] | None:
        old_audio_text = self.token_dim + self.text_dim
        old_audio_audio_text = self.token_dim * 2 + self.text_dim
        if weight.ndim != 2 or weight.shape[0] != self.audio_proj.weight.shape[0]:
            return None
        if weight.shape[1] == old_audio_audio_text:
            audio_weight = weight[:, : self.token_dim]
            text_weight = weight[:, self.token_dim * 2 :]
        elif weight.shape[1] == old_audio_text:
            audio_weight = weight[:, : self.token_dim]
            text_weight = weight[:, self.token_dim :]
        else:
            return None
        return audio_weight.contiguous(), text_weight.contiguous()

    def _migrate_old_audio_proj_weight(self, weight: Tensor) -> Tensor | None:
        if weight.ndim != 2 or weight.shape[0] != self.audio_proj.weight.shape[0]:
            return None
        if weight.shape[1] == self.token_dim * 2:
            return weight[:, : self.token_dim].contiguous()
        if weight.shape[1] == self.token_dim:
            return weight.contiguous()
        return None


class LatentTTSJointAttention(MaskedFlowWaveTokenTTS):
    def __init__(self, config: LatentTTSJointAttentionConfig) -> None:
        super().__init__(config)
        self.fm_encoder = JointAttentionFMEncoder(
            token_dim=config.latent_dim,
            hidden_dim=config.fm_hidden_dim,
            text_dim=config.text_dim,
            depth=config.fm_depth,
            adaln_every=config.fm_adaln_every,
            heads=config.fm_heads,
            dim_head=config.fm_dim_head,
            ffn_mult=config.fm_ffn_mult,
            dropout=config.dropout,
            rope_base=config.rope_base,
        )
        self.fm_head = FMHead(config.fm_hidden_dim, config.latent_dim)

    def forward(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        wav_clean_raw = batch["wav_clean"].float()
        wav_clean = self._scale_waveform(wav_clean_raw)
        patches, original_len = self.patchify.patchify(wav_clean)
        valid_patch_mask = _patch_mask_from_sample_mask(
            batch.get("wav_valid_mask"),
            patch_size=self.config.patch_size,
            num_patches=patches.shape[1],
            fallback_shape=wav_clean_raw.shape,
        )
        valid_latent_mask = _downsample_bool_mask(
            valid_patch_mask,
            factor=self.config.downsample_factor,
        )
        mask = self._latent_mask(batch, valid_patch_mask, valid_latent_mask)

        patch_tokens = self.token_norm(self.target_encoder(patches))
        z_clean = self._encode_patch_tokens(patch_tokens)
        z_stop = z_clean.detach()
        z_target = z_stop + self.config.target_grad_scale * (z_clean - z_stop)

        text_token_ids, text_lengths = self._condition_text_inputs(
            batch,
            mask=mask,
            valid_latent_mask=valid_latent_mask,
        )
        text_tokens = self._encode_text_ids(text_token_ids, text_lengths=text_lengths)

        steps = self.config.flow_steps_per_recon
        batch_size = z_target.shape[0]
        z_target_flow = _repeat_steps(z_target, steps)
        text_tokens_flow = _repeat_steps(text_tokens, steps)
        text_lengths_flow = _repeat_steps(text_lengths, steps) if isinstance(text_lengths, Tensor) else None
        mask_flow = _repeat_steps(mask, steps)
        valid_latent_mask_flow = _repeat_steps(valid_latent_mask, steps)
        if self.training and self.config.cfg_dropout > 0.0:
            condition_dropout_mask = (
                torch.rand(z_target_flow.shape[0], device=z_target.device)
                < float(self.config.cfg_dropout)
            )
        else:
            condition_dropout_mask = torch.zeros(z_target_flow.shape[0], device=z_target.device, dtype=torch.bool)

        eps = torch.randn_like(z_target_flow)
        t = self._sample_flow_time(
            z_target_flow.shape[0],
            device=z_target.device,
            dtype=z_target.dtype,
        )
        t_view = t[:, None, None]
        z_t = (1.0 - t_view) * eps + t_view * z_target_flow
        v_target = z_target_flow - eps
        prompt_mask_flow = (~mask_flow) & valid_latent_mask_flow
        z_x_all = torch.where(
            prompt_mask_flow.unsqueeze(-1),
            z_target_flow,
            z_t,
        )
        z_prompt_cond_all = torch.where(
            prompt_mask_flow.unsqueeze(-1),
            z_target_flow,
            torch.zeros_like(z_target_flow),
        )

        h = self.fm_encoder(
            z_x_all,
            text_tokens_flow,
            t,
            valid_audio_mask=valid_latent_mask_flow,
            prompt_audio_mask=prompt_mask_flow,
            condition_dropout_mask=condition_dropout_mask,
            text_lengths=text_lengths_flow,
        )
        fm_out = self.fm_head(h)
        if self.config.prediction == "v_pred_v_loss":
            v_pred = fm_out
            z_pred_all = z_t + (1.0 - t_view) * v_pred
            flow_loss = masked_mse(v_pred, v_target, mask_flow & valid_latent_mask_flow) * steps
        elif self.config.prediction == "x_pred_v_loss":
            z_pred_all = self._normalize_flow_x_pred(fm_out)
            denom = self._flow_xpred_denom(t_view)
            v_pred = (z_pred_all - z_t) / denom
            v_target = (z_target_flow - z_t) / denom
            flow_loss = masked_mse(v_pred, v_target, mask_flow & valid_latent_mask_flow) * steps
        elif self.config.prediction == "x_pred_v_loss_weight":
            z_pred_all = self._normalize_flow_x_pred(fm_out)
            denom = self._flow_xpred_denom(t_view.float())
            weight = denom.reciprocal().square()
            flow_loss = masked_weighted_mse(
                z_pred_all,
                z_target_flow,
                mask_flow & valid_latent_mask_flow,
                weight,
            ) * steps
        elif self.config.prediction == "x_pred_x_loss":
            z_pred_all = self._normalize_flow_x_pred(fm_out)
            flow_loss = masked_mse(z_pred_all, z_target_flow, mask_flow & valid_latent_mask_flow) * steps
        else:
            raise ValueError(f"unknown prediction mode: {self.config.prediction}")

        z_x = z_x_all.view(steps, batch_size, *z_target.shape[1:])[-1]
        z_cond = z_prompt_cond_all.view(steps, batch_size, *z_target.shape[1:])[-1]
        z_pred = z_pred_all.view(steps, batch_size, *z_target.shape[1:])[-1]

        valid_sample_mask = _sample_mask(batch.get("wav_valid_mask"), wav_clean_raw.shape)
        (
            teacher_waveform,
            teacher_waveform_model,
            recon_loss,
            mel_loss,
            recon_loss_raw_metric,
            mel_loss_raw_metric,
            decoder_noise_rms,
            decoder_noise_applied_fraction,
            z_decode,
        ) = self._teacher_reconstruction_losses(
            z_clean,
            z_pred,
            wav_clean,
            wav_clean_raw,
            valid_sample_mask,
            valid_latent_mask,
            mask,
            original_len=original_len,
        )

        total_loss = (
            self.config.lambda_flow * flow_loss
            + self.config.lambda_recon * recon_loss
            + self.config.lambda_mel * mel_loss
        )
        return {
            "loss": total_loss,
            "flow_loss": flow_loss,
            "recon_loss": recon_loss,
            "mel_loss": mel_loss,
            "recon_loss_raw_metric": recon_loss_raw_metric,
            "mel_loss_raw_metric": mel_loss_raw_metric,
            "decoder_latent_noise_rms": decoder_noise_rms,
            "decoder_latent_noise_applied_fraction": decoder_noise_applied_fraction,
            "z_decode": z_decode,
            "teacher_waveform": teacher_waveform,
            "teacher_waveform_model": teacher_waveform_model,
            "wav_model": wav_clean,
            "waveform_scale": wav_clean.new_tensor(float(self.config.waveform_scale)),
            "wav_input": z_x,
            "wav_cond": z_cond,
            "flow_time": t.view(steps, batch_size),
            "condition_dropout_fraction": condition_dropout_mask.detach().float().mean(),
            "mask": mask,
            "z_clean": z_clean,
            "z_pred": z_pred,
        }

    @torch.no_grad()
    def generate(
        self,
        prompt_wav: Tensor,
        text_token_ids: Tensor,
        *,
        target_num_tokens: int,
        num_steps: int = 16,
        text_token_lengths: Tensor | None = None,
        cfg_strength: float = 0.0,
        cfg_rescale: float = 0.0,
        solver: str = "euler",
    ) -> Tensor:
        if target_num_tokens < 1:
            raise ValueError("target_num_tokens must be positive")
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        solver = str(solver).lower()
        if solver not in {"euler", "heun", "rk4"}:
            raise ValueError("solver must be 'euler', 'heun', or 'rk4'")
        z_prompt = self.encode_wave(prompt_wav)
        total_tokens = z_prompt.shape[1] + target_num_tokens
        z_state = torch.randn(
            z_prompt.shape[0],
            total_tokens,
            self.config.latent_dim,
            device=z_prompt.device,
            dtype=z_prompt.dtype,
        )
        prompt_tokens = z_prompt.shape[1]
        z_state[:, :prompt_tokens] = z_prompt
        valid_mask = torch.ones(z_state.shape[:2], device=z_state.device, dtype=torch.bool)
        prompt_mask = torch.zeros_like(valid_mask)
        prompt_mask[:, :prompt_tokens] = True
        time_grid = self._flow_inference_time_grid(
            num_steps,
            device=z_prompt.device,
            dtype=z_prompt.dtype,
        )
        if text_token_lengths is not None:
            text_token_lengths = text_token_lengths.to(device=z_prompt.device, dtype=torch.long)
        text_tokens = self._encode_text_ids(text_token_ids, text_lengths=text_token_lengths, device=z_prompt.device)

        def with_prompt(state: Tensor) -> Tensor:
            state = state.clone()
            state[:, :prompt_tokens] = z_prompt
            return state

        def apply_velocity(state: Tensor, velocity: Tensor, scale: Tensor) -> Tensor:
            state = state.clone()
            state[:, prompt_tokens:] = state[:, prompt_tokens:] + scale * velocity[:, prompt_tokens:]
            state[:, :prompt_tokens] = z_prompt
            return state

        def eval_velocity(state: Tensor, time_value: float) -> Tensor:
            state = with_prompt(state)
            t = torch.full((z_prompt.shape[0],), time_value, device=z_prompt.device, dtype=z_prompt.dtype)
            if cfg_strength > 0.0:
                z_state_in = torch.cat((state, state), dim=0)
                text_tokens_in = torch.cat((text_tokens, text_tokens), dim=0)
                t_in = torch.cat((t, t), dim=0)
                valid_mask_in = torch.cat((valid_mask, valid_mask), dim=0)
                prompt_mask_in = torch.cat((prompt_mask, prompt_mask), dim=0)
                if text_token_lengths is None:
                    text_lengths_in = None
                else:
                    text_lengths_in = torch.cat((text_token_lengths, text_token_lengths), dim=0)
                condition_dropout_mask = torch.cat(
                    (
                        torch.zeros(z_prompt.shape[0], device=z_prompt.device, dtype=torch.bool),
                        torch.ones(z_prompt.shape[0], device=z_prompt.device, dtype=torch.bool),
                    ),
                    dim=0,
                )
                h = self.fm_encoder(
                    z_state_in,
                    text_tokens_in,
                    t_in,
                    valid_audio_mask=valid_mask_in,
                    prompt_audio_mask=prompt_mask_in,
                    condition_dropout_mask=condition_dropout_mask,
                    text_lengths=text_lengths_in,
                )
                fm_out_cond, fm_out_uncond = self.fm_head(h).chunk(2, dim=0)
                v_cond = self._fm_output_to_velocity(fm_out_cond, state, time_value)
                v_uncond = self._fm_output_to_velocity(fm_out_uncond, state, time_value)
                v = v_cond + float(cfg_strength) * (v_cond - v_uncond)
                if cfg_rescale > 0.0:
                    cond_std = v_cond.float().std(unbiased=False).clamp_min(1.0e-6)
                    guided_std = v.float().std(unbiased=False).clamp_min(1.0e-6)
                    v_rescaled = v * (cond_std / guided_std).to(dtype=v.dtype)
                    rescale = max(0.0, min(float(cfg_rescale), 1.0))
                    v = rescale * v_rescaled + (1.0 - rescale) * v
            else:
                h = self.fm_encoder(
                    state,
                    text_tokens,
                    t,
                    valid_audio_mask=valid_mask,
                    prompt_audio_mask=prompt_mask,
                    text_lengths=text_token_lengths,
                )
                fm_out = self.fm_head(h)
                v = self._fm_output_to_velocity(fm_out, state, time_value)
            return v

        for step in range(num_steps):
            time_value = float(time_grid[step].item())
            next_time_value = float(time_grid[step + 1].item())
            dt = (time_grid[step + 1] - time_grid[step]).to(dtype=z_state.dtype)
            if solver == "euler":
                k1 = eval_velocity(z_state, time_value)
                z_state = apply_velocity(z_state, k1, dt)
            elif solver == "heun":
                k1 = eval_velocity(z_state, time_value)
                z_euler = apply_velocity(z_state, k1, dt)
                k2 = eval_velocity(z_euler, next_time_value)
                z_state = apply_velocity(z_state, 0.5 * (k1 + k2), dt)
            else:
                mid_time_value = 0.5 * (time_value + next_time_value)
                half_dt = dt * 0.5
                k1 = eval_velocity(z_state, time_value)
                k2 = eval_velocity(apply_velocity(z_state, k1, half_dt), mid_time_value)
                k3 = eval_velocity(apply_velocity(z_state, k2, half_dt), mid_time_value)
                k4 = eval_velocity(apply_velocity(z_state, k3, dt), next_time_value)
                z_state = apply_velocity(z_state, (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0, dt)
        samples_per_latent = self.config.patch_size * self.config.downsample_factor
        prompt_latent_samples = z_prompt.shape[1] * samples_per_latent
        total_len = prompt_latent_samples + target_num_tokens * samples_per_latent
        z_out = z_state.clone()
        z_out[:, :prompt_tokens] = z_prompt
        return self.decode_tokens_raw(z_out, original_len=total_len)

    def _fm_output_to_velocity(self, fm_out: Tensor, z_state: Tensor, time_value: float) -> Tensor:
        if self.config.prediction == "v_pred_v_loss":
            return fm_out
        if self.config.prediction in {"x_pred_v_loss", "x_pred_v_loss_weight", "x_pred_x_loss"}:
            denom = max(float(self.config.flow_xpred_denom_min), 1.0 - time_value)
            z_pred = self._normalize_flow_x_pred(fm_out)
            return (z_pred - z_state) / denom
        raise ValueError(f"unknown prediction mode: {self.config.prediction}")
