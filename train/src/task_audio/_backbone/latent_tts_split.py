from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .latent_tts import (
    FMHead,
    MaskedFlowWaveTokenTTS,
    MaskedFlowWaveTokenTTSConfig,
    _downsample_bool_mask,
    _patch_mask_from_sample_mask,
    _repeat_steps,
    _sample_mask,
    _upsample_bool_mask,
    masked_mse,
    masked_weighted_mse,
)
from .latent_tts_joint_att import JointAttentionFMEncoder
from .latent_tts_megatts import ChannelConcatFMEncoder, MegaTTSAddFMEncoder, MegaTTSTextConditioner


LatentTTSSplitConfig = MaskedFlowWaveTokenTTSConfig


class LatentTTSSplit(MaskedFlowWaveTokenTTS):
    """Patch-split encoder path for joint latent prior training.

    The base model encodes the full utterance before masking. This variant
    chooses the prefix/target split in patch space first, then encodes the two
    segments independently so target positions cannot leak through encoder
    self-attention.
    """

    def __init__(self, config: LatentTTSSplitConfig) -> None:
        super().__init__(config)
        if self.config.mask_source not in {"prefix", "random_span"}:
            raise ValueError("LatentTTSSplit currently supports mask_source='prefix' or 'random_span'")
        if self.config.mask_source == "random_span" and self.config.use_target_token:
            raise ValueError("LatentTTSSplit random_span does not support use_target_token")
        if self.config.decoder_objective != "wave":
            raise ValueError("LatentTTSSplit starts from deterministic decoder_objective='wave'")
        if self.config.prediction not in {"x_pred_v_loss", "x_pred_v_loss_weight"}:
            raise ValueError(
                "LatentTTSSplit first pass requires prediction='x_pred_v_loss' "
                "or 'x_pred_v_loss_weight'"
            )
        if int(self.config.flow_steps_per_recon) < 1:
            raise ValueError("LatentTTSSplit requires flow_steps_per_recon >= 1")
        self.megatts_text_conditioner: MegaTTSTextConditioner | None = None
        self.latent_text_projector: nn.Module | None = None
        if self.config.fm_input_mode == "joint_seq":
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
        else:
            self.megatts_text_conditioner = MegaTTSTextConditioner(
                text_dim=config.text_dim,
                hidden_dim=config.fm_hidden_dim,
                downsample_factor=config.downsample_factor,
                alignment_mode=config.megatts_alignment_mode,
                anchor_mode=config.megatts_anchor_mode,
                anchor_ratio=config.megatts_anchor_ratio,
            )
            self.latent_text_projector = nn.Linear(config.text_dim, config.fm_hidden_dim)
            if self.config.fm_input_mode == "megatts_add":
                self.fm_encoder = MegaTTSAddFMEncoder(
                    token_dim=config.latent_dim,
                    hidden_dim=config.fm_hidden_dim,
                    depth=config.fm_depth,
                    adaln_every=config.fm_adaln_every,
                    heads=config.fm_heads,
                    dim_head=config.fm_dim_head,
                    ffn_mult=config.fm_ffn_mult,
                    dropout=config.dropout,
                    rope_base=config.rope_base,
                )
            elif self.config.fm_input_mode == "channel_concat":
                self.fm_encoder = ChannelConcatFMEncoder(
                    token_dim=config.latent_dim,
                    hidden_dim=config.fm_hidden_dim,
                    depth=config.fm_depth,
                    adaln_every=config.fm_adaln_every,
                    heads=config.fm_heads,
                    dim_head=config.fm_dim_head,
                    ffn_mult=config.fm_ffn_mult,
                    dropout=config.dropout,
                    rope_base=config.rope_base,
                )
            else:
                raise ValueError(f"unsupported fm_input_mode: {self.config.fm_input_mode}")
        self.fm_head = FMHead(config.fm_hidden_dim, config.latent_dim)
        self.target_encoder_ema: nn.Module | None = None
        self.token_norm_ema: nn.Module | None = None
        self.encoder_downsamples_ema: nn.Module | None = None
        self.encoder_down_blocks_ema: nn.Module | None = None
        self.encoder_latent_norm_ema: nn.Module | None = None
        self.encoder_to_latent_ema: nn.Module | None = None
        self.latent_norm_ema: nn.Module | None = None
        if self.config.split_target_encoder_ema:
            self.target_encoder_ema = deepcopy(self.target_encoder)
            self.token_norm_ema = deepcopy(self.token_norm)
            self.encoder_downsamples_ema = deepcopy(self.encoder_downsamples)
            self.encoder_down_blocks_ema = deepcopy(self.encoder_down_blocks)
            self.encoder_latent_norm_ema = deepcopy(self.encoder_latent_norm)
            self.encoder_to_latent_ema = deepcopy(self.encoder_to_latent)
            self.latent_norm_ema = deepcopy(self.latent_norm)
            for module in self._target_encoder_ema_modules():
                module.requires_grad_(False)
                module.eval()

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
        valid_latent_mask_seed = _downsample_bool_mask(
            valid_patch_mask,
            factor=self.config.downsample_factor,
        )
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

        flow_loss, z_x, z_cond, z_pred, flow_time, alignment_metrics = (
            self._split_flow_loss(
                batch,
                z_flow_target,
                text_tokens,
                text_lengths=text_lengths,
                mask=mask,
                valid_latent_mask=valid_latent_mask,
                condition_dropout_mask=condition_dropout_mask,
            )
        )

        valid_sample_mask = _sample_mask(batch.get("wav_valid_mask"), wav_clean_raw.shape).to(
            device=wav_clean.device,
            dtype=torch.bool,
        )
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
            z_recon_gt=z_flow_target,
            z_fm_input=z_x,
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

    def _encode_split_latents(
        self,
        patches: Tensor,
        *,
        valid_patch_mask: Tensor,
        valid_latent_mask: Tensor,
        seed_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if self.config.mask_source == "random_span":
            return self._encode_random_span_split_latents(
                patches,
                valid_patch_mask=valid_patch_mask,
                valid_latent_mask=valid_latent_mask,
                seed_mask=seed_mask,
            )

        factor = int(self.config.downsample_factor)
        base_max_latents = valid_latent_mask.shape[1]
        valid_patch_counts = valid_patch_mask.long().sum(dim=1)
        visible_latents = ((~seed_mask) & valid_latent_mask).long().sum(dim=1)

        prefix_patch_counts = torch.minimum(valid_patch_counts, visible_latents * factor)
        prefix_patch_counts = self._align_prefix_patch_counts_to_downsample_grid(
            prefix_patch_counts,
            valid_patch_counts,
            downsample_factor=factor,
        )
        prefix_patch_counts = torch.where(
            visible_latents > 0,
            prefix_patch_counts,
            torch.zeros_like(prefix_patch_counts),
        )
        target_patch_counts = (valid_patch_counts - prefix_patch_counts).clamp_min(0)
        segment_patch_counts = torch.cat((prefix_patch_counts, target_patch_counts), dim=0)
        max_segment_patches = int(segment_patch_counts.max().item()) if segment_patch_counts.numel() else 0

        batch_size, _, patch_dim = patches.shape
        if max_segment_patches > 0:
            segment_positions = torch.arange(max_segment_patches, device=patches.device)
            prefix_mask = segment_positions[None, :] < prefix_patch_counts[:, None]
            target_mask = segment_positions[None, :] < target_patch_counts[:, None]

            prefix_indices = segment_positions[None, :].expand(batch_size, -1)
            prefix_indices = prefix_indices.clamp(max=max(patches.shape[1] - 1, 0))
            prefix_segments = patches.gather(
                1,
                prefix_indices.unsqueeze(-1).expand(-1, -1, patch_dim),
            )

            target_indices = prefix_patch_counts[:, None] + segment_positions[None, :]
            target_indices = target_indices.clamp(max=max(patches.shape[1] - 1, 0))
            target_segments = patches.gather(
                1,
                target_indices.unsqueeze(-1).expand(-1, -1, patch_dim),
            )

            segments = torch.cat((prefix_segments, target_segments), dim=0)
            segment_mask = torch.cat((prefix_mask, target_mask), dim=0)
            segments = segments * segment_mask.to(dtype=segments.dtype).unsqueeze(-1)
        else:
            segments = patches.new_zeros((batch_size * 2, 0, patch_dim))
            segment_mask = torch.zeros((batch_size * 2, 0), device=patches.device, dtype=torch.bool)

        z_segments, latent_segment_mask = self._encode_patch_segments_batch(segments, segment_mask)
        z_prefix_segments = z_segments[: patches.shape[0]]
        z_target_segments = z_segments[patches.shape[0] :]
        prefix_latent_counts = latent_segment_mask[: patches.shape[0]].long().sum(dim=1)
        target_latent_counts = latent_segment_mask[patches.shape[0] :].long().sum(dim=1)
        z_target_flow_segments = z_target_segments
        if self.config.split_target_encoder_ema:
            with torch.no_grad():
                z_target_flow_segments, _ = self._encode_patch_segments_batch(
                    segments[patches.shape[0] :],
                    segment_mask[patches.shape[0] :],
                    use_target_ema=True,
                )

        max_latents = int(base_max_latents)
        total_latent_counts = (prefix_latent_counts + target_latent_counts).clamp(max=max_latents)
        latent_positions = torch.arange(max_latents, device=patches.device)
        valid = latent_positions[None, :] < total_latent_counts[:, None]
        prefix_region = latent_positions[None, :] < prefix_latent_counts[:, None]
        target_region = valid & ~prefix_region

        z_clean = z_prefix_segments.new_zeros((batch_size, max_latents, self.config.latent_dim))
        z_flow_target = z_clean.clone()
        if max_latents > 0 and z_prefix_segments.shape[1] > 0:
            prefix_indices = latent_positions[None, :].expand(batch_size, -1)
            prefix_indices = prefix_indices.clamp(max=z_prefix_segments.shape[1] - 1)
            gathered_prefix = z_prefix_segments.gather(
                1,
                prefix_indices.unsqueeze(-1).expand(-1, -1, self.config.latent_dim),
            )
            z_clean = torch.where(prefix_region.unsqueeze(-1), gathered_prefix, z_clean)
            z_flow_target = torch.where(prefix_region.unsqueeze(-1), gathered_prefix, z_flow_target)
        if max_latents > 0 and z_target_segments.shape[1] > 0:
            target_indices = latent_positions[None, :] - prefix_latent_counts[:, None]
            target_indices = target_indices.clamp(min=0, max=z_target_segments.shape[1] - 1)
            target_gather = target_indices.unsqueeze(-1).expand(-1, -1, self.config.latent_dim)
            gathered_target = z_target_segments.gather(1, target_gather)
            gathered_target_flow = z_target_flow_segments.gather(1, target_gather)
            z_clean = torch.where(target_region.unsqueeze(-1), gathered_target, z_clean)
            z_flow_target = torch.where(target_region.unsqueeze(-1), gathered_target_flow, z_flow_target)
        mask = target_region
        return z_clean, mask & valid, valid, z_flow_target, prefix_patch_counts

    def _encode_random_span_split_latents(
        self,
        patches: Tensor,
        *,
        valid_patch_mask: Tensor,
        valid_latent_mask: Tensor,
        seed_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        factor = int(self.config.downsample_factor)
        batch_size, num_patches, patch_dim = patches.shape
        max_latents = int(valid_latent_mask.shape[1])
        valid_patch_counts = valid_patch_mask.long().sum(dim=1)
        target_region = seed_mask.to(device=patches.device, dtype=torch.bool) & valid_latent_mask.to(
            device=patches.device,
            dtype=torch.bool,
        )

        latent_positions = torch.arange(max_latents, device=patches.device)
        has_target = target_region.any(dim=1)
        start_fallback = torch.full((batch_size, max_latents), max_latents, device=patches.device, dtype=torch.long)
        target_latent_starts = torch.where(target_region, latent_positions[None, :], start_fallback).amin(dim=1)
        target_latent_ends = torch.where(
            target_region,
            latent_positions[None, :] + 1,
            torch.zeros((batch_size, max_latents), device=patches.device, dtype=torch.long),
        ).amax(dim=1)
        target_latent_starts = torch.where(has_target, target_latent_starts, torch.zeros_like(target_latent_starts))
        target_latent_ends = torch.where(has_target, target_latent_ends, torch.zeros_like(target_latent_ends))

        target_patch_starts = (target_latent_starts * factor).clamp(max=num_patches)
        target_patch_ends = torch.minimum((target_latent_ends * factor).clamp(max=num_patches), valid_patch_counts)
        target_patch_counts = torch.where(
            has_target,
            (target_patch_ends - target_patch_starts).clamp_min(0),
            torch.zeros_like(target_patch_ends),
        )

        masked_patch_mask = _upsample_bool_mask(target_region, factor=factor, length=num_patches) & valid_patch_mask
        condition_patches = patches.masked_fill(masked_patch_mask.unsqueeze(-1), 0.0)
        z_condition, condition_latent_mask = self._encode_patch_segments_batch(
            condition_patches,
            valid_patch_mask,
        )
        z_condition = z_condition[:, :max_latents]
        condition_latent_mask = condition_latent_mask[:, :max_latents]

        max_target_patches = int(target_patch_counts.max().item()) if target_patch_counts.numel() else 0
        if max_target_patches > 0:
            segment_positions = torch.arange(max_target_patches, device=patches.device)
            target_segment_mask = segment_positions[None, :] < target_patch_counts[:, None]
            target_indices = target_patch_starts[:, None] + segment_positions[None, :]
            target_indices = target_indices.clamp(max=max(num_patches - 1, 0))
            target_segments = patches.gather(
                1,
                target_indices.unsqueeze(-1).expand(-1, -1, patch_dim),
            )
            target_segments = target_segments * target_segment_mask.to(dtype=target_segments.dtype).unsqueeze(-1)
        else:
            target_segments = patches.new_zeros((batch_size, 0, patch_dim))
            target_segment_mask = torch.zeros((batch_size, 0), device=patches.device, dtype=torch.bool)

        z_target_segments, _ = self._encode_patch_segments_batch(target_segments, target_segment_mask)
        z_target_flow_segments = z_target_segments
        if self.config.split_target_encoder_ema:
            with torch.no_grad():
                z_target_flow_segments, _ = self._encode_patch_segments_batch(
                    target_segments,
                    target_segment_mask,
                    use_target_ema=True,
                )

        valid = condition_latent_mask & valid_latent_mask
        target_region = target_region & valid
        z_clean = z_condition.new_zeros((batch_size, max_latents, self.config.latent_dim))
        z_flow_target = z_clean.clone()
        condition_region = valid & ~target_region
        z_clean = torch.where(condition_region.unsqueeze(-1), z_condition, z_clean)
        z_flow_target = torch.where(condition_region.unsqueeze(-1), z_condition, z_flow_target)

        if z_target_segments.shape[1] > 0:
            target_indices = latent_positions[None, :] - target_latent_starts[:, None]
            target_indices = target_indices.clamp(min=0, max=z_target_segments.shape[1] - 1)
            target_gather = target_indices.unsqueeze(-1).expand(-1, -1, self.config.latent_dim)
            gathered_target = z_target_segments.gather(1, target_gather)
            gathered_target_flow = z_target_flow_segments.gather(1, target_gather)
            z_clean = torch.where(target_region.unsqueeze(-1), gathered_target, z_clean)
            z_flow_target = torch.where(target_region.unsqueeze(-1), gathered_target_flow, z_flow_target)

        return z_clean, target_region, valid, z_flow_target, target_patch_starts

    @staticmethod
    def _align_prefix_patch_counts_to_downsample_grid(
        prefix_patch_counts: Tensor,
        valid_patch_counts: Tensor,
        *,
        downsample_factor: int,
    ) -> Tensor:
        factor = max(1, int(downsample_factor))
        if factor == 1:
            return torch.minimum(prefix_patch_counts, (valid_patch_counts - 1).clamp_min(0))
        max_prefix = ((valid_patch_counts - 1).clamp_min(0) // factor) * factor
        snapped = (prefix_patch_counts // factor) * factor
        snapped = torch.minimum(snapped, max_prefix)
        min_prefix = torch.where(
            valid_patch_counts > factor,
            torch.full_like(valid_patch_counts, factor),
            torch.zeros_like(valid_patch_counts),
        )
        return torch.maximum(snapped, min_prefix)

    def _encode_patch_segment(self, segment: Tensor) -> Tensor:
        if segment.shape[1] == 0:
            return segment.new_zeros((segment.shape[0], 0, self.config.latent_dim))
        valid_mask = torch.ones(segment.shape[:2], device=segment.device, dtype=torch.bool)
        z, _ = self._encode_patch_segments_batch(segment, valid_mask)
        return z

    def _encode_patch_segments_batch(
        self,
        segments: Tensor,
        valid_patch_mask: Tensor,
        *,
        use_target_ema: bool = False,
    ) -> tuple[Tensor, Tensor]:
        if segments.shape[1] == 0:
            z = segments.new_zeros((segments.shape[0], 0, self.config.latent_dim))
            mask = torch.zeros((segments.shape[0], 0), device=segments.device, dtype=torch.bool)
            return z, mask

        mask = valid_patch_mask.to(device=segments.device, dtype=torch.bool)
        attn_mask = self._nonempty_attention_mask(mask)
        (
            target_encoder,
            token_norm,
            encoder_downsamples,
            encoder_down_blocks,
            encoder_latent_norm,
            encoder_to_latent,
            latent_norm,
        ) = self._encoder_modules(use_target_ema=use_target_ema)
        h = target_encoder.in_proj(segments)
        h = h * mask.to(dtype=h.dtype).unsqueeze(-1)
        for block in target_encoder.blocks:
            h = block(h, mask=attn_mask)
            h = h * mask.to(dtype=h.dtype).unsqueeze(-1)
        h = token_norm(target_encoder.norm(h))
        h = h * mask.to(dtype=h.dtype).unsqueeze(-1)

        for downsample, blocks in zip(encoder_downsamples, encoder_down_blocks, strict=True):
            h = downsample(h)
            mask = _downsample_bool_mask(mask, factor=downsample.factor)
            attn_mask = self._nonempty_attention_mask(mask)
            h = h * mask.to(dtype=h.dtype).unsqueeze(-1)
            for block in blocks:
                h = block(h, mask=attn_mask)
                h = h * mask.to(dtype=h.dtype).unsqueeze(-1)

        h = encoder_latent_norm(h)
        z = latent_norm(encoder_to_latent(h))
        z = z * mask.to(dtype=z.dtype).unsqueeze(-1)
        return z, mask

    def _encoder_modules(
        self,
        *,
        use_target_ema: bool,
    ) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module, nn.Module, nn.Module]:
        if use_target_ema:
            if not self.config.split_target_encoder_ema or self.target_encoder_ema is None:
                raise RuntimeError("target encoder EMA is not enabled")
            for module in self._target_encoder_ema_modules():
                module.eval()
            return (
                self.target_encoder_ema,
                self.token_norm_ema,
                self.encoder_downsamples_ema,
                self.encoder_down_blocks_ema,
                self.encoder_latent_norm_ema,
                self.encoder_to_latent_ema,
                self.latent_norm_ema,
            )
        return (
            self.target_encoder,
            self.token_norm,
            self.encoder_downsamples,
            self.encoder_down_blocks,
            self.encoder_latent_norm,
            self.encoder_to_latent,
            self.latent_norm,
        )

    def _target_encoder_ema_modules(self) -> tuple[nn.Module, ...]:
        modules = (
            self.target_encoder_ema,
            self.token_norm_ema,
            self.encoder_downsamples_ema,
            self.encoder_down_blocks_ema,
            self.encoder_latent_norm_ema,
            self.encoder_to_latent_ema,
            self.latent_norm_ema,
        )
        return tuple(module for module in modules if module is not None)

    @torch.no_grad()
    def update_internal_ema(self, step: int) -> None:
        if not self.config.split_target_encoder_ema:
            return
        if step < int(self.config.split_target_encoder_ema_start_step):
            decay = 0.0
        else:
            decay = float(self.config.split_target_encoder_ema_decay)
        for online, ema in self._target_encoder_online_ema_pairs():
            self._update_ema_module(ema, online, decay=decay)

    def _target_encoder_online_ema_pairs(self) -> tuple[tuple[nn.Module, nn.Module], ...]:
        if not self.config.split_target_encoder_ema:
            return ()
        return (
            (self.target_encoder, self.target_encoder_ema),
            (self.token_norm, self.token_norm_ema),
            (self.encoder_downsamples, self.encoder_downsamples_ema),
            (self.encoder_down_blocks, self.encoder_down_blocks_ema),
            (self.encoder_latent_norm, self.encoder_latent_norm_ema),
            (self.encoder_to_latent, self.encoder_to_latent_ema),
            (self.latent_norm, self.latent_norm_ema),
        )

    @staticmethod
    def _update_ema_module(ema: nn.Module, online: nn.Module, *, decay: float) -> None:
        online_state = online.state_dict()
        ema_state = ema.state_dict()
        for key, ema_value in ema_state.items():
            value = online_state[key].detach().to(device=ema_value.device)
            if torch.is_floating_point(ema_value):
                ema_value.mul_(decay).add_(value.to(dtype=ema_value.dtype), alpha=1.0 - decay)
            else:
                ema_value.copy_(value)

    @staticmethod
    def _nonempty_attention_mask(mask: Tensor) -> Tensor:
        if mask.shape[1] == 0:
            return mask
        safe_mask = mask.clone()
        empty_rows = ~safe_mask.any(dim=1)
        safe_mask[empty_rows, 0] = True
        return safe_mask

    def _normalize_flow_x_pred(self, z_pred: Tensor) -> Tensor:
        return z_pred

    def _split_flow_loss(
        self,
        batch: dict[str, Any],
        z_clean: Tensor,
        text_tokens: Tensor,
        *,
        text_lengths: Tensor | None,
        mask: Tensor,
        valid_latent_mask: Tensor,
        condition_dropout_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
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

        if self.config.fm_input_mode == "joint_seq":
            text_tokens_flow = _repeat_steps(text_tokens, steps)
            text_lengths_flow = _repeat_steps(text_lengths, steps) if isinstance(text_lengths, Tensor) else None
            h = self.fm_encoder(
                z_x_all,
                text_tokens_flow,
                t,
                valid_audio_mask=valid_flow,
                prompt_audio_mask=prompt_flow,
                condition_dropout_mask=condition_dropout_mask_flow,
                text_lengths=text_lengths_flow,
            )
            alignment_metrics = self._zero_alignment_metrics(z_clean)
        else:
            aligned_text, alignment_metrics = self._aligned_text_condition(
                batch,
                text_tokens,
                num_latents=z_clean.shape[1],
                text_lengths=text_lengths,
                valid_latent_mask=valid_latent_mask,
            )
            aligned_text_flow = _repeat_steps(aligned_text, steps)
            if self.config.fm_input_mode == "megatts_add":
                h = self.fm_encoder(
                    z_x_all,
                    z_cond_all,
                    aligned_text_flow,
                    t,
                    valid_audio_mask=valid_flow,
                    prompt_audio_mask=prompt_flow,
                    condition_dropout_mask=condition_dropout_mask_flow,
                )
            elif self.config.fm_input_mode == "channel_concat":
                h = self.fm_encoder(
                    z_x_all,
                    z_cond_all,
                    aligned_text_flow,
                    t,
                    valid_audio_mask=valid_flow,
                    condition_dropout_mask=condition_dropout_mask_flow,
                )
            else:
                raise ValueError(f"unknown fm_input_mode: {self.config.fm_input_mode}")
        fm_out = self.fm_head(h)
        target_mask = mask_flow & valid_flow
        if self.config.prediction == "v_pred_v_loss":
            v_pred = fm_out
            z_pred_all = z_t + (1.0 - t_view) * v_pred
            flow_loss = masked_mse(v_pred, v_target, target_mask) * steps
        elif self.config.prediction == "x_pred_v_loss":
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
        elif self.config.prediction == "x_pred_x_loss":
            z_pred_all = self._normalize_flow_x_pred(fm_out)
            flow_loss = masked_mse(z_pred_all, z_target_flow, target_mask) * steps
        else:
            raise ValueError(f"unknown prediction mode: {self.config.prediction}")

        z_x = z_x_all.view(steps, batch_size, *z_clean.shape[1:])[-1]
        z_cond = z_cond_all.view(steps, batch_size, *z_clean.shape[1:])[-1]
        z_pred = z_pred_all.view(steps, batch_size, *z_clean.shape[1:])[-1]
        flow_time = t.view(steps, batch_size)
        return flow_loss, z_x, z_cond, z_pred, flow_time, alignment_metrics

    def _aligned_text_condition(
        self,
        batch: dict[str, Any],
        text_tokens: Tensor,
        *,
        num_latents: int,
        text_lengths: Tensor | None,
        valid_latent_mask: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if self.config.text_condition_mode == "megatts_expand":
            if self.megatts_text_conditioner is None:
                raise RuntimeError("MegaTTS text conditioner is not initialized")
            return self.megatts_text_conditioner(
                batch,
                text_tokens,
                num_latents=num_latents,
                text_lengths=text_lengths,
                valid_latent_mask=valid_latent_mask,
            )
        if self.latent_text_projector is None:
            raise RuntimeError("latent text projector is not initialized")
        text_cond = self._latent_text_condition(
            batch,
            text_tokens,
            num_latents,
            text_lengths=text_lengths,
        )
        return self.latent_text_projector(text_cond), self._zero_alignment_metrics(text_tokens)

    @staticmethod
    def _zero_alignment_metrics(reference: Tensor) -> dict[str, Tensor]:
        zero = reference.new_tensor(0.0)
        return {
            "alignment_coverage": zero,
            "patch2token_nonzero_rate": zero,
            "missing_token_span_rate": zero,
            "patch2token_collision_rate": zero,
        }
