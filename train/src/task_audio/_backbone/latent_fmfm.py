from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
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
    _sample_mask_from_patch_mask,
    masked_mse,
    masked_weighted_mse,
)
from .latent_tts_fm_decoder import WaveFMDecoderHead
from .latent_tts_megatts import MegaTTSTextConditioner
from .latent_tts_split import LatentTTSSplit
from .speaker_sequence import SpeakerCAMPPlusSequenceEncoder, SpeakerERes2NetV2SequenceEncoder
from .transformer import TimeEmbedding, TransformerBlock
from .wave_fm_udit import LinearUDiTUNetWaveFMDecoder


DEFAULT_CAMPPLUS_PRETRAINED_PATH = str(
    Path(__file__).resolve().parents[1] / "pretrained" / "campplus_cn_en_common.pt"
)


@dataclass
class LatentFMFMConfig(MaskedFlowWaveTokenTTSConfig):
    build_deterministic_decoder: bool = False
    speaker_encoder_type: str = "campplus_sequence"
    speaker_pretrained_path: str | None = DEFAULT_CAMPPLUS_PRETRAINED_PATH
    speaker_fbank_sample_rate: int = 16000
    speaker_fbank_bins: int = 80
    speaker_pool_window_size: int = 32
    speaker_pool_stride: int = 16
    speaker_embedding_dim: int = 192
    speaker_mean_norm: bool = True
    speaker_freeze_backbone: bool = True
    speaker_profile: bool = False
    profile_sections: bool = False
    fmfm_training_mode: str = "prior_stopgrad_wave_online_encoder"
    wave_fm_encoder_noise_std: float = 0.08

    def __post_init__(self) -> None:
        super().__post_init__()
        self.fmfm_training_mode = str(self.fmfm_training_mode).lower().replace("-", "_")
        if self.fmfm_training_mode in {"default", "joint", "joint_prior_wave", "prior"}:
            self.fmfm_training_mode = "prior_pred"
        if self.fmfm_training_mode in {
            "prior_stopgrad_wave_online",
            "prior_stopgrad_online_encoder",
            "prior_stopgrad_wave_encoder",
            "prior_stopgrad_wave_online_encoder",
            "prior_detach_wave_online",
            "prior_detach_wave_online_encoder",
            "encoder_noisy_detach",
            "encoder_noisy",
            "encoder_noisy_detached",
            "detached_encoder_noisy",
            "stopgrad_encoder_noisy",
            "stopgrad_encoder_noisy_wave",
        }:
            self.fmfm_training_mode = "prior_stopgrad_wave_online_encoder"
        if (
            self.speaker_encoder_type != "campplus_sequence"
            and self.speaker_pretrained_path == DEFAULT_CAMPPLUS_PRETRAINED_PATH
        ):
            self.speaker_pretrained_path = None
        if self.fmfm_training_mode not in {"prior_pred", "prior_stopgrad_wave_online_encoder"}:
            raise ValueError(
                "fmfm_training_mode must be 'prior_pred' or 'prior_stopgrad_wave_online_encoder'"
            )
        self.wave_fm_encoder_noise_std = float(self.wave_fm_encoder_noise_std)
        if self.wave_fm_encoder_noise_std < 0.0:
            raise ValueError("wave_fm_encoder_noise_std must be non-negative")
        if self.speaker_freeze_backbone and not self.speaker_pretrained_path:
            raise ValueError("speaker_freeze_backbone=True requires speaker_pretrained_path")
        if self.fm_input_mode != "megatts_add":
            raise ValueError("LatentFMFM requires fm_input_mode='megatts_add'")
        if self.text_condition_mode != "megatts_expand":
            raise ValueError("LatentFMFM requires text_condition_mode='megatts_expand'")
        if self.prediction not in {"x_pred_v_loss", "x_pred_v_loss_weight"}:
            raise ValueError("LatentFMFM requires x-pred prior/wave FM prediction")
        if self.decoder_objective != "wave_fm":
            raise ValueError("LatentFMFM requires decoder_objective='wave_fm'")
        if self.split_target_encoder_ema:
            raise ValueError("LatentFMFM does not use split target encoder EMA")
        if self.speaker_encoder_type not in {"eres2netv2_sequence", "campplus_sequence"}:
            raise ValueError("LatentFMFM supports speaker_encoder_type='campplus_sequence' or 'eres2netv2_sequence'")
        if self.speaker_pool_window_size < 1:
            raise ValueError("speaker_pool_window_size must be positive")
        if self.speaker_pool_stride < 1:
            raise ValueError("speaker_pool_stride must be positive")
        if self.speaker_embedding_dim < 1:
            raise ValueError("speaker_embedding_dim must be positive")


@dataclass
class LatentFMFMMaskedPriorConfig(MaskedFlowWaveTokenTTSConfig):
    build_deterministic_decoder: bool = False
    speaker_encoder_type: str = "none"
    speaker_pretrained_path: str | None = None
    speaker_freeze_backbone: bool = False
    speaker_profile: bool = False
    wave_speaker_encoder_type: str = "none"
    wave_speaker_pretrained_path: str | None = None
    wave_speaker_freeze_backbone: bool = True
    wave_speaker_pool_window_size: int = 32
    wave_speaker_pool_stride: int = 16
    wave_speaker_embedding_dim: int = 192
    wave_speaker_mean_norm: bool = True
    wave_speaker_profile: bool = False
    profile_sections: bool = False
    prior_cfg_dropout: float | None = 0.10
    prior_cfg_dropout_mode: str = "drop_text_audio"
    wave_cfg_dropout: float = 0.10
    fmfm_training_mode: str = "prior_stopgrad_wave_online_encoder"
    wave_fm_encoder_noise_std: float = 0.08

    def __post_init__(self) -> None:
        decoder_objective = str(self.decoder_objective).lower().replace("-", "_")
        if decoder_objective in {"standard", "deterministic"}:
            self.decoder_objective = "wave"
            self.build_deterministic_decoder = True
        elif decoder_objective in {"decoder_fm", "fm_decoder", "fm", "wave_fm"}:
            self.decoder_objective = "wave_fm"
            self.build_deterministic_decoder = False
        super().__post_init__()
        self.speaker_encoder_type = str(self.speaker_encoder_type or "none").lower().replace("-", "_")
        if self.speaker_encoder_type not in {"none", "off", "disabled", "no_speaker"}:
            raise ValueError("LatentFMFMMaskedPrior requires speaker_encoder_type='none'")
        self.speaker_encoder_type = "none"
        if self.speaker_pretrained_path in {"", "none", "None"}:
            self.speaker_pretrained_path = None
        self.speaker_freeze_backbone = False
        self.wave_speaker_encoder_type = str(self.wave_speaker_encoder_type or "none").lower().replace("-", "_")
        if self.wave_speaker_encoder_type in {"off", "disabled", "no_speaker"}:
            self.wave_speaker_encoder_type = "none"
        if self.wave_speaker_encoder_type in {"campplus_sequence", "camp_plus"}:
            self.wave_speaker_encoder_type = "campplus"
        if self.wave_speaker_encoder_type not in {"none", "campplus"}:
            raise ValueError("wave_speaker_encoder_type must be 'none' or 'campplus'")
        if self.wave_speaker_pretrained_path in {"", "none", "None"}:
            self.wave_speaker_pretrained_path = None
        self.wave_speaker_freeze_backbone = bool(self.wave_speaker_freeze_backbone)
        if self.wave_speaker_encoder_type == "campplus":
            if not self.wave_speaker_pretrained_path:
                raise ValueError("wave_speaker_encoder_type='campplus' requires wave_speaker_pretrained_path")
            if not self.wave_speaker_freeze_backbone:
                raise ValueError("LatentFMFMMaskedPrior wave speaker encoder is frozen by design")
        self.wave_speaker_pool_window_size = int(self.wave_speaker_pool_window_size)
        self.wave_speaker_pool_stride = int(self.wave_speaker_pool_stride)
        self.wave_speaker_embedding_dim = int(self.wave_speaker_embedding_dim)
        if min(self.wave_speaker_pool_window_size, self.wave_speaker_pool_stride, self.wave_speaker_embedding_dim) < 1:
            raise ValueError("wave speaker window/stride/embedding dimensions must be positive")
        self.fmfm_training_mode = str(self.fmfm_training_mode).lower().replace("-", "_")
        if self.fmfm_training_mode in {"default", "joint", "joint_prior_wave", "prior"}:
            self.fmfm_training_mode = "prior_pred"
        if self.fmfm_training_mode in {
            "prior_stopgrad_wave_online",
            "prior_stopgrad_online_encoder",
            "prior_stopgrad_wave_encoder",
            "prior_stopgrad_wave_online_encoder",
            "prior_detach_wave_online",
            "prior_detach_wave_online_encoder",
            "encoder_noisy_detach",
            "encoder_noisy",
            "encoder_noisy_detached",
            "detached_encoder_noisy",
            "stopgrad_encoder_noisy",
            "stopgrad_encoder_noisy_wave",
        }:
            self.fmfm_training_mode = "prior_stopgrad_wave_online_encoder"
        if self.fmfm_training_mode not in {"prior_pred", "prior_stopgrad_wave_online_encoder"}:
            raise ValueError(
                "fmfm_training_mode must be 'prior_pred' or 'prior_stopgrad_wave_online_encoder'"
            )
        self.wave_fm_encoder_noise_std = float(self.wave_fm_encoder_noise_std)
        if self.wave_fm_encoder_noise_std < 0.0:
            raise ValueError("wave_fm_encoder_noise_std must be non-negative")
        if self.fm_input_mode != "megatts_add":
            raise ValueError("LatentFMFMMaskedPrior requires fm_input_mode='megatts_add'")
        if self.text_condition_mode != "megatts_expand":
            raise ValueError("LatentFMFMMaskedPrior requires text_condition_mode='megatts_expand'")
        if self.prediction not in {"x_pred_v_loss", "x_pred_v_loss_weight"}:
            raise ValueError("LatentFMFMMaskedPrior requires x-pred prior/wave FM prediction")
        if self.decoder_objective == "wave" and not self.build_deterministic_decoder:
            raise ValueError("decoder_objective='wave' requires build_deterministic_decoder=True")
        if self.decoder_objective == "wave_fm" and self.build_deterministic_decoder:
            raise ValueError("decoder_objective='wave_fm' requires build_deterministic_decoder=False")
        if self.mask_source not in {"prefix", "random_span"}:
            raise ValueError("LatentFMFMMaskedPrior supports mask_source='prefix' or 'random_span'")
        if not self.split_target_encoder_ema:
            raise ValueError("LatentFMFMMaskedPrior requires split_target_encoder_ema=True")
        if self.use_target_token and self.mask_source == "random_span":
            raise ValueError("random_span masked prior does not support use_target_token")


class SpeakerPrefixMegaTTSAddFMEncoder(nn.Module):
    """MegaTTS-add audio tokens with speaker prefix in the sequence axis."""

    def __init__(
        self,
        *,
        token_dim: int,
        hidden_dim: int,
        depth: int,
        adaln_every: int,
        heads: int,
        dim_head: int,
        ffn_mult: int,
        dropout: float,
        rope_base: float,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("fm_depth must be positive for speaker-prefix prior FM")
        if adaln_every < 1:
            raise ValueError("fm_adaln_every must be positive")
        self.x_proj = nn.Linear(token_dim, hidden_dim)
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)
        self.speaker_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.audio_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.speaker_sep = nn.Parameter(torch.zeros(1, 1, hidden_dim))
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
        for block in self.blocks:
            self._init_active_adaln_gates(block)
        self.norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _init_active_adaln_gates(block: TransformerBlock) -> None:
        if not block.uses_cond or block.ada_norm is None:
            return
        linear = block.ada_norm[-1]
        hidden_dim = linear.bias.numel() // 6
        with torch.no_grad():
            linear.bias[2 * hidden_dim : 3 * hidden_dim].fill_(1.0)
            linear.bias[5 * hidden_dim : 6 * hidden_dim].fill_(1.0)

    def forward(
        self,
        z_x: Tensor,
        aligned_text: Tensor,
        speaker_tokens: Tensor,
        speaker_token_mask: Tensor,
        t: Tensor,
        *,
        valid_audio_mask: Tensor,
        condition_dropout_mask: Tensor | None = None,
    ) -> Tensor:
        if aligned_text.shape[:2] != z_x.shape[:2]:
            raise ValueError("aligned_text must have shape [B, T_audio, hidden_dim]")
        if speaker_tokens.ndim != 3:
            raise ValueError("speaker_tokens must have shape [B, K, H]")
        if speaker_tokens.shape[0] != z_x.shape[0]:
            raise ValueError("speaker_tokens batch must match z_x")
        if speaker_tokens.shape[-1] != aligned_text.shape[-1]:
            raise ValueError("speaker_tokens hidden dim must match aligned_text")
        if speaker_token_mask.shape != speaker_tokens.shape[:2]:
            raise ValueError("speaker_token_mask must have shape [B, K]")
        if valid_audio_mask.shape != z_x.shape[:2]:
            raise ValueError("valid_audio_mask must have shape [B, T_audio]")
        if t.shape != (z_x.shape[0],):
            raise ValueError("t must have shape [B]")

        text = aligned_text
        speaker = speaker_tokens + self.speaker_type.to(dtype=speaker_tokens.dtype)
        sep = self.speaker_sep.to(device=z_x.device, dtype=z_x.dtype).expand(z_x.shape[0], 1, -1)
        if condition_dropout_mask is not None:
            if condition_dropout_mask.shape != (z_x.shape[0],):
                raise ValueError("condition_dropout_mask must have shape [B]")
            drop = condition_dropout_mask.to(device=z_x.device, dtype=torch.bool)
            text = torch.where(drop[:, None, None], torch.zeros_like(text), text)
            speaker = torch.where(drop[:, None, None], torch.zeros_like(speaker), speaker)
            sep = torch.where(drop[:, None, None], torch.zeros_like(sep), sep)

        audio_h = self.x_proj(z_x) + self.text_proj(text) + self.audio_type.to(dtype=z_x.dtype)
        audio_h = audio_h * valid_audio_mask.unsqueeze(-1).to(device=z_x.device, dtype=audio_h.dtype)
        speaker = speaker * speaker_token_mask.unsqueeze(-1).to(device=z_x.device, dtype=speaker.dtype)

        h = torch.cat((speaker, sep, audio_h), dim=1)
        sep_mask = torch.ones((z_x.shape[0], 1), device=z_x.device, dtype=torch.bool)
        joint_mask = torch.cat(
            (
                speaker_token_mask.to(device=z_x.device, dtype=torch.bool),
                sep_mask,
                valid_audio_mask.to(device=z_x.device, dtype=torch.bool),
            ),
            dim=1,
        )
        h = h * joint_mask.unsqueeze(-1).to(dtype=h.dtype)

        time_cond = self.time_embed(t)
        for block in self.blocks:
            h = block(h, time_cond if block.uses_cond else None, mask=joint_mask)
            h = h * joint_mask.unsqueeze(-1).to(dtype=h.dtype)
        h = self.norm(h)
        audio_start = speaker_tokens.shape[1] + 1
        return h[:, audio_start:]


class MaskedMegaTTSAddFMEncoder(nn.Module):
    """MegaTTS-add FM encoder with visible latent conditioning and no speaker prefix."""

    def __init__(
        self,
        *,
        token_dim: int,
        hidden_dim: int,
        depth: int,
        adaln_every: int,
        heads: int,
        dim_head: int,
        ffn_mult: int,
        dropout: float,
        rope_base: float,
        input_fusion: str = "add",
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("fm_depth must be positive for masked-prior FM")
        if adaln_every < 1:
            raise ValueError("fm_adaln_every must be positive")
        self.input_fusion = str(input_fusion).lower().replace("-", "_")
        if self.input_fusion in {"sum", "additive"}:
            self.input_fusion = "add"
        if self.input_fusion in {"concat", "project_concat", "projected_fuse"}:
            self.input_fusion = "projected_concat"
        if self.input_fusion not in {"add", "projected_concat"}:
            raise ValueError("prior_fm_input_fusion must be 'add' or 'projected_concat'")
        self.x_proj = nn.Linear(token_dim, hidden_dim)
        self.cond_proj = nn.Linear(token_dim + 1, hidden_dim)
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)
        self.input_fuse = (
            nn.Linear(hidden_dim * 3, hidden_dim)
            if self.input_fusion == "projected_concat"
            else None
        )
        self.input_norm = (
            nn.LayerNorm(hidden_dim)
            if self.input_fusion == "projected_concat"
            else None
        )
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
        for block in self.blocks:
            self._init_active_adaln_gates(block)
        self.norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _init_active_adaln_gates(block: TransformerBlock) -> None:
        if not block.uses_cond or block.ada_norm is None:
            return
        linear = block.ada_norm[-1]
        hidden_dim = linear.bias.numel() // 6
        with torch.no_grad():
            linear.bias[2 * hidden_dim : 3 * hidden_dim].fill_(1.0)
            linear.bias[5 * hidden_dim : 6 * hidden_dim].fill_(1.0)

    def forward(
        self,
        z_x: Tensor,
        z_cond: Tensor,
        aligned_text: Tensor,
        t: Tensor,
        *,
        valid_audio_mask: Tensor,
        prompt_audio_mask: Tensor | None = None,
        condition_dropout_mask: Tensor | None = None,
        condition_dropout_mode: str = "drop_text",
    ) -> Tensor:
        if z_x.shape != z_cond.shape:
            raise ValueError("z_x and z_cond must have the same shape")
        if aligned_text.shape[:2] != z_x.shape[:2]:
            raise ValueError("aligned_text must have shape [B, T_audio, hidden_dim]")
        if valid_audio_mask.shape != z_x.shape[:2]:
            raise ValueError("valid_audio_mask must have shape [B, T_audio]")
        if t.shape != (z_x.shape[0],):
            raise ValueError("t must have shape [B]")
        if prompt_audio_mask is None:
            prompt = z_x.new_zeros(z_x.shape[:2])
        else:
            if prompt_audio_mask.shape != z_x.shape[:2]:
                raise ValueError("prompt_audio_mask must have shape [B, T_audio]")
            prompt = prompt_audio_mask.to(device=z_x.device, dtype=z_x.dtype)

        cond = torch.cat((z_cond, prompt.unsqueeze(-1)), dim=-1)
        text = aligned_text
        if condition_dropout_mask is not None:
            if condition_dropout_mask.shape != (z_x.shape[0],):
                raise ValueError("condition_dropout_mask must have shape [B]")
            drop = condition_dropout_mask.to(device=z_x.device, dtype=torch.bool)
            mode = str(condition_dropout_mode).lower().replace("-", "_")
            if mode in {"text", "text_only"}:
                mode = "drop_text"
            if mode in {
                "text_prompt",
                "drop_text_prompt",
                "both",
                "all",
                "text_audio",
                "drop_audio_text",
                "text_speech",
                "drop_text_speech",
                "text_prompt_speech",
            }:
                mode = "drop_text_audio"
            if mode not in {"drop_text", "drop_text_audio"}:
                raise ValueError("condition_dropout_mode must be 'drop_text' or 'drop_text_audio'")
            text = torch.where(drop[:, None, None], torch.zeros_like(text), text)
            if mode == "drop_text_audio":
                # Prior FM has two external conditions: aligned text and visible prompt speech.
                # drop_text_audio mode drops both. Target noisy x_t stays visible; only prompt
                # tokens in z_x plus z_cond/prompt flags are zeroed for unconditional rows.
                prompt_bool = prompt_audio_mask.to(device=z_x.device, dtype=torch.bool) if prompt_audio_mask is not None else None
                if prompt_bool is not None:
                    prompt_drop = drop[:, None, None] & prompt_bool[:, :, None]
                    z_x = torch.where(prompt_drop, torch.zeros_like(z_x), z_x)
                z_cond = torch.where(drop[:, None, None], torch.zeros_like(z_cond), z_cond)
                prompt = torch.where(drop[:, None], torch.zeros_like(prompt), prompt)
                cond = torch.cat((z_cond, prompt.unsqueeze(-1)), dim=-1)

        mask = valid_audio_mask.to(device=z_x.device, dtype=torch.bool)
        x_h = self.x_proj(z_x)
        cond_h = self.cond_proj(cond)
        text_h = self.text_proj(text)
        if self.input_fusion == "projected_concat":
            h = self.input_norm(self.input_fuse(torch.cat((x_h, cond_h, text_h), dim=-1)))
        else:
            h = x_h + cond_h + text_h
        h = h * mask.unsqueeze(-1).to(dtype=h.dtype)
        time_cond = self.time_embed(t)
        for block in self.blocks:
            h = block(h, time_cond if block.uses_cond else None, mask=mask)
            h = h * mask.unsqueeze(-1).to(dtype=h.dtype)
        return self.norm(h)


class LatentFMFM(MaskedFlowWaveTokenTTS):
    """Full-target latent prior FM plus waveform FM with speaker prefix."""

    def __init__(self, config: LatentFMFMConfig) -> None:
        super().__init__(config)
        self.config: LatentFMFMConfig
        self.megatts_text_conditioner = MegaTTSTextConditioner(
            text_dim=config.text_dim,
            hidden_dim=config.fm_hidden_dim,
            downsample_factor=config.downsample_factor,
            alignment_mode=config.megatts_alignment_mode,
            anchor_mode=config.megatts_anchor_mode,
            anchor_ratio=config.megatts_anchor_ratio,
        )
        speaker_encoder_cls = (
            SpeakerCAMPPlusSequenceEncoder
            if config.speaker_encoder_type == "campplus_sequence"
            else SpeakerERes2NetV2SequenceEncoder
        )
        self.speaker_encoder = speaker_encoder_cls(
            output_dim=config.fm_hidden_dim,
            sample_rate=config.speaker_fbank_sample_rate,
            num_mel_bins=config.speaker_fbank_bins,
            window_size=config.speaker_pool_window_size,
            stride=config.speaker_pool_stride,
            embedding_size=config.speaker_embedding_dim,
            pretrained_path=config.speaker_pretrained_path,
            mean_norm=config.speaker_mean_norm,
            freeze_backbone=config.speaker_freeze_backbone,
            profile=config.speaker_profile,
        )
        self.fm_encoder = SpeakerPrefixMegaTTSAddFMEncoder(
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
        self.fm_head = FMHead(config.fm_hidden_dim, config.latent_dim)
        self.wave_fm_decoder = WaveFMDecoderHead(
            patch_size=config.patch_size,
            cond_dim=config.latent_dim,
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
        raise NotImplementedError("LatentFMFM has no deterministic decoder; use wave FM sampling")

    def decode_tokens_raw(self, z: Tensor, original_len: int | None = None) -> Tensor:
        del z, original_len
        raise NotImplementedError("LatentFMFM has no deterministic decoder; use wave FM sampling")

    def forward(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        profile_metrics: dict[str, Tensor] = {}
        total_start = self._profile_start(next(iter(batch.values())))
        timer_start = total_start
        wav_clean_raw = batch["wav_clean"].float()
        wav_clean = self._scale_waveform(wav_clean_raw)
        patches, original_len = self.patchify.patchify(wav_clean)
        valid_patch_mask = _patch_mask_from_sample_mask(
            batch.get("wav_valid_mask"),
            patch_size=self.config.patch_size,
            num_patches=patches.shape[1],
            fallback_shape=wav_clean_raw.shape,
        ).to(device=patches.device, dtype=torch.bool)
        valid_latent_mask = _downsample_bool_mask(
            valid_patch_mask,
            factor=self.config.downsample_factor,
        )
        z_clean, z_raw = self._encode_full_latents(patches, return_raw=True)
        if valid_latent_mask.shape[1] < z_clean.shape[1]:
            valid_latent_mask = F.pad(valid_latent_mask, (0, z_clean.shape[1] - valid_latent_mask.shape[1]))
        valid_latent_mask = valid_latent_mask[:, : z_clean.shape[1]]
        self._profile_stop(profile_metrics, "profile_encoder_sec", timer_start, z_clean)

        timer_start = self._profile_start(z_clean)
        text_token_ids, text_lengths = self._condition_text_inputs(batch)
        text_tokens = self._encode_text_ids(text_token_ids, text_lengths=text_lengths)
        self._profile_stop(profile_metrics, "profile_text_sec", timer_start, text_tokens)

        timer_start = self._profile_start(z_clean)
        speaker_tokens, speaker_token_mask = self.speaker_encoder(
            batch["ref_wav_clean"].to(device=wav_clean.device, dtype=wav_clean.dtype),
            batch["ref_wav_valid_mask"].to(device=wav_clean.device, dtype=torch.bool),
        )
        self._profile_stop(profile_metrics, "profile_speaker_sec", timer_start, speaker_tokens)
        speaker_timing_metrics = {
            key: wav_clean.new_tensor(value)
            for key, value in self.speaker_encoder.last_timing.items()
        }
        prior_cfg_dropout = float(getattr(self.config, "prior_cfg_dropout", self.config.cfg_dropout))
        if self.training and prior_cfg_dropout > 0.0:
            prior_condition_dropout_mask = (
                torch.rand(z_clean.shape[0], device=z_clean.device)
                < prior_cfg_dropout
            )
        else:
            prior_condition_dropout_mask = torch.zeros(z_clean.shape[0], device=z_clean.device, dtype=torch.bool)
        wave_cfg_dropout = float(getattr(self.config, "wave_cfg_dropout", 0.0))
        if self.training and wave_cfg_dropout > 0.0:
            wave_condition_dropout_mask = (
                torch.rand(z_clean.shape[0], device=z_clean.device)
                < wave_cfg_dropout
            )
        else:
            wave_condition_dropout_mask = torch.zeros(z_clean.shape[0], device=z_clean.device, dtype=torch.bool)

        stopgrad_prior_wave_online = self.config.fmfm_training_mode == "prior_stopgrad_wave_online_encoder"
        prior_target_z = z_clean.detach() if stopgrad_prior_wave_online else z_clean
        (
            prior_fm_loss,
            z_x,
            z_pred,
            flow_time,
            z_pred_flow,
            alignment_metrics,
        ) = self._profile_call(
            profile_metrics,
            "profile_prior_sec",
            prior_target_z,
            self._prior_flow_loss,
            batch,
            prior_target_z,
            text_tokens,
            speaker_tokens,
            speaker_token_mask,
            text_lengths=text_lengths,
            valid_latent_mask=valid_latent_mask,
            condition_dropout_mask=prior_condition_dropout_mask,
        )
        wave_condition_flow = z_pred_flow
        wave_condition_noise_rms = z_clean.new_tensor(0.0)
        if stopgrad_prior_wave_online:
            wave_condition = z_clean
            noise_std = float(self.config.wave_fm_encoder_noise_std)
            if self.training and noise_std > 0.0:
                noise = torch.randn_like(wave_condition) * noise_std
                noise = noise * valid_latent_mask.unsqueeze(-1).to(dtype=noise.dtype)
                wave_condition = wave_condition + noise
                selected_noise = noise[valid_latent_mask]
                if selected_noise.numel() > 0:
                    wave_condition_noise_rms = selected_noise.float().pow(2).mean().sqrt()
            wave_condition_flow = _repeat_steps(wave_condition, int(self.config.flow_steps_per_recon))
        (
            wave_fm_loss,
            wave_pred_patches,
            wave_x,
            wave_target_mask,
            wave_valid_mask,
        ) = self._profile_call(
            profile_metrics,
            "profile_wave_sec",
            patches,
            self._wave_flow_loss,
            patches,
            wave_condition_flow,
            flow_time.reshape(-1),
            valid_patch_mask=valid_patch_mask,
            condition_dropout_mask=wave_condition_dropout_mask,
        )

        timer_start = self._profile_start(wave_pred_patches)
        pred_patches = torch.where(wave_target_mask.unsqueeze(-1), wave_pred_patches, patches)
        pred_patches = pred_patches * valid_patch_mask.unsqueeze(-1).to(dtype=pred_patches.dtype)
        teacher_waveform_model = self.patchify.unpatchify(pred_patches, original_len=original_len)
        teacher_waveform = self._unscale_waveform(teacher_waveform_model)
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
        self._profile_stop(profile_metrics, "profile_mel_sec", timer_start, mel_loss)
        reg_target = {
            "z_raw": z_raw,
            "z_clean": z_clean,
            "z_flow_target": z_clean,
            "z_pred": z_pred,
        }[str(self.config.latent_reg_target)]
        if stopgrad_prior_wave_online:
            latent_reg_loss = zero
            latent_reg_metrics = {}
        else:
            latent_reg_loss = self._latent_regularization_loss(
                reg_target,
                valid_latent_mask,
                target_latent_mask=valid_latent_mask,
            )
            latent_reg_metrics = self._latent_regularization_metrics(
                reg_target,
                valid_latent_mask,
                valid_latent_mask,
            )
        total_loss = (
            self.config.lambda_flow * prior_fm_loss
            + self.config.lambda_wave_fm * wave_fm_loss
            + self.config.lambda_mel * mel_loss
            + self.config.lambda_latent_reg * latent_reg_loss
        )
        valid_sample_mask = _sample_mask(batch.get("wav_valid_mask"), wav_clean_raw.shape).to(
            device=wav_clean.device,
            dtype=torch.bool,
        )
        speaker_token_count = speaker_token_mask.detach().float().sum(dim=1).mean()
        mask = valid_latent_mask
        self._profile_stop(profile_metrics, "profile_forward_total_sec", total_start, total_loss)
        return {
            "loss": total_loss,
            "flow_loss": prior_fm_loss,
            "prior_fm_loss": prior_fm_loss,
            "wave_fm_loss": wave_fm_loss,
            "recon_loss": zero,
            "mel_loss": mel_loss,
            "latent_reg_loss": latent_reg_loss,
            "weighted_latent_reg_loss": latent_reg_loss * float(self.config.lambda_latent_reg),
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
            "wav_cond": torch.zeros_like(z_clean),
            "wave_fm_input": wave_x,
            "wave_fm_pred_patches": wave_pred_patches,
            "wave_target_mask": wave_target_mask,
            "target_sample_mask": target_sample_mask,
            "flow_time": flow_time,
            "condition_dropout_fraction": prior_condition_dropout_mask.detach().float().mean(),
            "prior_condition_dropout_fraction": prior_condition_dropout_mask.detach().float().mean(),
            "wave_condition_dropout_fraction": wave_condition_dropout_mask.detach().float().mean(),
            "fmfm_training_mode_id": wav_clean.new_tensor(1.0 if stopgrad_prior_wave_online else 0.0),
            "prior_encoder_stop_gradient": wav_clean.new_tensor(1.0 if stopgrad_prior_wave_online else 0.0),
            "wave_encoder_online": wav_clean.new_tensor(1.0 if stopgrad_prior_wave_online else 0.0),
            "wave_fm_condition_source_id": wav_clean.new_tensor(1.0 if stopgrad_prior_wave_online else 0.0),
            "wave_fm_encoder_noise_std": wav_clean.new_tensor(float(self.config.wave_fm_encoder_noise_std)),
            "wave_fm_condition_noise_rms": wave_condition_noise_rms.detach(),
            "mask": mask,
            "valid_latent_mask": valid_latent_mask,
            "z_clean": z_clean,
            "z_raw": z_raw,
            "z_flow_target": z_clean,
            "split_target_encoder_ema": wav_clean.new_tensor(0.0),
            "z_pred": z_pred,
            "speaker_token_count": speaker_token_count,
            "valid_sample_mask": valid_sample_mask,
            **latent_reg_metrics,
            **profile_metrics,
            **speaker_timing_metrics,
            **alignment_metrics,
        }

    def _profile_start(self, anchor: Any) -> float:
        if not self.config.profile_sections:
            return 0.0
        if isinstance(anchor, Tensor):
            self._profile_sync(anchor)
        return time.perf_counter()

    def _profile_stop(
        self,
        metrics: dict[str, Tensor],
        name: str,
        start: float,
        anchor: Tensor,
    ) -> None:
        if not self.config.profile_sections:
            return
        self._profile_sync(anchor)
        metrics[name] = anchor.new_tensor(time.perf_counter() - start)

    def _profile_call(self, metrics: dict[str, Tensor], name: str, anchor: Tensor, fn: Any, *args: Any, **kwargs: Any) -> Any:
        start = self._profile_start(anchor)
        result = fn(*args, **kwargs)
        stop_anchor = result[0] if isinstance(result, tuple) and isinstance(result[0], Tensor) else anchor
        self._profile_stop(metrics, name, start, stop_anchor)
        return result

    @staticmethod
    def _profile_sync(anchor: Tensor) -> None:
        if anchor.device.type == "cuda":
            torch.cuda.synchronize(anchor.device)

    def _encode_full_latents(
        self,
        patches: Tensor,
        *,
        return_raw: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        patch_tokens = self.token_norm(self.target_encoder(patches))
        return self._encode_patch_tokens(patch_tokens, return_raw=return_raw)

    def _normalize_flow_x_pred(self, z_pred: Tensor) -> Tensor:
        return z_pred

    def _prior_flow_loss(
        self,
        batch: dict[str, Any],
        z_clean: Tensor,
        text_tokens: Tensor,
        speaker_tokens: Tensor,
        speaker_token_mask: Tensor,
        *,
        text_lengths: Tensor | None,
        valid_latent_mask: Tensor,
        condition_dropout_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        steps = int(self.config.flow_steps_per_recon)
        batch_size = z_clean.shape[0]
        z_stop = z_clean.detach()
        z_target = z_stop + float(self.config.target_grad_scale) * (z_clean - z_stop)
        z_target_flow = _repeat_steps(z_target, steps)
        valid_flow = _repeat_steps(valid_latent_mask, steps)
        speaker_tokens_flow = _repeat_steps(speaker_tokens, steps)
        speaker_mask_flow = _repeat_steps(speaker_token_mask, steps)
        condition_dropout_mask_flow = (
            _repeat_steps(condition_dropout_mask.to(device=z_clean.device, dtype=torch.bool), steps)
            if condition_dropout_mask is not None
            else None
        )

        eps = torch.randn_like(z_target_flow)
        t = self._sample_flow_time(
            z_target_flow.shape[0],
            device=z_clean.device,
            dtype=z_clean.dtype,
        )
        t_view = t[:, None, None]
        z_t = (1.0 - t_view) * eps + t_view * z_target_flow
        v_target = z_target_flow - eps
        z_t = z_t * valid_flow.unsqueeze(-1).to(dtype=z_t.dtype)

        aligned_text, alignment_metrics = self._aligned_text_condition(
            batch,
            text_tokens,
            num_latents=z_clean.shape[1],
            text_lengths=text_lengths,
            valid_latent_mask=valid_latent_mask,
        )
        aligned_text_flow = _repeat_steps(aligned_text, steps)
        h = self.fm_encoder(
            z_t,
            aligned_text_flow,
            speaker_tokens_flow,
            speaker_mask_flow,
            t,
            valid_audio_mask=valid_flow,
            condition_dropout_mask=condition_dropout_mask_flow,
        )
        fm_out = self.fm_head(h)
        target_mask = valid_flow
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
            raise ValueError(f"unsupported prediction mode for LatentFMFM prior: {self.config.prediction}")

        z_x = z_t.view(steps, batch_size, *z_clean.shape[1:])[-1]
        z_pred = z_pred_all.view(steps, batch_size, *z_clean.shape[1:])[-1]
        flow_time = t.view(steps, batch_size)
        return flow_loss, z_x, z_pred, flow_time, z_pred_all, alignment_metrics

    def _wave_flow_loss(
        self,
        patches: Tensor,
        z0_hat_flow: Tensor,
        t: Tensor,
        *,
        valid_patch_mask: Tensor,
        condition_dropout_mask: Tensor | None = None,
        speaker_emb: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size, num_patches, _ = patches.shape
        steps = int(self.config.flow_steps_per_recon)
        if z0_hat_flow.shape[0] != batch_size * steps:
            raise ValueError("z0_hat_flow batch does not match flow_steps_per_recon")
        if t.shape != (batch_size * steps,):
            raise ValueError("wave FM time tensor must have shape [B * flow_steps]")
        if condition_dropout_mask is not None:
            if condition_dropout_mask.shape == (batch_size,):
                wave_drop = _repeat_steps(
                    condition_dropout_mask.to(device=patches.device, dtype=torch.bool),
                    steps,
                )
            elif condition_dropout_mask.shape == (batch_size * steps,):
                wave_drop = condition_dropout_mask.to(device=patches.device, dtype=torch.bool)
            else:
                raise ValueError("condition_dropout_mask must have shape [B] or [B * flow_steps]")
        else:
            wave_drop = torch.zeros(batch_size * steps, device=patches.device, dtype=torch.bool)
        speaker_flow = None
        if speaker_emb is not None:
            if speaker_emb.shape[0] == batch_size:
                speaker_flow = _repeat_steps(speaker_emb, steps)
            elif speaker_emb.shape[0] == batch_size * steps:
                speaker_flow = speaker_emb
            else:
                raise ValueError("speaker_emb must have batch B or B * flow_steps")
            speaker_flow = speaker_flow.to(device=patches.device, dtype=patches.dtype)

        wave_target_mask = valid_patch_mask
        wave_target_mask_flow = _repeat_steps(wave_target_mask, steps)
        valid_patch_flow = _repeat_steps(valid_patch_mask, steps)
        patches_flow = _repeat_steps(patches, steps)
        eps = torch.randn_like(patches_flow)
        t_view = t[:, None, None]
        w_t = (1.0 - t_view) * eps + t_view * patches_flow
        v_target = patches_flow - eps
        wave_x_all = w_t * valid_patch_flow.unsqueeze(-1).to(dtype=w_t.dtype)

        z0_hat_wave = z0_hat_flow.repeat_interleave(int(self.config.downsample_factor), dim=1)
        if z0_hat_wave.shape[1] < num_patches:
            z0_hat_wave = F.pad(z0_hat_wave, (0, 0, 0, num_patches - z0_hat_wave.shape[1]))
        z0_hat_wave = z0_hat_wave[:, :num_patches]
        if wave_drop.any():
            # Wave FM unconditional rows drop the prior-z condition and wave-only speaker condition.
            z0_hat_wave = torch.where(wave_drop[:, None, None], torch.zeros_like(z0_hat_wave), z0_hat_wave)
            if speaker_flow is not None:
                speaker_flow = torch.where(wave_drop[:, None], torch.zeros_like(speaker_flow), speaker_flow)
        wave_kwargs: dict[str, Tensor | None] = {"valid_wave_mask": valid_patch_flow}
        if isinstance(self.wave_fm_decoder, LinearUDiTUNetWaveFMDecoder):
            wave_kwargs["speaker_emb"] = speaker_flow
        wave_pred_all = self.wave_fm_decoder(wave_x_all, z0_hat_wave, t, **wave_kwargs)
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
            raise ValueError(f"unsupported prediction mode for LatentFMFM wave FM: {self.config.prediction}")

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
        prior_steps: int | None = None,
        wave_steps: int | None = None,
        sample_batch: dict[str, Any],
        text_token_lengths: Tensor | None = None,
        cfg_strength: float = 0.0,
        cfg_rescale: float = 0.0,
        prior_cfg_strength: float | None = None,
        prior_cfg_rescale: float | None = None,
        wave_cfg_strength: float | None = None,
        wave_cfg_rescale: float | None = None,
        solver: str = "heun",
        sampler: str = "joint",
    ) -> Tensor:
        solver = str(solver).lower()
        if solver not in {"euler", "heun", "rk4"}:
            raise ValueError("solver must be 'euler', 'heun', or 'rk4'")
        sampler = str(sampler).lower().replace("-", "_")
        if sampler in {"2stage", "two_stage", "sequential"}:
            sampler = "twostage"
        if sampler not in {"joint", "twostage"}:
            raise ValueError("sampler must be 'joint' or 'twostage'")
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        prior_step_count = int(num_steps if prior_steps is None else prior_steps)
        wave_step_count = int(num_steps if wave_steps is None else wave_steps)
        if prior_step_count < 1:
            raise ValueError("prior_steps must be positive")
        if wave_step_count < 1:
            raise ValueError("wave_steps must be positive")
        if target_num_tokens < 1:
            raise ValueError("target_num_tokens must be positive")
        prior_cfg_strength = float(cfg_strength if prior_cfg_strength is None else prior_cfg_strength)
        prior_cfg_rescale = float(cfg_rescale if prior_cfg_rescale is None else prior_cfg_rescale)
        wave_cfg_strength = float(0.0 if wave_cfg_strength is None else wave_cfg_strength)
        wave_cfg_rescale = float(0.0 if wave_cfg_rescale is None else wave_cfg_rescale)

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
        valid_wave_mask = torch.ones(wave_state.shape[:2], device=wave_state.device, dtype=torch.bool)
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
        speaker_tokens, speaker_token_mask = self.speaker_encoder(
            sample_batch["ref_wav_clean"].to(device=z_prompt.device, dtype=torch.float32),
            sample_batch["ref_wav_valid_mask"].to(device=z_prompt.device, dtype=torch.bool),
        )
        speaker_tokens = speaker_tokens.to(dtype=z_prompt.dtype)
        aligned_text_cfg = torch.cat((aligned_text, aligned_text), dim=0)
        speaker_tokens_cfg = torch.cat((speaker_tokens, speaker_tokens), dim=0)
        speaker_token_mask_cfg = torch.cat((speaker_token_mask, speaker_token_mask), dim=0)
        valid_latent_mask_cfg = torch.cat((valid_latent_mask, valid_latent_mask), dim=0)

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
            return (z_pred - state) / denom

        def wave_velocity_from_x0(wave_pred: Tensor, state: Tensor, time_value: float) -> Tensor:
            denom = max(float(self.config.flow_xpred_denom_min), 1.0 - float(time_value))
            return (wave_pred - state) / denom

        def z_to_wave_cond(z0_hat: Tensor) -> Tensor:
            cond = z0_hat.repeat_interleave(factor, dim=1)
            if cond.shape[1] < total_patches:
                cond = F.pad(cond, (0, 0, 0, total_patches - cond.shape[1]))
            return cond[:, :total_patches]

        def prior_x0(
            z_current: Tensor,
            t: Tensor,
            *,
            condition_dropout_mask: Tensor | None,
        ) -> Tensor:
            is_cfg = z_current.shape[0] != batch_size
            h = self.fm_encoder(
                z_current,
                aligned_text_cfg if is_cfg else aligned_text,
                speaker_tokens_cfg if is_cfg else speaker_tokens,
                speaker_token_mask_cfg if is_cfg else speaker_token_mask,
                t,
                valid_audio_mask=valid_latent_mask_cfg if is_cfg else valid_latent_mask,
                condition_dropout_mask=condition_dropout_mask,
            )
            return self._normalize_flow_x_pred(self.fm_head(h))

        def eval_prior_velocity(z_current: Tensor, time_value: float) -> Tensor:
            z_current = with_prompt_z(z_current)
            t = torch.full((batch_size,), time_value, device=z_current.device, dtype=z_current.dtype)
            if prior_cfg_strength > 0.0:
                condition_dropout_mask = torch.cat(
                    (
                        torch.zeros(batch_size, device=z_current.device, dtype=torch.bool),
                        torch.ones(batch_size, device=z_current.device, dtype=torch.bool),
                    ),
                    dim=0,
                )
                z_cat = torch.cat((z_current, z_current), dim=0)
                t_cat = torch.cat((t, t), dim=0)
                z_pred_cond, z_pred_uncond = prior_x0(
                    z_cat,
                    t_cat,
                    condition_dropout_mask=condition_dropout_mask,
                ).chunk(2, dim=0)
                z_v_cond = z_velocity_from_x0(z_pred_cond, z_current, time_value)
                z_v_uncond = z_velocity_from_x0(z_pred_uncond, z_current, time_value)
                z_velocity = z_v_cond + prior_cfg_strength * (z_v_cond - z_v_uncond)
                if prior_cfg_rescale > 0.0:
                    mix = max(0.0, min(prior_cfg_rescale, 1.0))
                    z_velocity = self._rescale_guided_velocity(
                        z_velocity,
                        z_v_cond,
                        target_start=prompt_tokens,
                        mix=mix,
                    )
                return z_velocity

            z_pred = prior_x0(z_current, t, condition_dropout_mask=None)
            return z_velocity_from_x0(z_pred, z_current, time_value)

        def eval_wave_velocity(wave_current: Tensor, z0_hat: Tensor, time_value: float) -> Tensor:
            wave_current = with_prompt_wave(wave_current)
            z0_hat = with_prompt_z(z0_hat)
            t = torch.full((batch_size,), time_value, device=wave_current.device, dtype=wave_current.dtype)
            z0_cond = z_to_wave_cond(z0_hat)
            if wave_cfg_strength > 0.0:
                wave_pred_cond, wave_pred_uncond = self.wave_fm_decoder(
                    torch.cat((wave_current, wave_current), dim=0),
                    torch.cat((z0_cond, torch.zeros_like(z0_cond)), dim=0),
                    torch.cat((t, t), dim=0),
                    valid_wave_mask=torch.cat((valid_wave_mask, valid_wave_mask), dim=0),
                ).chunk(2, dim=0)
                wave_v_cond = wave_velocity_from_x0(wave_pred_cond, wave_current, time_value)
                wave_v_uncond = wave_velocity_from_x0(wave_pred_uncond, wave_current, time_value)
                wave_velocity = wave_v_cond + wave_cfg_strength * (wave_v_cond - wave_v_uncond)
                if wave_cfg_rescale > 0.0:
                    mix = max(0.0, min(wave_cfg_rescale, 1.0))
                    wave_velocity = self._rescale_guided_velocity(
                        wave_velocity,
                        wave_v_cond,
                        target_start=prompt_patches_count,
                        mix=mix,
                    )
                return wave_velocity

            wave_pred = self.wave_fm_decoder(
                wave_current,
                z0_cond,
                t,
                valid_wave_mask=valid_wave_mask,
            )
            return wave_velocity_from_x0(wave_pred, wave_current, time_value)

        def apply_prior_velocity(z_current: Tensor, z_velocity: Tensor, scale: Tensor) -> Tensor:
            z_next = z_current.clone()
            z_next[:, prompt_tokens:] = z_next[:, prompt_tokens:] + scale * z_velocity[:, prompt_tokens:]
            return with_prompt_z(z_next)

        def apply_wave_velocity(wave_current: Tensor, wave_velocity: Tensor, scale: Tensor) -> Tensor:
            wave_next = wave_current.clone()
            wave_next[:, prompt_patches_count:] = (
                wave_next[:, prompt_patches_count:]
                + scale.to(dtype=wave_next.dtype) * wave_velocity[:, prompt_patches_count:]
            )
            return with_prompt_wave(wave_next)

        if sampler == "twostage":
            prior_time_grid = self._flow_inference_time_grid(
                prior_step_count,
                device=z_prompt.device,
                dtype=z_prompt.dtype,
            )
            for step_idx in range(prior_step_count):
                time_value = float(prior_time_grid[step_idx].item())
                next_time_value = float(prior_time_grid[step_idx + 1].item())
                dt = (prior_time_grid[step_idx + 1] - prior_time_grid[step_idx]).to(dtype=z_state.dtype)
                if solver == "euler":
                    k1_z = eval_prior_velocity(z_state, time_value)
                    z_state = apply_prior_velocity(z_state, k1_z, dt)
                elif solver == "heun":
                    k1_z = eval_prior_velocity(z_state, time_value)
                    z_euler = apply_prior_velocity(z_state, k1_z, dt)
                    k2_z = eval_prior_velocity(z_euler, next_time_value)
                    z_state = apply_prior_velocity(z_state, 0.5 * (k1_z + k2_z), dt)
                else:
                    mid_time_value = 0.5 * (time_value + next_time_value)
                    half_dt = dt * 0.5
                    k1_z = eval_prior_velocity(z_state, time_value)
                    k2_z = eval_prior_velocity(apply_prior_velocity(z_state, k1_z, half_dt), mid_time_value)
                    k3_z = eval_prior_velocity(apply_prior_velocity(z_state, k2_z, half_dt), mid_time_value)
                    k4_z = eval_prior_velocity(apply_prior_velocity(z_state, k3_z, dt), next_time_value)
                    z_state = apply_prior_velocity(
                        z_state,
                        (k1_z + 2.0 * k2_z + 2.0 * k3_z + k4_z) / 6.0,
                        dt,
                    )

            z0_hat = with_prompt_z(z_state)
            wave_time_grid = self._flow_inference_time_grid(
                wave_step_count,
                device=z_prompt.device,
                dtype=z_prompt.dtype,
            )
            for step_idx in range(wave_step_count):
                time_value = float(wave_time_grid[step_idx].item())
                next_time_value = float(wave_time_grid[step_idx + 1].item())
                dt = (wave_time_grid[step_idx + 1] - wave_time_grid[step_idx]).to(dtype=wave_state.dtype)
                if solver == "euler":
                    k1_w = eval_wave_velocity(wave_state, z0_hat, time_value)
                    wave_state = apply_wave_velocity(wave_state, k1_w, dt)
                elif solver == "heun":
                    k1_w = eval_wave_velocity(wave_state, z0_hat, time_value)
                    w_euler = apply_wave_velocity(wave_state, k1_w, dt)
                    k2_w = eval_wave_velocity(w_euler, z0_hat, next_time_value)
                    wave_state = apply_wave_velocity(wave_state, 0.5 * (k1_w + k2_w), dt)
                else:
                    mid_time_value = 0.5 * (time_value + next_time_value)
                    half_dt = dt * 0.5
                    k1_w = eval_wave_velocity(wave_state, z0_hat, time_value)
                    k2_w = eval_wave_velocity(apply_wave_velocity(wave_state, k1_w, half_dt), z0_hat, mid_time_value)
                    k3_w = eval_wave_velocity(apply_wave_velocity(wave_state, k2_w, half_dt), z0_hat, mid_time_value)
                    k4_w = eval_wave_velocity(apply_wave_velocity(wave_state, k3_w, dt), z0_hat, next_time_value)
                    wave_state = apply_wave_velocity(
                        wave_state,
                        (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w) / 6.0,
                        dt,
                    )

            generated_model = self.patchify.unpatchify(with_prompt_wave(wave_state), original_len=total_len)
            return self._unscale_waveform(generated_model)

        def eval_velocity(z_current: Tensor, wave_current: Tensor, time_value: float) -> tuple[Tensor, Tensor]:
            z_current = with_prompt_z(z_current)
            wave_current = with_prompt_wave(wave_current)
            t = torch.full((batch_size,), time_value, device=z_current.device, dtype=z_current.dtype)
            if prior_cfg_strength > 0.0:
                condition_dropout_mask = torch.cat(
                    (
                        torch.zeros(batch_size, device=z_current.device, dtype=torch.bool),
                        torch.ones(batch_size, device=z_current.device, dtype=torch.bool),
                    ),
                    dim=0,
                )
                z_cat = torch.cat((z_current, z_current), dim=0)
                t_cat = torch.cat((t, t), dim=0)
                z_pred_cond, z_pred_uncond = prior_x0(
                    z_cat,
                    t_cat,
                    condition_dropout_mask=condition_dropout_mask,
                ).chunk(2, dim=0)
                z_v_cond = z_velocity_from_x0(z_pred_cond, z_current, time_value)
                z_v_uncond = z_velocity_from_x0(z_pred_uncond, z_current, time_value)
                z_velocity = z_v_cond + prior_cfg_strength * (z_v_cond - z_v_uncond)
                if prior_cfg_rescale > 0.0:
                    mix = max(0.0, min(prior_cfg_rescale, 1.0))
                    z_velocity = self._rescale_guided_velocity(
                        z_velocity,
                        z_v_cond,
                        target_start=prompt_tokens,
                        mix=mix,
                    )
                z_for_wave = z_pred_cond
            else:
                z_for_wave = prior_x0(z_current, t, condition_dropout_mask=None)
                z_velocity = z_velocity_from_x0(z_for_wave, z_current, time_value)

            z0_cond = z_to_wave_cond(z_for_wave)
            if wave_cfg_strength > 0.0:
                wave_pred_cond, wave_pred_uncond = self.wave_fm_decoder(
                    torch.cat((wave_current, wave_current), dim=0),
                    torch.cat((z0_cond, torch.zeros_like(z0_cond)), dim=0),
                    torch.cat((t, t), dim=0),
                    valid_wave_mask=torch.cat((valid_wave_mask, valid_wave_mask), dim=0),
                ).chunk(2, dim=0)
                wave_v_cond = wave_velocity_from_x0(wave_pred_cond, wave_current, time_value)
                wave_v_uncond = wave_velocity_from_x0(wave_pred_uncond, wave_current, time_value)
                wave_velocity = wave_v_cond + wave_cfg_strength * (wave_v_cond - wave_v_uncond)
                if wave_cfg_rescale > 0.0:
                    mix = max(0.0, min(wave_cfg_rescale, 1.0))
                    wave_velocity = self._rescale_guided_velocity(
                        wave_velocity,
                        wave_v_cond,
                        target_start=prompt_patches_count,
                        mix=mix,
                    )
                return z_velocity, wave_velocity

            wave_pred = self.wave_fm_decoder(
                wave_current,
                z0_cond,
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

        time_grid = self._flow_inference_time_grid(
            num_steps,
            device=z_prompt.device,
            dtype=z_prompt.dtype,
        )
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

    @torch.no_grad()
    def generate_fm_decoder_full(
        self,
        text_token_ids: Tensor,
        *,
        num_tokens: int,
        num_steps: int,
        prior_steps: int | None = None,
        wave_steps: int | None = None,
        sample_batch: dict[str, Any],
        text_token_lengths: Tensor | None = None,
        cfg_strength: float = 0.0,
        cfg_rescale: float = 0.0,
        prior_cfg_strength: float | None = None,
        prior_cfg_rescale: float | None = None,
        wave_cfg_strength: float | None = None,
        wave_cfg_rescale: float | None = None,
        solver: str = "heun",
    ) -> Tensor:
        solver = str(solver).lower()
        if solver not in {"euler", "heun", "rk4"}:
            raise ValueError("solver must be 'euler', 'heun', or 'rk4'")
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        prior_step_count = int(num_steps if prior_steps is None else prior_steps)
        wave_step_count = int(num_steps if wave_steps is None else wave_steps)
        if prior_step_count < 1:
            raise ValueError("prior_steps must be positive")
        if wave_step_count < 1:
            raise ValueError("wave_steps must be positive")
        if num_tokens < 1:
            raise ValueError("num_tokens must be positive")
        prior_cfg_strength = float(cfg_strength if prior_cfg_strength is None else prior_cfg_strength)
        prior_cfg_rescale = float(cfg_rescale if prior_cfg_rescale is None else prior_cfg_rescale)
        wave_cfg_strength = float(0.0 if wave_cfg_strength is None else wave_cfg_strength)
        wave_cfg_rescale = float(0.0 if wave_cfg_rescale is None else wave_cfg_rescale)

        param = next(self.parameters())
        device = param.device
        dtype = param.dtype
        text_token_ids = text_token_ids.to(device=device, dtype=torch.long)
        if text_token_lengths is not None:
            text_token_lengths = text_token_lengths.to(device=device, dtype=torch.long)
        sample_batch = self._move_batch_to_device(sample_batch, device)

        batch_size = int(text_token_ids.shape[0])
        total_tokens = int(num_tokens)
        factor = int(self.config.downsample_factor)
        patch_size = int(self.config.patch_size)
        total_patches = total_tokens * factor
        total_len = total_patches * patch_size
        valid_latent_mask = torch.ones((batch_size, total_tokens), device=device, dtype=torch.bool)
        valid_wave_mask = torch.ones((batch_size, total_patches), device=device, dtype=torch.bool)

        z_state = torch.randn(
            batch_size,
            total_tokens,
            self.config.latent_dim,
            device=device,
            dtype=dtype,
        )
        wave_state = torch.randn(
            batch_size,
            total_patches,
            patch_size,
            device=device,
            dtype=dtype,
        )

        text_tokens = self._encode_text_ids(
            text_token_ids,
            text_lengths=text_token_lengths,
            device=device,
        )
        aligned_text, _ = self._aligned_text_condition(
            sample_batch,
            text_tokens,
            num_latents=total_tokens,
            text_lengths=text_token_lengths,
            valid_latent_mask=valid_latent_mask,
        )
        speaker_tokens, speaker_token_mask = self.speaker_encoder(
            sample_batch["ref_wav_clean"].to(device=device, dtype=torch.float32),
            sample_batch["ref_wav_valid_mask"].to(device=device, dtype=torch.bool),
        )
        speaker_tokens = speaker_tokens.to(dtype=dtype)
        aligned_text_cfg = torch.cat((aligned_text, aligned_text), dim=0)
        speaker_tokens_cfg = torch.cat((speaker_tokens, speaker_tokens), dim=0)
        speaker_token_mask_cfg = torch.cat((speaker_token_mask, speaker_token_mask), dim=0)
        valid_latent_mask_cfg = torch.cat((valid_latent_mask, valid_latent_mask), dim=0)

        def z_to_wave_cond(z0_hat: Tensor) -> Tensor:
            cond = z0_hat.repeat_interleave(factor, dim=1)
            if cond.shape[1] < total_patches:
                cond = F.pad(cond, (0, 0, 0, total_patches - cond.shape[1]))
            return cond[:, :total_patches]

        def prior_x0(
            z_current: Tensor,
            t: Tensor,
            *,
            condition_dropout_mask: Tensor | None,
        ) -> Tensor:
            is_cfg = z_current.shape[0] != batch_size
            h = self.fm_encoder(
                z_current,
                aligned_text_cfg if is_cfg else aligned_text,
                speaker_tokens_cfg if is_cfg else speaker_tokens,
                speaker_token_mask_cfg if is_cfg else speaker_token_mask,
                t,
                valid_audio_mask=valid_latent_mask_cfg if is_cfg else valid_latent_mask,
                condition_dropout_mask=condition_dropout_mask,
            )
            return self._normalize_flow_x_pred(self.fm_head(h))

        def velocity_from_x0(x0_pred: Tensor, state: Tensor, time_value: float) -> Tensor:
            denom = max(float(self.config.flow_xpred_denom_min), 1.0 - float(time_value))
            return (x0_pred - state) / denom

        def eval_prior_velocity(z_current: Tensor, time_value: float) -> Tensor:
            t = torch.full((batch_size,), time_value, device=device, dtype=dtype)
            if prior_cfg_strength > 0.0:
                condition_dropout_mask = torch.cat(
                    (
                        torch.zeros(batch_size, device=device, dtype=torch.bool),
                        torch.ones(batch_size, device=device, dtype=torch.bool),
                    ),
                    dim=0,
                )
                z_cat = torch.cat((z_current, z_current), dim=0)
                t_cat = torch.cat((t, t), dim=0)
                z_pred_cond, z_pred_uncond = prior_x0(
                    z_cat,
                    t_cat,
                    condition_dropout_mask=condition_dropout_mask,
                ).chunk(2, dim=0)
                z_v_cond = velocity_from_x0(z_pred_cond, z_current, time_value)
                z_v_uncond = velocity_from_x0(z_pred_uncond, z_current, time_value)
                z_velocity = z_v_cond + prior_cfg_strength * (z_v_cond - z_v_uncond)
                if prior_cfg_rescale > 0.0:
                    mix = max(0.0, min(prior_cfg_rescale, 1.0))
                    z_velocity = self._rescale_guided_velocity(
                        z_velocity,
                        z_v_cond,
                        target_start=0,
                        mix=mix,
                    )
                return z_velocity
            return velocity_from_x0(prior_x0(z_current, t, condition_dropout_mask=None), z_current, time_value)

        def eval_wave_velocity(wave_current: Tensor, z0_hat: Tensor, time_value: float) -> Tensor:
            t = torch.full((batch_size,), time_value, device=device, dtype=dtype)
            z0_cond = z_to_wave_cond(z0_hat)
            if wave_cfg_strength > 0.0:
                wave_pred_cond, wave_pred_uncond = self.wave_fm_decoder(
                    torch.cat((wave_current, wave_current), dim=0),
                    torch.cat((z0_cond, torch.zeros_like(z0_cond)), dim=0),
                    torch.cat((t, t), dim=0),
                    valid_wave_mask=torch.cat((valid_wave_mask, valid_wave_mask), dim=0),
                ).chunk(2, dim=0)
                wave_v_cond = velocity_from_x0(wave_pred_cond, wave_current, time_value)
                wave_v_uncond = velocity_from_x0(wave_pred_uncond, wave_current, time_value)
                wave_velocity = wave_v_cond + wave_cfg_strength * (wave_v_cond - wave_v_uncond)
                if wave_cfg_rescale > 0.0:
                    mix = max(0.0, min(wave_cfg_rescale, 1.0))
                    wave_velocity = self._rescale_guided_velocity(
                        wave_velocity,
                        wave_v_cond,
                        target_start=0,
                        mix=mix,
                    )
                return wave_velocity

            wave_pred = self.wave_fm_decoder(
                wave_current,
                z0_cond,
                t,
                valid_wave_mask=valid_wave_mask,
            )
            return velocity_from_x0(wave_pred, wave_current, time_value)

        def solver_step(state: Tensor, eval_velocity: Any, time_grid: Tensor, step_idx: int) -> Tensor:
            time_value = float(time_grid[step_idx].item())
            next_time_value = float(time_grid[step_idx + 1].item())
            dt = (time_grid[step_idx + 1] - time_grid[step_idx]).to(dtype=state.dtype)
            if solver == "euler":
                return state + dt * eval_velocity(state, time_value)
            if solver == "heun":
                k1 = eval_velocity(state, time_value)
                k2 = eval_velocity(state + dt * k1, next_time_value)
                return state + dt * 0.5 * (k1 + k2)
            mid_time_value = 0.5 * (time_value + next_time_value)
            half_dt = dt * 0.5
            k1 = eval_velocity(state, time_value)
            k2 = eval_velocity(state + half_dt * k1, mid_time_value)
            k3 = eval_velocity(state + half_dt * k2, mid_time_value)
            k4 = eval_velocity(state + dt * k3, next_time_value)
            return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

        prior_time_grid = self._flow_inference_time_grid(prior_step_count, device=device, dtype=dtype)
        for step_idx in range(prior_step_count):
            z_state = solver_step(z_state, eval_prior_velocity, prior_time_grid, step_idx)

        z0_hat = z_state
        wave_time_grid = self._flow_inference_time_grid(wave_step_count, device=device, dtype=dtype)
        for step_idx in range(wave_step_count):
            wave_state = solver_step(
                wave_state,
                lambda state, time_value: eval_wave_velocity(state, z0_hat, time_value),
                wave_time_grid,
                step_idx,
            )

        generated_model = self.patchify.unpatchify(wave_state, original_len=total_len)
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

    def _aligned_text_condition(
        self,
        batch: dict[str, Any],
        text_tokens: Tensor,
        *,
        num_latents: int,
        text_lengths: Tensor | None,
        valid_latent_mask: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        return self.megatts_text_conditioner(
            batch,
            text_tokens,
            num_latents=num_latents,
            text_lengths=text_lengths,
            valid_latent_mask=valid_latent_mask,
        )


class LatentFMFMMaskedPrior(LatentFMFM):
    """Masked visible-conditioned prior FM plus waveform FM; speaker is wave-only."""

    _encode_split_latents = LatentTTSSplit._encode_split_latents
    _encode_random_span_split_latents = LatentTTSSplit._encode_random_span_split_latents
    _align_prefix_patch_counts_to_downsample_grid = staticmethod(
        LatentTTSSplit._align_prefix_patch_counts_to_downsample_grid
    )
    _encode_patch_segment = LatentTTSSplit._encode_patch_segment
    _encode_patch_segments_batch = LatentTTSSplit._encode_patch_segments_batch
    _encoder_modules = LatentTTSSplit._encoder_modules
    _target_encoder_ema_modules = LatentTTSSplit._target_encoder_ema_modules
    update_internal_ema = LatentTTSSplit.update_internal_ema
    _target_encoder_online_ema_pairs = LatentTTSSplit._target_encoder_online_ema_pairs
    _update_ema_module = staticmethod(LatentTTSSplit._update_ema_module)
    _nonempty_attention_mask = staticmethod(LatentTTSSplit._nonempty_attention_mask)

    def __init__(self, config: LatentFMFMMaskedPriorConfig) -> None:
        MaskedFlowWaveTokenTTS.__init__(self, config)
        self.config: LatentFMFMMaskedPriorConfig
        self.megatts_text_conditioner = MegaTTSTextConditioner(
            text_dim=config.text_dim,
            hidden_dim=config.fm_hidden_dim,
            downsample_factor=config.downsample_factor,
            alignment_mode=config.megatts_alignment_mode,
            anchor_mode=config.megatts_anchor_mode,
            anchor_ratio=config.megatts_anchor_ratio,
        )
        self.fm_encoder = MaskedMegaTTSAddFMEncoder(
            token_dim=config.latent_dim,
            hidden_dim=config.fm_hidden_dim,
            depth=config.fm_depth,
            adaln_every=config.fm_adaln_every,
            heads=config.fm_heads,
            dim_head=config.fm_dim_head,
            ffn_mult=config.fm_ffn_mult,
            dropout=config.dropout,
            rope_base=config.rope_base,
            input_fusion=config.prior_fm_input_fusion,
        )
        self.fm_head = FMHead(config.fm_hidden_dim, config.latent_dim)
        self.wave_speaker_encoder: nn.Module | None = None
        if config.wave_speaker_encoder_type == "campplus":
            self.wave_speaker_encoder = SpeakerCAMPPlusSequenceEncoder(
                output_dim=config.wave_speaker_embedding_dim,
                sample_rate=16000,
                num_mel_bins=80,
                window_size=config.wave_speaker_pool_window_size,
                stride=config.wave_speaker_pool_stride,
                embedding_size=config.wave_speaker_embedding_dim,
                pretrained_path=config.wave_speaker_pretrained_path,
                mean_norm=config.wave_speaker_mean_norm,
                freeze_backbone=True,
                profile=config.wave_speaker_profile,
            )
            self.wave_speaker_encoder.out_proj = nn.Identity()
            self.wave_speaker_encoder.requires_grad_(False)
            self.wave_speaker_encoder.eval()
        self.wave_fm_decoder: nn.Module | None = None
        if config.decoder_objective == "wave_fm":
            if config.wave_fm_backbone == "linear_udit_unet":
                if config.decoder_head != "patch":
                    raise ValueError("linear_udit_unet wave_fm_backbone requires decoder_head='patch'")
                self.wave_fm_decoder = LinearUDiTUNetWaveFMDecoder(
                    patch_size=config.patch_size,
                    cond_dim=config.latent_dim,
                    hidden_dim=config.wave_fm_hidden_dim,
                    depths=config.wave_fm_unet_depths,
                    adaln_every=config.wave_fm_adaln_every,
                    heads=config.wave_fm_heads,
                    dim_head=config.wave_fm_dim_head,
                    ffn_mult=config.wave_fm_ffn_mult,
                    dropout=config.dropout,
                    rope_base=config.rope_base,
                    input_fusion=config.wave_fm_input_fusion,
                    speaker_dim=config.wave_speaker_embedding_dim,
                )
            else:
                self.wave_fm_decoder = WaveFMDecoderHead(
                    patch_size=config.patch_size,
                    cond_dim=config.latent_dim,
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

    def decode_tokens(self, z: Tensor, original_len: int | None = None) -> Tensor:
        if self.config.decoder_objective != "wave":
            raise NotImplementedError("LatentFMFMMaskedPrior deterministic decode requires decoder.type='deterministic'")
        return MaskedFlowWaveTokenTTS.decode_tokens(self, z, original_len=original_len)

    def decode_tokens_raw(self, z: Tensor, original_len: int | None = None) -> Tensor:
        return self._unscale_waveform(self.decode_tokens(z, original_len=original_len))

    @torch.no_grad()
    def _wave_speaker_condition(
        self,
        batch: dict[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
        fallback_wav: Tensor,
        fallback_mask: Tensor | None,
    ) -> tuple[Tensor | None, Tensor, dict[str, Tensor]]:
        zero = torch.zeros((), device=device, dtype=dtype)
        if self.wave_speaker_encoder is None:
            return None, zero, {}
        wav = batch.get("ref_wav_clean")
        mask = batch.get("ref_wav_valid_mask")
        if wav is None or mask is None:
            wav = fallback_wav
            if fallback_mask is None:
                mask = torch.ones_like(fallback_wav, dtype=torch.bool)
            else:
                mask = fallback_mask
        wav = wav.to(device=device, dtype=torch.float32)
        mask = mask.to(device=device, dtype=torch.bool)
        tokens, token_mask = self.wave_speaker_encoder(wav, mask)
        token_mask = token_mask.to(device=device, dtype=torch.bool)
        weights = token_mask.to(device=device, dtype=tokens.dtype)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (tokens * weights.unsqueeze(-1)).sum(dim=1) / denom
        pooled = pooled.to(device=device, dtype=dtype)
        metrics = {
            f"wave_{key}": torch.tensor(value, device=device, dtype=dtype)
            for key, value in getattr(self.wave_speaker_encoder, "last_timing", {}).items()
        }
        return pooled, token_mask.detach().float().sum(dim=1).mean(), metrics

    def forward(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        profile_metrics: dict[str, Tensor] = {}
        total_start = self._profile_start(next(iter(batch.values())))
        timer_start = total_start
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
        z_raw = z_clean
        self._profile_stop(profile_metrics, "profile_encoder_sec", timer_start, z_clean)

        timer_start = self._profile_start(z_clean)
        text_token_ids, text_lengths = self._condition_text_inputs(
            batch,
            mask=mask,
            valid_latent_mask=valid_latent_mask,
            prompt_patch_ends=prompt_patch_ends,
        )
        text_tokens = self._encode_text_ids(text_token_ids, text_lengths=text_lengths)
        self._profile_stop(profile_metrics, "profile_text_sec", timer_start, text_tokens)

        timer_start = self._profile_start(z_clean)
        wave_speaker_emb, wave_speaker_token_count, wave_speaker_metrics = self._wave_speaker_condition(
            batch,
            device=wav_clean.device,
            dtype=z_clean.dtype,
            fallback_wav=wav_clean_raw,
            fallback_mask=batch.get("wav_valid_mask"),
        )
        self._profile_stop(profile_metrics, "profile_wave_speaker_sec", timer_start, z_clean)
        profile_metrics.update(wave_speaker_metrics)

        prior_cfg_dropout = float(getattr(self.config, "prior_cfg_dropout", self.config.cfg_dropout))
        if self.training and prior_cfg_dropout > 0.0:
            prior_condition_dropout_mask = (
                torch.rand(z_clean.shape[0], device=z_clean.device)
                < prior_cfg_dropout
            )
        else:
            prior_condition_dropout_mask = torch.zeros(z_clean.shape[0], device=z_clean.device, dtype=torch.bool)
        wave_cfg_dropout = float(getattr(self.config, "wave_cfg_dropout", 0.0))
        if self.training and wave_cfg_dropout > 0.0:
            wave_condition_dropout_mask = (
                torch.rand(z_clean.shape[0], device=z_clean.device)
                < wave_cfg_dropout
            )
        else:
            wave_condition_dropout_mask = torch.zeros(z_clean.shape[0], device=z_clean.device, dtype=torch.bool)

        stopgrad_prior_wave_online = self.config.fmfm_training_mode == "prior_stopgrad_wave_online_encoder"
        (
            prior_fm_loss,
            z_x,
            z_cond,
            z_pred,
            flow_time,
            z_pred_flow,
            alignment_metrics,
        ) = self._profile_call(
            profile_metrics,
            "profile_prior_sec",
            z_clean,
            self._masked_prior_flow_loss,
            batch,
            z_clean,
            z_flow_target,
            text_tokens,
            text_lengths=text_lengths,
            mask=mask,
            valid_latent_mask=valid_latent_mask,
            condition_dropout_mask=prior_condition_dropout_mask,
        )

        zero = wav_clean.new_tensor(0.0)
        valid_sample_mask = _sample_mask(batch.get("wav_valid_mask"), wav_clean_raw.shape).to(
            device=wav_clean.device,
            dtype=torch.bool,
        )
        wave_condition_noise_rms = zero
        wave_fm_loss = zero
        recon_loss = zero
        recon_loss_raw_metric = zero
        mel_loss_raw_metric = zero
        decoder_noise_rms = zero
        decoder_noise_applied_fraction = zero
        wave_pred_patches = patches.new_zeros(patches.shape)
        wave_x = patches.new_zeros(patches.shape)
        wave_target_mask = torch.zeros_like(valid_patch_mask)
        wave_valid_mask = valid_patch_mask

        if self.config.decoder_objective == "wave_fm":
            wave_condition_flow = z_pred_flow
            if stopgrad_prior_wave_online:
                wave_condition = z_clean
                noise_std = float(self.config.wave_fm_encoder_noise_std)
                noise_mode = str(
                    getattr(self.config, "decoder_latent_noise_mode", "additive")
                )
                noise_probability = float(
                    getattr(self.config, "decoder_latent_noise_prob", 1.0)
                )
                if noise_mode not in {"none", "additive"}:
                    raise ValueError(
                        "FM decoder condition noise must be 'none' or 'additive'"
                    )
                if (
                    self.training
                    and noise_mode == "additive"
                    and noise_std > 0.0
                    and noise_probability > 0.0
                ):
                    noise = torch.randn_like(wave_condition) * noise_std
                    noise = noise * valid_latent_mask.unsqueeze(-1).to(dtype=noise.dtype)
                    apply_mask = (
                        torch.rand(wave_condition.shape[0], device=wave_condition.device)
                        < noise_probability
                    ).view(wave_condition.shape[0], 1, 1)
                    noise = torch.where(apply_mask, noise, torch.zeros_like(noise))
                    decoder_noise_applied_fraction = apply_mask.detach().float().mean()
                    wave_condition = wave_condition + noise
                    selected_noise = noise[valid_latent_mask]
                    if selected_noise.numel() > 0:
                        wave_condition_noise_rms = selected_noise.float().pow(2).mean().sqrt()
                    decoder_noise_rms = wave_condition_noise_rms
                wave_condition_flow = _repeat_steps(wave_condition, int(self.config.flow_steps_per_recon))

            (
                wave_fm_loss,
                wave_pred_patches,
                wave_x,
                wave_target_mask,
                wave_valid_mask,
            ) = self._profile_call(
                profile_metrics,
                "profile_wave_sec",
                patches,
                self._wave_flow_loss,
                patches,
                wave_condition_flow,
                flow_time.reshape(-1),
                valid_patch_mask=valid_patch_mask,
                condition_dropout_mask=wave_condition_dropout_mask,
                speaker_emb=wave_speaker_emb,
            )

            timer_start = self._profile_start(wave_pred_patches)
            pred_patches = torch.where(wave_target_mask.unsqueeze(-1), wave_pred_patches, patches)
            pred_patches = pred_patches * valid_patch_mask.unsqueeze(-1).to(dtype=pred_patches.dtype)
            teacher_waveform_model = self.patchify.unpatchify(pred_patches, original_len=original_len)
            teacher_waveform = self._unscale_waveform(teacher_waveform_model)
            target_sample_mask = _sample_mask_from_patch_mask(
                wave_target_mask & wave_valid_mask,
                patch_size=self.config.patch_size,
                original_len=original_len,
            ).to(device=wav_clean.device, dtype=torch.bool)
            mel_loss = (
                self.mel_loss_fn(teacher_waveform_model, wav_clean, sample_mask=target_sample_mask)
                if self.config.lambda_mel != 0.0
                else zero
            )
            mel_loss_raw_metric = mel_loss.detach()
            self._profile_stop(profile_metrics, "profile_mel_sec", timer_start, mel_loss)
        else:
            timer_start = self._profile_start(z_pred)
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
            target_sample_mask = valid_sample_mask
            self._profile_stop(profile_metrics, "profile_mel_sec", timer_start, mel_loss)

        if stopgrad_prior_wave_online or self.config.lambda_latent_reg == 0.0:
            latent_reg_loss = zero
            latent_reg_metrics = {}
        else:
            reg_target = {
                "z_raw": z_raw,
                "z_clean": z_clean,
                "z_flow_target": z_flow_target,
                "z_pred": z_pred,
            }[str(self.config.latent_reg_target)]
            latent_reg_loss = self._latent_regularization_loss(
                reg_target,
                valid_latent_mask,
                target_latent_mask=mask,
            )
            latent_reg_metrics = self._latent_regularization_metrics(
                reg_target,
                valid_latent_mask,
                mask,
            )

        total_loss = (
            self.config.lambda_flow * prior_fm_loss
            + self.config.lambda_wave_fm * wave_fm_loss
            + self.config.lambda_recon * recon_loss
            + self.config.lambda_mel * mel_loss
            + self.config.lambda_latent_reg * latent_reg_loss
        )
        self._profile_stop(profile_metrics, "profile_forward_total_sec", total_start, total_loss)
        return {
            "loss": total_loss,
            "flow_loss": prior_fm_loss,
            "prior_fm_loss": prior_fm_loss,
            "wave_fm_loss": wave_fm_loss,
            "recon_loss": recon_loss,
            "mel_loss": mel_loss,
            "latent_reg_loss": latent_reg_loss,
            "weighted_latent_reg_loss": latent_reg_loss * float(self.config.lambda_latent_reg),
            "recon_loss_raw_metric": recon_loss_raw_metric,
            "mel_loss_raw_metric": mel_loss_raw_metric,
            "decoder_latent_noise_rms": decoder_noise_rms,
            "decoder_latent_noise_applied_fraction": decoder_noise_applied_fraction,
            "z_decode": z_decode if self.config.decoder_objective == "wave" else z_pred,
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
            "condition_dropout_fraction": prior_condition_dropout_mask.detach().float().mean(),
            "prior_condition_dropout_fraction": prior_condition_dropout_mask.detach().float().mean(),
            "wave_condition_dropout_fraction": wave_condition_dropout_mask.detach().float().mean(),
            "fmfm_training_mode_id": wav_clean.new_tensor(1.0 if stopgrad_prior_wave_online else 0.0),
            "prior_encoder_stop_gradient": wav_clean.new_tensor(0.0),
            "wave_encoder_online": wav_clean.new_tensor(1.0 if stopgrad_prior_wave_online else 0.0),
            "wave_fm_condition_source_id": wav_clean.new_tensor(1.0 if stopgrad_prior_wave_online else 0.0),
            "wave_fm_encoder_noise_std": wav_clean.new_tensor(float(self.config.wave_fm_encoder_noise_std)),
            "wave_fm_condition_noise_rms": wave_condition_noise_rms.detach(),
            "mask": mask,
            "valid_latent_mask": valid_latent_mask,
            "z_clean": z_clean,
            "z_raw": z_raw,
            "z_flow_target": z_flow_target,
            "split_target_encoder_ema": wav_clean.new_tensor(float(self.config.split_target_encoder_ema)),
            "z_pred": z_pred,
            "speaker_token_count": wave_speaker_token_count.detach(),
            "valid_sample_mask": valid_sample_mask,
            **latent_reg_metrics,
            **profile_metrics,
            **alignment_metrics,
        }

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
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        steps = int(self.config.flow_steps_per_recon)
        batch_size = z_clean.shape[0]
        z_stop = z_flow_target.detach()
        z_target = z_stop + float(self.config.target_grad_scale) * (z_flow_target - z_stop)
        z_target_flow = _repeat_steps(z_target, steps)
        z_visible_flow = _repeat_steps(z_clean, steps)
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

        z_x_all = torch.where(prompt_flow.unsqueeze(-1), z_visible_flow, z_t)
        z_cond_all = torch.where(
            prompt_flow.unsqueeze(-1),
            z_visible_flow,
            torch.zeros_like(z_visible_flow),
        )
        z_x_all = z_x_all * valid_flow.unsqueeze(-1).to(dtype=z_x_all.dtype)
        z_cond_all = z_cond_all * valid_flow.unsqueeze(-1).to(dtype=z_cond_all.dtype)

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
            condition_dropout_mode=self.config.prior_cfg_dropout_mode,
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
            raise ValueError(f"unsupported prediction mode for LatentFMFMMaskedPrior prior: {self.config.prediction}")

        z_x = z_x_all.view(steps, batch_size, *z_clean.shape[1:])[-1]
        z_cond = z_cond_all.view(steps, batch_size, *z_clean.shape[1:])[-1]
        z_pred = z_pred_all.view(steps, batch_size, *z_clean.shape[1:])[-1]
        flow_time = t.view(steps, batch_size)
        return flow_loss, z_x, z_cond, z_pred, flow_time, z_pred_all, alignment_metrics

    @torch.no_grad()
    def generate_fm_decoder_prefix(
        self,
        prompt_wav: Tensor,
        text_token_ids: Tensor,
        *,
        target_num_tokens: int,
        num_steps: int,
        prior_steps: int | None = None,
        wave_steps: int | None = None,
        sample_batch: dict[str, Any],
        text_token_lengths: Tensor | None = None,
        cfg_strength: float = 0.0,
        cfg_rescale: float = 0.0,
        prior_cfg_strength: float | None = None,
        prior_cfg_rescale: float | None = None,
        wave_cfg_strength: float | None = None,
        wave_cfg_rescale: float | None = None,
        solver: str = "heun",
        sampler: str = "joint",
    ) -> Tensor:
        solver = str(solver).lower()
        if solver not in {"euler", "heun", "rk4"}:
            raise ValueError("solver must be 'euler', 'heun', or 'rk4'")
        sampler = str(sampler).lower().replace("-", "_")
        if sampler in {"2stage", "two_stage", "sequential", "joint"}:
            sampler = "twostage"
        if sampler != "twostage":
            raise ValueError("LatentFMFMMaskedPrior inference uses sampler='twostage'")
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        prior_step_count = int(num_steps if prior_steps is None else prior_steps)
        wave_step_count = int(num_steps if wave_steps is None else wave_steps)
        if prior_step_count < 1:
            raise ValueError("prior_steps must be positive")
        if wave_step_count < 1:
            raise ValueError("wave_steps must be positive")
        if target_num_tokens < 1:
            raise ValueError("target_num_tokens must be positive")
        prior_cfg_strength = float(cfg_strength if prior_cfg_strength is None else prior_cfg_strength)
        prior_cfg_rescale = float(cfg_rescale if prior_cfg_rescale is None else prior_cfg_rescale)
        wave_cfg_strength = float(0.0 if wave_cfg_strength is None else wave_cfg_strength)
        wave_cfg_rescale = float(0.0 if wave_cfg_rescale is None else wave_cfg_rescale)

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
        z_prompt_full = torch.zeros_like(z_state)
        z_prompt_full[:, :prompt_tokens] = z_prompt
        prompt_audio_mask = torch.zeros((batch_size, total_tokens), device=z_prompt.device, dtype=torch.bool)
        prompt_audio_mask[:, :prompt_tokens] = True
        wave_state = torch.randn(
            batch_size,
            total_patches,
            patch_size,
            device=z_prompt.device,
            dtype=z_prompt.dtype,
        )
        wave_state[:, :prompt_patches_count] = prompt_patches.to(device=z_prompt.device, dtype=z_prompt.dtype)
        wave_prompt_full = torch.zeros_like(wave_state)
        wave_prompt_full[:, :prompt_patches_count] = prompt_patches.to(
            device=wave_state.device,
            dtype=wave_state.dtype,
        )

        valid_latent_mask = torch.ones(z_state.shape[:2], device=z_state.device, dtype=torch.bool)
        valid_wave_mask = torch.ones(wave_state.shape[:2], device=wave_state.device, dtype=torch.bool)
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
        aligned_text_cfg = torch.cat((aligned_text, aligned_text), dim=0)
        valid_latent_mask_cfg = torch.cat((valid_latent_mask, valid_latent_mask), dim=0)
        z_prompt_full_cfg = torch.cat((z_prompt_full, z_prompt_full), dim=0)
        prompt_audio_mask_cfg = torch.cat((prompt_audio_mask, prompt_audio_mask), dim=0)
        prompt_sample_mask = torch.ones_like(prompt_wav, device=z_prompt.device, dtype=torch.bool)
        wave_speaker_emb, _, _ = self._wave_speaker_condition(
            sample_batch,
            device=z_prompt.device,
            dtype=z_prompt.dtype,
            fallback_wav=prompt_wav.to(device=z_prompt.device),
            fallback_mask=prompt_sample_mask,
        )

        def with_prompt_z(state: Tensor) -> Tensor:
            state = state.clone()
            state[:, :prompt_tokens] = z_prompt
            return state

        def with_prompt_wave(state: Tensor) -> Tensor:
            state = state.clone()
            state[:, :prompt_patches_count] = wave_prompt_full[:, :prompt_patches_count]
            return state

        def velocity_from_x0(x0_pred: Tensor, state: Tensor, time_value: float) -> Tensor:
            denom = max(float(self.config.flow_xpred_denom_min), 1.0 - float(time_value))
            return (x0_pred - state) / denom

        def z_to_wave_cond(z0_hat: Tensor) -> Tensor:
            cond = z0_hat.repeat_interleave(factor, dim=1)
            if cond.shape[1] < total_patches:
                cond = F.pad(cond, (0, 0, 0, total_patches - cond.shape[1]))
            return cond[:, :total_patches]

        def prior_x0(
            z_current: Tensor,
            t: Tensor,
            *,
            condition_dropout_mask: Tensor | None,
        ) -> Tensor:
            is_cfg = z_current.shape[0] != batch_size
            h = self.fm_encoder(
                z_current,
                z_prompt_full_cfg if is_cfg else z_prompt_full,
                aligned_text_cfg if is_cfg else aligned_text,
                t,
                valid_audio_mask=valid_latent_mask_cfg if is_cfg else valid_latent_mask,
                prompt_audio_mask=prompt_audio_mask_cfg if is_cfg else prompt_audio_mask,
                condition_dropout_mask=condition_dropout_mask,
                condition_dropout_mode=self.config.prior_cfg_dropout_mode,
            )
            return self._normalize_flow_x_pred(self.fm_head(h))

        def eval_prior_velocity(z_current: Tensor, time_value: float) -> Tensor:
            z_current = with_prompt_z(z_current)
            t = torch.full((batch_size,), time_value, device=z_current.device, dtype=z_current.dtype)
            if prior_cfg_strength > 0.0:
                condition_dropout_mask = torch.cat(
                    (
                        torch.zeros(batch_size, device=z_current.device, dtype=torch.bool),
                        torch.ones(batch_size, device=z_current.device, dtype=torch.bool),
                    ),
                    dim=0,
                )
                z_cat = torch.cat((z_current, z_current), dim=0)
                t_cat = torch.cat((t, t), dim=0)
                z_pred_cond, z_pred_uncond = prior_x0(
                    z_cat,
                    t_cat,
                    condition_dropout_mask=condition_dropout_mask,
                ).chunk(2, dim=0)
                z_v_cond = velocity_from_x0(z_pred_cond, z_current, time_value)
                z_v_uncond = velocity_from_x0(z_pred_uncond, z_current, time_value)
                z_velocity = z_v_cond + prior_cfg_strength * (z_v_cond - z_v_uncond)
                if prior_cfg_rescale > 0.0:
                    mix = max(0.0, min(prior_cfg_rescale, 1.0))
                    z_velocity = self._rescale_guided_velocity(
                        z_velocity,
                        z_v_cond,
                        target_start=prompt_tokens,
                        mix=mix,
                    )
                return z_velocity
            z_pred = prior_x0(z_current, t, condition_dropout_mask=None)
            return velocity_from_x0(z_pred, z_current, time_value)

        def eval_wave_velocity(wave_current: Tensor, z0_hat: Tensor, time_value: float) -> Tensor:
            wave_current = with_prompt_wave(wave_current)
            z0_hat = with_prompt_z(z0_hat)
            t = torch.full((batch_size,), time_value, device=wave_current.device, dtype=wave_current.dtype)
            z0_cond = z_to_wave_cond(z0_hat)
            if wave_cfg_strength > 0.0:
                wave_kwargs: dict[str, Tensor | None] = {
                    "valid_wave_mask": torch.cat((valid_wave_mask, valid_wave_mask), dim=0),
                }
                if isinstance(self.wave_fm_decoder, LinearUDiTUNetWaveFMDecoder):
                    if wave_speaker_emb is None:
                        speaker_cat = None
                    else:
                        speaker_cat = torch.cat((wave_speaker_emb, torch.zeros_like(wave_speaker_emb)), dim=0)
                    wave_kwargs["speaker_emb"] = speaker_cat
                wave_pred_cond, wave_pred_uncond = self.wave_fm_decoder(
                    torch.cat((wave_current, wave_current), dim=0),
                    torch.cat((z0_cond, torch.zeros_like(z0_cond)), dim=0),
                    torch.cat((t, t), dim=0),
                    **wave_kwargs,
                ).chunk(2, dim=0)
                wave_v_cond = velocity_from_x0(wave_pred_cond, wave_current, time_value)
                wave_v_uncond = velocity_from_x0(wave_pred_uncond, wave_current, time_value)
                wave_velocity = wave_v_cond + wave_cfg_strength * (wave_v_cond - wave_v_uncond)
                if wave_cfg_rescale > 0.0:
                    mix = max(0.0, min(wave_cfg_rescale, 1.0))
                    wave_velocity = self._rescale_guided_velocity(
                        wave_velocity,
                        wave_v_cond,
                        target_start=prompt_patches_count,
                        mix=mix,
                    )
                return wave_velocity

            wave_kwargs = {"valid_wave_mask": valid_wave_mask}
            if isinstance(self.wave_fm_decoder, LinearUDiTUNetWaveFMDecoder):
                wave_kwargs["speaker_emb"] = wave_speaker_emb
            wave_pred = self.wave_fm_decoder(wave_current, z0_cond, t, **wave_kwargs)
            return velocity_from_x0(wave_pred, wave_current, time_value)

        def apply_prior_velocity(z_current: Tensor, z_velocity: Tensor, scale: Tensor) -> Tensor:
            z_next = z_current.clone()
            z_next[:, prompt_tokens:] = z_next[:, prompt_tokens:] + scale * z_velocity[:, prompt_tokens:]
            return with_prompt_z(z_next)

        def apply_wave_velocity(wave_current: Tensor, wave_velocity: Tensor, scale: Tensor) -> Tensor:
            wave_next = wave_current.clone()
            wave_next[:, prompt_patches_count:] = (
                wave_next[:, prompt_patches_count:]
                + scale.to(dtype=wave_next.dtype) * wave_velocity[:, prompt_patches_count:]
            )
            return with_prompt_wave(wave_next)

        prior_time_grid = self._flow_inference_time_grid(
            prior_step_count,
            device=z_prompt.device,
            dtype=z_prompt.dtype,
        )
        for step_idx in range(prior_step_count):
            time_value = float(prior_time_grid[step_idx].item())
            next_time_value = float(prior_time_grid[step_idx + 1].item())
            dt = (prior_time_grid[step_idx + 1] - prior_time_grid[step_idx]).to(dtype=z_state.dtype)
            if solver == "euler":
                k1_z = eval_prior_velocity(z_state, time_value)
                z_state = apply_prior_velocity(z_state, k1_z, dt)
            elif solver == "heun":
                k1_z = eval_prior_velocity(z_state, time_value)
                z_euler = apply_prior_velocity(z_state, k1_z, dt)
                k2_z = eval_prior_velocity(z_euler, next_time_value)
                z_state = apply_prior_velocity(z_state, 0.5 * (k1_z + k2_z), dt)
            else:
                mid_time_value = 0.5 * (time_value + next_time_value)
                half_dt = dt * 0.5
                k1_z = eval_prior_velocity(z_state, time_value)
                k2_z = eval_prior_velocity(apply_prior_velocity(z_state, k1_z, half_dt), mid_time_value)
                k3_z = eval_prior_velocity(apply_prior_velocity(z_state, k2_z, half_dt), mid_time_value)
                k4_z = eval_prior_velocity(apply_prior_velocity(z_state, k3_z, dt), next_time_value)
                z_state = apply_prior_velocity(
                    z_state,
                    (k1_z + 2.0 * k2_z + 2.0 * k3_z + k4_z) / 6.0,
                    dt,
                )

        z0_hat = with_prompt_z(z_state)
        if self.config.decoder_objective == "wave":
            # The deterministic decoder is conditioned only on the sampled
            # prior latents. It deliberately bypasses the waveform FM stage.
            return self.decode_tokens_raw(z0_hat, original_len=total_len)
        if self.wave_fm_decoder is None:
            raise RuntimeError("FM decoder sampling requires wave_fm_decoder")
        wave_time_grid = self._flow_inference_time_grid(
            wave_step_count,
            device=z_prompt.device,
            dtype=z_prompt.dtype,
        )
        for step_idx in range(wave_step_count):
            time_value = float(wave_time_grid[step_idx].item())
            next_time_value = float(wave_time_grid[step_idx + 1].item())
            dt = (wave_time_grid[step_idx + 1] - wave_time_grid[step_idx]).to(dtype=wave_state.dtype)
            if solver == "euler":
                k1_w = eval_wave_velocity(wave_state, z0_hat, time_value)
                wave_state = apply_wave_velocity(wave_state, k1_w, dt)
            elif solver == "heun":
                k1_w = eval_wave_velocity(wave_state, z0_hat, time_value)
                w_euler = apply_wave_velocity(wave_state, k1_w, dt)
                k2_w = eval_wave_velocity(w_euler, z0_hat, next_time_value)
                wave_state = apply_wave_velocity(wave_state, 0.5 * (k1_w + k2_w), dt)
            else:
                mid_time_value = 0.5 * (time_value + next_time_value)
                half_dt = dt * 0.5
                k1_w = eval_wave_velocity(wave_state, z0_hat, time_value)
                k2_w = eval_wave_velocity(apply_wave_velocity(wave_state, k1_w, half_dt), z0_hat, mid_time_value)
                k3_w = eval_wave_velocity(apply_wave_velocity(wave_state, k2_w, half_dt), z0_hat, mid_time_value)
                k4_w = eval_wave_velocity(apply_wave_velocity(wave_state, k3_w, dt), z0_hat, next_time_value)
                wave_state = apply_wave_velocity(
                    wave_state,
                    (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w) / 6.0,
                    dt,
                )

        generated_model = self.patchify.unpatchify(with_prompt_wave(wave_state), original_len=total_len)
        return self._unscale_waveform(generated_model)

    @torch.no_grad()
    def generate_fm_decoder_full(
        self,
        text_token_ids: Tensor,
        *,
        num_tokens: int,
        num_steps: int,
        prior_steps: int | None = None,
        wave_steps: int | None = None,
        sample_batch: dict[str, Any],
        text_token_lengths: Tensor | None = None,
        cfg_strength: float = 0.0,
        cfg_rescale: float = 0.0,
        solver: str = "heun",
    ) -> Tensor:
        del text_token_ids, num_tokens, num_steps, prior_steps, wave_steps
        del sample_batch, text_token_lengths, cfg_strength, cfg_rescale, solver
        raise NotImplementedError(
            "LatentFMFMMaskedPrior requires visible prompt audio; use prefix/twostage inference"
        )


def _use_fm_adaln(idx: int, depth: int, every: int) -> bool:
    return idx == 0 or idx == depth - 1 or idx % every == 0
