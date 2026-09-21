from __future__ import annotations

from copy import deepcopy
from typing import Any

from torch import nn

from task_audio._backbone.latent_tts import (
    DeterministicWaveDecoder,
    FixedRMSNorm,
    TargetEncoder,
    TokenDownsample,
    TokenUpsample,
    WavePatchify,
)
from task_audio._backbone.latent_tts_fm_decoder import WaveFMDecoderHead
from task_audio._backbone.losses import MelSpectrogramLoss
from task_audio._backbone.transformer import TransformerBlock
from task_audio._backbone.wave_fm_udit import LinearUDiTUNetWaveFMDecoder


def initialize_tts_audio_components(model: nn.Module, config: Any) -> None:
    """Attach the reusable TTS audio stack without constructing TTS text code.

    Attribute names intentionally match the existing TTS model so audio-only
    checkpoint initialization can use exact state-dict keys.
    """

    model.patchify = WavePatchify(config.patch_size)
    model.target_encoder = TargetEncoder(
        patch_size=config.patch_size,
        token_dim=config.token_dim,
        depth=config.encoder_same_depth,
        heads=config.heads,
        dim_head=config.dim_head,
        ffn_mult=config.ffn_mult,
        dropout=config.dropout,
        rope_base=config.rope_base,
    )
    model.token_norm = nn.LayerNorm(config.token_dim)
    model.encoder_downsamples = nn.ModuleList(
        [TokenDownsample(config.token_dim, 2) for _ in config.encoder_down_depths]
    )
    model.encoder_down_blocks = nn.ModuleList(
        [
            nn.ModuleList(
                [
                    TransformerBlock(
                        config.token_dim,
                        heads=config.heads,
                        dim_head=config.dim_head,
                        ffn_mult=config.ffn_mult,
                        dropout=config.dropout,
                        rope_base=config.rope_base,
                    )
                    for _ in range(depth)
                ]
            )
            for depth in config.encoder_down_depths
        ]
    )
    model.encoder_latent_norm = nn.LayerNorm(config.token_dim)
    model.encoder_to_latent = nn.Linear(config.token_dim, config.latent_dim)
    model.latent_norm = _build_latent_norm(config)

    if config.build_deterministic_decoder:
        model.decoder_up_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        TransformerBlock(
                            config.token_dim,
                            heads=config.heads,
                            dim_head=config.dim_head,
                            ffn_mult=config.ffn_mult,
                            dropout=config.dropout,
                            rope_base=config.rope_base,
                        )
                        for _ in range(depth)
                    ]
                )
                for depth in config.decoder_up_depths
            ]
        )
        model.decoder_upsamples = nn.ModuleList(
            [TokenUpsample(config.token_dim, 2) for _ in config.decoder_up_depths]
        )
        model.decoder_latent_norm = nn.LayerNorm(config.latent_dim)
        model.latent_to_decoder = nn.Linear(config.latent_dim, config.token_dim)
        model.decoder_token_norm = nn.LayerNorm(config.token_dim)
        model.decoder = DeterministicWaveDecoder(
            backbone=config.decoder_backbone,
            patch_size=config.patch_size,
            token_dim=config.token_dim,
            hidden_dim=config.hidden_dim,
            depth=config.decoder_same_depth,
            unet_depths=config.decoder_unet_depths,
            heads=config.heads,
            dim_head=config.dim_head,
            ffn_mult=config.ffn_mult,
            dropout=config.dropout,
            rope_base=config.rope_base,
            input_fusion=config.decoder_input_fusion,
            head_type=config.decoder_head,
            istft_n_fft_factor=config.decoder_istft_n_fft_factor,
            istft_mag_clip=config.decoder_istft_mag_clip,
        )
    else:
        model.decoder_up_blocks = nn.ModuleList()
        model.decoder_upsamples = nn.ModuleList()
        model.decoder_latent_norm = nn.Identity()
        model.latent_to_decoder = nn.Identity()
        model.decoder_token_norm = nn.Identity()
        model.decoder = nn.Identity()

    model.mel_loss_fn = MelSpectrogramLoss(
        sample_rate=config.sample_rate,
        max_chunk_samples=config.mel_max_chunk_samples,
        max_stft_samples=config.mel_max_stft_samples,
        stft_device=config.mel_stft_device,
    )
    model.wave_speaker_encoder = None
    model.wave_fm_decoder = _build_wave_fm_decoder(config)

    model.target_encoder_ema = None
    model.token_norm_ema = None
    model.encoder_downsamples_ema = None
    model.encoder_down_blocks_ema = None
    model.encoder_latent_norm_ema = None
    model.encoder_to_latent_ema = None
    model.latent_norm_ema = None
    if config.split_target_encoder_ema:
        model.target_encoder_ema = deepcopy(model.target_encoder)
        model.token_norm_ema = deepcopy(model.token_norm)
        model.encoder_downsamples_ema = deepcopy(model.encoder_downsamples)
        model.encoder_down_blocks_ema = deepcopy(model.encoder_down_blocks)
        model.encoder_latent_norm_ema = deepcopy(model.encoder_latent_norm)
        model.encoder_to_latent_ema = deepcopy(model.encoder_to_latent)
        model.latent_norm_ema = deepcopy(model.latent_norm)
        for module in model._target_encoder_ema_modules():
            module.requires_grad_(False)
            module.eval()


def _build_latent_norm(config: Any) -> nn.Module:
    if config.latent_norm_mode == "layernorm":
        return nn.LayerNorm(config.latent_dim)
    if config.latent_norm_mode == "fixed_rms":
        return FixedRMSNorm(
            config.latent_dim,
            scale=config.latent_norm_scale,
            eps=config.latent_norm_eps,
        )
    if config.latent_norm_mode == "identity":
        return nn.Identity()
    raise ValueError(f"unknown latent_norm_mode: {config.latent_norm_mode}")


def _build_wave_fm_decoder(config: Any) -> nn.Module | None:
    if config.decoder_objective != "wave_fm":
        return None
    if config.wave_fm_backbone == "linear_udit_unet":
        if config.decoder_head != "patch":
            raise ValueError("linear_udit_unet wave decoder requires decoder.head='patch'")
        return LinearUDiTUNetWaveFMDecoder(
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
    return WaveFMDecoderHead(
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


__all__ = ["initialize_tts_audio_components"]
