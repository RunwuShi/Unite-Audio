from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .decoder import DeterministicWaveDecoder, FMHead, FixedRMSNorm, TokenUpsample
from .network import TransformerBlock
from .prior import TextConditionedPriorFMEncoder
from .text_conditioner import FlanT5Conditioner, FlanT5ConditionerConfig


_IGNORED_STATE_PREFIXES = (
    "text_conditioner.",
    "target_encoder.",
    "target_encoder_ema.",
    "token_norm.",
    "token_norm_ema.",
    "encoder_downsamples.",
    "encoder_downsamples_ema.",
    "encoder_down_blocks.",
    "encoder_down_blocks_ema.",
    "encoder_latent_norm.",
    "encoder_latent_norm_ema.",
    "encoder_to_latent.",
    "encoder_to_latent_ema.",
    "mel_loss_fn.",
)


class UniteAudio(nn.Module):
    """Inference network for the bundled deterministic-decoder checkpoints."""

    def __init__(self, config: Mapping[str, Any], text_conditioner: nn.Module) -> None:
        super().__init__()
        self.config = SimpleNamespace(**dict(config))
        c = self.config
        self.text_conditioner = text_conditioner
        self.latent_norm = FixedRMSNorm(
            int(c.latent_dim), scale=float(c.latent_norm_scale), eps=float(c.latent_norm_eps)
        )
        self.decoder_latent_norm = nn.LayerNorm(int(c.latent_dim))
        self.latent_to_decoder = nn.Linear(int(c.latent_dim), int(c.token_dim))
        self.decoder_token_norm = nn.LayerNorm(int(c.token_dim))
        self.decoder_up_blocks = nn.ModuleList([
            nn.ModuleList([
                TransformerBlock(int(c.token_dim), heads=int(c.heads), dim_head=int(c.dim_head),
                                 ffn_mult=int(c.ffn_mult), dropout=float(c.dropout),
                                 rope_base=float(c.rope_base))
                for _ in range(int(depth))
            ]) for depth in c.decoder_up_depths
        ])
        self.decoder_upsamples = nn.ModuleList([
            TokenUpsample(int(c.token_dim), 2) for _ in c.decoder_up_depths
        ])
        self.decoder = DeterministicWaveDecoder(
            backbone="dit", patch_size=int(c.patch_size), token_dim=int(c.token_dim),
            hidden_dim=int(c.hidden_dim), depth=int(c.decoder_same_depth), unet_depths=None,
            heads=int(c.heads), dim_head=int(c.dim_head), ffn_mult=int(c.ffn_mult),
            dropout=float(c.dropout), rope_base=float(c.rope_base), head_type="patch",
        )
        self.fm_encoder = TextConditionedPriorFMEncoder(
            token_dim=int(c.latent_dim), text_dim=int(c.text_dim), hidden_dim=int(c.fm_hidden_dim),
            depth=int(c.fm_depth), adaln_every=int(c.fm_adaln_every), heads=int(c.fm_heads),
            dim_head=int(c.fm_dim_head), ffn_mult=int(c.fm_ffn_mult), dropout=float(c.dropout),
            rope_base=float(c.rope_base), injection="mmdit", input_fusion="add",
            mmdit_fused_depth=int(c.fm_mmdit_fused_depth), global_condition_dim=None,
        )
        self.fm_head = FMHead(int(c.fm_hidden_dim), int(c.latent_dim))

    def filter_checkpoint_state(self, state: Mapping[str, Tensor]) -> dict[str, Tensor]:
        return {
            key: value
            for key, value in state.items()
            if not key.startswith(_IGNORED_STATE_PREFIXES)
        }

    def checkpoint_model_state(self, *, to_cpu: bool = True) -> dict[str, Tensor]:
        state = {key: value for key, value in self.state_dict().items() if not key.startswith("text_conditioner.encoder.")}
        return {key: value.detach().cpu() if to_cpu else value for key, value in state.items()}

    def _null_text_sequence(self, batch_size: int, sequence_length: int, *, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        first, first_mask = self.text_conditioner.null_context(batch_size, device=device, dtype=dtype)
        if sequence_length == 1:
            return first, first_mask
        tokens = torch.cat((first, first.new_zeros((batch_size, sequence_length - 1, first.shape[-1]))), dim=1)
        mask = torch.zeros((batch_size, sequence_length), device=device, dtype=torch.bool)
        mask[:, 0] = True
        return tokens, mask

    def _normalize_flow_x_pred(self, value: Tensor) -> Tensor:
        return self.latent_norm(value)

    def _unscale_waveform(self, value: Tensor) -> Tensor:
        return value / float(self.config.waveform_scale)

    def _flow_inference_time_grid(self, steps: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        del dtype
        grid = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=torch.float32)
        mapping = str(self.config.flow_inference_timestep_mapping).replace("-", "_").lower()
        if mapping == "power":
            grid = grid.pow(float(self.config.flow_inference_timestep_power))
        elif mapping not in {"uniform", "linear", "none"}:
            raise ValueError(f"unsupported timestep mapping: {mapping}")
        shift = float(self.config.flow_inference_timestep_shift)
        if shift != 1.0:
            grid = grid / (grid + shift * (1.0 - grid)).clamp_min(1.0e-6)
        grid[0], grid[-1] = 0.0, 1.0
        return grid

    def _velocity_from_x0(self, x0: Tensor, state: Tensor, time_value: float) -> Tensor:
        return (x0 - state) / max(float(self.config.flow_xpred_denom_min), 1.0 - time_value)

    @staticmethod
    def _rescale_guided_velocity(guided: Tensor, cond: Tensor, *, target_start: int, mix: float) -> Tensor:
        cond_std = cond[:, target_start:].float().std(unbiased=False).clamp_min(1.0e-6)
        guided_std = guided[:, target_start:].float().std(unbiased=False).clamp_min(1.0e-6)
        return mix * guided * (cond_std / guided_std).to(dtype=guided.dtype) + (1.0 - mix) * guided

    @staticmethod
    def _solver_step(state: Tensor, velocity_fn: Any, grid: Tensor, index: int, solver: str) -> Tensor:
        time_value, next_time = float(grid[index]), float(grid[index + 1])
        dt = (grid[index + 1] - grid[index]).to(dtype=state.dtype)
        if solver == "euler":
            return state + dt * velocity_fn(state, time_value)
        if solver == "heun":
            k1 = velocity_fn(state, time_value)
            k2 = velocity_fn(state + dt * k1, next_time)
            return state + dt * (k1 + k2) * 0.5
        half_dt, middle = dt * 0.5, (time_value + next_time) * 0.5
        k1 = velocity_fn(state, time_value)
        k2 = velocity_fn(state + half_dt * k1, middle)
        k3 = velocity_fn(state + half_dt * k2, middle)
        k4 = velocity_fn(state + dt * k3, next_time)
        return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    def decode_tokens_raw(self, z: Tensor, original_len: int) -> Tensor:
        patches = (original_len + int(self.config.patch_size) - 1) // int(self.config.patch_size)
        z = self.decoder_token_norm(self.latent_to_decoder(self.decoder_latent_norm(z)))
        for blocks, upsample in zip(self.decoder_up_blocks, self.decoder_upsamples, strict=True):
            for block in blocks:
                z = block(z)
            z = upsample(z)
        return self._unscale_waveform(self.decoder(z[:, :patches], original_len=original_len))

    @torch.no_grad()
    def generate(self, captions: list[str] | str, **kwargs: Any) -> Tensor:
        from .sampler import sample
        return sample(self, captions, **kwargs)


def build_model(config: Mapping[str, Any], *, text_conditioner: nn.Module | None = None) -> UniteAudio:
    raw = dict(config.get("model", config))
    decoder = dict(raw.get("decoder", {}))
    text = dict(raw.get("text", {}))
    if decoder.get("type") != "deterministic" or decoder.get("head") != "patch":
        raise ValueError("this runtime supports bundled deterministic patch decoders only")
    if str(text.get("injection", "")).replace("-", "_") != "mmdit":
        raise ValueError("this runtime supports bundled FLAN-T5 configurations only")
    raw.update({
        "decoder_up_depths": tuple(int(value) for value in decoder["upsample_depths"]),
        "decoder_same_depth": int(decoder["depth"]),
        "decoder_objective": "wave",
        "text_dim": int(text["text_dim"]),
        "fm_mmdit_fused_depth": int(raw["fm_mmdit_fused_depth"]),
    })
    conditioner = text_conditioner or FlanT5Conditioner(FlanT5ConditionerConfig(
        name_or_path=str(text["name_or_path"]), text_dim=int(text["text_dim"]),
        max_length=int(text["max_length"]), cfg_dropout=0.0, freeze=True,
        local_files_only=bool(text.get("local_files_only", True)),
        padding_mode=str(text.get("padding_mode", "zero")),
        tokenizer_padding=str(text.get("tokenizer_padding", "max_length")),
    ))
    return UniteAudio(raw, conditioner)
