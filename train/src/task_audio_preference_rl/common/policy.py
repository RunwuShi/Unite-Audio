from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn

from task_audio.training.checkpointing import (
    load_tta_model_state,
    resolve_inference_state_path,
)
from task_audio_pmf.proven.model import XPredModel, build_model

from ..config import RolloutConfig
from .trajectory import RolloutTrajectory, cps_step, sample_group_windows


@dataclass
class PriorConditioning:
    text: Tensor
    text_mask: Tensor
    valid: Tensor
    prompt: Tensor
    z_condition: Tensor
    global_text: Tensor | None = None

    def to(self, device: torch.device) -> "PriorConditioning":
        return PriorConditioning(
            text=self.text.to(device),
            text_mask=self.text_mask.to(device),
            valid=self.valid.to(device),
            prompt=self.prompt.to(device),
            z_condition=self.z_condition.to(device),
            global_text=(
                None if self.global_text is None else self.global_text.to(device)
            ),
        )

    def index_select(self, indices: Tensor) -> "PriorConditioning":
        return PriorConditioning(
            text=self.text.index_select(0, indices),
            text_mask=self.text_mask.index_select(0, indices),
            valid=self.valid.index_select(0, indices),
            prompt=self.prompt.index_select(0, indices),
            z_condition=self.z_condition.index_select(0, indices),
            global_text=(
                None
                if self.global_text is None
                else self.global_text.index_select(0, indices)
            ),
        )


class PriorPolicy(nn.Module):
    """The trainable x-pred prior, separated so DDP wraps only FM modules."""

    def __init__(self, source: XPredModel, *, clone: bool = False) -> None:
        super().__init__()
        take = copy.deepcopy if clone else (lambda value: value)
        self.fm_encoder = take(source.fm_encoder)
        self.fm_head = take(source.fm_head)
        self.latent_norm = take(source.latent_norm)
        self.prediction = str(source.config.prediction)
        self.flow_xpred_denom_min = float(source.config.flow_xpred_denom_min)
        if clone:
            self.requires_grad_(False)
            self.eval()

    def predict_x0(self, state: Tensor, time: Tensor, cond: PriorConditioning) -> Tensor:
        hidden = self.fm_encoder(
            state,
            cond.z_condition,
            cond.text,
            time,
            valid_audio_mask=cond.valid,
            text_mask=cond.text_mask,
            prompt_audio_mask=cond.prompt,
            global_text_embedding=cond.global_text,
        )
        prediction = self.fm_head(hidden)
        if self.prediction == "v_pred_v_loss":
            raise ValueError("preference Stage 2 requires x-pred source weights")
        return self.latent_norm(prediction)

    def forward(self, state: Tensor, time: Tensor, cond: PriorConditioning) -> Tensor:
        x0 = self.predict_x0(state, time, cond)
        denom = (1.0 - time).clamp_min(self.flow_xpred_denom_min)
        return (x0 - state) / denom[:, None, None]


@dataclass
class PolicyBundle:
    model: XPredModel
    policy: PriorPolicy
    reference: PriorPolicy
    source_state: Path
    resolved_config: dict


def load_policy_bundle(
    resolved_config_path: str | Path,
    checkpoint: str | Path,
    *,
    device: torch.device,
) -> PolicyBundle:
    config_path = Path(resolved_config_path).expanduser().resolve()
    resolved = json.loads(config_path.read_text(encoding="utf-8"))
    model_config = dict(resolved["model"])
    model_config["mask"] = dict(resolved["mask"])
    model = build_model(model_config)
    if not isinstance(model, XPredModel):
        raise TypeError("preference Stage 2 source must build XPredModel")
    state_path, _ = resolve_inference_state_path(checkpoint, prefer_ema=True)
    load_tta_model_state(model, state_path)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    for module in (model.fm_encoder, model.fm_head):
        module.requires_grad_(True)
    policy = PriorPolicy(model, clone=False)
    reference = PriorPolicy(model, clone=True).to(device)
    return PolicyBundle(
        model=model,
        policy=policy,
        reference=reference,
        source_state=state_path,
        resolved_config=resolved,
    )


class XPredRolloutAdapter:
    def __init__(self, model: XPredModel, policy: nn.Module) -> None:
        self.model = model
        self.policy = policy

    @property
    def device(self) -> torch.device:
        return next(self.model.fm_encoder.parameters()).device

    def prepare_conditioning(
        self,
        captions: Sequence[str],
        *,
        seconds: float,
    ) -> tuple[PriorConditioning, int]:
        captions = [str(value) for value in captions]
        if not captions:
            raise ValueError("captions cannot be empty")
        parameter = next(self.model.fm_encoder.parameters())
        device, dtype = parameter.device, parameter.dtype
        sample_count = max(
            1, int(round(float(seconds) * int(self.model.config.sample_rate)))
        )
        required_patches = math.ceil(sample_count / int(self.model.config.patch_size))
        token_count = math.ceil(
            required_patches / int(self.model.config.downsample_factor)
        )
        batch = len(captions)
        valid = torch.ones((batch, token_count), device=device, dtype=torch.bool)
        prompt = torch.zeros_like(valid)
        z_condition = torch.zeros(
            (batch, token_count, int(self.model.config.latent_dim)),
            device=device,
            dtype=dtype,
        )
        no_dropout = torch.zeros(batch, device=device, dtype=torch.bool)
        text, text_mask = self.model.text_conditioner(
            captions, device=device, dropout_mask=no_dropout
        )
        global_text = self.model._encode_clap_captions(
            captions, device=device, dtype=dtype
        )
        return (
            PriorConditioning(
                text=text.to(device=device),
                text_mask=text_mask.to(device=device, dtype=torch.bool),
                valid=valid,
                prompt=prompt,
                z_condition=z_condition,
                global_text=global_text,
            ),
            sample_count,
        )

    def rollout(
        self,
        unique_captions: Sequence[str],
        config: RolloutConfig,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[RolloutTrajectory, Tensor, PriorConditioning]:
        if config.cfg_strength != 0.0:
            raise NotImplementedError(
                "v1 policy rollout uses cfg_strength=0 for fast on-policy log-probs"
            )
        captions = [
            caption
            for caption in unique_captions
            for _ in range(config.group_size)
        ]
        cond, sample_count = self.prepare_conditioning(
            captions, seconds=config.seconds
        )
        parameter = next(self.model.fm_encoder.parameters())
        state = torch.randn(
            (
                len(captions),
                cond.valid.shape[1],
                int(self.model.config.latent_dim),
            ),
            device=parameter.device,
            dtype=parameter.dtype,
            generator=generator,
        )
        grid = self.model._flow_inference_time_grid(
            config.steps, device=state.device, dtype=state.dtype
        )
        starts, stochastic_mask = sample_group_windows(
            len(unique_captions),
            config.group_size,
            config.steps,
            window_size=config.window_size,
            start_min=config.window_start_min,
            start_max=config.window_start_max,
            device=state.device,
            generator=generator,
        )
        states: list[Tensor] = []
        next_states: list[Tensor] = []
        old_log_probs: list[Tensor] = []
        times: list[Tensor] = []
        next_times: list[Tensor] = []
        for step in range(config.steps):
            current_t = torch.full(
                (state.shape[0],),
                float(grid[step].item()),
                device=state.device,
                dtype=state.dtype,
            )
            following_t = torch.full_like(current_t, float(grid[step + 1].item()))
            velocity = self.policy(state, current_t, cond)
            dt = (following_t - current_t)[:, None, None]
            ode_next = state + dt * velocity
            if config.stochastic_type == "cps":
                sampled = cps_step(
                    state,
                    velocity,
                    current_t,
                    following_t,
                    noise_level=config.noise_level,
                    generator=generator,
                )
                row_mask = stochastic_mask[:, step]
                chosen_next = torch.where(
                    row_mask[:, None, None], sampled.next_state, ode_next
                )
                log_prob = torch.where(
                    row_mask, sampled.log_prob, torch.zeros_like(sampled.log_prob)
                )
            else:
                chosen_next = ode_next
                log_prob = torch.zeros(
                    state.shape[0], device=state.device, dtype=torch.float32
                )
            states.append(state)
            next_states.append(chosen_next)
            old_log_probs.append(log_prob)
            times.append(current_t)
            next_times.append(following_t)
            state = chosen_next
        final_latents = state
        waveform = self.model.decode_tokens_raw(
            final_latents, original_len=sample_count
        )
        trajectory = RolloutTrajectory(
            states=torch.stack(states, dim=1),
            next_states=torch.stack(next_states, dim=1),
            times=torch.stack(times, dim=1),
            next_times=torch.stack(next_times, dim=1),
            old_log_probs=torch.stack(old_log_probs, dim=1),
            stochastic_mask=stochastic_mask,
            window_starts=starts,
            final_latents=final_latents,
        ).detach()
        return trajectory, waveform.detach(), cond

    @staticmethod
    def transition_log_prob(
        policy: nn.Module,
        trajectory: RolloutTrajectory,
        conditioning: PriorConditioning,
        step: int,
        *,
        noise_level: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        state = trajectory.states[:, step]
        next_state = trajectory.next_states[:, step]
        current = trajectory.times[:, step]
        following = trajectory.next_times[:, step]
        velocity = policy(state, current, conditioning)
        result = cps_step(
            state,
            velocity,
            current,
            following,
            noise_level=noise_level,
            next_state=next_state,
        )
        return result.log_prob, result.mean, result.std


__all__ = [
    "PolicyBundle",
    "PriorConditioning",
    "PriorPolicy",
    "XPredRolloutAdapter",
    "load_policy_bundle",
]
