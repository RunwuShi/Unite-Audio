from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class CPSStep:
    next_state: Tensor
    log_prob: Tensor
    mean: Tensor
    std: Tensor


@dataclass
class RolloutTrajectory:
    states: Tensor
    next_states: Tensor
    times: Tensor
    next_times: Tensor
    old_log_probs: Tensor
    stochastic_mask: Tensor
    window_starts: Tensor
    final_latents: Tensor

    def detach(self) -> "RolloutTrajectory":
        return RolloutTrajectory(
            **{name: getattr(self, name).detach() for name in self.__dataclass_fields__}
        )


def gaussian_log_prob(value: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """Mean Gaussian log probability over all non-batch dimensions."""

    value = value.float()
    mean = mean.float()
    std = std.float().clamp_min(1.0e-8)
    log_prob = (
        -0.5 * ((value.detach() - mean) / std).square()
        - torch.log(std)
        - 0.5 * math.log(2.0 * math.pi)
    )
    return log_prob.mean(dim=tuple(range(1, log_prob.ndim)))


def cps_step(
    state: Tensor,
    velocity: Tensor,
    current_t: Tensor | float,
    next_t: Tensor | float,
    *,
    noise_level: float,
    next_state: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> CPSStep:
    """Coefficient-preserving stochastic step for x-pred's t=0 noise -> t=1 data.

    The equivalent FlowMatch scheduler noise coordinate is sigma=1-t.  At
    noise_level=0 this exactly reduces to the Euler/straight-flow update.
    """

    if state.shape != velocity.shape:
        raise ValueError("state and velocity must have the same shape")
    device = state.device
    shape = (state.shape[0],) + (1,) * (state.ndim - 1)
    current = torch.as_tensor(current_t, device=device, dtype=torch.float32)
    following = torch.as_tensor(next_t, device=device, dtype=torch.float32)
    if current.ndim == 0:
        current = current.expand(state.shape[0])
    if following.ndim == 0:
        following = following.expand(state.shape[0])
    if current.shape != (state.shape[0],) or following.shape != (state.shape[0],):
        raise ValueError("time inputs must be scalar or shape [B]")
    if bool((following <= current).any().item()):
        raise ValueError("x-pred CPS requires increasing t")
    sigma = (1.0 - current).view(shape)
    sigma_next = (1.0 - following).view(shape)
    state32, velocity32 = state.float(), velocity.float()
    predicted_clean = state32 + sigma * velocity32
    predicted_noise = state32 - current.view(shape) * velocity32
    std = sigma_next * math.sin(float(noise_level) * math.pi / 2.0)
    noise_coefficient = torch.sqrt((sigma_next.square() - std.square()).clamp_min(0.0))
    mean = following.view(shape) * predicted_clean + noise_coefficient * predicted_noise
    if next_state is None:
        noise = torch.randn(
            state.shape,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        sampled = mean + std * noise
    else:
        sampled = next_state.float()
    if float(noise_level) == 0.0:
        # A degenerate transition has no policy density.  Keep a finite zero
        # for parity tests; GRPO rejects zero-noise stochastic windows.
        log_prob = torch.zeros(state.shape[0], device=device, dtype=torch.float32)
    else:
        log_prob = gaussian_log_prob(sampled, mean, std)
    return CPSStep(
        next_state=sampled.to(dtype=state.dtype),
        log_prob=log_prob,
        mean=mean,
        std=std,
    )


def sample_group_windows(
    prompt_count: int,
    group_size: int,
    steps: int,
    *,
    window_size: int,
    start_min: int,
    start_max: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    if start_max + window_size > steps:
        raise ValueError("window exceeds trajectory")
    starts = torch.randint(
        start_min,
        start_max + 1,
        (prompt_count,),
        device=device,
        generator=generator,
    )
    sample_starts = starts.repeat_interleave(group_size)
    positions = torch.arange(steps, device=device)[None, :]
    mask = (positions >= sample_starts[:, None]) & (
        positions < sample_starts[:, None] + window_size
    )
    return sample_starts, mask


def _group_z(values: Tensor, group_size: int, eps: float) -> Tensor:
    if values.ndim != 1 or values.numel() % group_size:
        raise ValueError("reward vector must contain complete prompt groups")
    grouped = values.float().view(-1, group_size)
    return (
        (grouped - grouped.mean(dim=1, keepdim=True))
        / grouped.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    ).reshape_as(values)


def balanced_group_advantage(
    clap: Tensor,
    passt_kl: Tensor,
    repetition: Tensor,
    *,
    group_size: int,
    clap_weight: float,
    passt_weight: float,
    repetition_weight: float,
    fd_marginal: Tensor | None = None,
    fd_weight: float = 0.0,
    fad_marginal: Tensor | None = None,
    fad_weight: float = 0.0,
    eps: float = 1.0e-5,
) -> Tensor:
    composite = (
        float(clap_weight) * _group_z(clap, group_size, eps)
        + float(passt_weight) * _group_z(-passt_kl, group_size, eps)
        + float(repetition_weight) * _group_z(-repetition, group_size, eps)
    )
    if fd_marginal is not None and float(fd_weight) > 0:
        composite = composite + float(fd_weight) * _group_z(
            fd_marginal, group_size, eps
        )
    if fad_marginal is not None and float(fad_weight) > 0:
        composite = composite + float(fad_weight) * _group_z(
            fad_marginal, group_size, eps
        )
    return _group_z(composite, group_size, eps)
