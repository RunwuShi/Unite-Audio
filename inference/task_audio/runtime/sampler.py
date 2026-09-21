from __future__ import annotations

from typing import Any

import math
import torch
from torch import Tensor

def sample(model: Any, captions: list[str] | str, *, seconds: float = 10.0, num_steps: int = 16, prior_steps: int | None = None, wave_steps: int | None = None, solver: str = "euler", cfg_strength: float = 0.0, cfg_rescale: float = 0.0, generator: torch.Generator | None = None) -> Tensor:
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

    param = next(model.fm_encoder.parameters())
    device, dtype = param.device, param.dtype
    sample_count = max(1, int(round(float(seconds) * int(model.config.sample_rate))))
    required_patches = math.ceil(sample_count / int(model.config.patch_size))
    total_tokens = math.ceil(required_patches / int(model.config.downsample_factor))
    total_patches = total_tokens * int(model.config.downsample_factor)
    batch_size = len(captions)
    valid_latent_mask = torch.ones(
        (batch_size, total_tokens), device=device, dtype=torch.bool
    )

    no_dropout = torch.zeros(batch_size, device=device, dtype=torch.bool)
    text_tokens, text_mask = model.text_conditioner(
        captions,
        device=device,
        dropout_mask=no_dropout,
    )
    text_tokens = text_tokens.to(device=device)
    text_mask = text_mask.to(device=device, dtype=torch.bool)
    z_state = torch.randn(
        (batch_size, total_tokens, int(model.config.latent_dim)),
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
    ) -> Tensor:
        t = torch.full((state.shape[0],), time_value, device=device, dtype=dtype)
        valid = valid_latent_mask
        cond = z_cond
        prompt = prompt_mask
        if state.shape[0] != batch_size:
            valid = torch.cat((valid, valid), dim=0)
            cond = torch.cat((cond, cond), dim=0)
            prompt = torch.cat((prompt, prompt), dim=0)
        h = model.fm_encoder(
            state,
            cond,
            context,
            t,
            valid_audio_mask=valid,
            text_mask=context_mask,
            prompt_audio_mask=prompt,
            global_text_embedding=None,
        )
        prediction = model.fm_head(h)
        if model.config.prediction == "v_pred_v_loss":
            return prediction
        return model._normalize_flow_x_pred(prediction)

    if float(cfg_strength) > 0.0:
        null_tokens, null_mask = model._null_text_sequence(
            batch_size,
            text_tokens.shape[1],
            device=device,
            dtype=text_tokens.dtype,
        )
        cfg_text = torch.cat((text_tokens, null_tokens), dim=0)
        cfg_text_mask = torch.cat((text_mask, null_mask), dim=0)
        cfg_global_text = None
    else:
        cfg_text = text_tokens
        cfg_text_mask = text_mask
        cfg_global_text = None

    def prior_velocity(state: Tensor, time_value: float) -> Tensor:
        if float(cfg_strength) <= 0.0:
            prediction = prior_prediction(
                state,
                time_value,
                cfg_text,
                cfg_text_mask,
            )
            if model.config.prediction == "v_pred_v_loss":
                return prediction
            return model._velocity_from_x0(prediction, state, time_value)
        state_cat = torch.cat((state, state), dim=0)
        pred_cond, pred_uncond = prior_prediction(
            state_cat,
            time_value,
            cfg_text,
            cfg_text_mask,
        ).chunk(2, dim=0)
        if model.config.prediction == "v_pred_v_loss":
            v_cond, v_uncond = pred_cond, pred_uncond
        else:
            v_cond = model._velocity_from_x0(pred_cond, state, time_value)
            v_uncond = model._velocity_from_x0(pred_uncond, state, time_value)
        guided = v_cond + float(cfg_strength) * (v_cond - v_uncond)
        if float(cfg_rescale) > 0.0:
            guided = model._rescale_guided_velocity(
                guided,
                v_cond,
                target_start=0,
                mix=max(0.0, min(float(cfg_rescale), 1.0)),
            )
        return guided

    prior_grid = model._flow_inference_time_grid(
        prior_step_count, device=device, dtype=dtype
    )
    for idx in range(prior_step_count):
        z_state = model._solver_step(
            z_state, prior_velocity, prior_grid, idx, solver
        )

    if model.config.decoder_objective == "wave":
        return model.decode_tokens_raw(z_state, original_len=sample_count)
