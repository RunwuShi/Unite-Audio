from pathlib import Path

from task_audio_pmf.proven.configs import (
    build_xpred_stage1_config as _build_xpred_stage1_config,
)


SOURCE_RUN = Path(
    "/nativemm2/share/cpfs/shirunwu.sru/537342/workflow_60970279/"
    "workspace/latentaudio/experiments/task_audio_xpred_prior512d18_stage1/"
    "20260730_130100_from110k_replay_original_spectral"
)


def build_config():
    config = _build_xpred_stage1_config()

    # Preserve the exact 110k-200k model and loss recipe.
    config["training"]["discriminator"]["update_every"] = 2
    config["model"]["loss"]["adversarial"]["weight_transition"] = {
        "enabled": True,
        "source_fingerprint": (
            "ef1b509cb64395b5ab543e838b2984600095ed8e484176354ee85382b5091cbe"
        ),
        "source_step": 42_000,
        "source_generator_weight": 0.1,
        "source_feature_matching_weight": 5.0,
        "transition_steps": 0,
        "reason": (
            "Continue from 42k with discriminator update_every changed from 1 to 2"
        ),
    }
    latent_noise = config["model"]["decoder"]["latent_noise"]
    latent_noise["mode"] = "unite"
    latent_noise["probability"] = 0.5
    latent_noise["t_start"] = 0.7

    # Keep the successful 110k-200k source update distribution.
    config["data"]["source_update_weights"] = {
        "audiosetcaps": 0.5,
        "wavcaps": 0.5,
    }
    config["experiment"]["name"] = (
        "task_audio_xpred_prior512d18_stage1_5050_accum2"
    )
    config["training"].update(
        {
            "max_steps": 240_000,
            "resume_from": str(
                SOURCE_RUN / "checkpoint" / "checkpoint-00200000"
            ),
            # Two ranks x accumulation 2 preserves the old four-rank
            # effective global batch while retaining the same per-rank batch.
            "grad_accumulation_steps": 2,
            "auto_continue_data_cursor": True,
            "reset_data_cursor": False,
        }
    )

    # Step 200k is already fully evaluated in SOURCE_RUN.
    config["audiocaps_eval"]["skip_steps"] = [200_000]
    milestone_steps = list(config["audiocaps_eval"]["milestone_cfg_steps"])
    if 240_000 not in milestone_steps:
        milestone_steps.append(240_000)
    config["audiocaps_eval"]["milestone_cfg_steps"] = milestone_steps
    return config


CONFIG = build_config()
