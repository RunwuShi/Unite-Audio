from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "expeirment_0804"
    / "20260808_124801_from200k_noisy_online_masked_loss_p50_strict_stream_240k"
    / "checkpoint"
    / "checkpoint-00240000"
)
DEFAULT_RESOLVED_CONFIG = (
    ROOT
    / "expeirment_0804"
    / "20260808_124801_from200k_noisy_online_masked_loss_p50_strict_stream_240k"
    / "config"
    / "resolved_config.json"
)
DEFAULT_AUDIOCAPS_MANIFEST = (
    ROOT.parent / "dataset" / "AudioCaps" / "train_metadata.csv"
)
DEFAULT_AUDIOCAPS_ROOT = ROOT.parent / "dataset" / "AudioCaps"
DEFAULT_OUTPUT_ROOT = (
    ROOT / "experiments" / "task_audio_xpred_stage2_audiocaps_preference_rl"
)
DEFAULT_CLAP_CHECKPOINT = (
    ROOT
    / "refer"
    / "stable-audio-metrics"
    / "load"
    / "clap_score"
    / "630k-audioset-fusion-best.pt"
)
DEFAULT_VGGISH_HUB_DIR = ROOT / "evaluation" / "cache" / "torch" / "hub"
DEFAULT_PANNS_REPO = ROOT / "refer" / "audioldm_eval"
DEFAULT_PANNS_CHECKPOINT_DIR = ROOT / "evaluation" / "cache" / "audioldm_eval" / "ckpt"


@dataclass(frozen=True)
class PathsConfig:
    source_checkpoint: str = str(DEFAULT_SOURCE)
    source_resolved_config: str = str(DEFAULT_RESOLVED_CONFIG)
    audiocaps_manifest: str = str(DEFAULT_AUDIOCAPS_MANIFEST)
    audiocaps_root: str = str(DEFAULT_AUDIOCAPS_ROOT)
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    passt_reference_cache: str = str(
        DEFAULT_OUTPUT_ROOT / "cache" / "audiocaps_train_passt.pt"
    )
    clap_checkpoint: str = str(DEFAULT_CLAP_CHECKPOINT)
    vggish_hub_dir: str = str(DEFAULT_VGGISH_HUB_DIR)
    vggish_reference_cache: str = str(
        DEFAULT_OUTPUT_ROOT / "cache" / "audiocaps_train_vggish_fad.pt"
    )
    vggish_fad_cache: str = str(
        DEFAULT_OUTPUT_ROOT / "cache" / "audiocaps_train_vggish_fad.pt"
    )
    panns_repo: str = str(DEFAULT_PANNS_REPO)
    panns_checkpoint_dir: str = str(DEFAULT_PANNS_CHECKPOINT_DIR)
    panns_fd_reference_cache: str = str(
        DEFAULT_OUTPUT_ROOT / "cache" / "audiocaps_train_panns_fd_reference_256d.pt"
    )
    panns_fd_cache: str = str(
        DEFAULT_OUTPUT_ROOT / "cache" / "audiocaps_train_panns_fd.pt"
    )


@dataclass(frozen=True)
class RolloutConfig:
    steps: int = 16
    seconds: float = 10.0
    solver: str = "euler"
    cfg_strength: float = 0.0
    group_size: int = 4
    prompts_per_rank: int = 4
    stochastic_type: str = "cps"
    noise_level: float = 0.4
    window_size: int = 4
    window_start_min: int = 1
    window_start_max: int = 4

    def __post_init__(self) -> None:
        if self.steps < 1 or self.group_size < 2 or self.prompts_per_rank < 1:
            raise ValueError(
                "steps/prompts must be positive and group_size must be >= 2"
            )
        if self.solver != "euler":
            raise ValueError("preference rollout currently requires solver='euler'")
        if self.stochastic_type not in {"cps", "ode"}:
            raise ValueError("stochastic_type must be 'cps' or 'ode'")
        if not 0.0 <= self.noise_level < 1.0:
            raise ValueError("noise_level must be in [0,1)")
        if self.window_size < 1:
            raise ValueError("window_size must be positive")
        if not 0 <= self.window_start_min <= self.window_start_max:
            raise ValueError("invalid stochastic window range")
        if self.window_start_max + self.window_size > self.steps:
            raise ValueError("stochastic window exceeds rollout steps")

    @property
    def samples_per_rank(self) -> int:
        return self.prompts_per_rank * self.group_size


@dataclass(frozen=True)
class RewardConfig:
    clap_weight: float = 0.60
    passt_weight: float = 0.30
    repetition_weight: float = 0.10
    clap_backend: str = "630k"
    clap_batch_size: int = 32
    passt_batch_size: int = 32
    repetition_min_lag_seconds: float = 0.25
    repetition_max_lag_seconds: float = 2.0
    fd_enabled: bool = False
    fd_projection_dim: int = 256
    fd_queue_capacity: int = 2048
    fd_target_scale: float = 1.0
    fd_dual_lr: float = 0.002
    fd_lambda_init: float = 0.0
    fd_lambda_max: float = 0.35
    fd_exact_every: int = 10
    fd_embedding_batch_size: int = 16
    fad_enabled: bool = False
    fad_queue_capacity: int = 2048
    fad_dual_lr: float = 0.05
    fad_lambda_init: float = 0.0
    fad_lambda_max: float = 2.0
    fad_target_scale: float = 1.0
    fad_interval: int = 1
    fad_exact_every: int = 10
    fad_embedding_batch_size: int = 256
    eps: float = 1.0e-5

    def __post_init__(self) -> None:
        weights = (self.clap_weight, self.passt_weight, self.repetition_weight)
        if any(value < 0 or not math.isfinite(value) for value in weights):
            raise ValueError("reward weights must be finite and non-negative")
        if sum(weights) <= 0:
            raise ValueError("at least one reward weight must be positive")
        if self.clap_backend not in {"630k", "htsat_fused", "disabled"}:
            raise ValueError("unsupported CLAP reward backend")
        if not 1 <= self.fd_projection_dim <= 2048:
            raise ValueError("fd_projection_dim must be in [1, 2048]")
        if self.fd_queue_capacity < 2:
            raise ValueError("fd_queue_capacity must be at least 2")
        if not 0 < self.fd_target_scale <= 1:
            raise ValueError("fd_target_scale must be in (0, 1]")
        if self.fd_dual_lr < 0 or not math.isfinite(self.fd_dual_lr):
            raise ValueError("fd_dual_lr must be finite and non-negative")
        if not 0 <= self.fd_lambda_init <= self.fd_lambda_max:
            raise ValueError("fd_lambda_init must be in [0, fd_lambda_max]")
        if self.fd_exact_every < 1 or self.fd_embedding_batch_size < 1:
            raise ValueError("FD intervals and batch sizes must be positive")
        if self.fad_queue_capacity < 2:
            raise ValueError("fad_queue_capacity must be at least 2")
        if not 0 < self.fad_target_scale <= 1:
            raise ValueError("fad_target_scale must be in (0, 1]")
        if self.fad_dual_lr < 0 or not math.isfinite(self.fad_dual_lr):
            raise ValueError("fad_dual_lr must be finite and non-negative")
        if not 0 <= self.fad_lambda_init <= self.fad_lambda_max:
            raise ValueError("fad_lambda_init must be in [0, fad_lambda_max]")
        if (
            self.fad_interval < 1
            or self.fad_exact_every < 1
            or self.fad_embedding_batch_size < 1
        ):
            raise ValueError("FAD intervals and batch sizes must be positive")


@dataclass(frozen=True)
class GRPOConfig:
    prompt_epochs: int = 3
    learning_rate: float = 1.0e-6
    weight_decay: float = 1.0e-4
    ppo_clip_range: float = 1.0e-4
    advantage_clip: float = 5.0
    reference_kl_coefficient: float = 0.01
    grad_clip_norm: float = 1.0
    mixed_precision: str = "fp16"
    save_every_updates: int = 500
    log_every_updates: int = 10
    eta_warmup_updates: int = 100


@dataclass(frozen=True)
class DPOConfig:
    prompt_epochs: int = 3
    collection_prompt_epochs: int = 1
    learning_rate: float = 1.0e-6
    weight_decay: float = 1.0e-4
    beta: float = 2000.0
    chosen_anchor_weight: float = 1.0
    minimum_preference_gap: float = 0.5
    pairs_per_rank_batch: int = 8
    mixed_precision: str = "bf16"
    save_every_updates: int = 500
    log_every_updates: int = 10


@dataclass(frozen=True)
class ExperimentConfig:
    algorithm: str = "flow_grpo"
    seed: int = 1234
    paths: PathsConfig = field(default_factory=PathsConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)

    def __post_init__(self) -> None:
        if self.algorithm not in {"flow_grpo", "flow_dpo"}:
            raise ValueError("algorithm must be flow_grpo or flow_dpo")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _section(cls: type[Any], values: Mapping[str, Any] | None) -> Any:
    return cls(**dict(values or {}))


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path).expanduser().resolve()
    values = json.loads(source.read_text(encoding="utf-8"))
    return ExperimentConfig(
        algorithm=str(values.get("algorithm", "flow_grpo")),
        seed=int(values.get("seed", 1234)),
        paths=_section(PathsConfig, values.get("paths")),
        rollout=_section(RolloutConfig, values.get("rollout")),
        reward=_section(RewardConfig, values.get("reward")),
        grpo=_section(GRPOConfig, values.get("grpo")),
        dpo=_section(DPOConfig, values.get("dpo")),
    )


__all__ = [
    "DPOConfig",
    "ExperimentConfig",
    "GRPOConfig",
    "PathsConfig",
    "RewardConfig",
    "RolloutConfig",
    "load_config",
]
