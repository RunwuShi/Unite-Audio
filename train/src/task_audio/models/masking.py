from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor

# Training scripts add the LatentAudio repository itself to PYTHONPATH.
from task_audio._backbone.latent_tts import sample_span_mask as _tts_sample_span_mask


MaskMetric = Tensor | float

MASK_MODE_SSL = 0
MASK_MODE_FULL_GENERATION = 1


@dataclass
class MaskPlan:
    """A sampled prior target mask and its logging metadata."""

    target_mask: Tensor
    policy_name: str
    metrics: dict[str, MaskMetric]
    encoding_mode: str = "contiguous_span"
    full_generation_mask: Tensor | None = None
    mode_ids: Tensor | None = None

    def __post_init__(self) -> None:
        if self.encoding_mode not in {"contiguous_span", "arbitrary_mask"}:
            raise ValueError(
                "encoding_mode must be 'contiguous_span' or 'arbitrary_mask'"
            )
        batch_size = int(self.target_mask.shape[0]) if self.target_mask.ndim >= 1 else 0
        if (
            self.full_generation_mask is not None
            and self.full_generation_mask.shape != (batch_size,)
        ):
            raise ValueError("full_generation_mask must have shape [B]")
        if self.mode_ids is not None and self.mode_ids.shape != (batch_size,):
            raise ValueError("mode_ids must have shape [B]")

    @property
    def mask(self) -> Tensor:
        """Compatibility alias for earlier TTA mask-policy experiments."""

        return self.target_mask

    @property
    def actual_ratio(self) -> Tensor:
        return self.metrics["actual_ratio"]  # type: ignore[return-value]


@runtime_checkable
class PriorMaskPolicy(Protocol):
    """Extension point for future TTA prior-mask strategies."""

    def sample(
        self,
        valid_latent_mask: Tensor,
        batch: Mapping[str, Any] | None = None,
        progress: float | None = None,
        generator: torch.Generator | None = None,
    ) -> MaskPlan: ...


@dataclass(frozen=True)
class RandomSpanMaskConfig:
    policy: str = "random_span"
    min_ratio: float = 0.70
    max_ratio: float = 1.00

    def __post_init__(self) -> None:
        if self.policy not in {"random_span", "tts_random_span"}:
            raise ValueError("RandomSpanMaskConfig requires policy='random_span'")
        if not 0.0 < float(self.min_ratio) <= float(self.max_ratio) <= 1.0:
            raise ValueError("mask ratios must satisfy 0 < min_ratio <= max_ratio <= 1")


class RandomSpanMaskPolicy:
    """TTS-compatible single contiguous target-span sampling."""

    policy_name = "random_span"

    def __init__(
        self,
        config: RandomSpanMaskConfig | Mapping[str, Any] | None = None,
        *,
        min_ratio: float | None = None,
        max_ratio: float | None = None,
    ) -> None:
        if config is None:
            values: dict[str, Any] = {}
        elif isinstance(config, RandomSpanMaskConfig):
            values = {
                "policy": config.policy,
                "min_ratio": config.min_ratio,
                "max_ratio": config.max_ratio,
            }
        else:
            raw = dict(config)
            values = {
                "policy": raw.get("policy", "random_span"),
                "min_ratio": raw.get(
                    "min_ratio", raw.get("random_span_min_ratio", 0.70)
                ),
                "max_ratio": raw.get(
                    "max_ratio", raw.get("random_span_max_ratio", 1.00)
                ),
            }
        if min_ratio is not None:
            values["min_ratio"] = min_ratio
        if max_ratio is not None:
            values["max_ratio"] = max_ratio
        self.config = RandomSpanMaskConfig(**values)

    def sample(
        self,
        valid_latent_mask: Tensor,
        batch: Mapping[str, Any] | None = None,
        progress: float | None = None,
        generator: torch.Generator | None = None,
    ) -> MaskPlan:
        del batch, progress  # Reserved for policies that use data or schedules.
        valid = _validate_valid_mask(valid_latent_mask)
        batch_size, max_len = valid.shape
        lengths = valid.long().sum(dim=1)
        if generator is None:
            target = _tts_sample_span_mask(
                batch_size,
                lengths,
                min_ratio=float(self.config.min_ratio),
                max_ratio=float(self.config.max_ratio),
                device=valid.device,
                max_len=max_len,
            )
        else:
            # The TTS helper uses the global RNG and has no generator argument.
            # This is the exact same algorithm with an explicit generator.
            target = _sample_span_with_generator(
                valid,
                min_ratio=float(self.config.min_ratio),
                max_ratio=float(self.config.max_ratio),
                generator=generator,
            )
        target = target.to(dtype=torch.bool) & valid
        _validate_target_mask(target, valid)

        masked_tokens = target.long().sum(dim=1)
        valid_tokens = valid.long().sum(dim=1)
        visible_tokens = valid_tokens - masked_tokens
        actual_ratio = masked_tokens.float() / valid_tokens.clamp_min(1).float()
        nonempty = valid_tokens > 0
        mean_ratio = (
            actual_ratio[nonempty].mean()
            if bool(nonempty.any().item())
            else actual_ratio.new_zeros(())
        )
        return MaskPlan(
            target_mask=target,
            policy_name=self.policy_name,
            encoding_mode="contiguous_span",
            full_generation_mask=torch.zeros(
                batch_size, device=valid.device, dtype=torch.bool
            ),
            mode_ids=torch.full(
                (batch_size,),
                MASK_MODE_SSL,
                device=valid.device,
                dtype=torch.long,
            ),
            metrics={
                "requested_ratio_min": float(self.config.min_ratio),
                "requested_ratio_max": float(self.config.max_ratio),
                "actual_ratio": actual_ratio,
                "actual_ratio_mean": mean_ratio,
                "masked_tokens": masked_tokens,
                "visible_tokens": visible_tokens,
                "valid_tokens": valid_tokens,
            },
        )


@dataclass(frozen=True)
class BucketSSLMaskConfig:
    """TTA-only bucket SSL mixed with caption-conditioned full generation."""

    policy: str = "bucket_ssl"
    full_generation_probability: float = 0.25
    unit_tokens: int = 3
    bucket_tokens: int = 50
    visible_ratio_min: float = 0.0
    visible_ratio_max: float = 0.30
    random_bucket_offset: bool = False
    ensure_bucket_coverage: bool = False

    def __post_init__(self) -> None:
        if self.policy not in {
            "bucket_ssl",
            "tta_bucket_ssl",
            "bucket_ssl_shifted",
            "bucket_ssl_coverage",
            "bucket_ssl_shifted_coverage",
        }:
            raise ValueError("BucketSSLMaskConfig requires a bucket_ssl policy")
        if not 0.0 <= float(self.full_generation_probability) <= 1.0:
            raise ValueError("full_generation_probability must be in [0, 1]")
        if int(self.unit_tokens) < 1:
            raise ValueError("unit_tokens must be positive")
        if int(self.bucket_tokens) < int(self.unit_tokens):
            raise ValueError("bucket_tokens must be at least unit_tokens")
        if (
            not 0.0
            <= float(self.visible_ratio_min)
            <= float(self.visible_ratio_max)
            <= 1.0
        ):
            raise ValueError(
                "visible ratios must satisfy 0 <= visible_ratio_min <= visible_ratio_max <= 1"
            )


class BucketSSLMaskPolicy:
    """Sample TTA bucket SSL masks with an example-level full-generation mix.

    Each valid sequence is divided into fixed-size buckets. Visible tokens are
    selected as indivisible contiguous units, and all remaining valid tokens
    become prior targets. A stratified stochastic count keeps the requested
    full-generation ratio low-variance within each batch (B=4 and p=0.25 gives
    exactly one full-generation row).
    """

    policy_name = "bucket_ssl"

    def __init__(
        self,
        config: BucketSSLMaskConfig | Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> None:
        if config is None:
            values: dict[str, Any] = {}
        elif isinstance(config, BucketSSLMaskConfig):
            values = {
                "policy": config.policy,
                "full_generation_probability": config.full_generation_probability,
                "unit_tokens": config.unit_tokens,
                "bucket_tokens": config.bucket_tokens,
                "visible_ratio_min": config.visible_ratio_min,
                "visible_ratio_max": config.visible_ratio_max,
                "random_bucket_offset": config.random_bucket_offset,
                "ensure_bucket_coverage": config.ensure_bucket_coverage,
            }
        else:
            raw = dict(config)
            policy = (
                str(raw.get("policy", "bucket_ssl")).strip().lower().replace("-", "_")
            )
            values = {
                "policy": policy,
                "full_generation_probability": raw.get(
                    "full_generation_probability",
                    raw.get("full_generation_prob", 0.25),
                ),
                "unit_tokens": raw.get("unit_tokens", 3),
                "bucket_tokens": raw.get("bucket_tokens", 50),
                "visible_ratio_min": raw.get("visible_ratio_min", 0.0),
                "visible_ratio_max": raw.get("visible_ratio_max", 0.30),
                "random_bucket_offset": raw.get(
                    "random_bucket_offset",
                    policy in {"bucket_ssl_shifted", "bucket_ssl_shifted_coverage"},
                ),
                "ensure_bucket_coverage": raw.get(
                    "ensure_bucket_coverage",
                    policy in {"bucket_ssl_coverage", "bucket_ssl_shifted_coverage"},
                ),
            }
        values.update(overrides)
        self.config = BucketSSLMaskConfig(**values)

    def sample(
        self,
        valid_latent_mask: Tensor,
        batch: Mapping[str, Any] | None = None,
        progress: float | None = None,
        generator: torch.Generator | None = None,
    ) -> MaskPlan:
        del batch, progress
        valid = _validate_valid_mask(valid_latent_mask)
        batch_size, _ = valid.shape
        valid_tokens = valid.long().sum(dim=1)
        nonempty = valid_tokens > 0
        full_generation = self._sample_full_generation_rows(
            nonempty,
            generator=generator,
        )
        visible = torch.zeros_like(valid)
        short_ssl_fallback = torch.zeros(
            batch_size, device=valid.device, dtype=torch.bool
        )
        extra_totals: dict[str, float] = {}

        for row in range(batch_size):
            if not bool(nonempty[row].item()) or bool(full_generation[row].item()):
                continue
            length = int(valid_tokens[row].item())
            row_visible, has_eligible_unit, row_metrics = self._sample_ssl_visible_row(
                length,
                device=valid.device,
                generator=generator,
            )
            if not has_eligible_unit:
                # A clip shorter than the atomic unit (or an incompatible
                # ratio bound) cannot be a genuine SSL example.
                full_generation[row] = True
                short_ssl_fallback[row] = True
                continue
            visible[row, :length] = row_visible
            for name, value in row_metrics.items():
                extra_totals[name] = extra_totals.get(name, 0.0) + float(value)

        target = valid & ~visible
        _validate_mask_inside_valid(target, valid)
        masked_tokens = target.long().sum(dim=1)
        visible_tokens = visible.long().sum(dim=1)
        actual_ratio = masked_tokens.float() / valid_tokens.clamp_min(1).float()
        visible_ratio = visible_tokens.float() / valid_tokens.clamp_min(1).float()
        ssl_rows = nonempty & ~full_generation
        ssl_count = ssl_rows.float().sum().clamp_min(1.0)
        block_count, block_tokens = _visible_run_statistics(visible, ssl_rows)
        nonempty_count = nonempty.float().sum().clamp_min(1.0)
        actual_ratio_mean = (
            actual_ratio[nonempty].mean()
            if bool(nonempty.any().item())
            else actual_ratio.new_zeros(())
        )
        ssl_visible_ratio_mean = (
            visible_ratio[ssl_rows].mean()
            if bool(ssl_rows.any().item())
            else visible_ratio.new_zeros(())
        )
        mode_ids = torch.where(
            full_generation,
            torch.full_like(valid_tokens, MASK_MODE_FULL_GENERATION),
            torch.full_like(valid_tokens, MASK_MODE_SSL),
        )
        metrics: dict[str, MaskMetric] = {
            "requested_ratio_min": 1.0 - float(self.config.visible_ratio_max),
            "requested_ratio_max": 1.0 - float(self.config.visible_ratio_min),
            "actual_ratio": actual_ratio,
            "actual_ratio_mean": actual_ratio_mean,
            "visible_ratio": visible_ratio,
            "ssl_visible_ratio_mean": ssl_visible_ratio_mean,
            "masked_tokens": masked_tokens,
            "visible_tokens": visible_tokens,
            "valid_tokens": valid_tokens,
            "full_generation_fraction": full_generation.float().sum() / nonempty_count,
            "ssl_fraction": ssl_rows.float().sum() / nonempty_count,
            "short_ssl_fallback_fraction": short_ssl_fallback.float().sum()
            / nonempty_count,
            "visible_block_count_mean": block_count / ssl_count,
            "visible_block_length_mean": block_tokens / block_count.clamp_min(1.0),
            "unit_tokens": float(self.config.unit_tokens),
            "bucket_tokens": float(self.config.bucket_tokens),
            "configured_full_generation_probability": float(
                self.config.full_generation_probability
            ),
            "random_bucket_offset": float(self.config.random_bucket_offset),
            "ensure_bucket_coverage": float(self.config.ensure_bucket_coverage),
        }
        for name, total in extra_totals.items():
            metrics[name] = total / float(max(int(ssl_rows.sum().item()), 1))
        return MaskPlan(
            target_mask=target,
            policy_name=str(self.config.policy),
            encoding_mode="arbitrary_mask",
            full_generation_mask=full_generation,
            mode_ids=mode_ids,
            metrics=metrics,
        )

    def _sample_full_generation_rows(
        self,
        nonempty: Tensor,
        *,
        generator: torch.Generator | None,
    ) -> Tensor:
        result = torch.zeros_like(nonempty)
        indices = nonempty.nonzero(as_tuple=False).flatten()
        count = int(indices.numel())
        if count == 0:
            return result
        expected = count * float(self.config.full_generation_probability)
        selected_count = int(expected)
        fractional = expected - selected_count
        if fractional > 0.0:
            draw = torch.rand((), device=nonempty.device, generator=generator)
            selected_count += int(float(draw.item()) < fractional)
        if selected_count == 0:
            return result
        order = torch.randperm(count, device=nonempty.device, generator=generator)
        result[indices[order[:selected_count]]] = True
        return result

    def _sample_ssl_visible_row(
        self,
        length: int,
        *,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, bool, dict[str, float]]:
        visible = torch.zeros(length, device=device, dtype=torch.bool)
        eligible_groups: list[tuple[int, int]] = []
        bucket_size = int(self.config.bucket_tokens)
        unit_size = int(self.config.unit_tokens)
        maximum_ratio = float(self.config.visible_ratio_max)
        offset = 0
        if bool(self.config.random_bucket_offset) and length > 1:
            offset = int(
                torch.randint(
                    min(bucket_size, length),
                    (),
                    device=device,
                    generator=generator,
                ).item()
            )
        bucket_records: list[dict[str, list[tuple[int, int]]]] = []

        for bucket_start, bucket_end in _shifted_bucket_ranges(
            length, bucket_size=bucket_size, offset=offset
        ):
            bucket_length = bucket_end - bucket_start
            groups = _atomic_token_groups(bucket_start, bucket_end, unit_size)
            eligible = [
                group
                for group in groups
                if (group[1] - group[0]) / max(bucket_length, 1)
                <= maximum_ratio + 1.0e-12
            ]
            eligible_groups.extend(eligible)
            if not eligible:
                continue
            ratio = _uniform_scalar(
                float(self.config.visible_ratio_min),
                maximum_ratio,
                device=device,
                generator=generator,
            )
            budget = int(math.floor(bucket_length * ratio + 1.0e-12))
            order = torch.randperm(
                len(eligible), device=device, generator=generator
            ).tolist()
            selected_tokens = 0
            selected: list[tuple[int, int]] = []
            for index in order:
                start, end = eligible[index]
                group_tokens = end - start
                if selected_tokens + group_tokens <= budget:
                    selected.append((start, end))
                    selected_tokens += group_tokens
            bucket_records.append({"eligible": eligible, "selected": selected})

        if not eligible_groups:
            return visible, False, {}

        coverage_relocations = 0
        coverage_additions = 0
        if bool(self.config.ensure_bucket_coverage):
            empty_records = [
                record for record in bucket_records if not record["selected"]
            ]
            for record in empty_records:
                destination = min(
                    record["eligible"], key=lambda span: (span[1] - span[0], span[0])
                )
                destination_tokens = destination[1] - destination[0]
                exact_donors = [
                    donor
                    for donor in bucket_records
                    if len(donor["selected"]) > 1
                    and any(
                        end - start == destination_tokens
                        for start, end in donor["selected"]
                    )
                ]
                donors = exact_donors or [
                    donor for donor in bucket_records if len(donor["selected"]) > 1
                ]
                if donors:
                    donor = donors[
                        _randint(0, len(donors) - 1, device=device, generator=generator)
                    ]
                    matching = [
                        index
                        for index, (start, end) in enumerate(donor["selected"])
                        if end - start == destination_tokens
                    ]
                    donor_index = (
                        matching[0] if matching else len(donor["selected"]) - 1
                    )
                    donor["selected"].pop(donor_index)
                    coverage_relocations += 1
                else:
                    coverage_additions += 1
                record["selected"].append(destination)

        for record in bucket_records:
            for start, end in record["selected"]:
                visible[start:end] = True

        if not bool(visible.any().item()):
            # Keep an SSL row acoustically conditioned even when every sampled
            # bucket budget rounds below one atomic unit.
            selected = int(
                torch.randint(
                    len(eligible_groups),
                    (),
                    device=device,
                    generator=generator,
                ).item()
            )
            start, end = eligible_groups[selected]
            visible[start:end] = True
        covered_buckets = sum(bool(record["selected"]) for record in bucket_records)
        return (
            visible,
            True,
            {
                "bucket_offset_tokens_mean": float(offset),
                "eligible_bucket_count_mean": float(len(bucket_records)),
                "bucket_coverage_fraction_mean": float(covered_buckets)
                / float(max(len(bucket_records), 1)),
                "coverage_relocations_mean": float(coverage_relocations),
                "coverage_additions_mean": float(coverage_additions),
            },
        )


STRUCTURAL_MASK_POLICIES = (
    "bucket_scattered",
    "bucket_uniform_scattered",
    "bucket_span",
    "global_scattered",
    "global_variable_block_matched",
    "global_span",
)


@dataclass(frozen=True)
class StructuralSSLMaskConfig:
    """Budget-matched 2x2 ablation of bucket scope and span layout."""

    policy: str = "bucket_scattered"
    full_generation_probability: float = 0.25
    unit_tokens: int = 3
    bucket_tokens: int = 50
    block_lengths: tuple[int, ...] = (3, 5, 8)
    visible_ratio_min: float = 0.0
    visible_ratio_max: float = 0.30
    seed: int = 1234

    def __post_init__(self) -> None:
        if self.policy not in STRUCTURAL_MASK_POLICIES:
            raise ValueError(
                f"StructuralSSLMaskConfig policy must be one of {STRUCTURAL_MASK_POLICIES}"
            )
        if not 0.0 <= float(self.full_generation_probability) <= 1.0:
            raise ValueError("full_generation_probability must be in [0, 1]")
        if int(self.unit_tokens) < 1:
            raise ValueError("unit_tokens must be positive")
        if int(self.bucket_tokens) < int(self.unit_tokens):
            raise ValueError("bucket_tokens must be at least unit_tokens")
        if not self.block_lengths or any(int(value) < 1 for value in self.block_lengths):
            raise ValueError("block_lengths must all be positive")
        if self.policy == "global_variable_block_matched" and any(
            int(value) < int(self.unit_tokens) for value in self.block_lengths
        ):
            raise ValueError(
                "block_lengths must all be at least unit_tokens for "
                "global_variable_block_matched"
            )
        if (
            not 0.0
            <= float(self.visible_ratio_min)
            <= float(self.visible_ratio_max)
            <= 1.0
        ):
            raise ValueError(
                "visible ratios must satisfy 0 <= visible_ratio_min <= visible_ratio_max <= 1"
            )
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative")


class StructuralSSLMaskPolicy:
    """Sample a causal 2x2 mask ablation with exactly matched row budgets.

    A master generator draws full-generation rows plus one budget/layout seed
    pair per row. Budget sampling therefore consumes exactly the same random
    stream for every topology, while topology-specific draws stay in the
    independent row-local layout generator. When callers do not provide a
    generator, the policy owns a dedicated generator and never advances the
    model's global RNG.
    """

    def __init__(
        self,
        config: StructuralSSLMaskConfig | Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> None:
        self.config = _coerce_dataclass_config(
            StructuralSSLMaskConfig, config, overrides
        )
        self.policy_name = str(self.config.policy)
        self._generators: dict[str, torch.Generator] = {}

    def sample(
        self,
        valid_latent_mask: Tensor,
        batch: Mapping[str, Any] | None = None,
        progress: float | None = None,
        generator: torch.Generator | None = None,
    ) -> MaskPlan:
        del batch, progress
        valid = _validate_valid_mask(valid_latent_mask)
        batch_size, _ = valid.shape
        valid_tokens = valid.long().sum(dim=1)
        nonempty = valid_tokens > 0
        master = generator or self._generator(valid.device)
        full_generation = _sample_full_generation_rows(
            nonempty,
            probability=float(self.config.full_generation_probability),
            generator=master,
        )
        row_seeds = torch.randint(
            0,
            2**31 - 1,
            (batch_size, 2),
            device=valid.device,
            generator=master,
        )
        visible = torch.zeros_like(valid)
        short_ssl_fallback = torch.zeros_like(nonempty)
        requested_visible = torch.zeros_like(valid_tokens)
        layout_totals: dict[str, float] = {}

        for row in range(batch_size):
            if not bool(nonempty[row].item()) or bool(full_generation[row].item()):
                continue
            length = int(valid_tokens[row].item())
            budget_generator = torch.Generator(device=valid.device).manual_seed(
                int(row_seeds[row, 0].item())
            )
            layout_generator = torch.Generator(device=valid.device).manual_seed(
                int(row_seeds[row, 1].item())
            )
            bucket_units = self._sample_bucket_visible_units(
                length,
                device=valid.device,
                generator=budget_generator,
            )
            unit_count = sum(value[2] for value in bucket_units)
            if unit_count == 0:
                eligible = [
                    index
                    for index, (start, end, _) in enumerate(bucket_units)
                    if end - start >= int(self.config.unit_tokens)
                ]
                if not eligible:
                    full_generation[row] = True
                    short_ssl_fallback[row] = True
                    continue
                selected = eligible[
                    _randint(
                        0,
                        len(eligible) - 1,
                        device=valid.device,
                        generator=budget_generator,
                    )
                ]
                start, end, _ = bucket_units[selected]
                bucket_units[selected] = (start, end, 1)
                unit_count = 1
            requested_visible[row] = unit_count * int(self.config.unit_tokens)
            row_visible, row_layout_metrics = self._layout_visible(
                length,
                bucket_units=bucket_units,
                device=valid.device,
                generator=layout_generator,
            )
            visible[row, :length] = row_visible
            row_layout_metrics.update(
                self._local_bucket_metrics(row_visible, bucket_units=bucket_units)
            )
            for name, value in row_layout_metrics.items():
                layout_totals[name] = layout_totals.get(name, 0.0) + float(value)

        target = valid & ~visible
        _validate_mask_inside_valid(target, valid)
        masked_tokens = target.long().sum(dim=1)
        visible_tokens = visible.long().sum(dim=1)
        if bool(
            (
                visible_tokens[nonempty & ~full_generation]
                != requested_visible[nonempty & ~full_generation]
            )
            .any()
            .item()
        ):
            raise RuntimeError(
                "structural mask layout changed the sampled visible budget"
            )
        ssl_rows = nonempty & ~full_generation
        ssl_count = ssl_rows.float().sum().clamp_min(1.0)
        nonempty_count = nonempty.float().sum().clamp_min(1.0)
        actual_ratio = masked_tokens.float() / valid_tokens.clamp_min(1).float()
        visible_ratio = visible_tokens.float() / valid_tokens.clamp_min(1).float()
        visible_count, visible_span_tokens, visible_longest = _run_statistics(
            visible, ssl_rows
        )
        target_count, target_span_tokens, target_longest = _run_statistics(
            target, ssl_rows
        )
        policy = str(self.config.policy)
        metrics: dict[str, MaskMetric] = {
            "requested_ratio_min": 1.0 - float(self.config.visible_ratio_max),
            "requested_ratio_max": 1.0 - float(self.config.visible_ratio_min),
            "actual_ratio": actual_ratio,
            "actual_ratio_mean": actual_ratio[nonempty].mean()
            if bool(nonempty.any().item())
            else actual_ratio.new_zeros(()),
            "visible_ratio": visible_ratio,
            "ssl_visible_ratio_mean": visible_ratio[ssl_rows].mean()
            if bool(ssl_rows.any().item())
            else visible_ratio.new_zeros(()),
            "masked_tokens": masked_tokens,
            "visible_tokens": visible_tokens,
            "valid_tokens": valid_tokens,
            "sampled_visible_tokens": requested_visible,
            "full_generation_fraction": full_generation.float().sum() / nonempty_count,
            "ssl_fraction": ssl_rows.float().sum() / nonempty_count,
            "short_ssl_fallback_fraction": short_ssl_fallback.float().sum()
            / nonempty_count,
            "visible_block_count_mean": visible_count / ssl_count,
            "visible_block_length_mean": visible_span_tokens
            / visible_count.clamp_min(1.0),
            "visible_longest_span_mean": visible_longest / ssl_count,
            "target_span_count_mean": target_count / ssl_count,
            "target_span_length_mean": target_span_tokens / target_count.clamp_min(1.0),
            "target_longest_span_mean": target_longest / ssl_count,
            "unit_tokens": float(self.config.unit_tokens),
            "bucket_tokens": float(self.config.bucket_tokens),
            "configured_full_generation_probability": float(
                self.config.full_generation_probability
            ),
            "bucket_scope": float(policy.startswith("bucket_")),
            "span_layout": float(policy.endswith("_span")),
        }
        for name, total in layout_totals.items():
            metrics[name] = total / float(max(int(ssl_rows.sum().item()), 1))
        mode_ids = torch.where(
            full_generation,
            torch.full_like(valid_tokens, MASK_MODE_FULL_GENERATION),
            torch.full_like(valid_tokens, MASK_MODE_SSL),
        )
        return MaskPlan(
            target_mask=target,
            policy_name=policy,
            encoding_mode="arbitrary_mask",
            full_generation_mask=full_generation,
            mode_ids=mode_ids,
            metrics=metrics,
        )

    def _generator(self, device: torch.device) -> torch.Generator:
        key = str(device)
        generator = self._generators.get(key)
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(self.config.seed))
            self._generators[key] = generator
        return generator

    def _sample_bucket_visible_units(
        self,
        length: int,
        *,
        device: torch.device,
        generator: torch.Generator,
    ) -> list[tuple[int, int, int]]:
        unit = int(self.config.unit_tokens)
        records: list[tuple[int, int, int]] = []
        for start in range(0, length, int(self.config.bucket_tokens)):
            end = min(start + int(self.config.bucket_tokens), length)
            ratio = _uniform_scalar(
                float(self.config.visible_ratio_min),
                float(self.config.visible_ratio_max),
                device=device,
                generator=generator,
            )
            units = int(math.floor((end - start) * ratio + 1.0e-12)) // unit
            units = min(units, (end - start) // unit)
            records.append((start, end, units))
        return records

    def _layout_visible(
        self,
        length: int,
        *,
        bucket_units: list[tuple[int, int, int]],
        device: torch.device,
        generator: torch.Generator,
    ) -> tuple[Tensor, dict[str, float]]:
        visible = torch.zeros(length, device=device, dtype=torch.bool)
        unit = int(self.config.unit_tokens)
        policy = str(self.config.policy)
        layout_metrics = {
            "variable_partition_fallback_fraction": 0.0,
            "variable_selected_blocks_mean": 0.0,
        }
        if policy == "bucket_uniform_scattered":
            regions = self._uniform_bucket_units(
                bucket_units,
                requested_units=sum(value[2] for value in bucket_units),
                device=device,
                generator=generator,
            )
        elif policy.startswith("bucket_"):
            regions = bucket_units
        else:
            regions = [(0, length, sum(value[2] for value in bucket_units))]

        if policy == "global_variable_block_matched":
            selected, used_fallback = _select_exact_variable_partition_spans(
                length,
                budget=sum(value[2] for value in bucket_units) * unit,
                block_lengths=tuple(int(value) for value in self.config.block_lengths),
                unit_tokens=unit,
                device=device,
                generator=generator,
            )
            for start, end in selected:
                visible[start:end] = True
            layout_metrics.update(
                {
                    "variable_partition_fallback_fraction": float(used_fallback),
                    "variable_selected_blocks_mean": float(len(selected)),
                }
            )
            return visible, layout_metrics

        for start, end, requested_units in regions:
            if requested_units <= 0:
                continue
            available_units = (end - start) // unit
            if requested_units > available_units:
                raise RuntimeError("visible unit budget exceeds layout capacity")
            if policy.endswith("_scattered"):
                selected = torch.randperm(
                    available_units,
                    device=device,
                    generator=generator,
                )[:requested_units]
                for index in selected.tolist():
                    left = start + int(index) * unit
                    visible[left : left + unit] = True
            else:
                left_units = _randint(
                    0,
                    requested_units,
                    device=device,
                    generator=generator,
                )
                right_units = requested_units - left_units
                visible[start : start + left_units * unit] = True
                if right_units:
                    visible[end - right_units * unit : end] = True
        return visible, layout_metrics

    def _uniform_bucket_units(
        self,
        bucket_units: list[tuple[int, int, int]],
        *,
        requested_units: int,
        device: torch.device,
        generator: torch.Generator,
    ) -> list[tuple[int, int, int]]:
        """Redistribute a row budget as evenly as bucket capacities allow."""

        counts = [0] * len(bucket_units)
        unit = int(self.config.unit_tokens)
        capacities = [(end - start) // unit for start, end, _ in bucket_units]
        order = torch.randperm(
            len(bucket_units), device=device, generator=generator
        ).tolist()
        remaining = int(requested_units)
        while remaining > 0:
            progressed = False
            for index in order:
                if remaining == 0:
                    break
                if counts[index] >= capacities[index]:
                    continue
                counts[index] += 1
                remaining -= 1
                progressed = True
            if not progressed:
                raise RuntimeError("uniform bucket layout cannot fit visible budget")
        return [
            (start, end, counts[index])
            for index, (start, end, _) in enumerate(bucket_units)
        ]

    def _local_bucket_metrics(
        self,
        visible: Tensor,
        *,
        bucket_units: list[tuple[int, int, int]],
    ) -> dict[str, float]:
        ratios = visible.new_tensor(
            [
                float(visible[start:end].sum().item()) / float(max(end - start, 1))
                for start, end, _ in bucket_units
            ],
            dtype=torch.float32,
        )
        if ratios.numel() == 0:
            return {
                "local_visible_ratio_mean": 0.0,
                "local_visible_ratio_std": 0.0,
                "zero_visible_bucket_fraction": 0.0,
                "local_visible_ratio_min": 0.0,
                "local_visible_ratio_max": 0.0,
            }
        return {
            "local_visible_ratio_mean": float(ratios.mean().item()),
            "local_visible_ratio_std": float(ratios.std(unbiased=False).item()),
            "zero_visible_bucket_fraction": float((ratios == 0).float().mean().item()),
            "local_visible_ratio_min": float(ratios.min().item()),
            "local_visible_ratio_max": float(ratios.max().item()),
        }


@dataclass(frozen=True)
class GlobalVariableBlockMaskConfig:
    """Global visible blocks without fixed bucket boundaries."""

    policy: str = "global_variable_block"
    full_generation_probability: float = 0.25
    block_lengths: tuple[int, ...] = (3, 5, 8)
    # ``bucketed`` reproduces the original implementation, which samples one
    # visible ratio per virtual budget bucket. ``global`` samples exactly one
    # ratio for the whole SSL example; block placement remains global in both
    # modes.
    budget_mode: str = "bucketed"
    budget_bucket_tokens: int = 50
    visible_ratio_min: float = 0.0
    visible_ratio_max: float = 0.30

    def __post_init__(self) -> None:
        if self.policy not in {"global_variable_block", "global_blocks"}:
            raise ValueError(
                "GlobalVariableBlockMaskConfig requires policy='global_variable_block'"
            )
        if _normalize_budget_mode(self.budget_mode) not in {"bucketed", "global"}:
            raise ValueError("budget_mode must be 'bucketed' or 'global'")
        _validate_structured_mask_config(
            full_generation_probability=self.full_generation_probability,
            visible_ratio_min=self.visible_ratio_min,
            visible_ratio_max=self.visible_ratio_max,
            budget_bucket_tokens=self.budget_bucket_tokens,
            block_lengths=self.block_lengths,
        )


@dataclass(frozen=True)
class GlobalDynamicBlockV1MaskConfig:
    """Production global-budget dynamic blocks with no bucketed mode switch."""

    policy: str = "global_dynamic_block_v1"
    full_generation_probability: float = 0.25
    block_lengths: tuple[int, ...] = (3, 5, 8)
    visible_ratio_min: float = 0.0
    visible_ratio_max: float = 0.30

    def __post_init__(self) -> None:
        if self.policy != "global_dynamic_block_v1":
            raise ValueError(
                "GlobalDynamicBlockV1MaskConfig requires "
                "policy='global_dynamic_block_v1'"
            )
        _validate_structured_mask_config(
            full_generation_probability=self.full_generation_probability,
            visible_ratio_min=self.visible_ratio_min,
            visible_ratio_max=self.visible_ratio_max,
            budget_bucket_tokens=3,
            block_lengths=self.block_lengths,
        )


@dataclass(frozen=True)
class MultiAnchorMaskConfig:
    """A small number of information-rich visible anchors."""

    policy: str = "multi_anchor"
    full_generation_probability: float = 0.25
    anchor_count_min: int = 2
    anchor_count_max: int = 6
    anchor_tokens_min: int = 5
    anchor_tokens_max: int = 12
    anchor_gap_tokens: int = 3
    budget_bucket_tokens: int = 50
    visible_ratio_min: float = 0.0
    visible_ratio_max: float = 0.30

    def __post_init__(self) -> None:
        if self.policy not in {"multi_anchor", "multi_anchor_context"}:
            raise ValueError("MultiAnchorMaskConfig requires policy='multi_anchor'")
        _validate_structured_mask_config(
            full_generation_probability=self.full_generation_probability,
            visible_ratio_min=self.visible_ratio_min,
            visible_ratio_max=self.visible_ratio_max,
            budget_bucket_tokens=self.budget_bucket_tokens,
            block_lengths=(self.anchor_tokens_min, self.anchor_tokens_max),
        )
        if not 1 <= int(self.anchor_count_min) <= int(self.anchor_count_max):
            raise ValueError(
                "anchor counts must satisfy 1 <= anchor_count_min <= anchor_count_max"
            )
        if int(self.anchor_tokens_min) < 3:
            raise ValueError("anchor_tokens_min must be at least 3")
        if int(self.anchor_tokens_min) > int(self.anchor_tokens_max):
            raise ValueError(
                "anchor token lengths must satisfy anchor_tokens_min <= anchor_tokens_max"
            )
        if int(self.anchor_gap_tokens) < 0:
            raise ValueError("anchor_gap_tokens must be non-negative")


@dataclass(frozen=True)
class HierarchicalMacroMicroMaskConfig:
    """Macro-region dropout with short visible micro blocks."""

    policy: str = "hierarchical_macro_micro"
    full_generation_probability: float = 0.25
    macro_tokens_min: int = 25
    macro_tokens_max: int = 50
    macro_active_probability: float = 0.50
    micro_block_lengths: tuple[int, ...] = (3, 5, 8)
    budget_bucket_tokens: int = 50
    visible_ratio_min: float = 0.0
    visible_ratio_max: float = 0.30

    def __post_init__(self) -> None:
        if self.policy not in {"hierarchical_macro_micro", "macro_micro"}:
            raise ValueError(
                "HierarchicalMacroMicroMaskConfig requires "
                "policy='hierarchical_macro_micro'"
            )
        _validate_structured_mask_config(
            full_generation_probability=self.full_generation_probability,
            visible_ratio_min=self.visible_ratio_min,
            visible_ratio_max=self.visible_ratio_max,
            budget_bucket_tokens=self.budget_bucket_tokens,
            block_lengths=self.micro_block_lengths,
        )
        if int(self.macro_tokens_min) < 3:
            raise ValueError("macro_tokens_min must be at least 3")
        if int(self.macro_tokens_min) > int(self.macro_tokens_max):
            raise ValueError(
                "macro token lengths must satisfy macro_tokens_min <= macro_tokens_max"
            )
        if not 0.0 < float(self.macro_active_probability) < 1.0:
            raise ValueError("macro_active_probability must be in (0, 1)")


class _StructuredSSLMaskPolicy:
    """Shared full-generation mixing and diagnostics for topology ablations."""

    policy_name = "structured_ssl"
    config: Any

    def sample(
        self,
        valid_latent_mask: Tensor,
        batch: Mapping[str, Any] | None = None,
        progress: float | None = None,
        generator: torch.Generator | None = None,
    ) -> MaskPlan:
        del batch, progress
        valid = _validate_valid_mask(valid_latent_mask)
        batch_size, _ = valid.shape
        valid_tokens = valid.long().sum(dim=1)
        nonempty = valid_tokens > 0
        full_generation = _sample_full_generation_rows(
            nonempty,
            probability=float(self.config.full_generation_probability),
            generator=generator,
        )
        visible = torch.zeros_like(valid)
        short_ssl_fallback = torch.zeros_like(nonempty)
        extra_totals: dict[str, float] = {}

        for row in range(batch_size):
            if not bool(nonempty[row].item()) or bool(full_generation[row].item()):
                continue
            length = int(valid_tokens[row].item())
            row_visible, eligible, row_metrics = self._sample_ssl_visible_row(
                length,
                device=valid.device,
                generator=generator,
            )
            if not eligible or not bool(row_visible.any().item()):
                full_generation[row] = True
                short_ssl_fallback[row] = True
                continue
            visible[row, :length] = row_visible
            for name, value in row_metrics.items():
                extra_totals[name] = extra_totals.get(name, 0.0) + float(value)

        target = valid & ~visible
        _validate_mask_inside_valid(target, valid)
        masked_tokens = target.long().sum(dim=1)
        visible_tokens = visible.long().sum(dim=1)
        actual_ratio = masked_tokens.float() / valid_tokens.clamp_min(1).float()
        visible_ratio = visible_tokens.float() / valid_tokens.clamp_min(1).float()
        ssl_rows = nonempty & ~full_generation
        ssl_count = ssl_rows.float().sum().clamp_min(1.0)
        nonempty_count = nonempty.float().sum().clamp_min(1.0)
        block_count, block_tokens = _visible_run_statistics(visible, ssl_rows)
        mode_ids = torch.where(
            full_generation,
            torch.full_like(valid_tokens, MASK_MODE_FULL_GENERATION),
            torch.full_like(valid_tokens, MASK_MODE_SSL),
        )
        metrics: dict[str, MaskMetric] = {
            "requested_ratio_min": 1.0 - float(self.config.visible_ratio_max),
            "requested_ratio_max": 1.0 - float(self.config.visible_ratio_min),
            "actual_ratio": actual_ratio,
            "actual_ratio_mean": actual_ratio[nonempty].mean()
            if bool(nonempty.any().item())
            else actual_ratio.new_zeros(()),
            "visible_ratio": visible_ratio,
            "ssl_visible_ratio_mean": visible_ratio[ssl_rows].mean()
            if bool(ssl_rows.any().item())
            else visible_ratio.new_zeros(()),
            "masked_tokens": masked_tokens,
            "visible_tokens": visible_tokens,
            "valid_tokens": valid_tokens,
            "full_generation_fraction": full_generation.float().sum() / nonempty_count,
            "ssl_fraction": ssl_rows.float().sum() / nonempty_count,
            "short_ssl_fallback_fraction": short_ssl_fallback.float().sum()
            / nonempty_count,
            "visible_block_count_mean": block_count / ssl_count,
            "visible_block_length_mean": block_tokens / block_count.clamp_min(1.0),
            "configured_full_generation_probability": float(
                self.config.full_generation_probability
            ),
            "budget_mode_global": float(self._budget_mode() == "global"),
            "configured_budget_bucket_tokens": float(
                getattr(self.config, "budget_bucket_tokens", 0)
            ),
        }
        for name, total in extra_totals.items():
            metrics[name] = total / float(max(int(ssl_rows.sum().item()), 1))
        return MaskPlan(
            target_mask=target,
            policy_name=self.policy_name,
            encoding_mode="arbitrary_mask",
            full_generation_mask=full_generation,
            mode_ids=mode_ids,
            metrics=metrics,
        )

    def _sample_ssl_visible_row(
        self,
        length: int,
        *,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, bool, dict[str, float]]:
        raise NotImplementedError

    def _visible_budget(
        self,
        length: int,
        *,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> int:
        budget_mode = self._budget_mode()
        return _sample_visible_budget(
            length,
            # A single bucket spanning the valid sequence means the ratio is
            # drawn once globally. The legacy path keeps independently drawn
            # ratios per virtual bucket for checkpoint/run reproducibility.
            bucket_tokens=(
                length
                if budget_mode == "global"
                else int(self.config.budget_bucket_tokens)
            ),
            ratio_min=float(self.config.visible_ratio_min),
            ratio_max=float(self.config.visible_ratio_max),
            minimum_unit=3,
            device=device,
            generator=generator,
        )

    def _budget_mode(self) -> str:
        return _normalize_budget_mode(getattr(self.config, "budget_mode", "bucketed"))


class GlobalVariableBlockMaskPolicy(_StructuredSSLMaskPolicy):
    policy_name = "global_variable_block"

    def __init__(
        self,
        config: GlobalVariableBlockMaskConfig | Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> None:
        self.config = _coerce_dataclass_config(
            GlobalVariableBlockMaskConfig, config, overrides
        )

    def _sample_ssl_visible_row(
        self,
        length: int,
        *,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, bool, dict[str, float]]:
        visible = torch.zeros(length, device=device, dtype=torch.bool)
        spans = _random_partition_spans(
            0,
            length,
            tuple(int(value) for value in self.config.block_lengths),
            device=device,
            generator=generator,
        )
        if not spans:
            return visible, False, {}
        budget = self._visible_budget(length, device=device, generator=generator)
        selected = _select_spans_with_budget(
            spans, budget=budget, device=device, generator=generator
        )
        for start, end in selected:
            visible[start:end] = True
        return visible, True, {"selected_atomic_blocks_mean": float(len(selected))}


class GlobalDynamicBlockV1MaskPolicy(GlobalVariableBlockMaskPolicy):
    """Versioned production policy: one visible-ratio draw per SSL example."""

    policy_name = "global_dynamic_block_v1"

    def __init__(
        self,
        config: GlobalDynamicBlockV1MaskConfig | Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> None:
        self.config = _coerce_dataclass_config(
            GlobalDynamicBlockV1MaskConfig, config, overrides
        )

    def _budget_mode(self) -> str:
        return "global"

    def _visible_budget(
        self,
        length: int,
        *,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> int:
        return _sample_visible_budget(
            length,
            bucket_tokens=length,
            ratio_min=float(self.config.visible_ratio_min),
            ratio_max=float(self.config.visible_ratio_max),
            minimum_unit=3,
            device=device,
            generator=generator,
        )


class MultiAnchorMaskPolicy(_StructuredSSLMaskPolicy):
    policy_name = "multi_anchor"

    def __init__(
        self,
        config: MultiAnchorMaskConfig | Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> None:
        self.config = _coerce_dataclass_config(MultiAnchorMaskConfig, config, overrides)

    def _sample_ssl_visible_row(
        self,
        length: int,
        *,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, bool, dict[str, float]]:
        visible = torch.zeros(length, device=device, dtype=torch.bool)
        if length < 3:
            return visible, False, {}
        budget = self._visible_budget(length, device=device, generator=generator)
        minimum = int(self.config.anchor_tokens_min)
        maximum = int(self.config.anchor_tokens_max)
        if length < minimum or budget < minimum:
            span = min(max(budget, 3), length)
            start = _randint(0, length - span, device=device, generator=generator)
            visible[start : start + span] = True
            return (
                visible,
                True,
                {
                    "anchor_count_mean": 1.0,
                    "anchor_short_budget_fallback_mean": 1.0,
                },
            )

        feasible_max = min(
            int(self.config.anchor_count_max),
            max(1, budget // minimum),
        )
        feasible_min = min(int(self.config.anchor_count_min), feasible_max)
        requested = _randint(
            feasible_min, feasible_max, device=device, generator=generator
        )
        gap = int(self.config.anchor_gap_tokens)
        spans: list[tuple[int, int]] = []
        remaining = budget
        attempts = max(64, requested * 32)
        for _ in range(attempts):
            if len(spans) >= requested or remaining < minimum:
                break
            span_length = _randint(
                minimum,
                min(maximum, remaining, length),
                device=device,
                generator=generator,
            )
            start = _randint(
                0, length - span_length, device=device, generator=generator
            )
            end = start + span_length
            if any(
                not (end + gap <= left or right + gap <= start) for left, right in spans
            ):
                continue
            spans.append((start, end))
            remaining -= span_length
        if not spans:
            span_length = min(maximum, budget, length)
            start = _randint(
                0, length - span_length, device=device, generator=generator
            )
            spans.append((start, start + span_length))
        for start, end in spans:
            visible[start:end] = True
        return (
            visible,
            True,
            {
                "anchor_count_mean": float(len(spans)),
                "anchor_short_budget_fallback_mean": 0.0,
            },
        )


class HierarchicalMacroMicroMaskPolicy(_StructuredSSLMaskPolicy):
    policy_name = "hierarchical_macro_micro"

    def __init__(
        self,
        config: HierarchicalMacroMicroMaskConfig | Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> None:
        self.config = _coerce_dataclass_config(
            HierarchicalMacroMicroMaskConfig, config, overrides
        )

    def _sample_ssl_visible_row(
        self,
        length: int,
        *,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, bool, dict[str, float]]:
        visible = torch.zeros(length, device=device, dtype=torch.bool)
        if length < 3:
            return visible, False, {}
        macros = _random_range_partition_spans(
            length,
            minimum=int(self.config.macro_tokens_min),
            maximum=int(self.config.macro_tokens_max),
            device=device,
            generator=generator,
        )
        if not macros:
            return visible, False, {}
        active = torch.rand(len(macros), device=device, generator=generator) < float(
            self.config.macro_active_probability
        )
        if not bool(active.any().item()):
            active[_randint(0, len(macros) - 1, device=device, generator=generator)] = (
                True
            )
        if len(macros) > 1 and bool(active.all().item()):
            active_indices = active.nonzero(as_tuple=False).flatten()
            hidden_index = int(
                active_indices[
                    _randint(
                        0,
                        int(active_indices.numel()) - 1,
                        device=device,
                        generator=generator,
                    )
                ].item()
            )
            active[hidden_index] = False

        candidates: list[tuple[int, int]] = []
        micro_lengths = tuple(int(value) for value in self.config.micro_block_lengths)
        for index, (start, end) in enumerate(macros):
            if bool(active[index].item()):
                candidates.extend(
                    _random_partition_spans(
                        start,
                        end,
                        micro_lengths,
                        device=device,
                        generator=generator,
                    )
                )
        if not candidates:
            return visible, False, {}
        budget = self._visible_budget(length, device=device, generator=generator)
        selected = _select_spans_with_budget(
            candidates,
            budget=budget,
            minimum_gap=1,
            device=device,
            generator=generator,
        )
        for start, end in selected:
            visible[start:end] = True
        return (
            visible,
            True,
            {
                "macro_count_mean": float(len(macros)),
                "macro_active_fraction_mean": float(active.float().mean().item()),
                "selected_atomic_blocks_mean": float(len(selected)),
            },
        )


def build_mask_policy(
    config: (
        RandomSpanMaskConfig
        | BucketSSLMaskConfig
        | StructuralSSLMaskConfig
        | GlobalVariableBlockMaskConfig
        | GlobalDynamicBlockV1MaskConfig
        | MultiAnchorMaskConfig
        | HierarchicalMacroMicroMaskConfig
        | Mapping[str, Any]
        | str
        | None
    ) = None,
) -> PriorMaskPolicy:
    """Build a TTA mask policy without changing the speech-task registry."""

    if isinstance(config, str):
        config = {"policy": config}
    if isinstance(config, BucketSSLMaskConfig):
        return BucketSSLMaskPolicy(config)
    if isinstance(config, StructuralSSLMaskConfig):
        return StructuralSSLMaskPolicy(config)
    if isinstance(config, GlobalVariableBlockMaskConfig):
        return GlobalVariableBlockMaskPolicy(config)
    if isinstance(config, GlobalDynamicBlockV1MaskConfig):
        return GlobalDynamicBlockV1MaskPolicy(config)
    if isinstance(config, MultiAnchorMaskConfig):
        return MultiAnchorMaskPolicy(config)
    if isinstance(config, HierarchicalMacroMicroMaskConfig):
        return HierarchicalMacroMicroMaskPolicy(config)
    if isinstance(config, RandomSpanMaskConfig):
        return RandomSpanMaskPolicy(config)
    policy = (
        "random_span"
        if config is None
        else str(dict(config).get("policy", "random_span"))
    )
    policy = policy.strip().lower().replace("-", "_")
    if policy in {"random_span", "tts_random_span"}:
        return RandomSpanMaskPolicy(config)
    if policy in {
        "bucket_ssl",
        "tta_bucket_ssl",
        "bucket_ssl_shifted",
        "bucket_ssl_coverage",
        "bucket_ssl_shifted_coverage",
    }:
        return BucketSSLMaskPolicy(config)
    if policy in STRUCTURAL_MASK_POLICIES:
        return StructuralSSLMaskPolicy(config)
    if policy in {"global_variable_block", "global_blocks"}:
        return GlobalVariableBlockMaskPolicy(config)
    if policy == "global_dynamic_block_v1":
        return GlobalDynamicBlockV1MaskPolicy(config)
    if policy in {"multi_anchor", "multi_anchor_context"}:
        return MultiAnchorMaskPolicy(config)
    if policy in {"hierarchical_macro_micro", "macro_micro"}:
        return HierarchicalMacroMicroMaskPolicy(config)
    raise ValueError(
        "mask.policy must be 'random_span', a 'bucket_ssl' variant, "
        "a structural 2x2 policy, 'global_variable_block', 'multi_anchor', "
        "or 'hierarchical_macro_micro'"
    )


def _validate_valid_mask(valid_mask: Tensor) -> Tensor:
    if not isinstance(valid_mask, Tensor) or valid_mask.ndim != 2:
        raise ValueError("valid_latent_mask must be a Tensor with shape [B, T]")
    valid = valid_mask.to(dtype=torch.bool)
    # TTS span sampling derives each row from its length, so padding must be a
    # suffix. Rejecting holes prevents silently changing the requested ratio.
    if valid.shape[1] > 1:
        has_hole = ((~valid[:, :-1]) & valid[:, 1:]).any()
        if bool(has_hole.item()):
            raise ValueError(
                "valid_latent_mask must be left-aligned with suffix padding"
            )
    return valid


def _validate_target_mask(target: Tensor, valid: Tensor) -> None:
    _validate_mask_inside_valid(target, valid)
    for row in target:
        indices = row.nonzero(as_tuple=False).flatten()
        if indices.numel() > 1 and int(indices[-1] - indices[0] + 1) != int(
            indices.numel()
        ):
            raise RuntimeError(
                "random_span policy must return at most one contiguous span per row"
            )


def _validate_mask_inside_valid(target: Tensor, valid: Tensor) -> None:
    if target.shape != valid.shape:
        raise RuntimeError("mask policy returned an invalid target-mask shape")
    if bool((target & ~valid).any().item()):
        raise RuntimeError("mask policy selected padded latent positions")


def _atomic_token_groups(
    start: int, end: int, unit_tokens: int
) -> list[tuple[int, int]]:
    """Partition a bucket into units, merging a short remainder backward."""

    length = end - start
    full_units = length // unit_tokens
    if full_units == 0:
        return []
    remainder = length % unit_tokens
    if remainder == 0:
        return [
            (position, position + unit_tokens)
            for position in range(start, end, unit_tokens)
        ]
    groups = [
        (start + index * unit_tokens, start + (index + 1) * unit_tokens)
        for index in range(max(full_units - 1, 0))
    ]
    final_start = start + max(full_units - 1, 0) * unit_tokens
    groups.append((final_start, end))
    return groups


def _shifted_bucket_ranges(
    length: int,
    *,
    bucket_size: int,
    offset: int,
) -> list[tuple[int, int]]:
    """Partition a sequence using a random bucket phase without wrapping time."""

    if length <= 0:
        return []
    if not 0 <= offset < min(bucket_size, length):
        raise ValueError("bucket offset must stay inside the first bucket")
    ranges: list[tuple[int, int]] = []
    position = 0
    if offset > 0:
        ranges.append((0, offset))
        position = offset
    while position < length:
        end = min(position + bucket_size, length)
        ranges.append((position, end))
        position = end
    return ranges


def _uniform_scalar(
    minimum: float,
    maximum: float,
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> float:
    if minimum == maximum:
        return minimum
    value = torch.empty((), device=device).uniform_(
        minimum, maximum, generator=generator
    )
    return float(value.item())


def _sample_span_with_generator(
    valid: Tensor,
    *,
    min_ratio: float,
    max_ratio: float,
    generator: torch.Generator,
) -> Tensor:
    batch_size, max_len = valid.shape
    if max_len == 0:
        return torch.zeros_like(valid)
    lengths = valid.long().sum(dim=1)
    ratios = torch.empty((batch_size,), device=valid.device).uniform_(
        min_ratio,
        max_ratio,
        generator=generator,
    )
    spans = (lengths.float() * ratios).long().clamp_min(1)
    spans = torch.minimum(spans, lengths.clamp_min(0))
    start_max = (lengths - spans).clamp_min(0)
    starts = (
        torch.rand((batch_size,), device=valid.device, generator=generator)
        * (start_max + 1).float()
    ).long()
    positions = torch.arange(max_len, device=valid.device)[None, :]
    return (
        (positions >= starts[:, None]) & (positions < (starts + spans)[:, None]) & valid
    )


def _validate_structured_mask_config(
    *,
    full_generation_probability: float,
    visible_ratio_min: float,
    visible_ratio_max: float,
    budget_bucket_tokens: int,
    block_lengths: tuple[int, ...],
) -> None:
    if not 0.0 <= float(full_generation_probability) <= 1.0:
        raise ValueError("full_generation_probability must be in [0, 1]")
    if not 0.0 <= float(visible_ratio_min) <= float(visible_ratio_max) <= 1.0:
        raise ValueError(
            "visible ratios must satisfy 0 <= visible_ratio_min <= visible_ratio_max <= 1"
        )
    if int(budget_bucket_tokens) < 3:
        raise ValueError("budget_bucket_tokens must be at least 3")
    if not block_lengths or any(int(value) < 3 for value in block_lengths):
        raise ValueError("structured mask block lengths must all be at least 3")


def _coerce_dataclass_config(config_type, config, overrides: Mapping[str, Any]):
    if config is None:
        values: dict[str, Any] = {}
    elif isinstance(config, config_type):
        if not overrides:
            return config
        values = dict(vars(config))
    else:
        values = dict(config)
    values.update(dict(overrides))
    for name in ("block_lengths", "micro_block_lengths"):
        if name in values:
            values[name] = tuple(int(value) for value in values[name])
    return config_type(**values)


def _sample_full_generation_rows(
    nonempty: Tensor,
    *,
    probability: float,
    generator: torch.Generator | None,
) -> Tensor:
    result = torch.zeros_like(nonempty)
    indices = nonempty.nonzero(as_tuple=False).flatten()
    count = int(indices.numel())
    if count == 0:
        return result
    expected = count * float(probability)
    selected_count = int(expected)
    fractional = expected - selected_count
    if fractional > 0.0:
        draw = torch.rand((), device=nonempty.device, generator=generator)
        selected_count += int(float(draw.item()) < fractional)
    if selected_count == 0:
        return result
    order = torch.randperm(count, device=nonempty.device, generator=generator)
    result[indices[order[:selected_count]]] = True
    return result


def _sample_visible_budget(
    length: int,
    *,
    bucket_tokens: int,
    ratio_min: float,
    ratio_max: float,
    minimum_unit: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> int:
    """Use virtual buckets to match the current bucket policy's budget scale."""

    if length < minimum_unit:
        return 0
    budget = 0
    for start in range(0, length, bucket_tokens):
        bucket_length = min(bucket_tokens, length - start)
        ratio = _uniform_scalar(
            ratio_min,
            ratio_max,
            device=device,
            generator=generator,
        )
        bucket_budget = int(math.floor(bucket_length * ratio + 1.0e-12))
        budget += bucket_budget - (bucket_budget % minimum_unit)
    return min(max(budget, minimum_unit), length)


def _normalize_budget_mode(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def _randint(
    minimum: int,
    maximum: int,
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> int:
    if maximum < minimum:
        raise ValueError(f"invalid randint range [{minimum}, {maximum}]")
    if minimum == maximum:
        return int(minimum)
    return int(
        torch.randint(
            minimum,
            maximum + 1,
            (),
            device=device,
            generator=generator,
        ).item()
    )


def _random_partition_spans(
    start: int,
    end: int,
    lengths: tuple[int, ...],
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> list[tuple[int, int]]:
    choices = tuple(sorted(set(int(value) for value in lengths)))
    minimum = min(choices)
    spans: list[tuple[int, int]] = []
    position = start
    while end - position >= minimum:
        remaining = end - position
        eligible = tuple(value for value in choices if value <= remaining)
        span_length = eligible[
            _randint(0, len(eligible) - 1, device=device, generator=generator)
        ]
        spans.append((position, position + span_length))
        position += span_length
    if position < end and spans:
        previous_start, _ = spans[-1]
        spans[-1] = (previous_start, end)
    return spans


def _select_exact_variable_partition_spans(
    length: int,
    *,
    budget: int,
    block_lengths: tuple[int, ...],
    unit_tokens: int,
    device: torch.device,
    generator: torch.Generator,
    max_attempts: int = 32,
) -> tuple[list[tuple[int, int]], bool]:
    """Select complete variable blocks whose lengths exactly match ``budget``."""

    if budget <= 0:
        return [], False
    allowed = tuple(sorted(set(int(value) for value in block_lengths)))
    for _ in range(max_attempts):
        spans = _random_partition_spans(
            0,
            length,
            allowed,
            device=device,
            generator=generator,
        )
        eligible = [span for span in spans if span[1] - span[0] in allowed]
        if not eligible:
            continue
        order = torch.randperm(
            len(eligible), device=device, generator=generator
        ).tolist()
        reachable: dict[int, tuple[int, ...]] = {0: ()}
        for index in order:
            span_length = eligible[index][1] - eligible[index][0]
            for used, selected in tuple(reachable.items())[::-1]:
                updated = used + span_length
                if updated > budget or updated in reachable:
                    continue
                reachable[updated] = (*selected, index)
            if budget in reachable:
                indices = reachable[budget]
                return sorted((eligible[index] for index in indices)), False

    if budget % unit_tokens:
        raise RuntimeError("matched variable-block budget must align to unit_tokens")
    spans = [
        (start, start + unit_tokens)
        for start in range(0, length - unit_tokens + 1, unit_tokens)
    ]
    requested = budget // unit_tokens
    if requested > len(spans):
        raise RuntimeError("fallback partition cannot fit visible budget")
    order = torch.randperm(len(spans), device=device, generator=generator).tolist()
    return sorted(spans[index] for index in order[:requested]), True


def _random_range_partition_spans(
    length: int,
    *,
    minimum: int,
    maximum: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> list[tuple[int, int]]:
    if length < 3:
        return []
    if length < minimum:
        return [(0, length)]
    spans: list[tuple[int, int]] = []
    position = 0
    while length - position >= minimum:
        remaining = length - position
        span_length = _randint(
            minimum,
            min(maximum, remaining),
            device=device,
            generator=generator,
        )
        spans.append((position, position + span_length))
        position += span_length
    if position < length:
        previous_start, _ = spans[-1]
        spans[-1] = (previous_start, length)
    return spans


def _select_spans_with_budget(
    spans: list[tuple[int, int]],
    *,
    budget: int,
    minimum_gap: int = 0,
    device: torch.device,
    generator: torch.Generator | None,
) -> list[tuple[int, int]]:
    if not spans:
        return []
    order = torch.randperm(len(spans), device=device, generator=generator).tolist()
    selected: list[tuple[int, int]] = []
    used = 0
    for index in order:
        span = spans[index]
        span_length = span[1] - span[0]
        if any(
            not (
                span[1] + minimum_gap <= selected_start
                or selected_end + minimum_gap <= span[0]
            )
            for selected_start, selected_end in selected
        ):
            continue
        if used + span_length <= budget:
            selected.append(span)
            used += span_length
    if not selected:
        selected.append(min(spans, key=lambda value: value[1] - value[0]))
    return selected


def _visible_run_statistics(visible: Tensor, ssl_rows: Tensor) -> tuple[Tensor, Tensor]:
    block_count = visible.new_zeros((), dtype=torch.float32)
    block_tokens = visible.new_zeros((), dtype=torch.float32)
    for row in ssl_rows.nonzero(as_tuple=False).flatten().tolist():
        row_visible = visible[row]
        padded = torch.cat(
            (
                torch.zeros(1, device=visible.device, dtype=torch.bool),
                row_visible,
                torch.zeros(1, device=visible.device, dtype=torch.bool),
            )
        )
        changes = padded[1:].to(torch.int8) - padded[:-1].to(torch.int8)
        starts = (changes == 1).nonzero(as_tuple=False).flatten()
        ends = (changes == -1).nonzero(as_tuple=False).flatten()
        block_count += float(starts.numel())
        if starts.numel():
            block_tokens += (ends - starts).float().sum()
    return block_count, block_tokens


def _run_statistics(
    mask: Tensor, selected_rows: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Return total run count, total tokens, and sum of per-row longest runs."""

    run_count = mask.new_zeros((), dtype=torch.float32)
    run_tokens = mask.new_zeros((), dtype=torch.float32)
    longest_total = mask.new_zeros((), dtype=torch.float32)
    for row in selected_rows.nonzero(as_tuple=False).flatten().tolist():
        padded = torch.cat(
            (
                torch.zeros(1, device=mask.device, dtype=torch.bool),
                mask[row],
                torch.zeros(1, device=mask.device, dtype=torch.bool),
            )
        )
        changes = padded[1:].to(torch.int8) - padded[:-1].to(torch.int8)
        starts = (changes == 1).nonzero(as_tuple=False).flatten()
        ends = (changes == -1).nonzero(as_tuple=False).flatten()
        lengths = (ends - starts).float()
        run_count += float(starts.numel())
        if lengths.numel():
            run_tokens += lengths.sum()
            longest_total += lengths.max()
    return run_count, run_tokens, longest_total


__all__ = [
    "BucketSSLMaskConfig",
    "BucketSSLMaskPolicy",
    "GlobalVariableBlockMaskConfig",
    "GlobalVariableBlockMaskPolicy",
    "HierarchicalMacroMicroMaskConfig",
    "HierarchicalMacroMicroMaskPolicy",
    "MASK_MODE_FULL_GENERATION",
    "MASK_MODE_SSL",
    "MaskPlan",
    "MultiAnchorMaskConfig",
    "MultiAnchorMaskPolicy",
    "PriorMaskPolicy",
    "RandomSpanMaskConfig",
    "RandomSpanMaskPolicy",
    "STRUCTURAL_MASK_POLICIES",
    "StructuralSSLMaskConfig",
    "StructuralSSLMaskPolicy",
    "build_mask_policy",
]
