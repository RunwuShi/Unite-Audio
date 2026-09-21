from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


GLOBAL_DYNAMIC_BLOCK_V1 = "global_dynamic_block_v1"


@dataclass(frozen=True)
class MaskContract:
    contract_id: str
    policy_config: dict[str, Any]
    migrated_from: str | None = None

    @property
    def canonical_json(self) -> str:
        payload = {
            "contract_id": self.contract_id,
            "policy_config": self.policy_config,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    def metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.contract_id,
            "fingerprint": self.fingerprint,
            "canonical": json.loads(self.canonical_json),
        }
        if self.migrated_from is not None:
            result["migrated_from"] = self.migrated_from
        return result


def global_dynamic_block_v1(
    *,
    full_generation_probability: float = 0.25,
    block_lengths: tuple[int, ...] | list[int] = (3, 5, 8),
    visible_ratio_min: float = 0.0,
    visible_ratio_max: float = 0.30,
) -> dict[str, Any]:
    """Return the complete production mask preset without hidden defaults."""

    return {
        "policy": GLOBAL_DYNAMIC_BLOCK_V1,
        "full_generation_probability": float(full_generation_probability),
        "block_lengths": [int(value) for value in block_lengths],
        "visible_ratio_min": float(visible_ratio_min),
        "visible_ratio_max": float(visible_ratio_max),
    }


def resolve_mask_contract(
    mask: Mapping[str, Any],
    *,
    legacy_model_policy: Any | None = None,
) -> MaskContract:
    """Parse mask semantics once and return an immutable, hashable contract.

    New production configs use ``global_dynamic_block_v1``. The only automatic
    migration is the exact historical spelling used by the current run:
    ``global_variable_block`` with ``budget_mode=global``. Bucketed behavior is
    deliberately assigned a different legacy contract and cannot become the
    production policy through a missing default.
    """

    raw = dict(mask)
    policy = _normalize(raw.get("policy", legacy_model_policy or "random_span"))
    legacy = _normalize(legacy_model_policy) if legacy_model_policy is not None else None
    if legacy and legacy != policy:
        raise ValueError(
            "conflicting mask selectors are forbidden: "
            f"mask.policy={policy!r} model.mask_policy={legacy!r}"
        )

    if policy == GLOBAL_DYNAMIC_BLOCK_V1:
        return MaskContract(GLOBAL_DYNAMIC_BLOCK_V1, _strict_global_v1(raw))

    if policy in {"global_variable_block", "global_blocks"}:
        budget_mode = _normalize(raw.get("budget_mode", "bucketed"))
        if budget_mode == "global":
            migrated = {
                "policy": GLOBAL_DYNAMIC_BLOCK_V1,
                "full_generation_probability": raw.get(
                    "full_generation_probability", 0.25
                ),
                "block_lengths": raw.get("block_lengths", (3, 5, 8)),
                "visible_ratio_min": raw.get("visible_ratio_min", 0.0),
                "visible_ratio_max": raw.get("visible_ratio_max", 0.30),
            }
            return MaskContract(
                GLOBAL_DYNAMIC_BLOCK_V1,
                _strict_global_v1(migrated),
                migrated_from="global_variable_block:budget_mode=global",
            )
        if budget_mode != "bucketed":
            raise ValueError(f"unsupported legacy global block budget_mode={budget_mode!r}")
        normalized = dict(raw)
        normalized["policy"] = "legacy_bucketed_variable_block_v1"
        normalized["budget_mode"] = "bucketed"
        return MaskContract("legacy_bucketed_variable_block_v1", normalized)

    normalized = dict(raw)
    normalized["policy"] = policy
    return MaskContract(f"legacy:{policy}", normalized)


def attach_mask_contract(config: dict[str, Any]) -> MaskContract:
    model = config.setdefault("model", {})
    mask = config.setdefault("mask", {})
    contract = resolve_mask_contract(
        mask,
        legacy_model_policy=model.get("mask_policy"),
    )
    config["mask"] = dict(contract.policy_config)
    config["mask_contract"] = contract.metadata()
    model.pop("mask_policy", None)
    model["mask_contract_id"] = contract.contract_id
    model["mask_contract_fingerprint"] = contract.fingerprint
    return contract


def validate_resume_mask_contract(
    contract: MaskContract,
    checkpoint: str | Path,
    *,
    transition: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Reject resume across mask semantics, including legacy checkpoints."""

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    state_path = checkpoint_path / "trainer_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(
            f"resume checkpoint has no trainer_state.json: {checkpoint_path}"
        )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    saved = state.get("mask_contract_fingerprint")
    if saved is not None:
        return validate_mask_transition_request(
            saved_fingerprint=str(saved),
            saved_step=int(state.get("global_step", -1)),
            target_fingerprint=contract.fingerprint,
            transition=transition,
            checkpoint=checkpoint_path,
        )

    resolved = _find_resolved_config(checkpoint_path)
    old = json.loads(resolved.read_text(encoding="utf-8"))
    old_model = old.get("model", {})
    old_contract = resolve_mask_contract(
        old.get("mask", {}),
        legacy_model_policy=(
            old_model.get("mask_policy") if isinstance(old_model, Mapping) else None
        ),
    )
    if old_contract.fingerprint != contract.fingerprint:
        raise ValueError(
            "legacy resume mask contract mismatch after explicit migration: "
            f"saved={old_contract.fingerprint} current={contract.fingerprint}"
        )
    return None


def validate_mask_transition_request(
    *,
    saved_fingerprint: str,
    saved_step: int,
    target_fingerprint: str,
    transition: Mapping[str, Any] | None,
    checkpoint: str | Path,
) -> dict[str, Any] | None:
    """Validate one explicitly declared stage-boundary mask transition."""

    saved = str(saved_fingerprint)
    target = str(target_fingerprint)
    if saved == target:
        return None

    request = dict(transition or {})
    path = Path(checkpoint).expanduser().resolve()
    if not bool(request.get("enabled", False)):
        raise ValueError(
            "resume mask contract mismatch: "
            f"saved={saved} current={target} path={path}"
        )

    expected_source = str(request.get("source_fingerprint", ""))
    expected_step = int(request.get("source_step", -1))
    expected_target = str(request.get("target_fingerprint", target))
    reason = str(request.get("reason", "")).strip()
    if not expected_source or expected_source != saved:
        raise ValueError(
            "mask transition source fingerprint mismatch: "
            f"declared={expected_source!r} saved={saved!r} path={path}"
        )
    if expected_step < 0 or expected_step != int(saved_step):
        raise ValueError(
            "mask transition source step mismatch: "
            f"declared={expected_step} saved={saved_step} path={path}"
        )
    if expected_target != target:
        raise ValueError(
            "mask transition target fingerprint mismatch: "
            f"declared={expected_target!r} current={target!r}"
        )
    if not reason:
        raise ValueError("mask transition requires a non-empty reason")
    return {
        "source_fingerprint": saved,
        "source_step": int(saved_step),
        "target_fingerprint": target,
        "reason": reason,
    }


def _find_resolved_config(checkpoint: Path) -> Path:
    for directory in (checkpoint, *checkpoint.parents):
        for candidate in (
            directory / "resolved_config.json",
            directory / "config" / "resolved_config.json",
        ):
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f"cannot validate legacy mask contract above checkpoint: {checkpoint}"
    )


def _strict_global_v1(raw: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "policy",
        "full_generation_probability",
        "block_lengths",
        "visible_ratio_min",
        "visible_ratio_max",
    }
    extra = sorted(set(raw).difference(allowed))
    if extra:
        raise ValueError(
            f"{GLOBAL_DYNAMIC_BLOCK_V1} rejects hidden/unknown fields: {extra}"
        )
    result = global_dynamic_block_v1(
        full_generation_probability=float(
            raw.get("full_generation_probability", 0.25)
        ),
        block_lengths=tuple(int(value) for value in raw.get("block_lengths", (3, 5, 8))),
        visible_ratio_min=float(raw.get("visible_ratio_min", 0.0)),
        visible_ratio_max=float(raw.get("visible_ratio_max", 0.30)),
    )
    probability = float(result["full_generation_probability"])
    minimum = float(result["visible_ratio_min"])
    maximum = float(result["visible_ratio_max"])
    lengths = result["block_lengths"]
    if not 0.0 <= probability <= 1.0:
        raise ValueError("full_generation_probability must be in [0, 1]")
    if not 0.0 <= minimum <= maximum <= 1.0:
        raise ValueError("visible ratios must satisfy 0 <= min <= max <= 1")
    if not lengths or any(int(value) < 3 for value in lengths):
        raise ValueError("block_lengths must contain integers >= 3")
    return result


def _normalize(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


__all__ = [
    "GLOBAL_DYNAMIC_BLOCK_V1",
    "MaskContract",
    "attach_mask_contract",
    "global_dynamic_block_v1",
    "resolve_mask_contract",
    "validate_mask_transition_request",
    "validate_resume_mask_contract",
]
