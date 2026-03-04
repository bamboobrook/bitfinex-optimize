"""
System policy loader for closed-loop behavior controls.

This module centralizes strategy defaults and optional file-based overrides.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


def _default_policy() -> Dict[str, Any]:
    return {
        "policy_version": "2026.03.auto.v1",
        "objective_priority": "execution_revenue_first",
        "acceptance_primary": "follow_error_and_stability",
        "automation": {
            # Hard freshness gate. Predictions older than this are rejected.
            "stale_data_warn_minutes": 60,
            "stale_data_hard_minutes": 120,
            # If a combo has no recent validated sample, allow refresh probe.
            "refresh_probe_lookback_hours": 24,
            "refresh_probe_min_validations": 1,
            "refresh_probe_trigger_cycles": 6,
            "refresh_probe_max_per_cycle": 4,
        },
        "retrain_trigger": {
            "score_threshold": 1.0,
            "follow_mae_ratio_threshold": 0.65,
            "direction_match_threshold": 0.40,
            "p120_step_p95_threshold": 0.05,
            "global_exec_low": 0.40,
            "global_exec_high": 0.60,
        },
        "period_policy": {
            "short_periods": [2, 3, 4, 5, 6, 7],
            "medium_periods": [10, 14, 15, 20, 30],
            "long_periods": [60, 90, 120],
            # Per-period hard cap for one-round price step changes.
            # 120d is intentionally strict to keep long-cycle stability.
            "step_caps_pct": {
                "120": 0.05,
            },
            # Keep current allocation behavior unless explicitly changed.
            "p120_order_share": "current_ratio",
            "p120_stability_mode": "strong",
        },
    }


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_default_policy_path() -> str:
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    return os.path.join(base_dir, "data", "system_policy.json")


def get_tracked_policy_path() -> str:
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    return os.path.join(base_dir, "config", "system_policy.json")


def _candidate_policy_paths(policy_path: Optional[str] = None) -> List[str]:
    candidates: List[str] = []
    if policy_path:
        candidates.append(policy_path)

    env_path = os.getenv("SYSTEM_POLICY_PATH")
    if env_path:
        candidates.append(env_path)

    # Prefer tracked config path over ignored /data path.
    candidates.append(get_tracked_policy_path())
    candidates.append(get_default_policy_path())
    return candidates


def load_system_policy(policy_path: Optional[str] = None) -> Dict[str, Any]:
    policy = _default_policy()

    for path in _candidate_policy_paths(policy_path):
        if not os.path.exists(path):
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
        except Exception:
            # Keep running with safe defaults if policy file is malformed.
            continue

        if not isinstance(loaded, dict):
            continue

        policy = _deep_merge(policy, loaded)
        policy["_policy_source_path"] = path
        break

    return policy


def get_step_cap_pct(policy: Dict[str, Any], period: int) -> Optional[float]:
    caps = (
        policy.get("period_policy", {})
        .get("step_caps_pct", {})
    )
    if not isinstance(caps, dict):
        return None

    value = caps.get(str(period))
    if value is None:
        return None

    try:
        cap = float(value)
    except (TypeError, ValueError):
        return None

    if cap <= 0:
        return None
    return cap


def get_policy_version(policy: Dict[str, Any]) -> str:
    value = policy.get("policy_version")
    if value is None:
        return "unknown"
    return str(value)
