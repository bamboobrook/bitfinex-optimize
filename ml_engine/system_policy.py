"""
System policy loader for closed-loop behavior controls.

This module centralizes strategy defaults and optional file-based overrides.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


def _default_policy() -> Dict[str, Any]:
    return {
        "objective_priority": "execution_revenue_first",
        "acceptance_primary": "follow_error_and_stability",
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


def load_system_policy(policy_path: Optional[str] = None) -> Dict[str, Any]:
    policy = _default_policy()

    path = policy_path or os.getenv("SYSTEM_POLICY_PATH") or get_default_policy_path()
    if not os.path.exists(path):
        return policy

    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except Exception:
        # Keep running with safe defaults if policy file is malformed.
        return policy

    if not isinstance(loaded, dict):
        return policy

    return _deep_merge(policy, loaded)


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
