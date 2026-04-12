from dataclasses import dataclass

import numpy as np


@dataclass
class AnchorSnapshot:
    low: float
    mid: float
    high: float
    window_profile: str
    source: str


@dataclass
class RateCandidate:
    currency: str
    period: int
    rate: float
    band: str
    distance_from_mid: float


def _window_profile_for_period(period: int) -> str:
    if period <= 7:
        return "2h-6h-12h"
    if period <= 30:
        return "6h-12h-24h"
    return "12h-24h-48h"


def build_anchor_snapshot(currency: str, period: int, market_rows: list[dict]) -> AnchorSnapshot:
    executed = [float(r["executed"]) for r in market_rows if r.get("executed") is not None]
    closes = [float(r["close"]) for r in market_rows if r.get("close") is not None]
    base = executed if len(executed) >= max(3, len(closes) // 2) else closes
    if not base:
        raise ValueError("No valid market rows for anchor snapshot")
    source = "executed" if base is executed else "close"
    low = float(np.percentile(base, 25))
    mid = float(np.percentile(base, 50))
    high = float(np.percentile(base, 75))
    return AnchorSnapshot(
        low=low,
        mid=mid,
        high=high,
        window_profile=_window_profile_for_period(period),
        source=source,
    )


def generate_rate_candidates(
    currency: str,
    period: int,
    anchor: AnchorSnapshot,
    hard_cap_pct: float,
    max_candidates: int,
) -> list[RateCandidate]:
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")

    raw = [
        ("safe_fill", anchor.low),
        ("balanced_low", (anchor.low + anchor.mid) / 2.0),
        ("balanced_mid", anchor.mid),
        ("premium", (anchor.mid + anchor.high) / 2.0),
        ("stretch_premium", anchor.high * (1.0 + hard_cap_pct * 2.0)),
    ]
    cap = anchor.high * (1.0 + hard_cap_pct)
    return [
        RateCandidate(currency, period, min(rate, cap), band, min(rate, cap) - anchor.mid)
        for band, rate in raw[:max_candidates]
    ]
