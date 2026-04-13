from dataclasses import dataclass
import math

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


def _candidate_field(candidate, field: str):
    if isinstance(candidate, dict):
        return candidate[field]
    return getattr(candidate, field)


def _priority_bucket(value: float, step: float) -> int:
    value = float(value or 0.0)
    step = float(step or 0.0)
    if step <= 0:
        return int(round(value * 1000))
    return int(math.floor((value + 1e-12) / step))


def choose_combo_beam(candidates, scored, beam_width: int):
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")

    revenue_step = 0.50
    fill_step = 0.02
    beams = [([], ())]

    for _rank in range(5):
        next_beams = []
        for partial, hard_key in beams:
            used_pairs = {
                (_candidate_field(item, "currency"), int(_candidate_field(item, "period")))
                for item in partial
            }
            for candidate in candidates:
                pair = (_candidate_field(candidate, "currency"), int(_candidate_field(candidate, "period")))
                if pair in used_pairs:
                    continue

                metrics = scored[(
                    _candidate_field(candidate, "currency"),
                    int(_candidate_field(candidate, "period")),
                    float(_candidate_field(candidate, "rate")),
                )]
                candidate_key = (
                    _priority_bucket(float(metrics.get("candidate_path_ev", 0.0) or 0.0), revenue_step),
                    _priority_bucket(float(metrics.get("fill_quality", 0.0) or 0.0), fill_step),
                    float(metrics.get("tenor_value", _candidate_field(candidate, "period")) or 0.0),
                    float(metrics.get(
                        "currency_priority",
                        1.0 if _candidate_field(candidate, "currency") == "fUSD" else 0.0,
                    ) or 0.0),
                    float(metrics.get("candidate_path_ev", 0.0) or 0.0),
                    float(metrics.get("fill_quality", 0.0) or 0.0),
                )
                next_beams.append((
                    partial + [candidate],
                    hard_key + candidate_key,
                ))

        if not next_beams:
            break

        next_beams.sort(
            key=lambda item: item[1],
            reverse=True,
        )
        beams = next_beams[:beam_width]

    return beams[0][0] if beams else []
