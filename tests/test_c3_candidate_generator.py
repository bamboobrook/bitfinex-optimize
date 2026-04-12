import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_engine.c3_combo_optimizer import build_anchor_snapshot, generate_rate_candidates


def test_generate_rate_candidates_uses_dynamic_window_and_caps_distance():
    market_rows = [
        {"close": 8.9, "executed": 8.8},
        {"close": 9.1, "executed": 9.0},
        {"close": 9.3, "executed": 9.2},
        {"close": 9.0, "executed": 8.9},
        {"close": 9.2, "executed": 9.1},
    ]

    anchor = build_anchor_snapshot(currency="fUSD", period=30, market_rows=market_rows)
    candidates = generate_rate_candidates(
        currency="fUSD",
        period=30,
        anchor=anchor,
        hard_cap_pct=0.02,
        max_candidates=5,
    )

    assert anchor.window_profile == "6h-12h-24h"
    assert len(candidates) == 5
    capped = next(c for c in candidates if c.band == "stretch_premium")
    cap = anchor.high * 1.02
    assert capped.rate == pytest.approx(cap)
    assert capped.distance_from_mid == pytest.approx(capped.rate - anchor.mid)
    assert max(c.rate for c in candidates) <= cap
    assert {c.band for c in candidates} >= {"safe_fill", "balanced_mid", "premium"}
    assert candidates[2].distance_from_mid == pytest.approx(0.0)


def test_build_anchor_snapshot_falls_back_to_close_when_executed_insufficient():
    market_rows = [
        {"close": 8.9, "executed": None},
        {"close": 9.1, "executed": 9.0},
        {"close": 9.3, "executed": None},
        {"close": 9.0, "executed": None},
        {"close": 9.2, "executed": None},
    ]

    anchor = build_anchor_snapshot(currency="fUSD", period=30, market_rows=market_rows)

    assert anchor.source == "close"
    assert anchor.low == pytest.approx(9.0)
    assert anchor.mid == pytest.approx(9.1)
    assert anchor.high == pytest.approx(9.2)


@pytest.mark.parametrize(
    "market_rows",
    [
        [],
        [{"close": None, "executed": None}],
    ],
)
def test_build_anchor_snapshot_raises_clear_error_when_no_valid_market_rows(market_rows):
    with pytest.raises(ValueError, match="No valid market rows for anchor snapshot"):
        build_anchor_snapshot(currency="fUSD", period=30, market_rows=market_rows)


@pytest.mark.parametrize("max_candidates", [0, -1, -3])
def test_generate_rate_candidates_raises_clear_error_when_max_candidates_is_non_positive(max_candidates):
    market_rows = [
        {"close": 8.9, "executed": 8.8},
        {"close": 9.1, "executed": 9.0},
        {"close": 9.3, "executed": 9.2},
    ]
    anchor = build_anchor_snapshot(currency="fUSD", period=30, market_rows=market_rows)

    with pytest.raises(ValueError, match="max_candidates must be positive"):
        generate_rate_candidates(
            currency="fUSD",
            period=30,
            anchor=anchor,
            hard_cap_pct=0.20,
            max_candidates=max_candidates,
        )
