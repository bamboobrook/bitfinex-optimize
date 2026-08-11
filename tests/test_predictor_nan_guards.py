import math
import sqlite3

import numpy as np
import pandas as pd
import pytest

from test_predictor_rank6 import EnsemblePredictor


def test_sanitize_rate_falls_back_for_nan_long_anchor():
    predictor = EnsemblePredictor.__new__(EnsemblePredictor)

    assert predictor._finite_rate(float("nan"), 8.5) == 8.5
    assert predictor._finite_rate(None, 8.5) == 8.5
    assert predictor._finite_rate(9.25, 8.5) == 9.25


def test_sanitize_rate_uses_first_finite_fallback():
    predictor = EnsemblePredictor.__new__(EnsemblePredictor)

    value = predictor._finite_rate(float("nan"), float("nan"), 7.25, 6.0)

    assert math.isclose(value, 7.25)


def _make_rate_predictor(tmp_path, rows):
    db_path = tmp_path / "rates.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE funding_rates (
                currency TEXT,
                period INTEGER,
                close_annual REAL,
                datetime TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO funding_rates VALUES ('fUST', 30, ?, ?)",
            rows,
        )

    predictor = EnsemblePredictor.__new__(EnsemblePredictor)
    predictor.db_path = str(db_path)
    predictor.policy = {}
    return predictor


def test_latest_rate_replaces_isolated_extreme_tick_with_recent_median(tmp_path):
    predictor = _make_rate_predictor(
        tmp_path,
        [
            (0.027, "2026-07-28 07:03:00"),
            (8.0, "2026-07-28 06:50:00"),
            (8.0, "2026-07-28 06:30:00"),
            (8.0, "2026-07-28 06:10:00"),
        ],
    )

    rate, timestamp = predictor.get_latest_rate_from_db("fUST", 30)

    assert math.isclose(rate, 8.0)
    assert timestamp == "2026-07-28 06:50:00"


def test_latest_rate_keeps_normal_market_move(tmp_path):
    predictor = _make_rate_predictor(
        tmp_path,
        [
            (7.2, "2026-07-28 07:03:00"),
            (8.0, "2026-07-28 06:50:00"),
            (7.8, "2026-07-28 06:30:00"),
            (8.1, "2026-07-28 06:10:00"),
        ],
    )

    rate, timestamp = predictor.get_latest_rate_from_db("fUST", 30)

    assert math.isclose(rate, 7.2)
    assert timestamp == "2026-07-28 07:03:00"


def test_predict_with_ensemble_rejects_missing_positive_weight_component():
    class _CatModel:
        @staticmethod
        def predict(X, thread_count=None):
            return np.full(len(X), 8.0)

    predictor = EnsemblePredictor.__new__(EnsemblePredictor)
    predictor.infer_threads = 1
    predictor.models = {
        "fUSD": {
            "model_balanced": {
                "cat": _CatModel(),
            }
        }
    }
    predictor.meta_info = {
        "fUSD": {
            "model_balanced": {
                "weights": {"xgb": 0.7, "cat": 0.3},
                "task_type": "regression",
            }
        }
    }

    with pytest.raises(ValueError, match="missing.*xgb"):
        predictor.predict_with_ensemble(
            pd.DataFrame({"signal": [1.0]}),
            "fUSD",
            "model_balanced",
        )


def test_low_liquidity_floor_never_clips_candidate_upward():
    guarded, suppressed = EnsemblePredictor._guard_low_liquidity_min_bound(
        candidate_rate=6.63,
        min_bound=8.23,
        current_rate=4.38,
        execution_rate=0.10,
    )

    assert guarded <= 6.63
    assert suppressed is True


def test_healthy_liquidity_keeps_existing_floor():
    guarded, suppressed = EnsemblePredictor._guard_low_liquidity_min_bound(
        candidate_rate=6.63,
        min_bound=8.23,
        current_rate=4.38,
        execution_rate=0.50,
    )

    assert guarded == pytest.approx(8.23)
    assert suppressed is False


def test_funding_book_signal_filters_by_period_and_executable_rate():
    predictor = EnsemblePredictor.__new__(EnsemblePredictor)
    predictor._funding_book_cache = {}
    predictor._fetch_bitfinex_public_json = lambda *args, **kwargs: [
        [0.00020, 90, 1, -50_000.0],   # 7.30%, executable
        [0.00010, 90, 1, -100_000.0],  # 3.65%, below target
        [0.00050, 120, 1, -500_000.0], # wrong period
        [0.00030, 90, 1, 900_000.0],   # ask, not borrower bid
    ]

    signal = predictor._get_realtime_non2d_liquidity_signal(
        "fUSD", period=90, target_rate=6.0
    )

    assert signal["available"] is True
    assert signal["period_depth"] == pytest.approx(150_000.0)
    assert signal["executable_depth"] == pytest.approx(50_000.0)
    assert signal["best_bid_rate"] == pytest.approx(7.3)
    assert signal["fillability_signal"] > 0.0

    blocked = predictor._get_realtime_non2d_liquidity_signal(
        "fUSD", period=90, target_rate=8.0
    )
    assert blocked["executable_depth"] == 0.0
    assert blocked["fillability_signal"] == 0.0
