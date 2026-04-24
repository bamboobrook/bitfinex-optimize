import sys
import json
import sqlite3
import tempfile
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_engine.c3_combo_optimizer import RateCandidate, choose_combo_beam
from test_predictor_rank6 import EnsemblePredictor, _make_prediction


def _make_shadow_predictor(tmp: str, beam_width):
    predictor = EnsemblePredictor.__new__(EnsemblePredictor)
    predictor.policy = {"combo_optimizer": {"combo_mode": "shadow", "beam_width": beam_width}}
    predictor.policy_version = "test-policy"
    predictor.db_path = str(Path(tmp) / "lending_history.db")
    predictor.refresh_probe_state_path = str(Path(tmp) / "refresh_probe_state.json")
    predictor.order_manager = None
    predictor._funding_book_cache = {}
    predictor._stale_issues = []
    predictor._persist_prediction_history = lambda ranked_predictions: None
    predictor._calc_market_liquidity = lambda preds: {
        "fUSD": {"level": "medium", "score": 60.0, "fast_score": 0.62, "fillability_signal": 0.60, "volume_ratio_24h": 0.84},
        "fUST": {"level": "medium", "score": 52.0, "fast_score": 0.50, "fillability_signal": 0.48, "volume_ratio_24h": 0.62},
    }
    predictor.get_latest_predictions = lambda: [
        _make_prediction("fUSD", 120, 12.4, exec_prob=0.62),
        _make_prediction("fUSD", 30, 9.2, exec_prob=0.58),
        _make_prediction("fUST", 30, 11.4, exec_prob=0.64),
        _make_prediction("fUSD", 90, 11.8, exec_prob=0.60),
        _make_prediction("fUSD", 14, 8.8, exec_prob=0.56),
        _make_prediction("fUSD", 2, 5.2, exec_prob=0.67),
    ]
    predictor._apply_path_ranking = lambda valid_preds, market_liquidity, fusd_2d_pred: sorted(
        valid_preds,
        key=lambda pred: pred["predicted_rate"],
        reverse=True,
    )
    return predictor


def test_choose_combo_beam_rejects_duplicate_pairs_and_prefers_fusd_unless_fust_is_clearly_better():
    candidates = [
        RateCandidate("fUSD", 120, 12.4, "premium", 0.4),
        RateCandidate("fUSD", 120, 12.6, "stretch_premium", 0.6),
        RateCandidate("fUSD", 30, 9.2, "balanced_mid", 0.1),
        RateCandidate("fUST", 30, 9.8, "premium", 0.6),
        RateCandidate("fUST", 30, 11.4, "stretch_premium", 1.8),
        RateCandidate("fUSD", 90, 11.8, "premium", 0.3),
        RateCandidate("fUSD", 14, 8.8, "balanced_mid", 0.2),
    ]
    scored = {
        ("fUSD", 120, 12.4): {"candidate_path_ev": 8.8, "fill_quality": 0.58},
        ("fUSD", 120, 12.6): {"candidate_path_ev": 8.9, "fill_quality": 0.40},
        ("fUSD", 30, 9.2): {"candidate_path_ev": 7.0, "fill_quality": 0.72},
        ("fUST", 30, 9.8): {"candidate_path_ev": 7.1, "fill_quality": 0.55},
        ("fUST", 30, 11.4): {"candidate_path_ev": 7.4, "fill_quality": 0.28},
        ("fUSD", 90, 11.8): {"candidate_path_ev": 8.1, "fill_quality": 0.52},
        ("fUSD", 14, 8.8): {"candidate_path_ev": 6.8, "fill_quality": 0.66},
    }

    combo = choose_combo_beam(candidates, scored, beam_width=12, policy={
        "combo_optimizer": {"hard_sort_revenue_step": 0.50, "hard_sort_fill_step": 0.02}
    })

    assert len({(item.currency, item.period) for item in combo}) == 5
    assert combo[0].currency == "fUSD"
    assert ("fUST", 30, 11.4) not in [(c.currency, c.period, c.rate) for c in combo]


def test_choose_combo_beam_prefers_longer_tenor_when_revenue_and_fill_share_same_tier():
    candidates = [
        RateCandidate("fUSD", 120, 12.4, "premium", 0.4),
        RateCandidate("fUSD", 60, 10.9, "premium", 0.3),
        RateCandidate("fUST", 14, 8.6, "balanced_mid", 0.2),
        RateCandidate("fUSD", 14, 8.1, "balanced_mid", 0.1),
        RateCandidate("fUSD", 30, 7.5, "balanced_mid", 0.1),
        RateCandidate("fUSD", 90, 7.5, "balanced_mid", 0.1),
    ]
    scored = {
        ("fUSD", 120, 12.4): {"candidate_path_ev": 12.8, "fill_quality": 0.58},
        ("fUSD", 60, 10.9): {"candidate_path_ev": 11.7, "fill_quality": 0.57},
        ("fUST", 14, 8.6): {"candidate_path_ev": 10.9, "fill_quality": 0.61},
        ("fUSD", 14, 8.1): {"candidate_path_ev": 10.3, "fill_quality": 0.63},
        ("fUSD", 30, 7.5): {"candidate_path_ev": 6.02, "fill_quality": 0.60},
        ("fUSD", 90, 7.5): {"candidate_path_ev": 6.01, "fill_quality": 0.60},
    }

    combo = choose_combo_beam(candidates, scored, beam_width=12)

    combo_keys = [(item.currency, item.period, item.rate) for item in combo]
    assert ("fUSD", 90, 7.5) in combo_keys
    assert ("fUSD", 30, 7.5) not in combo_keys


def test_choose_combo_beam_returns_empty_for_empty_candidates():
    assert choose_combo_beam([], {}, 4) == []


def test_choose_combo_beam_rejects_non_positive_beam_width():
    with pytest.raises(ValueError):
        choose_combo_beam([], {}, 0)


def test_generate_recommendations_in_shadow_mode_keeps_result_schema_without_shadow_fields():
    with tempfile.TemporaryDirectory() as tmp:
        output_path = Path(tmp) / "optimal_combination.json"

        predictor = EnsemblePredictor.__new__(EnsemblePredictor)
        predictor.policy = {"combo_optimizer": {"combo_mode": "shadow", "beam_width": 12}}
        predictor.policy_version = "test-policy"
        predictor.db_path = str(Path(tmp) / "lending_history.db")
        predictor.refresh_probe_state_path = str(Path(tmp) / "refresh_probe_state.json")
        predictor.order_manager = None
        predictor._funding_book_cache = {}
        predictor._stale_issues = []
        predictor._persist_prediction_history = lambda ranked_predictions: None
        predictor._calc_market_liquidity = lambda preds: {
            "fUSD": {"level": "medium", "score": 60.0, "fast_score": 0.62, "fillability_signal": 0.60, "volume_ratio_24h": 0.84},
            "fUST": {"level": "medium", "score": 52.0, "fast_score": 0.50, "fillability_signal": 0.48, "volume_ratio_24h": 0.62},
        }

        def _decorate(pred, path_value, fill_quality, candidate_band="balanced_mid"):
            pred["path_value_score"] = path_value
            pred["final_rank_score"] = path_value
            pred["weighted_score"] = path_value
            pred["stage1_fill_probability"] = fill_quality
            pred["fast_liquidity_score"] = fill_quality
            pred["candidate_band"] = candidate_band
            return pred

        ranked_preds = [
            _decorate(_make_prediction("fUSD", 120, 12.4, exec_prob=0.62), 8.8, 0.58, "premium"),
            _decorate(_make_prediction("fUSD", 30, 9.2, exec_prob=0.58), 7.0, 0.72),
            _decorate(_make_prediction("fUST", 30, 11.4, exec_prob=0.64), 7.4, 0.28, "stretch_premium"),
            _decorate(_make_prediction("fUSD", 90, 11.8, exec_prob=0.60), 8.1, 0.52, "premium"),
            _decorate(_make_prediction("fUSD", 14, 8.8, exec_prob=0.56), 6.8, 0.66),
            _decorate(_make_prediction("fUSD", 2, 5.2, exec_prob=0.67), 4.9, 0.76),
        ]

        predictor.get_latest_predictions = lambda: ranked_preds
        predictor._apply_path_ranking = lambda valid_preds, market_liquidity, fusd_2d_pred: sorted(
            valid_preds,
            key=lambda pred: pred["predicted_rate"],
            reverse=True,
        )

        predictor.generate_recommendations(str(output_path))

        result = json.loads(output_path.read_text())

        assert result["recommendations"][0]["currency"] == "fUSD"
        assert result["recommendations"][0]["period"] == 120
        assert "shadow_combo" not in result
        assert "shadow_combo_metrics" not in result


def test_generate_recommendations_ignores_invalid_shadow_beam_width_and_keeps_main_output():
    with tempfile.TemporaryDirectory() as tmp:
        output_path = Path(tmp) / "optimal_combination.json"
        predictor = _make_shadow_predictor(tmp, -1)

        predictor.generate_recommendations(str(output_path))

        result = json.loads(output_path.read_text())

        assert result["status"] == "success"
        assert len(result["recommendations"]) == 6
        assert result["recommendations"][0]["currency"] == "fUSD"
        assert result["recommendations"][0]["period"] == 120
        assert "shadow_combo" not in result


def test_generate_recommendations_ignores_zero_shadow_beam_width_and_keeps_main_output():
    with tempfile.TemporaryDirectory() as tmp:
        output_path = Path(tmp) / "optimal_combination.json"
        predictor = _make_shadow_predictor(tmp, 0)

        predictor.generate_recommendations(str(output_path))

        result = json.loads(output_path.read_text())

        assert result["status"] == "success"
        assert len(result["recommendations"]) == 6
        assert result["recommendations"][0]["currency"] == "fUSD"
        assert result["recommendations"][0]["period"] == 120
        assert "shadow_combo" not in result


def test_generate_recommendations_ignores_shadow_builder_failure_and_keeps_main_output():
    with tempfile.TemporaryDirectory() as tmp:
        output_path = Path(tmp) / "optimal_combination.json"
        predictor = _make_shadow_predictor(tmp, 12)
        predictor._build_shadow_combo = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("shadow boom"))

        predictor.generate_recommendations(str(output_path))

        result = json.loads(output_path.read_text())

        assert result["status"] == "success"
        assert len(result["recommendations"]) == 6
        assert result["recommendations"][0]["currency"] == "fUSD"
        assert result["recommendations"][0]["period"] == 120
        assert "shadow_combo" not in result


def test_build_shadow_combo_uses_anchor_candidates_with_real_bands():
    with tempfile.TemporaryDirectory() as tmp:
        predictor = _make_shadow_predictor(tmp, 12)

        with sqlite3.connect(predictor.db_path) as conn:
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
                """
                INSERT INTO funding_rates(currency, period, close_annual, datetime)
                VALUES (?, ?, ?, ?)
                """,
                [
                    ("fUSD", 120, 10.0, "2026-04-13 00:00:00"),
                    ("fUSD", 120, 12.0, "2026-04-13 01:00:00"),
                    ("fUSD", 120, 14.0, "2026-04-13 02:00:00"),
                    ("fUST", 30, 8.0, "2026-04-13 00:00:00"),
                    ("fUST", 30, 9.0, "2026-04-13 01:00:00"),
                    ("fUST", 30, 10.0, "2026-04-13 02:00:00"),
                    ("fUSD", 90, 9.5, "2026-04-13 00:00:00"),
                    ("fUSD", 90, 10.5, "2026-04-13 01:00:00"),
                    ("fUSD", 90, 11.5, "2026-04-13 02:00:00"),
                    ("fUSD", 14, 6.0, "2026-04-13 00:00:00"),
                    ("fUSD", 14, 7.0, "2026-04-13 01:00:00"),
                    ("fUSD", 14, 8.0, "2026-04-13 02:00:00"),
                    ("fUST", 14, 5.5, "2026-04-13 00:00:00"),
                    ("fUST", 14, 6.2, "2026-04-13 01:00:00"),
                    ("fUST", 14, 6.9, "2026-04-13 02:00:00"),
                ],
            )
            conn.execute(
                """
                CREATE TABLE virtual_orders (
                    currency TEXT,
                    period INTEGER,
                    status TEXT,
                    execution_rate REAL,
                    executed_at TEXT
                )
                """
            )
            conn.executemany(
                """
                INSERT INTO virtual_orders(currency, period, status, execution_rate, executed_at)
                VALUES (?, ?, 'EXECUTED', ?, ?)
                """,
                [
                    ("fUSD", 120, 11.5, "2026-04-13 03:00:00"),
                    ("fUST", 30, 9.6, "2026-04-13 03:00:00"),
                    ("fUSD", 90, 11.0, "2026-04-13 03:00:00"),
                    ("fUSD", 14, 7.5, "2026-04-13 03:00:00"),
                    ("fUST", 14, 6.7, "2026-04-13 03:00:00"),
                ],
            )
            conn.commit()

        ranked_predictions = [
            _make_prediction("fUSD", 120, 12.4, exec_prob=0.62),
            _make_prediction("fUST", 30, 9.4, exec_prob=0.64),
            _make_prediction("fUSD", 90, 11.1, exec_prob=0.60),
            _make_prediction("fUSD", 14, 7.2, exec_prob=0.56),
            _make_prediction("fUST", 14, 6.4, exec_prob=0.58),
            _make_prediction("fUSD", 2, 5.2, exec_prob=0.67),
        ]
        for pred in ranked_predictions:
            pred["path_value_score"] = pred["predicted_rate"]
            pred["final_rank_score"] = pred["predicted_rate"]
            pred["weighted_score"] = pred["predicted_rate"]
            pred["stage1_fill_probability"] = pred["execution_probability"]
            pred["fast_liquidity_score"] = pred["execution_probability"]

        combo, metrics = predictor._build_shadow_combo(ranked_predictions, "cycle-anchor", 12)

        assert len(combo) == 5
        assert metrics["beam_width"] == 12
        assert any(item["candidate_band"] != "balanced_mid" for item in combo)
        assert any(item["candidate_band"] in {"premium", "stretch_premium"} for item in combo)
        assert any("-premium" in item["candidate_id"] for item in combo)


def test_build_shadow_combo_uses_path_value_as_primary_revenue_metric(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        predictor = _make_shadow_predictor(tmp, 12)
        predictor._load_market_anchor_rows = lambda currency, period: []

        ranked_predictions = [
            _make_prediction("fUSD", 120, 12.4, exec_prob=0.62),
            _make_prediction("fUSD", 90, 11.8, exec_prob=0.60),
            _make_prediction("fUSD", 30, 9.2, exec_prob=0.58),
            _make_prediction("fUST", 30, 11.4, exec_prob=0.64),
            _make_prediction("fUSD", 14, 8.8, exec_prob=0.56),
            _make_prediction("fUSD", 2, 5.2, exec_prob=0.67),
        ]
        for idx, pred in enumerate(ranked_predictions, start=1):
            pred["path_value_score"] = 9.0 - idx * 0.2
            pred["final_rank_score"] = 2.0 + idx * 0.1
            pred["weighted_score"] = pred["final_rank_score"]
            pred["stage1_fill_probability"] = 0.55 + idx * 0.01
            pred["fast_liquidity_score"] = 0.60
            pred["candidate_band"] = "balanced_mid"

        def _fake_score_shadow_candidate(base_pred, candidate, market_liquidity, rank6_rate):
            return {
                **base_pred,
                "currency": candidate.currency,
                "period": candidate.period,
                "predicted_rate": candidate.rate,
                "candidate_band": candidate.band,
                "candidate_id": f"{candidate.currency}-{candidate.period}-{candidate.band}",
                "path_value_score": base_pred["path_value_score"],
                "final_rank_score": base_pred["final_rank_score"],
                "weighted_score": base_pred["weighted_score"],
                "stage1_fill_probability": base_pred["stage1_fill_probability"],
                "fast_liquidity_score": base_pred["fast_liquidity_score"],
            }

        predictor._score_shadow_candidate = _fake_score_shadow_candidate
        captured = {}

        def _capture_choose_combo_beam(candidates, scored, beam_width, policy=None):
            captured["scored"] = scored
            return candidates[:5]

        monkeypatch.setattr("ml_engine.predictor.choose_combo_beam", _capture_choose_combo_beam)

        predictor._build_shadow_combo(ranked_predictions, "cycle-path-primary", 12)

        key = ("fUSD", 120, 12.4)
        assert captured["scored"][key]["candidate_path_ev"] == ranked_predictions[0]["path_value_score"]
        assert captured["scored"][key]["candidate_path_ev"] != ranked_predictions[0]["final_rank_score"]


def test_choose_combo_beam_prefers_anchor_backed_candidates_within_same_priority_tier():
    """Beam should prefer anchor-backed candidates only after core C3 priority ties."""
    candidates = [
        RateCandidate("fUSD", 120, 12.4, "premium", 0.4),
        RateCandidate("fUSD", 30, 9.2, "balanced_mid", 0.1),
        RateCandidate("fUSD", 90, 11.8, "premium", 0.3),
        RateCandidate("fUST", 30, 9.8, "premium", 0.6),
        RateCandidate("fUST", 14, 7.5, "balanced_mid", 0.1),
        RateCandidate("fUST", 14, 7.6, "balanced_mid", 0.2),
    ]
    scored = {
        ("fUSD", 120, 12.4): {"candidate_path_ev": 8.8, "fill_quality": 0.58, "anchor_backed": 1},
        ("fUSD", 30, 9.2): {"candidate_path_ev": 7.0, "fill_quality": 0.72, "anchor_backed": 1},
        ("fUSD", 90, 11.8): {"candidate_path_ev": 8.1, "fill_quality": 0.52, "anchor_backed": 1},
        ("fUST", 30, 9.8): {"candidate_path_ev": 7.1, "fill_quality": 0.55, "anchor_backed": 0},
        ("fUST", 14, 7.5): {"candidate_path_ev": 6.8, "fill_quality": 0.66, "anchor_backed": 0},
        ("fUST", 14, 7.6): {"candidate_path_ev": 6.8, "fill_quality": 0.66, "anchor_backed": 1},
    }

    combo = choose_combo_beam(candidates, scored, beam_width=12, policy={
        "combo_optimizer": {"hard_sort_revenue_step": 0.50, "hard_sort_fill_step": 0.02}
    })

    assert len(combo) == 5
    combo_keys = [(item.currency, item.period, item.rate) for item in combo]
    assert ("fUSD", 120, 12.4) in combo_keys
    assert ("fUST", 14, 7.6) in combo_keys
    assert ("fUST", 14, 7.5) not in combo_keys


def test_choose_combo_beam_keeps_revenue_bucket_ahead_of_anchor_backed_status():
    candidates = [
        RateCandidate("fUSD", 120, 12.4, "premium", 0.4),
        RateCandidate("fUSD", 90, 11.8, "premium", 0.3),
        RateCandidate("fUSD", 60, 10.5, "premium", 0.2),
        RateCandidate("fUSD", 30, 9.2, "balanced_mid", 0.1),
        RateCandidate("fUSD", 14, 8.8, "balanced_mid", 0.2),
        RateCandidate("fUST", 14, 10.2, "premium", 0.6),
    ]
    scored = {
        ("fUSD", 120, 12.4): {"candidate_path_ev": 9.0, "fill_quality": 0.62, "anchor_backed": 1},
        ("fUSD", 90, 11.8): {"candidate_path_ev": 8.8, "fill_quality": 0.61, "anchor_backed": 1},
        ("fUSD", 60, 10.5): {"candidate_path_ev": 8.6, "fill_quality": 0.60, "anchor_backed": 1},
        ("fUSD", 30, 9.2): {"candidate_path_ev": 8.4, "fill_quality": 0.59, "anchor_backed": 1},
        ("fUSD", 14, 8.8): {"candidate_path_ev": 6.8, "fill_quality": 0.80, "anchor_backed": 1},
        ("fUST", 14, 10.2): {"candidate_path_ev": 8.2, "fill_quality": 0.58, "anchor_backed": 0},
    }

    combo = choose_combo_beam(candidates, scored, beam_width=12, policy={
        "combo_optimizer": {"hard_sort_revenue_step": 0.50, "hard_sort_fill_step": 0.02}
    })

    combo_keys = [(item.currency, item.period) for item in combo]
    assert ("fUST", 14) in combo_keys
    assert ("fUSD", 14) not in combo_keys
