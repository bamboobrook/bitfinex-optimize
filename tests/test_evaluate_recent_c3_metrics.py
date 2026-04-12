import json
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_recent_optimization import fetch_path_metrics
from test_predictor_rank6 import EnsemblePredictor, _make_prediction


def test_fetch_path_metrics_reports_label_coverage_and_terminal_matrix(tmp_path):
    db_path = tmp_path / "metrics.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE virtual_orders (
                order_timestamp TEXT,
                validated_at TEXT,
                status TEXT,
                path_value_score REAL,
                stage1_fill_probability REAL,
                expected_terminal_mode TEXT,
                realized_terminal_mode TEXT,
                realized_terminal_value REAL
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO virtual_orders VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "2026-04-01 00:00:00",
                    "2026-04-01 06:00:00",
                    "EXECUTED",
                    8.0,
                    0.7,
                    "stage1_fixed",
                    "FIXED",
                    10.8,
                ),
                (
                    "2026-04-01 01:00:00",
                    "2026-04-01 13:00:00",
                    "FAILED",
                    7.1,
                    0.3,
                    "stage2_frr",
                    "FRR_PROXY",
                    9.2,
                ),
            ],
        )
        start = datetime(2026, 4, 1, 0, 0, 0)
        end = start + timedelta(days=1)
        metrics = fetch_path_metrics(conn, start, end)
    finally:
        conn.close()

    assert "path_label_coverage" in metrics
    assert "terminal_mode_matrix" in metrics
    assert metrics["path_label_coverage"] > 0.0
    assert metrics["avg_realized_terminal_value"] == 10.0
    assert metrics["terminal_mode_matrix"]["stage1_fixed->FIXED"] == 1
    assert metrics["terminal_mode_matrix"]["stage2_frr->FRR_PROXY"] == 1


def test_fetch_path_metrics_handles_legacy_schema_without_path_columns(tmp_path):
    db_path = tmp_path / "legacy_metrics.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE virtual_orders (
                order_timestamp TEXT,
                status TEXT,
                terminal_mode TEXT
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO virtual_orders VALUES (?, ?, ?)
            """,
            [
                ("2026-04-01 00:00:00", "EXECUTED", "FRR_PROXY"),
                ("2026-04-01 01:00:00", "FAILED", "RANK6_PROXY"),
            ],
        )
        start = datetime(2026, 4, 1, 0, 0, 0)
        end = start + timedelta(days=1)
        metrics = fetch_path_metrics(conn, start, end)
    finally:
        conn.close()

    assert metrics == {
        "avg_path_value_score": 0.0,
        "avg_stage1_fill_probability": 0.0,
        "frr_terminal_ratio": 0.5,
        "rank6_terminal_ratio": 0.5,
        "path_label_coverage": 1.0,
        "avg_realized_terminal_value": None,
        "terminal_mode_matrix": {},
    }


def test_generate_recommendations_uses_c3_combo_as_live_output_when_enabled():
    with tempfile.TemporaryDirectory() as tmp:
        output_path = Path(tmp) / "optimal_combination.json"

        predictor = EnsemblePredictor.__new__(EnsemblePredictor)
        predictor.policy = {"combo_optimizer": {"combo_mode": "live", "beam_width": 12}}
        predictor.policy_version = "test-policy"
        predictor.db_path = str(Path(tmp) / "lending_history.db")
        predictor.refresh_probe_state_path = str(Path(tmp) / "refresh_probe_state.json")
        predictor.order_manager = None
        predictor._funding_book_cache = {}
        predictor._stale_issues = []
        predictor._persist_prediction_history = lambda ranked_predictions: None
        predictor._calc_market_liquidity = lambda preds: {
            "fUSD": {
                "level": "medium",
                "score": 60.0,
                "fast_score": 0.62,
                "fillability_signal": 0.60,
                "volume_ratio_24h": 0.84,
            },
            "fUST": {
                "level": "medium",
                "score": 52.0,
                "fast_score": 0.50,
                "fillability_signal": 0.48,
                "volume_ratio_24h": 0.62,
            },
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

        live_combo = [
            dict(ranked_preds[2], recommendation_rank=1, decision_mode="exploit", candidate_id="fUST-30-stretch-premium", anchor_backed=True),
            dict(ranked_preds[0], recommendation_rank=2, decision_mode="exploit", candidate_id="fUSD-120-premium", anchor_backed=True),
            dict(ranked_preds[3], recommendation_rank=3, decision_mode="exploit", candidate_id="fUSD-90-premium", anchor_backed=True),
            dict(ranked_preds[1], recommendation_rank=4, decision_mode="exploit", candidate_id="fUSD-30-balanced-mid", anchor_backed=True),
            dict(ranked_preds[4], recommendation_rank=5, decision_mode="exploit", candidate_id="fUSD-14-balanced-mid", anchor_backed=True),
        ]
        predictor._build_shadow_combo = lambda sorted_preds, update_cycle_id, beam_width: (
            live_combo,
            {
                "beam_width": beam_width,
                "combo_revenue_ev": 8.0,
                "combo_fill_quality": 0.6,
                "anchor_backed_pair_count": 5,
            },
        )

        predictor.generate_recommendations(str(output_path))

        result = json.loads(output_path.read_text())

        assert [item["rank"] for item in result["recommendations"]] == [1, 2, 3, 4, 5, 6]
        assert [
            (item["currency"], item["period"]) for item in result["recommendations"][:5]
        ] == [
            ("fUST", 30),
            ("fUSD", 120),
            ("fUSD", 90),
            ("fUSD", 30),
            ("fUSD", 14),
        ]
        assert result["recommendations"][5]["currency"] == "fUSD"
        assert result["recommendations"][5]["period"] == 2


def test_generate_recommendations_fail_closes_when_live_combo_build_raises():
    with tempfile.TemporaryDirectory() as tmp:
        output_path = Path(tmp) / "optimal_combination.json"

        predictor = EnsemblePredictor.__new__(EnsemblePredictor)
        predictor.policy = {"combo_optimizer": {"combo_mode": "live", "beam_width": 12}}
        predictor.policy_version = "test-policy"
        predictor.db_path = str(Path(tmp) / "lending_history.db")
        predictor.refresh_probe_state_path = str(Path(tmp) / "refresh_probe_state.json")
        predictor.order_manager = None
        predictor._funding_book_cache = {}
        predictor._stale_issues = []
        predictor._persist_prediction_history = lambda ranked_predictions: None
        predictor._calc_market_liquidity = lambda preds: {
            "fUSD": {"level": "medium", "score": 60.0, "volume_ratio_24h": 0.84},
            "fUST": {"level": "medium", "score": 52.0, "volume_ratio_24h": 0.62},
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
        predictor._build_shadow_combo = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("combo boom"))

        predictor.generate_recommendations(str(output_path))

        result = json.loads(output_path.read_text())

        assert result["status"] == "error"
        assert result["fail_closed"] is True
        assert result["error"] == "C3 live mode fail-closed: combo build failed"
        assert result["recommendations"] == []


def test_generate_recommendations_fail_closes_when_live_combo_is_incomplete():
    with tempfile.TemporaryDirectory() as tmp:
        output_path = Path(tmp) / "optimal_combination.json"

        predictor = EnsemblePredictor.__new__(EnsemblePredictor)
        predictor.policy = {"combo_optimizer": {"combo_mode": "live", "beam_width": 12}}
        predictor.policy_version = "test-policy"
        predictor.db_path = str(Path(tmp) / "lending_history.db")
        predictor.refresh_probe_state_path = str(Path(tmp) / "refresh_probe_state.json")
        predictor.order_manager = None
        predictor._funding_book_cache = {}
        predictor._stale_issues = []
        predictor._persist_prediction_history = lambda ranked_predictions: None
        predictor._calc_market_liquidity = lambda preds: {
            "fUSD": {"level": "medium", "score": 60.0, "volume_ratio_24h": 0.84},
            "fUST": {"level": "medium", "score": 52.0, "volume_ratio_24h": 0.62},
        }
        ranked_preds = [
            _make_prediction("fUSD", 120, 12.4, exec_prob=0.62),
            _make_prediction("fUSD", 30, 9.2, exec_prob=0.58),
            _make_prediction("fUST", 30, 11.4, exec_prob=0.64),
            _make_prediction("fUSD", 90, 11.8, exec_prob=0.60),
            _make_prediction("fUSD", 14, 8.8, exec_prob=0.56),
            _make_prediction("fUSD", 2, 5.2, exec_prob=0.67),
        ]
        predictor.get_latest_predictions = lambda: ranked_preds
        predictor._apply_path_ranking = lambda valid_preds, market_liquidity, fusd_2d_pred: sorted(
            valid_preds,
            key=lambda pred: pred["predicted_rate"],
            reverse=True,
        )
        predictor._build_shadow_combo = lambda *args, **kwargs: (
            [
                dict(ranked_preds[0], recommendation_rank=1, decision_mode="exploit", candidate_id="fUSD-120-balanced-mid"),
                dict(ranked_preds[2], recommendation_rank=2, decision_mode="exploit", candidate_id="fUST-30-balanced-mid"),
                dict(ranked_preds[3], recommendation_rank=3, decision_mode="exploit", candidate_id="fUSD-90-balanced-mid"),
                dict(ranked_preds[1], recommendation_rank=4, decision_mode="exploit", candidate_id="fUSD-30-balanced-mid"),
            ],
            {
                "beam_width": 12,
                "combo_revenue_ev": 7.0,
                "combo_fill_quality": 0.5,
                "anchor_backed_pair_count": 5,
            },
        )

        predictor.generate_recommendations(str(output_path))

        result = json.loads(output_path.read_text())

        assert result["status"] == "error"
        assert result["fail_closed"] is True
    assert result["error"] == "C3 live mode fail-closed: incomplete combo"
    assert result["recommendations"] == []


def test_generate_recommendations_fail_closes_when_live_combo_has_no_anchor_backed_pool():
    with tempfile.TemporaryDirectory() as tmp:
        output_path = Path(tmp) / "optimal_combination.json"

        predictor = EnsemblePredictor.__new__(EnsemblePredictor)
        predictor.policy = {"combo_optimizer": {"combo_mode": "live", "beam_width": 12}}
        predictor.policy_version = "test-policy"
        predictor.db_path = str(Path(tmp) / "lending_history.db")
        predictor.refresh_probe_state_path = str(Path(tmp) / "refresh_probe_state.json")
        predictor.order_manager = None
        predictor._funding_book_cache = {}
        predictor._stale_issues = []
        predictor._persist_prediction_history = lambda ranked_predictions: None
        predictor._calc_market_liquidity = lambda preds: {
            "fUSD": {"level": "medium", "score": 60.0, "volume_ratio_24h": 0.84},
            "fUST": {"level": "medium", "score": 52.0, "volume_ratio_24h": 0.62},
        }
        ranked_preds = [
            _make_prediction("fUSD", 120, 12.4, exec_prob=0.62),
            _make_prediction("fUSD", 30, 9.2, exec_prob=0.58),
            _make_prediction("fUST", 30, 11.4, exec_prob=0.64),
            _make_prediction("fUSD", 90, 11.8, exec_prob=0.60),
            _make_prediction("fUSD", 14, 8.8, exec_prob=0.56),
            _make_prediction("fUSD", 2, 5.2, exec_prob=0.67),
        ]
        predictor.get_latest_predictions = lambda: ranked_preds
        predictor._apply_path_ranking = lambda valid_preds, market_liquidity, fusd_2d_pred: sorted(
            valid_preds,
            key=lambda pred: pred["predicted_rate"],
            reverse=True,
        )
        predictor._load_market_anchor_rows = lambda currency, period: []

        predictor.generate_recommendations(str(output_path))

        result = json.loads(output_path.read_text())

        assert result["status"] == "error"
        assert result["fail_closed"] is True
    assert result["error"] == "C3 live mode fail-closed: insufficient anchor-backed candidate pool"
    assert result["recommendations"] == []


def test_generate_recommendations_fail_closes_when_live_combo_contains_suspended_pair():
    with tempfile.TemporaryDirectory() as tmp:
        output_path = Path(tmp) / "optimal_combination.json"

        predictor = EnsemblePredictor.__new__(EnsemblePredictor)
        predictor.policy = {"combo_optimizer": {"combo_mode": "live", "beam_width": 12}}
        predictor.policy_version = "test-policy"
        predictor.db_path = str(Path(tmp) / "lending_history.db")
        predictor.refresh_probe_state_path = str(Path(tmp) / "refresh_probe_state.json")
        predictor.order_manager = None
        predictor._funding_book_cache = {}
        predictor._stale_issues = []
        predictor._persist_prediction_history = lambda ranked_predictions: None
        predictor._calc_market_liquidity = lambda preds: {
            "fUSD": {"level": "medium", "score": 60.0, "volume_ratio_24h": 0.84},
            "fUST": {"level": "medium", "score": 52.0, "volume_ratio_24h": 0.62},
        }
        ranked_preds = [
            _make_prediction("fUSD", 120, 12.4, exec_prob=0.62),
            _make_prediction("fUSD", 30, 9.2, exec_prob=0.58),
            _make_prediction("fUST", 30, 11.4, exec_prob=0.64),
            _make_prediction("fUSD", 90, 11.8, exec_prob=0.60),
            _make_prediction("fUSD", 14, 8.8, exec_prob=0.56),
            _make_prediction("fUSD", 2, 5.2, exec_prob=0.67),
        ]
        predictor.get_latest_predictions = lambda: ranked_preds
        predictor._apply_path_ranking = lambda valid_preds, market_liquidity, fusd_2d_pred: sorted(
            valid_preds,
            key=lambda pred: pred["predicted_rate"],
            reverse=True,
        )
        predictor._build_shadow_combo = lambda *args, **kwargs: (
            [
                dict(ranked_preds[2], recommendation_rank=1, decision_mode="exploit", candidate_id="fUST-30-stretch-premium", anchor_backed=True),
                dict(ranked_preds[0], recommendation_rank=2, decision_mode="exploit", candidate_id="fUSD-120-premium", anchor_backed=True),
                dict(ranked_preds[3], recommendation_rank=3, decision_mode="exploit", candidate_id="fUSD-90-premium", anchor_backed=True),
                dict(ranked_preds[1], recommendation_rank=4, decision_mode="exploit", candidate_id="fUSD-30-balanced-mid", anchor_backed=True),
                dict(ranked_preds[4], recommendation_rank=5, decision_mode="exploit", candidate_id="fUSD-14-balanced-mid", anchor_backed=True),
            ],
            {
                "beam_width": 12,
                "combo_revenue_ev": 8.0,
                "combo_fill_quality": 0.6,
                "anchor_backed_pair_count": 5,
            },
        )
        predictor._is_zero_liquidity_suspended = lambda currency, period: (currency, period) == ("fUST", 30)

        predictor.generate_recommendations(str(output_path))

        result = json.loads(output_path.read_text())

        assert result["status"] == "error"
        assert result["fail_closed"] is True
        assert result["error"] == "C3 live mode fail-closed: suspended live combo pairs: fUST-30d"
        assert result["recommendations"] == []


def test_generate_recommendations_fail_closes_when_live_top5_contains_non_anchor_backed_item():
    with tempfile.TemporaryDirectory() as tmp:
        output_path = Path(tmp) / "optimal_combination.json"

        predictor = EnsemblePredictor.__new__(EnsemblePredictor)
        predictor.policy = {"combo_optimizer": {"combo_mode": "live", "beam_width": 12}}
        predictor.policy_version = "test-policy"
        predictor.db_path = str(Path(tmp) / "lending_history.db")
        predictor.refresh_probe_state_path = str(Path(tmp) / "refresh_probe_state.json")
        predictor.order_manager = None
        predictor._funding_book_cache = {}
        predictor._stale_issues = []
        predictor._persist_prediction_history = lambda ranked_predictions: None
        predictor._calc_market_liquidity = lambda preds: {
            "fUSD": {"level": "medium", "score": 60.0, "volume_ratio_24h": 0.84},
            "fUST": {"level": "medium", "score": 52.0, "volume_ratio_24h": 0.62},
        }
        ranked_preds = [
            _make_prediction("fUSD", 120, 12.4, exec_prob=0.62),
            _make_prediction("fUSD", 30, 9.2, exec_prob=0.58),
            _make_prediction("fUST", 30, 11.4, exec_prob=0.64),
            _make_prediction("fUSD", 90, 11.8, exec_prob=0.60),
            _make_prediction("fUSD", 14, 8.8, exec_prob=0.56),
            _make_prediction("fUSD", 2, 5.2, exec_prob=0.67),
        ]
        predictor.get_latest_predictions = lambda: ranked_preds
        predictor._apply_path_ranking = lambda valid_preds, market_liquidity, fusd_2d_pred: sorted(
            valid_preds,
            key=lambda pred: pred["predicted_rate"],
            reverse=True,
        )
        predictor._build_shadow_combo = lambda *args, **kwargs: (
            [
                dict(ranked_preds[2], recommendation_rank=1, decision_mode="exploit", candidate_id="fUST-30-stretch-premium", anchor_backed=True),
                dict(ranked_preds[0], recommendation_rank=2, decision_mode="exploit", candidate_id="fUSD-120-premium", anchor_backed=True),
                dict(ranked_preds[3], recommendation_rank=3, decision_mode="exploit", candidate_id="fUSD-90-premium", anchor_backed=True),
                dict(ranked_preds[1], recommendation_rank=4, decision_mode="exploit", candidate_id="fUSD-30-balanced-mid", anchor_backed=False),
                dict(ranked_preds[4], recommendation_rank=5, decision_mode="exploit", candidate_id="fUSD-14-balanced-mid", anchor_backed=True),
            ],
            {
                "beam_width": 12,
                "combo_revenue_ev": 8.0,
                "combo_fill_quality": 0.6,
                "anchor_backed_pair_count": 5,
            },
        )

        predictor.generate_recommendations(str(output_path))

        result = json.loads(output_path.read_text())

        assert result["status"] == "error"
        assert result["fail_closed"] is True
        assert result["error"] == "C3 live mode fail-closed: combo contains non-anchor-backed candidate"
        assert result["recommendations"] == []


def test_generate_recommendations_fail_closes_when_live_prediction_history_persist_fails():
    class _OrderManagerStub:
        def __init__(self):
            self.created = 0

        def get_order_count(self, currency, period):
            return 0

        def needs_refresh_probe(self, currency, period, lookback_hours, min_validations):
            return False

        def create_virtual_order(self, pred):
            self.created += 1
            return f"order-{self.created}"

    with tempfile.TemporaryDirectory() as tmp:
        output_path = Path(tmp) / "optimal_combination.json"
        order_manager = _OrderManagerStub()

        predictor = EnsemblePredictor.__new__(EnsemblePredictor)
        predictor.policy = {"combo_optimizer": {"combo_mode": "live", "beam_width": 12}}
        predictor.policy_version = "test-policy"
        predictor.db_path = str(Path(tmp) / "lending_history.db")
        predictor.refresh_probe_state_path = str(Path(tmp) / "refresh_probe_state.json")
        predictor.order_manager = order_manager
        predictor._funding_book_cache = {}
        predictor._stale_issues = []
        predictor._persist_prediction_history = lambda ranked_predictions: (_ for _ in ()).throw(RuntimeError("history boom"))
        predictor._calc_market_liquidity = lambda preds: {
            "fUSD": {"level": "medium", "score": 60.0, "volume_ratio_24h": 0.84},
            "fUST": {"level": "medium", "score": 52.0, "volume_ratio_24h": 0.62},
        }
        ranked_preds = [
            _make_prediction("fUSD", 120, 12.4, exec_prob=0.62),
            _make_prediction("fUSD", 30, 9.2, exec_prob=0.58),
            _make_prediction("fUST", 30, 11.4, exec_prob=0.64),
            _make_prediction("fUSD", 90, 11.8, exec_prob=0.60),
            _make_prediction("fUSD", 14, 8.8, exec_prob=0.56),
            _make_prediction("fUSD", 2, 5.2, exec_prob=0.67),
        ]
        predictor.get_latest_predictions = lambda: ranked_preds
        predictor._apply_path_ranking = lambda valid_preds, market_liquidity, fusd_2d_pred: sorted(
            valid_preds,
            key=lambda pred: pred["predicted_rate"],
            reverse=True,
        )
        predictor._build_shadow_combo = lambda *args, **kwargs: (
            [
                dict(ranked_preds[2], recommendation_rank=1, decision_mode="exploit", candidate_id="fUST-30-stretch-premium", anchor_backed=True),
                dict(ranked_preds[0], recommendation_rank=2, decision_mode="exploit", candidate_id="fUSD-120-premium", anchor_backed=True),
                dict(ranked_preds[3], recommendation_rank=3, decision_mode="exploit", candidate_id="fUSD-90-premium", anchor_backed=True),
                dict(ranked_preds[1], recommendation_rank=4, decision_mode="exploit", candidate_id="fUSD-30-balanced-mid", anchor_backed=True),
                dict(ranked_preds[4], recommendation_rank=5, decision_mode="exploit", candidate_id="fUSD-14-balanced-mid", anchor_backed=True),
            ],
            {
                "beam_width": 12,
                "combo_revenue_ev": 8.0,
                "combo_fill_quality": 0.6,
                "anchor_backed_pair_count": 5,
            },
        )
        predictor._is_zero_liquidity_suspended = lambda currency, period: False
        predictor._load_refresh_probe_state = lambda: {}
        predictor._save_refresh_probe_state = lambda state: None
        predictor._policy_value = lambda section, key, default=None: default

        predictor.generate_recommendations(str(output_path))

        result = json.loads(output_path.read_text())

        assert result["status"] == "error"
        assert result["fail_closed"] is True
        assert result["error"] == "C3 live mode fail-closed: prediction_history persist failed"
        assert result["recommendations"] == []
        assert order_manager.created == 0
