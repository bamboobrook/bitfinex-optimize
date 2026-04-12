import sqlite3
import sys
import tempfile
import unittest
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_predictor_rank6 import EnsemblePredictor, _make_prediction
from ml_engine.order_manager import OrderManager


class C3IdentityChainTest(unittest.TestCase):
    def _make_predictor_for_generate_recommendations(self, db_path: Path, state_path: Path):
        predictor = EnsemblePredictor.__new__(EnsemblePredictor)
        predictor.db_path = str(db_path)
        predictor.policy = {}
        predictor.policy_version = "test-policy"
        predictor.refresh_probe_state_path = str(state_path)
        predictor._stale_issues = []
        predictor._funding_book_cache = {}
        predictor.order_manager = OrderManager(db_path)
        predictor._calc_market_liquidity = lambda preds: {
            "fUSD": {"level": "medium", "score": 56.6, "avg_exec_rate": 0.41, "volume_ratio_24h": 0.019},
            "fUST": {"level": "medium", "score": 52.0, "avg_exec_rate": 0.38, "volume_ratio_24h": 0.18},
        }
        predictor._apply_path_ranking = lambda valid_preds, market_liquidity, fusd_2d_pred: sorted(
            valid_preds,
            key=lambda pred: pred["predicted_rate"],
            reverse=True,
        )
        predictor._is_zero_liquidity_suspended = lambda currency, period: False
        return predictor

    def test_prediction_history_and_virtual_orders_share_identity_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "lending_history.db"

            predictor = EnsemblePredictor.__new__(EnsemblePredictor)
            predictor.db_path = str(db_path)
            predictor.policy = {}
            predictor.policy_version = "test-policy"
            predictor._stale_issues = []
            predictor._ensure_prediction_history_schema()

            pred = _make_prediction("fUSD", 120, 12.8, exec_prob=0.62)
            pred.update({
                "update_cycle_id": "cycle-1",
                "recommendation_rank": 1,
                "rank_weight": 0.60,
                "candidate_id": "fUSD-120-balanced-mid",
                "decision_mode": "exploit",
            })
            predictor._persist_prediction_history([pred])

            manager = OrderManager(db_path)
            manager.create_virtual_order(pred)

            with sqlite3.connect(db_path) as conn:
                history = conn.execute(
                    "SELECT update_cycle_id, recommendation_rank, rank_weight, candidate_id, decision_mode "
                    "FROM prediction_history"
                ).fetchone()
                order = conn.execute(
                    "SELECT update_cycle_id, recommendation_rank, rank_weight, candidate_id, decision_mode "
                    "FROM virtual_orders"
                ).fetchone()

            self.assertEqual(history, ("cycle-1", 1, 0.60, "fUSD-120-balanced-mid", "exploit"))
            self.assertEqual(order, ("cycle-1", 1, 0.60, "fUSD-120-balanced-mid", "exploit"))

    def test_generate_recommendations_assigns_shared_identity_chain_on_main_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "lending_history.db"
            output_path = Path(tmp) / "optimal_combination.json"
            state_path = Path(tmp) / "refresh_probe_state.json"

            predictor = self._make_predictor_for_generate_recommendations(db_path, state_path)
            predictor.get_latest_predictions = lambda: [
                _make_prediction("fUSD", 120, 12.8, exec_prob=0.62),
                _make_prediction("fUST", 30, 11.4, exec_prob=0.58),
                _make_prediction("fUSD", 2, 5.2, exec_prob=0.67),
            ]

            predictor.generate_recommendations(str(output_path))

            with sqlite3.connect(db_path) as conn:
                history = conn.execute(
                    """
                    SELECT update_cycle_id, recommendation_rank, rank_weight, candidate_id, decision_mode
                    FROM prediction_history
                    WHERE currency = 'fUSD' AND period = 120
                    ORDER BY id DESC
                    LIMIT 1
                    """
                ).fetchone()
                order = conn.execute(
                    """
                    SELECT update_cycle_id, recommendation_rank, rank_weight, candidate_id, decision_mode
                    FROM virtual_orders
                    WHERE currency = 'fUSD' AND period = 120
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                ).fetchone()

            self.assertIsNotNone(history)
            self.assertIsNotNone(order)
            self.assertEqual(history, order)
            self.assertEqual(history[1], 1)
            self.assertEqual(history[2], 0.60)
            self.assertEqual(history[4], "exploit")
            self.assertTrue(all(value not in (None, "") for value in history))

    def test_live_combo_unifies_json_prediction_history_and_exploit_orders(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "lending_history.db"
            output_path = Path(tmp) / "optimal_combination.json"
            state_path = Path(tmp) / "refresh_probe_state.json"

            predictor = self._make_predictor_for_generate_recommendations(db_path, state_path)
            predictor.policy = {"combo_optimizer": {"combo_mode": "live", "beam_width": 12}}
            predictor.order_manager.get_order_count = lambda currency, period: 0
            predictor.order_manager.needs_refresh_probe = lambda currency, period, lookback_hours, min_validations: False

            ranked_preds = [
                _make_prediction("fUSD", 120, 12.8, exec_prob=0.62),
                _make_prediction("fUST", 30, 11.4, exec_prob=0.58),
                _make_prediction("fUSD", 90, 11.7, exec_prob=0.61),
                _make_prediction("fUSD", 30, 9.1, exec_prob=0.55),
                _make_prediction("fUST", 14, 8.9, exec_prob=0.52),
                _make_prediction("fUSD", 2, 5.2, exec_prob=0.67),
            ]
            predictor.get_latest_predictions = lambda: ranked_preds

            live_pairs = [
                ("fUST", 30, "fUST-30-stretch-premium"),
                ("fUSD", 120, "fUSD-120-premium"),
                ("fUSD", 90, "fUSD-90-premium"),
                ("fUSD", 30, "fUSD-30-balanced-mid"),
                ("fUST", 14, "fUST-14-balanced-low"),
            ]

            def _build_live_combo(sorted_preds, update_cycle_id, beam_width):
                lookup = {(pred["currency"], pred["period"]): pred for pred in sorted_preds}
                combo = []
                for rank, (currency, period, candidate_id) in enumerate(live_pairs, start=1):
                    combo.append({
                        **lookup[(currency, period)],
                        "update_cycle_id": update_cycle_id,
                        "recommendation_rank": rank,
                        "rank_weight": 0.60 if rank == 1 else 0.10,
                        "candidate_id": candidate_id,
                        "decision_mode": "exploit",
                        "anchor_backed": True,
                    })
                return combo, {
                    "beam_width": beam_width,
                    "combo_revenue_ev": 8.0,
                    "combo_fill_quality": 0.6,
                    "anchor_backed_pair_count": 5,
                }

            predictor._build_shadow_combo = _build_live_combo

            predictor.generate_recommendations(str(output_path))

            result = json.loads(output_path.read_text())
            expected_top5 = [
                (item["currency"], item["period"], live_pairs[index][2])
                for index, item in enumerate(result["recommendations"][:5])
            ]

            with sqlite3.connect(db_path) as conn:
                history_rows = conn.execute(
                    """
                    SELECT currency, period, candidate_id, update_cycle_id
                    FROM prediction_history
                    WHERE recommendation_rank BETWEEN 1 AND 5
                    ORDER BY recommendation_rank ASC
                    """
                ).fetchall()
                exploit_rows = conn.execute(
                    """
                    SELECT currency, period, candidate_id, update_cycle_id
                    FROM virtual_orders
                    WHERE decision_mode = 'exploit'
                    ORDER BY recommendation_rank ASC, created_at ASC
                    """
                ).fetchall()
                probe_modes = conn.execute(
                    """
                    SELECT DISTINCT decision_mode
                    FROM virtual_orders
                    WHERE decision_mode IS NOT NULL AND decision_mode != 'exploit'
                    """
                ).fetchall()

            self.assertEqual([row[:3] for row in history_rows], expected_top5)
            self.assertEqual([row[:3] for row in exploit_rows], expected_top5)
            self.assertEqual(len(exploit_rows), 5)
            self.assertTrue(all(row[3] == history_rows[0][3] for row in history_rows))
            self.assertTrue(all(row[3] == exploit_rows[0][3] for row in exploit_rows))
            self.assertTrue(all(mode[0] == "probe" for mode in probe_modes))

    def test_live_combo_creates_current_cycle_exploit_orders_even_when_old_pending_rows_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "lending_history.db"
            output_path = Path(tmp) / "optimal_combination.json"
            state_path = Path(tmp) / "refresh_probe_state.json"

            predictor = self._make_predictor_for_generate_recommendations(db_path, state_path)
            predictor.policy = {"combo_optimizer": {"combo_mode": "live", "beam_width": 12}}
            predictor.order_manager.get_order_count = lambda currency, period: 0
            predictor.order_manager.needs_refresh_probe = lambda currency, period, lookback_hours, min_validations: False

            shared_data_timestamp = "2026-04-13 12:00:00"
            ranked_preds = [
                _make_prediction("fUSD", 120, 12.8, exec_prob=0.62),
                _make_prediction("fUST", 30, 11.4, exec_prob=0.58),
                _make_prediction("fUSD", 90, 11.7, exec_prob=0.61),
                _make_prediction("fUSD", 30, 9.1, exec_prob=0.55),
                _make_prediction("fUST", 14, 8.9, exec_prob=0.52),
                _make_prediction("fUSD", 2, 5.2, exec_prob=0.67),
            ]
            for pred in ranked_preds:
                pred["data_timestamp"] = shared_data_timestamp
            predictor.get_latest_predictions = lambda: ranked_preds

            live_pairs = [
                ("fUST", 30, "fUST-30-stretch-premium"),
                ("fUSD", 120, "fUSD-120-premium"),
                ("fUSD", 90, "fUSD-90-premium"),
                ("fUSD", 30, "fUSD-30-balanced-mid"),
                ("fUST", 14, "fUST-14-balanced-low"),
            ]

            def _build_live_combo(sorted_preds, update_cycle_id, beam_width):
                lookup = {(pred["currency"], pred["period"]): pred for pred in sorted_preds}
                combo = []
                for rank, (currency, period, candidate_id) in enumerate(live_pairs, start=1):
                    combo.append({
                        **lookup[(currency, period)],
                        "update_cycle_id": update_cycle_id,
                        "recommendation_rank": rank,
                        "rank_weight": 0.60 if rank == 1 else 0.10,
                        "candidate_id": candidate_id,
                        "decision_mode": "exploit",
                        "anchor_backed": True,
                    })
                return combo, {
                    "beam_width": beam_width,
                    "combo_revenue_ev": 8.0,
                    "combo_fill_quality": 0.6,
                    "anchor_backed_pair_count": 5,
                }

            predictor._build_shadow_combo = _build_live_combo

            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS virtual_orders (
                        order_id TEXT PRIMARY KEY,
                        currency TEXT NOT NULL,
                        period INTEGER NOT NULL,
                        predicted_rate REAL NOT NULL,
                        order_timestamp TEXT NOT NULL,
                        validation_window_hours INTEGER NOT NULL,
                        status TEXT DEFAULT 'PENDING',
                        executed_at TEXT,
                        execution_rate REAL,
                        execution_delay_minutes INTEGER,
                        max_market_rate REAL,
                        rate_gap REAL,
                        model_version TEXT,
                        prediction_confidence TEXT,
                        prediction_strategy TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        validated_at TIMESTAMP,
                        update_cycle_id TEXT,
                        recommendation_rank INTEGER,
                        rank_weight REAL,
                        candidate_id TEXT,
                        decision_mode TEXT
                    )
                    """
                )
                for rank, (currency, period, candidate_id) in enumerate(live_pairs, start=1):
                    conn.execute(
                        """
                        INSERT INTO virtual_orders (
                            order_id, currency, period, predicted_rate, order_timestamp,
                            validation_window_hours, status, model_version, created_at,
                            update_cycle_id, recommendation_rank, rank_weight, candidate_id, decision_mode
                        ) VALUES (?, ?, ?, ?, ?, ?, 'PENDING', 'v0', ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            f"old-{currency}-{period}",
                            currency,
                            period,
                            1.0,
                            shared_data_timestamp,
                            24,
                            "2026-04-13 11:00:00",
                            "old-cycle",
                            rank,
                            0.60 if rank == 1 else 0.10,
                            candidate_id,
                            "exploit",
                        ),
                    )
                conn.commit()

            predictor.generate_recommendations(str(output_path))

            result = json.loads(output_path.read_text())
            expected_top5 = [
                (item["currency"], item["period"], live_pairs[index][2])
                for index, item in enumerate(result["recommendations"][:5])
            ]

            with sqlite3.connect(db_path) as conn:
                history_cycle_id = conn.execute(
                    """
                    SELECT update_cycle_id
                    FROM prediction_history
                    WHERE recommendation_rank = 1
                    ORDER BY id DESC
                    LIMIT 1
                    """
                ).fetchone()[0]
                history_rows = conn.execute(
                    """
                    SELECT currency, period, candidate_id
                    FROM prediction_history
                    WHERE update_cycle_id = ?
                      AND recommendation_rank BETWEEN 1 AND 5
                    ORDER BY recommendation_rank ASC
                    """,
                    (history_cycle_id,),
                ).fetchall()
                exploit_rows = conn.execute(
                    """
                    SELECT currency, period, candidate_id
                    FROM virtual_orders
                    WHERE update_cycle_id = ?
                      AND decision_mode = 'exploit'
                    ORDER BY recommendation_rank ASC, created_at ASC
                    """,
                    (history_cycle_id,),
                ).fetchall()
                old_cycle_rows = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM virtual_orders
                    WHERE update_cycle_id = 'old-cycle'
                      AND decision_mode = 'exploit'
                    """
                ).fetchone()[0]

            self.assertEqual([row for row in history_rows], expected_top5)
            self.assertEqual([row for row in exploit_rows], expected_top5)
            self.assertEqual(len(exploit_rows), 5)
            self.assertEqual(old_cycle_rows, 5)

    def test_live_combo_cleans_history_when_virtual_order_creation_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "lending_history.db"
            output_path = Path(tmp) / "optimal_combination.json"
            state_path = Path(tmp) / "refresh_probe_state.json"

            predictor = self._make_predictor_for_generate_recommendations(db_path, state_path)
            predictor.policy = {"combo_optimizer": {"combo_mode": "live", "beam_width": 12}}
            predictor.order_manager.get_order_count = lambda currency, period: 0
            predictor.order_manager.needs_refresh_probe = lambda currency, period, lookback_hours, min_validations: False
            predictor.order_manager.create_virtual_order = lambda pred: (_ for _ in ()).throw(RuntimeError("order boom"))

            ranked_preds = [
                _make_prediction("fUSD", 120, 12.8, exec_prob=0.62),
                _make_prediction("fUST", 30, 11.4, exec_prob=0.58),
                _make_prediction("fUSD", 90, 11.7, exec_prob=0.61),
                _make_prediction("fUSD", 30, 9.1, exec_prob=0.55),
                _make_prediction("fUST", 14, 8.9, exec_prob=0.52),
                _make_prediction("fUSD", 2, 5.2, exec_prob=0.67),
            ]
            shared_data_timestamp = "2026-04-13 12:00:00"
            for pred in ranked_preds:
                pred["data_timestamp"] = shared_data_timestamp
            predictor.get_latest_predictions = lambda: ranked_preds

            live_pairs = [
                ("fUST", 30, "fUST-30-stretch-premium"),
                ("fUSD", 120, "fUSD-120-premium"),
                ("fUSD", 90, "fUSD-90-premium"),
                ("fUSD", 30, "fUSD-30-balanced-mid"),
                ("fUST", 14, "fUST-14-balanced-low"),
            ]

            def _build_live_combo(sorted_preds, update_cycle_id, beam_width):
                lookup = {(pred["currency"], pred["period"]): pred for pred in sorted_preds}
                combo = []
                for rank, (currency, period, candidate_id) in enumerate(live_pairs, start=1):
                    combo.append({
                        **lookup[(currency, period)],
                        "update_cycle_id": update_cycle_id,
                        "recommendation_rank": rank,
                        "rank_weight": 0.60 if rank == 1 else 0.10,
                        "candidate_id": candidate_id,
                        "decision_mode": "exploit",
                        "anchor_backed": True,
                    })
                return combo, {
                    "beam_width": beam_width,
                    "combo_revenue_ev": 8.0,
                    "combo_fill_quality": 0.6,
                    "anchor_backed_pair_count": 5,
                }

            predictor._build_shadow_combo = _build_live_combo

            predictor.generate_recommendations(str(output_path))

            result = json.loads(output_path.read_text())
            with sqlite3.connect(db_path) as conn:
                history_rows = conn.execute(
                    "SELECT COUNT(*) FROM prediction_history"
                ).fetchone()[0]
                order_rows = conn.execute(
                    "SELECT COUNT(*) FROM virtual_orders WHERE update_cycle_id IS NOT NULL"
                ).fetchone()[0]

            self.assertEqual(result["status"], "error")
            self.assertTrue(result["fail_closed"])
            self.assertEqual(result["error"], "C3 live mode fail-closed: virtual order creation failed")
            self.assertEqual(result["recommendations"], [])
            self.assertEqual(history_rows, 0)
            self.assertEqual(order_rows, 0)


if __name__ == "__main__":
    unittest.main()
