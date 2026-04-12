import sqlite3
import sys
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
