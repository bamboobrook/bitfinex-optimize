import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _install_predictor_import_stubs():
    xgb = types.ModuleType("xgboost")

    class Booster:
        def load_model(self, *args, **kwargs):
            return None

        def set_param(self, *args, **kwargs):
            return None

        def predict(self, *args, **kwargs):
            return []

    class DMatrix:
        def __init__(self, *args, **kwargs):
            pass

    xgb.Booster = Booster
    xgb.DMatrix = DMatrix
    sys.modules.setdefault("xgboost", xgb)

    lgb = types.ModuleType("lightgbm")

    class LightGBMBooster:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, *args, **kwargs):
            return []

    lgb.Booster = LightGBMBooster
    sys.modules.setdefault("lightgbm", lgb)

    cat = types.ModuleType("catboost")

    class CatBoostRegressor:
        def load_model(self, *args, **kwargs):
            return None

        def predict(self, *args, **kwargs):
            return []

    class CatBoostClassifier:
        def load_model(self, *args, **kwargs):
            return None

        def predict_proba(self, *args, **kwargs):
            return []

    cat.CatBoostRegressor = CatBoostRegressor
    cat.CatBoostClassifier = CatBoostClassifier
    sys.modules.setdefault("catboost", cat)

    loguru = types.ModuleType("loguru")

    class Logger:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    loguru.logger = Logger()
    sys.modules.setdefault("loguru", loguru)


_install_predictor_import_stubs()

from ml_engine.predictor import EnsemblePredictor


def _make_prediction(currency: str, period: int, predicted_rate: float, exec_prob: float = 0.6):
    return {
        "currency": currency,
        "period": period,
        "current_rate": max(predicted_rate - 1.0, 0.1),
        "predicted_rate": predicted_rate,
        "execution_probability": exec_prob,
        "calibrated_execution_prob": exec_prob,
        "exec_rate_raw": exec_prob,
        "liquidity_score": 55.0,
        "liquidity_level": "medium",
        "order_count": 12,
        "avg_rate_gap_failed": 0.0,
        "conservative_rate": max(predicted_rate - 0.5, 0.1),
        "aggressive_rate": predicted_rate + 0.5,
        "balanced_rate": predicted_rate,
        "trend_factor": 1.0,
        "strategy": "test-strategy",
        "confidence": "High",
        "data_age_minutes": 0.0,
        "execution_rate_7d": exec_prob,
        "execution_rate_slow": exec_prob,
        "market_follow_error": 0.0,
    }


class PredictorRank6FallbackTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.output_path = Path(self.temp_dir.name) / "optimal_combination.json"
        self.db_path = Path(self.temp_dir.name) / "lending_history.db"

        predictor = EnsemblePredictor.__new__(EnsemblePredictor)
        predictor.policy = {}
        predictor.policy_version = "test-policy"
        predictor.db_path = str(self.db_path)
        predictor.refresh_probe_state_path = str(Path(self.temp_dir.name) / "refresh_probe_state.json")
        predictor.order_manager = None
        predictor._funding_book_cache = {}
        predictor._stale_issues = []
        predictor._ensure_prediction_history_schema()
        predictor._persist_prediction_history([
            _make_prediction("fUSD", 2, 5.1416, exec_prob=0.58),
        ])
        predictor._persist_prediction_history = lambda ranked_predictions: None
        predictor._calc_market_liquidity = lambda preds: {
            "fUSD": {"level": "medium", "score": 56.6, "avg_exec_rate": 0.41, "volume_ratio_24h": 0.019},
            "fUST": {"level": "low", "score": 36.0, "avg_exec_rate": 0.37, "volume_ratio_24h": 0.14},
        }

        def fake_get_latest_predictions():
            predictor._stale_issues = [
                {
                    "currency": "fUSD",
                    "period": 2,
                    "age_minutes": 355.16,
                    "source_timestamp": "2026-03-26 09:30:00",
                }
            ]
            return [
                _make_prediction("fUSD", 90, 16.5500, exec_prob=0.72),
                _make_prediction("fUSD", 120, 7.6048, exec_prob=0.65),
                _make_prediction("fUST", 20, 12.6664, exec_prob=0.64),
                _make_prediction("fUST", 5, 7.0791, exec_prob=0.63),
                _make_prediction("fUSD", 30, 6.5767, exec_prob=0.61),
            ]

        predictor.get_latest_predictions = fake_get_latest_predictions
        self.predictor = predictor

    def test_generate_recommendations_adds_fusd_2d_to_rank6_from_history(self):
        self.predictor.generate_recommendations(str(self.output_path))

        result = json.loads(self.output_path.read_text())
        recommendations = result["recommendations"]

        self.assertEqual(len(recommendations), 6)
        self.assertEqual(recommendations[-1]["rank"], 6)
        self.assertEqual(recommendations[-1]["currency"], "fUSD")
        self.assertEqual(recommendations[-1]["period"], 2)
        self.assertAlmostEqual(recommendations[-1]["rate"], 5.1416, places=4)

    def test_generate_recommendations_writes_strict_json_for_numpy_scalars(self):
        self.predictor._calc_market_liquidity = lambda preds: {
            "fUSD": {
                "level": "medium",
                "score": np.float64(56.6),
                "avg_exec_rate": np.float64(0.41),
                "volume_ratio_24h": np.float64(np.nan),
                "book_live": np.bool_(True),
            },
            "fUST": {
                "level": "low",
                "score": np.float64(36.0),
                "avg_exec_rate": np.float64(0.37),
                "volume_ratio_24h": np.float64(0.14),
                "book_live": np.bool_(False),
            },
        }

        self.predictor.generate_recommendations(str(self.output_path))

        raw = self.output_path.read_text()
        result = json.loads(raw)

        self.assertNotIn("NaN", raw)
        self.assertIs(result["market_liquidity"]["fUSD"]["book_live"], True)
        self.assertIsNone(result["market_liquidity"]["fUSD"]["volume_ratio_24h"])

    def test_generate_recommendations_keeps_previous_json_when_replace_fails(self):
        self.output_path.write_text(json.dumps({"status": "old", "recommendations": []}))

        with patch("ml_engine.predictor.os.replace", side_effect=OSError("replace blocked")):
            with self.assertRaises(OSError):
                self.predictor.generate_recommendations(str(self.output_path))

        self.assertEqual(
            json.loads(self.output_path.read_text()),
            {"status": "old", "recommendations": []},
        )


if __name__ == "__main__":
    unittest.main()
