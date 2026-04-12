import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path


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


def test_retraining_main_returns_nonzero_when_training_did_not_deploy(tmp_path):
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        "\n".join(
            [
                "import builtins",
                "import sys",
                "import types",
                "",
                "_orig_build_class = builtins.__build_class__",
                "",
                "def _patched_build_class(func, name, *args, **kwargs):",
                "    cls = _orig_build_class(func, name, *args, **kwargs)",
                "    if name == 'RetrainingScheduler':",
                "        cls.run = lambda self, force=False: False",
                "    return cls",
                "",
                "_data_processor = types.ModuleType('ml_engine.data_processor')",
                "_data_processor.DataProcessor = type('DataProcessor', (), {})",
                "sys.modules['ml_engine.data_processor'] = _data_processor",
                "",
                "_predictor = types.ModuleType('ml_engine.predictor')",
                "_predictor.EnsemblePredictor = type('EnsemblePredictor', (), {})",
                "sys.modules['ml_engine.predictor'] = _predictor",
                "",
                "_system_policy = types.ModuleType('ml_engine.system_policy')",
                "_system_policy.load_system_policy = lambda: {}",
                "sys.modules['ml_engine.system_policy'] = _system_policy",
                "",
                "builtins.__build_class__ = _patched_build_class",
                "",
            ]
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(PROJECT_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    result = subprocess.run(
        [sys.executable, "-m", "ml_engine.retraining_scheduler", "--force"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env=env,
    )

    assert result.returncode != 0


def test_generate_recommendations_live_mode_marks_empty_stale_predictions_fail_closed():
    _install_predictor_import_stubs()

    from ml_engine.predictor import EnsemblePredictor

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "optimal_combination.json"
        predictor = EnsemblePredictor.__new__(EnsemblePredictor)
        predictor.policy = {"combo_optimizer": {"combo_mode": "live"}}
        predictor.policy_version = "test-policy"
        predictor.order_manager = None
        predictor._funding_book_cache = {}
        predictor._stale_issues = []

        def fake_get_latest_predictions():
            predictor._stale_issues = [
                {
                    "currency": "fUSD",
                    "period": 2,
                    "age_minutes": 355.16,
                    "source_timestamp": "2026-03-26 09:30:00",
                }
            ]
            return []

        predictor.get_latest_predictions = fake_get_latest_predictions

        predictor.generate_recommendations(str(output_path))

        result = json.loads(output_path.read_text())
        assert result["fail_closed"] is True
        assert result["status"] in {"error", "failed"}
        assert "fail-closed" in result["error"]
