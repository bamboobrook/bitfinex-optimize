import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import pytest

from tests.test_predictor_rank6 import _make_prediction


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


def _install_scheduler_import_stubs(monkeypatch):
    data_processor = types.ModuleType("ml_engine.data_processor")
    data_processor.DataProcessor = type("DataProcessor", (), {})
    monkeypatch.setitem(sys.modules, "ml_engine.data_processor", data_processor)

    predictor = types.ModuleType("ml_engine.predictor")
    predictor.EnsemblePredictor = type("EnsemblePredictor", (), {})
    monkeypatch.setitem(sys.modules, "ml_engine.predictor", predictor)

    system_policy = types.ModuleType("ml_engine.system_policy")
    system_policy.load_system_policy = lambda: {}
    monkeypatch.setitem(sys.modules, "ml_engine.system_policy", system_policy)


def _write_model_meta_files(model_dir: Path, model_prefixes):
    for model_prefix in model_prefixes:
        (model_dir / f"{model_prefix}_meta.json").write_text("{}", encoding="utf-8")


@pytest.fixture
def scheduler_module(monkeypatch):
    _install_scheduler_import_stubs(monkeypatch)
    sys.modules.pop("ml_engine.retraining_scheduler", None)
    import ml_engine.retraining_scheduler as retraining_scheduler

    return retraining_scheduler


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


def test_generate_recommendations_live_mode_marks_stale_pairs_fail_closed():
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
                    "currency": "fUST",
                    "period": 30,
                    "age_minutes": 180.0,
                    "source_timestamp": "2026-04-13 12:00:00",
                }
            ]
            return [
                _make_prediction("fUSD", 120, 12.4, exec_prob=0.62),
                _make_prediction("fUST", 30, 11.4, exec_prob=0.64),
                _make_prediction("fUSD", 2, 5.2, exec_prob=0.67),
            ]

        predictor.get_latest_predictions = fake_get_latest_predictions
        predictor._calc_market_liquidity = lambda preds: {
            "fUSD": {"level": "medium", "score": 60.0, "volume_ratio_24h": 0.84},
            "fUST": {"level": "medium", "score": 52.0, "volume_ratio_24h": 0.62},
        }

        predictor.generate_recommendations(str(output_path))

        result = json.loads(output_path.read_text())
        assert result["fail_closed"] is True
        assert result["status"] == "error"
        assert "stale pairs detected" in result["error"]


def test_compare_models_returns_false_when_new_dir_drops_existing_enhanced_models(
    tmp_path, scheduler_module, monkeypatch
):
    old_model_dir = tmp_path / "old_models"
    new_model_dir = tmp_path / "new_models"
    old_model_dir.mkdir()
    new_model_dir.mkdir()

    base_models = [
        "fUSD_model_execution_prob",
        "fUSD_model_conservative",
        "fUSD_model_aggressive",
        "fUSD_model_balanced",
        "fUST_model_execution_prob",
        "fUST_model_conservative",
        "fUST_model_aggressive",
        "fUST_model_balanced",
    ]
    old_enhanced_models = [
        "fUSD_model_execution_prob_v2",
        "fUST_model_execution_prob_v2",
    ]

    _write_model_meta_files(old_model_dir, base_models + old_enhanced_models)
    _write_model_meta_files(new_model_dir, base_models)

    scheduler = scheduler_module.RetrainingScheduler(
        production_model_dir=str(old_model_dir),
        backup_dir=str(tmp_path / "backup"),
        log_dir=str(tmp_path / "logs"),
    )
    monkeypatch.setattr(scheduler, "_compare_model_performance", lambda *args: True)

    is_better, comparison = scheduler.compare_models(
        str(old_model_dir), str(new_model_dir)
    )

    assert is_better is False
    assert comparison["checks"]["enhanced_models"] is False


def test_compare_models_allows_complete_enhanced_models_without_false_positive(
    tmp_path, scheduler_module, monkeypatch
):
    old_model_dir = tmp_path / "old_models"
    new_model_dir = tmp_path / "new_models"
    old_model_dir.mkdir()
    new_model_dir.mkdir()

    base_models = [
        "fUSD_model_execution_prob",
        "fUSD_model_conservative",
        "fUSD_model_aggressive",
        "fUSD_model_balanced",
        "fUST_model_execution_prob",
        "fUST_model_conservative",
        "fUST_model_aggressive",
        "fUST_model_balanced",
    ]
    enhanced_models = [
        "fUSD_model_execution_prob_v2",
        "fUSD_model_revenue_optimized",
        "fUST_model_execution_prob_v2",
        "fUST_model_revenue_optimized",
    ]

    _write_model_meta_files(old_model_dir, base_models + enhanced_models)
    _write_model_meta_files(new_model_dir, base_models + enhanced_models)

    scheduler = scheduler_module.RetrainingScheduler(
        production_model_dir=str(old_model_dir),
        backup_dir=str(tmp_path / "backup"),
        log_dir=str(tmp_path / "logs"),
    )
    monkeypatch.setattr(scheduler, "_compare_model_performance", lambda *args: True)

    is_better, comparison = scheduler.compare_models(
        str(old_model_dir), str(new_model_dir)
    )

    assert is_better is True
    assert comparison["checks"]["enhanced_models"] is True


def test_compare_models_requires_enhanced_models_even_when_production_is_missing_them(
    tmp_path, scheduler_module, monkeypatch
):
    old_model_dir = tmp_path / "old_models"
    new_model_dir = tmp_path / "new_models"
    old_model_dir.mkdir()
    new_model_dir.mkdir()

    base_models = [
        "fUSD_model_execution_prob",
        "fUSD_model_conservative",
        "fUSD_model_aggressive",
        "fUSD_model_balanced",
        "fUST_model_execution_prob",
        "fUST_model_conservative",
        "fUST_model_aggressive",
        "fUST_model_balanced",
    ]

    _write_model_meta_files(old_model_dir, base_models)
    _write_model_meta_files(new_model_dir, base_models)

    scheduler = scheduler_module.RetrainingScheduler(
        production_model_dir=str(old_model_dir),
        backup_dir=str(tmp_path / "backup"),
        log_dir=str(tmp_path / "logs"),
    )
    monkeypatch.setattr(scheduler, "_compare_model_performance", lambda *args: True)

    is_better, comparison = scheduler.compare_models(
        str(old_model_dir), str(new_model_dir)
    )

    assert is_better is False
    assert comparison["checks"]["enhanced_models"] is False
    assert comparison["checks"]["enhanced_model_retention"] is False
    assert comparison["missing_enhanced_models"] == [
        "fUSD_model_execution_prob_v2",
        "fUSD_model_revenue_optimized",
        "fUST_model_execution_prob_v2",
        "fUST_model_revenue_optimized",
    ]


def test_follow_stability_and_divergence_checks_handle_missing_market_median_column(
    tmp_path, scheduler_module
):
    db_path = tmp_path / "lending_history.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE virtual_orders (
                predicted_rate REAL,
                period INTEGER,
                validated_at TEXT,
                status TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO virtual_orders VALUES (10.0, 30, '2026-04-13', 'EXECUTED')
            """
        )
        conn.commit()
    finally:
        conn.close()

    scheduler = scheduler_module.RetrainingScheduler(
        db_path=str(db_path),
        production_model_dir=str(tmp_path / "models"),
        backup_dir=str(tmp_path / "backup"),
        log_dir=str(tmp_path / "logs"),
    )

    metrics = scheduler._get_follow_stability_metrics()
    divergence = scheduler._check_market_divergence_trigger()

    assert metrics == {
        "samples": 0,
        "follow_mae": 0.0,
        "follow_mae_ratio": 0.0,
        "direction_match_rate": 0.0,
        "p120_samples": 0,
        "p120_step_p95": 0.0,
    }
    assert divergence is False
