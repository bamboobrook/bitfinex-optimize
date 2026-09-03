import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_predictor_rank6 import _make_prediction


def _install_predictor_import_stubs():
    def _has_real_module(name):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "__file__", None):
            return True
        if module is not None:
            sys.modules.pop(name, None)
        try:
            __import__(name)
            return True
        except ImportError:
            return False

    if not _has_real_module("xgboost"):
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
        sys.modules["xgboost"] = xgb

    if not _has_real_module("lightgbm"):
        lgb = types.ModuleType("lightgbm")

        class LightGBMBooster:
            def __init__(self, *args, **kwargs):
                pass

            def predict(self, *args, **kwargs):
                return []

        lgb.Booster = LightGBMBooster
        sys.modules["lightgbm"] = lgb

    if not _has_real_module("catboost"):
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
        cat.Pool = object
        sys.modules["catboost"] = cat

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


def _write_complete_currency_artifacts(model_dir: Path, currency: str, marker: str):
    for model_type in [
        "model_execution_prob",
        "model_conservative",
        "model_aggressive",
        "model_balanced",
        "model_execution_prob_v2",
        "model_revenue_optimized",
    ]:
        prefix = f"{currency}_{model_type}"
        (model_dir / f"{prefix}_meta.json").write_text(
            json.dumps({
                "weights": {"cat": 1.0},
                "feature_cols": ["signal"],
                "task_type": "classification" if "execution_prob" in model_type else "regression",
                "marker": marker,
            }),
            encoding="utf-8",
        )
        (model_dir / f"{prefix}_cat.cbm").write_text(marker, encoding="utf-8")


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


def test_retraining_main_returns_zero_when_no_retraining_needed(tmp_path):
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
                "        cls.should_retrain = lambda self: (False, None)",
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
        [sys.executable, "-m", "ml_engine.retraining_scheduler"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env=env,
    )

    assert result.returncode == 0


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


def test_model_performance_approves_fusd_without_fust(tmp_path, scheduler_module, monkeypatch):
    scheduler = scheduler_module.RetrainingScheduler(
        production_model_dir=str(tmp_path / "models"),
        backup_dir=str(tmp_path / "backup"),
        log_dir=str(tmp_path / "logs"),
    )
    validation = pd.DataFrame({
        "actual_execution_binary": [0.0, 1.0] * 60,
        "revenue_optimized_target": [1.0] * 120,
    })
    monkeypatch.setattr(
        scheduler,
        "_prepare_champion_validation_data",
        lambda days=7, warmup_days=21: {"fUSD": validation, "fUST": validation},
    )
    old_eval = {
        "overall_score": 0.64,
        "currency_scores": {"fUSD": 0.63, "fUST": 0.64},
        "metrics": {
            "fUSD": {
                "model_execution_prob_v2_eligible_samples": 60,
                "model_execution_prob_v2_auc": 0.71,
                "model_execution_prob_v2_brier": 0.22,
                "model_revenue_optimized_eligible_samples": 60,
                "model_revenue_optimized_mae": 0.42,
            },
            "fUST": {
                "model_execution_prob_v2_eligible_samples": 60,
                "model_execution_prob_v2_auc": 0.63,
                "model_execution_prob_v2_brier": 0.27,
                "model_revenue_optimized_eligible_samples": 60,
                "model_revenue_optimized_mae": 0.50,
            },
        },
    }
    new_eval = {
        "overall_score": 0.65,
        "currency_scores": {"fUSD": 0.65, "fUST": 0.65},
        "metrics": {
            "fUSD": {
                "model_execution_prob_v2_eligible_samples": 60,
                "model_execution_prob_v2_auc": 0.704,
                "model_execution_prob_v2_brier": 0.207,
                "model_revenue_optimized_eligible_samples": 60,
                "model_revenue_optimized_mae": 0.43,
            },
            "fUST": {
                "model_execution_prob_v2_eligible_samples": 60,
                "model_execution_prob_v2_auc": 0.629,
                "model_execution_prob_v2_brier": 0.289,
                "model_revenue_optimized_eligible_samples": 60,
                "model_revenue_optimized_mae": 0.574,
            },
        },
    }
    monkeypatch.setattr(
        scheduler,
        "_evaluate_model_dir_on_validation",
        lambda model_dir, val_data: new_eval if model_dir == "new" else old_eval,
    )
    monkeypatch.setattr(scheduler, "_sanity_check_new_models", lambda *args: True)
    monkeypatch.setattr(scheduler, "_evaluate_follow_and_stability", lambda days=7: (True, {}))

    comparison = {"checks": {}, "metrics": {}}
    approved = scheduler._compare_model_performance(
        "old", "new", comparison, ["fUSD", "fUST"]
    )

    assert approved is True
    assert comparison["approved_currencies"] == ["fUSD"]
    assert "fUST" in comparison["rejected_currencies"]
    assert comparison["checks"]["performance"] == "partial"


def test_partial_currency_deployment_preserves_rejected_currency(
    tmp_path, scheduler_module, monkeypatch
):
    production = tmp_path / "models"
    challenger = tmp_path / "challenger"
    production.mkdir()
    challenger.mkdir()
    for currency in ["fUSD", "fUST"]:
        _write_complete_currency_artifacts(production, currency, "old")
        _write_complete_currency_artifacts(challenger, currency, "new")

    scheduler = scheduler_module.RetrainingScheduler(
        production_model_dir=str(production),
        backup_dir=str(tmp_path / "backup"),
        log_dir=str(tmp_path / "logs"),
    )
    monkeypatch.setattr(scheduler, "_sanity_check_new_models", lambda *args: True)

    assert scheduler.deploy_new_models(str(challenger), ["fUSD"]) is True
    assert (production / "fUSD_model_balanced_cat.cbm").read_text() == "new"
    assert (production / "fUST_model_balanced_cat.cbm").read_text() == "old"


def test_live_rollback_restores_only_failed_currency_and_logs_source(
    tmp_path, scheduler_module, monkeypatch
):
    backup = tmp_path / "backup" / "production_20260823_003241"
    backup.mkdir(parents=True)
    scheduler = scheduler_module.RetrainingScheduler(
        production_model_dir=str(tmp_path / "models"),
        backup_dir=str(tmp_path / "backup"),
        log_dir=str(tmp_path),
    )
    gate = {
        "currency": "fUSD",
        "rollback_required": True,
        "deployed_at": "2026-08-23 00:32:41",
        "current_model_version": "model_20260823_003241",
        "previous_model_version": "model_20260819_223827",
    }
    deployed = []
    logged = []
    monkeypatch.setattr(
        scheduler, "evaluate_live_model_gate", lambda currency: gate
    )
    monkeypatch.setattr(
        scheduler,
        "deploy_new_models",
        lambda model_dir, currencies: deployed.append((model_dir, currencies)) or True,
    )
    monkeypatch.setattr(
        scheduler, "_currency_artifact_digest", lambda *args: "unchanged"
    )
    monkeypatch.setattr(
        scheduler, "log_retraining_event", lambda **kwargs: logged.append(kwargs)
    )

    outcome = scheduler.rollback_live_model("fUSD")

    assert outcome == "rolled_back"
    assert deployed == [(str(backup), ["fUSD"])]
    assert logged[0]["outcome"] == "rolled_back"
    assert logged[0]["source_model_version"] == "model_20260819_223827"


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


def test_compare_model_performance_does_not_reject_better_model_only_due_to_live_follow_gate(
    tmp_path, scheduler_module, monkeypatch
):
    scheduler = scheduler_module.RetrainingScheduler(
        production_model_dir=str(tmp_path / "models"),
        backup_dir=str(tmp_path / "backup"),
        log_dir=str(tmp_path / "logs"),
    )

    monkeypatch.setattr(
        scheduler,
        "_prepare_champion_validation_data",
        lambda days=7, warmup_days=21: {"fUSD": list(range(250))},
    )

    old_eval = {
        "overall_score": 0.60,
        "currency_scores": {"fUSD": 0.60},
        "metrics": {"fUSD": {"model_balanced_mae": 1.08}},
    }
    new_eval = {
        "overall_score": 0.75,
        "currency_scores": {"fUSD": 0.75},
        "metrics": {"fUSD": {"model_balanced_mae": 0.48}},
    }

    monkeypatch.setattr(
        scheduler,
        "_evaluate_model_dir_on_validation",
        lambda model_dir, val_data: new_eval if model_dir == "new-models" else old_eval,
    )
    monkeypatch.setattr(scheduler, "_sanity_check_new_models", lambda model_dir: True)
    monkeypatch.setattr(
        scheduler,
        "_evaluate_follow_and_stability",
        lambda days=7: (
            False,
            {
                "follow_mae_7d": 1.71,
                "follow_mae_ratio_7d": 0.289,
                "direction_match_rate_7d": 0.48,
                "p120_step_p95_7d": 0.20,
            },
        ),
    )

    comparison = {"checks": {}, "metrics": {}}
    is_better = scheduler._compare_model_performance(
        "old-models",
        "new-models",
        comparison,
    )

    assert is_better is True
    assert comparison["checks"]["performance"] == "passed"
    assert comparison["metrics"]["p120_step_p95_7d"] == 0.20


def test_compare_model_performance_rejects_degraded_enhanced_model(
    tmp_path, scheduler_module, monkeypatch
):
    scheduler = scheduler_module.RetrainingScheduler(
        production_model_dir=str(tmp_path / "models"),
        backup_dir=str(tmp_path / "backup"),
        log_dir=str(tmp_path / "logs"),
    )
    monkeypatch.setattr(
        scheduler,
        "_prepare_champion_validation_data",
        lambda days=7, warmup_days=21: {"fUSD": list(range(250))},
    )
    old_eval = {
        "overall_score": 0.70,
        "currency_scores": {"fUSD": 0.70},
        "metrics": {
            "fUSD": {
                "model_execution_prob_v2_eligible_samples": 60,
                "model_execution_prob_v2_auc": 0.75,
                "model_execution_prob_v2_brier": 0.18,
                "model_revenue_optimized_eligible_samples": 60,
                "model_revenue_optimized_mae": 0.50,
            }
        },
    }
    new_eval = {
        "overall_score": 0.71,
        "currency_scores": {"fUSD": 0.71},
        "metrics": {
            "fUSD": {
                "model_execution_prob_v2_eligible_samples": 60,
                "model_execution_prob_v2_auc": 0.70,
                "model_execution_prob_v2_brier": 0.22,
                "model_revenue_optimized_eligible_samples": 60,
                "model_revenue_optimized_mae": 0.60,
            }
        },
    }
    monkeypatch.setattr(
        scheduler,
        "_evaluate_model_dir_on_validation",
        lambda model_dir, val_data: (
            new_eval if model_dir == "new-models" else old_eval
        ),
    )
    monkeypatch.setattr(
        scheduler,
        "_sanity_check_new_models",
        lambda model_dir: True,
    )
    monkeypatch.setattr(
        scheduler,
        "_evaluate_follow_and_stability",
        lambda days=7: (True, {}),
    )

    comparison = {"checks": {}, "metrics": {}}
    is_better = scheduler._compare_model_performance(
        "old-models",
        "new-models",
        comparison,
    )

    assert is_better is False
    assert comparison["checks"]["enhanced_performance"] == "degraded"


def test_compare_model_performance_enforces_absolute_enhanced_floors_without_champion_metrics(
    tmp_path, scheduler_module, monkeypatch
):
    scheduler = scheduler_module.RetrainingScheduler(
        production_model_dir=str(tmp_path / "models"),
        backup_dir=str(tmp_path / "backup"),
        log_dir=str(tmp_path / "logs"),
    )
    monkeypatch.setattr(
        scheduler,
        "_prepare_champion_validation_data",
        lambda days=7, warmup_days=21: {"fUSD": list(range(250))},
    )
    old_eval = {
        "overall_score": 0.70,
        "currency_scores": {"fUSD": 0.70},
        "metrics": {"fUSD": {}},
    }
    new_eval = {
        "overall_score": 0.80,
        "currency_scores": {"fUSD": 0.80},
        "metrics": {
            "fUSD": {
                "model_execution_prob_v2_eligible_samples": 60,
                "model_execution_prob_v2_auc": 0.0,
                "model_execution_prob_v2_brier": 1.0,
                "model_revenue_optimized_eligible_samples": 60,
                "model_revenue_optimized_mae": 49.0,
            }
        },
    }
    monkeypatch.setattr(
        scheduler,
        "_evaluate_model_dir_on_validation",
        lambda model_dir, val_data: (
            new_eval if model_dir == "new-models" else old_eval
        ),
    )
    monkeypatch.setattr(
        scheduler,
        "_sanity_check_new_models",
        lambda model_dir: True,
    )
    monkeypatch.setattr(
        scheduler,
        "_evaluate_follow_and_stability",
        lambda days=7: (True, {}),
    )

    comparison = {"checks": {}, "metrics": {}}
    is_better = scheduler._compare_model_performance(
        "old-models",
        "new-models",
        comparison,
    )

    assert is_better is False
    assert comparison["checks"]["enhanced_performance"] == "degraded"


def test_compare_model_performance_rejects_non_finite_enhanced_metrics(
    tmp_path, scheduler_module, monkeypatch
):
    scheduler = scheduler_module.RetrainingScheduler(
        production_model_dir=str(tmp_path / "models"),
        backup_dir=str(tmp_path / "backup"),
        log_dir=str(tmp_path / "logs"),
    )
    monkeypatch.setattr(
        scheduler,
        "_prepare_champion_validation_data",
        lambda days=7, warmup_days=21: {"fUSD": list(range(250))},
    )
    old_eval = {
        "overall_score": 0.70,
        "currency_scores": {"fUSD": 0.70},
        "metrics": {"fUSD": {}},
    }
    new_eval = {
        "overall_score": 0.80,
        "currency_scores": {"fUSD": 0.80},
        "metrics": {
            "fUSD": {
                "model_execution_prob_v2_eligible_samples": 60,
                "model_execution_prob_v2_auc": float("nan"),
                "model_execution_prob_v2_brier": float("nan"),
                "model_revenue_optimized_eligible_samples": 60,
                "model_revenue_optimized_mae": float("inf"),
            }
        },
    }
    monkeypatch.setattr(
        scheduler,
        "_evaluate_model_dir_on_validation",
        lambda model_dir, val_data: (
            new_eval if model_dir == "new-models" else old_eval
        ),
    )
    monkeypatch.setattr(
        scheduler,
        "_sanity_check_new_models",
        lambda model_dir: True,
    )
    monkeypatch.setattr(
        scheduler,
        "_evaluate_follow_and_stability",
        lambda days=7: (True, {}),
    )

    comparison = {"checks": {}, "metrics": {}}
    is_better = scheduler._compare_model_performance(
        "old-models",
        "new-models",
        comparison,
    )

    assert is_better is False
    assert comparison["checks"]["enhanced_performance"] == "degraded"


def test_small_validation_slice_still_checks_mature_feedback_labels(
    tmp_path, scheduler_module, monkeypatch
):
    scheduler = scheduler_module.RetrainingScheduler(
        production_model_dir=str(tmp_path / "models"),
        backup_dir=str(tmp_path / "backup"),
        log_dir=str(tmp_path / "logs"),
    )
    validation = pd.DataFrame(
        {
            "actual_execution_binary": [1.0] * 50 + [np.nan] * 149,
            "revenue_optimized_target": [8.0] * 50 + [np.nan] * 149,
        }
    )
    monkeypatch.setattr(
        scheduler,
        "_prepare_champion_validation_data",
        lambda days=7, warmup_days=21: {"fUSD": validation},
    )
    old_eval = {
        "overall_score": 0.70,
        "currency_scores": {"fUSD": 0.70},
        "metrics": {"fUSD": {}},
    }
    new_eval = {
        "overall_score": 0.80,
        "currency_scores": {"fUSD": 0.80},
        "metrics": {
            "fUSD": {
                "model_execution_prob_v2_eligible_samples": 50,
                "model_execution_prob_v2_auc": 0.0,
                "model_execution_prob_v2_brier": 1.0,
                "model_revenue_optimized_eligible_samples": 50,
                "model_revenue_optimized_mae": 49.0,
            }
        },
    }
    evaluation_calls = []

    def _evaluate(model_dir, val_data):
        evaluation_calls.append(model_dir)
        return new_eval if model_dir == "new-models" else old_eval

    monkeypatch.setattr(
        scheduler,
        "_evaluate_model_dir_on_validation",
        _evaluate,
    )
    monkeypatch.setattr(
        scheduler,
        "_sanity_check_new_models",
        lambda model_dir: True,
    )
    monkeypatch.setattr(
        scheduler,
        "_evaluate_follow_and_stability",
        lambda days=7: (True, {}),
    )

    comparison = {"checks": {}, "metrics": {}}
    is_better = scheduler._compare_model_performance(
        "old-models",
        "new-models",
        comparison,
    )

    assert evaluation_calls == ["old-models", "new-models"]
    assert is_better is False


def test_model_artifact_check_requires_every_positive_weight_component(
    tmp_path, scheduler_module
):
    model_prefix = "fUSD_model_execution_prob_v2"
    (tmp_path / f"{model_prefix}_meta.json").write_text(
        json.dumps(
            {
                "weights": {"xgb": 0.7, "lgb": 0.0, "cat": 0.3},
                "feature_cols": ["signal"],
                "task_type": "classification",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / f"{model_prefix}_cat.cbm").write_text(
        "cat",
        encoding="utf-8",
    )

    complete, reason = scheduler_module.RetrainingScheduler._check_model_artifact_set(
        str(tmp_path),
        model_prefix,
    )
    assert complete is False
    assert "xgb" in reason

    (tmp_path / f"{model_prefix}_xgb.json").write_text(
        "xgb",
        encoding="utf-8",
    )
    complete, reason = scheduler_module.RetrainingScheduler._check_model_artifact_set(
        str(tmp_path),
        model_prefix,
    )
    assert complete is True
    assert reason == ""


def test_validation_evaluator_scores_v2_and_revenue_models(
    tmp_path, scheduler_module, monkeypatch
):
    model_types = [
        "model_execution_prob",
        "model_conservative",
        "model_aggressive",
        "model_balanced",
        "model_execution_prob_v2",
        "model_revenue_optimized",
    ]

    class _Predictor:
        def __init__(self, model_dir, max_workers):
            self.meta_info = {
                "fUSD": {
                    model_type: {"feature_cols": ["signal"]}
                    for model_type in model_types
                }
            }

        def predict_with_ensemble(self, X, currency, model_type):
            if "execution_prob" in model_type:
                return X["signal"].to_numpy()
            return (5.0 + X["signal"]).to_numpy()

    monkeypatch.setattr(scheduler_module, "EnsemblePredictor", _Predictor)
    scheduler = scheduler_module.RetrainingScheduler(
        production_model_dir=str(tmp_path / "models"),
        backup_dir=str(tmp_path / "backup"),
        log_dir=str(tmp_path / "logs"),
    )
    signal = np.linspace(0.01, 0.99, 50)
    validation = pd.DataFrame(
        {
            "signal": signal,
            "future_execution_prob": (signal >= 0.5).astype(float),
            "future_conservative": 5.0 + signal,
            "future_aggressive": 5.0 + signal,
            "future_balanced": 5.0 + signal,
            "actual_execution_binary": (signal >= 0.5).astype(float),
            "revenue_optimized_target": 5.0 + signal,
        }
    )

    result = scheduler._evaluate_model_dir_on_validation(
        "models",
        {"fUSD": validation},
    )
    metrics = result["metrics"]["fUSD"]

    assert metrics["model_execution_prob_v2_eligible_samples"] == 50
    assert metrics["model_execution_prob_v2_auc"] == pytest.approx(1.0)
    assert metrics["model_execution_prob_v2_brier"] < 0.1
    assert metrics["model_revenue_optimized_eligible_samples"] == 50
    assert metrics["model_revenue_optimized_mae"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("labels", "scores", "expected"),
    [
        ([0, 1], [0.5, 0.5], 0.5),
        ([0, 1, 0, 1], [0.2, 0.2, 0.8, 0.8], 0.5),
        ([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5], 0.5),
        ([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 1.0),
    ],
)
def test_safe_auc_uses_average_ranks_for_tied_scores(
    scheduler_module, labels, scores, expected
):
    assert scheduler_module.RetrainingScheduler._safe_auc(labels, scores) == pytest.approx(
        expected
    )


def test_get_production_model_age_days_uses_newest_meta_file(tmp_path, scheduler_module):
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    stale = model_dir / "old_meta.json"
    fresh = model_dir / "new_meta.json"
    stale.write_text("{}", encoding="utf-8")
    fresh.write_text("{}", encoding="utf-8")

    now_ts = 1_700_000_000
    os.utime(stale, (now_ts - 9 * 86400, now_ts - 9 * 86400))
    os.utime(fresh, (now_ts - 2 * 86400, now_ts - 2 * 86400))

    scheduler = scheduler_module.RetrainingScheduler(
        production_model_dir=str(model_dir),
        backup_dir=str(tmp_path / "backup"),
        log_dir=str(tmp_path / "logs"),
    )

    class _FrozenDatetime:
        @classmethod
        def now(cls):
            return __import__("datetime").datetime.fromtimestamp(now_ts)

        @classmethod
        def fromtimestamp(cls, ts):
            return __import__("datetime").datetime.fromtimestamp(ts)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(scheduler_module, "datetime", _FrozenDatetime)
    try:
        assert scheduler._get_production_model_age_days() == 2
    finally:
        monkeypatch.undo()


def test_production_deployment_time_prefers_successful_history_event(
    tmp_path, scheduler_module
):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    meta = model_dir / "fUSD_model_balanced_meta.json"
    meta.write_text("{}", encoding="utf-8")
    os.utime(meta, (1_700_000_000, 1_700_000_000))

    history = [
        {
            "timestamp": "2026-04-01 10:05:00",
            "retrained": True,
            "deployed": True,
        },
        {
            "timestamp": "2026-04-02 10:05:00",
            "retrained": True,
            "deployed": False,
        },
    ]
    (tmp_path / "retraining_history.json").write_text(
        json.dumps(history),
        encoding="utf-8",
    )

    scheduler = scheduler_module.RetrainingScheduler(
        production_model_dir=str(model_dir),
        backup_dir=str(tmp_path / "backup"),
        log_dir=str(tmp_path),
    )

    assert scheduler._get_production_model_deployed_at() == (
        __import__("datetime").datetime(2026, 4, 1, 10, 5, 0)
    )


def test_production_deployment_time_is_tracked_per_currency(
    tmp_path, scheduler_module
):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    history = [
        {
            "timestamp": "2026-04-01 10:05:00",
            "retrained": True,
            "deployed": True,
        },
        {
            "timestamp": "2026-04-10 12:00:00",
            "retrained": True,
            "deployed": True,
            "deployed_currencies": ["fUSD"],
        },
    ]
    (tmp_path / "retraining_history.json").write_text(
        json.dumps(history), encoding="utf-8"
    )
    scheduler = scheduler_module.RetrainingScheduler(
        production_model_dir=str(model_dir),
        backup_dir=str(tmp_path / "backup"),
        log_dir=str(tmp_path),
    )

    assert scheduler._get_production_model_deployed_at("fUSD") == (
        __import__("datetime").datetime(2026, 4, 10, 12, 0, 0)
    )
    assert scheduler._get_production_model_deployed_at("fUST") == (
        __import__("datetime").datetime(2026, 4, 1, 10, 5, 0)
    )


def test_live_model_gate_waits_for_40_then_requires_rollback(
    tmp_path, scheduler_module
):
    db_path = tmp_path / "live_gate.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE virtual_orders (
                order_id TEXT PRIMARY KEY, currency TEXT, period INTEGER,
                status TEXT, validation_window_hours INTEGER, created_at TEXT,
                model_version TEXT, decision_mode TEXT, update_cycle_id TEXT,
                candidate_id TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE prediction_history (
                id INTEGER PRIMARY KEY, update_cycle_id TEXT, currency TEXT,
                period INTEGER, candidate_id TEXT,
                calibrated_execution_probability REAL
            )
            """
        )
        rows = []
        predictions = []
        for prefix, version, count, executed, created_at, probability in [
            ("old", "model_20260819_223827", 80, 64, "2026-08-20 00:00:00", 0.75),
            ("new", "model_20260823_003241", 40, 10, "2026-08-23 01:00:00", 0.70),
        ]:
            for index in range(count):
                cycle = f"{prefix}_{index}"
                status = "EXECUTED" if index < executed else "FAILED"
                rows.append((
                    cycle, "fUSD", 3, status, 24, created_at, version,
                    "exploit", cycle, cycle,
                ))
                predictions.append((cycle, "fUSD", 3, cycle, probability))
        conn.executemany(
            "INSERT INTO virtual_orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.executemany(
            """
            INSERT INTO prediction_history(
                update_cycle_id, currency, period, candidate_id,
                calibrated_execution_probability
            ) VALUES (?, ?, ?, ?, ?)
            """,
            predictions,
        )

    history = [
        {
            "timestamp": "2026-08-19 22:38:27",
            "deployed": True,
            "deployed_currencies": ["fUSD"],
        },
        {
            "timestamp": "2026-08-23 00:32:41",
            "deployed": True,
            "deployed_currencies": ["fUSD"],
        },
    ]
    (tmp_path / "retraining_history.json").write_text(
        json.dumps(history), encoding="utf-8"
    )
    scheduler = scheduler_module.RetrainingScheduler(
        db_path=str(db_path),
        production_model_dir=str(tmp_path / "models"),
        backup_dir=str(tmp_path / "backup"),
        log_dir=str(tmp_path),
    )
    scheduler.policy = {
        "live_model_gate": {
            "enabled": True,
            "min_current_decided": 40,
            "min_champion_decided": 80,
            "min_execution_rate_drop": 0.15,
            "max_brier": 0.25,
            "max_ece": 0.15,
            "max_observation_age_hours": 100000,
        }
    }

    gate = scheduler.evaluate_live_model_gate("fUSD")

    assert gate["ready"] is True
    assert gate["current"]["decided"] == 40
    assert gate["previous"]["decided"] == 80
    assert gate["rollback_required"] is True
    assert gate["execution_rate_drop"] == pytest.approx(0.55)


def _live_gate_scheduler_with_segments(
    tmp_path,
    scheduler_module,
    old_segments,
    current_segments,
    policy_overrides=None,
):
    db_path = tmp_path / "segmented_live_gate.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE virtual_orders (
                order_id TEXT PRIMARY KEY, currency TEXT, period INTEGER,
                status TEXT, validation_window_hours INTEGER, created_at TEXT,
                model_version TEXT, decision_mode TEXT, update_cycle_id TEXT,
                candidate_id TEXT, realized_terminal_value_net_wait REAL,
                market_median REAL, path_value_score REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE prediction_history (
                id INTEGER PRIMARY KEY, update_cycle_id TEXT, currency TEXT,
                period INTEGER, candidate_id TEXT,
                calibrated_execution_probability REAL,
                probability_calibration_method TEXT
            )
            """
        )
        order_rows = []
        prediction_rows = []
        sequence = 0
        for prefix, version, created_at, segments in [
            ("old", "model_20260819_223827", "2026-08-20 00:00:00", old_segments),
            ("new", "model_20260823_003241", "2026-08-23 01:00:00", current_segments),
        ]:
            for segment in segments:
                decided = int(segment.get("decided", 0))
                pending = int(segment.get("pending", 0))
                executed = int(segment.get("executed", 0))
                premium = float(segment.get("premium", 0.5))
                probability = float(segment.get("probability", 0.7))
                method = segment.get("method", "traditional_fallback")
                statuses = ["EXECUTED"] * executed
                statuses += ["FAILED"] * max(decided - executed, 0)
                statuses += ["PENDING"] * pending
                for status in statuses:
                    sequence += 1
                    cycle = f"{prefix}_{sequence}"
                    realized = 10.0 + premium if status != "PENDING" else None
                    order_rows.append((
                        cycle,
                        "fUSD",
                        int(segment["period"]),
                        status,
                        int(segment["window"]),
                        created_at,
                        version,
                        "exploit",
                        cycle,
                        cycle,
                        realized,
                        10.0,
                        realized,
                    ))
                    prediction_rows.append((
                        cycle,
                        "fUSD",
                        int(segment["period"]),
                        cycle,
                        probability,
                        method,
                    ))
        conn.executemany(
            "INSERT INTO virtual_orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            order_rows,
        )
        conn.executemany(
            """
            INSERT INTO prediction_history(
                update_cycle_id, currency, period, candidate_id,
                calibrated_execution_probability, probability_calibration_method
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            prediction_rows,
        )

    history = [
        {
            "timestamp": "2026-08-19 22:38:27",
            "deployed": True,
            "deployed_currencies": ["fUSD"],
        },
        {
            "timestamp": "2026-08-23 00:32:41",
            "deployed": True,
            "deployed_currencies": ["fUSD"],
        },
    ]
    (tmp_path / "retraining_history.json").write_text(
        json.dumps(history), encoding="utf-8"
    )
    scheduler = scheduler_module.RetrainingScheduler(
        db_path=str(db_path),
        production_model_dir=str(tmp_path / "models"),
        backup_dir=str(tmp_path / "backup"),
        log_dir=str(tmp_path),
    )
    gate_policy = {
        "enabled": True,
        "min_current_decided": 40,
        "min_champion_decided": 80,
        "min_execution_rate_drop": 0.15,
        "max_brier": 0.25,
        "max_ece": 0.15,
        "min_value_samples": 40,
        "min_realized_premium_drop": 0.20,
        "max_observation_age_hours": 100000,
    }
    gate_policy.update(policy_overrides or {})
    scheduler.policy = {"live_model_gate": gate_policy}
    return scheduler


def test_live_model_gate_reweights_champion_to_current_period_window_mix(
    tmp_path, scheduler_module
):
    scheduler = _live_gate_scheduler_with_segments(
        tmp_path,
        scheduler_module,
        old_segments=[
            {"period": 3, "window": 24, "decided": 100, "executed": 100},
            {"period": 90, "window": 72, "decided": 100, "executed": 0},
        ],
        current_segments=[
            {"period": 3, "window": 24, "decided": 10, "executed": 10},
            {"period": 90, "window": 72, "decided": 30, "executed": 15},
        ],
    )

    gate = scheduler.evaluate_live_model_gate("fUSD")

    assert gate["ready"] is True
    assert gate["comparison_mode"] == "current_period_and_window_reweighted"
    assert gate["previous"]["source_execution_rate"] == pytest.approx(0.5)
    assert gate["previous"]["execution_rate"] == pytest.approx(0.25)
    assert gate["champion_segment_coverage"] == pytest.approx(1.0)


def test_live_model_gate_requires_each_configured_window_to_mature(
    tmp_path, scheduler_module
):
    scheduler = _live_gate_scheduler_with_segments(
        tmp_path,
        scheduler_module,
        old_segments=[
            {"period": 3, "window": 24, "decided": 100, "executed": 80},
            {"period": 10, "window": 48, "decided": 100, "executed": 80},
            {"period": 90, "window": 72, "decided": 100, "executed": 80},
        ],
        current_segments=[
            {"period": 3, "window": 24, "decided": 30, "executed": 24},
            {"period": 10, "window": 48, "decided": 9, "executed": 7},
            {"period": 90, "window": 72, "decided": 1, "executed": 1},
        ],
        policy_overrides={
            "required_validation_windows": [24, 48, 72],
            "min_current_decided_by_window": {"24": 10, "48": 10, "72": 10},
            "min_champion_decided_by_window": {"24": 20, "48": 20, "72": 20},
        },
    )

    gate = scheduler.evaluate_live_model_gate("fUSD")

    assert gate["ready"] is False
    assert gate["reason"] == "insufficient_window_maturity"
    assert gate["window_maturity"]["48"]["current_decided"] == 9
    assert gate["window_maturity"]["72"]["current_decided"] == 1


def test_live_model_gate_rolls_back_material_realized_value_drop(
    tmp_path, scheduler_module
):
    scheduler = _live_gate_scheduler_with_segments(
        tmp_path,
        scheduler_module,
        old_segments=[{
            "period": 3, "window": 24, "decided": 80, "executed": 64,
            "premium": 0.5, "probability": 0.8,
        }],
        current_segments=[{
            "period": 3, "window": 24, "decided": 40, "executed": 32,
            "premium": 0.1, "probability": 0.8,
        }],
    )

    gate = scheduler.evaluate_live_model_gate("fUSD")

    assert gate["ready"] is True
    assert gate["execution_rate_drop"] == pytest.approx(0.0)
    assert gate["value_gate"]["realized_premium_drop"] == pytest.approx(0.4)
    assert gate["rollback_required"] is True
    assert gate["rollback_trigger_reasons"] == ["realized_premium_degraded"]


def test_live_model_gate_tracks_v2_identity_maturity_separately(
    tmp_path, scheduler_module
):
    scheduler = _live_gate_scheduler_with_segments(
        tmp_path,
        scheduler_module,
        old_segments=[{
            "period": 3, "window": 24, "decided": 80, "executed": 64,
        }],
        current_segments=[
            {
                "period": 3, "window": 24, "decided": 40, "executed": 32,
                "method": "traditional_fallback",
            },
            {
                "period": 3, "window": 24, "pending": 12,
                "method": "v2_identity_persisted",
            },
        ],
    )

    gate = scheduler.evaluate_live_model_gate("fUSD")
    cohorts = gate["calibration_method_cohorts"]

    assert cohorts["traditional_fallback"]["decided"] == 40
    assert cohorts["v2_identity"]["decided"] == 0
    assert cohorts["v2_identity"]["pending"] == 12
    assert cohorts["v2_identity"]["ready"] is False


def test_partial_deploy_rejection_backoff_is_currency_specific(
    tmp_path, scheduler_module
):
    history = [
        {
            "timestamp": "2026-08-20 00:00:00",
            "deployed": True,
            "deployed_currencies": ["fUSD"],
            "comparison": {"rejected_currencies": {"fUST": ["brier"]}},
        },
        {
            "timestamp": "2026-08-23 00:00:00",
            "deployed": True,
            "deployed_currencies": ["fUSD"],
            "comparison": {"rejected_currencies": {"fUST": ["mae"]}},
        },
    ]
    (tmp_path / "retraining_history.json").write_text(
        json.dumps(history), encoding="utf-8"
    )
    scheduler = scheduler_module.RetrainingScheduler(
        production_model_dir=str(tmp_path / "models"),
        backup_dir=str(tmp_path / "backup"),
        log_dir=str(tmp_path),
    )
    scheduler.policy = {
        "currency_retrain_backoff": {"base_hours": 24, "max_hours": 168}
    }

    assert scheduler._currency_retrain_backoff_until("fUSD") is None
    assert scheduler._currency_retrain_backoff_until("fUST") == (
        __import__("datetime").datetime(2026, 8, 25, 0, 0, 0)
    )


def test_should_retrain_skips_quality_triggers_during_post_deploy_grace(
    tmp_path, scheduler_module, monkeypatch
):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    meta = model_dir / "fUSD_model_balanced_meta.json"
    meta.write_text("{}", encoding="utf-8")

    now_ts = 1_700_000_000
    os.utime(meta, (now_ts - 3600, now_ts - 3600))

    scheduler = scheduler_module.RetrainingScheduler(
        production_model_dir=str(model_dir),
        backup_dir=str(tmp_path / "backup"),
        log_dir=str(tmp_path / "logs"),
    )

    class _FrozenDatetime:
        @classmethod
        def now(cls):
            return __import__("datetime").datetime.fromtimestamp(now_ts)

        @classmethod
        def fromtimestamp(cls, ts):
            return __import__("datetime").datetime.fromtimestamp(ts)

        @classmethod
        def strptime(cls, value, fmt):
            return __import__("datetime").datetime.strptime(value, fmt)

    monkeypatch.setattr(scheduler_module, "datetime", _FrozenDatetime)
    monkeypatch.setattr(scheduler, "count_new_execution_results", lambda since_date: 0)
    monkeypatch.setattr(scheduler, "_count_orders_since", lambda since_dt, currency=None: 100)
    monkeypatch.setattr(
        scheduler,
        "_count_long_window_orders_since",
            lambda since_dt, currency=None: 0,
    )
    monkeypatch.setattr(scheduler, "get_recent_execution_rate", lambda days=7, since_dt=None, currency=None: 0.51)
    monkeypatch.setattr(
        scheduler,
        "get_per_period_execution_anomalies",
            lambda days=7, since_dt=None, currency=None: [
            {
                "currency": "fUST",
                "period": 120,
                "exec_rate": 0.0,
                "total": 44,
                "severity": "critical",
            }
        ],
    )
    monkeypatch.setattr(
        scheduler,
        "_get_follow_stability_metrics",
            lambda days=7, since_dt=None, currency=None: {
            "samples": 100,
            "follow_mae": 1.7,
            "follow_mae_ratio": 0.28,
            "direction_match_rate": 0.48,
            "p120_samples": 40,
            "p120_step_p95": 0.20,
        },
    )
    monkeypatch.setattr(scheduler, "_check_zero_liquidity_anomaly", lambda since_dt=None, currency=None: [])
    monkeypatch.setattr(scheduler, "_check_market_divergence_trigger", lambda since_dt=None, currency=None: False)

    should_retrain, reason = scheduler.should_retrain()

    assert should_retrain is False
    assert reason is None


def test_post_deploy_order_count_uses_created_at_for_model_attribution(
    tmp_path, scheduler_module
):
    db_path = tmp_path / "orders.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE virtual_orders (
                order_timestamp TEXT,
                created_at TEXT,
                status TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO virtual_orders VALUES (?, ?, ?)",
            [
                ("2026-04-01 09:50:00", "2026-04-01 10:05:00", "EXECUTED"),
                ("2026-04-01 10:10:00", "2026-04-01 09:55:00", "FAILED"),
            ],
        )

    scheduler = scheduler_module.RetrainingScheduler(
        db_path=str(db_path),
        production_model_dir=str(tmp_path / "models"),
        backup_dir=str(tmp_path / "backup"),
        log_dir=str(tmp_path / "logs"),
    )
    deployed_at = __import__("datetime").datetime(2026, 4, 1, 10, 0, 0)

    assert scheduler._count_orders_since(deployed_at) == 1


def test_zero_liquidity_requires_resolved_samples_and_zero_executions(
    tmp_path, scheduler_module, monkeypatch
):
    db_path = tmp_path / "orders.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE virtual_orders (
                currency TEXT,
                period INTEGER,
                created_at TEXT,
                status TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO virtual_orders VALUES (?, ?, ?, ?)",
            [
                ("fUST", 30, f"2026-04-01 0{hour}:00:00", "FAILED")
                for hour in range(4)
            ] + [
                ("fUST", 60, f"2026-04-01 0{hour}:00:00", "PENDING")
                for hour in range(6)
            ] + [
                ("fUST", 90, f"2026-04-01 0{hour}:00:00", "EXPIRED")
                for hour in range(6)
            ],
        )

    scheduler = scheduler_module.RetrainingScheduler(
        db_path=str(db_path),
        production_model_dir=str(tmp_path / "models"),
        backup_dir=str(tmp_path / "backup"),
        log_dir=str(tmp_path / "logs"),
    )
    scheduler.policy = {"retrain_trigger": {"zero_liq_min_decided_orders": 5}}
    since_dt = __import__("datetime").datetime(2026, 4, 1, 0, 0, 0)

    class _FrozenDatetime:
        @classmethod
        def now(cls):
            return __import__("datetime").datetime(2026, 4, 2, 0, 0, 0)

    monkeypatch.setattr(scheduler_module, "datetime", _FrozenDatetime)

    assert scheduler._check_zero_liquidity_anomaly(since_dt=since_dt) == []

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO virtual_orders VALUES (?, ?, ?, ?)",
            ("fUST", 30, "2026-04-01 04:00:00", "FAILED"),
        )

    anomalies = scheduler._check_zero_liquidity_anomaly(since_dt=since_dt)
    assert [(row[0], row[1], row[3], row[4]) for row in anomalies] == [
        ("fUST", 30, 5, 0)
    ]

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO virtual_orders VALUES (?, ?, ?, ?)",
            ("fUST", 30, "2026-04-01 05:00:00", "EXECUTED"),
        )

    assert scheduler._check_zero_liquidity_anomaly(since_dt=since_dt) == []


def test_should_retrain_uses_post_deploy_window_for_quality_triggers(
    tmp_path, scheduler_module, monkeypatch
):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    meta = model_dir / "fUSD_model_balanced_meta.json"
    meta.write_text("{}", encoding="utf-8")

    now_ts = 1_700_000_000
    deployed_ts = now_ts - 18 * 3600
    os.utime(meta, (deployed_ts, deployed_ts))

    scheduler = scheduler_module.RetrainingScheduler(
        production_model_dir=str(model_dir),
        backup_dir=str(tmp_path / "backup"),
        log_dir=str(tmp_path / "logs"),
    )

    class _FrozenDatetime:
        @classmethod
        def now(cls):
            return __import__("datetime").datetime.fromtimestamp(now_ts)

        @classmethod
        def fromtimestamp(cls, ts):
            return __import__("datetime").datetime.fromtimestamp(ts)

        @classmethod
        def strptime(cls, value, fmt):
            return __import__("datetime").datetime.strptime(value, fmt)

    captured = {}

    monkeypatch.setattr(scheduler_module, "datetime", _FrozenDatetime)
    monkeypatch.setattr(scheduler, "count_new_execution_results", lambda since_date: 0)

    def _recent_exec(days=7, since_dt=None, currency=None):
        captured["exec_since"] = since_dt
        return 0.51

    def _period_anomalies(days=7, since_dt=None, currency=None):
        captured["period_since"] = since_dt
        return [
            {
                "currency": "fUST",
                "period": 120,
                "exec_rate": 0.0,
                "total": 8,
                "severity": "critical",
            }
        ]

    monkeypatch.setattr(scheduler, "get_recent_execution_rate", _recent_exec)
    monkeypatch.setattr(scheduler, "get_per_period_execution_anomalies", _period_anomalies)
    monkeypatch.setattr(
        scheduler,
        "_get_follow_stability_metrics",
        lambda days=7, since_dt=None, currency=None: {
            "samples": 20,
            "follow_mae": 0.0,
            "follow_mae_ratio": 0.0,
            "direction_match_rate": 0.0,
            "p120_samples": 0,
            "p120_step_p95": 0.0,
        },
    )
    monkeypatch.setattr(scheduler, "_check_zero_liquidity_anomaly", lambda since_dt=None, currency=None: [])
    monkeypatch.setattr(scheduler, "_check_market_divergence_trigger", lambda since_dt=None, currency=None: False)
    monkeypatch.setattr(
        scheduler,
        "_count_orders_since",
        lambda since_dt, currency=None: 60,
    )
    monkeypatch.setattr(
        scheduler,
        "_count_long_window_orders_since",
        lambda since_dt, currency=None: 0,
    )

    should_retrain, reason = scheduler.should_retrain()

    assert should_retrain is True
    assert "单period成交率极低" in reason
    assert captured["exec_since"] == __import__("datetime").datetime.fromtimestamp(deployed_ts)
    assert captured["period_since"] == __import__("datetime").datetime.fromtimestamp(deployed_ts)


def test_should_retrain_defers_high_execution_until_full_validation_window(
    tmp_path, scheduler_module, monkeypatch, capsys
):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    meta = model_dir / "fUSD_model_balanced_meta.json"
    meta.write_text("{}", encoding="utf-8")

    now_ts = 1_700_000_000
    deployed_ts = now_ts - 36 * 3600
    os.utime(meta, (deployed_ts, deployed_ts))

    scheduler = scheduler_module.RetrainingScheduler(
        production_model_dir=str(model_dir),
        backup_dir=str(tmp_path / "backup"),
        log_dir=str(tmp_path / "logs"),
    )

    class _FrozenDatetime:
        @classmethod
        def now(cls):
            return __import__("datetime").datetime.fromtimestamp(now_ts)

        @classmethod
        def fromtimestamp(cls, ts):
            return __import__("datetime").datetime.fromtimestamp(ts)

        @classmethod
        def strptime(cls, value, fmt):
            return __import__("datetime").datetime.strptime(value, fmt)

    high_anomaly = {
        "currency": "fUSD",
        "period": 7,
        "exec_rate": 1.0,
        "total": 10,
        "severity": "critical",
    }
    monkeypatch.setattr(scheduler_module, "datetime", _FrozenDatetime)
    monkeypatch.setattr(scheduler, "count_new_execution_results", lambda since_date: 0)
    monkeypatch.setattr(scheduler, "_count_orders_since", lambda since_dt, currency=None: 100)
    monkeypatch.setattr(
        scheduler,
        "_count_long_window_orders_since",
            lambda since_dt, currency=None: 0,
    )
    monkeypatch.setattr(scheduler, "get_recent_execution_rate", lambda days=7, since_dt=None, currency=None: 0.90)
    monkeypatch.setattr(
        scheduler,
        "get_per_period_execution_anomalies",
            lambda days=7, since_dt=None, currency=None: [high_anomaly],
    )
    monkeypatch.setattr(
        scheduler,
        "_get_follow_stability_metrics",
            lambda days=7, since_dt=None, currency=None: {
            "samples": 100,
            "follow_mae": 0.0,
            "follow_mae_ratio": 0.0,
            "direction_match_rate": 0.50,
            "p120_samples": 0,
            "p120_step_p95": 0.0,
        },
    )
    monkeypatch.setattr(scheduler, "_check_zero_liquidity_anomaly", lambda since_dt=None, currency=None: [])
    monkeypatch.setattr(scheduler, "_check_market_divergence_trigger", lambda since_dt=None, currency=None: False)

    should_retrain, reason = scheduler.should_retrain()
    output = capsys.readouterr().out

    assert should_retrain is False
    assert reason is None
    assert "全局成交率: 90.00% (偏高，等待 72h/120条已决/20条72h结果后再判断)" in output
    assert "分组异常: 1条高成交率信号等待成熟" in output
    assert "全局成交率: 90.00% (正常范围" not in output


def test_should_retrain_allows_high_execution_after_full_validation_window(
    tmp_path, scheduler_module, monkeypatch
):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    meta = model_dir / "fUSD_model_balanced_meta.json"
    meta.write_text("{}", encoding="utf-8")

    now_ts = 1_700_000_000
    deployed_ts = now_ts - 80 * 3600
    os.utime(meta, (deployed_ts, deployed_ts))

    scheduler = scheduler_module.RetrainingScheduler(
        production_model_dir=str(model_dir),
        backup_dir=str(tmp_path / "backup"),
        log_dir=str(tmp_path / "logs"),
    )

    class _FrozenDatetime:
        @classmethod
        def now(cls):
            return __import__("datetime").datetime.fromtimestamp(now_ts)

        @classmethod
        def fromtimestamp(cls, ts):
            return __import__("datetime").datetime.fromtimestamp(ts)

        @classmethod
        def strptime(cls, value, fmt):
            return __import__("datetime").datetime.strptime(value, fmt)

    monkeypatch.setattr(scheduler_module, "datetime", _FrozenDatetime)
    monkeypatch.setattr(scheduler, "count_new_execution_results", lambda since_date: 0)
    monkeypatch.setattr(scheduler, "_count_orders_since", lambda since_dt, currency=None: 160)
    monkeypatch.setattr(
        scheduler,
        "_count_long_window_orders_since",
            lambda since_dt, currency=None: 25,
    )
    monkeypatch.setattr(scheduler, "get_recent_execution_rate", lambda days=7, since_dt=None, currency=None: 0.90)
    monkeypatch.setattr(scheduler, "get_per_period_execution_anomalies", lambda days=7, since_dt=None, currency=None: [])
    monkeypatch.setattr(
        scheduler,
        "_get_follow_stability_metrics",
            lambda days=7, since_dt=None, currency=None: {
            "samples": 160,
            "follow_mae": 0.0,
            "follow_mae_ratio": 0.0,
            "direction_match_rate": 0.50,
            "p120_samples": 0,
            "p120_step_p95": 0.0,
        },
    )

    should_retrain, reason = scheduler.should_retrain()

    assert should_retrain is True
    assert "全局成交率过高" in reason


def test_log_retraining_event_keeps_multiple_same_day_entries(tmp_path, scheduler_module, monkeypatch):
    scheduler = scheduler_module.RetrainingScheduler(
        production_model_dir=str(tmp_path / "models"),
        backup_dir=str(tmp_path / "backup"),
        log_dir=str(tmp_path / "logs"),
    )

    times = iter(
        [
            __import__("datetime").datetime(2026, 4, 23, 10, 0, 0),
            __import__("datetime").datetime(2026, 4, 23, 10, 0, 0),
            __import__("datetime").datetime(2026, 4, 23, 12, 0, 0),
            __import__("datetime").datetime(2026, 4, 23, 12, 0, 0),
        ]
    )

    class _FrozenDatetime:
        @classmethod
        def now(cls):
            return next(times)

        @classmethod
        def fromtimestamp(cls, ts):
            return __import__("datetime").datetime.fromtimestamp(ts)

        @classmethod
        def strptime(cls, value, fmt):
            return __import__("datetime").datetime.strptime(value, fmt)

    monkeypatch.setattr(scheduler_module, "datetime", _FrozenDatetime)

    scheduler.log_retraining_event("first", retrained=True, deployed=False)
    scheduler.log_retraining_event("second", retrained=True, deployed=True)

    history = json.loads(Path(scheduler.history_log_path).read_text(encoding="utf-8"))

    assert isinstance(history, list)
    assert len(history) == 2
    assert history[0]["trigger"] == "first"
    assert history[1]["trigger"] == "second"


def test_training_label_snapshot_tracks_only_mature_training_window_labels(
    tmp_path, scheduler_module
):
    db_path = tmp_path / "labels.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE virtual_orders (
                order_id TEXT, order_timestamp TEXT, validated_at TEXT,
                status TEXT, decision_mode TEXT, data_quality_label TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO virtual_orders VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("a", "2026-01-10 00:00:00", "2026-01-11 00:00:00", "EXECUTED", "exploit", "STRONG"),
                ("b", "2026-01-12 00:00:00", "2026-01-13 00:00:00", "FAILED", "exploit", "WEAK_PROXY"),
                ("future", "2026-01-14 00:00:00", "2026-02-02 00:00:00", "FAILED", "exploit", "WEAK_PROXY"),
                ("probe", "2026-01-15 00:00:00", "2026-01-16 00:00:00", "FAILED", "probe", "WEAK_PROXY"),
                ("censored", "2026-01-16 00:00:00", "2026-01-17 00:00:00", "FAILED", "exploit", "CENSORED"),
            ],
        )

    scheduler = scheduler_module.RetrainingScheduler(
        db_path=str(db_path),
        production_model_dir=str(tmp_path / "models"),
        backup_dir=str(tmp_path / "backup"),
        log_dir=str(tmp_path / "logs"),
    )
    training_end = __import__("datetime").datetime(2026, 2, 1, 0, 0, 0)

    first = scheduler.get_training_label_snapshot(training_end)
    second = scheduler.get_training_label_snapshot(training_end)

    assert first["label_count"] == 2
    assert first["positive_count"] == 1
    assert first["negative_count"] == 1
    assert first["fingerprint"] == second["fingerprint"]


def test_cleanup_preserves_latest_rejected_challenger(tmp_path, scheduler_module):
    model_root = tmp_path / "data"
    production = model_root / "models"
    old_challenger = model_root / "models_retrained_old"
    latest_challenger = model_root / "models_retrained_latest"
    production.mkdir(parents=True)
    old_challenger.mkdir()
    latest_challenger.mkdir()

    scheduler = scheduler_module.RetrainingScheduler(
        db_path=str(tmp_path / "db.sqlite"),
        production_model_dir=str(production),
        backup_dir=str(tmp_path / "backup"),
        log_dir=str(tmp_path / "logs"),
    )
    scheduler.cleanup_old_artifacts(
        str(latest_challenger), preserve_retrained=True
    )

    assert latest_challenger.exists()
    assert not old_challenger.exists()
