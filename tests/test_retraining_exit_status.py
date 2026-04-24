import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_predictor_rank6 import _make_prediction


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
    monkeypatch.setattr(scheduler, "get_recent_execution_rate", lambda days=7, since_dt=None: 0.51)
    monkeypatch.setattr(
        scheduler,
        "get_per_period_execution_anomalies",
        lambda days=7, since_dt=None: [
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
        lambda days=7, since_dt=None: {
            "samples": 100,
            "follow_mae": 1.7,
            "follow_mae_ratio": 0.28,
            "direction_match_rate": 0.48,
            "p120_samples": 40,
            "p120_step_p95": 0.20,
        },
    )
    monkeypatch.setattr(scheduler, "_check_zero_liquidity_anomaly", lambda since_dt=None: [])
    monkeypatch.setattr(scheduler, "_check_market_divergence_trigger", lambda since_dt=None: False)

    should_retrain, reason = scheduler.should_retrain()

    assert should_retrain is False
    assert reason is None


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

    def _recent_exec(days=7, since_dt=None):
        captured["exec_since"] = since_dt
        return 0.51

    def _period_anomalies(days=7, since_dt=None):
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
        lambda days=7, since_dt=None: {
            "samples": 20,
            "follow_mae": 0.0,
            "follow_mae_ratio": 0.0,
            "direction_match_rate": 0.0,
            "p120_samples": 0,
            "p120_step_p95": 0.0,
        },
    )
    monkeypatch.setattr(scheduler, "_check_zero_liquidity_anomaly", lambda since_dt=None: [])
    monkeypatch.setattr(scheduler, "_check_market_divergence_trigger", lambda since_dt=None: False)
    monkeypatch.setattr(
        scheduler,
        "_count_orders_since",
        lambda since_dt: 60,
    )

    should_retrain, reason = scheduler.should_retrain()

    assert should_retrain is True
    assert "单period成交率极低" in reason
    assert captured["exec_since"] == __import__("datetime").datetime.fromtimestamp(deployed_ts)
    assert captured["period_since"] == __import__("datetime").datetime.fromtimestamp(deployed_ts)


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
