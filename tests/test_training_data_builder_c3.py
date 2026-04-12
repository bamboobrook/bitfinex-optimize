import sqlite3
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
ML_ENGINE_ROOT = PROJECT_ROOT / "ml_engine"
if str(ML_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ENGINE_ROOT))


def _install_training_lib_stubs():
    if "xgboost" not in sys.modules:
        xgb_module = types.ModuleType("xgboost")
        xgb_module.DMatrix = object
        xgb_module.train = lambda *args, **kwargs: None
        sys.modules["xgboost"] = xgb_module

    if "lightgbm" not in sys.modules:
        lgb_module = types.ModuleType("lightgbm")
        lgb_module.Dataset = object
        lgb_module.train = lambda *args, **kwargs: None
        lgb_module.early_stopping = lambda *args, **kwargs: None
        lgb_module.log_evaluation = lambda *args, **kwargs: None
        sys.modules["lightgbm"] = lgb_module

    catboost_module = sys.modules.get("catboost")
    if catboost_module is None:
        catboost_module = types.ModuleType("catboost")
        sys.modules["catboost"] = catboost_module

    class _DummyModel:
        def __init__(self, *args, **kwargs):
            pass

    if not hasattr(catboost_module, "CatBoostRegressor"):
        catboost_module.CatBoostRegressor = _DummyModel
    if not hasattr(catboost_module, "CatBoostClassifier"):
        catboost_module.CatBoostClassifier = _DummyModel
    if not hasattr(catboost_module, "Pool"):
        catboost_module.Pool = _DummyModel


_install_training_lib_stubs()

from ml_engine.training_data_builder import TrainingDataBuilder
from ml_engine.model_trainer_v2 import EnhancedModelTrainer


class _DummySavedModel:
    def save_model(self, *_args, **_kwargs):
        pass


class _RecordingTrainer(EnhancedModelTrainer):
    def __init__(self, db_path, model_dir):
        super().__init__(db_path=str(db_path), model_dir=str(model_dir))
        self.calls = []
        self.saved = None

    def train_xgboost_regression(self, X_train, y_train, X_val, y_val, sample_weight=None):
        self.calls.append(("xgb_reg", len(X_train), len(X_val)))
        return _DummySavedModel(), 0.1

    def train_lightgbm_regression(self, X_train, y_train, X_val, y_val, sample_weight=None):
        self.calls.append(("lgb_reg", len(X_train), len(X_val)))
        return _DummySavedModel(), 0.2

    def train_catboost_regression(self, X_train, y_train, X_val, y_val, sample_weight=None):
        self.calls.append(("cat_reg", len(X_train), len(X_val)))
        return _DummySavedModel(), 0.3

    def train_xgboost_classification(self, X_train, y_train, X_val, y_val, sample_weight=None):
        self.calls.append(("xgb_cls", len(X_train), len(X_val)))
        return _DummySavedModel(), 0.6

    def train_lightgbm_classification(self, X_train, y_train, X_val, y_val, sample_weight=None):
        self.calls.append(("lgb_cls", len(X_train), len(X_val)))
        return _DummySavedModel(), 0.7

    def train_catboost_classification(self, X_train, y_train, X_val, y_val, sample_weight=None):
        self.calls.append(("cat_cls", len(X_train), len(X_val)))
        return _DummySavedModel(), 0.8

    def save_ensemble_models(
        self,
        currency: str,
        prefix: str,
        models: dict,
        weights: dict,
        feature_cols: list,
        task_type: str
    ):
        self.saved = {
            "currency": currency,
            "prefix": prefix,
            "weights": weights,
            "feature_cols": feature_cols,
            "task_type": task_type,
        }


def _make_training_frame(target_name, target_values):
    row_count = len(target_values)
    return pd.DataFrame(
        {
            "feature_a": np.linspace(0.0, 1.0, row_count),
            "feature_b": np.arange(row_count, dtype=float),
            target_name: target_values,
        }
    )


def _seed_training_db(db_path):
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE funding_rates (
                currency TEXT NOT NULL,
                period INTEGER NOT NULL,
                timestamp INTEGER,
                datetime TEXT NOT NULL,
                open_annual REAL,
                close_annual REAL,
                high_annual REAL,
                low_annual REAL,
                volume REAL,
                hour INTEGER,
                day_of_week INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE virtual_orders (
                order_timestamp TEXT NOT NULL,
                currency TEXT NOT NULL,
                period INTEGER NOT NULL,
                predicted_rate REAL NOT NULL,
                status TEXT NOT NULL,
                execution_confidence REAL,
                total_score REAL,
                market_median REAL,
                execution_rate REAL,
                decision_mode TEXT,
                data_quality_label TEXT,
                validation_label TEXT,
                realized_terminal_mode TEXT,
                realized_terminal_value REAL,
                realized_wait_hours REAL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO funding_rates VALUES
            ('fUSD', 30, 1, '2026-04-01 00:00:00', 10.0, 10.2, 10.4, 9.8, 1000.0, 0, 3)
            """
        )
        conn.executemany(
            """
            INSERT INTO virtual_orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    '2026-04-01 00:30:00',
                    'fUSD',
                    30,
                    10.8,
                    'EXECUTED',
                    0.8,
                    80.0,
                    10.1,
                    10.7,
                    'exploit',
                    'STRONG',
                    'PATH_STAGE1_FILLED',
                    'FIXED',
                    10.7,
                    0.5,
                ),
                (
                    '2026-04-01 00:35:00',
                    'fUSD',
                    30,
                    11.2,
                    'FAILED',
                    0.3,
                    30.0,
                    10.1,
                    None,
                    'exploit',
                    'WEAK_PROXY',
                    'PATH_STAGE2_PROXY',
                    'FRR_PROXY',
                    9.2,
                    12.0,
                ),
                (
                    '2026-04-01 00:37:00',
                    'fUSD',
                    30,
                    11.5,
                    'EXPIRED',
                    0.0,
                    0.0,
                    10.1,
                    None,
                    'exploit',
                    'CENSORED',
                    'UNKNOWN',
                    'UNKNOWN',
                    None,
                    None,
                ),
                (
                    '2026-04-01 00:40:00',
                    'fUSD',
                    30,
                    11.0,
                    'FAILED',
                    0.4,
                    40.0,
                    10.1,
                    None,
                    'probe',
                    'WEAK_PROXY',
                    'PATH_STAGE2_PROXY',
                    'FRR_PROXY',
                    9.2,
                    12.0,
                ),
            ],
        )
        conn.commit()
    finally:
        conn.close()


def _seed_minimal_execution_db(db_path):
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE funding_rates (
                currency TEXT NOT NULL,
                period INTEGER NOT NULL,
                timestamp INTEGER,
                datetime TEXT NOT NULL,
                open_annual REAL,
                close_annual REAL,
                high_annual REAL,
                low_annual REAL,
                volume REAL,
                hour INTEGER,
                day_of_week INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE virtual_orders (
                order_timestamp TEXT NOT NULL,
                currency TEXT NOT NULL,
                period INTEGER NOT NULL,
                predicted_rate REAL NOT NULL,
                status TEXT NOT NULL,
                execution_rate REAL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO funding_rates VALUES
            ('fUSD', 30, 1, '2026-04-01 00:00:00', 10.0, 10.2, 10.4, 9.8, 1000.0, 0, 3)
            """
        )
        conn.execute(
            """
            INSERT INTO virtual_orders VALUES
            ('2026-04-01 00:30:00', 'fUSD', 30, 10.8, 'EXECUTED', 10.7)
            """
        )
        conn.commit()
    finally:
        conn.close()


def _seed_mixed_schema_db(db_path, order_row=None):
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE funding_rates (
                currency TEXT NOT NULL,
                period INTEGER NOT NULL,
                timestamp INTEGER,
                datetime TEXT NOT NULL,
                open_annual REAL,
                close_annual REAL,
                high_annual REAL,
                low_annual REAL,
                volume REAL,
                hour INTEGER,
                day_of_week INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE virtual_orders (
                order_timestamp TEXT NOT NULL,
                currency TEXT NOT NULL,
                period INTEGER NOT NULL,
                predicted_rate REAL NOT NULL,
                status TEXT NOT NULL,
                execution_confidence REAL,
                total_score REAL,
                market_median REAL,
                execution_rate REAL,
                terminal_mode TEXT,
                decision_mode TEXT,
                data_quality_label TEXT,
                validation_label TEXT,
                realized_terminal_mode TEXT,
                realized_terminal_value REAL,
                realized_wait_hours REAL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO funding_rates VALUES
            ('fUSD', 30, 1, '2026-04-01 00:00:00', 10.0, 10.2, 10.4, 9.8, 1000.0, 0, 3)
            """
        )
        default_row = (
            '2026-04-01 00:30:00',
            'fUSD',
            30,
            10.8,
            'EXECUTED',
            0.8,
            80.0,
            10.1,
            10.7,
            'FIXED',
            'exploit',
            'STRONG',
            'PATH_STAGE1_FILLED',
            None,
            None,
            None,
        )
        conn.execute(
            """
            INSERT INTO virtual_orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            order_row or default_row,
        )
        conn.commit()
    finally:
        conn.close()


def test_build_training_data_keeps_exploit_positive_and_negative_execution_samples(tmp_path):
    db_path = tmp_path / "db.sqlite"
    _seed_training_db(db_path)
    builder = TrainingDataBuilder(str(db_path))

    df = builder.build_training_data("2026-04-01", "2026-04-02", include_execution_results=True)

    assert set(df["decision_mode"].unique()) == {"exploit"}
    assert set(df["status"].unique()) == {"EXECUTED", "FAILED", "EXPIRED"}
    assert set(df["data_quality_label"].unique()) == {"STRONG", "WEAK_PROXY", "CENSORED"}
    assert set(df["actual_execution_binary"].unique()) == {0.0, 1.0}

    by_status = df.set_index("status")
    assert by_status.loc["EXECUTED", "actual_execution_binary"] == 1.0
    assert by_status.loc["FAILED", "actual_execution_binary"] == 0.0
    assert by_status.loc["EXPIRED", "actual_execution_binary"] == 0.0


def test_build_training_data_handles_execution_results_with_missing_optional_columns(tmp_path):
    db_path = tmp_path / "minimal.sqlite"
    _seed_minimal_execution_db(db_path)
    builder = TrainingDataBuilder(str(db_path))

    df = builder.build_training_data("2026-04-01", "2026-04-02", include_execution_results=True)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["currency"] == "fUSD"
    assert row["actual_execution_binary"] == 1.0
    assert "execution_confidence" in df.columns
    assert "total_score" in df.columns
    assert "market_median" in df.columns


def test_build_training_data_only_assigns_path_targets_to_strong_rows(tmp_path):
    db_path = tmp_path / "path_gating.sqlite"
    _seed_training_db(db_path)
    builder = TrainingDataBuilder(str(db_path))

    df = builder.build_training_data("2026-04-01", "2026-04-02", include_execution_results=True)

    strong_rows = df[df["data_quality_label"] == "STRONG"]
    weak_rows = df[df["data_quality_label"] != "STRONG"]

    assert len(df) == 3
    assert len(strong_rows) == 1
    assert len(weak_rows) == 2
    assert strong_rows["path_terminal_value"].notna().all()
    assert strong_rows["path_stage1_success"].notna().all()
    assert strong_rows["path_wait_hours"].notna().all()

    assert weak_rows["path_terminal_value"].isna().all()
    assert weak_rows["path_stage1_success"].isna().all()
    assert weak_rows["path_wait_hours"].isna().all()


def test_build_training_data_falls_back_when_realized_values_are_null(tmp_path):
    db_path = tmp_path / "mixed_schema.sqlite"
    _seed_mixed_schema_db(db_path)
    builder = TrainingDataBuilder(str(db_path))

    df = builder.build_training_data("2026-04-01", "2026-04-02", include_execution_results=True)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["path_terminal_value"] == 10.7
    assert row["path_stage1_success"] == 1.0


def test_build_training_data_does_not_mark_failed_fixed_row_as_stage1_success(tmp_path):
    db_path = tmp_path / "failed_fixed.sqlite"
    _seed_mixed_schema_db(
        db_path,
        order_row=(
            '2026-04-01 00:30:00',
            'fUSD',
            30,
            10.8,
            'FAILED',
            0.8,
            80.0,
            10.1,
            None,
            'FIXED',
            'exploit',
            'STRONG',
            'PATH_STAGE1_FILLED',
            None,
            None,
            None,
        ),
    )
    builder = TrainingDataBuilder(str(db_path))

    df = builder.build_training_data("2026-04-01", "2026-04-02", include_execution_results=True)

    assert len(df) == 1
    assert df.iloc[0]["path_stage1_success"] == 0.0


def test_build_training_data_falls_back_when_realized_values_are_empty_strings(tmp_path):
    db_path = tmp_path / "empty_realized.sqlite"
    _seed_mixed_schema_db(
        db_path,
        order_row=(
            '2026-04-01 00:30:00',
            'fUSD',
            30,
            10.8,
            'EXECUTED',
            0.8,
            80.0,
            10.1,
            10.7,
            'FIXED',
            'exploit',
            'STRONG',
            'PATH_STAGE1_FILLED',
            '',
            '',
            None,
        ),
    )
    builder = TrainingDataBuilder(str(db_path))

    df = builder.build_training_data("2026-04-01", "2026-04-02", include_execution_results=True)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["path_terminal_value"] == 10.7
    assert row["path_stage1_success"] == 1.0


def test_prepare_features_excludes_path_identity_and_validation_columns(tmp_path):
    db_path = tmp_path / "features.sqlite"
    _seed_training_db(db_path)
    builder = TrainingDataBuilder(str(db_path))
    df = builder.build_training_data("2026-04-01", "2026-04-02", include_execution_results=True)

    row_count = len(df)
    df["candidate_id"] = [f"cand-{i}" for i in range(row_count)]
    df["update_cycle_id"] = list(range(101, 101 + row_count))
    df["recommendation_rank"] = list(range(1, 1 + row_count))
    df["rank_weight"] = np.linspace(1.0, 0.5, row_count)

    trainer = EnhancedModelTrainer(db_path=str(db_path), model_dir=str(tmp_path / "models"))
    feature_cols = trainer.prepare_features(df)

    assert "path_terminal_value" not in feature_cols
    assert "path_stage1_success" not in feature_cols
    assert "path_wait_hours" not in feature_cols
    assert "candidate_id" not in feature_cols
    assert "update_cycle_id" not in feature_cols
    assert "recommendation_rank" not in feature_cols
    assert "rank_weight" not in feature_cols
    assert "data_quality_label" not in feature_cols
    assert "validation_label" not in feature_cols
    assert "realized_terminal_mode" not in feature_cols
    assert "realized_terminal_value" not in feature_cols
    assert "realized_wait_hours" not in feature_cols


def test_train_single_target_skips_classification_when_target_has_single_class(tmp_path):
    trainer = _RecordingTrainer(db_path=tmp_path / "unused.sqlite", model_dir=tmp_path / "models")
    df = _make_training_frame("actual_execution_binary", np.zeros(120, dtype=int))

    result = trainer.train_single_target(
        currency="fUSD",
        df=df,
        target_name="actual_execution_binary",
        task_type="classification",
        output_prefix="model_execution_prob_v2",
    )

    assert result is None
    assert trainer.calls == []
    assert trainer.saved is None


def test_train_single_target_skips_classification_when_validation_split_has_single_class(tmp_path):
    trainer = _RecordingTrainer(db_path=tmp_path / "unused.sqlite", model_dir=tmp_path / "models")
    target = np.array([0] * 60 + [1] * 48 + [0] * 12, dtype=int)
    df = _make_training_frame("actual_execution_binary", target)

    result = trainer.train_single_target(
        currency="fUSD",
        df=df,
        target_name="actual_execution_binary",
        task_type="classification",
        output_prefix="model_execution_prob_v2",
    )

    assert result is None
    assert trainer.calls == []
    assert trainer.saved is None


def test_train_single_target_regression_path_still_trains(tmp_path):
    trainer = _RecordingTrainer(db_path=tmp_path / "unused.sqlite", model_dir=tmp_path / "models")
    df = _make_training_frame("future_conservative", np.linspace(1.0, 2.0, 120))

    result = trainer.train_single_target(
        currency="fUSD",
        df=df,
        target_name="future_conservative",
        task_type="regression",
        output_prefix="model_conservative",
    )

    assert result is not None
    models, weights = result
    assert set(models) == {"xgb", "lgb", "cat"}
    assert set(weights) == {"xgb", "lgb", "cat"}
    assert [call[0] for call in trainer.calls] == ["xgb_reg", "lgb_reg", "cat_reg"]
    assert trainer.saved["task_type"] == "regression"
