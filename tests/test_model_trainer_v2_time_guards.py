import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_ENGINE_ROOT = PROJECT_ROOT / "ml_engine"
for path in (PROJECT_ROOT, ML_ENGINE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ml_engine.model_trainer_v2 import (
    EnhancedModelTrainer,
    add_training_technical_features,
)


class _DummySavedModel:
    def save_model(self, *_args, **_kwargs):
        pass


class _RecordingTrainer(EnhancedModelTrainer):
    """Keep the split test independent of the actual ML libraries."""

    def __init__(self):
        self.model_dir = "."
        self.validation_signals = []
        self.training_sizes = []
        self.cpu_threads = 3
        self.xgb_params = {"nthread": 3, "n_jobs": 3}
        self.lgb_params = {"num_threads": 3, "n_jobs": 3}
        self.catboost_params = {"thread_count": 3}

    def prepare_features(self, _df):
        return ["signal"]

    def _record_validation(self, X_train, X_val):
        self.training_sizes.append(len(X_train))
        self.validation_signals.append(X_val["signal"].tolist())
        return _DummySavedModel(), 1.0

    def train_xgboost_regression(self, X_train, _y_train, X_val, _y_val, sample_weight=None):
        return self._record_validation(X_train, X_val)

    def train_lightgbm_regression(self, X_train, _y_train, X_val, _y_val, sample_weight=None):
        return self._record_validation(X_train, X_val)

    def train_catboost_regression(self, X_train, _y_train, X_val, _y_val, sample_weight=None):
        return self._record_validation(X_train, X_val)

    def save_ensemble_models(self, *_args, **_kwargs):
        pass


def test_model_trainer_v2_imports_with_project_root_only(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import ml_engine.model_trainer_v2",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_future_execution_label_stays_nan_without_a_forward_window():
    group = pd.DataFrame(
        {
            "currency": ["fUSD"] * 130,
            "period": [30] * 130,
            "datetime": pd.date_range("2026-01-01", periods=130, freq="h"),
            "low_annual": np.linspace(5.0, 8.0, 130),
            "close_annual": np.linspace(6.0, 9.0, 130),
        }
    )

    labeled = EnhancedModelTrainer._compute_traditional_targets(group)

    # No future close exists for the final row; it is unknown, not a failed event.
    assert pd.isna(labeled.iloc[-1]["future_execution_prob"])
    assert labeled.iloc[-2]["future_execution_prob"] in (0.0, 1.0)


def test_train_single_target_splits_by_datetime_after_unsorted_input():
    timestamps = pd.date_range("2026-01-01", periods=1000, freq="h")
    ordered = pd.DataFrame(
        {
            "currency": ["fUSD"] * 1000,
            "period": np.repeat([2, 3, 4, 5, 6], 200),
            "datetime": timestamps,
            "signal": np.arange(1000, dtype=float),
            "future_balanced": np.arange(1000, dtype=float),
        }
    )
    # Simulate the upstream currency/period-blocked or otherwise unsorted frame.
    shuffled = ordered.sample(frac=1.0, random_state=7).reset_index(drop=True)
    trainer = _RecordingTrainer()

    trainer.train_single_target(
        currency="fUSD",
        df=shuffled,
        target_name="future_balanced",
        task_type="regression",
        output_prefix="model_balanced",
    )

    assert len(trainer.validation_signals) == 3
    assert all(
        signals == list(np.arange(900, 1000, dtype=float))
        for signals in trainer.validation_signals
    )


def test_order_rows_reuse_backward_matched_market_features():
    class _Processor:
        @staticmethod
        def add_technical_indicators(group):
            result = group.copy()
            result["rolling_marker"] = result["close_annual"].expanding().sum()
            result["future_execution_prob"] = 1.0
            return result

    market_times = pd.date_range("2026-01-01", periods=3, freq="h")
    market = pd.DataFrame(
        {
            "currency": ["fUSD"] * 3,
            "period": [30] * 3,
            "datetime": market_times,
            "close_annual": [1.0, 2.0, 3.0],
            "_sample_role": ["market_dense"] * 3,
        }
    )
    order = pd.DataFrame(
        {
            "currency": ["fUSD"],
            "period": [30],
            "datetime": [market_times[0]],
            "order_timestamp": [market_times[0] + pd.Timedelta(minutes=10)],
            "close_annual": [1.0],
            "actual_execution_binary": [1.0],
            "_sample_role": ["order_supervision"],
        }
    )

    featured = add_training_technical_features(
        pd.concat([market, order], ignore_index=True, sort=False),
        _Processor(),
    )
    order_row = featured[featured["_sample_role"] == "order_supervision"].iloc[0]

    assert order_row["rolling_marker"] == 1.0
    assert order_row["actual_execution_binary"] == 1.0
    assert pd.isna(order_row["future_execution_prob"])


def test_feedback_split_embargoes_labels_resolved_in_validation_period():
    row_count = 300
    timestamps = pd.date_range("2026-01-01", periods=row_count, freq="h")
    frame = pd.DataFrame(
        {
            "currency": ["fUSD"] * row_count,
            "period": [30] * row_count,
            "datetime": timestamps,
            "order_timestamp": timestamps,
            "validated_at": timestamps + pd.Timedelta(hours=72),
            "validation_window_hours": [72] * row_count,
            "signal": np.arange(row_count, dtype=float),
            "revenue_optimized_target": np.linspace(1.0, 2.0, row_count),
        }
    )
    trainer = _RecordingTrainer()

    trainer.train_single_target(
        currency="fUSD",
        df=frame,
        target_name="revenue_optimized_target",
        task_type="regression",
        output_prefix="model_revenue_optimized",
    )

    assert trainer.training_sizes == [198, 198, 198]
    assert all(
        signals == list(np.arange(270, 300, dtype=float))
        for signals in trainer.validation_signals
    )
