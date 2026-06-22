import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test_predictor_rank6 import EnsemblePredictor
import ml_engine.execution_features as execution_features
import ml_engine.predictor as predictor_module


def test_period_prediction_batch_skips_stalled_period(monkeypatch):
    predictor = EnsemblePredictor.__new__(EnsemblePredictor)
    predictor.max_workers = 2

    monkeypatch.setenv("PREDICT_PERIOD_TIMEOUT", "0.05")

    def fake_predict_single_period(row_data, feature_cols, currency):
        if int(row_data["period"]) == 90:
            time.sleep(30)
        return {
            "currency": currency,
            "period": int(row_data["period"]),
            "predicted_rate": 5.0,
        }

    predictor.predict_single_period = fake_predict_single_period
    latest_data = pd.DataFrame([{"period": 2}, {"period": 90}])

    started_at = time.monotonic()
    results = predictor._run_period_prediction_batch(latest_data, [], "fUSD")
    elapsed = time.monotonic() - started_at

    assert elapsed < 1.0
    assert results == [{"currency": "fUSD", "period": 2, "predicted_rate": 5.0}]


def test_execution_features_uses_short_sqlite_busy_timeout(monkeypatch):
    calls = []
    pragmas = []

    class FakeConnection:
        def execute(self, sql):
            pragmas.append(sql)

    def fake_connect(db_path, **kwargs):
        calls.append((db_path, kwargs))
        return FakeConnection()

    monkeypatch.setenv("PREDICT_SQLITE_BUSY_TIMEOUT", "1.25")
    monkeypatch.setattr(execution_features.sqlite3, "connect", fake_connect)

    conn = execution_features.ExecutionFeatures("test.db")._get_connection()

    assert isinstance(conn, FakeConnection)
    assert calls == [("test.db", {"timeout": 1.25})]
    assert "PRAGMA busy_timeout = 1250" in pragmas
    assert "PRAGMA query_only = ON" in pragmas


def test_predictor_read_connection_uses_short_sqlite_busy_timeout(monkeypatch):
    predictor = EnsemblePredictor.__new__(EnsemblePredictor)
    predictor.db_path = "test.db"
    calls = []
    pragmas = []

    class FakeConnection:
        def execute(self, sql):
            pragmas.append(sql)

    def fake_connect(db_path, **kwargs):
        calls.append((db_path, kwargs))
        return FakeConnection()

    monkeypatch.setenv("PREDICT_SQLITE_BUSY_TIMEOUT", "1.25")
    monkeypatch.setattr(predictor_module.sqlite3, "connect", fake_connect)

    conn = predictor._connect_db(read_only=True)

    assert isinstance(conn, FakeConnection)
    assert calls == [("test.db", {"timeout": 1.25})]
    assert "PRAGMA busy_timeout = 1250" in pragmas
    assert "PRAGMA query_only = ON" in pragmas
