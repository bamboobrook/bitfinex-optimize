import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ml_engine.data_processor as data_processor_module
from ml_engine.data_processor import DataProcessor
from ml_engine.execution_features import ExecutionFeatures


def _seed_orders(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE virtual_orders (
                currency TEXT,
                period INTEGER,
                order_timestamp TEXT,
                validated_at TEXT,
                status TEXT,
                execution_rate REAL,
                predicted_rate REAL,
                rate_gap REAL,
                execution_delay_minutes REAL
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO virtual_orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "fUSD",
                    2,
                    "2026-01-01 00:00:00",
                    "2026-01-04 00:00:00",
                    "EXECUTED",
                    8.0,
                    7.0,
                    None,
                    30.0,
                ),
                (
                    "fUSD",
                    2,
                    "2026-01-01 01:00:00",
                    "2026-01-02 00:00:00",
                    "FAILED",
                    None,
                    9.0,
                    2.0,
                    None,
                ),
                (
                    "fUSD",
                    3,
                    "2026-01-10 00:00:00",
                    "2026-01-11 00:00:00",
                    "FAILED",
                    None,
                    9.0,
                    2.0,
                    None,
                ),
            ],
        )
        conn.commit()
    finally:
        conn.close()


def test_execution_features_only_use_outcomes_observable_as_of_cutoff(tmp_path):
    db_path = tmp_path / "orders.sqlite"
    _seed_orders(db_path)
    features = ExecutionFeatures(str(db_path))
    before_execution = datetime(2026, 1, 2, 12)
    after_execution = datetime(2026, 1, 5)

    rate_before = features.calculate_execution_rate(
        "fUSD",
        2,
        7,
        as_of_date=before_execution,
    )
    rate_after = features.calculate_execution_rate(
        "fUSD",
        2,
        7,
        as_of_date=after_execution,
    )

    assert rate_before == pytest.approx(0.5225)
    assert rate_after == pytest.approx(0.545)
    assert features.calculate_avg_spread(
        "fUSD",
        2,
        7,
        as_of_date=before_execution,
    ) == 0.0
    assert features.calculate_avg_spread(
        "fUSD",
        2,
        7,
        as_of_date=after_execution,
    ) == 1.0
    assert features.calculate_avg_rate_gap(
        "fUSD",
        2,
        7,
        as_of_date=before_execution,
    ) == 2.0
    assert features.calculate_execution_delay_percentile(
        "fUSD",
        2,
        7,
        0.5,
        as_of_date=before_execution,
    ) == 0.0
    assert features.calculate_execution_delay_percentile(
        "fUSD",
        2,
        7,
        0.5,
        as_of_date=after_execution,
    ) == 30.0


def test_cold_start_query_does_not_count_orders_after_as_of_date(tmp_path):
    db_path = tmp_path / "orders.sqlite"
    _seed_orders(db_path)
    features = ExecutionFeatures(str(db_path))

    rate = features.calculate_execution_rate(
        "fUSD",
        3,
        3,
        as_of_date=datetime(2026, 1, 2),
    )

    assert rate == pytest.approx(0.55)


def test_data_processor_does_not_broadcast_current_execution_snapshot(tmp_path):
    db_path = tmp_path / "orders.sqlite"
    _seed_orders(db_path)
    processor = DataProcessor(str(db_path))
    datetimes = pd.to_datetime(
        [
            "2026-01-01 00:00:00",
            "2026-01-03 00:00:00",
            "2026-01-05 00:00:00",
        ]
    )
    market = pd.DataFrame(
        {
            "currency": ["fUSD"] * 3,
            "period": [2] * 3,
            "datetime": datetimes,
            "open_annual": [6.0, 6.1, 6.2],
            "close_annual": [6.0, 6.1, 6.2],
            "high_annual": [6.2, 6.3, 6.4],
            "low_annual": [5.8, 5.9, 6.0],
            "volume": [100.0, 100.0, 100.0],
            "hour": [0, 0, 0],
            "day_of_week": [3, 5, 0],
        }
    )

    result = processor.add_technical_indicators(market)

    assert result["avg_spread_profile"].tolist() == [0.0, 0.0, 1.0]
    assert result["exec_delay_p50"].tolist() == [0.0, 0.0, 30.0]
    assert result["exec_rate_fast"].iloc[0] == pytest.approx(0.55)
    assert result["exec_rate_fast"].iloc[-1] == pytest.approx(0.50875)


def test_failed_early_sample_uses_neutral_default_not_future_snapshot(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "orders.sqlite"
    _seed_orders(db_path)

    def _flaky_snapshot(self, currency, period, as_of_date=None):
        if as_of_date.day == 1:
            raise sqlite3.OperationalError("database is busy")
        return {
            "exec_rate_fast": 2.0,
            "exec_rate_slow": 2.0,
            "avg_spread_profile": 1.0,
            "avg_spread_7d": 1.0,
            "avg_spread_30d": 1.0,
            "avg_rate_gap_failed_profile": 1.0,
            "avg_rate_gap_failed_7d": 1.0,
            "avg_rate_gap_failed_30d": 1.0,
            "exec_delay_p50": 1.0,
            "exec_delay_p90": 1.0,
        }

    monkeypatch.setattr(
        ExecutionFeatures,
        "get_all_features",
        _flaky_snapshot,
    )
    processor = DataProcessor(str(db_path))
    datetimes = pd.to_datetime(
        ["2026-01-01 00:00:00", "2026-01-02 00:00:00"]
    )
    market = pd.DataFrame(
        {
            "currency": ["fUSD"] * 2,
            "period": [2] * 2,
            "datetime": datetimes,
            "open_annual": [6.0, 6.1],
            "close_annual": [6.0, 6.1],
            "high_annual": [6.2, 6.3],
            "low_annual": [5.8, 5.9],
            "volume": [100.0, 100.0],
            "hour": [0, 0],
            "day_of_week": [3, 4],
        }
    )

    result = processor.add_technical_indicators(market)

    assert result["exec_rate_fast"].tolist() == [0.55, 2.0]
    assert result["avg_spread_profile"].tolist() == [0.0, 1.0]


def test_process_currency_workers_keep_custom_database_path(
    tmp_path,
    monkeypatch,
):
    submitted = {}

    class _ImmediateFuture:
        def __init__(self, fn, args):
            self._result = fn(*args)

        def result(self):
            return self._result

    class _ImmediateExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, fn, *args):
            return _ImmediateFuture(fn, args)

    def _record_worker(period_df, db_path):
        submitted["db_path"] = db_path
        return period_df

    market = pd.DataFrame(
        {
            "currency": ["fUSD"],
            "period": [2],
            "datetime": pd.to_datetime(["2026-01-01 00:00:00"]),
        }
    )
    custom_db = tmp_path / "custom.sqlite"
    processor = DataProcessor(str(custom_db))

    monkeypatch.setattr(processor, "load_data", lambda _currency: market)
    monkeypatch.setattr(
        processor,
        "_process_single_period",
        _record_worker,
    )
    monkeypatch.setattr(
        data_processor_module,
        "ProcessPoolExecutor",
        _ImmediateExecutor,
    )
    monkeypatch.setattr(
        data_processor_module,
        "as_completed",
        lambda futures: iter(futures),
    )
    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        lambda self, path, index=False: None,
    )

    output = processor.process_currency(
        "fUSD",
        output_dir=str(tmp_path),
        max_workers=1,
    )

    assert output == str(tmp_path / "fUSD_features.parquet")
    assert submitted["db_path"] == str(custom_db)
