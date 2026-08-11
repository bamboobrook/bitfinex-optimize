import sqlite3
from datetime import datetime as RealDateTime

import ml_engine.api_server as api_server
import ml_engine.execution_validator as validator_module
import ml_engine.metrics as metrics_module


class FixedDateTime(RealDateTime):
    @classmethod
    def now(cls, tz=None):
        value = cls(2035, 1, 8, 12, 0, 0)
        return value if tz is None else value.replace(tzinfo=tz)


def _create_orders_db(path):
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE virtual_orders (
            order_id TEXT PRIMARY KEY,
            currency TEXT NOT NULL,
            period INTEGER NOT NULL,
            predicted_rate REAL NOT NULL,
            status TEXT NOT NULL,
            order_timestamp TEXT NOT NULL,
            validated_at TEXT,
            max_market_rate REAL,
            rate_gap REAL,
            stage1_rate_gap REAL
        )
        """
    )
    return conn


def _insert_boundary_orders(conn):
    conn.executemany(
        """
        INSERT INTO virtual_orders (
            order_id, currency, period, predicted_rate, status,
            order_timestamp, validated_at, max_market_rate, rate_gap,
            stage1_rate_gap
        ) VALUES (?, 'fUSD', 2, 8.0, ?, ?, ?, 7.0, 1.0, 1.0)
        """,
        [
            ("inside", "EXECUTED", "2035-01-01 13:00:00", "2035-01-08 11:00:00"),
            ("outside", "FAILED", "2035-01-01 04:00:00", "2035-01-08 06:00:00"),
        ],
    )
    conn.commit()


def test_api_statistics_use_local_wall_clock_cutoff(tmp_path, monkeypatch):
    db_path = tmp_path / "orders.db"
    conn = _create_orders_db(db_path)
    _insert_boundary_orders(conn)
    conn.close()

    monkeypatch.setattr(api_server, "DB_FILE", str(db_path))
    monkeypatch.setattr(api_server, "datetime", FixedDateTime)

    overall = api_server.get_db_statistics()["execution_rate_7d_overall"]

    assert overall == {"total": 1, "executed": 1, "exec_rate": 100.0}


def test_metrics_use_local_wall_clock_cutoff(tmp_path, monkeypatch):
    db_path = tmp_path / "orders.db"
    conn = _create_orders_db(db_path)
    _insert_boundary_orders(conn)
    conn.close()

    monkeypatch.setattr(metrics_module, "datetime", FixedDateTime)

    result = metrics_module.MetricsCollector(str(db_path)).get_execution_metrics()

    assert result["execution_rate"] == "100.0%"
    assert result["window_total_decided"] == 1
    assert result["window_executed"] == 1


def test_validator_activity_windows_use_local_wall_clock_cutoffs(tmp_path, monkeypatch):
    db_path = tmp_path / "orders.db"
    conn = _create_orders_db(db_path)
    _insert_boundary_orders(conn)
    conn.close()

    monkeypatch.setattr(validator_module, "datetime", FixedDateTime)
    validator = validator_module.ExecutionValidator(str(db_path))

    assert validator._get_recent_validation_count(2, "fUSD", hours=4) == 1
    assert validator._get_execution_threshold(2, "fUSD") == 38.0
