import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_engine.execution_validator import ExecutionValidator


def _init_test_db(db_path):
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE funding_rates (
                currency TEXT NOT NULL,
                period INTEGER NOT NULL,
                timestamp INTEGER,
                datetime TEXT NOT NULL,
                high_annual REAL,
                close_annual REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE virtual_orders (
                order_id TEXT PRIMARY KEY,
                currency TEXT NOT NULL,
                period INTEGER NOT NULL,
                predicted_rate REAL NOT NULL,
                order_timestamp TEXT NOT NULL,
                validation_window_hours INTEGER NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT,
                validated_at TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO funding_rates(currency, period, timestamp, datetime, high_annual, close_annual) VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("fUSD", 30, 1, "2026-03-01 00:30:00", 13.5, 13.2),
                ("fUSD", 30, 2, "2026-03-01 02:30:00", 13.2, 12.9),
                ("fUSD", 120, 3, "2026-03-01 07:30:00", 10.6, 10.4),
                ("fUSD", 120, 4, "2026-03-01 08:30:00", 10.5, 10.3),
            ],
        )
        conn.execute(
            """
            INSERT INTO virtual_orders(order_id, currency, period, predicted_rate, order_timestamp, validation_window_hours, status, created_at)
            VALUES ('o-1', 'fUSD', 30, 14.0, '2026-03-01 00:00:00', 48, 'PENDING', '2026-03-01 00:00:00')
            """
        )
        conn.commit()
    finally:
        conn.close()


def test_validate_single_order_marks_frr_proxy_when_fixed_stage_misses(tmp_path):
    db_path = tmp_path / "path_labels.db"
    _init_test_db(db_path)

    validator = ExecutionValidator(str(db_path))
    result = validator.validate_single_order(
        {
            "order_id": "o-1",
            "currency": "fUSD",
            "period": 30,
            "predicted_rate": 14.0,
            "order_timestamp": "2026-03-01 00:00:00",
            "validation_window_hours": 48,
            "status": "PENDING",
        }
    )

    assert result["status"] == "FAILED"
    assert result["path_stage_outcome"] == "FIXED_MISS"
    assert result["terminal_mode"] == "FRR_PROXY"
    assert result["stage2_frr_proxy_rate"] > 0.0
