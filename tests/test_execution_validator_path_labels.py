import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_engine.execution_validator import ExecutionValidator


def _init_test_db(db_path, funding_rows, order_row):
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
            funding_rows,
        )
        conn.execute(
            """
            INSERT INTO virtual_orders(order_id, currency, period, predicted_rate, order_timestamp, validation_window_hours, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            order_row,
        )
        conn.commit()
    finally:
        conn.close()


def _fetch_order_fields(db_path, order_id):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT path_stage_outcome, terminal_mode, execution_confidence, market_median, executed_at
            FROM virtual_orders
            WHERE order_id = ?
            """,
            (order_id,),
        ).fetchone()
        return dict(row)
    finally:
        conn.close()


def test_validate_single_order_marks_frr_proxy_when_fixed_stage_misses(tmp_path):
    db_path = tmp_path / "path_labels.db"
    _init_test_db(
        db_path,
        [
            ("fUSD", 30, 1, "2026-03-01 00:30:00", 13.5, 13.2),
            ("fUSD", 30, 2, "2026-03-01 02:30:00", 13.2, 12.9),
            ("fUSD", 120, 3, "2026-03-01 07:30:00", 10.6, 10.4),
            ("fUSD", 120, 4, "2026-03-01 08:30:00", 10.5, 10.3),
        ],
        ("o-1", "fUSD", 30, 14.0, "2026-03-01 00:00:00", 48, "PENDING", "2026-03-01 00:00:00"),
    )

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
    assert result["execution_confidence"] is not None
    assert result["market_median"] > 0.0

    persisted = _fetch_order_fields(db_path, "o-1")
    assert persisted["path_stage_outcome"] == "FIXED_MISS"
    assert persisted["terminal_mode"] == "FRR_PROXY"
    assert persisted["execution_confidence"] is not None
    assert persisted["market_median"] > 0.0


def test_validate_single_order_marks_fixed_fill_with_real_tick_timestamp(tmp_path):
    db_path = tmp_path / "path_labels_fixed.db"
    _init_test_db(
        db_path,
        [
            ("fUSD", 30, 1, "2026-03-01 00:30:00", 13.5, 13.2),
            ("fUSD", 30, 2, "2026-03-01 01:30:00", 13.3, 13.0),
            ("fUSD", 120, 3, "2026-03-01 07:30:00", 10.6, 10.4),
        ],
        ("o-2", "fUSD", 30, 13.0, "2026-03-01 00:00:00", 48, "PENDING", "2026-03-01 00:00:00"),
    )

    validator = ExecutionValidator(str(db_path))
    result = validator.validate_single_order(
        {
            "order_id": "o-2",
            "currency": "fUSD",
            "period": 30,
            "predicted_rate": 13.0,
            "order_timestamp": "2026-03-01 00:00:00",
            "validation_window_hours": 48,
            "status": "PENDING",
        }
    )

    assert result["status"] == "EXECUTED"
    assert result["path_stage_outcome"] == "FIXED_FILLED"
    assert result["terminal_mode"] == "FIXED"
    assert result["executed_at"] == "2026-03-01 00:30:00"
    assert result["execution_delay_minutes"] == 30

    persisted = _fetch_order_fields(db_path, "o-2")
    assert persisted["path_stage_outcome"] == "FIXED_FILLED"
    assert persisted["terminal_mode"] == "FIXED"
    assert persisted["executed_at"] == "2026-03-01 00:30:00"


def test_validate_single_order_marks_rank6_proxy_when_frr_proxy_missing(tmp_path):
    db_path = tmp_path / "path_labels_rank6.db"
    _init_test_db(
        db_path,
        [
            ("fUSD", 30, 1, "2026-03-01 00:30:00", 13.5, 13.2),
            ("fUSD", 30, 2, "2026-03-01 02:30:00", 13.2, 12.9),
        ],
        ("o-3", "fUSD", 30, 14.0, "2026-03-01 00:00:00", 48, "PENDING", "2026-03-01 00:00:00"),
    )

    validator = ExecutionValidator(str(db_path))
    result = validator.validate_single_order(
        {
            "order_id": "o-3",
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
    assert result["terminal_mode"] == "RANK6_PROXY"
    assert result["stage2_frr_proxy_rate"] == 0.0
