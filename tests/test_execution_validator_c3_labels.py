import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_engine.execution_validator import ExecutionValidator


def _init_validator_db(db_path, funding_rows=None, order_row=None):
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

        if funding_rows:
            conn.executemany(
                "INSERT INTO funding_rates(currency, period, timestamp, datetime, high_annual, close_annual) VALUES (?, ?, ?, ?, ?, ?)",
                funding_rows,
            )

        order_row = order_row or (
            "o-1",
            "fUSD",
            30,
            11.0,
            "2026-04-01 00:00:00",
            48,
            "PENDING",
            "2026-04-01 00:00:00",
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


def _fetch_c3_fields(db_path, order_id):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT data_quality_label, validation_label, realized_terminal_mode, realized_terminal_value, realized_wait_hours
            FROM virtual_orders
            WHERE order_id = ?
            """,
            (order_id,),
        ).fetchone()
        return dict(row)
    finally:
        conn.close()


def test_validate_single_order_marks_no_market_data_as_censored_not_negative(tmp_path):
    db_path = tmp_path / "validator_c3.db"
    _init_validator_db(db_path)

    validator = ExecutionValidator(str(db_path))
    result = validator.validate_single_order({
        "order_id": "o-1",
        "currency": "fUSD",
        "period": 30,
        "predicted_rate": 11.0,
        "order_timestamp": "2026-04-01 00:00:00",
        "validation_window_hours": 48,
        "status": "PENDING",
    })

    assert result["status"] == "EXPIRED"
    assert result["data_quality_label"] == "CENSORED"
    assert result["validation_label"] == "UNKNOWN"
    assert result["realized_terminal_mode"] == "UNKNOWN"

    persisted = _fetch_c3_fields(db_path, "o-1")
    assert persisted["data_quality_label"] == "CENSORED"
    assert persisted["validation_label"] == "UNKNOWN"
    assert persisted["realized_terminal_mode"] == "UNKNOWN"


def test_validate_single_order_marks_stage1_fill_as_strong_label(tmp_path):
    db_path = tmp_path / "validator_c3_stage1.db"
    _init_validator_db(
        db_path,
        funding_rows=[
            ("fUSD", 30, 1, "2026-03-01 00:30:00", 13.5, 13.2),
            ("fUSD", 30, 2, "2026-03-01 01:30:00", 13.3, 13.0),
            ("fUSD", 120, 3, "2026-03-01 07:30:00", 10.6, 10.4),
        ],
        order_row=("o-2", "fUSD", 30, 13.0, "2026-03-01 00:00:00", 48, "PENDING", "2026-03-01 00:00:00"),
    )

    validator = ExecutionValidator(str(db_path))
    result = validator.validate_single_order({
        "order_id": "o-2",
        "currency": "fUSD",
        "period": 30,
        "predicted_rate": 13.0,
        "order_timestamp": "2026-03-01 00:00:00",
        "validation_window_hours": 48,
        "status": "PENDING",
    })

    assert result["status"] == "EXECUTED"
    assert result["data_quality_label"] == "STRONG"
    assert result["validation_label"] == "PATH_STAGE1_FILLED"
    assert result["realized_terminal_mode"] == "FIXED"
    assert result["realized_terminal_value"] == 13.0
    assert result["realized_wait_hours"] == 0.5

    persisted = _fetch_c3_fields(db_path, "o-2")
    assert persisted["data_quality_label"] == "STRONG"
    assert persisted["validation_label"] == "PATH_STAGE1_FILLED"
    assert persisted["realized_terminal_mode"] == "FIXED"
    assert persisted["realized_terminal_value"] == 13.0
    assert persisted["realized_wait_hours"] == 0.5


def test_validate_single_order_marks_stage2_proxy_as_weak_proxy_label(tmp_path):
    db_path = tmp_path / "validator_c3_stage2.db"
    _init_validator_db(
        db_path,
        funding_rows=[
            ("fUSD", 30, 1, "2026-03-01 00:30:00", 13.5, 13.2),
            ("fUSD", 30, 2, "2026-03-01 02:30:00", 13.2, 12.9),
            ("fUSD", 120, 3, "2026-03-01 07:30:00", 10.6, 10.4),
            ("fUSD", 120, 4, "2026-03-01 08:30:00", 10.5, 10.3),
        ],
        order_row=("o-3", "fUSD", 30, 14.0, "2026-03-01 00:00:00", 48, "PENDING", "2026-03-01 00:00:00"),
    )

    validator = ExecutionValidator(str(db_path))
    result = validator.validate_single_order({
        "order_id": "o-3",
        "currency": "fUSD",
        "period": 30,
        "predicted_rate": 14.0,
        "order_timestamp": "2026-03-01 00:00:00",
        "validation_window_hours": 48,
        "status": "PENDING",
    })

    assert result["status"] == "FAILED"
    assert result["data_quality_label"] == "WEAK_PROXY"
    assert result["validation_label"] == "PATH_STAGE2_PROXY"
    assert result["realized_terminal_mode"] == "FRR_PROXY"
    assert result["realized_terminal_value"] > 0.0
    assert result["realized_wait_hours"] == 12.0

    persisted = _fetch_c3_fields(db_path, "o-3")
    assert persisted["data_quality_label"] == "WEAK_PROXY"
    assert persisted["validation_label"] == "PATH_STAGE2_PROXY"
    assert persisted["realized_terminal_mode"] == "FRR_PROXY"
    assert persisted["realized_terminal_value"] > 0.0
    assert persisted["realized_wait_hours"] == 12.0


def test_validate_single_order_marks_rank6_proxy_with_explicit_stage3_label(tmp_path):
    db_path = tmp_path / "validator_c3_rank6.db"
    _init_validator_db(
        db_path,
        funding_rows=[
            ("fUSD", 30, 1, "2026-03-01 00:30:00", 13.5, 13.2),
            ("fUSD", 30, 2, "2026-03-01 02:30:00", 13.2, 12.9),
        ],
        order_row=("o-4", "fUSD", 30, 14.0, "2026-03-01 00:00:00", 48, "PENDING", "2026-03-01 00:00:00"),
    )

    validator = ExecutionValidator(str(db_path))
    result = validator.validate_single_order({
        "order_id": "o-4",
        "currency": "fUSD",
        "period": 30,
        "predicted_rate": 14.0,
        "order_timestamp": "2026-03-01 00:00:00",
        "validation_window_hours": 48,
        "status": "PENDING",
    })

    assert result["status"] == "FAILED"
    assert result["data_quality_label"] == "WEAK_PROXY"
    assert result["validation_label"] == "PATH_STAGE3_RANK6_PROXY"
    assert result["realized_terminal_mode"] == "RANK6_PROXY"
    assert result["realized_terminal_value"] is None
    assert result["realized_wait_hours"] == 12.0

    persisted = _fetch_c3_fields(db_path, "o-4")
    assert persisted["data_quality_label"] == "WEAK_PROXY"
    assert persisted["validation_label"] == "PATH_STAGE3_RANK6_PROXY"
    assert persisted["realized_terminal_mode"] == "RANK6_PROXY"
    assert persisted["realized_terminal_value"] is None
    assert persisted["realized_wait_hours"] == 12.0
