import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_engine.training_data_builder import TrainingDataBuilder


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


def test_build_training_data_excludes_censored_and_probe_rows(tmp_path):
    db_path = tmp_path / "db.sqlite"
    _seed_training_db(db_path)
    builder = TrainingDataBuilder(str(db_path))

    df = builder.build_training_data("2026-04-01", "2026-04-02", include_execution_results=True)

    assert set(df["decision_mode"].unique()) == {"exploit"}
    assert set(df["data_quality_label"].unique()) == {"STRONG"}
    assert "path_terminal_value" in df.columns
    assert df["path_terminal_value"].notna().all()


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
