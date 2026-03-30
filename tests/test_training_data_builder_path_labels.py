import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_engine.training_data_builder import TrainingDataBuilder


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
                path_value_score REAL,
                stage1_fill_probability REAL,
                stage2_frr_proxy_rate REAL,
                terminal_mode TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO funding_rates(
                currency, period, timestamp, datetime,
                open_annual, close_annual, high_annual, low_annual,
                volume, hour, day_of_week
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "fUSD",
                30,
                1740787200,
                "2026-03-01 00:00:00",
                12.0,
                12.1,
                12.5,
                11.8,
                1000.0,
                0,
                6,
            ),
        )
        conn.execute(
            """
            INSERT INTO virtual_orders(
                order_timestamp, currency, period, predicted_rate, status,
                execution_confidence, total_score, market_median, execution_rate,
                path_value_score, stage1_fill_probability, stage2_frr_proxy_rate, terminal_mode
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "2026-03-01 00:30:00",
                "fUSD",
                30,
                12.8,
                "EXECUTED",
                0.76,
                0.88,
                12.1,
                12.6,
                0.92,
                0.73,
                11.4,
                "FIXED",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def test_build_training_data_includes_path_labels(tmp_path):
    db_path = tmp_path / "training_data_builder_path_labels.db"
    _init_test_db(db_path)

    builder = TrainingDataBuilder(str(db_path))
    result = builder.build_training_data(
        "2026-03-01",
        "2026-03-02",
        include_execution_results=True,
    )

    assert len(result) == 1
    row = result.iloc[0]

    assert "path_value_score" in result.columns
    assert "stage1_fill_probability" in result.columns
    assert "stage2_frr_proxy_rate" in result.columns
    assert "path_terminal_value" in result.columns
    assert "path_stage1_success" in result.columns
    assert row["path_value_score"] == 0.92
    assert row["stage1_fill_probability"] == 0.73
    assert row["stage2_frr_proxy_rate"] == 11.4
    assert row["path_stage1_success"] == 1.0
    assert row["path_terminal_value"] > 0
