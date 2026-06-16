import sqlite3
from pathlib import Path

from ml_engine.data_processor import DataProcessor


def _insert_rate(conn, currency, period, minute):
    ts = 1_700_000_000_000 + minute * 60_000
    conn.execute(
        """
        INSERT INTO funding_rates (
            currency, period, timestamp, datetime,
            open_annual, close_annual, high_annual, low_annual, volume,
            hour, day_of_week, month
        ) VALUES (?, ?, ?, ?, 5.0, 5.0, 5.0, 5.0, 10.0, 0, 0, 1)
        """,
        (currency, period, ts, f"2026-01-01 00:{minute:02d}:00"),
    )


def test_load_data_filters_unsupported_bitfinex_periods(tmp_path: Path):
    db_path = tmp_path / "rates.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE funding_rates (
                currency TEXT,
                period INTEGER,
                timestamp INTEGER,
                datetime TEXT,
                open_annual REAL,
                close_annual REAL,
                high_annual REAL,
                low_annual REAL,
                volume REAL,
                hour INTEGER,
                day_of_week INTEGER,
                month INTEGER
            )
            """
        )
        _insert_rate(conn, "fUSD", 2, 0)
        _insert_rate(conn, "fUSD", 8, 1)
        _insert_rate(conn, "fUSD", 9, 2)
        _insert_rate(conn, "fUSD", 120, 3)
        conn.commit()
    finally:
        conn.close()

    df = DataProcessor(str(db_path)).load_data("fUSD")

    assert sorted(df["period"].unique().tolist()) == [2, 120]
