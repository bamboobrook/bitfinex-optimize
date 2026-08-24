#!/usr/bin/env python3
"""
本地优化效果评估脚本

用途:
1. 对比最近 N 天 vs 前 N 天整体/分币种/分组合执行率
2. 输出 funding_rates 新鲜度摘要
3. 检查 prediction_history 是否持续落库
4. 展示当前结果文件的 stale 状态与 Top 推荐
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_engine.system_policy import (
    get_freshness_thresholds_minutes,
    load_system_policy,
)


DEFAULT_DB = PROJECT_ROOT / "data" / "lending_history.db"
DEFAULT_RESULT = PROJECT_ROOT / "data" / "optimal_combination.json"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "data" / "models"
SUPPORTED_CURRENCIES = {"fUSD", "fUST"}
SUPPORTED_PERIODS = {2, 3, 4, 5, 6, 7, 10, 14, 15, 20, 30, 60, 90, 120}


@dataclass
class WindowMetric:
    total: int = 0
    executed: int = 0
    failed: int = 0
    expired: int = 0
    avg_failed_gap: float | None = None

    @property
    def execution_rate(self) -> float:
        return (self.executed / self.total) if self.total else 0.0


def infer_model_deployment_times(model_dir: Path) -> dict[str, datetime | None]:
    deployed = {currency: None for currency in sorted(SUPPORTED_CURRENCIES)}
    history_path = model_dir.parent / "retraining_history.json"
    if history_path.exists():
        try:
            raw_history = json.loads(history_path.read_text(encoding="utf-8"))
            entries = (
                raw_history
                if isinstance(raw_history, list)
                else list(raw_history.values())
                if isinstance(raw_history, dict)
                else []
            )
            for entry in reversed(entries):
                if not isinstance(entry, dict) or entry.get("deployed") is not True:
                    continue
                selected = entry.get("deployed_currencies")
                try:
                    timestamp = datetime.strptime(
                        entry.get("timestamp", ""), "%Y-%m-%d %H:%M:%S"
                    )
                except (TypeError, ValueError):
                    continue
                for currency in deployed:
                    if deployed[currency] is not None:
                        continue
                    if isinstance(selected, list) and currency not in selected:
                        continue
                    deployed[currency] = timestamp
        except (OSError, json.JSONDecodeError):
            pass

    for currency in deployed:
        if deployed[currency] is not None:
            continue
        meta_files = list(model_dir.glob(f"{currency}_*_meta.json"))
        if meta_files:
            deployed[currency] = datetime.fromtimestamp(
                max(path.stat().st_mtime for path in meta_files)
            )
    return deployed


def infer_model_deployment_time(model_dir: Path) -> datetime | None:
    """Backward-compatible latest deployment helper."""
    values = [value for value in infer_model_deployment_times(model_dir).values() if value]
    return max(values) if values else None


def _binary_auc(labels: list[int], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _expected_calibration_error(
    labels: list[int],
    scores: list[float],
    bins: int = 10,
) -> float | None:
    if not labels:
        return None
    total = len(labels)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = [
            (label, score)
            for label, score in zip(labels, scores)
            if lower <= score < upper or (index == bins - 1 and score == 1.0)
        ]
        if not selected:
            continue
        observed = sum(label for label, _ in selected) / len(selected)
        predicted = sum(score for _, score in selected) / len(selected)
        error += len(selected) / total * abs(observed - predicted)
    return error


def _probability_metrics(labels: list[int], scores: list[float]) -> dict:
    if not labels:
        return {"mean": None, "auc": None, "brier": None, "ece": None}
    clipped = [max(0.0, min(1.0, float(score))) for score in scores]
    return {
        "mean": sum(clipped) / len(clipped),
        "auc": _binary_auc(labels, clipped),
        "brier": sum((score - label) ** 2 for label, score in zip(labels, clipped)) / len(labels),
        "ece": _expected_calibration_error(labels, clipped),
    }


def _pearson(values_x: list[float], values_y: list[float]) -> float | None:
    if len(values_x) < 3 or len(values_x) != len(values_y):
        return None
    mean_x = sum(values_x) / len(values_x)
    mean_y = sum(values_y) / len(values_y)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(values_x, values_y))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in values_x))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in values_y))
    if denom_x <= 1e-12 or denom_y <= 1e-12:
        return None
    return numerator / (denom_x * denom_y)


def fetch_deployment_cohort_metrics(
    conn: sqlite3.Connection,
    deployed_at: datetime,
    currency: str | None = None,
) -> dict:
    deployed_text = deployed_at.strftime("%Y-%m-%d %H:%M:%S")
    virtual_columns = _get_virtual_order_columns(conn)
    prediction_columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(prediction_history)").fetchall()
    }
    stage1_select = (
        "v.stage1_fill_probability"
        if "stage1_fill_probability" in virtual_columns
        else "NULL"
    )
    traditional_select = (
        "p.traditional_execution_probability"
        if "traditional_execution_probability" in prediction_columns
        else "NULL"
    )
    v2_select = (
        "p.v2_execution_probability"
        if "v2_execution_probability" in prediction_columns
        else "NULL"
    )
    model_version_select = (
        "v.model_version"
        if "model_version" in virtual_columns
        else "NULL"
    )
    order_market_select = (
        "COALESCE(p.current_market_rate, v.market_median)"
        if "current_market_rate" in prediction_columns
        else "v.market_median"
    )
    realized_value_select = (
        _net_terminal_value_expression(virtual_columns, prefix="v.")
        if "realized_terminal_value" in virtual_columns
        else "NULL"
    )
    currency_clause = " AND v.currency = ?" if currency else ""
    params = [deployed_text]
    if currency:
        params.append(currency)
    rows = conn.execute(
        f"""
        SELECT
            v.status,
            v.validation_window_hours,
            v.predicted_rate,
            {order_market_select} AS order_market_rate,
            v.path_value_score,
            {realized_value_select} AS realized_terminal_value_net_wait,
            p.execution_probability,
            COALESCE(
                p.calibrated_execution_probability,
                p.execution_probability
            ) AS calibrated_probability,
            {stage1_select} AS stage1_probability,
            {traditional_select} AS traditional_probability,
            {v2_select} AS v2_probability,
            {model_version_select} AS model_version
        FROM virtual_orders AS v
        LEFT JOIN prediction_history AS p
          ON p.rowid = (
              SELECT p2.rowid
              FROM prediction_history AS p2
              WHERE p2.update_cycle_id = v.update_cycle_id
                AND p2.currency = v.currency
                AND p2.period = v.period
                AND (
                    v.candidate_id IS NULL
                    OR p2.candidate_id = v.candidate_id
                )
              ORDER BY p2.rowid DESC
              LIMIT 1
          )
        WHERE v.created_at >= ? {currency_clause}
        """,
        tuple(params),
    ).fetchall()

    status_counts = defaultdict(int)
    maturity_by_window = defaultdict(lambda: {"total": 0, "decided": 0})
    labels = []
    raw_scores = []
    calibrated_scores = []
    stage1_labels = []
    stage1_scores = []
    traditional_labels = []
    traditional_scores = []
    v2_labels = []
    v2_scores = []
    model_versions = defaultdict(int)
    realized_values = []
    market_values = []
    predicted_values = []
    path_values = []

    for row in rows:
        (
            status,
            window_hours,
            predicted_rate,
            market_median,
            path_value,
            realized_value,
            raw_probability,
            calibrated_probability,
            stage1_probability,
            traditional_probability,
            v2_probability,
            model_version,
        ) = row
        status_counts[status] += 1
        model_versions[model_version or "unknown"] += 1
        window = int(window_hours or 0)
        maturity_by_window[window]["total"] += 1
        if status in {"EXECUTED", "FAILED"}:
            maturity_by_window[window]["decided"] += 1

        if status in {"EXECUTED", "FAILED"}:
            label = 1 if status == "EXECUTED" else 0
            if raw_probability is not None and calibrated_probability is not None:
                labels.append(label)
                raw_scores.append(float(raw_probability))
                calibrated_scores.append(float(calibrated_probability))
            if stage1_probability is not None:
                stage1_labels.append(label)
                stage1_scores.append(float(stage1_probability))
            if traditional_probability is not None:
                traditional_labels.append(label)
                traditional_scores.append(float(traditional_probability))
            if v2_probability is not None:
                v2_labels.append(label)
                v2_scores.append(float(v2_probability))

        if (
            status in {"EXECUTED", "FAILED"}
            and realized_value is not None
            and market_median is not None
        ):
            realized_values.append(float(realized_value))
            market_values.append(float(market_median))
            predicted_values.append(float(predicted_rate))
            path_values.append(float(path_value) if path_value is not None else math.nan)

    decided = sum(status_counts[name] for name in ("EXECUTED", "FAILED"))
    path_pairs = [
        (path, realized)
        for path, realized in zip(path_values, realized_values)
        if math.isfinite(path)
    ]
    return {
        "deployed_at": deployed_text,
        "currency": currency,
        "total": len(rows),
        "decided": decided,
        "pending": status_counts["PENDING"],
        "executed": status_counts["EXECUTED"],
        "failed": status_counts["FAILED"],
        "expired": status_counts["EXPIRED"],
        "model_versions": dict(sorted(model_versions.items())),
        "execution_rate": (
            status_counts["EXECUTED"] / (status_counts["EXECUTED"] + status_counts["FAILED"])
            if status_counts["EXECUTED"] + status_counts["FAILED"]
            else None
        ),
        "maturity_by_window": dict(sorted(maturity_by_window.items())),
        "raw_probability": _probability_metrics(labels, raw_scores),
        "calibrated_probability": _probability_metrics(labels, calibrated_scores),
        "stage1_probability": _probability_metrics(stage1_labels, stage1_scores),
        "traditional_probability": _probability_metrics(
            traditional_labels,
            traditional_scores,
        ),
        "v2_probability": _probability_metrics(v2_labels, v2_scores),
        "value_samples": len(realized_values),
        "avg_realized_value": (
            sum(realized_values) / len(realized_values) if realized_values else None
        ),
        "realized_premium_vs_market": (
            sum(realized - market for realized, market in zip(realized_values, market_values))
            / len(realized_values)
            if realized_values
            else None
        ),
        "predicted_premium_vs_market": (
            sum(predicted - market for predicted, market in zip(predicted_values, market_values))
            / len(predicted_values)
            if predicted_values
            else None
        ),
        "path_mae": (
            sum(abs(path - realized) for path, realized in path_pairs) / len(path_pairs)
            if path_pairs
            else None
        ),
        "path_pearson": _pearson(
            [path for path, _ in path_pairs],
            [realized for _, realized in path_pairs],
        ),
    }


def format_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def format_num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def fetch_window_metric(conn: sqlite3.Connection, start: datetime, end: datetime) -> WindowMetric:
    gap_expr = _failed_gap_expression(_get_virtual_order_columns(conn))
    row = conn.execute(
        f"""
        SELECT
            SUM(CASE WHEN status IN ('EXECUTED', 'FAILED') THEN 1 ELSE 0 END) AS total,
            SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) AS executed,
            SUM(CASE WHEN status='FAILED' THEN 1 ELSE 0 END) AS failed,
            SUM(CASE WHEN status='EXPIRED' THEN 1 ELSE 0 END) AS expired,
            AVG(CASE WHEN status='FAILED' THEN {gap_expr} END) AS avg_failed_gap
        FROM virtual_orders
        WHERE order_timestamp >= ?
          AND order_timestamp < ?
          AND status IN ('EXECUTED', 'FAILED', 'EXPIRED')
        """,
        (start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S")),
    ).fetchone()
    return WindowMetric(
        total=row[0] or 0,
        executed=row[1] or 0,
        failed=row[2] or 0,
        expired=row[3] or 0,
        avg_failed_gap=row[4],
    )


def fetch_group_metrics(conn: sqlite3.Connection, start: datetime, end: datetime, group_by: str) -> dict:
    if group_by not in {"currency", "combo"}:
        raise ValueError(f"Unsupported group_by: {group_by}")

    if group_by == "currency":
        select_expr = "currency"
        group_expr = "currency"
    else:
        select_expr = "currency || '-' || period || 'd'"
        group_expr = "currency, period"

    gap_expr = _failed_gap_expression(_get_virtual_order_columns(conn))
    rows = conn.execute(
        f"""
        SELECT
            {select_expr} AS grp,
            SUM(CASE WHEN status IN ('EXECUTED', 'FAILED') THEN 1 ELSE 0 END) AS total,
            SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) AS executed,
            AVG(CASE WHEN status='FAILED' THEN {gap_expr} END) AS avg_failed_gap
        FROM virtual_orders
        WHERE order_timestamp >= ?
          AND order_timestamp < ?
          AND status IN ('EXECUTED', 'FAILED', 'EXPIRED')
        GROUP BY {group_expr}
        """,
        (start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S")),
    ).fetchall()

    result = {}
    for group_name, total, executed, avg_failed_gap in rows:
        metric = WindowMetric(
            total=total or 0,
            executed=executed or 0,
            failed=0,
            expired=0,
            avg_failed_gap=avg_failed_gap,
        )
        result[group_name] = metric
    return result


def fetch_freshness(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        f"""
        SELECT currency, period, MAX(datetime) AS latest_dt
        FROM funding_rates
        WHERE currency IN ({','.join('?' for _ in SUPPORTED_CURRENCIES)})
          AND period IN ({','.join('?' for _ in SUPPORTED_PERIODS)})
        GROUP BY currency, period
        ORDER BY currency, period
        """,
        (*sorted(SUPPORTED_CURRENCIES), *sorted(SUPPORTED_PERIODS)),
    ).fetchall()

    now = datetime.now()
    policy = load_system_policy()
    report = []
    for currency, period, latest_dt in rows:
        age_minutes = None
        status = "missing"
        if latest_dt:
            latest = datetime.strptime(latest_dt, "%Y-%m-%d %H:%M:%S")
            age_minutes = max(0.0, (now - latest).total_seconds() / 60.0)
            _, hard_minutes = get_freshness_thresholds_minutes(
                policy, currency, int(period)
            )
            status = "fresh" if age_minutes <= hard_minutes else "stale"
        report.append({
            "currency": currency,
            "period": int(period),
            "latest": latest_dt,
            "age_minutes": age_minutes,
            "hard_minutes": hard_minutes if latest_dt else None,
            "status": status,
        })
    return report


def fetch_prediction_history_status(conn: sqlite3.Connection) -> dict:
    row = conn.execute(
        """
        SELECT COUNT(*), MIN(prediction_timestamp), MAX(prediction_timestamp),
               COUNT(DISTINCT update_cycle_id)
        FROM prediction_history
        """
    ).fetchone()
    latest_cycles = conn.execute(
        """
        SELECT update_cycle_id, COUNT(*) AS rows,
               MAX(prediction_timestamp) AS prediction_timestamp
        FROM prediction_history
        GROUP BY update_cycle_id
        ORDER BY prediction_timestamp DESC
        LIMIT 3
        """
    ).fetchall()
    return {
        "count": row[0] or 0,
        "min_created_at": row[1],
        "max_created_at": row[2],
        "cycle_count": row[3] or 0,
        "latest_cycles": latest_cycles,
    }


def _get_virtual_order_columns(conn: sqlite3.Connection) -> set[str]:
    return {row[1] for row in conn.execute("PRAGMA table_info(virtual_orders)").fetchall()}


def _failed_gap_expression(columns: set[str]) -> str:
    if "stage1_rate_gap" in columns:
        return (
            "COALESCE(stage1_rate_gap, "
            "CASE WHEN max_market_rate IS NOT NULL "
            "THEN predicted_rate - max_market_rate END)"
        )
    if "max_market_rate" in columns:
        return "predicted_rate - max_market_rate"
    return "rate_gap"


def _net_terminal_value_expression(columns: set[str], prefix: str = "") -> str:
    col = lambda name: f"{prefix}{name}"
    if "realized_terminal_value_net_wait" in columns:
        return f"COALESCE({col('realized_terminal_value_net_wait')}, {col('realized_terminal_value')})"
    if {"period", "realized_wait_hours"}.issubset(columns):
        return (
            f"CASE WHEN {col('realized_terminal_value')} IS NOT NULL THEN "
            f"{col('realized_terminal_value')} * ({col('period')} * 24.0) / "
            f"NULLIF({col('period')} * 24.0 + COALESCE({col('realized_wait_hours')}, 0.0), 0.0) END"
        )
    return col("realized_terminal_value")


def fetch_path_metrics(conn: sqlite3.Connection, start: datetime, end: datetime) -> dict:
    columns = _get_virtual_order_columns(conn)
    path_value_expr = "AVG(path_value_score)" if "path_value_score" in columns else "NULL"
    stage1_fill_expr = "AVG(stage1_fill_probability)" if "stage1_fill_probability" in columns else "NULL"
    realized_mode_col = (
        "realized_terminal_mode"
        if "realized_terminal_mode" in columns
        else ("terminal_mode" if "terminal_mode" in columns else None)
    )
    expected_mode_col = "expected_terminal_mode" if "expected_terminal_mode" in columns else None
    realized_value_col = (
        _net_terminal_value_expression(columns)
        if "realized_terminal_value" in columns
        else None
    )
    params = (start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S"))

    row = conn.execute(
        f"""
        SELECT
            {path_value_expr},
            {stage1_fill_expr},
            {"AVG(CASE WHEN " + realized_mode_col + "='FRR_PROXY' THEN 1 ELSE 0 END)" if realized_mode_col else "0.0"},
            {"AVG(CASE WHEN " + realized_mode_col + "='RANK6_PROXY' THEN 1 ELSE 0 END)" if realized_mode_col else "0.0"},
            {"AVG(CASE WHEN " + realized_mode_col + " IS NOT NULL THEN 1 ELSE 0 END)" if realized_mode_col else "0.0"},
            {f"AVG({realized_value_col})" if realized_value_col else "NULL"}
        FROM virtual_orders
        WHERE order_timestamp >= ?
          AND order_timestamp < ?
          AND status IN ('EXECUTED', 'FAILED', 'EXPIRED')
        """,
        params,
    ).fetchone()

    terminal_mode_matrix = {}
    if expected_mode_col and realized_mode_col:
        matrix_rows = conn.execute(
            f"""
            SELECT
                {expected_mode_col} AS expected_mode,
                {realized_mode_col} AS realized_mode,
                COUNT(*) AS total
            FROM virtual_orders
            WHERE order_timestamp >= ?
              AND order_timestamp < ?
              AND status IN ('EXECUTED', 'FAILED', 'EXPIRED')
              AND {expected_mode_col} IS NOT NULL
              AND {realized_mode_col} IS NOT NULL
            GROUP BY {expected_mode_col}, {realized_mode_col}
            ORDER BY {expected_mode_col}, {realized_mode_col}
            """,
            params,
        ).fetchall()
        terminal_mode_matrix = {
            f"{expected_mode}->{realized_mode}": total
            for expected_mode, realized_mode, total in matrix_rows
        }

    return {
        "avg_path_value_score": row[0] or 0.0,
        "avg_stage1_fill_probability": row[1] or 0.0,
        "frr_terminal_ratio": row[2] or 0.0,
        "rank6_terminal_ratio": row[3] or 0.0,
        "path_label_coverage": row[4] or 0.0,
        "avg_realized_terminal_value": row[5],
        "terminal_mode_matrix": terminal_mode_matrix,
    }


def load_result_file(result_path: Path) -> dict | None:
    if not result_path.exists():
        return None
    with result_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_combo_delta(recent: dict, previous: dict, min_total: int) -> list[dict]:
    combos = sorted(set(recent) | set(previous))
    deltas = []
    for combo in combos:
        recent_metric = recent.get(combo)
        prev_metric = previous.get(combo)
        recent_total = recent_metric.total if recent_metric else 0
        prev_total = prev_metric.total if prev_metric else 0
        # Avoid presenting an immature/missing cohort as a real 0% regression.
        if recent_total < min_total or prev_total < min_total:
            continue
        recent_rate = recent_metric.execution_rate if recent_metric else 0.0
        prev_rate = prev_metric.execution_rate if prev_metric else 0.0
        deltas.append({
            "combo": combo,
            "recent_rate": recent_rate,
            "prev_rate": prev_rate,
            "delta": recent_rate - prev_rate,
            "recent_total": recent_total,
            "prev_total": prev_total,
        })
    return deltas


def print_window_delta(title: str, recent: WindowMetric, previous: WindowMetric):
    print(title)
    print(
        f"- 执行率: {format_pct(previous.execution_rate)} -> {format_pct(recent.execution_rate)} "
        f"({recent.executed}/{recent.total} vs {previous.executed}/{previous.total})"
    )
    print(
        f"- 失败单平均价差: {format_num(previous.avg_failed_gap, 2)} -> "
        f"{format_num(recent.avg_failed_gap, 2)}"
    )


def main():
    parser = argparse.ArgumentParser(description="评估最近优化效果")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite 数据库路径")
    parser.add_argument("--result", default=str(DEFAULT_RESULT), help="预测结果 JSON 路径")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="生产模型目录")
    parser.add_argument(
        "--deployment-at",
        help="显式指定模型部署时间 (YYYY-MM-DD HH:MM:SS)，默认从 meta mtime 推断",
    )
    parser.add_argument("--days", type=int, default=7, help="对比窗口天数")
    parser.add_argument("--min-combo-orders", type=int, default=5, help="组合对比最小订单数")
    args = parser.parse_args()

    db_path = Path(args.db)
    result_path = Path(args.result)
    model_dir = Path(args.model_dir)
    now = datetime.now()
    recent_start = now - timedelta(days=args.days)
    prev_start = now - timedelta(days=args.days * 2)

    if args.deployment_at:
        explicit_deployed_at = datetime.strptime(
            args.deployment_at, "%Y-%m-%d %H:%M:%S"
        )
        deployed_by_currency = {
            currency: explicit_deployed_at for currency in SUPPORTED_CURRENCIES
        }
    else:
        deployed_by_currency = infer_model_deployment_times(model_dir)

    with sqlite3.connect(db_path) as conn:
        recent_overall = fetch_window_metric(conn, recent_start, now)
        previous_overall = fetch_window_metric(conn, prev_start, recent_start)

        recent_currency = fetch_group_metrics(conn, recent_start, now, "currency")
        previous_currency = fetch_group_metrics(conn, prev_start, recent_start, "currency")

        recent_combo = fetch_group_metrics(conn, recent_start, now, "combo")
        previous_combo = fetch_group_metrics(conn, prev_start, recent_start, "combo")

        freshness = fetch_freshness(conn)
        history_status = fetch_prediction_history_status(conn)
        recent_path = fetch_path_metrics(conn, recent_start, now)
        previous_path = fetch_path_metrics(conn, prev_start, recent_start)
        deployment_metrics = {
            currency: (
                fetch_deployment_cohort_metrics(
                    conn, deployed_at, currency=currency
                )
                if deployed_at is not None else None
            )
            for currency, deployed_at in deployed_by_currency.items()
        }

    result_file = load_result_file(result_path)
    combo_deltas = build_combo_delta(recent_combo, previous_combo, args.min_combo_orders)
    combo_deltas_sorted = sorted(combo_deltas, key=lambda x: x["delta"], reverse=True)

    stale_items = [item for item in freshness if item["status"] != "fresh"]
    stale_by_currency = defaultdict(int)
    for item in stale_items:
        stale_by_currency[item["currency"]] += 1

    print("=== 优化效果评估 ===")
    print(f"- 数据库: {db_path}")
    print(f"- 时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"- 对比窗口: 最近 {args.days} 天 vs 前 {args.days} 天")
    print()

    print_window_delta("一、整体表现", recent_overall, previous_overall)
    print()

    print("二、按币种表现")
    for currency in ["fUSD", "fUST"]:
        recent_metric = recent_currency.get(currency, WindowMetric())
        previous_metric = previous_currency.get(currency, WindowMetric())
        print(
            f"- {currency}: {format_pct(previous_metric.execution_rate)} -> "
            f"{format_pct(recent_metric.execution_rate)} "
            f"({recent_metric.executed}/{recent_metric.total} vs {previous_metric.executed}/{previous_metric.total})"
        )
    print()

    print("三、分组合变化")
    print("- 改善最多")
    for item in combo_deltas_sorted[:5]:
        print(
            f"  {item['combo']}: {format_pct(item['prev_rate'])} -> {format_pct(item['recent_rate'])} "
            f"(Δ {item['delta'] * 100:+.1f}pct)"
        )
    print("- 退化最多")
    for item in list(reversed(combo_deltas_sorted[-5:])):
        print(
            f"  {item['combo']}: {format_pct(item['prev_rate'])} -> {format_pct(item['recent_rate'])} "
            f"(Δ {item['delta'] * 100:+.1f}pct)"
        )
    print()

    print("四、市场数据新鲜度")
    print(
        f"- stale 组合: {len(stale_items)}/{len(freshness)} "
        f"(fUSD={stale_by_currency.get('fUSD', 0)}, fUST={stale_by_currency.get('fUST', 0)})"
    )
    for item in stale_items[:10]:
        age = "N/A" if item["age_minutes"] is None else f"{item['age_minutes']:.0f} min"
        print(f"  {item['currency']}-{item['period']}d: {item['latest']} (age={age})")
    print()

    print("五、prediction_history 落库")
    print(
        f"- 总记录: {history_status['count']}, 周期数: {history_status['cycle_count']}, "
        f"时间范围: {history_status['min_created_at']} -> {history_status['max_created_at']}"
    )
    for cycle_id, rows, created_at in history_status["latest_cycles"]:
        print(f"  cycle={cycle_id} rows={rows} created_at={created_at}")
    print()

    print("六、当前结果文件")
    if not result_file:
        print("- 未找到结果文件")
    else:
        print(
            f"- stale_data={result_file.get('stale_data')} "
            f"stale_minutes={result_file.get('stale_minutes')} "
            f"policy_version={result_file.get('policy_version')}"
        )
        recommendations = result_file.get("recommendations", [])[:5]
        for item in recommendations:
            print(
                f"  rank{item.get('rank')}: {item.get('currency')}-{item.get('period')}d "
                f"rate={item.get('rate')} confidence={item.get('confidence')}"
            )
    print()

    print("七、路径质量")
    print(
        f"- path_value_score: {format_num(previous_path['avg_path_value_score'], 3)} -> "
        f"{format_num(recent_path['avg_path_value_score'], 3)}"
    )
    print(
        f"- stage1_fill_probability: {format_pct(previous_path['avg_stage1_fill_probability'])} -> "
        f"{format_pct(recent_path['avg_stage1_fill_probability'])}"
    )
    print(
        f"- terminal FRR ratio: {format_pct(previous_path['frr_terminal_ratio'])} -> "
        f"{format_pct(recent_path['frr_terminal_ratio'])}"
    )
    print(
        f"- terminal rank6 ratio: {format_pct(previous_path['rank6_terminal_ratio'])} -> "
        f"{format_pct(recent_path['rank6_terminal_ratio'])}"
    )
    print(
        f"- path_label_coverage: {format_pct(previous_path['path_label_coverage'])} -> "
        f"{format_pct(recent_path['path_label_coverage'])}"
    )
    print(
        f"- avg_realized_terminal_value_net_wait: {format_num(previous_path['avg_realized_terminal_value'], 3)} -> "
        f"{format_num(recent_path['avg_realized_terminal_value'], 3)}"
    )
    print(
        f"- terminal_mode_matrix(prev): "
        f"{json.dumps(previous_path['terminal_mode_matrix'], ensure_ascii=False, sort_keys=True)}"
    )
    print(
        f"- terminal_mode_matrix(recent): "
        f"{json.dumps(recent_path['terminal_mode_matrix'], ensure_ascii=False, sort_keys=True)}"
    )
    print()

    print("八、当前生产模型成熟度")
    if not any(deployment_metrics.values()):
        print("- 无法识别生产模型部署时间")
    for currency in ["fUSD", "fUST"]:
        metric = deployment_metrics.get(currency)
        deployed_at = deployed_by_currency.get(currency)
        if metric is None or deployed_at is None:
            print(f"- {currency}: 无法识别生产模型部署时间")
            continue
        model_age_hours = (now - deployed_at).total_seconds() / 3600.0
        print(
            f"- {currency} 部署时间: {metric['deployed_at']} (age={model_age_hours:.1f}h), "
            f"成熟: {metric['decided']}/{metric['total']}, pending={metric['pending']}"
        )
        print(
            f"- 模型版本分布: "
            f"{json.dumps(metric['model_versions'], ensure_ascii=False, sort_keys=True)}"
        )
        for window, counts in metric["maturity_by_window"].items():
            print(
                f"  {window}h: {counts['decided']}/{counts['total']} 已决"
            )
        print(
            f"- 已决成交率: {format_pct(metric['execution_rate'])} "
            f"({metric['executed']}/{metric['executed'] + metric['failed']})"
        )
        for label, key in [
            ("生产主排序原始", "raw_probability"),
            ("生产校准概率", "calibrated_probability"),
            ("候选Stage1", "stage1_probability"),
            ("传统模型", "traditional_probability"),
            ("v2模型", "v2_probability"),
        ]:
            probability = metric[key]
            if probability["mean"] is None:
                continue
            print(
                f"- {label}: mean={format_pct(probability['mean'])}, "
                f"AUC={format_num(probability['auc'], 3)}, "
                f"Brier={format_num(probability['brier'], 4)}, "
                f"ECE={format_pct(probability['ece'])}"
            )
        print(
            f"- 兑现溢价/下单时市场: {format_num(metric['realized_premium_vs_market'], 3)}, "
            f"预测溢价/下单时市场: {format_num(metric['predicted_premium_vs_market'], 3)}"
        )
        print(
            f"- 路径 MAE={format_num(metric['path_mae'], 3)}, "
            f"Pearson={format_num(metric['path_pearson'], 3)}, "
            f"samples={metric['value_samples']}"
        )


if __name__ == "__main__":
    main()
