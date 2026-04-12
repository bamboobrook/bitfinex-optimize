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
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "lending_history.db"
DEFAULT_RESULT = Path(__file__).resolve().parent.parent / "data" / "optimal_combination.json"
FRESHNESS_TARGETS = {"fUSD": 300.0, "fUST": 900.0}


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


def format_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def format_num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def fetch_window_metric(conn: sqlite3.Connection, start: datetime, end: datetime) -> WindowMetric:
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) AS executed,
            SUM(CASE WHEN status='FAILED' THEN 1 ELSE 0 END) AS failed,
            SUM(CASE WHEN status='EXPIRED' THEN 1 ELSE 0 END) AS expired,
            AVG(CASE WHEN status='FAILED' THEN rate_gap END) AS avg_failed_gap
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

    rows = conn.execute(
        f"""
        SELECT
            {select_expr} AS grp,
            COUNT(*) AS total,
            SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) AS executed,
            AVG(CASE WHEN status='FAILED' THEN rate_gap END) AS avg_failed_gap
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
        """
        SELECT currency, period, MAX(datetime) AS latest_dt
        FROM funding_rates
        GROUP BY currency, period
        ORDER BY currency, period
        """
    ).fetchall()

    now = datetime.now()
    report = []
    for currency, period, latest_dt in rows:
        age_minutes = None
        status = "missing"
        if latest_dt:
            latest = datetime.strptime(latest_dt, "%Y-%m-%d %H:%M:%S")
            age_minutes = max(0.0, (now - latest).total_seconds() / 60.0)
            status = "fresh" if age_minutes <= FRESHNESS_TARGETS.get(currency, 300.0) else "stale"
        report.append({
            "currency": currency,
            "period": int(period),
            "latest": latest_dt,
            "age_minutes": age_minutes,
            "status": status,
        })
    return report


def fetch_prediction_history_status(conn: sqlite3.Connection) -> dict:
    row = conn.execute(
        """
        SELECT COUNT(*), MIN(created_at), MAX(created_at), COUNT(DISTINCT update_cycle_id)
        FROM prediction_history
        """
    ).fetchone()
    latest_cycles = conn.execute(
        """
        SELECT update_cycle_id, COUNT(*) AS rows, MAX(created_at) AS created_at
        FROM prediction_history
        GROUP BY update_cycle_id
        ORDER BY created_at DESC
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
    realized_value_col = "realized_terminal_value" if "realized_terminal_value" in columns else None
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
        if max(recent_total, prev_total) < min_total:
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
    parser.add_argument("--days", type=int, default=7, help="对比窗口天数")
    parser.add_argument("--min-combo-orders", type=int, default=5, help="组合对比最小订单数")
    args = parser.parse_args()

    db_path = Path(args.db)
    result_path = Path(args.result)
    now = datetime.now()
    recent_start = now - timedelta(days=args.days)
    prev_start = now - timedelta(days=args.days * 2)

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
        f"- avg_realized_terminal_value: {format_num(previous_path['avg_realized_terminal_value'], 3)} -> "
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


if __name__ == "__main__":
    main()
