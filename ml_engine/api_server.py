from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
import json
import os
import asyncio
import logging
from datetime import datetime, timedelta
from loguru import logger
import sys
from pathlib import Path
import sqlite3
import tempfile
import signal
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent.parent))

# Pipeline concurrency lock (L1) - prevent overlapping pipeline runs
_pipeline_lock = asyncio.Lock()

logger.add('/home/bumblebee/Project/optimize/log/ml_optimizer.log', retention='7 days', rotation="10 MB")
# Suppress Uvicorn access logs (e.g. "GET /status 200 OK").
logging.getLogger("uvicorn.access").disabled = True

app = FastAPI(title="Lending Optimization API", version="3.0")

# Use absolute paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = str(BASE_DIR / "data" / "optimal_combination.json")
STATUS_FILE = str(BASE_DIR / "data" / "service_status.json")
DB_FILE = str(BASE_DIR / "data" / "lending_history.db")
RETRAIN_STATE_FILE = str(BASE_DIR / "data" / "retraining_state.json")
RETRAIN_HISTORY_FILE = str(BASE_DIR / "data" / "retraining_history.json")
CALIBRATION_STATE_FILE = str(BASE_DIR / "data" / "v2_calibration_state.json")
_live_gate_status_cache = {"loaded_at": None, "value": {}}


def _atomic_write_json(path: str, payload: dict, indent: int = 4):
    target_dir = os.path.dirname(path) or "."
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-json-", suffix=".json", dir=target_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise

# --- 辅助函数: 状态管理 ---
def update_status(status: str, step: str = "", details: str = ""):
    """更新服务状态文件"""
    info = {
        "status": status,         # 'online'/'processing'/'degraded'/'error'
        "current_step": step,     # 当前正在执行的步骤
        "last_update": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "details": details
    }
    _atomic_write_json(STATUS_FILE, info, indent=4)

def get_current_status():
    """读取当前状态"""
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read status file: {e}")
            pass
    return {"status": "unknown", "details": "Status file not found"}


def load_retraining_state():
    """Load retraining cooldown state from disk."""
    if not os.path.exists(RETRAIN_STATE_FILE):
        return {
            "last_forced_retrain_time": None,
            "last_reason": "",
            "last_outcome": None,
            "rejection_streak": 0,
            "backoff_until": None,
        }
    try:
        with open(RETRAIN_STATE_FILE, 'r') as f:
            state = json.load(f)
        if not isinstance(state, dict):
            return {"last_forced_retrain_time": None, "last_reason": ""}
        return state
    except Exception as e:
        logger.warning(f"Failed to load retraining state: {e}")
        return {"last_forced_retrain_time": None, "last_reason": ""}


def save_retraining_state(
    last_forced_retrain_time: datetime = None,
    reason: str = "",
    outcome: str = None,
    training_data: dict = None,
):
    """Persist retraining cooldown state to disk."""
    state = load_retraining_state()
    state["last_forced_retrain_time"] = (
            last_forced_retrain_time.strftime('%Y-%m-%d %H:%M:%S')
            if last_forced_retrain_time else None
        )
    state["last_reason"] = reason or state.get("last_reason", "")
    state["updated_at"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if outcome:
        state["last_outcome"] = outcome
        if outcome in {"trained_not_better", "trained_not_deployed"}:
            streak = int(state.get("rejection_streak", 0) or 0) + 1
            backoff_hours = 12 if streak == 1 else 24
            state["rejection_streak"] = streak
            state["backoff_until"] = (
                datetime.now() + timedelta(hours=backoff_hours)
            ).strftime('%Y-%m-%d %H:%M:%S')
        elif outcome in {"deployed", "partially_deployed", "rolled_back"}:
            state["rejection_streak"] = 0
            state["backoff_until"] = None

    if isinstance(training_data, dict):
        state["training_data_fingerprint"] = training_data.get("fingerprint")
        state["training_label_count"] = int(training_data.get("label_count", 0) or 0)
        state["training_positive_count"] = int(training_data.get("positive_count", 0) or 0)
        state["training_negative_count"] = int(training_data.get("negative_count", 0) or 0)
        state["training_window_end"] = training_data.get("training_end")
    try:
        _atomic_write_json(RETRAIN_STATE_FILE, state, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save retraining state: {e}")


def parse_datetime_safe(dt_str: str):
    if not dt_str:
        return None
    try:
        return datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
    except Exception:
        return None


def _parse_retraining_outcome(stdout: str) -> str:
    import re

    match = re.search(
        r"重训练结果:\s*(partially_deployed|deployed|rolled_back|trained_not_deployed|trained_not_better|not_needed|train_failed)",
        stdout or "",
    )
    return match.group(1) if match else "unknown"


def _parse_training_data_snapshot(stdout: str):
    marker = "TRAINING_DATA_SNAPSHOT="
    for line in (stdout or "").splitlines():
        if marker not in line:
            continue
        try:
            payload = json.loads(line.split(marker, 1)[1])
        except (json.JSONDecodeError, IndexError):
            return None
        return payload if isinstance(payload, dict) else None
    return None


def _parse_live_model_gates(stdout: str) -> dict:
    marker = "LIVE_MODEL_GATES="
    for line in (stdout or "").splitlines():
        if marker not in line:
            continue
        try:
            payload = json.loads(line.split(marker, 1)[1])
        except (json.JSONDecodeError, IndexError):
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}


def _load_prediction_result():
    if not os.path.exists(DATA_FILE):
        return None
    try:
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to inspect prediction result file: {e}")
        return None


def _model_deployment_times() -> dict:
    currencies = ("fUSD", "fUST")
    deployed_at = {currency: None for currency in currencies}
    try:
        with open(RETRAIN_HISTORY_FILE, "r", encoding="utf-8") as history_file:
            raw = json.load(history_file)
        entries = raw if isinstance(raw, list) else list(raw.values()) if isinstance(raw, dict) else []
        for entry in reversed(entries):
            if not isinstance(entry, dict) or entry.get("deployed") is not True:
                continue
            timestamp = parse_datetime_safe(entry.get("timestamp"))
            if timestamp is None:
                continue
            selected = entry.get("deployed_currencies")
            for currency in currencies:
                if deployed_at[currency] is not None:
                    continue
                if isinstance(selected, list) and currency not in selected:
                    continue
                deployed_at[currency] = timestamp
    except Exception as exc:
        logger.warning(f"Failed to read model deployment history: {exc}")

    model_dir = BASE_DIR / "data" / "models"
    for currency in currencies:
        if deployed_at[currency] is not None or not model_dir.exists():
            continue
        meta_files = list(model_dir.glob(f"{currency}_*_meta.json"))
        if meta_files:
            deployed_at[currency] = datetime.fromtimestamp(
                max(path.stat().st_mtime for path in meta_files)
            )
    return deployed_at


def _model_cohort_status(deployed_at: dict) -> dict:
    cohorts = {}
    try:
        with sqlite3.connect(DB_FILE) as conn:
            for currency, timestamp in deployed_at.items():
                if timestamp is None:
                    cohorts[currency] = {"decided": 0, "executed": 0, "pending": 0, "hit_rate": None}
                    continue
                row = conn.execute(
                    """
                    SELECT
                        SUM(status IN ('EXECUTED', 'FAILED')),
                        SUM(status = 'EXECUTED'),
                        SUM(status = 'PENDING')
                    FROM virtual_orders
                    WHERE currency = ? AND created_at >= ?
                    """,
                    (currency, timestamp.strftime('%Y-%m-%d %H:%M:%S')),
                ).fetchone()
                decided, executed, pending = (int(value or 0) for value in row)
                cohorts[currency] = {
                    "decided": decided,
                    "executed": executed,
                    "pending": pending,
                    "hit_rate": (executed / decided) if decided else None,
                    "age_hours": round((datetime.now() - timestamp).total_seconds() / 3600.0, 1),
                }
    except Exception as exc:
        logger.warning(f"Failed to calculate model cohort status: {exc}")
    return cohorts


def _calibration_runtime_status() -> dict:
    state = {}
    try:
        if os.path.exists(CALIBRATION_STATE_FILE):
            with open(CALIBRATION_STATE_FILE, "r", encoding="utf-8") as state_file:
                loaded = json.load(state_file)
            state = loaded if isinstance(loaded, dict) else {}
    except Exception as exc:
        logger.warning(f"Failed to read calibration state: {exc}")
    latest_by_currency = {}
    try:
        with sqlite3.connect(DB_FILE) as conn:
            latest_cycle = conn.execute(
                "SELECT update_cycle_id FROM prediction_history ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if latest_cycle:
                for currency, method, failures, reason, model_version in conn.execute(
                    """
                    SELECT currency, probability_calibration_method,
                           MAX(probability_calibration_failure_count),
                           MAX(probability_calibration_reason), MAX(model_version)
                    FROM prediction_history
                    WHERE update_cycle_id = ?
                    GROUP BY currency, probability_calibration_method
                    """,
                    (latest_cycle[0],),
                ):
                    latest_by_currency[currency] = {
                        "method": method,
                        "consecutive_failures": int(failures or 0),
                        "last_failure_reason": reason,
                        "model_version": model_version,
                    }
    except Exception as exc:
        logger.warning(f"Failed to read latest calibration methods: {exc}")

    result_methods = set()
    latest_result = _load_prediction_result() or {}
    for item in latest_result.get("recommendations", []) if isinstance(latest_result, dict) else []:
        if not isinstance(item, dict):
            continue
        details = item.get("details") if isinstance(item.get("details"), dict) else {}
        method = item.get("probability_calibration_method") or details.get(
            "probability_calibration_method"
        )
        if method:
            result_methods.add(str(method))

    state_by_currency = (
        state.get("currencies", {})
        if state.get("version") == 2 and isinstance(state.get("currencies"), dict)
        else {}
    )
    by_currency = {}
    for currency in ("fUSD", "fUST"):
        persisted = state_by_currency.get(currency, {})
        latest = latest_by_currency.get(currency, {})
        active = persisted.get("active_calibrator") or {}
        by_currency[currency] = {
            **latest,
            "method": latest.get("method") or active.get("method") or "traditional_fallback",
            "model_version": latest.get("model_version") or persisted.get("model_version"),
            "consecutive_failures": int(
                persisted.get("consecutive_failures", latest.get("consecutive_failures", 0)) or 0
            ),
            "last_failure_reason": (
                persisted.get("last_failure_reason")
                or latest.get("last_failure_reason")
                if int(persisted.get("consecutive_failures", latest.get("consecutive_failures", 0)) or 0) > 0
                else None
            ),
            "last_success_at": persisted.get("last_success_at"),
            "updated_at": persisted.get("updated_at"),
            "last_gate": persisted.get("last_gate"),
        }
    state_methods = {
        by_currency[currency]["method"]
        for currency in by_currency
        if currency in latest_by_currency or currency in state_by_currency
    }
    methods = sorted(state_methods | result_methods)
    max_failures = max(
        (int(item.get("consecutive_failures", 0) or 0) for item in by_currency.values()),
        default=int(state.get("consecutive_failures", 0) or 0),
    )
    successful_times = [
        item.get("last_success_at") for item in by_currency.values()
        if item.get("last_success_at")
    ]
    updated_times = [
        item.get("updated_at") for item in by_currency.values()
        if item.get("updated_at")
    ]
    aggregate_failure_reason = next((
        item.get("last_failure_reason")
        for item in sorted(
            by_currency.values(),
            key=lambda value: int(value.get("consecutive_failures", 0) or 0),
            reverse=True,
        )
        if item.get("last_failure_reason")
    ), None)
    return {
        "methods": methods,
        "consecutive_failures": max_failures,
        "last_failure_reason": aggregate_failure_reason or state.get("last_failure_reason"),
        "last_success_at": max(successful_times) if successful_times else state.get("last_success_at"),
        "updated_at": max(updated_times) if updated_times else state.get("updated_at"),
        "by_currency": by_currency,
    }


def _live_model_gate_status() -> dict:
    loaded_at = _live_gate_status_cache.get("loaded_at")
    if loaded_at and (datetime.now() - loaded_at).total_seconds() < 60:
        return _live_gate_status_cache.get("value", {})
    try:
        from ml_engine.retraining_scheduler import RetrainingScheduler

        scheduler = RetrainingScheduler(
            db_path=DB_FILE,
            production_model_dir=str(BASE_DIR / "data" / "models"),
            backup_dir=str(BASE_DIR / "data" / "models_backup"),
            log_dir=str(BASE_DIR / "data"),
        )
        value = scheduler.evaluate_live_model_gates()
        _live_gate_status_cache.update({"loaded_at": datetime.now(), "value": value})
        return value
    except Exception as exc:
        logger.warning(f"Failed to calculate live model gates: {exc}")
        return {}


def _extract_prediction_failure(result):
    if not isinstance(result, dict):
        return None

    if result.get("fail_closed"):
        return result.get("error") or result.get("reason") or "Prediction fail-closed"

    status = result.get("status")
    if status in {"error", "failed"}:
        return result.get("error") or result.get("details") or "Prediction reported failure"

    return None

# --- 辅助函数: 获取数据库统计 ---
def get_db_statistics():
    """获取虚拟订单统计信息"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # 订单状态统计
        cursor.execute("""
            SELECT status, COUNT(*) as count
            FROM virtual_orders
            GROUP BY status
            ORDER BY
                CASE status
                    WHEN 'PENDING' THEN 1
                    WHEN 'EXECUTED' THEN 2
                    WHEN 'FAILED' THEN 3
                    ELSE 4
                END
        """)
        status_stats = [{"status": row[0], "count": row[1]} for row in cursor.fetchall()]

        # order_timestamp is stored in local wall-clock time. SQLite's
        # datetime('now') is UTC, so build the cutoff in Python explicitly.
        cutoff_7d = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute("""
            SELECT
                currency,
                period,
                COUNT(*) as total,
                SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) as executed,
                ROUND(100.0 * SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) / COUNT(*), 1) as exec_rate
            FROM virtual_orders
            WHERE order_timestamp >= ?
              AND status IN ('EXECUTED', 'FAILED')
            GROUP BY currency, period
            ORDER BY total DESC
        """, (cutoff_7d,))
        exec_rate_7d = [
            {
                "currency": row[0],
                "period": row[1],
                "total": row[2],
                "executed": row[3],
                "exec_rate": row[4]
            }
            for row in cursor.fetchall()
        ]
        total_7d = sum(item["total"] for item in exec_rate_7d)
        executed_7d = sum(item["executed"] for item in exec_rate_7d)

        # 最新订单
        cursor.execute("""
            SELECT
                substr(order_id, 1, 8) as id,
                currency,
                period,
                predicted_rate,
                status,
                order_timestamp
            FROM virtual_orders
            ORDER BY order_timestamp DESC
            LIMIT 10
        """)
        latest_orders = [
            {
                "id": row[0],
                "combo": f"{row[1]}-{row[2]}d",
                "rate": f"{row[3]:.2f}%",
                "status": row[4],
                "created": row[5]
            }
            for row in cursor.fetchall()
        ]

        conn.close()

        return {
            "status_summary": status_stats,
            "execution_rate_7d": exec_rate_7d,
            "execution_rate_7d_overall": {
                "total": total_7d,
                "executed": executed_7d,
                "exec_rate": round(executed_7d / total_7d * 100.0, 1) if total_7d else None,
            },
            "latest_orders": latest_orders
        }
    except Exception as e:
        logger.error(f"Failed to get DB statistics: {e}")
        return {"error": str(e)}

# --- 验证函数: 从 validate_fixes.py 整合 ---

def test_timestamp_correctness(conn):
    """
    检查订单数据时间戳延迟（理想情况<5分钟）
    注意: 回测订单(order_timestamp是历史数据时间)会被自动排除,只检查实时订单
    5-60分钟的延迟会标记为WARNING(正常的API延迟),不影响交易决策
    """
    cursor = conn.cursor()

    # 查询实时订单中的延迟(时间差在5-60分钟之间,通常是API调用延迟)
    # 时间差>60分钟的被认为是回测订单,不算延迟
    cursor.execute("""
        SELECT
            order_id,
            currency,
            period,
            order_timestamp,
            created_at,
            ROUND((julianday(created_at) - julianday(order_timestamp)) * 24 * 60, 1) as delay_minutes
        FROM virtual_orders
        WHERE ABS((julianday(created_at) - julianday(order_timestamp)) * 24 * 60) > 5
          AND ABS((julianday(created_at) - julianday(order_timestamp)) * 24 * 60) < 60
          AND status != 'EXPIRED'
        ORDER BY created_at DESC
        LIMIT 10
    """)

    anomalies = cursor.fetchall()

    # 统计回测订单数量(仅供参考)
    cursor.execute("""
        SELECT COUNT(*)
        FROM virtual_orders
        WHERE ABS((julianday(created_at) - julianday(order_timestamp)) * 24 * 60) >= 60
          AND status != 'EXPIRED'
    """)
    backtest_orders = cursor.fetchone()[0]

    if not anomalies:
        return {
            "status": "PASS",
            "message": f"All real-time orders have correct timestamps (found {backtest_orders} backtest orders, excluded from check)",
            "anomaly_count": 0,
            "backtest_orders": backtest_orders,
            "anomalies": []
        }
    else:
        return {
            "status": "WARNING",
            "message": f"Found {len(anomalies)} orders with data timestamp delays (5-60min, normal due to API latency)",
            "anomaly_count": len(anomalies),
            "backtest_orders": backtest_orders,
            "anomalies": [
                {
                    "currency": row[1],
                    "period": row[2],
                    "order_timestamp": row[3],
                    "created_at": row[4],
                    "delay_minutes": row[5]
                }
                for row in anomalies[:5]
            ]
        }


def test_validation_window(conn):
    """
    验证订单在窗口结束后验证（防止 look-ahead bias）
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            order_id,
            currency,
            period,
            order_timestamp,
            validation_window_hours,
            validated_at,
            ROUND((julianday(validated_at) - julianday(order_timestamp)) * 24, 1) as actual_hours
        FROM virtual_orders
        WHERE status IN ('EXECUTED', 'FAILED')
          AND validated_at IS NOT NULL
          AND actual_hours < validation_window_hours - 1
        ORDER BY validated_at DESC
        LIMIT 10
    """)

    early_validations = cursor.fetchall()

    if not early_validations:
        return {
            "status": "PASS",
            "message": "All validations occurred after window end",
            "early_validation_count": 0,
            "early_validations": []
        }
    else:
        return {
            "status": "FAIL",
            "message": f"Found {len(early_validations)} orders validated too early (look-ahead bias)",
            "early_validation_count": len(early_validations),
            "early_validations": [
                {
                    "currency": row[1],
                    "period": row[2],
                    "window_hours": row[4],
                    "actual_hours": row[6]
                }
                for row in early_validations[:5]
            ]
        }


def test_sampling_coverage(conn):
    """
    验证采样覆盖率（7天内 >80% 组合覆盖）
    """
    cursor = conn.cursor()

    cutoff_7d = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')

    cursor.execute("""
        SELECT
            COUNT(DISTINCT currency || '-' || period) as tested_combinations
        FROM virtual_orders
        WHERE order_timestamp >= ?
    """, (cutoff_7d,))

    tested_7d = cursor.fetchone()[0]
    total_combinations = 2 * 14  # 2 currencies × 14 periods
    coverage_pct = (tested_7d / total_combinations) * 100

    # Get coverage by combination
    cursor.execute("""
        SELECT
            currency,
            period,
            COUNT(*) as order_count,
            SUM(CASE WHEN status = 'EXECUTED' THEN 1 ELSE 0 END) as executed,
            SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) as failed,
            SUM(CASE WHEN status = 'PENDING' THEN 1 ELSE 0 END) as pending
        FROM virtual_orders
        WHERE order_timestamp >= ?
        GROUP BY currency, period
        ORDER BY order_count DESC
    """, (cutoff_7d,))

    coverage_details = [
        {
            "currency": row[0],
            "period": row[1],
            "total": row[2],
            "executed": row[3],
            "failed": row[4],
            "pending": row[5]
        }
        for row in cursor.fetchall()
    ]

    if tested_7d == 0:
        return {
            "status": "WARNING",
            "message": "No orders in last 7 days (system may not be running)",
            "tested_combinations": tested_7d,
            "total_combinations": total_combinations,
            "coverage_pct": 0,
            "coverage_details": []
        }
    elif coverage_pct >= 80:
        return {
            "status": "PASS",
            "message": f"Good sampling coverage ({coverage_pct:.1f}%)",
            "tested_combinations": tested_7d,
            "total_combinations": total_combinations,
            "coverage_pct": coverage_pct,
            "coverage_details": coverage_details[:10]
        }
    else:
        return {
            "status": "WARNING",
            "message": f"Low sampling coverage ({coverage_pct:.1f}%), expect improvement over time",
            "tested_combinations": tested_7d,
            "total_combinations": total_combinations,
            "coverage_pct": coverage_pct,
            "coverage_details": coverage_details[:10]
        }


def test_execution_rate_realism(conn):
    """
    检查执行率是否真实（<90%）
    """
    cursor = conn.cursor()

    cutoff_7d = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')

    cursor.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status = 'EXECUTED' THEN 1 ELSE 0 END) as executed
        FROM virtual_orders
        WHERE order_timestamp >= ?
          AND status IN ('EXECUTED', 'FAILED', 'EXPIRED')
    """, (cutoff_7d,))

    row = cursor.fetchone()
    total = row[0] or 0
    executed = row[1] or 0

    if total == 0:
        return {
            "status": "WARNING",
            "message": "No validated orders in last 7 days (insufficient data)",
            "total_7d": 0,
            "executed_7d": 0,
            "exec_rate_7d": None,
            "total_30d": None,
            "executed_30d": None,
            "exec_rate_30d": None
        }

    exec_rate = (executed / total) * 100

    # 30-day execution rate
    cutoff_30d = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')

    cursor.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status = 'EXECUTED' THEN 1 ELSE 0 END) as executed
        FROM virtual_orders
        WHERE order_timestamp >= ?
          AND status IN ('EXECUTED', 'FAILED', 'EXPIRED')
    """, (cutoff_30d,))

    row = cursor.fetchone()
    total_30d = row[0] or 0
    executed_30d = row[1] or 0

    exec_rate_30d = (executed_30d / total_30d) * 100 if total_30d > 0 else None

    # Evaluation
    if exec_rate > 90:
        status = "FAIL"
        message = f"Execution rate too high ({exec_rate:.1f}%), likely look-ahead bias"
    elif exec_rate >= 40 and exec_rate <= 70:
        status = "PASS"
        message = f"Execution rate is realistic ({exec_rate:.1f}%)"
    else:
        status = "WARNING"
        message = f"Execution rate unusual ({exec_rate:.1f}%), needs more data"

    return {
        "status": status,
        "message": message,
        "total_7d": total,
        "executed_7d": executed,
        "exec_rate_7d": exec_rate,
        "total_30d": total_30d,
        "executed_30d": executed_30d,
        "exec_rate_30d": exec_rate_30d
    }


def test_cold_start_detection(conn):
    """
    验证冷启动组合检测
    """
    cursor = conn.cursor()
    cutoff_7d = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')

    cursor.execute("""
        SELECT
            currency,
            period,
            COUNT(*) as total_orders,
            SUM(CASE WHEN order_timestamp >= ? THEN 1 ELSE 0 END) as recent_orders
        FROM virtual_orders
        WHERE status IN ('EXECUTED', 'FAILED', 'PENDING')
        GROUP BY currency, period
        HAVING total_orders < 10
        ORDER BY recent_orders DESC
    """, (cutoff_7d,))

    cold_start_combos = cursor.fetchall()

    if not cold_start_combos:
        return {
            "status": "PASS",
            "message": "All combinations have sufficient data (>10 orders)",
            "cold_start_count": 0,
            "recent_tested_count": 0,
            "cold_start_combos": []
        }
    else:
        recent_tested = sum(1 for c in cold_start_combos if c[3] > 0)

        combos_details = [
            {
                "currency": row[0],
                "period": row[1],
                "total_orders": row[2],
                "recent_orders": row[3]
            }
            for row in cold_start_combos[:5]
        ]

        if recent_tested >= len(cold_start_combos) * 0.5:
            status = "PASS"
            message = f"Cold start combinations being explored ({recent_tested} tested)"
        else:
            status = "WARNING"
            message = "Some cold start combinations not tested recently"

        return {
            "status": status,
            "message": message,
            "cold_start_count": len(cold_start_combos),
            "recent_tested_count": recent_tested,
            "cold_start_combos": combos_details
        }


def test_expired_orders_validated(conn):
    """
    测试6: 所有EXPIRED订单都有validated_at时间戳

    这确保订单经过了正确的验证流程,而不是被直接标记为EXPIRED
    防止未来再次出现未验证的EXPIRED订单
    """
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*) as unvalidated_count
        FROM virtual_orders
        WHERE status = 'EXPIRED'
          AND validated_at IS NULL
    """)

    unvalidated_count = cursor.fetchone()[0]

    # Also get total EXPIRED count for context
    cursor.execute("""
        SELECT COUNT(*) as total_expired
        FROM virtual_orders
        WHERE status = 'EXPIRED'
    """)

    total_expired = cursor.fetchone()[0]

    # Check if all EXPIRED orders have validated_at
    if unvalidated_count == 0:
        status = "PASS"
        message = f"All {total_expired} EXPIRED orders have validated_at timestamp"
    else:
        status = "FAIL"
        message = f"Found {unvalidated_count} EXPIRED orders without validated_at (out of {total_expired} total)"

    return {
        "status": status,
        "message": message,
        "unvalidated_expired_orders": unvalidated_count,
        "total_expired_orders": total_expired,
        "validation_coverage": f"{100.0 * (total_expired - unvalidated_count) / max(total_expired, 1):.1f}%"
    }


def run_all_validation_tests():
    """
    运行所有5个验证测试
    """
    if not os.path.exists(DB_FILE):
        return {
            "error": f"Database not found: {DB_FILE}",
            "overall_status": "ERROR"
        }

    try:
        conn = sqlite3.connect(DB_FILE)

        # Run all tests
        results = {
            'timestamp_correctness': test_timestamp_correctness(conn),
            'validation_window': test_validation_window(conn),
            'sampling_coverage': test_sampling_coverage(conn),
            'execution_rate': test_execution_rate_realism(conn),
            'cold_start_detection': test_cold_start_detection(conn),
            'expired_orders_validated': test_expired_orders_validated(conn),
        }

        conn.close()

        # Calculate summary
        passed = sum(1 for v in results.values() if v.get('status') == 'PASS')
        failed = sum(1 for v in results.values() if v.get('status') == 'FAIL')
        warnings = sum(1 for v in results.values() if v.get('status') == 'WARNING')

        if failed == 0 and passed >= 3:
            overall_status = "PASS"
        elif failed > 0:
            overall_status = "FAIL"
        else:
            overall_status = "WARNING"

        return {
            "summary": {
                "passed": passed,
                "failed": failed,
                "warnings": warnings,
                "overall_status": overall_status
            },
            "tests": results
        }

    except Exception as e:
        logger.error(f"Validation tests failed: {e}")
        return {
            "error": str(e),
            "overall_status": "ERROR"
        }

# --- 核心逻辑: 完整更新流水线 (每2小时运行) - 闭环优化版 ---

# Subprocess timeout constants
TIMEOUT_DOWNLOAD = 1200   # 20 minutes (Bitfinex refresh is serial and can stall on network retries)
TIMEOUT_TRAIN = 4200      # 70 minutes (12 models × 3.5 min + v2 models + compare + buffer)
TIMEOUT_PREDICT = 900     # 15 minutes (56 tasks + 24 order creations)
TIMEOUT_VALIDATE = 300    # 5 minutes
TIMEOUT_ORDERS = 300      # 5 minutes


def _build_subprocess_env():
    """
    Build runtime environment for training/prediction subprocesses.
    Enforces multi-core CPU usage and keeps GPU visible.
    """
    env = os.environ.copy()
    cpu_threads = int(env.get("ML_CPU_THREADS", str(max(1, min(os.cpu_count() or 8, 32)))))

    # Threaded math/runtime libs
    env["ML_CPU_THREADS"] = str(cpu_threads)
    env["OMP_NUM_THREADS"] = str(cpu_threads)
    env["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
    env["MKL_NUM_THREADS"] = str(cpu_threads)
    env["NUMEXPR_NUM_THREADS"] = str(cpu_threads)
    env["VECLIB_MAXIMUM_THREADS"] = str(cpu_threads)

    # Inference-level parallel controls
    # 预测并行度保持保守：max_workers=8（14 period 并行足够），infer_threads=4
    # 避免在高负载宿主机上 24×24=576 线程需求导致 CPU 上下文切换风暴
    # （训练侧不受影响，训练的线程限制由 model_trainer_v2 P1-1 单独控制）
    env.setdefault("PREDICT_MAX_WORKERS", str(max(4, min(cpu_threads, 8))))
    env.setdefault("PREDICT_INFER_THREADS", str(min(cpu_threads, 4)))
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    return env


def _check_db_data_freshness(fUSD_max: int = None, fUST_max: int = None) -> bool:
    """Check if funding_rates has fresh data for at least one currency. Returns True if any currency is fresh."""
    # Read thresholds from system_policy.json, fall back to hardcoded defaults
    _auto = {}
    try:
        from ml_engine.system_policy import load_system_policy
        _policy = load_system_policy()
        _auto = _policy.get("automation", {})
        if fUSD_max is None:
            fUSD_max = int(_auto.get("stale_data_hard_minutes", 300))
        if fUST_max is None:
            fUST_max = int(_auto.get("stale_data_hard_minutes_fUST", _auto.get("stale_data_hard_minutes", 900)))
    except Exception:
        if fUSD_max is None:
            fUSD_max = 300
        if fUST_max is None:
            fUST_max = 900
    try:
        import sqlite3 as _sqlite3
        from datetime import datetime as _dt
        db_path = str(BASE_DIR / "data" / "lending_history.db")
        conn = _sqlite3.connect(db_path)
        cursor = conn.cursor()
        thresholds = {'fUSD': fUSD_max, 'fUST': fUST_max}
        any_fresh = False
        for currency, max_age in thresholds.items():
            cursor.execute("SELECT MAX(timestamp) FROM funding_rates WHERE currency = ?", (currency,))
            row = cursor.fetchone()
            if not row or not row[0]:
                continue
            latest_ts = row[0]
            if isinstance(latest_ts, str):
                try:
                    latest_dt = _dt.fromisoformat(latest_ts)
                except ValueError:
                    latest_dt = _dt.strptime(latest_ts, '%Y-%m-%d %H:%M:%S')
            elif isinstance(latest_ts, (int, float)):
                # funding_rates timestamps may be stored as milliseconds (13-digit)
                # vs seconds (10-digit); detect by magnitude
                if latest_ts > 1e10:
                    latest_dt = _dt.fromtimestamp(latest_ts / 1000)
                else:
                    latest_dt = _dt.fromtimestamp(latest_ts)
            else:
                latest_dt = latest_ts
            age_minutes = (datetime.now() - latest_dt).total_seconds() / 60
            if age_minutes <= max_age:
                logger.info(f"✅ {currency} data is fresh (age={age_minutes:.0f}min <= {max_age}min)")
                any_fresh = True
        # 修复4: fUST 长周期 per-period 诊断（流动性枯竭预警）
        _fust_tier = _auto.get("stale_data_hard_minutes_fUST_by_period_tier", {})
        _fust_long_threshold = int(_fust_tier.get("long", 10080))
        for period in [30, 60, 90]:
            cursor.execute("SELECT MAX(timestamp) FROM funding_rates WHERE currency='fUST' AND period=?", (period,))
            p_row = cursor.fetchone()
            if p_row and p_row[0]:
                p_ts = p_row[0]
                if p_ts > 1e10:
                    p_dt = _dt.fromtimestamp(p_ts / 1000)
                else:
                    p_dt = _dt.fromtimestamp(p_ts)
                p_age = (_dt.now() - p_dt).total_seconds() / 60
                if p_age > _fust_long_threshold:
                    logger.warning(f"  WARN: fUST-{period} is {p_age:.0f}min old (>{_fust_long_threshold}min, possible liquidity drought)")
        # 优化3: fUSD 长周期 per-period 诊断（数据源异常预警）
        _fusd_tier = _auto.get("stale_data_hard_minutes_by_period_tier", {})
        _fusd_warn_threshold = int(_fusd_tier.get("long", 1440))
        for period in [15, 20, 60, 90, 120]:
            cursor.execute("SELECT MAX(timestamp) FROM funding_rates WHERE currency='fUSD' AND period=?", (period,))
            p_row = cursor.fetchone()
            if p_row and p_row[0]:
                p_ts = p_row[0]
                if p_ts > 1e10:
                    p_dt = _dt.fromtimestamp(p_ts / 1000)
                else:
                    p_dt = _dt.fromtimestamp(p_ts)
                p_age = (_dt.now() - p_dt).total_seconds() / 60
                if p_age > _fusd_warn_threshold:
                    logger.warning(f"  WARN: fUSD-{period} is {p_age:.0f}min old (>{_fusd_warn_threshold}min, check Bitfinex data supply)")
        conn.close()
        return any_fresh
    except Exception as e:
        logger.warning(f"_check_db_data_freshness error: {e}")
        return False


async def _run_subprocess_with_timeout(cmd, cwd, timeout, step_name):
    """Run subprocess with timeout, kill on timeout. Returns (stdout, stderr, returncode).

    使用 start_new_session=True 让子进程成为新进程组的 leader，
    超时时用 os.killpg 杀整个进程组（含 GPU/OpenMP worker），避免孤儿进程占用资源。
    采用 SIGTERM → 8s grace → SIGKILL 两阶段，给模型 save_model 留优雅退出时间。
    """
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd),
        env=_build_subprocess_env(),
        start_new_session=True,  # 子进程成为新会话/进程组leader，便于killpg
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        return stdout.decode().strip(), stderr.decode().strip(), process.returncode
    except asyncio.TimeoutError:
        logger.error(f"TIMEOUT: {step_name} exceeded {timeout}s, killing process group")
        # 两阶段终止：先SIGTERM（优雅退出，让save_model完成），8s后SIGKILL
        await _terminate_process_group(process, step_name)
        # 读取已缓冲的 stdout/stderr，避免超时时丢失全部子进程输出
        try:
            remaining_stdout, remaining_stderr = await process.communicate()
            partial_out = (remaining_stdout or b"").decode().strip()
            partial_err = (remaining_stderr or b"").decode().strip()
        except Exception:
            partial_out = ""
            partial_err = ""
        if not partial_out and not partial_err:
            partial_err = f"Subprocess timed out after {timeout}s"
        return partial_out, partial_err, -1


async def _terminate_process_group(process, step_name: str, grace_seconds: int = 8):
    """优雅终止整个子进程组：SIGTERM → 等待 grace → SIGKILL。

    覆盖训练子进程派生的 GPU/OpenMP worker，避免孤儿进程占用 GPU/CPU。
    """
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        # 子进程已退出，无需处理
        return
    except Exception as e:
        logger.warning(f"{step_name}: 无法获取进程组({e}), 回退到 kill 单进程")
        try:
            process.kill()
        except Exception:
            pass
        return

    # 阶段1：SIGTERM 整个进程组
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception as e:
        logger.warning(f"{step_name}: SIGTERM进程组失败({e})")

    # 等待 grace period 让子进程优雅退出
    try:
        await asyncio.wait_for(process.wait(), timeout=grace_seconds)
        return  # 已优雅退出
    except asyncio.TimeoutError:
        pass  # 超时，进入阶段2

    # 阶段2：SIGKILL 强制清理整个进程组
    logger.warning(f"{step_name}: SIGTERM后{grace_seconds}s未退出, SIGKILL进程组")
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except Exception as e:
        logger.warning(f"{step_name}: SIGKILL进程组失败({e}), 回退到 kill 单进程")
        try:
            process.kill()
        except Exception:
            pass


def _is_service_shutdown_returncode(rc: int) -> bool:
    """True when a child process exited because the service is shutting down."""
    return rc in {-signal.SIGTERM, 128 + signal.SIGTERM, -signal.SIGINT, 128 + signal.SIGINT}


def _is_partial_download_with_stale(stdout: str, stderr: str, rc: int) -> bool:
    """
    下载器已执行完成，但仅因为仍有 stale/missing 组合而返回 rc=2。
    这种情况不应阻塞后续 pipeline，freshness 交给预测阶段判定。

    P1-3: 改用精确 exit code 判定（rc=2 = 部分stale软失败），不再依赖文本 marker 匹配。
      rc=0: 全部fresh成功；rc=2: 部分stale（软失败）；rc=1: 真实失败（异常）；-1: 超时。
    """
    return rc == 2


async def _download_with_retry(cwd, max_retries=3):
    """Download with exponential backoff retry. Partial stale refresh should not block pipeline.

    exit code 语义（P1-3）：
      rc=0  → 全部 fresh，放行
      rc=2  → 部分 stale（软失败），放行，freshness 由 predictor 判定
      rc=1  → 真实失败（异常），重试
      rc=-1 → 超时，重试
    """
    download_cmd = [sys.executable, "-m", "funding_history_downloader", "--days", "30"]
    for attempt in range(max_retries):
        stdout, stderr, rc = await _run_subprocess_with_timeout(
            download_cmd, cwd, TIMEOUT_DOWNLOAD, "Data Download"
        )
        if stdout:
            logger.info(f"Download output (last 500 chars):\n{stdout[-500:]}")
        if stderr:
            logger.warning(f"Download stderr:\n{stderr[-500:]}")
        if rc == 0:
            logger.info("✅ Market data download completed (all fresh)")
            return True
        if _is_partial_download_with_stale(stdout, stderr, rc):
            logger.warning(
                "⚠️  Market data download completed with stale/missing pairs (rc=2). "
                "Continuing pipeline; freshness gate will be enforced at prediction stage."
            )
            return True
        # rc=1（真实失败）或 rc=-1（超时）：触发重试
        wait_time = 60 * (2 ** attempt)  # 60s, 120s, 240s
        if attempt < max_retries - 1:
            logger.warning(f"⚠️  Download failed (attempt {attempt+1}/{max_retries}, exit={rc}), retrying in {wait_time}s")
            await asyncio.sleep(wait_time)
        else:
            logger.warning(f"⚠️  Download failed after {max_retries} attempts (exit={rc}), checking data freshness...")
    # All retries exhausted: check DB freshness
    data_age_ok = _check_db_data_freshness()
    if data_age_ok:
        logger.info("✅ Existing DB data is fresh enough, continuing pipeline despite download failure")
        return True
    logger.error("❌ Data download failed and DB data is stale")
    return False



_initial_retrain_state = load_retraining_state()
_last_forced_retrain_time = parse_datetime_safe(_initial_retrain_state.get("last_forced_retrain_time"))

async def run_full_pipeline():
    """
    后台任务: 执行完整的闭环优化流程

    闭环流程:
    1. 验证虚拟订单（获取执行反馈）
    2. 下载最新市场数据（确保数据新鲜）⭐
    3. 检查是否需要重训练（基于执行结果）
    4. 如果需要，执行模型重训练（学习市场反馈）
    5. 生成新预测（使用最新模型和数据）
    6. 创建新虚拟订单（进入下一轮循环）

    适合每2小时运行一次，形成持续优化闭环
    """
    global _last_forced_retrain_time

    # Acquire concurrency lock (L1) - prevent overlapping pipeline runs
    if _pipeline_lock.locked():
        logger.warning("Pipeline already running, skipping this invocation")
        return

    async with _pipeline_lock:
        logger.info("=" * 80)
        logger.info(">>> 🔄 Starting CLOSED-LOOP Optimization Pipeline")
        logger.info(f">>> Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        pipeline_degraded_reasons = []

        # ========== STEP 1: 下载最新市场数据 (必须！) ⭐ ==========
        update_status("processing", "1. Downloading Data", "Downloading latest market data...")
        logger.info("Step 1: 📥 Downloading latest market data (CRITICAL)")

        try:
            download_ok = await _download_with_retry(BASE_DIR)
            if not download_ok:
                update_status("error", "1. Downloading Data", "Download failed and data is stale")
                return
        except Exception as e:
            logger.warning(f"⚠️  Data download system error: {e}, checking data freshness...")
            data_age_ok = _check_db_data_freshness()
            if data_age_ok:
                logger.info("✅ Existing DB data is fresh enough, continuing pipeline despite download error")
            else:
                logger.error(f"❌ Data download system error and DB data is stale, aborting pipeline")
                update_status("error", "1. Downloading Data", str(e))
                return
        # ====================================================================

        # ========== STEP 2: 验证虚拟订单 (获取执行反馈) ==========
        update_status("processing", "2. Validating Orders", "Checking pending orders for execution feedback...")
        logger.info("Step 2: 🔍 Validating pending virtual orders (execution feedback)")

        try:
            stdout, stderr, rc = await _run_subprocess_with_timeout(
                [sys.executable, str(BASE_DIR / "ml_engine" / "execution_validator.py")],
                BASE_DIR, TIMEOUT_VALIDATE, "Order Validation"
            )
            if rc != 0:
                logger.warning(f"Order validation had issues: {stderr}")
            else:
                logger.info(f"✅ Order validation completed: {stdout[-500:]}")
        except Exception as e:
            # Order validation failure is non-critical, continue pipeline
            logger.warning(f"⚠️  Order validation failed: {e}, continuing with pipeline")
        # ====================================================================

        # ========== STEP 3: 检查是否需要重训练 (闭环核心) ==========
        update_status("processing", "3. Checking Retraining", "Evaluating if model retraining is needed...")
        logger.info("Step 3: 🤖 Checking if model retraining is needed (closed-loop)")

        should_retrain = False
        live_rollback_currencies = []
        try:
            stdout, stderr, rc = await _run_subprocess_with_timeout(
                [sys.executable, str(BASE_DIR / "ml_engine" / "retraining_scheduler.py"), "--dry-run"],
                BASE_DIR, TIMEOUT_VALIDATE, "Retraining Check"
            )
            live_gates = _parse_live_model_gates(stdout or "")
            live_rollback_currencies = sorted(
                currency
                for currency, gate in live_gates.items()
                if isinstance(gate, dict) and gate.get("rollback_required") is True
            )
            if live_rollback_currencies:
                logger.warning(
                    "Live model rollback required: "
                    + ", ".join(live_rollback_currencies)
                )

            # 检查输出中是否包含实际触发信号（排除标题"🔍 检查是否需要重训练"的误匹配）
            # 方案B: 匹配实际触发行前缀 "⚠️  需要重训练" 或 "✅ 需要重训练"
            if live_rollback_currencies:
                should_retrain = False
            elif "⚠️  需要重训练" in stdout or "✅ 需要重训练" in stdout:
                # 极端低流动性（exec_rate < 10%）：绕过冷却强制重训
                _bypass_cooldown = False
                if "全局成交率过低" in stdout:
                    import re as _re
                    _m = _re.search(r'(\d+(?:\.\d+)?)%\s*<', stdout)
                    if _m and float(_m.group(1)) < 10.0:
                        logger.warning(f"极端低流动性 ({_m.group(1)}% < 10%)，绕过冷却强制重训")
                        _bypass_cooldown = True

                if _bypass_cooldown:
                    should_retrain = True
                    logger.info(f"✅ Retraining trigger detected (bypass_cooldown=True): {stdout[-300:]}")
                else:
                    # 紧急重训练(单period异常)冷却6h
                    # 全局成交率过低(exec<30%)冷却12h（比普通24h更短，但比紧急6h保守）
                    # 普通重训练冷却24h
                    is_urgent = "紧急重训练" in stdout
                    is_exec_low = "全局成交率过低" in stdout
                    if is_urgent:
                        cooldown = timedelta(hours=6)
                        cooldown_label = "6h"
                    elif is_exec_low:
                        cooldown = timedelta(hours=12)
                        cooldown_label = "12h"
                    else:
                        cooldown = timedelta(hours=24)
                        cooldown_label = "24h"

                    # Fix6: 每次从磁盘重新读取，而非依赖内存变量（重启后状态也能正确加载）
                    _disk_state = load_retraining_state()
                    _disk_last_retrain = parse_datetime_safe(_disk_state.get("last_forced_retrain_time"))
                    _backoff_until = parse_datetime_safe(_disk_state.get("backoff_until"))
                    _dry_run_snapshot = _parse_training_data_snapshot(stdout or "")
                    _same_rejected_training_data = (
                        _disk_state.get("last_outcome") in {
                            "trained_not_better", "trained_not_deployed"
                        }
                        and isinstance(_dry_run_snapshot, dict)
                        and bool(_disk_state.get("training_data_fingerprint"))
                        and _dry_run_snapshot.get("fingerprint")
                        == _disk_state.get("training_data_fingerprint")
                    )
                    if _same_rejected_training_data:
                        logger.info(
                            "ℹ️  Retraining trigger suppressed: mature training labels "
                            f"unchanged since rejected challenger "
                            f"(fingerprint={_dry_run_snapshot.get('fingerprint')}, "
                            f"labels={_dry_run_snapshot.get('label_count')})"
                        )
                        should_retrain = False
                    elif _backoff_until and datetime.now() < _backoff_until:
                        logger.info(
                            "ℹ️  Retraining trigger suppressed after rejected challenger: "
                            f"backoff until {_backoff_until.strftime('%Y-%m-%d %H:%M:%S')} "
                            f"(streak={int(_disk_state.get('rejection_streak', 0) or 0)})"
                        )
                        should_retrain = False
                    elif _disk_last_retrain and (datetime.now() - _disk_last_retrain) < cooldown:
                        logger.info(f"ℹ️  Retraining triggered but skipped: last retrained {datetime.now() - _disk_last_retrain} ago (< {cooldown_label})")
                        should_retrain = False
                    else:
                        should_retrain = True
                        logger.info(f"✅ Retraining trigger detected (urgent={is_urgent}): {stdout[-300:]}")
            else:
                logger.info(f"ℹ️  No retraining needed: {stdout[-300:]}")

        except Exception as e:
            logger.warning(f"⚠️  Retraining check failed: {e}, skipping retraining")
        # ====================================================================

        # ========== STEP 4: live rollback 或模型重训练 ==========
        if live_rollback_currencies:
            update_status(
                "processing", "4. Rolling Back Models",
                "Applying live quality gate rollback...",
            )
            for currency in live_rollback_currencies:
                stdout, stderr, rc = await _run_subprocess_with_timeout(
                    [
                        sys.executable,
                        str(BASE_DIR / "ml_engine" / "retraining_scheduler.py"),
                        "--apply-live-rollback",
                        currency,
                    ],
                    BASE_DIR,
                    TIMEOUT_VALIDATE,
                    f"Live Rollback {currency}",
                )
                if stdout:
                    logger.info(f"Live rollback output ({currency}):\n{stdout[-2000:]}")
                if rc != 0:
                    detail = stderr or stdout or "unknown rollback failure"
                    logger.error(f"Live rollback failed for {currency}: {detail[-500:]}")
                    degraded_msg = f"Live rollback failed: {currency}"
                    pipeline_degraded_reasons.append(degraded_msg)
        elif should_retrain:
            update_status("processing", "4. Retraining Models", "Training models with execution feedback...")
            logger.info("Step 4: 🚀 Retraining models with execution feedback (CLOSED-LOOP)")

            try:
                stdout, stderr, rc = await _run_subprocess_with_timeout(
                    [sys.executable, str(BASE_DIR / "ml_engine" / "retraining_scheduler.py"), "--force"],
                    BASE_DIR, TIMEOUT_TRAIN, "Model Retraining"
                )

                if stdout:
                    logger.info(f"Retraining output (last 10000 chars):\n{stdout[-10000:]}")
                if stderr:
                    logger.warning(f"Retraining stderr (last 500 chars):\n{stderr[-500:]}")

                # exit code: 0=部署成功或训练成功但未部署, 1=训练失败, 2=无需重训练, -1=超时
                if rc == 0:
                    retrain_outcome = _parse_retraining_outcome(stdout or "")
                    training_snapshot = _parse_training_data_snapshot(stdout or "")
                    deployed = retrain_outcome in {'deployed', 'partially_deployed'}
                    _last_forced_retrain_time = datetime.now()
                    save_retraining_state(
                        _last_forced_retrain_time,
                        reason="forced_by_pipeline",
                        outcome=retrain_outcome,
                        training_data=training_snapshot,
                    )
                    if deployed:
                        logger.info(f"✅ Retraining completed and new models deployed")
                    else:
                        logger.info(
                            "✅ Retraining completed but models not deployed "
                            f"(outcome={retrain_outcome}, keeping current)"
                        )
                elif rc == 2:
                    logger.info(f"ℹ️ Retraining check: not needed")
                elif rc == -1:
                    # P1-2: 训练超时也更新冷却时间，防止每2h反复触发70min超时训练耗尽槽位
                    # 真实失败(rc==1)不更新冷却，让下轮重试；仅超时更新，避免无效占用
                    logger.error(f"❌ Retraining timed out (70min)")
                    _last_forced_retrain_time = datetime.now()
                    save_retraining_state(_last_forced_retrain_time, reason="retrain_timeout_cooldown")
                    degraded_msg = "Retraining timed out (rc=-1)"
                    pipeline_degraded_reasons.append(degraded_msg)
                    update_status("degraded", "4. Retraining Models", degraded_msg)
                else:
                    # rc == 1: 训练失败，不更新冷却（让下轮重试）
                    logger.error(f"❌ Retraining failed (exit code {rc})")
                    # Extract meaningful error: prefer stdout error lines, filter normal stderr noise
                    err_detail = ""
                    if stdout:
                        for line in reversed(stdout.splitlines()):
                            if '❌' in line or '失败' in line or 'ERROR' in line or 'Traceback' in line:
                                err_detail = line.strip()
                                break
                    if not err_detail and stderr:
                        noise_patterns = ['Capping', 'extreme', 'percentile', 'UserWarning', 'FutureWarning']
                        stderr_lines = [l.strip() for l in stderr.splitlines()
                                        if l.strip() and not any(p in l for p in noise_patterns)]
                        if stderr_lines:
                            err_detail = stderr_lines[-1]
                    if err_detail:
                        logger.error(f"Retraining error details: {err_detail}")
                    degraded_msg = f"Retraining failed (rc={rc})"
                    pipeline_degraded_reasons.append(degraded_msg)
                    update_status("degraded", "4. Retraining Models", degraded_msg)

            except Exception as e:
                logger.error(f"❌ Retraining system error: {e}")
                degraded_msg = f"Retraining system error: {e}"
                pipeline_degraded_reasons.append(degraded_msg)
                update_status("degraded", "4. Retraining Models", degraded_msg)
                # 继续流程，使用现有模型
        else:
            logger.info("Step 4: ⏭️  Skipping retraining (not needed)")
        # ====================================================================

        # ========== STEP 5: 生成新预测 (使用最新模型) ==========
        update_status("processing", "5. Generating Predictions", "Creating new predictions...")
        logger.info("Step 5: 📊 Generating new predictions with latest models")

        try:
            stdout, stderr, rc = await _run_subprocess_with_timeout(
                [sys.executable, "-m", "ml_engine.predictor"],
                BASE_DIR, TIMEOUT_PREDICT, "Prediction"
            )

            if stdout:
                logger.info(f"Prediction output (last 500 chars):\n{stdout[-500:]}")
            if stderr:
                logger.warning(f"Prediction stderr:\n{stderr[-500:]}")

            if rc != 0:
                if _is_service_shutdown_returncode(rc):
                    logger.info(
                        f"Prediction subprocess terminated during service shutdown (exit code {rc}); "
                        "leaving previous prediction result intact"
                    )
                    update_status("online", "Idle", "Service shutdown interrupted prediction")
                    return
                err_msg = stderr or stdout or "Prediction failed"
                logger.error(f"❌ Prediction failed (exit code {rc}): {err_msg}")
                update_status("error", "5. Generating Predictions", f"Failed: {err_msg[-200:]}")
                return

            logger.info(f"✅ Prediction completed successfully")

            prediction_failure = _extract_prediction_failure(_load_prediction_result())
            if prediction_failure:
                logger.error(f"❌ Prediction reported failure result: {prediction_failure}")
                update_status("error", "5. Generating Predictions", f"Failed: {prediction_failure[-200:]}")
                return

            # Freshness gate may produce stale_data result with empty recommendations.
            try:
                if os.path.exists(DATA_FILE):
                    with open(DATA_FILE, "r") as f:
                        latest_result = json.load(f)
                    if isinstance(latest_result, dict) and latest_result.get("stale_data"):
                        logger.warning(
                            "Prediction finished with stale_data gate: "
                            f"{latest_result.get('stale_reason', '')}"
                        )
            except Exception as parse_err:
                logger.warning(f"Failed to inspect prediction result freshness fields: {parse_err}")

        except Exception as e:
            logger.error(f"❌ Prediction system error: {e}")
            update_status("error", "5. Generating Predictions", str(e))
            return
        # ====================================================================

        # ========== STEP 6: 虚拟订单已在 Step 5 预测阶段创建 ==========
        # 注意: 订单由 predictor.py generate_recommendations() 内部调用
        # self.order_manager.create_virtual_order(pred) 创建，无需额外 subprocess
        # 旧版通过 `python -m ml_engine.order_manager` subprocess 调用会执行
        # order_manager.py 的 __main__ 测试代码，导致每轮多创建一个
        # fUSD-30d predicted_rate=12.5 的假订单（已累计185条，已清理）
        logger.info("Step 6: ✅ Virtual orders already created in Step 5 (prediction phase)")
        # ====================================================================

        # 全部完成 - 记录统计信息
        logger.info("=" * 80)
        logger.info("<<< ✅ CLOSED-LOOP Pipeline completed successfully!")
        logger.info(f"<<< End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 获取并记录统计信息
        stats = get_db_statistics()
        logger.info("=== Virtual Orders Summary ===")
        logger.info(f"Status: {stats.get('status_summary', [])}")
        logger.info(f"7-day execution rate: {len(stats.get('execution_rate_7d', []))} combinations validated")

        # 计算成交率
        exec_stats = stats.get('execution_rate_7d', [])
        if exec_stats:
            total_orders = sum(item['total'] for item in exec_stats)
            total_executed = sum(item['executed'] for item in exec_stats)
            overall_exec_rate = (total_executed / total_orders * 100) if total_orders > 0 else 0
            logger.info(f"Overall 7-day execution rate: {overall_exec_rate:.2f}% ({total_executed}/{total_orders})")

        logger.info("=" * 80)

        if pipeline_degraded_reasons:
            degraded_summary = "; ".join(dict.fromkeys(pipeline_degraded_reasons))
            update_status(
                "degraded",
                "Idle",
                f"Last closed-loop update at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}; {degraded_summary}"
            )
        else:
            update_status("online", "Idle", f"Last closed-loop update at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- API 接口定义 ---

# 1. 确认 API 是否在线 (Health Check)
@app.get("/status")
def check_status():
    """
    返回 API 在线状态及当前后台任务进度
    """
    status_info = get_current_status()

    deployment_times = _model_deployment_times()
    age_by_currency = {
        currency: (
            max(0, (datetime.now() - deployed_at).days)
            if deployed_at is not None
            else 999
        )
        for currency, deployed_at in deployment_times.items()
    }
    status_info["model_age_days_by_currency"] = age_by_currency
    status_info["model_deployed_at_by_currency"] = {
        currency: deployed_at.strftime('%Y-%m-%d %H:%M:%S') if deployed_at else None
        for currency, deployed_at in deployment_times.items()
    }
    status_info["model_cohort_by_currency"] = _model_cohort_status(deployment_times)
    status_info["probability_calibration"] = _calibration_runtime_status()
    status_info["live_model_gate_by_currency"] = _live_model_gate_status()
    status_info["model_age_days"] = max(age_by_currency.values(), default=999)

    stale_currencies = [
        currency for currency, age in age_by_currency.items() if age > 14
    ]
    for currency, age in age_by_currency.items():
        if age > 7:
            logger.warning(f"Production {currency} models are {age} days old!")
    if stale_currencies:
        status_info["status"] = "degraded"
        stale_summary = ", ".join(
            f"{currency}={age_by_currency[currency]}d"
            for currency in stale_currencies
        )
        status_info["details"] = f"Models stale: {stale_summary}"

    return {
        "api_online": True,
        "service_info": status_info
    }

# 2. 获取预测结果 JSON (Get Results)
@app.get("/result")
def get_result():
    """
    返回最新的预测结果 (optimal_combination.json)
    """
    if not os.path.exists(DATA_FILE):
        return JSONResponse(status_code=404, content={
            "error": "No predictions found.",
            "suggestion": "Please call /update to generate data."
        })
    
    try:
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict):
            data.setdefault("stale_data", False)
            data.setdefault("stale_minutes", 0)
            data.setdefault("stale_reason", "")
            data.setdefault("policy_version", "unknown")
        return data
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to read data: {str(e)}"})

# 3. 重新下载/训练/预测 (Trigger Update)
@app.post("/update")
async def trigger_update(background_tasks: BackgroundTasks):
    """
    触发后台全量更新: 下载新数据 -> 重新训练模型 -> 生成新预测
    """
    if _pipeline_lock.locked():
        return JSONResponse(status_code=409, content={
            "status": "busy",
            "message": "Pipeline is already running"
        })

    current = get_current_status()
    if current.get("status") == "processing":
        return JSONResponse(status_code=409, content={
            "status": "busy",
            "message": f"Pipeline is already running: {current.get('current_step')}"
        })

    # 添加到后台任务队列，立即响应
    background_tasks.add_task(run_full_pipeline)

    return {
        "status": "accepted",
        "message": "Full update pipeline started. Check /status for progress."
    }

# 4. 查询执行统计 (Query Execution Stats) - 新增
@app.get("/execution_stats")
def get_execution_stats(currency: str = "fUSD", period: int = 30, days: int = 7):
    """
    查询指定币种-周期的成交统计

    参数:
        currency: fUSD 或 fUST
        period: 借贷周期 (2-120)
        days: 回溯天数 (默认7天)
    """
    try:
        from ml_engine.order_manager import OrderManager
        manager = OrderManager()
        stats = manager.get_execution_stats(currency, period, days)
        return {
            "currency": currency,
            "period": period,
            "lookback_days": days,
            "statistics": stats
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# 5. 查询虚拟订单列表 (Query Virtual Orders) - 新增
@app.get("/orders")
def get_orders(status: str = None, limit: int = 100):
    """
    查询虚拟订单列表

    参数:
        status: 过滤状态 (PENDING/EXECUTED/FAILED)
        limit: 返回数量限制
    """
    try:
        from ml_engine.order_manager import OrderManager
        manager = OrderManager()
        orders = manager.get_orders(status=status, limit=limit)
        return {
            "total": len(orders),
            "filter_status": status or "all",
            "orders": orders
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# 6. 获取数据库统计信息 (Get Statistics) - 新增
@app.get("/stats")
def get_stats():
    """
    获取虚拟订单统计信息，类似 daily_run.sh 的输出

    返回:
        - 订单状态汇总 (PENDING/EXECUTED/FAILED)
        - 7天执行率 (按币种-周期分组)
        - 最新10个订单
    """
    return get_db_statistics()

# 7. 运行完整系统验证 (Run All Validation Tests) - 新增
@app.get("/validate")
def validate_system():
    """
    运行所有6个系统验证测试,返回完整验证报告

    测试包括:
        1. 时间戳正确性(5分钟内)
        2. 验证窗口合规性(防止 look-ahead bias)
        3. 采样覆盖率(7天内 >80% 组合覆盖)
        4. 执行率真实性(<90%)
        5. 冷启动组合检测
        6. EXPIRED订单验证覆盖率(确保所有EXPIRED订单都经过验证)
    """
    return run_all_validation_tests()

# 8. 运行单个验证测试 (Run Single Validation Test) - 新增
@app.get("/validate/{test_name}")
def validate_single_test(test_name: str):
    """
    运行单个验证测试

    参数:
        test_name: 测试名称
            - timestamp_correctness: 时间戳正确性
            - validation_window: 验证窗口合规性
            - sampling_coverage: 采样覆盖率
            - execution_rate: 执行率真实性
            - cold_start_detection: 冷启动检测
            - expired_orders_validated: EXPIRED订单验证覆盖率
    """
    test_functions = {
        'timestamp_correctness': test_timestamp_correctness,
        'validation_window': test_validation_window,
        'sampling_coverage': test_sampling_coverage,
        'execution_rate': test_execution_rate_realism,
        'cold_start_detection': test_cold_start_detection,
        'expired_orders_validated': test_expired_orders_validated,
    }

    if test_name not in test_functions:
        return JSONResponse(status_code=404, content={
            "error": f"Test '{test_name}' not found",
            "available_tests": list(test_functions.keys())
        })

    if not os.path.exists(DB_FILE):
        return JSONResponse(status_code=500, content={
            "error": f"Database not found: {DB_FILE}"
        })

    try:
        conn = sqlite3.connect(DB_FILE)
        result = test_functions[test_name](conn)
        conn.close()

        return {
            "test_name": test_name,
            "result": result
        }
    except Exception as e:
        logger.error(f"Test '{test_name}' failed: {e}")
        return JSONResponse(status_code=500, content={
            "error": str(e)
        })

# 9. 触发单独的数据下载 (Trigger Data Download) - 新增
@app.post("/download_data")
async def trigger_download(background_tasks: BackgroundTasks):
    """
    触发历史数据下载任务
    """
    current = get_current_status()
    if current.get("status") == "processing":
        return JSONResponse(status_code=409, content={
            "status": "busy",
            "message": f"Another task is running: {current.get('current_step')}"
        })

    async def run_download():
        update_status("processing", "Downloading Data", "Executing...")
        logger.info("Starting data download task")

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "funding_history_downloader",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(BASE_DIR),
                env=_build_subprocess_env()
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                err_msg = stderr.decode().strip() or stdout.decode().strip()
                logger.error(f"Data download failed: {err_msg}")
                update_status("error", "Download Data", f"Failed: {err_msg[-200:]}")
            else:
                logger.info("Data download completed successfully")
                update_status("online", "Idle", f"Download completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        except Exception as e:
            logger.error(f"Download task error: {e}")
            update_status("error", "Download Data", str(e))

    background_tasks.add_task(run_download)

    return {
        "status": "accepted",
        "message": "Data download task started. Check /status for progress."
    }

# 10. 触发特征处理 (Trigger Feature Processing) - 新增
@app.post("/process_features")
async def trigger_feature_processing(background_tasks: BackgroundTasks):
    """
    触发特征处理任务
    """
    current = get_current_status()
    if current.get("status") == "processing":
        return JSONResponse(status_code=409, content={
            "status": "busy",
            "message": f"Another task is running: {current.get('current_step')}"
        })

    async def run_processing():
        update_status("processing", "Processing Features", "Executing...")
        logger.info("Starting feature processing task")

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "ml_engine.data_processor",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(BASE_DIR),
                env=_build_subprocess_env()
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                err_msg = stderr.decode().strip() or stdout.decode().strip()
                logger.error(f"Feature processing failed: {err_msg}")
                update_status("error", "Process Features", f"Failed: {err_msg[-200:]}")
            else:
                logger.info("Feature processing completed successfully")
                update_status("online", "Idle", f"Processing completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        except Exception as e:
            logger.error(f"Processing task error: {e}")
            update_status("error", "Process Features", str(e))

    background_tasks.add_task(run_processing)

    return {
        "status": "accepted",
        "message": "Feature processing task started. Check /status for progress."
    }

# 11. 触发模型训练 (Trigger Model Training) - 新增
@app.post("/train")
async def trigger_training(background_tasks: BackgroundTasks):
    """
    触发模型训练任务
    """
    current = get_current_status()
    if current.get("status") == "processing":
        return JSONResponse(status_code=409, content={
            "status": "busy",
            "message": f"Another task is running: {current.get('current_step')}"
        })

    async def run_training():
        update_status("processing", "Training Models", "Executing...")
        logger.info("Starting model training task")

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "ml_engine.model_trainer",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(BASE_DIR),
                env=_build_subprocess_env()
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                err_msg = stderr.decode().strip() or stdout.decode().strip()
                logger.error(f"Model training failed: {err_msg}")
                update_status("error", "Train Models", f"Failed: {err_msg[-200:]}")
            else:
                logger.info("Model training completed successfully")
                update_status("online", "Idle", f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        except Exception as e:
            logger.error(f"Training task error: {e}")
            update_status("error", "Train Models", str(e))

    background_tasks.add_task(run_training)

    return {
        "status": "accepted",
        "message": "Model training task started. Check /status for progress."
    }

# 12. 触发预测生成 (Trigger Prediction) - 新增
@app.post("/predict")
async def trigger_prediction(background_tasks: BackgroundTasks):
    """
    触发预测生成任务
    """
    current = get_current_status()
    if current.get("status") == "processing":
        return JSONResponse(status_code=409, content={
            "status": "busy",
            "message": f"Another task is running: {current.get('current_step')}"
        })

    async def run_prediction():
        update_status("processing", "Generating Predictions", "Executing...")
        logger.info("Starting prediction generation task")

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "ml_engine.predictor",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(BASE_DIR),
                env=_build_subprocess_env()
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                err_msg = stderr.decode().strip() or stdout.decode().strip()
                logger.error(f"Prediction generation failed: {err_msg}")
                update_status("error", "Generate Predictions", f"Failed: {err_msg[-200:]}")
            else:
                prediction_failure = _extract_prediction_failure(_load_prediction_result())
                if prediction_failure:
                    logger.error(f"Prediction generation reported failure result: {prediction_failure}")
                    update_status("error", "Generate Predictions", f"Failed: {prediction_failure[-200:]}")
                    return
                logger.info("Prediction generation completed successfully")
                update_status("online", "Idle", f"Prediction completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        except Exception as e:
            logger.error(f"Prediction task error: {e}")
            update_status("error", "Generate Predictions", str(e))

    background_tasks.add_task(run_prediction)

    return {
        "status": "accepted",
        "message": "Prediction generation task started. Check /status for progress."
    }

# 13. 触发虚拟订单验证 (Trigger Order Validation) - 新增
@app.post("/validate_orders")
async def trigger_order_validation(background_tasks: BackgroundTasks):
    """
    手动触发虚拟订单验证
    """
    current = get_current_status()
    if current.get("status") == "processing":
        return JSONResponse(status_code=409, content={
            "status": "busy",
            "message": f"Another task is running: {current.get('current_step')}"
        })

    async def run_validation():
        update_status("processing", "Validating Orders", "Executing...")
        logger.info("Starting order validation task")

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(BASE_DIR / "ml_engine" / "execution_validator.py"),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(BASE_DIR),
                env=_build_subprocess_env()
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                err_msg = stderr.decode().strip() or stdout.decode().strip()
                logger.error(f"Order validation failed: {err_msg}")
                update_status("error", "Validate Orders", f"Failed: {err_msg[-200:]}")
            else:
                output = stdout.decode().strip()
                logger.info(f"Order validation completed: {output}")
                update_status("online", "Idle", f"Validation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        except Exception as e:
            logger.error(f"Validation task error: {e}")
            update_status("error", "Validate Orders", str(e))

    background_tasks.add_task(run_validation)

    return {
        "status": "accepted",
        "message": "Order validation task started. Check /status for progress."
    }

# 14. 触发闭环重训练 (Trigger Closed-Loop Retraining) - 新增 ⭐
@app.post("/retrain")
async def trigger_retraining(background_tasks: BackgroundTasks, force: bool = False):
    """
    触发闭环模型重训练

    参数:
    - force: 是否强制重训练（忽略判断条件）

    闭环流程:
    1. 检查是否需要重训练（基于执行结果）
    2. 如果需要，使用增强版训练器重新训练
    3. 融合虚拟订单执行反馈
    4. 自动对比新旧模型
    5. 如果新模型更好，自动部署
    """
    current = get_current_status()
    if current.get("status") == "processing":
        return JSONResponse(status_code=409, content={
            "status": "busy",
            "message": f"Another task is running: {current.get('current_step')}"
        })

    async def run_retraining():
        global _last_forced_retrain_time
        update_status("processing", "Closed-Loop Retraining", "Checking and retraining models...")
        logger.info("🔄 Starting closed-loop retraining task")

        try:
            # 构建命令
            cmd = [sys.executable, str(BASE_DIR / "ml_engine" / "retraining_scheduler.py")]
            if force:
                cmd.append("--force")

            stdout_text, stderr_text, rc = await _run_subprocess_with_timeout(
                cmd, BASE_DIR, TIMEOUT_TRAIN, "Closed-Loop Retraining"
            )

            if stdout_text:
                logger.info(f"Retraining output (last 10000 chars):\n{stdout_text[-10000:]}")
            if stderr_text:
                logger.warning(f"Retraining stderr (last 500 chars):\n{stderr_text[-500:]}")

            # exit code: 0=部署成功或训练成功但未部署, 1=训练失败, 2=无需重训练
            if rc == 1:
                # Extract meaningful error: prefer stdout error lines, filter normal stderr noise
                err_detail = ""
                if stdout_text:
                    for line in reversed(stdout_text.splitlines()):
                        if '❌' in line or '失败' in line or 'ERROR' in line or 'Traceback' in line:
                            err_detail = line.strip()
                            break
                if not err_detail and stderr_text:
                    noise_patterns = ['Capping', 'extreme', 'percentile', 'UserWarning', 'FutureWarning']
                    stderr_lines = [l.strip() for l in stderr_text.splitlines()
                                    if l.strip() and not any(p in l for p in noise_patterns)]
                    if stderr_lines:
                        err_detail = stderr_lines[-1]
                err_msg = err_detail or "Retraining failed"
                logger.error(f"❌ Closed-loop retraining failed: {err_msg}")
                update_status("degraded", "Closed-Loop Retraining", f"Training failed: {err_msg[-200:]}")
            elif rc == 2:
                logger.info(f"ℹ️ Closed-loop retraining: not needed")
                update_status("online", "Idle", f"Retraining not needed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                # rc == 0: 训练成功（可能部署也可能未部署）
                retrain_outcome = _parse_retraining_outcome(stdout_text or "")
                training_snapshot = _parse_training_data_snapshot(stdout_text or "")
                deployed = retrain_outcome in {'deployed', 'partially_deployed'}
                if force:
                    _last_forced_retrain_time = datetime.now()
                    save_retraining_state(
                        _last_forced_retrain_time,
                        reason="manual_force",
                        outcome=retrain_outcome,
                        training_data=training_snapshot,
                    )
                if deployed:
                    logger.info("✅ Closed-loop retraining completed and new models deployed")
                else:
                    logger.info("✅ Closed-loop retraining completed but models not deployed (keeping current)")
                update_status("online", "Idle", f"Retraining completed at {datetime.now().strftime('%Y-%m-%d:%S')}")

        except Exception as e:
            logger.error(f"❌ Retraining task error: {e}")
            update_status("error", "Closed-Loop Retraining", str(e))

    background_tasks.add_task(run_retraining)

    return {
        "status": "accepted",
        "message": f"Closed-loop retraining task started (force={force}). Check /status for progress.",
        "force_mode": force
    }

# --- Built-in Scheduler (S3) ---
_scheduler_task = None
_scheduler_lock_fd = None

async def _scheduled_pipeline_loop():
    """Background loop that runs the pipeline immediately, then every 2 hours."""
    logger.info("Built-in scheduler started: first run now, then every 2 hours")
    while True:
        try:
            logger.info("Scheduled pipeline trigger")
            await run_full_pipeline()
        except Exception as e:
            logger.error(f"Scheduled pipeline error: {e}")
        await asyncio.sleep(2 * 60 * 60)  # Wait 2 hours

@app.on_event("startup")
async def startup_event():
    """Start the built-in scheduler on server startup."""
    global _scheduler_task
    if _scheduler_task is not None and not _scheduler_task.done():
        logger.info("Scheduler already running, skipping duplicate registration")
        return
    # 文件锁防止多进程重复注册调度器 (持有全局引用防止GC回收fd)
    import fcntl
    global _scheduler_lock_fd
    _lock_path = os.path.join(BASE_DIR, ".scheduler.lock")
    _scheduler_lock_fd = open(_lock_path, 'w')
    try:
        fcntl.lockf(_scheduler_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (IOError, OSError):
        _scheduler_lock_fd.close()
        _scheduler_lock_fd = None
        logger.info("Another scheduler instance is running (lock held), skipping")
        return
    _scheduler_task = asyncio.create_task(_scheduled_pipeline_loop())
    logger.info(
        "Built-in scheduler registered "
        f"(last_forced_retrain={_last_forced_retrain_time.strftime('%Y-%m-%d %H:%M:%S') if _last_forced_retrain_time else 'none'})"
    )

@app.on_event("shutdown")
async def shutdown_event():
    """Cancel the scheduler on server shutdown."""
    global _scheduler_task
    if _scheduler_task:
        _scheduler_task.cancel()
        logger.info("Built-in scheduler cancelled")

if __name__ == "__main__":
    # 初始化状态
    update_status("online", "Idle", "Service started")
    # 启动服务
    uvicorn.run(app, host="0.0.0.0", port=5000, access_log=False)
