from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
import json
import os
import asyncio
from datetime import datetime, timedelta
from loguru import logger
import sys
from pathlib import Path
import sqlite3
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent.parent))

logger.add('/home/bumblebee/Project/optimize/log/ml_optimizer.log', retention='7 days', rotation="10 MB")

app = FastAPI(title="Lending Optimization API", version="3.0")

# Use absolute paths
BASE_DIR = Path(__file__).parent.parent
DATA_FILE = str(BASE_DIR / "data" / "optimal_combination.json")
STATUS_FILE = str(BASE_DIR / "data" / "service_status.json")
DB_FILE = str(BASE_DIR / "data" / "lending_history.db")

# --- 辅助函数: 状态管理 ---
def update_status(status: str, step: str = "", details: str = ""):
    """更新服务状态文件"""
    info = {
        "status": status,         # 'online' (空闲在线), 'processing' (正在运行任务), 'error' (出错)
        "current_step": step,     # 当前正在执行的步骤
        "last_update": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "details": details
    }
    with open(STATUS_FILE, 'w') as f:
        json.dump(info, f, indent=4)

def get_current_status():
    """读取当前状态"""
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"status": "unknown", "details": "Status file not found"}

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

        # 7天执行率
        cursor.execute("""
            SELECT
                currency,
                period,
                COUNT(*) as total,
                SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) as executed,
                ROUND(100.0 * SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) / COUNT(*), 1) as exec_rate
            FROM virtual_orders
            WHERE order_timestamp >= datetime('now', '-7 days')
              AND status IN ('EXECUTED', 'FAILED')
            GROUP BY currency, period
            ORDER BY total DESC
            LIMIT 10
        """)
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
            "latest_orders": latest_orders
        }
    except Exception as e:
        logger.error(f"Failed to get DB statistics: {e}")
        return {"error": str(e)}

# --- 验证函数: 从 validate_fixes.py 整合 ---

def test_timestamp_correctness(conn):
    """
    验证订单时间戳正确性（5分钟内）
    """
    cursor = conn.cursor()
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
        ORDER BY created_at DESC
        LIMIT 10
    """)

    anomalies = cursor.fetchall()

    if not anomalies:
        return {
            "status": "PASS",
            "message": "All order timestamps are within 5 minutes of creation time",
            "anomaly_count": 0,
            "anomalies": []
        }
    else:
        return {
            "status": "FAIL",
            "message": f"Found {len(anomalies)} orders with suspicious timestamps",
            "anomaly_count": len(anomalies),
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
          AND status IN ('EXECUTED', 'FAILED')
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
          AND status IN ('EXECUTED', 'FAILED')
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

# --- 核心逻辑: 完整更新流水线 (每2小时运行) ---
async def run_full_pipeline():
    """
    后台任务: 执行完整的数据更新与模型训练流程
    步骤: 验证订单 -> 下载数据 -> 处理特征 -> 训练模型 -> 生成预测
    适合每2小时运行一次
    """
    logger.info("=" * 70)
    logger.info(">>> Starting full update pipeline (2-hour cycle)...")
    logger.info(f">>> Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    # ========== STEP 1: 验证虚拟订单 (Validate Virtual Orders) ==========
    update_status("processing", "1. Validating Orders", "Checking pending orders...")
    logger.info("Step 1: Validating pending virtual orders")

    try:
        process = await asyncio.create_subprocess_exec(
            "python", str(BASE_DIR / "ml_engine" / "execution_validator.py"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(BASE_DIR)
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.warning(f"Order validation had issues: {stderr.decode()}")
        else:
            logger.info(f"Order validation completed: {stdout.decode().strip()}")
    except Exception as e:
        logger.warning(f"Order validation failed: {e}, continuing with pipeline")
    # ====================================================================

    # 定义流水线步骤 (显示名称, Python模块路径)
    steps = [
        ("2. Downloading Data", ["python", "-m", "funding_history_downloader"]),
        ("3. Processing Features", ["python", "-m", "ml_engine.data_processor"]),
        ("4. Re-training Models", ["python", "-m", "ml_engine.model_trainer"]),
        ("5. Generating Predictions", ["python", "-m", "ml_engine.predictor"])
    ]

    for step_name, command in steps:
        update_status("processing", step_name, "Executing...")
        logger.info(f"Running: {step_name}")

        try:
            # 使用子进程调用脚本，确保内存独立且环境干净
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(BASE_DIR)
            )
            # 等待完成
            stdout, stderr = await process.communicate()

            # Capture and log subprocess output
            stdout_text = stdout.decode().strip()
            stderr_text = stderr.decode().strip()

            # Log subprocess output
            if stdout_text:
                logger.info(f"{step_name} output:\n{stdout_text}")
            if stderr_text:
                logger.warning(f"{step_name} stderr:\n{stderr_text}")

            if process.returncode != 0:
                err_msg = stderr_text or stdout_text or "Process exited with non-zero code"
                logger.error(f"{step_name} Failed (exit code {process.returncode}): {err_msg}")
                update_status("error", step_name, f"Failed: {err_msg[-200:]}")  # 只记录最后200字符
                return

            logger.info(f"{step_name} completed successfully (exit code 0)")

        except Exception as e:
            logger.error(f"Pipeline System Error: {e}")
            update_status("error", step_name, str(e))
            return

    # 全部完成 - 记录统计信息
    logger.info("=" * 70)
    logger.info("<<< Pipeline completed successfully!")
    logger.info(f"<<< End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 获取并记录统计信息
    stats = get_db_statistics()
    logger.info("=== Virtual Orders Summary ===")
    logger.info(f"Status: {stats.get('status_summary', [])}")
    logger.info(f"7-day execution rate: {len(stats.get('execution_rate_7d', []))} combinations validated")
    logger.info("=" * 70)

    update_status("online", "Idle", f"Last update completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- API 接口定义 ---

# 1. 确认 API 是否在线 (Health Check)
@app.get("/status")
def check_status():
    """
    返回 API 在线状态及当前后台任务进度
    """
    return {
        "api_online": True,
        "service_info": get_current_status()
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
        return data
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to read data: {str(e)}"})

# 3. 重新下载/训练/预测 (Trigger Update)
@app.post("/update")
async def trigger_update(background_tasks: BackgroundTasks):
    """
    触发后台全量更新: 下载新数据 -> 重新训练模型 -> 生成新预测
    """
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
    运行所有5个系统验证测试，返回完整验证报告

    测试包括:
        1. 时间戳正确性（5分钟内）
        2. 验证窗口合规性（防止 look-ahead bias）
        3. 采样覆盖率（7天内 >80% 组合覆盖）
        4. 执行率真实性（<90%）
        5. 冷启动组合检测
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
    """
    test_functions = {
        'timestamp_correctness': test_timestamp_correctness,
        'validation_window': test_validation_window,
        'sampling_coverage': test_sampling_coverage,
        'execution_rate': test_execution_rate_realism,
        'cold_start_detection': test_cold_start_detection,
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
                "python", "-m", "funding_history_downloader",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(BASE_DIR)
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
                "python", "-m", "ml_engine.data_processor",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(BASE_DIR)
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
                "python", "-m", "ml_engine.model_trainer",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(BASE_DIR)
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
                "python", "-m", "ml_engine.predictor",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(BASE_DIR)
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                err_msg = stderr.decode().strip() or stdout.decode().strip()
                logger.error(f"Prediction generation failed: {err_msg}")
                update_status("error", "Generate Predictions", f"Failed: {err_msg[-200:]}")
            else:
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
                "python", str(BASE_DIR / "ml_engine" / "execution_validator.py"),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(BASE_DIR)
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

if __name__ == "__main__":
    # 初始化状态
    update_status("online", "Idle", "Service started")
    # 启动服务
    uvicorn.run(app, host="0.0.0.0", port=5000)