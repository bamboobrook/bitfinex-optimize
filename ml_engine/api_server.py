from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
import json
import os
import asyncio
from datetime import datetime
from loguru import logger
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger.add('/home/bumblebee/Project/optimize/log/ml_optimizer.log', retention='7 days', rotation="10 MB")

app = FastAPI(title="Lending Optimization API", version="3.0")

DATA_FILE = "../data/optimal_combination.json"
STATUS_FILE = "../data/service_status.json"

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

# --- 核心逻辑: 全量更新流水线 ---
async def run_full_pipeline():
    """
    后台任务: 执行完整的数据更新与模型训练流程
    步骤: 验证订单 -> 下载 -> 清洗 -> 训练 -> 预测
    """
    logger.info(">>> Starting full update pipeline...")

    # ========== STEP 0: 验证虚拟订单 (Validate Virtual Orders) ==========
    update_status("processing", "0. Validating Orders", "Checking pending orders...")
    logger.info("Step 0: Validating pending virtual orders")

    try:
        process = await asyncio.create_subprocess_exec(
            "python", "./execution_validator.py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.warning(f"Order validation had issues: {stderr.decode()}")
        else:
            logger.info(f"Order validation completed: {stdout.decode().strip()}")
    except Exception as e:
        logger.warning(f"Order validation failed: {e}, continuing with pipeline")
    # ====================================================================

    # 定义流水线步骤 (显示名称, 命令行指令)
    steps = [
        ("1. Downloading Data", ["python", "../funding_history_downloader.py"]),
        ("2. Processing Features", ["python", "./data_processor.py"]),
        ("3. Re-training Models", ["python", "./model_trainer.py"]),
        ("4. Generating Predictions", ["python", "./predictor.py"])
    ]

    for step_name, command in steps:
        update_status("processing", step_name, "Executing...")
        logger.info(f"Running: {step_name}")

        try:
            # 使用子进程调用脚本，确保内存独立且环境干净
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            # 等待完成
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                err_msg = stderr.decode().strip() or stdout.decode().strip()
                logger.error(f"{step_name} Failed: {err_msg}")
                update_status("error", step_name, f"Failed: {err_msg[-200:]}") # 只记录最后200字符
                return

        except Exception as e:
            logger.error(f"Pipeline System Error: {e}")
            update_status("error", step_name, str(e))
            return

    # 全部完成
    logger.info("<<< Pipeline completed successfully.")
    update_status("online", "Idle", "Last update completed at " + datetime.now().strftime('%H:%M:%S'))

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

if __name__ == "__main__":
    # 初始化状态
    update_status("online", "Idle", "Service started")
    # 启动服务
    uvicorn.run(app, host="0.0.0.0", port=5000)