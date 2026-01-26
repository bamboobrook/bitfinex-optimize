from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
import json
import os
import asyncio
from datetime import datetime
from loguru import logger

app = FastAPI(title="Lending Optimization API", version="3.0")

DATA_FILE = "data/optimal_combination.json"
STATUS_FILE = "data/service_status.json"

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
    步骤: 下载 -> 清洗 -> 训练 -> 预测
    """
    logger.info(">>> Starting full update pipeline...")
    
    # 定义流水线步骤 (显示名称, 命令行指令)
    steps = [
        ("1. Downloading Data", ["python", "funding_history_downloader.py"]),
        ("2. Processing Features", ["python", "ml_engine/data_processor.py"]),
        ("3. Re-training Models", ["python", "ml_engine/model_trainer.py"]),
        ("4. Generating Predictions", ["python", "ml_engine/predictor.py"])
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
    返回 API 在线状态及当前后台��务进度
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

if __name__ == "__main__":
    # 初始化状态
    update_status("online", "Idle", "Service started")
    # 启动服务
    uvicorn.run(app, host="0.0.0.0", port=8000)