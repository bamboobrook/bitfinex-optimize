from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import json
import os
import sys
import subprocess
from loguru import logger
from datetime import datetime
import asyncio

# 添加父目录到 path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_engine.predictor import LendingPredictor

app = FastAPI(title="Lending Optimization API", version="2.0")

DATA_FILE = "data/optimal_combination.json"
STATUS_FILE = "data/service_status.json"

# 全局 Predictor 实例 (Lazy loading)
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = LendingPredictor()
    return _predictor

def update_status(status: str, details: str = ""):
    with open(STATUS_FILE, 'w') as f:
        json.dump({
            "status": status,
            "last_update": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "details": details
        }, f)

async def run_pipeline():
    """
    运行完整的数据更新和预测流水线
    """
    logger.info("Starting pipeline update...")
    update_status("running", "Downloading data...")
    
    try:
        # 1. 下载数据 (调用现有脚本)
        # 使用 subprocess 调用 funding_history_downloader.py
        # 注意：这里假设 downloader 可以在当前环境下运行
        logger.info("Running downloader...")
        process = await asyncio.create_subprocess_exec(
            "python", "../funding_history_downloader.py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Downloader failed: {stderr.decode()}")
            update_status("error", "Downloader failed")
            return

        # 2. 预测生成 (使用我们的高效模块)
        logger.info("Generating recommendations...")
        update_status("running", "Predicting...")
        
        predictor = get_predictor()
        # 注意：Predictor 每次都会重新加载最新数据，因为 load_data 是读文件的
        # 但为了效率，我们可以只让它读最新的 (目前逻辑是读全部，鉴于速度够快先保持)
        predictor.generate_recommendations(DATA_FILE)
        
        update_status("idle", "Update completed successfully")
        logger.info("Pipeline completed.")
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        update_status("error", str(e))

@app.get("/")
def health_check():
    status = {"status": "unknown"}
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, 'r') as f:
            status = json.load(f)
    return {
        "service": "Lending Optimization API", 
        "version": "2.0",
        "pipeline_status": status
    }

@app.get("/recommendation")
def get_recommendation():
    """
    获取最新的放贷组合建议
    """
    if not os.path.exists(DATA_FILE):
        return JSONResponse(status_code=404, content={"error": "No recommendations available yet. Please trigger an update."})
    
    with open(DATA_FILE, 'r') as f:
        data = json.load(f)
    return data

@app.post("/trigger_update")
async def trigger_update(background_tasks: BackgroundTasks):
    """
    手动触发一次数据更新和预测
    """
    background_tasks.add_task(run_pipeline)
    return {"message": "Update pipeline triggered in background."}

if __name__ == "__main__":
    # 确保 data 目录存在
    os.makedirs("data", exist_ok=True)
    update_status("idle", "Service started")
    
    # 启动服务器
    uvicorn.run(app, host="0.0.0.0", port=8000)
