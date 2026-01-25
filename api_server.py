"""
FastAPI REST API服务
提供预测结果的HTTP接口
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Optional
import logging
from datetime import datetime
import json
from pathlib import Path
import asyncio

from prediction_engine import PredictionEngine
from hardware_monitor import HardwareMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="Lending Rate Prediction API",
    description="预测最优放贷组合的API服务",
    version="1.0.0"
)

# 全局变量
prediction_engine: Optional[PredictionEngine] = None
hardware_monitor: Optional[HardwareMonitor] = None
latest_prediction: Optional[Dict] = None
prediction_in_progress: bool = False


class PredictionResponse(BaseModel):
    """预测响应模型"""
    optimal_combination: Dict
    detailed_metrics: Dict
    top_alternatives: list
    analysis_timestamp: str
    lookback_days_used: list
    total_combinations_evaluated: int
    hardware_utilization: Dict
    status: str


@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    global prediction_engine, hardware_monitor

    logger.info("="*60)
    logger.info("Starting Lending Rate Prediction API")
    logger.info("="*60)

    # 初始化硬件监控
    hardware_monitor = HardwareMonitor()
    hardware_monitor.optimize_pytorch()

    # 初始化预测引擎
    logger.info("Initializing prediction engine...")
    prediction_engine = PredictionEngine(
        db_path='data/lending_history.db',
        device='cuda'
    )

    # 加载数据
    logger.info("Loading historical data...")
    prediction_engine.load_data()

    logger.info("API server ready!")
    logger.info("="*60)


@app.on_event("shutdown")
async def shutdown_event():
    """关闭时清理"""
    logger.info("Shutting down API server...")


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "Lending Rate Prediction API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "prediction": "/api/v1/predict",
            "latest": "/api/v1/latest",
            "health": "/api/v1/health",
            "hardware": "/api/v1/hardware"
        }
    }


@app.get("/api/v1/health")
async def health_check():
    """健康检查"""
    if prediction_engine is None:
        raise HTTPException(status_code=503, detail="Prediction engine not initialized")

    stats = hardware_monitor.get_all_stats()

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_loaded": len(prediction_engine.data_cache) > 0,
        "gpu_available": hardware_monitor.has_gpu,
        "memory_usage_percent": stats.memory_percent,
        "gpu_memory_percent": stats.gpu_memory_percent
    }


@app.get("/api/v1/hardware")
async def get_hardware_stats():
    """获取硬件使用情况"""
    if hardware_monitor is None:
        raise HTTPException(status_code=503, detail="Hardware monitor not initialized")

    stats = hardware_monitor.get_all_stats()

    return {
        "cpu_usage_percent": stats.cpu_usage_percent,
        "memory_used_gb": stats.memory_used_gb,
        "memory_total_gb": stats.memory_total_gb,
        "memory_percent": stats.memory_percent,
        "gpu_memory_used_gb": stats.gpu_memory_used_gb,
        "gpu_memory_total_gb": stats.gpu_memory_total_gb,
        "gpu_memory_percent": stats.gpu_memory_percent,
        "gpu_utilization_percent": stats.gpu_utilization_percent,
        "gpu_temperature": stats.gpu_temperature,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/v1/latest")
async def get_latest_prediction():
    """获取最新的预测结果"""
    if latest_prediction is None:
        raise HTTPException(status_code=404, detail="No prediction available yet")

    return JSONResponse(content=latest_prediction)


async def run_prediction_async():
    """异步运行预测"""
    global latest_prediction, prediction_in_progress

    try:
        prediction_in_progress = True
        logger.info("Starting prediction...")

        # 运行预测
        result = prediction_engine.run_prediction()

        if result:
            # 添加硬件使用情况
            stats = hardware_monitor.get_all_stats()
            result['hardware_utilization'] = {
                'cpu_usage_percent': stats.cpu_usage_percent,
                'memory_used_gb': stats.memory_used_gb,
                'memory_total_gb': stats.memory_total_gb,
                'memory_percent': stats.memory_percent,
                'gpu_memory_used_gb': stats.gpu_memory_used_gb,
                'gpu_memory_total_gb': stats.gpu_memory_total_gb,
                'gpu_memory_percent': stats.gpu_memory_percent
            }
            result['status'] = 'success'

            # 保存最新预测
            latest_prediction = result

            # 保存到文件
            output_file = Path('data/optimal_combination.json')
            output_file.parent.mkdir(exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)

            logger.info(f"Prediction completed and saved to {output_file}")

        else:
            logger.error("Prediction failed")

    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)

    finally:
        prediction_in_progress = False


@app.post("/api/v1/predict")
async def trigger_prediction(background_tasks: BackgroundTasks):
    """触发新的预测"""
    global prediction_in_progress

    if prediction_engine is None:
        raise HTTPException(status_code=503, detail="Prediction engine not initialized")

    if prediction_in_progress:
        raise HTTPException(status_code=409, detail="Prediction already in progress")

    # 在后台运行预测
    background_tasks.add_task(run_prediction_async)

    return {
        "status": "prediction_started",
        "message": "Prediction started in background. Use /api/v1/latest to get results.",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/v1/predict/sync")
async def predict_sync():
    """同步预测（阻塞直到完成）"""
    global latest_prediction, prediction_in_progress

    if prediction_engine is None:
        raise HTTPException(status_code=503, detail="Prediction engine not initialized")

    if prediction_in_progress:
        raise HTTPException(status_code=409, detail="Prediction already in progress")

    try:
        prediction_in_progress = True
        logger.info("Starting synchronous prediction...")

        # 运行预测
        result = prediction_engine.run_prediction()

        if result:
            # 添加硬件使用情况
            stats = hardware_monitor.get_all_stats()
            result['hardware_utilization'] = {
                'cpu_usage_percent': stats.cpu_usage_percent,
                'memory_used_gb': stats.memory_used_gb,
                'memory_total_gb': stats.memory_total_gb,
                'memory_percent': stats.memory_percent,
                'gpu_memory_used_gb': stats.gpu_memory_used_gb,
                'gpu_memory_total_gb': stats.gpu_memory_total_gb,
                'gpu_memory_percent': stats.gpu_memory_percent
            }
            result['status'] = 'success'

            # 保存最新预测
            latest_prediction = result

            # 保存到文件
            output_file = Path('data/optimal_combination.json')
            output_file.parent.mkdir(exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)

            logger.info(f"Prediction completed and saved to {output_file}")

            return JSONResponse(content=result)

        else:
            raise HTTPException(status_code=500, detail="Prediction failed")

    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        prediction_in_progress = False


@app.get("/api/v1/status")
async def get_status():
    """获取预测状态"""
    return {
        "prediction_in_progress": prediction_in_progress,
        "latest_prediction_available": latest_prediction is not None,
        "latest_prediction_time": latest_prediction.get('analysis_timestamp') if latest_prediction else None,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn

    # 运行服务器
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        workers=1  # 单worker以避免多进程问题
    )
