每次执行命令的时候,请叫我达拉崩吧.

## Project Overview

Bitfinex lending rate prediction system. Predicts optimal fUSD/fUST lending rates across 14 periods (2-120 days) using an XGBoost/LightGBM/CatBoost ensemble. Closed-loop pipeline: predict → virtual orders → validate against market → adjust.

- **Stack**: Python, FastAPI, XGBoost, LightGBM, CatBoost, SQLite, loguru
- **Currencies**: fUSD, fUST
- **Periods**: 2, 3, 4, 5, 6, 7, 10, 14, 15, 20, 30, 60, 90, 120 days

## Commands

```bash
# API server (port 5000)
python ml_engine/api_server.py

# Download historical market data (fUSD + fUST, 1200 days, 1-min candles)
python funding_history_downloader.py

# Trigger full closed-loop pipeline
curl -X POST http://localhost:5000/update

# Check server status
curl http://localhost:5000/status

# Run all validation tests
curl http://localhost:5000/validate

# View logs
tail -f log/ml_optimizer.log

# Query database
sqlite3 data/lending_history.db
```

No requirements.txt, no test framework, no linter configured.

## Architecture

**Closed-loop pipeline** (triggered by POST `/update`):

1. **Validate orders** — check pending virtual orders against market for execution feedback
2. **Download data** — fetch latest Bitfinex funding rate candles
3. **Check retraining** — evaluate if models need retraining based on execution results
4. **Retrain models** (if needed) — train with execution feedback features
5. **Generate predictions + create virtual orders** — predict rates, place new virtual orders

**Key modules** (`ml_engine/`):

| File | Role |
|------|------|
| `api_server.py` | FastAPI server, pipeline orchestration |
| `data_processor.py` | Feature engineering (~100+ features: lags, RSI, MACD, Bollinger, execution feedback) |
| `model_trainer.py` | XGBoost/LightGBM/CatBoost ensemble training (GPU-accelerated) |
| `model_trainer_v2.py` | Enhanced training with revenue-optimized targets |
| `predictor.py` | Rate prediction using trained ensemble |
| `order_manager.py` | Virtual order creation, validation, execution stats |
| `execution_validator.py` | System validation tests |
| `execution_features.py` | Execution-based feature computation |
| `training_data_builder.py` | Fuses market data + virtual order execution results |
| `retraining_scheduler.py` | Decides when retraining is needed |
| `metrics.py` | Metric tracking |

**4 model targets**: `execution_prob` (binary classification), `conservative`, `aggressive`, `balanced` (regression). Ensemble weights: 1/MAE for regression, AUC for classification.

## API Endpoints

| Method | Route | Purpose |
|--------|-------|---------|
| GET | `/status` | Health check, current task progress |
| GET | `/result` | Latest predictions (optimal_combination.json) |
| GET | `/execution_stats` | Execution stats by currency/period (params: currency, period, days) |
| GET | `/orders` | Virtual orders list (params: status, limit) |
| GET | `/stats` | DB statistics, 7-day execution rates |
| GET | `/validate` | Run all 6 validation tests |
| GET | `/validate/{test_name}` | Run single test |
| POST | `/update` | Full closed-loop pipeline |
| POST | `/download_data` | Download market data only |
| POST | `/process_features` | Feature processing only |
| POST | `/train` | Model training only |
| POST | `/predict` | Prediction only |
| POST | `/validate_orders` | Manual order validation |
| POST | `/retrain` | Closed-loop retraining (param: force) |

## Key Patterns

- **Hardcoded paths**: BASE_DIR derived from `__file__`, all paths relative to it
- **Chinese comments**: Comments and log messages throughout codebase are in Chinese
- **Background tasks**: FastAPI BackgroundTasks for pipeline steps
- **Logging**: loguru → `log/ml_optimizer.log`
- **Env vars** (`.env`): `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`, `TELEGRAM_TEST_MODE`, `TEST_MODE`
- **No formal tests**: Validation via `/validate` endpoint (6 tests: timestamp_correctness, validation_window, sampling_coverage, execution_rate, cold_start_detection, expired_orders_validated)
- **Virtual order lifecycle**: PENDING → EXECUTED/FAILED/EXPIRED (validation windows: 24h for ≤7d periods, 48h for 8-30d, 72h for >30d)

## 2025-02-10 大修后核心审核项（下次对话时执行）

以下3项是本次大修（18项修改，5个阶段）后需要验证的核心关注点。用户下次咨询时，按此清单逐项审核：

### 审核1: v2模型是否参与预测（S1修复验证）
- **为什么重要**: 之前 model_trainer_v2 训练了 execution_prob_v2 和 revenue_optimized 模型，但 predictor.py 从未使用，闭环断裂
- **怎么查**: 查日志 `grep "v2_execution_prob\|v2_revenue_rate\|V2 model" log/ml_optimizer.log`
- **期望结果**: 预测输出中应包含 v2_execution_prob 和 v2_revenue_rate 字段值
- **如果没有**: 检查 `data/models/` 下是否存在 `*_v2*` 和 `*revenue*` 模型文件；检查 predictor.py 中 v2 加载逻辑

### 审核2: execution_statistics 表是否有数据（问题6修复验证）
- **为什么重要**: 该表是训练反馈的基础，之前一直为空（0行），导致闭环训练缺少执行反馈数据
- **怎么查**: `sqlite3 data/lending_history.db "SELECT COUNT(*) FROM execution_statistics; SELECT * FROM execution_statistics ORDER BY stat_date DESC LIMIT 5;"`
- **期望结果**: 行数 > 0，且有近期日期的聚合记录
- **如果仍为空**: 检查 execution_validator.py 末尾是否调用了 aggregate_execution_statistics()；检查 virtual_orders 表中是否有已验证订单

### 审核3: 2小时自动调度是否正常运行（S3修复验证）
- **为什么重要**: 之前无内置调度，完全依赖外部手动调用，如果外部调度断了系统就停了
- **怎么查**: `GET /status` 查看 last_update 是否每2小时刷新；`grep "Scheduled pipeline trigger" log/ml_optimizer.log`
- **期望结果**: 日志中出现 "Scheduled pipeline trigger (every 2h)" 且间隔约2小时
- **如果没有**: 确认服务是否重启过（重启后 scheduler 自动注册）；检查日志中 startup_event 是否执行

### 次要关注
- clipping_rate 是否从 38% 下降（收紧了 clipping bounds）
- 重复订单是否消失（加了去重逻辑）
- 是否出现 subprocess timeout 日志（加了超时保护）
- retraining_history.json 中是否有 MAE/AUC 对比数据（模型性能对比修复）

## Data Files (all in .gitignore)

| Path | Content |
|------|---------|
| `data/lending_history.db` | Main SQLite DB (~570MB). Tables: `funding_rates`, `virtual_orders`, `execution_statistics`, `order_execution`, `prediction_history` |
| `data/optimal_combination.json` | Latest prediction results |
| `data/service_status.json` | API status |
| `data/system_metrics.json` | System metrics |
| `data/models/` | Trained model files |
| `data/processed/` | Processed feature data |
| `log/ml_optimizer.log` | Application log |
