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
- **怎么查**: `sqlite3 data/lending_history.db "SELECT COUNT(*) FROM execution_statistics; SELECT * FROM execution_statistics ORDER BY date DESC LIMIT 5;"`
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

## 2025-02-22 长周期执行率优化后审核项（下次对话时执行）

本次修改解决120天执行率低而60/90天执行率高、但系统无法有效自动纠偏的问题。修改了3个文件，下次审核重点如下：

### 审核4: 按period分组重训练是否生效（retraining_scheduler.py 修复验证）
- **为什么重要**: 之前重训练触发用的是全局执行率，120天低执行率被60/90天稀释，永远不会触发紧急重训练
- **改了什么**: 新增 `get_per_period_execution_anomalies()` 方法，`should_retrain()` 增加第3个触发条件——单个 currency+period 执行率 < 30% 或 > 75% 即触发
- **怎么查**:
  ```bash
  grep "单period成交率" log/ml_optimizer.log
  grep "按 period 分组" log/ml_optimizer.log
  ```
- **期望结果**: 如果120天执行率 < 30%，日志中应出现 "单period成交率过低 (fUSD 120天=XX%), 紧急重训练"
- **如果没有**: 检查120天近7天是否有 >= 5条已验证订单（min_orders阈值）；用SQL查：
  ```sql
  SELECT currency, period, COUNT(*), SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) as exec
  FROM virtual_orders
  WHERE order_timestamp >= date('now', '-7 days') AND status IN ('EXECUTED','FAILED')
  GROUP BY currency, period;
  ```

### 审核5: 长周期调整系数是否加大（predictor.py 修复验证）
- **为什么重要**: 之前所有周期统一调整上限15%，对120天远远不够；且长周期+5%奖励在低执行率时抵消了惩罚
- **改了什么**:
  - `_calculate_execution_adjustment()` 新增 `period` 参数，>=60天使用更大步长（最大降幅从18%→30%）
  - `_get_currency_adjustment_factor()` 长周期+5%奖励改为仅 exec_rate >= 40% 时生效
- **怎么查**: 对比修改前后120天的预测利率
  ```bash
  # 查看最新预测中120天的数据
  python -c "
  import json
  data = json.load(open('data/optimal_combination.json'))
  for r in data.get('results', data) if isinstance(data, list) else data.get('results', []):
      if r.get('period') == 120:
          print(f\"{r.get('currency')} 120d: rate={r.get('optimal_rate', r.get('predicted_rate'))}, exec_adj={r.get('execution_adjustment', 'N/A')}\")
  "
  ```
- **期望结果**: 120天预测利率应低于修改前（降幅更大），且执行率应在后续几个周期内上升
- **量化验证**: 运行2-3个周期（4-6小时）后：
  ```sql
  SELECT period,
         COUNT(*) as total,
         SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) as executed,
         ROUND(100.0 * SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) / COUNT(*), 1) as exec_rate
  FROM virtual_orders
  WHERE order_timestamp >= date('now', '-2 days')
    AND status IN ('EXECUTED','FAILED')
    AND period IN (60, 90, 120)
  GROUP BY period;
  ```
  120天执行率应从当前水平开始上升，目标是靠近50%

### 审核6: 长周期冷启动阈值是否生效（execution_features.py 修复验证）
- **为什么重要**: 120天订单少，之前需要 >= 10条才脱离冷启动默认值0.6，可能长期用不上真实执行率
- **改了什么**: period >= 60 天冷启动阈值从10条降到5条
- **怎么查**:
  ```sql
  SELECT currency, period, COUNT(*) as order_count
  FROM virtual_orders
  WHERE status IN ('EXECUTED','FAILED')
    AND order_timestamp >= date('now', '-7 days')
    AND period IN (60, 90, 120)
  GROUP BY currency, period;
  ```
- **期望结果**: 如果某个长周期组合有 5-9 条订单，现在应该使用真实执行率而非默认0.6
- **关联验证**: 对比 exec_rate_7d 是否与SQL查出的真实比率一致（而非0.6）

### 综合回归验证
- 60/90天执行率不应显著下降（它们不受长周期修改影响，除非全局重训练被更频繁触发）
- 短周期(2-30天)行为不变（`_calculate_execution_adjustment` 对 period < 60 走原逻辑）
- 系统无报错：`grep "Error\|Exception\|Traceback" log/ml_optimizer.log | tail -20`

## 2025-02-22 执行率均衡优化审核项（下次对话时执行）

本次修改解决全周期执行率不均衡问题：60/90天过高(96-100%)、30天过低(20-26%)、2d/4d偏低(34-37%)、10-15d偏高(77-83%)。修改了4个文件+CLAUDE.md。

### 审核7: 中周期分类+上调分级是否生效（predictor.py 修复验证）
- **为什么重要**: 之前只有长/短二分法，30天归为短周期调整步长不够；上调仅1级(1.02/1.04/1.08)严重不对称
- **改了什么**:
  - `_calculate_execution_adjustment()` 新增 `is_medium_period` (20-59天)，下调和上调均3级分级
  - 上调步长: 短周期1.05/1.10, 中周期1.06/1.12, 长周期1.08/1.15
  - 安全边界分级: 短[0.82,1.18], 中[0.76,1.20], 长[0.70,1.25]
- **怎么查**: 对比修改前后30天和60/90天预测利率
  ```bash
  python -c "
  import json
  data = json.load(open('data/optimal_combination.json'))
  for r in data.get('recommendations', []):
      p = r.get('period', 0)
      if p in [2, 4, 10, 15, 30, 60, 90, 120]:
          print(f\"{r.get('currency')} {p}d: rate={r.get('rate')}\")
  "
  ```
- **期望结果**: 30天利率降低(→执行率升高), 60/90天利率升高(→执行率降低)

### 审核8: risk_adjustment_factor 对称化是否一致（execution_features.py + data_processor.py）
- **为什么重要**: 训练时和推理时的 risk_adjustment_factor 必须一致,否则特征分布不匹配导致模型预测偏移
- **改了什么**: 两处均改为7级: [<0.3→0.90, <0.4→0.94, <0.5→0.97, ≤0.6→1.0, ≤0.7→1.03, ≤0.85→1.06, >0.85→1.10]
- **怎么查**:
  ```bash
  grep -n "risk_adjustment_factor" ml_engine/execution_features.py ml_engine/data_processor.py
  ```
- **期望结果**: 两处逻辑完全一致（阈值和值都相同）

### 审核9: 币种调整因子上调能力是否增强（predictor.py _get_currency_adjustment_factor）
- **改了什么**: 上调斜率0.12→0.15, 长周期奖励从统一+5%改为分级(exec_rate≥0.70→+6%, ≥0.40→+3%), 上限1.18→1.22
- **怎么查**: 日志中搜索高执行率周期的调整因子
- **期望结果**: 60/90天(执行率>90%)复合因子应显著大于1.0

### 审核10: 高执行率异常阈值是否收紧（retraining_scheduler.py）
- **改了什么**: `get_per_period_execution_anomalies` 中高执行率阈值从 0.75 → 0.70
- **怎么查**: 如果任何 period 执行率 > 70%,应触发重训练
  ```bash
  grep "单period成交率过高" log/ml_optimizer.log
  ```

### 量化验证（部署后4-6小时）
```sql
SELECT currency, period,
       COUNT(*) as total,
       SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) as executed,
       ROUND(100.0 * SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) / COUNT(*), 1) as exec_rate
FROM virtual_orders
WHERE order_timestamp >= datetime('now', '-6 hours')
  AND status IN ('EXECUTED','FAILED')
GROUP BY currency, period
ORDER BY currency, period;
```
目标:
- 30天: 20-26% → 35-45%
- 60天: 96% → 55-70%
- 90天: 100% → 55-70%
- 2d/4d: 34-37% → 40-45%
- 10-15d: 77-83% → 60-70%
- 20天: ~50% 维持

## 2025-02-24 v5预测引擎重构审核项（下次对话时执行）

本次修改解决核心问题：三因子乘法级联导致调整信号重复计算、校准概率双重惩罚、策略选择硬阈值跳变。修改了4个文件共10项优化。

### 修改前基线数据（2025-02-24 修改前7日执行率）

| 币种-周期 | 修改前执行率 | 目标 | 问题 |
|-----------|-------------|------|------|
| fUSD 60d | 100% | 50% | 极高 |
| fUSD 90d | 100% | 50% | 极高 |
| fUST 60d | 100% | 50% | 极高 |
| fUST 90d | 100% | 50% | 极高 |
| fUST 14d | 95.5% | 50% | 极高 |
| fUSD 10d | 81.5% | 50% | 偏高 |
| fUSD 14d | 82.1% | 50% | 偏高 |
| fUSD 15d | 84.0% | 50% | 偏高 |
| fUST 7d | 81.6% | 50% | 偏高 |
| fUST 10d | 86.2% | 50% | 偏高 |
| fUSD 30d | 29.6% | 50% | 偏低 |
| fUST 2d | 20.0% | 50% | 极低 |
| fUST 120d | 17.6% | 50% | 极低 |
| fUSD 2d | 37.5% | 50% | 偏低 |
| fUSD 3d | 28.6% | 50% | 偏低 |

### 修改部署后首次预测的调整因子（2025-02-24 15:09 验证通过）

| 币种-周期 | exec_rate_7d | adj因子 | 方向 | clipping_rate: 38%→20% |
|-----------|-------------|---------|------|------------------------|
| fUSD-60d | 1.000 | 1.3500 | 上调利率 | 策略分布: balanced 68% |
| fUSD-90d | 1.000 | 1.3500 | 上调利率 | conservative 24% |
| fUST-60d | 1.000 | 1.3500 | 上调利率 | aggressive 8% |
| fUST-120d | 0.188 | 0.7007 | 大幅下调 | |
| fUSD-30d | 0.288 | 0.7746 | 下调利率 | |
| fUST-2d | 0.250 | 0.8200 | 下调利率 | |
| fUSD-2d | 0.414 | 0.8527 | 下调利率 | |
| 纠偏方向正确率 | | | 17/17=100% | |

### 审核11: 统一调整因子是否生效（P0-1，predictor.py）
- **改了什么**: 合并原三因子(`_calculate_execution_adjustment × risk_adjustment × _get_currency_adjustment_factor`)为单一函数 `_get_unified_adjustment()`。非线性映射+周期感知灵敏度。删除了 `_calculate_execution_adjustment()` 和 `_get_currency_adjustment_factor()` 两个方法
- **怎么查**:
  ```bash
  grep "PREDICTION_DIAG" log/ml_optimizer.log | tail -28
  ```
- **期望结果**: 日志中 `adj=` 值应在 [0.70, 1.35] 范围内；exec_rate=1.0 的长周期 adj=1.35；exec_rate<0.3 的长周期 adj<0.80
- **如果异常**: 检查 `_get_unified_adjustment` 函数，确认 `max_up/max_down` 是否按周期分级

### 审核12: 贝叶斯校准概率是否正确（P0-2，predictor.py）
- **改了什么**: 移除旧版 `raw_calibrated = prob × (exec_rate_7d / 0.7)` + 硬性下限覆盖。改为 `evidence_weight = min(order_count/50, 0.5)`，`calibrated_prob = (1-ew)*prob + ew*exec_rate_7d`
- **怎么查**:
  ```bash
  grep "calib_prob" log/ml_optimizer.log | tail -28
  ```
- **期望结果**: exec_rate=1.0 + orders=156 时 calib_prob ≈ 0.74（不是旧版的0.85）；exec_rate=0.20 + 少量orders时 calib_prob 接近模型原始prob
- **关键**: 不应再出现 `v4 calibration floor` 日志（该逻辑已删除）

### 审核13: 策略连续插值是否消除跳变（P1-1，predictor.py）
- **改了什么**: 硬阈值(0.40/0.75)改为连续权重插值 `w_cons/w_bal/w_aggr`
- **怎么查**:
  ```bash
  grep "strategy=" log/ml_optimizer.log | grep "PREDICTION_DIAG" | tail -28
  ```
- **期望结果**: 只出现 `Balanced`、`Conservative-leaning`、`Aggressive-leaning`。不应出现旧标签 `High Certainty` 或 `High Yield`

### 审核14: 市场锚定动态权重（P1-2，predictor.py）
- **改了什么**: 原来 `0.5*model_rate + 0.5*(current_rate*combined_factor)` 改为 `conf_weight*model_rate + (1-conf_weight)*current_rate`，conf_weight 基于 confidence（High=0.65, Medium=0.50, Low=0.35）
- **怎么查**: PREDICTION_DIAG 日志中 `conf_w=` 字段
- **期望结果**: confidence=High 时 conf_w=0.65，Medium=0.50，Low=0.35

### 审核15: 趋势因子按周期分窗口（P2-1，predictor.py）
- **改了什么**: 所有周期统一用 `rate_chg_60`(1h) + weight=0.10 → 改为长周期用 `rate_chg_1440`(24h)+3%、中周期 `rate_chg_240`(4h)+5%、短周期 `rate_chg_60`(1h)+8%
- **期望结果**: PREDICTION_DIAG 中长周期的 `trend_adj` 绝对值较小

### 审核16: 重训练阈值严重性分级（P2-2，retraining_scheduler.py）
- **改了什么**: 原阈值 `<0.30 或 >0.70` → 新增分级 critical(`<0.20 或 >0.85`) + warning(`<0.30 或 >0.65`)
- **怎么查**:
  ```bash
  grep "单period成交率" log/ml_optimizer.log | tail -10
  ```
- **期望结果**: 日志中应出现 `[critical]` 或 `[warning]` 标签

### 审核17: 安全边界基于均线（P2-3，predictor.py）
- **改了什么**: bounds 基准从 `current_rate` 改为 `0.6×ma_1440 + 0.4×current_rate`
- **期望结果**: 当 current_rate 处于低点时，bounds 不会过度收紧

### 审核18: 冷启动默认值差异化（P3-1，execution_features.py）
- **改了什么**: 统一默认0.6 → `period<=7: 0.55, 8-30: 0.50, >30: 0.45`
- **怎么查**: 看冷启动组合（order_count < threshold）的 exec_rate_7d 是否匹配新默认值
- **关联**: data_processor.py 中 `_apply_default_exec_features` 的默认值仍为0.6（这是异常回退值，与冷启动默认值不同）

### 审核19: 训练特征时间化（P2-4，data_processor.py）
- **改了什么**: exec_rate_7d/30d 从广播（所有样本用同一值）改为每周采样滚动计算（`as_of_date` + `ffill`）
- **怎么查**: 下次重训练时观察日志，确认无报错
- **注意**: 此项需要重训练才能验证效果

### 量化验证（部署后6-12小时执行）

```bash
# 1. 检查 PREDICTION_DIAG 日志完整性（应有28条，14 periods × 2 currencies）
grep "PREDICTION_DIAG" log/ml_optimizer.log | tail -28 | wc -l

# 2. 检查无错误
grep -E "Error|Exception|Traceback" log/ml_optimizer.log | grep "$(date +%Y-%m-%d)" | tail -20

# 3. 执行率趋势（对比修改前基线）
sqlite3 data/lending_history.db "
SELECT currency, period,
       COUNT(*) as total,
       SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) as executed,
       ROUND(100.0 * SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) / COUNT(*), 1) as exec_rate
FROM virtual_orders
WHERE order_timestamp >= datetime('now', '-12 hours')
  AND status IN ('EXECUTED','FAILED')
GROUP BY currency, period
ORDER BY currency, period;"

# 4. 对比预测利率与市场价偏差
python3 -c "
import json
data = json.load(open('data/optimal_combination.json'))
for r in data.get('recommendations', []):
    p, c, rate = r['period'], r['currency'], r['rate']
    cur = r.get('details', {}).get('current', 0)
    dev = ((rate - cur) / cur * 100) if cur > 0 else 0
    print(f'{c} {p:>3}d: predicted={rate:.4f} current={cur:.4f} deviation={dev:+.1f}%')
"
```

**目标执行率（修改后趋向）**:
- 60/90d: 100% → 下降趋向 60-75%（第一轮可能降幅有限，需要2-3轮）
- 30d fUSD: 29.6% → 上升趋向 35-45%
- 2d fUST: 20.0% → 上升趋向 30-40%
- 120d fUST: 17.6% → 上升趋向 25-35%
- 10-15d: 80-86% → 下降趋向 65-75%

**注意**: 如果12小时后60/90d执行率仍为100%，说明市场持续承接，模型调整需要更多轮次积累。不应视为修改失败。

## 2025-02-24 v5 预测引擎重构审核项（下次对话时执行）

本次修改是对预测引擎的全面重构，解决三因子乘法级联、校准概率双重惩罚、策略硬阈值跳变等根本性问题。修改了4个文件共10项优化。部署时间: 2026-02-24 15:07。

### 修改前基线数据（7日执行率，2026-02-24 采集）

| 币种-周期 | 修改前执行率 | 目标 |
|-----------|-------------|------|
| fUSD 60d | 100% | 50% |
| fUSD 90d | 100% | 50% |
| fUST 60d | 100% | 50% |
| fUST 90d | 100% | 50% |
| fUST 14d | 95.5% | 50% |
| fUST 10d | 86.2% | 50% |
| fUSD 15d | 84.0% | 50% |
| fUSD 14d | 82.1% | 50% |
| fUSD 10d | 81.5% | 50% |
| fUST 7d | 81.6% | 50% |
| fUSD 20d | 72.4% | 50% |
| fUST 4d | 66.7% | 50% |
| fUST 6d | 65.0% | 50% |
| fUSD 120d | 64.3% | 50% |
| fUST 30d | 56.7% | 50% |
| fUST 20d | 52.6% | 50% |
| fUST 3d | 52.6% | 50% |
| fUSD 6d | 52.8% | 50% |
| fUSD 5d | 47.5% | 50% |
| fUSD 7d | 45.7% | 50% |
| fUSD 4d | 40.6% | 50% |
| fUSD 2d | 37.5% | 50% |
| fUSD 30d | 29.6% | 50% |
| fUSD 3d | 28.6% | 50% |
| fUST 120d | 17.6% | 50% |
| fUST 2d | 20.0% | 50% |

修改前系统指标: clipping_rate=38%, 策略分布不明

### 修改后首次预测验证数据（2026-02-24 15:09）

部署后首次预测的 PREDICTION_DIAG 关键数据:

| 币种-周期 | current | adj因子 | final | exec_7d | calib_prob | 策略 | clipped |
|-----------|---------|--------|-------|---------|-----------|------|---------|
| fUSD-60d | 16.60 | 1.3500 | 12.75 | 1.000 | 0.740 | Aggressive | No |
| fUSD-90d | 16.60 | 1.3500 | 13.40 | 1.000 | 0.836 | Aggressive | No |
| fUSD-120d | 6.61 | 1.0639 | 6.73 | 0.720 | 0.577 | Balanced | No |
| fUSD-30d | 0.15 | 0.7746 | 4.70 | 0.288 | 0.343 | Balanced | Yes |
| fUST-120d | 15.46 | 0.7007 | 8.70 | 0.188 | 0.286 | Balanced | No |
| fUST-60d | 15.48 | 1.3500 | 13.59 | 1.000 | 0.821 | Aggressive | No |
| fUST-2d | 6.57 | 0.8200 | 4.72 | 0.250 | 0.379 | Balanced | No |
| fUSD-2d | 4.19 | 0.8527 | 3.26 | 0.414 | 0.497 | Balanced | Yes |

修改后系统指标: clipping_rate=20%, 策略分布: balanced 68%, conservative 24%, aggressive 8%

### 审核11: 统一调整因子是否生效（P0-1, predictor.py）
- **改了什么**: 删除 `_calculate_execution_adjustment()` 和 `_get_currency_adjustment_factor()`，合并为 `_get_unified_adjustment()`。原来三因子 `execution_adj × risk_adj × currency_factor` 都对 exec_rate_7d 反应，同一信号被计入三次。新函数单一非线性映射，周期感知灵敏度: 长周期[0.70,1.35], 中周期[0.76,1.28], 短周期[0.82,1.20]
- **怎么查**:
  ```bash
  grep "PREDICTION_DIAG" log/ml_optimizer.log | tail -28
  ```
- **期望结果**: adj 字段不应超过 1.35（长周期）或 1.20（短周期）。exec_rate=1.0 时 adj=1.35（命中上限），exec_rate=0.5 时 adj≈1.0
- **已验证**: 首次预测纠偏方向 17/17 = 100% 正确

### 审核12: 贝叶斯校准是否消除双重惩罚（P0-2, predictor.py）
- **改了什么**: 删除旧逻辑 `raw_calibrated = prob × (exec_rate/0.7)` + 硬性下限覆盖 `max(calib, exec_rate×0.85)`。改为 `evidence_weight = min(order_count/50, 0.5)`，`calibrated_prob = (1-ew)*prob + ew*exec_rate_7d`
- **怎么查**: 日志中 calib_prob 字段。exec_rate=1.0 时 calib_prob 不应是固定的 0.85
- **期望结果**: fUSD-60d(exec=1.0, orders=156) → ew=0.5, calib=0.5*prob+0.5*1.0。首次验证: calib=0.740 (正确)

### 审核13: 策略连续插值是否消除跳变（P1-1, predictor.py）
- **改了什么**: 删除硬阈值 0.40/0.75。改为 `w_cons=clip((0.45-calib)/0.35,0,1)`, `w_aggr=clip((calib-0.55)/0.35,0,1)`, `w_bal=1-w_cons-w_aggr`
- **怎么查**: 日志 strategy 字段不应出现旧标签 "High Certainty" / "High Yield"
- **期望结果**: 只有 "Conservative-leaning" / "Balanced" / "Aggressive-leaning"。首次验证: 只出现 Balanced 和 Aggressive-leaning

### 审核14: 市场锚定动态权重是否生效（P1-2, predictor.py）
- **改了什么**: 删除 `market_anchored = current_rate × combined_factor`。改为 `conf_weight = {High:0.65, Medium:0.50, Low:0.35}[confidence]`，`adjusted = conf_weight * model_rate + (1-conf_weight) * current_rate`。current_rate 不再乘系数
- **怎么查**: 日志 conf_w 字段
- **期望结果**: High→0.65, Medium→0.50, Low→0.35。首次验证: 正确

### 审核15: 趋势因子按周期调整是否生效（P2-1, predictor.py）
- **改了什么**: 长周期(>=60d)用 rate_chg_1440(24h) weight=0.03; 中周期(20-59d)用 rate_chg_240(4h) weight=0.05; 短周期(<20d)用 rate_chg_60(1h) weight=0.08
- **怎么查**: 日志 trend_adj 字段。长周期应比短周期更小
- **期望结果**: 120d 的 trend_adj 绝对值应 < 1h周期的 trend_adj

### 审核16: 重训练阈值严重性分级（P2-2, retraining_scheduler.py）
- **改了什么**: 新增严重性分级: critical(<0.20 或 >0.85), warning(<0.30 或 >0.65)。旧版只有单一阈值 <0.30 或 >0.70
- **怎么查**:
  ```bash
  grep "单period成交率" log/ml_optimizer.log | tail -10
  ```
- **期望结果**: 日志中应出现 `[critical]` 或 `[warning]` 标签

### 审核17: 安全边界基于均线（P2-3, predictor.py）
- **改了什么**: bounds 基准从 `current_rate` 改为 `bound_base = 0.6*ma_1440 + 0.4*current_rate`
- **怎么查**: 日志 bounds 字段。当 current_rate 异常低时，bounds 不应过度收紧
- **期望结果**: fUSD-30d current=0.15 但 bounds=[4.70,7.23]（因为 ma_1440 >> current_rate）

### 审核18: 冷启动按周期差异化（P3-1, execution_features.py）
- **改了什么**: 冷启动默认值: period<=7→0.55, 8-30→0.50, >30→0.45（旧版统一0.6）
- **怎么查**: 对新出现的冷启动组合检查 exec_rate_7d 是否符合预期
- **影响**: 长周期冷启动默认更保守(0.45)，减少冷启动期间定价过低

### 审核19: 训练特征时间化（P2-4, data_processor.py）
- **改了什么**: 训练数据中 exec_rate_7d/exec_rate_30d 从"所有样本共享同一快照值"改为"每周采样滚动计算"
- **怎么查**: 下次重训练时观察日志是否有异常。特征处理时间可能略增
- **影响**: 模型能学习执行率的时间变化规律，而非只看当前值

### 量化验证（部署后 6-12 小时运行）

```bash
# 1. 检查诊断日志
grep "PREDICTION_DIAG" log/ml_optimizer.log | tail -28

# 2. 检查执行率变化趋势
sqlite3 data/lending_history.db "
SELECT currency, period,
       COUNT(*) as total,
       SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) as executed,
       ROUND(100.0 * SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) / COUNT(*), 1) as exec_rate
FROM virtual_orders
WHERE order_timestamp >= datetime('now', '-12 hours')
  AND status IN ('EXECUTED','FAILED')
GROUP BY currency, period
ORDER BY currency, period;"

# 3. 对比修改前后
# 目标方向（不是绝对值，关注趋势）:
# - 60/90d 100% → 应开始下降（利率被上调了+35%）
# - 30d fUSD 29.6% → 应开始上升（利率被下调了-22.5%）
# - fUST 2d 20.0% → 应开始上升（利率被下调了-18%）
# - fUST 120d 17.6% → 应开始上升（利率被下调了-30%）
# - 10-15d 80%+ → 应开始下降（利率被微上调+2~3%）

# 4. 检查系统无报错
grep "Error\|Exception\|Traceback" log/ml_optimizer.log | grep "$(date +%Y-%m-%d)" | tail -20

# 5. 检查 clipping_rate 是否维持低位
python3 -c "
import json
m = json.load(open('data/system_metrics.json'))
print(f\"clipping_rate: {m['clipping_metrics']['clipping_rate']}\")
print(f\"strategy_dist: {m['clipping_metrics']['strategy_distribution']}\")
"
```

### 如果效果不理想的调参方向

1. **60/90d 仍然100%**: `_get_unified_adjustment` 中 max_up 从 0.35 → 0.45，或市场锚定 conf_weight High 从 0.65 → 0.75（更信任模型上调）
2. **短周期过度下调**: max_down 从 0.18 → 0.12，或贝叶斯 evidence_weight 上限从 0.5 → 0.3
3. **clipping_rate 回升**: 检查 bound_base 计算，ma_1440 是否有异常值
4. **需要重训练**: P2-4 修改后首次重训练会使模型适应时间化特征，可能需要 `curl -X POST http://localhost:5000/retrain?force=true`

## 2025-02-25 v5.1 中周期调整力度优化（部署后24h审核+调参）

v5 部署24h后审核发现: 10-20d 周期执行率仍极高(85-97%)，因被归为"短周期" max_up=0.20 调整力度不足。2-7d 近48h急跌主要是市场因素。本次修改3项，仅改 `predictor.py`。

### 修改前基线数据（v5部署24h后，2026-02-25 采集）

| 问题类别 | 币种-周期 | 7日执行率 | 偏离目标 |
|---------|----------|----------|---------|
| 极高(>85%) | fUSD 15d | 96.6% | +46.6pp |
| | fUSD 10d | 93.5% | +43.5pp |
| | fUST 10d | 93.8% | +43.8pp |
| | fUST 14d | 92.6% | +42.6pp |
| | fUST 15d | 88.9% | +38.9pp |
| | fUSD 14d | 87.5% | +37.5pp |
| | fUSD 20d | 87.5% | +37.5pp |
| | fUST 7d | 85.4% | +35.4pp |
| 偏低(<40%) | fUST 2d | 20.0% | -30.0pp |
| | fUST 120d | 27.8% | -22.2pp |
| | fUSD 3d | 37.1% | -12.9pp |
| | fUSD 30d | 40.3% | -9.7pp |
| 接近目标 | fUSD 5d=46%, fUSD 6d=54%, fUST 3d=54%, fUST 20d=57% | ≈50% | OK |

系统指标: clipping_rate=7.4%, 策略分布: balanced 63%, conservative 26%, aggressive 11%

### 审核20: 4级周期分类是否生效（P0, predictor.py _get_unified_adjustment）
- **改了什么**: 原3级(长>=60/中>=20/短<20) → 4级: 长(>=60d)+35%/-30%, 中(>=20d)+28%/-24%, 近中(>=8d)+26%/-20%, 超短(<8d)+18%/-16%
- **为什么**: 10-15d 市场特性(48h验证窗口、较低波动性)更接近中周期，不应与2-4d共用超短参数
- **怎么查**:
  ```bash
  grep "PREDICTION_DIAG" log/ml_optimizer.log | tail -28
  ```
- **期望结果**:
  - fUSD 10d(exec≈93.5%): adj 从 ~1.18 → ~1.24 (+6pp)
  - fUSD 15d(exec≈96.6%): adj 从 ~1.19 → ~1.26 (+7pp)
  - fUST 2d(exec≈20%): adj 从 ~0.89 → ~0.90 (调整更温和)
  - fUSD 2d(exec≈41%): adj 几乎不变（偏离不大）

### 审核21: 极端偏离加速响应是否生效（P0, predictor.py _get_unified_adjustment）
- **改了什么**: norm_dev>0.8(exec>90%或exec<10%)时，从纯 power 0.7 改为混合 `0.5*(norm_dev^0.7) + 0.5*(norm_dev^0.5)`
- **为什么**: power 0.7 在高偏离区接近饱和，exec从90%→100%几乎不增加调整力度
- **怎么查**: 对比 exec_rate>90% 的组合adj值是否比修改前更大
- **期望结果**: exec=93.5%时 adj_factor 从 0.895→0.911 (+1.6pp)，仅影响极端情况

### 审核22: 市场锚定极端增强是否生效（P1, predictor.py）
- **改了什么**: 当 adjustment>1.15 或 <0.85 时，conf_weight 额外增加 `min(|adj-1.0|-0.15, 0.15)`，上限0.80
- **为什么**: adj=1.23时市场锚定以50%权重混合current_rate，实际调整被减半。极端纠偏时应更信任模型
- **怎么查**: PREDICTION_DIAG 日志中 `conf_w=` 字段
- **期望结果**:
  - fUSD 10d(adj≈1.24): conf_w 从0.50→0.59
  - fUSD 60d(adj=1.35): conf_w 从0.65→0.80(命中上限)
  - 接近目标的组合(adj≈1.0): conf_w 不变

### 量化验证（部署后48h运行）

```bash
# 1. 检查诊断日志（确认4级分类和conf_w变化）
grep "PREDICTION_DIAG" log/ml_optimizer.log | tail -28

# 2. 10-15d 执行率趋势（核心验证）
sqlite3 data/lending_history.db "
SELECT currency, period,
       COUNT(*) as total,
       SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) as executed,
       ROUND(100.0 * SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) / COUNT(*), 1) as exec_rate
FROM virtual_orders
WHERE order_timestamp >= datetime('now', '-48 hours')
  AND status IN ('EXECUTED','FAILED')
  AND period IN (10, 14, 15, 20)
GROUP BY currency, period ORDER BY currency, period;"

# 3. 2-7d 超短周期是否过调（应比修改前更温和）
sqlite3 data/lending_history.db "
SELECT currency, period,
       COUNT(*) as total,
       SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) as executed,
       ROUND(100.0 * SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) / COUNT(*), 1) as exec_rate
FROM virtual_orders
WHERE order_timestamp >= datetime('now', '-48 hours')
  AND status IN ('EXECUTED','FAILED')
  AND period IN (2, 3, 4, 5, 6, 7)
GROUP BY currency, period ORDER BY currency, period;"

# 4. 无报错
grep -E "Error|Exception|Traceback" log/ml_optimizer.log | tail -20
```

**目标执行率（v5.1修改后趋向）**:
- 10-15d: 85-97% → 下降趋向 65-80%（第一轮预计降5-10pp）
- 2-7d: 维持或略升（降低了max_down从0.18→0.16，减少过调）
- 20-30d/60-120d: 不受影响（参数未变）

**如果效果不足的调参方向**:
1. **10-15d 仍>85%**: 近中周期 max_up 从 0.26 → 0.30
2. **2-7d 继续下跌**: max_down 从 0.16 → 0.12
3. **conf_w boost 过强**: 上限从 0.80 → 0.70

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
