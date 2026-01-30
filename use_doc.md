# 执行反馈优化系统 - 使用文档

## 一、系统概述

执行反馈系统是 Bitfinex 借贷利率预测系统的核心组件，通过模拟订单执行、验证预测准确性，并利用历史执行数据动态调整未来预测，实现预测质量的持续优化。

### 核心功能

1. **虚拟订单管理** - 为每个预测创建虚拟订单
2. **执行验证** - 根据实际市场数据验证订单是否能成交
3. **特征计算** - 基于执行历史计算 15 个反馈特征
4. **动态调整** - 根据执行率自动调整预测利率
5. **统计查询** - 提供 API 接口查询执行统计数据

### 预期效果

- **第 1 周**：执行率从 <50% 提升至 >65%
- **第 1 月**：执行率稳定在 >70%
- **长期**：预测利率更接近实际市场成交水平

---

## 二、系统架构

### 2.1 核心组件

```
┌─────────────────────────────────────────────────────┐
│  POST /update (每 2 小时触发一次)                    │
└────────────┬────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────┐
│ STEP 0: 订单验证                                   │
│  - 查询 PENDING 状态且超过验证窗口的订单           │
│  - 检查预测利率是否在市场中成交                    │
│  - 更新状态: EXECUTED/FAILED                       │
└────────────┬───────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────┐
│ STEP 1-2: 数据下载与处理                          │
│  - 下载最新 funding_rates 数据                     │
│  - 计算 15 个执行反馈特征:                         │
│    * 执行率: exec_rate_7d, exec_rate_30d           │
│    * 价差: avg_spread_7d, avg_spread_30d           │
│    * 失败订单利率差: avg_rate_gap_failed_7d        │
│    * 执行延迟: exec_delay_p50, exec_delay_p90      │
│    * 风险调整因子: risk_adjustment_factor          │
│    * 趋势指标: exec_rate_trend, rate_gap_trend     │
└────────────┬───────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────┐
│ STEP 3: 模型训练                                   │
│  - 使用 72 个特征训练 (57 基础 + 15 反馈)          │
└────────────┬───────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────┐
│ STEP 4: 生成预测                                   │
│  - 基础预测 (4 个模型集成)                        │
│  - 应用执行反馈调整:                               │
│    * 执行率 <50%: 降低 10%                         │
│    * 执行率 50-60%: 降低 5%                        │
│    * 执行率 60-70%: 降低 2%                        │
│    * 执行率 >85%: 提高 2%                          │
│  - 创建 Top 10 预测的虚拟订单                      │
└─────────────────────────────────────────────────────┘
```

### 2.2 文件结构

#### 新增文件 (4个)

1. **ml_engine/init_execution_tables.py** - 数据库初始化
2. **ml_engine/order_manager.py** - 虚拟订单管理
3. **ml_engine/execution_validator.py** - 订单验证引擎
4. **ml_engine/execution_features.py** - 特征计算器

#### 修改文件 (3个)

1. **ml_engine/data_processor.py** - 添加 15 个执行特征
2. **ml_engine/predictor.py** - 添加执行调整逻辑
3. **ml_engine/api_server.py** - 添加验证步骤和新接口

#### 测试文件 (1个)

1. **validate_system.py** - 系统验证脚本

---

## 三、数据库设计

### 3.1 virtual_orders 表

存储模拟订单记录：

```sql
CREATE TABLE virtual_orders (
    order_id TEXT PRIMARY KEY,              -- 订单ID (UUID)
    currency TEXT NOT NULL,                 -- 币种 (fUSD/fUST)
    period INTEGER NOT NULL,                -- 借贷期限 (2-120 天)
    predicted_rate REAL NOT NULL,           -- 预测利率 (%)
    order_timestamp TEXT NOT NULL,          -- 订单创建时间
    validation_window_hours INTEGER,        -- 验证窗口 (24/48/72 小时)
    status TEXT DEFAULT 'PENDING',          -- 状态: PENDING/EXECUTED/FAILED/EXPIRED

    -- 执行详情
    executed_at TEXT,                       -- 成交时间
    execution_rate REAL,                    -- 成交时的市场利率
    execution_delay_minutes INTEGER,        -- 成交延迟 (分钟)
    max_market_rate REAL,                   -- 验证窗口内最高市场利率
    rate_gap REAL,                          -- 利率差 (predicted - max_market)

    -- 元数据
    model_version TEXT,                     -- 模型版本
    prediction_confidence TEXT,             -- 置信度 (Low/Medium/High)
    prediction_strategy TEXT,               -- 策略名称
    created_at TIMESTAMP,                   -- 记录创建时间
    validated_at TIMESTAMP,                 -- 验证时间

    CHECK(status IN ('PENDING', 'EXECUTED', 'FAILED', 'EXPIRED'))
);
```

### 3.2 验证窗口规则

| 期限范围 (天) | 验证窗口 |
|--------------|---------|
| 2-7          | 24 小时 |
| 10-30        | 48 小时 |
| 60-120       | 72 小时 |

### 3.3 执行判断标准

订单标记为 **EXECUTED** 的条件：
- 验证窗口内**任意分钟**的市场利率 (high_annual) ≥ 预测利率

否则标记为 **FAILED**，并记录利率差值。

---

## 四、执行调整算法

### 4.1 基础调整逻辑

```python
# 根据 7 天执行率确定基础调整系数
if exec_rate_7d < 0.5:
    adjustment = 0.90      # 执行率太低，降低 10%
elif exec_rate_7d < 0.6:
    adjustment = 0.95      # 执行率偏低，降低 5%
elif exec_rate_7d < 0.7:
    adjustment = 0.98      # 执行率略低，降低 2%
elif exec_rate_7d > 0.85:
    adjustment = 1.02      # 执行率很高，提高 2%
else:
    adjustment = 1.0       # 执行率正常，不调整
```

### 4.2 利率差惩罚

如果失败订单的平均利率差过大，进一步降低预测：

```python
if avg_gap > 0:
    gap_penalty = min(avg_gap / base_rate, 0.08)  # 最多惩罚 8%
    adjustment *= (1.0 - gap_penalty)
```

### 4.3 趋势调整

根据执行率趋势进行微调：

```python
exec_trend = exec_rate_7d / exec_rate_30d

if exec_trend < 0.8:       # 执行率恶化
    adjustment *= 0.97
elif exec_trend > 1.2:     # 执行率改善
    adjustment *= 1.01
```

### 4.4 安全边界

最终调整系数限制在 **0.85 - 1.05** 之间，防止过度调整。

---

## 五、使用指南

### 5.1 初始化系统

```bash
# 1. 创建数据库表
python ml_engine/init_execution_tables.py

# 2. 验证系统
python validate_system.py
```

预期输出：

```
✅ ALL VALIDATION TESTS PASSED
Current Virtual Orders Status:
  PENDING: X
Market Data: 1,997,156 funding rate records
```

### 5.2 启动服务

```bash
# 启动 API 服务器
python ml_engine/api_server.py

# 在另一个终端触发更新
curl -X POST http://localhost:5000/update
```

### 5.3 API 接口

#### 现有接口

- `GET /status` - API 健康检查和流水线状态
- `GET /result` - 获取最新预测 (optimal_combination.json)
- `POST /update` - 触发完整流水线 (包含订单验证)

#### 新增接口

**查询执行统计**

```bash
# fUSD 30天期限，最近 7 天的执行统计
curl "http://localhost:5000/execution_stats?currency=fUSD&period=30&days=7" | jq

# 返回示例
{
  "total_orders": 45,
  "executed_orders": 32,
  "execution_rate": 0.711,
  "avg_execution_delay": 245,   # 平均成交延迟（分钟）
  "avg_spread": 0.85,            # 平均价差
  "avg_rate_gap": 1.23,          # 失败订单平均差距
  "has_sufficient_data": true
}
```

**查询虚拟订单**

```bash
# 查看所有待验证订单
curl "http://localhost:5000/orders?status=PENDING&limit=50" | jq

# 查看已成交订单
curl "http://localhost:5000/orders?status=EXECUTED&limit=50" | jq

# 查看失败订单
curl "http://localhost:5000/orders?status=FAILED&limit=50" | jq
```

### 5.4 日常运维

#### 检查系统状态

```bash
# 快速验证
python validate_system.py
```

#### 查看订单统计

```bash
sqlite3 data/lending_history.db "
SELECT
    status,
    COUNT(*) as count,
    ROUND(AVG(predicted_rate), 2) as avg_rate
FROM virtual_orders
GROUP BY status;"
```

#### 查看 7 天执行率

```bash
sqlite3 data/lending_history.db "
SELECT
    currency,
    period,
    COUNT(*) as total,
    SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) as executed,
    ROUND(100.0 * SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) / COUNT(*), 1) as exec_rate_pct
FROM virtual_orders
WHERE order_timestamp >= datetime('now', '-7 days')
  AND status IN ('EXECUTED', 'FAILED')
GROUP BY currency, period
ORDER BY exec_rate_pct DESC;"
```

#### 分析失败订单

```bash
sqlite3 data/lending_history.db "
SELECT
    currency,
    period,
    predicted_rate,
    max_market_rate,
    rate_gap,
    datetime(order_timestamp) as time
FROM virtual_orders
WHERE status = 'FAILED'
  AND rate_gap > 2.0
ORDER BY rate_gap DESC
LIMIT 10;"
```

---

## 六、监控指标

### 6.1 关键指标

| 指标名称 | 目标值 | 说明 |
|---------|-------|------|
| 7天执行率 | >75% | 最近 7 天成交订单占比 |
| 30天执行率 | >70% | 最近 30 天成交订单占比 |
| 平均价差 (已成交) | >0.5% | execution_rate - predicted_rate |
| 平均利率差 (失败) | <2% | predicted_rate - max_market_rate |
| 成交延迟中位数 | <12小时 | 从订单创建到成交的时间 |

### 6.2 每周回顾查询

```bash
sqlite3 data/lending_history.db "
SELECT
    'Last 7 Days' as period,
    COUNT(*) as orders,
    ROUND(100.0 * SUM(CASE WHEN status='EXECUTED' THEN 1 END) / COUNT(*), 1) as exec_pct,
    ROUND(AVG(CASE WHEN status='FAILED' THEN rate_gap END), 2) as avg_gap,
    ROUND(AVG(CASE WHEN status='EXECUTED' THEN execution_delay_minutes END)/60, 1) as avg_delay_hrs
FROM virtual_orders
WHERE order_timestamp >= datetime('now', '-7 days')
  AND status IN ('EXECUTED', 'FAILED')

UNION ALL

SELECT
    'Last 30 Days' as period,
    COUNT(*) as orders,
    ROUND(100.0 * SUM(CASE WHEN status='EXECUTED' THEN 1 END) / COUNT(*), 1) as exec_pct,
    ROUND(AVG(CASE WHEN status='FAILED' THEN rate_gap END), 2) as avg_gap,
    ROUND(AVG(CASE WHEN status='EXECUTED' THEN execution_delay_minutes END)/60, 1) as avg_delay_hrs
FROM virtual_orders
WHERE order_timestamp >= datetime('now', '-30 days')
  AND status IN ('EXECUTED', 'FAILED');"
```

---

## 七、故障排查

### 7.1 订单未创建

```bash
# 检查日志
grep "Created.*virtual orders" log/ml_optimizer.log | tail -5

# 手动测试订单创建
python -c "
from ml_engine.order_manager import OrderManager
from datetime import datetime
manager = OrderManager()
pred = {
    'currency': 'fUSD',
    'period': 30,
    'predicted_rate': 12.5,
    'confidence': 'High',
    'strategy': 'Test',
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}
print(manager.create_virtual_order(pred))
"
```

### 7.2 订单未验证

```bash
# 手动运行验证器
python ml_engine/execution_validator.py

# 检查验证日志
grep "validation" log/ml_optimizer.log | tail -10
```

### 7.3 执行率过低 (<50%)

这表示预测利率过于激进，系统会自动调整。也可以手动排查：

```bash
# 查找利率差大的失败订单
sqlite3 data/lending_history.db "
SELECT
    currency,
    period,
    predicted_rate,
    max_market_rate,
    rate_gap
FROM virtual_orders
WHERE status = 'FAILED'
  AND rate_gap > 2.0
ORDER BY rate_gap DESC
LIMIT 10;"
```

### 7.4 数据库错误

```bash
# 检查表是否存在
sqlite3 data/lending_history.db ".tables" | grep virtual

# 重建表（警告：会删除所有数据）
python ml_engine/init_execution_tables.py
```

### 7.5 特征计算错误

```bash
# 测试执行特征
python -c "
from ml_engine.execution_features import ExecutionFeatures
calc = ExecutionFeatures()
features = calc.get_all_features('fUSD', 30)
print('exec_rate_7d:', features['exec_rate_7d'])
print('risk_adjustment_factor:', features['risk_adjustment_factor'])
"
```

---

## 八、高级配置

### 8.1 调整执行率阈值

编辑 `ml_engine/predictor.py` (约第 245 行)：

```python
def _calculate_execution_adjustment(...):
    if exec_rate_7d < 0.5:
        adjustment = 0.90  # 修改此处
    elif exec_rate_7d < 0.6:
        adjustment = 0.95  # 修改此处
    # ... 其他阈值
```

### 8.2 调整验证窗口

编辑 `ml_engine/order_manager.py` (约第 20 行)：

```python
def determine_validation_window(period: int) -> int:
    if period <= 7:
        return 24  # 短期：24 小时
    elif period <= 30:
        return 48  # 中期：48 小时
    else:
        return 72  # 长期：72 小时
```

### 8.3 修改冷启动默认值

编辑 `ml_engine/execution_features.py` (约第 77 行)：

```python
if total < 10:  # 数据不足时的冷启动阈值
    exec_rate = 0.7  # 默认执行率 70%
```

---

## 九、数据维护

### 9.1 备份数据

```bash
# 备份虚拟订单
sqlite3 data/lending_history.db ".dump virtual_orders" > backup_orders_$(date +%Y%m%d).sql

# 备份执行统计
sqlite3 data/lending_history.db ".dump execution_statistics" > backup_stats_$(date +%Y%m%d).sql
```

### 9.2 归档旧数据

```bash
# 删除 90 天前的已完成订单
sqlite3 data/lending_history.db "
DELETE FROM virtual_orders
WHERE order_timestamp < datetime('now', '-90 days')
  AND status IN ('EXECUTED', 'FAILED', 'EXPIRED');"
```

### 9.3 清理测试数据

```bash
# 删除测试订单
sqlite3 data/lending_history.db "
DELETE FROM virtual_orders
WHERE model_version = 'test_v1.0'
  OR prediction_strategy = 'Test';"
```

---

## 十、应急操作

### 10.1 临时禁用执行反馈

如果系统出现问题，可以暂时禁用：

1. 停止 API 服务器：`pkill -f api_server.py`
2. 编辑 `ml_engine/predictor.py`，注释掉订单创建代码（约 384-397 行）
3. 重启 API 服务器

### 10.2 重置执行数据

```bash
# ⚠️ 警告：此操作会删除所有执行历史
sqlite3 data/lending_history.db "
DELETE FROM virtual_orders;
DELETE FROM execution_statistics;"
```

### 10.3 从备份恢复

```bash
sqlite3 data/lending_history.db < backup_orders_YYYYMMDD.sql
sqlite3 data/lending_history.db < backup_stats_YYYYMMDD.sql
```

---

## 十一、冷启动策略

系统初始运行时（历史订单 < 10 个），使用默认值：

| 特征 | 默认值 | 说明 |
|-----|-------|------|
| exec_rate_7d | 0.7 | 70% 执行率（乐观估计） |
| exec_rate_30d | 0.7 | 70% 执行率 |
| risk_adjustment_factor | 1.0 | 无风险调整 |
| avg_spread_* | 0.0 | 无价差数据 |
| avg_rate_gap_* | 0.0 | 无利率差数据 |
| exec_delay_* | 0.0 | 无延迟数据 |

这样可以保证系统在没有历史数据时也能正常工作，并随着数据积累逐步优化。

---

## 十二、成功标准

- ✅ 系统为预测创建虚拟订单
- ✅ 订单根据市场数据验证
- ✅ 计算并集成执行特征
- ✅ 基于执行历史调整预测
- ✅ API 接口提供执行统计
- ✅ 优雅处理冷启动
- ✅ 数据库架构支持所有操作
- ✅ 代码模块化且易维护

---

## 十三、技术支持

遇到问题时：

1. 查看日志：`tail -f log/ml_optimizer.log`
2. 运行验证：`python validate_system.py`
3. 检查文档：本文档或 git commit 记录

---

## 附录：15个执行反馈特征

| 特征名称 | 描述 | 冷启动默认值 |
|---------|------|------------|
| exec_rate_7d | 7天执行率 | 0.7 |
| exec_rate_30d | 30天执行率 | 0.7 |
| avg_spread_7d | 7天平均价差（已成交） | 0.0 |
| avg_spread_30d | 30天平均价差 | 0.0 |
| avg_rate_gap_failed_7d | 7天失败订单平均利率差 | 0.0 |
| avg_rate_gap_failed_30d | 30天失败订单平均利率差 | 0.0 |
| exec_delay_p50 | 成交延迟中位数（分钟） | 0.0 |
| exec_delay_p90 | 成交延迟90分位数 | 0.0 |
| market_competitiveness | 市场竞争度 (current_rate / ma_1440) | 1.0 |
| exec_likelihood_score | 执行概率综合评分 | 0.7 |
| risk_adjustment_factor | 动态风险调整因子 | 1.0 |
| exec_rate_trend | 执行率趋势 (7d/30d) | 1.0 |
| rate_gap_trend | 利率差趋势 | 0.0 |
| reserved_1 | 预留特征1 | 0.0 |
| reserved_2 | 预留特征2 | 0.0 |

---

**文档版本**: 1.0
**最后更新**: 2025-01

---

## 十四、API 接口完整指南

本系统提供完整的 RESTful API 接口，支持通过 HTTP 请求执行所有操作，无需手动运行脚本。

### 14.1 服务启动

```bash
# 启动 API 服务器（端口 5000）
python ml_engine/api_server.py

# API 将在以下地址可用
# http://localhost:5000
```

### 14.2 健康检查与状态

#### GET /status

查看 API 状态和当前任务进度。

**请求示例**:
```bash
curl http://localhost:5000/status | jq
```

**返回示例**:
```json
{
  "api_online": true,
  "service_info": {
    "status": "online",
    "current_step": "Idle",
    "last_update": "2025-01-30 14:30:00",
    "details": "Last update completed at 2025-01-30 14:30:00"
  }
}
```

**状态说明**:
- `online` - 服务在线且空闲，可接受新任务
- `processing` - 正在执行任务
- `error` - 上次任务出错

**使用场景**:
- 启动后验证服务是否正常
- 检查后台任务执行进度
- 确认服务是否空闲（执行新任务前）

---

### 14.3 数据获取

#### GET /result

获取最新的预测结果（`optimal_combination.json`）。

**请求示例**:
```bash
curl http://localhost:5000/result | jq
```

**返回示例**:
```json
{
  "timestamp": "2025-01-30 14:30:00",
  "optimal_combination": {
    "currency": "fUSD",
    "period": 30,
    "predicted_rate": 12.45,
    "confidence": "High",
    "strategy": "ensemble_v3"
  },
  "top_10_predictions": [...]
}
```

**使用场景**:
- 获取最新借贷利率预测
- 查看 Top 10 推荐组合
- 集成到交易系统或监控面板

---

#### GET /stats

获取虚拟订单统计信息。

**请求示例**:
```bash
curl http://localhost:5000/stats | jq
```

**返回示例**:
```json
{
  "status_summary": [
    {"status": "PENDING", "count": 15},
    {"status": "EXECUTED", "count": 342},
    {"status": "FAILED", "count": 158}
  ],
  "execution_rate_7d": [
    {
      "currency": "fUSD",
      "period": 30,
      "total": 45,
      "executed": 32,
      "exec_rate": 71.1
    }
  ],
  "latest_orders": [
    {
      "id": "a1b2c3d4",
      "combo": "fUSD-30d",
      "rate": "12.45%",
      "status": "EXECUTED",
      "created": "2025-01-30 14:00:00"
    }
  ]
}
```

**使用场景**:
- 查看订单状态分布
- 监控 7 天执行率
- 查看最新订单记录

---

#### GET /execution_stats

查询指定币种-周期的执行统计。

**参数**:
- `currency` (可选，默认 `fUSD`): 币种（`fUSD` 或 `fUST`）
- `period` (可选，默认 `30`): 借贷周期（2-120）
- `days` (可选，默认 `7`): 回溯天数

**请求示例**:
```bash
# fUSD 30天期限，最近 7 天的执行统计
curl "http://localhost:5000/execution_stats?currency=fUSD&period=30&days=7" | jq

# fUST 60天期限，最近 30 天的执行统计
curl "http://localhost:5000/execution_stats?currency=fUST&period=60&days=30" | jq
```

**返回示例**:
```json
{
  "currency": "fUSD",
  "period": 30,
  "lookback_days": 7,
  "statistics": {
    "total_orders": 45,
    "executed_orders": 32,
    "execution_rate": 0.711,
    "avg_execution_delay": 245,
    "avg_spread": 0.85,
    "avg_rate_gap": 1.23,
    "has_sufficient_data": true
  }
}
```

**使用场景**:
- 分析特定组合的执行表现
- 优化预测策略
- 生成性能报告

---

#### GET /orders

查询虚拟订单列表。

**参数**:
- `status` (可选): 过滤状态（`PENDING` / `EXECUTED` / `FAILED`）
- `limit` (可选，默认 `100`): 返回数量限制

**请求示例**:
```bash
# 查看所有待验证订单
curl "http://localhost:5000/orders?status=PENDING&limit=50" | jq

# 查看最近 100 个已成交订单
curl "http://localhost:5000/orders?status=EXECUTED&limit=100" | jq

# 查看所有订单（不过滤状态）
curl "http://localhost:5000/orders?limit=200" | jq
```

**返回示例**:
```json
{
  "total": 15,
  "filter_status": "PENDING",
  "orders": [
    {
      "order_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "currency": "fUSD",
      "period": 30,
      "predicted_rate": 12.45,
      "status": "PENDING",
      "order_timestamp": "2025-01-30 14:00:00",
      "validation_window_hours": 48
    }
  ]
}
```

**使用场景**:
- 监控待验证订单数量
- 审查失败订单
- 导出订单数据用于分析

---

### 14.4 任务触发

#### POST /update

触发完整流水线（推荐使用，每 2 小时运行一次）。

**流程**:
1. 验证虚拟订单（检查 PENDING 订单是否成交）
2. 下载最新历史数据
3. 处理特征（包含 15 个执行反馈特征）
4. 训练模型（使用最新数据）
5. 生成预测（创建 Top 10 虚拟订单）

**请求示例**:
```bash
curl -X POST http://localhost:5000/update
```

**返回示例**:
```json
{
  "status": "accepted",
  "message": "Full update pipeline started. Check /status for progress."
}
```

**监控进度**:
```bash
# 持续监控任务状态
watch -n 5 'curl -s http://localhost:5000/status | jq'
```

**使用场景**:
- 定时更新（推荐每 2 小时）
- 确保系统使用最新数据
- 完整的端到端流程

---

#### POST /download_data

仅下载历史数据（不训练模型）。

**请求示例**:
```bash
curl -X POST http://localhost:5000/download_data
```

**返回示例**:
```json
{
  "status": "accepted",
  "message": "Data download task started. Check /status for progress."
}
```

**使用场景**:
- 手动补充缺失数据
- 数据下载失败后重试
- 不需要重新训练模型时

---

#### POST /process_features

仅处理特征（不下载数据或训练模型）。

**请求示例**:
```bash
curl -X POST http://localhost:5000/process_features
```

**返回示例**:
```json
{
  "status": "accepted",
  "message": "Feature processing task started. Check /status for progress."
}
```

**使用场景**:
- 特征处理失败后重试
- 调整特征计算逻辑后重新处理
- 测试新的特征工程

---

#### POST /train

仅训练模型（不下载数据或生成预测）。

**请求示例**:
```bash
curl -X POST http://localhost:5000/train
```

**返回示例**:
```json
{
  "status": "accepted",
  "message": "Model training task started. Check /status for progress."
}
```

**使用场景**:
- 调整模型参数后重新训练
- 训练失败后重试
- 测试新的模型架构

---

#### POST /predict

仅生成预测（不训练模型）。

**请求示例**:
```bash
curl -X POST http://localhost:5000/predict
```

**返回示例**:
```json
{
  "status": "accepted",
  "message": "Prediction generation task started. Check /status for progress."
}
```

**使用场景**:
- 使用现有模型生成新预测
- 预测失败后重试
- 快速获取预测结果（不重新训练）

---

#### POST /validate_orders

手动触发虚拟订单验证。

**请求示例**:
```bash
curl -X POST http://localhost:5000/validate_orders
```

**返回示例**:
```json
{
  "status": "accepted",
  "message": "Order validation task started. Check /status for progress."
}
```

**使用场景**:
- 立即验证 PENDING 订单
- 测试验证逻辑
- 在完整流水线之外单独运行验证

---

### 14.5 系统验证

#### GET /validate

运行完整系统验证（5 个测试）。

**测试内容**:
1. **时间戳正确性** - 验证订单时间戳在创建时间 5 分钟内
2. **验证窗口合规** - 验证订单在窗口结束后才验证（防止 look-ahead bias）
3. **采样覆盖率** - 验证 7 天内覆盖 >80% 的组合
4. **执行率真实性** - 检查执行率 <90%（过高可能有 look-ahead bias）
5. **冷启动检测** - 验证冷启动组合（<10 订单）是否被测试

**请求示例**:
```bash
curl http://localhost:5000/validate | jq
```

**返回示例**:
```json
{
  "summary": {
    "passed": 3,
    "failed": 0,
    "warnings": 2,
    "overall_status": "PASS"
  },
  "tests": {
    "timestamp_correctness": {
      "status": "PASS",
      "message": "All order timestamps are within 5 minutes of creation time",
      "anomaly_count": 0,
      "anomalies": []
    },
    "validation_window": {
      "status": "PASS",
      "message": "All validations occurred after window end",
      "early_validation_count": 0,
      "early_validations": []
    },
    "sampling_coverage": {
      "status": "WARNING",
      "message": "Low sampling coverage (75.0%), expect improvement over time",
      "tested_combinations": 21,
      "total_combinations": 28,
      "coverage_pct": 75.0,
      "coverage_details": [...]
    },
    "execution_rate": {
      "status": "PASS",
      "message": "Execution rate is realistic (65.5%)",
      "total_7d": 45,
      "executed_7d": 32,
      "exec_rate_7d": 71.1,
      "total_30d": 180,
      "executed_30d": 118,
      "exec_rate_30d": 65.5
    },
    "cold_start_detection": {
      "status": "WARNING",
      "message": "Some cold start combinations not tested recently",
      "cold_start_count": 7,
      "recent_tested_count": 3,
      "cold_start_combos": [...]
    }
  }
}
```

**状态说明**:
- `PASS` - 测试通过
- `FAIL` - 测试失败（需要修复）
- `WARNING` - 警告（可能正常，观察即可）
- `overall_status`:
  - `PASS` - 无失败，至少 3 个通过
  - `FAIL` - 有测试失败
  - `WARNING` - 无失败但通过数 <3

**使用场景**:
- 系统初始化后验证
- 定期健康检查（每周）
- 发现异常后诊断问题

---

#### GET /validate/{test_name}

运行单个验证测试。

**可用测试名称**:
- `timestamp_correctness` - 时间戳正确性
- `validation_window` - 验证窗口合规性
- `sampling_coverage` - 采样覆盖率
- `execution_rate` - 执行率真实性
- `cold_start_detection` - 冷启动检测

**请求示例**:
```bash
# 只检查时间戳正确性
curl http://localhost:5000/validate/timestamp_correctness | jq

# 只检查执行率
curl http://localhost:5000/validate/execution_rate | jq

# 只检查采样覆盖率
curl http://localhost:5000/validate/sampling_coverage | jq
```

**返回示例**:
```json
{
  "test_name": "execution_rate",
  "result": {
    "status": "PASS",
    "message": "Execution rate is realistic (65.5%)",
    "total_7d": 45,
    "executed_7d": 32,
    "exec_rate_7d": 71.1,
    "total_30d": 180,
    "executed_30d": 118,
    "exec_rate_30d": 65.5
  }
}
```

**使用场景**:
- 针对性检查特定问题
- 开发/调试特定功能
- 减少响应时间（不运行所有测试）

---

### 14.6 完整工作流示例

#### 场景 1: 系统初始化

```bash
# 1. 启动 API 服务器
python ml_engine/api_server.py

# 2. 检查服务状态
curl http://localhost:5000/status | jq

# 3. 运行系统验证
curl http://localhost:5000/validate | jq

# 4. 触发完整流水线
curl -X POST http://localhost:5000/update

# 5. 监控进度（每 5 秒刷新）
watch -n 5 'curl -s http://localhost:5000/status | jq'

# 6. 获取预测结果
curl http://localhost:5000/result | jq
```

---

#### 场景 2: 定期更新（每 2 小时）

```bash
# 使用 crontab 定时触发
# 编辑 crontab: crontab -e
# 添加以下行（每 2 小时运行）:
0 */2 * * * curl -X POST http://localhost:5000/update

# 或使用 while 循环手动调度
while true; do
    echo "[$(date)] Triggering update..."
    curl -X POST http://localhost:5000/update
    sleep 7200  # 2 小时
done
```

---

#### 场景 3: 故障排查

```bash
# 1. 查看服务状态
curl http://localhost:5000/status | jq

# 2. 运行完整验证
curl http://localhost:5000/validate | jq

# 3. 查看订单统计
curl http://localhost:5000/stats | jq

# 4. 检查失败订单
curl "http://localhost:5000/orders?status=FAILED&limit=50" | jq

# 5. 查看特定组合的执行统计
curl "http://localhost:5000/execution_stats?currency=fUSD&period=30&days=7" | jq

# 6. 如果发现问题，单独重试失败的步骤
curl -X POST http://localhost:5000/validate_orders  # 重新验证订单
curl -X POST http://localhost:5000/train           # 重新训练模型
curl -X POST http://localhost:5000/predict         # 重新生成预测
```

---

#### 场景 4: 性能监控

```bash
# 创建监控脚本: monitor.sh
#!/bin/bash

echo "=== System Status ==="
curl -s http://localhost:5000/status | jq -r '.service_info | "Status: \(.status)\nStep: \(.current_step)\nLast Update: \(.last_update)"'

echo -e "\n=== Order Statistics ==="
curl -s http://localhost:5000/stats | jq -r '.status_summary[] | "\(.status): \(.count)"'

echo -e "\n=== 7-Day Execution Rates ==="
curl -s http://localhost:5000/stats | jq -r '.execution_rate_7d[] | "\(.currency)-\(.period)d: \(.exec_rate)% (\(.executed)/\(.total))"'

echo -e "\n=== Validation Tests ==="
curl -s http://localhost:5000/validate | jq -r '.tests | to_entries[] | "\(.key): \(.value.status)"'

# 运行监控
chmod +x monitor.sh
./monitor.sh
```

---

### 14.7 错误处理

#### 错误码说明

- **404** - 资源不存在（如预测结果未生成、测试名称错误）
- **409** - 冲突（服务正在执行其他任务）
- **500** - 服务器内部错误（数据库连接失败、模块执行错误）

#### 常见错误示例

**错误 1: 任务冲突**
```json
{
  "status": "busy",
  "message": "Another task is running: Training Models"
}
```

**解决**: 等待当前任务完成，通过 `/status` 监控进度。

---

**错误 2: 预测结果不存在**
```json
{
  "error": "No predictions found.",
  "suggestion": "Please call /update to generate data."
}
```

**解决**: 执行 `POST /update` 生成预测结果。

---

**错误 3: 数据库未找到**
```json
{
  "error": "Database not found: /path/to/lending_history.db"
}
```

**解决**: 运行 `python ml_engine/init_execution_tables.py` 初始化数据库。

---

### 14.8 API 接口总结表

| 方法 | 路径 | 功能 | 是否后台执行 |
|------|------|------|------------|
| GET | /status | 查看 API 状态和任务进度 | ❌ |
| GET | /result | 获取最新预测结果 | ❌ |
| GET | /stats | 获取订单统计信息 | ❌ |
| GET | /execution_stats | 查询执行统计（指定组合） | ❌ |
| GET | /orders | 查询虚拟订单列表 | ❌ |
| POST | /update | 完整流水线（推荐） | ✅ |
| POST | /download_data | 仅下载数据 | ✅ |
| POST | /process_features | 仅处理特征 | ✅ |
| POST | /train | 仅训练模型 | ✅ |
| POST | /predict | 仅生成预测 | ✅ |
| POST | /validate_orders | 仅验证订单 | ✅ |
| GET | /validate | 运行所有验证测试 | ❌ |
| GET | /validate/{test_name} | 运行单个验证测试 | ❌ |

---

### 14.9 最佳实践

1. **定期更新**: 使用 `POST /update` 每 2 小时更新一次
2. **监控状态**: 定期检查 `/status` 和 `/validate`
3. **日志审查**: 查看 `log/ml_optimizer.log` 了解详细执行日志
4. **错误恢复**: 任务失败时，使用单独的端点重试失败步骤
5. **性能优化**: 根据 `/execution_stats` 调整预测策略
6. **数据备份**: 定期备份 `data/lending_history.db`

---

