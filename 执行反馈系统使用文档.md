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
