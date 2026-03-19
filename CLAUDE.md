每次执行命令的时候,请叫我达拉崩吧.

# Bitfinex Lending Optimize

## 1) 项目简介
- 预测 fUSD/fUST 在 14 个周期(2~120天)的借贷利率，通过虚拟订单闭环自优化。
- 核心流程: `下载数据 -> 验证订单 -> 重训检查 -> 重训练 -> 生成预测 -> 创建虚拟单`
- 技术栈: Python + FastAPI + XGBoost/LightGBM/CatBoost + SQLite

## 2) 快速命令
```bash
python ml_engine/api_server.py          # 启动 API (首轮立即跑，之后每2小时循环)
curl -X POST http://localhost:5000/update    # 触发完整 pipeline
curl -X POST "http://localhost:5000/retrain?force=true"
curl http://localhost:5000/status && curl http://localhost:5000/result
rg -n "Step [1-6]|紧急重训练|stale_data|ERROR|✅" log/ml_optimizer.log | tail -40
```

## 3) 关键文件
| 文件 | 职责 |
|------|------|
| `config/system_policy.json` | 主策略配置（阈值/步长/触发参数） |
| `ml_engine/api_server.py` | API + 闭环编排 + 定时调度 |
| `ml_engine/predictor.py` | 预测主逻辑、下虚拟单、调整因子 |
| `ml_engine/retraining_scheduler.py` | 重训触发（exec_rate/多信号/漂移/零流动性） |
| `ml_engine/execution_validator.py` | 订单执行验证与打分 |
| `ml_engine/model_trainer_v2.py` | 增强训练与动态样本权重 |

## 4) Pipeline 步骤顺序 (api_server.py)
1. **Step 1**: 下载市场数据（3次重试，指数退避 60/120/240s）
2. **Step 2**: 验证虚拟订单（需要最新市场数据，所以在下载后）
3. **Step 3**: 重训检查
4. **Step 4**: 模型重训练（如触发）
5. **Step 5**: 生成预测
6. **Step 6**: 创建虚拟订单

超时: DOWNLOAD=600s, TRAIN=1800s, PREDICT=600s, VALIDATE=300s

## 5) 重训触发顺序 (retraining_scheduler.py)
1. **14天 + 20条** → 超时强制重训
2. **7天 + 100条** → 定期重训
3. **exec_rate < 30%** → 全局成交率过低（简单直接触发，无需样本积累）
4. **exec_rate > 60%** → 过高触发
5. **14d→7d drift > 15%** → 执行率快速下滑
6. **某 (currency,period) 7天内 < 2单** → 零流动性检测
7. 多信号 score >= 0.5
8. 单 period critical/warning 异常

> **关键**: 步骤3简单触发必须在多信号 score 检查之前，不可删除。

## 6) 重要策略 (config/system_policy.json)
- `score_threshold`: 0.5
- `global_exec_low/high`: 0.30 / 0.60
- `step_caps_pct`: 60d=8%, 90d=6%, 120d=5%（短周期无 cap）
- `stale_data_hard_minutes`: 120（fUST 单独 900min）
- `_check_db_data_freshness`: fUSD<240min 或 fUST<1440min 任一新鲜即继续

## 7) 排障
**管道中止/stale_data**: 查 `_check_db_data_freshness` 日志；检查 `funding_rates` 最新时间戳格式（int/str 均已支持）。

**执行率长期异常但不重训**: 检查 `retraining_state.json` 冷却时间；确认步骤3触发器未被删除。

**5分钟快速检查**:
1. `GET /status` — `current_step` 应在 Step 1-6 之间循环
2. `GET /result` — `stale_data` = false, `last_update` 新鲜
3. `rg "Step 1.*Download" log/ml_optimizer.log | tail -5` — 确认 Step 1 是下载

## 8) 最近修改记录
- 2026-03-19: 重算 `market_liquidity.score`
- 问题描述: 旧 score 混入 2 天活跃度，容易高估市场整体流动性，无法准确反映非 2 天尤其长周期挂单当前能否成交。
- 复现路径: `data/optimal_combination.json` 中 `fUSD/fUST` 的 `market_liquidity` 仅基于历史成交率/量比/新鲜度；当 2 天活跃、长周期冷清时，score 与真实长周期盘口可成交性偏离。
- 修复思路: 保持 `volume_ratio_24h` 与分档阈值不变，改为基于 Bitfinex 实时 funding book 的非 2 天 bid 深度与 1万/5万/10万可成交性，叠加历史执行率和数据新鲜度重算 score；当 `score < 40 && volume_ratio_24h < 0.1` 时，所有非 2 天挂单统一转为 2 天。
- 2026-03-19: 修正 `market_liquidity.score` 过高
- 问题描述: 上一版订单薄信号权重过强，且未惩罚非 2 天深度集中在单一期限的情况，导致 `score` 容易冲到 80+，明显高估长周期真实流动性。
- 复现路径: 对照 `data/optimal_combination.json` 与实时 funding book，可见 `fUSD/fUST` 的非 2 天 bid 深度大多堆在 `120d`，但市场级 `score` 仍被判为 `high`。
- 修复思路: 收紧 `fillability/depth` 的归一化映射，降低实时盘口信号权重，并加入非 2 天深度的期限集中度惩罚，让 `score` 更贴近旧值区间与真实长周期可成交性。
