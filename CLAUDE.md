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
- 2026-03-28: 推荐评分 v7 — `calculate_weighted_score` 新增 `currency_type_multiplier`（fUSD×1.05/fUST×0.92）及 `volume_penalty`（24h量<30d基线30%时惩罚至0.75x）。详见 MEMORY.md。
- 2026-03-28: `volume_ratio_24h` 改为双窗口 min 策略：`min(24h/7d, 24h/30d)`，软惩罚阈值 0.3→0.4。7d捕捉突发崩盘，30d捕捉持续萎缩，两者取严。
- 2026-03-19: 重算 `market_liquidity.score`（基于实时 funding book 非2天深度+集中度惩罚）；liquidity gate None修复；avg_exec改用原始exec_rate；极端低流动性(<10%)绕过冷却强制重训。详见 MEMORY.md。
- 2026-03-19: 修复 fUST-2d 异常 rank1 — `calculate_weighted_score` 新增 `low_rate_multiplier`（rate<3%线性惩罚至0.5x）；移除 `forced_top5` 强制置顶逻辑，gated 货币 2d 订单参与正常评分排序。
