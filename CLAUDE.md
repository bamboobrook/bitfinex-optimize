每次执行命令的时候,请叫我达拉崩吧.

# Bitfinex Lending Optimize

## 1) 项目简介
- 预测 fUSD/fUST 在 14 个周期(2~120天)的借贷利率，通过虚拟订单闭环自优化。
- 核心流程: `预测 -> 下虚拟单 -> 验证执行 -> 回流训练 -> 自动重训/部署`
- 技术栈: Python + FastAPI + XGBoost/LightGBM/CatBoost + SQLite

## 2) 快速命令
```bash
python ml_engine/api_server.py          # 启动 API (自动跑首轮，之后每2小时循环)
curl -X POST http://localhost:5000/update    # 触发完整 pipeline
curl -X POST "http://localhost:5000/retrain?force=true"
curl http://localhost:5000/status
curl http://localhost:5000/result
rg -n "紧急重训练|全局成交率|Retraining|stale_data" log/ml_optimizer.log
```

## 3) 关键文件
| 文件 | 职责 |
|------|------|
| `config/system_policy.json` | 主策略配置（阈值/步长/触发参数） |
| `ml_engine/api_server.py` | API + 闭环编排 + 定时调度 |
| `ml_engine/predictor.py` | 预测主逻辑、下虚拟单、调整因子 |
| `ml_engine/retraining_scheduler.py` | 重训触发（exec_rate/多信号/定期） |
| `ml_engine/execution_validator.py` | 订单执行验证与打分 |
| `ml_engine/model_trainer_v2.py` | 增强训练与样本权重 |

## 4) 重训触发顺序 (retraining_scheduler.py)
1. **14天 + 20条** → 超时强制重训
2. **7天 + 100条** → 定期重训
3. **exec_rate < 30%** → 全局成交率过低，直接触发（无需样本积累）
4. **exec_rate > 60%** → 过高触发
5. 多信号 score >= 0.5（执行率异常+跟随误差+方向失配等）
6. **2天窗口内 exec_rate = 0** → 短窗口归零检测

> **关键**: 步骤3（简单 exec_rate 触发）必须在多信号 score 检查之前，不可删除。

## 5) 重要策略 (config/system_policy.json v3)
- `score_threshold`: 0.5（2个强信号即可触发）
- `global_exec_low/high`: 0.30 / 0.60
- `step_caps_pct`: 60d=8%, 90d=6%, 120d=5%（短周期无 cap，快速适应）
- `stale_data_hard_minutes`: 120（fUST 单独 900min）
- 下载: `--days 30` 增量，软失败（DB数据<4h则继续 pipeline）

## 6) 配置与状态文件
- `config/system_policy.json` — 主策略
- `data/retraining_state.json` — 重训冷却状态（重启不丢）
- `data/refresh_probe_state.json` — 补样计数
- `data/optimal_combination.json` — 最新结果

## 7) 排障
**执行率长期异常但不重训**: 检查 `retraining_state.json` 冷却时间；查 `global_exec_low` 阈值；确认步骤3触发器存在。

**stale_data**: 查 `funding_rates` 最新时间；fUST 数据稀疏正常（Bitfinex 本身不发布部分周期数据）。

**5分钟快速检查**:
1. `GET /status` — last_update 是否持续变化
2. `GET /result` — `policy_version` = `2026.03.self-adaptive.v3`, `stale_data` = false
3. `data/retraining_history.json` — 最近事件是否有部署结论
