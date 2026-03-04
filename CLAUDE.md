每次执行命令的时候,请叫我达拉崩吧.

# Bitfinex Lending Optimize - 审核速查版

## 1) 项目简介
- 目标: 预测 fUSD/fUST 在 14 个周期(2~120天)的借贷利率,并通过虚拟订单闭环自优化。
- 核心流程: `预测 -> 下虚拟单 -> 验证执行 -> 回流训练 -> 自动重训/部署`。
- 技术栈: Python + FastAPI + XGBoost/LightGBM/CatBoost + SQLite。

## 2) 快速命令
```bash
# 启动 API
python ml_engine/api_server.py

# 触发完整闭环流程
curl -X POST http://localhost:5000/update

# 单步触发
curl -X POST http://localhost:5000/predict
curl -X POST http://localhost:5000/validate_orders
curl -X POST "http://localhost:5000/retrain?force=true"

# 状态/结果/订单
curl http://localhost:5000/status
curl http://localhost:5000/result
curl "http://localhost:5000/orders?status=PENDING&limit=20"

# 日志
rg -n "PREDICTION_DIAG|stale_data|Retraining|Scheduled pipeline" log/ml_optimizer.log
```

## 3) 关键文件职责
- `ml_engine/api_server.py`: API + 闭环编排 + 定时调度。
- `ml_engine/predictor.py`: 预测主逻辑、下虚拟单、策略诊断。
- `ml_engine/order_manager.py`: 虚拟订单建单/查询/统计。
- `ml_engine/execution_validator.py`: 订单执行验证与打分。
- `ml_engine/training_data_builder.py`: 市场数据+执行反馈融合。
- `ml_engine/model_trainer_v2.py`: 增强训练与样本权重。
- `ml_engine/retraining_scheduler.py`: 重训触发、模型对比、部署门禁。
- `ml_engine/system_policy.py`: 闭环策略与阈值加载。
- `config/system_policy.json`: 可追踪策略配置(推荐主配置源)。

## 4) 当前生产闭环规则(简版)

### 4.1 自动运行
- API 启动后: **立即执行1轮 pipeline**, 之后每2小时执行1轮。
- 并发保护: pipeline 有锁,避免重入。

### 4.2 预测侧
- 预测含诊断字段:
  - `market_follow_error`
  - `direction_match`
  - `step_change_pct`
  - `step_capped`
  - `policy_step_cap_pct`
- 120d 单步变化硬限幅: 默认 `5%`。
- 数据新鲜度硬门控:
  - `warn`: 60分钟
  - `hard`: 120分钟
  - 触发 hard 时拒绝本轮建单并将 `/result` 写为 `stale_data=true`。

### 4.3 下单与反馈采样
- 去重: `currency + period + order_timestamp + PENDING`。
- 新增 `probe_type`:
  - `normal`
  - `refresh_probe` (低权重补样)
- 当某组合在回看窗口内长期无新验证样本,且连续多轮被去重跳过后,自动插入 `refresh_probe`。

### 4.4 验证侧
- 写回结构化拒绝原因:
  - `SCORE_GATE / Q40_GATE / FILL_GATE / Q40_AND_FILL_GATE / NO_MARKET_DATA / NO_VALID_MARKET_DATA`
- 写回字段:
  - `execution_threshold`
  - `market_percentile_40`
  - `follow_error_at_order`

### 4.5 训练侧
- 执行反馈标签参与训练:
  - `actual_execution_binary`
  - `revenue_reward`
  - `follow_error`
- `refresh_probe` 样本在训练时降权(默认 0.3x),减少探针样本对主模型扰动。

### 4.6 重训触发与部署门禁
- 触发方式:
  - 定期触发: `>=7天` 且新增反馈足够。
  - 多信号触发分数: 执行率异常 + 分组异常 + 跟随误差 + 方向失配 + 120d稳定性。
- 部署门禁:
  - 新旧模型在同一验证切片做 champion/challenger 对比。
  - 同时必须通过闭环质量门禁(跟随误差/方向一致/120d稳定)。

## 5) 配置与状态文件
- 主策略配置: `config/system_policy.json`
- 兼容旧策略路径: `data/system_policy.json` (仅fallback)
- 重训冷却状态: `data/retraining_state.json`
- 补样状态计数: `data/refresh_probe_state.json`
- 最新结果: `data/optimal_combination.json`

## 6) API 输出关键字段
`GET /result` 至少应包含:
- `timestamp`
- `status`
- `policy_version`
- `stale_data`
- `stale_minutes`
- `stale_reason`
- `recommendations`

## 7) 排障清单(高频)

### A. 执行率长期异常但模型不动
1. 看 `/status` 是否有周期性 pipeline 更新。
2. 看 `retraining_scheduler --dry-run` 是否触发重训。
3. 查 `virtual_orders` 是否大量旧 `PENDING` 阻塞样本更新。
4. 查是否频繁 `Duplicate order skipped`。

### B. /result 一直 stale_data
1. 检查数据下载步骤是否成功。
2. 查 `funding_rates` 最新时间是否落后 >120分钟。
3. 检查服务时区与数据库时间格式。

### C. 120d 波动太大
1. 检查 `config/system_policy.json` 的 `step_caps_pct.120`。
2. 查 `p120_step_p95_7d` 是否超过阈值。

## 8) 本次更新记录 (2026-03-04)

### 8.1 闭环增强
- 新增策略系统与默认配置版本(`policy_version`)。
- 预测写入跟随/方向/步长诊断字段。
- 验证写入结构化 gate 拒绝原因与阈值字段。
- 训练融合新增反馈字段并排除泄漏特征。
- 监控新增 follow/stability 指标。

### 8.2 全自动化增强
- API 启动即跑首轮 pipeline, 后续2小时循环。
- 重训冷却状态持久化到磁盘, 重启不丢。
- 数据新鲜度硬门控(120分钟), stale 时停止建单。
- `refresh_probe` 自动补样机制上线, 训练低权重吸收。
- 重训触发新增多信号 `trigger_score`。
- 新旧模型改为同验证切片对比(Champion/Challenger)后再部署。

### 8.3 日志治理
- 已关闭 Uvicorn access log, 不再打印常规 `200` 访问日志。

## 9) 下次审核最小检查 (5分钟)
1. `GET /status` 看是否 online 且 last_update 持续变化。
2. `GET /result` 看 `policy_version/stale_data` 字段。
3. 查日志是否仍出现 `"GET /... 200"` 访问日志(应无)。
4. 查 `virtual_orders` 最近验证是否持续新增。
5. 查 `retraining_history.json` 最近事件是否有对比结果与部署结论。
