# 2026-03-25 优化记录

## 背景

基于本地真实市场数据 review，当前系统存在 4 个核心问题:

1. `funding_rates` 下载步骤会把旧 candle / 重复 candle 也记成“成功”，导致部分组合长期 stale 但流水线仍显示下载成功。
2. `execution_probability` 仍然主要靠统一公式校准，`fUSD` 中长周期在执行率塌陷时仍可能维持偏高概率。
3. 推荐排序对 `fUSD-60d/120d/7d` 这类高 gap / 低执行率组合降权不够，局部退化会继续透传到榜单前排。
4. `prediction_history` 没有持续落库，导致优化前后无法稳定追踪每轮快照。

## 本轮优化计划

### P1. 修数据新鲜度

- 修正下载器成功判定:
  下载后必须检查 `latest timestamp` 是否前进，且最新数据年龄是否达到 fresh 阈值。
- 增加自适应回补:
  主刷新窗口无效时，自动扩展到 `72h -> 7d -> 30d`。
- 对仍旧 stale 的组合返回失败，而不是“假成功”。

### P2. 分币种 / 分周期校准

- 在预测器里新增 `currency + tier(short/medium/long)` 校准画像。
- `fUSD` 尤其是中长周期，更快信任真实执行反馈，降低高 gap 场景的模型乐观偏差。
- 对长期无成交组合增加概率上限，避免继续给出虚高执行概率。

### P3. 重新校准 execution probability

- 由单一 `model_prob + exec_rate_fast`，升级为:
  `model_prob + fast_exec + slow_exec + prior` 的分层混合。
- 新增两层保护:
  - `avg_rate_gap/current_rate` 过高时降权
  - 最终报价相对当前市场溢价过大时，再次压低 calibrated probability

### P4. 补齐可观测性 / 回测闭环

- 恢复 `prediction_history` 每轮快照落库。
- 保存 `weighted_score / calibrated_execution_probability / liquidity_score / data_age_minutes / market_follow_error / stale_data`。
- 新增本地评估脚本，一键输出:
  - 最近 7 天 vs 前 7 天整体执行率
  - 按币种表现
  - 改善/退化最大的组合
  - 当前 stale 组合
  - `prediction_history` 落库状态

## 已完成改动

### 1. 下载器

文件: `funding_history_downloader.py`

- 新增 `get_latest_timestamp / get_latest_age_minutes / freshness_target_minutes`
- 下载后不再只看“处理了多少条”，而是看:
  - 最新时间戳是否推进
  - 最新数据年龄是否达到阈值
- 主刷新失败后自动扩大 lookback
- `download_multiple()` 现在会把未修复的 stale 组合当成失败返回
- `main()` 在仍有 stale/failed 组合时以非 0 退出，避免 API 误判为完全成功

### 2. 预测器校准

文件: `ml_engine/predictor.py`

- 新增 `_get_probability_calibration_profile`
- 新增 `_calibrate_execution_probability`
- 新增 `_apply_probability_divergence_guard`
- 新增 `_get_recommendation_regime_multiplier`
- `fUSD` 中长周期在:
  - 低执行率
  - 高 `avg_rate_gap`
  - 长时间无成交
  - 最终报价偏离当前市场过大
  时会被更强地下调概率和排序权重

### 3. prediction_history 恢复

文件: `ml_engine/predictor.py`

- 新增 `_ensure_prediction_history_schema`
- 新增 `_persist_prediction_history`
- 每轮会保存完整排序快照，而不是只依赖结果文件和日志

### 4. 本地评估脚本

文件: `scripts/evaluate_recent_optimization.py`

用法:

```bash
python scripts/evaluate_recent_optimization.py
python scripts/evaluate_recent_optimization.py --days 7 --min-combo-orders 5
```

## 验证建议

### 代码级验证

```bash
python -m py_compile funding_history_downloader.py ml_engine/predictor.py scripts/evaluate_recent_optimization.py
python scripts/evaluate_recent_optimization.py
```

### 服务级验证

1. 手动重启 API 服务
2. 观察首轮 pipeline
3. 检查:
   - `/status`
   - `/result`
   - `log/ml_optimizer.log`
4. 重点确认:
   - `prediction_history` 是否继续增长
   - `stale_data` 是否下降
   - `fUSD-60d/120d/7d` 是否不再高位误排

## 重点观察项

- 如果下载器仍反复报某些组合 stale:
  优先查 Bitfinex 该周期是否真实缺单，而不是继续相信“下载成功”日志。
- 如果 `fUSD` 执行率仍明显落后:
  下一轮优先看 `model_trainer_v2.py` 的目标定义和样本权重，而不是继续只改排序层。
