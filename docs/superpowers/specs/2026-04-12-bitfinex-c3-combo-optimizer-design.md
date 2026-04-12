# Bitfinex Lending C3 Combo Optimizer Design

## 背景

当前系统已经有预测、排序、虚拟单验证、重训练闭环，但线上暴露出的核心问题不是“输出格式不对”，而是内部优化目标与真实业务路径不一致：

- 模型经常直接给出偏离市场较远的高利率
- 长时间挂单后仍无法成交
- 排序更像“高名义利率 + 长周期 + 手工偏置”，不是真正的组合级最优
- 验证只真实覆盖前 `6h` 固定价阶段，后续 `FRR` / `rank6` 主要是代理值
- 训练样本把历史订单字段、缺行情样本、旧 schema 样本混进来，导致闭环会越学越偏

用户当前明确目标是：

`收益优先 > 成交 > 长周期 > fUSD优先`

同时，用户要求：

- 外部输出格式不变
- 仍然输出 `rank1-5`
- `rank6` 固定为 `fUSD-2d`
- 外部机器人无需改动

## 目标

### 主目标

把当前内部逻辑从“模型直接报最终利率的启发式排序器”，升级为：

`市场锚定候选 + 组合搜索 + 路径闭环验证`

系统内部真正优化的是：

`在 rank1=60%、rank2-5=各10%、固定价 -> FRR -> rank6 的真实规则下，rank1-5 组合的最终路径价值`

### 次目标

- 显著减少偏离最近真实成交区间的假高利率
- 让推荐随最近几小时市场成交/close 区间动态变化
- 让 `fUSD` 默认优先，但不是绝对锁死
- 让 `fUST` 只有在“收益明显更高，且成交不能明显更差”时，才允许排到更前
- 让训练真正学习路径终值与等待成本，而不是历史排序残影

## 非目标

- 不改变 `/result` 的 JSON 结构
- 不改变 `rank6 = fUSD-2d`
- 不改外部机器人资金分配逻辑
- 不新增独立服务
- 不把当前系统拆成多服务架构

## 方案选择

本轮采用 `C3 混合式组合优化`，不采用：

- `A` 止血修补：只能缓解脏样本/stale/部署问题，不能从根上解决“模型直报利率失真”
- `B` 单候选路径价值排序：比当前强，但仍主要是“单候选评分后排序”，不是组合级最优

`C3` 的核心思想是：

- 模型不再直接决定最终挂单利率
- 最终挂单利率来自最近几小时真实成交/close 区间生成的候选集合
- 组合搜索器在候选集合上直接优化 `rank1-5` 的整体路径价值

## 总体设计

### 外部接口

外部接口保持不变：

- `rank1-5` 继续输出 5 个推荐位
- `rank6` 固定为 `fUSD-2d`
- `/result` 结构保持兼容

### 内部职责重排

系统内部拆成 4 个逻辑层，但尽量仍落在现有模块中：

1. `Market-Anchored Candidate Generator`
   为每个 `(currency, period)` 生成市场锚定候选利率

2. `Path Evaluator`
   估计每个候选的路径终值、等待成本、fallback 风险

3. `Combo Searcher`
   在候选集合上搜索 `rank1-5` 最优组合

4. `Path Validation + Training Loop`
   按 exploit 组合真实回放，回写路径标签并训练路径侧模型

## 业务规则

### 优先级

组合比较严格按以下顺序：

1. `组合最终收益 EV`
2. `组合成交/等待质量`
3. `长周期质量`
4. `fUSD 默认优先`

### 市场锚点

最终利率的市场锚点采用：

`最近几小时真实成交 / close 区间`

窗口动态切换：

- `2d-7d`：`2h / 6h / 12h`
- `10d-30d`：`6h / 12h / 24h`
- `60d-120d`：`12h / 24h / 48h`

### 重复规则

- 禁止完全重复的 `(currency, period)`
- 允许同币种不同周期同时出现在 `rank1-5`

### 等待成本

等待时间采用 `中处罚`：

- 不会因为多等一点就直接否掉高收益机会
- 但会明显压低长期占资且最终收益兑现不稳的组合

### fUST 上位规则

`fUST` 只有在满足以下条件时，才允许压过更优先的 `fUSD`：

- 组合收益分明显更高
- 成交概率不能明显更差

这不是简单乘数，而是组合比较时的门槛规则。

## 候选利率生成

### 原则

每个 `(currency, period)` 不再只输出 1 个最终利率，而是输出少量候选利率。

目标是：

- 保留收益空间
- 防止偏离市场太远
- 为组合搜索提供可比较的离散决策点

### 候选来源

每个动态窗口提取：

- 真实成交中位数/分位数
- close 中位数/分位数
- 近期波动宽度

再合成稳健区间：

- `anchor_low`
- `anchor_mid`
- `anchor_high`

### 候选集合

每个 `(currency, period)` 默认生成以下候选带：

- `safe_fill`
- `balanced_low`
- `balanced_mid`
- `premium`
- `stretch_premium`

当长周期窗口本身足够强时，再额外生成：

- `long_premium`

### 偏离约束

采用“软约束 + 硬外边界”：

- 候选仍允许高于市场锚点区间
- 但越偏离市场，后续组合评分惩罚越重
- 同时设置硬外边界，禁止离谱报价

### fUST 特殊处理

`fUST` 的高价候选更保守：

- 高价候选数量更少
- 外边界更紧
- 只有最近窗口确实出现更强收益带，才允许放出更激进候选

## 路径估值

### 单候选路径

每个候选都按真实业务路径估值：

1. `0-6h` 固定利率阶段，每小时降 `1%`
2. `6-12h` fallback 到 `FRR-120d`
3. `12h+` fallback 到 `rank6=fUSD-2d`

### 单候选核心指标

每个候选至少计算：

- `stage1_fill_prob`
- `expected_wait_hours`
- `fallback_stage2_prob`
- `fallback_rank6_prob`
- `final_path_terminal_value`
- `market_distance_penalty`
- `candidate_path_ev`

### 约束思想

收益是主目标，但不会允许以下情况轻易上位：

- 偏离最近真实成交区间太远
- 长时间占资
- 过高概率掉进 `rank6`

## 组合评分

### 真实资金权重

组合总分直接按外部真实资金权重计算：

- `rank1 = 60%`
- `rank2-5 = 各10%`

### 主分

组合主分定义为：

`combo_revenue_ev = Σ(rank_weight × candidate_path_ev)`

这是第一优先级。

### 二级约束

当两个组合收益接近时，再比较：

- `combo_fill_quality`
- `combo_wait_cost`

其中等待成本使用中处罚。

### 三级偏好

在前两层无法拉开明显差距时，再比较：

- 长周期质量
- `fUSD` 默认优先

### 组合级惩罚

搜索与比较过程中加入：

- 过度集中同类风险惩罚
- fallback 到 `rank6` 概率过高惩罚
- 整体等待时间过长惩罚

## 组合搜索

### 搜索策略

不做全量穷举，采用：

`候选压缩 + Beam Search`

### 候选压缩

每个 `(currency, period)` 最多保留 `2-3` 个候选：

- 一个偏稳成交
- 一个平衡
- 一个偏收益

### 搜索顺序

因为 `rank1` 权重最高，搜索先重点确定 `rank1`，再逐步扩展 `rank2-5`。

### 剪枝规则

以下情况直接丢弃：

- 完全重复 `(currency, period)`
- stale / 数据不足候选
- 超出市场硬外边界候选
- `fUST` 未达“明显更高且成交不能明显更差”门槛却想压过 `fUSD`

## 验证闭环

### exploit 与 probe 分离

后续验证与训练中必须区分：

- `exploit`：最优组合产生的正式推荐
- `probe`：探索/刷新样本

probe 不得再污染主推荐评价。

### 统一身份

推荐、建单、验证、训练统一携带：

- `update_cycle_id`
- `recommendation_rank`
- `rank_weight`
- `candidate_id`
- `decision_mode`

### 真实路径标签

验证器回写时至少产生：

- `filled_in_stage1`
- `fell_to_stage2`
- `fell_to_stage3`
- `realized_wait_hours`
- `realized_terminal_value`
- `realized_terminal_mode`

### 弱标签

若当前没有可靠 FRR 历史，则：

- `stage1` 与 `stage3` 可作为强标签
- `stage2` 先记为 `weak_proxy`
- 训练时显式降权

### 缺数据样本

`NO_MARKET_DATA / stale / 缺历史` 统一视为：

- `UNKNOWN`
- `CENSORED`
- `WEAK_PROXY`

绝不能继续直接当成负样本。

## 训练重构

### 双链训练

训练拆成两条链：

1. `市场侧模型`
   只学习最近市场窗口与 regime

2. `路径侧模型`
   只学习 exploit 单的路径结果、等待成本、fallback 风险

### 训练目标

核心目标从“直接报利率/单点成交概率”转为：

- `stage1_fill_prob`
- `expected_wait_hours`
- `fallback_stage2_prob`
- `fallback_rank6_prob`
- `final_path_terminal_value`

### 历史脏数据处理

- 老 schema 且缺关键路径字段的样本，不进核心训练
- 旧数据最多只作为市场侧辅助统计
- exploit 样本是路径侧模型的主训练集

## 模块边界

优先遵守“如无必要，勿增实体”。

### 主要修改文件

- `/home/bumblebee/Project/optimize/ml_engine/predictor.py`
- `/home/bumblebee/Project/optimize/ml_engine/execution_validator.py`
- `/home/bumblebee/Project/optimize/ml_engine/training_data_builder.py`
- `/home/bumblebee/Project/optimize/ml_engine/model_trainer_v2.py`
- `/home/bumblebee/Project/optimize/ml_engine/order_manager.py`
- `/home/bumblebee/Project/optimize/scripts/evaluate_recent_optimization.py`

### 允许新增的小型 helper

仅当 `predictor.py` 过于臃肿时，允许新增一个小型 helper 文件，用于：

- 候选利率生成
- 组合搜索
- 组合评分

不新增独立服务。

## 迁移与上线策略

### Phase 1

补齐身份链路、diagnostics、fail-closed 机制。

### Phase 2

新候选利率生成器 shadow 运行，不直接替换线上输出。

### Phase 3

组合搜索器 shadow 运行，与旧排序并行对比。

### Phase 4

重写 validator 与训练标签，改成路径语言。

### Phase 5

确认 shadow 指标优于旧逻辑后，切换线上主逻辑。

### Phase 6

保留快速回退开关：

- 关闭组合搜索
- 关闭新候选生成
- 回退旧排序
- 回退旧训练目标

## 验收标准

方案切换到正式线上前，至少满足：

- 候选利率显著更贴近最近真实成交/close 区间
- 明显假高利率显著减少
- `rank1-5` shadow 组合的 `combo_revenue_ev` 高于旧逻辑
- fallback 到 `rank6` 的比例下降
- exploit 单的 realized terminal value 高于旧逻辑
- 重训练能稳定部署，且不会再因订单侧特征缺列而失败
- 输出格式保持兼容，`rank6 = fUSD-2d`

## 风险与边界

### 主要风险

- FRR 历史数据不足，导致 stage2 仍需弱标签
- 历史脏样本太多，需要严格过滤
- 组合搜索若剪枝不当，可能过早丢失真正优解

### 控制方式

- 先 shadow 再切换
- 统一 exploit/probe 身份
- 对弱标签显式降权
- 保留完整回退开关

## 结论

`C3` 保留现有外部格式，但把内部从“模型直接报最终利率的启发式排序器”，升级成“市场锚定候选 + 组合搜索 + 路径闭环验证”的最优组合引擎。

这比单纯修补 A/B 更贴近用户真实目标，也更有机会解决“报价不准、长时间挂不成交、闭环越学越偏”的根问题。
