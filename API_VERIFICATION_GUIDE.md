# API 验证指导说明

## 一、快速健康检查（日常）

```bash
# 1. 服务在线检查
curl https://leo.cpolar.cn/status
# 期望: status="online", current_step="Idle"

# 2. 验证测试
curl https://leo.cpolar.cn/validate
# 期望: passed >= 5, failed = 0, overall_status = "PASS"
# WARNING 可接受（通常是 API 延迟导致的时间戳偏差）
```

## 二、完整流程验证（模型更新后）

按顺序执行以下 4 步：

### Step 1: 记录基线
```bash
# 记录当前执行率和预测结果
curl https://leo.cpolar.cn/stats > baseline_stats.json
curl https://leo.cpolar.cn/result > baseline_result.json
```

### Step 2: 触发完整更新
```bash
curl -X POST https://leo.cpolar.cn/update
# 等待完成（轮询 /status 直到 status="online"）
while true; do
  status=$(curl -s https://leo.cpolar.cn/status | python3 -c "import sys,json; print(json.load(sys.stdin)['service_info']['status'])")
  [ "$status" = "online" ] && echo "Done" && break
  echo "Processing... ($status)"
  sleep 15
done
```

### Step 3: 对比结果
```bash
curl https://leo.cpolar.cn/result > new_result.json
curl https://leo.cpolar.cn/validate > new_validate.json
```

检查要点：
- `/validate` 仍然 overall_status = "PASS"
- `/result` 中 execution_probability 应 > 0.5
- 推荐利率应低于市场当前利率（current 字段）
- 不应出现极端利率（< 1% 或 > 50%）

### Step 4: 24-48h 后复查
```bash
curl https://leo.cpolar.cn/stats
```

关注 `execution_rate_7d` 字段：
- **目标**: 55-65% 整体执行率
- **过低 (<45%)**: 利率仍偏高，需进一步降低
- **过高 (>75%)**: 利率过于保守，损失收益

## 三、关键指标判断标准

| 指标 | 健康范围 | 异常处理 |
|------|----------|----------|
| 整体执行率 (7d) | 50-70% | <45% 需降利率; >80% 可适当提高 |
| 单组合执行率 | >20% | <10% 检查该组合是否应被排除 |
| 验证通过数 | >= 5/6 | < 5 需排查失败项 |
| 冷启动组合数 | 0 | >0 说明新组合数据不足 |
| PENDING 订单数 | < 总量 30% | >50% 说明验证流程可能卡住 |
| 推荐利率 vs 市场价 | 50-90% | <30% 过保守; >100% 必定失败 |

## 四、异常排查

```bash
# 服务不在线
curl https://leo.cpolar.cn/status
# → 检查 cpolar 隧道和后端进程是否运行

# 重训练失败
curl -X POST "https://leo.cpolar.cn/retrain?force=true"
# → 检查服务器日志: tail -100 nohup.out

# 执行率异常低
curl https://leo.cpolar.cn/stats
# → 查看各组合 exec_rate，定位问题组合
# → 检查市场数据是否正常更新

# 预测结果为空
curl https://leo.cpolar.cn/result
# → 重新触发: curl -X POST https://leo.cpolar.cn/predict
```
