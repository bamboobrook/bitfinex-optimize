# ML 预测系统 API 健康检测指引

> **API Base URL**: `https://leo.cpolar.cn`
> **系统位置**: `ml_engine/`
> **用途**: 为外部监控程序提供系统健康状态判断标准

---

## 📋 目录

1. [快速开始](#快速开始)
2. [核心监控端点](#核心监控端点)
3. [健康判断标准](#健康判断标准)
4. [异常检测规则](#异常检测规则)
5. [监控最佳实践](#监控最佳实践)
6. [故障排查流程](#故障排查流程)

---

## 🚀 快速开始

### 最简单的健康检查（推荐）

```bash
# 1. 检查 API 是否在线
curl https://leo.cpolar.cn/status

# 预期响应:
{
  "api_online": true,
  "service_info": {
    "status": "online",           # 或 "processing" / "error"
    "current_step": "",
    "last_update": "2025-01-15 10:30:45",
    "details": ""
  }
}
```

**判断标准**:
- ✅ **正常**: `api_online == true` 且 `service_info.status == "online"` 或 `"processing"`
- ❌ **异常**: 无响应、超时、`service_info.status == "error"`

---

## 🔍 核心监控端点

### 1️⃣ 基础健康检查 `/status`

**用途**: 确认 API 在线状态及后台任务进度

```bash
GET https://leo.cpolar.cn/status
```

**响应示例**:
```json
{
  "api_online": true,
  "service_info": {
    "status": "processing",
    "current_step": "正在训练模型...",
    "last_update": "2025-01-15 10:35:12",
    "details": "处理中"
  }
}
```

**字段说明**:

| 字段 | 值 | 含义 |
|------|------|------|
| `status` | `"online"` | 空闲在线,可接受新任务 |
|  | `"processing"` | 正在执行后台任务（正常） |
|  | `"error"` | 出现错误（需关注） |
|  | `"unknown"` | 状态文件丢失（异常） |
| `current_step` | 字符串 | 当前执行步骤描述 |
| `last_update` | 时间戳 | 最后更新时间 |

---

### 2️⃣ 预测结果检查 `/result`

**用途**: 验证系统是否产生有效预测数据

```bash
GET https://leo.cpolar.cn/result
```

**正常响应示例**:
```json
{
  "optimal_combination": {
    "currency": "fUSD",
    "period": 30,
    "predicted_rate": 15.23,
    "profit": 0.012,
    "recommendation": "建议下单"
  },
  "timestamp": "2025-01-15 10:30:00",
  "model_version": "v3.0"
}
```

**异常响应**:
```json
// HTTP 404
{
  "error": "No predictions found.",
  "suggestion": "Please call /update to generate data."
}

// HTTP 500
{
  "error": "Failed to read data: [错误详情]"
}
```

**判断标准**:
- ✅ **正常**: HTTP 200 + 包含 `optimal_combination` 字段
- ⚠️ **警告**: HTTP 404（首次启动或数据未生成,可触发 `/update`）
- ❌ **异常**: HTTP 500（文件损坏或读取错误）

---

### 3️⃣ 数据库统计 `/stats`

**用途**: 检查订单系统运行健康度

```bash
GET https://leo.cpolar.cn/stats
```

**响应示例**:
```json
{
  "status_summary": [
    {"status": "PENDING", "count": 12},
    {"status": "EXECUTED", "count": 245},
    {"status": "FAILED", "count": 18}
  ],
  "execution_rate_7d": [
    {
      "currency": "fUSD",
      "period": 30,
      "total": 50,
      "executed": 42,
      "exec_rate": 84.0
    }
  ],
  "latest_orders": [
    {
      "id": "abc12345",
      "combo": "fUSD-30d",
      "rate": "15.23%",
      "status": "EXECUTED",
      "created": "2025-01-15 10:20:00"
    }
  ]
}
```

**健康指标**:

| 指标 | 正常范围 | 异常阈值 |
|------|---------|---------|
| **7天执行率** (`exec_rate`) | 60% - 90% | < 50% 或 > 95% |
| **PENDING 订单数** | < 20 | > 50（订单堆积） |
| **最新订单时间** | < 3小时前 | > 6小时前（系统停滞） |

---

### 4️⃣ 系统验证 `/validate`

**用途**: 运行完整的数据完整性和逻辑正确性验证

```bash
GET https://leo.cpolar.cn/validate
```

**响应示例**:
```json
{
  "overall_status": "PASS",  // "PASS" 或 "FAIL"
  "total_tests": 6,
  "passed": 6,
  "failed": 0,
  "tests": [
    {
      "name": "时间戳正确性",
      "status": "PASS",
      "details": "所有订单时间戳在±5分钟内"
    },
    {
      "name": "验证窗口合规性",
      "status": "PASS",
      "details": "无 look-ahead bias"
    },
    {
      "name": "采样覆盖率",
      "status": "PASS",
      "details": "7天内覆盖 95.2% 组合"
    }
  ],
  "timestamp": "2025-01-15 10:40:00"
}
```

**判断标准**:
- ✅ **正常**: `overall_status == "PASS"` 且 `passed == total_tests`
- ⚠️ **警告**: `passed >= 4` (大部分通过)
- ❌ **异常**: `overall_status == "FAIL"` 或 `passed < 4`

---

## ✅ 健康判断标准

### 分级监控策略

#### 🟢 Level 1: 基础可用性（每 5 分钟检查）

```python
# 伪代码示例
def check_basic_health():
    response = requests.get("https://leo.cpolar.cn/status", timeout=10)

    if response.status_code != 200:
        return "CRITICAL: API 不可访问"

    data = response.json()
    if not data.get("api_online"):
        return "CRITICAL: API 离线"

    status = data.get("service_info", {}).get("status")
    if status == "error":
        return "ERROR: 服务报错"

    return "OK"
```

**指标**:
- ✅ HTTP 200 响应 < 3秒
- ✅ `api_online == true`
- ✅ `status` 为 `online` 或 `processing`

---

#### 🟡 Level 2: 功能健康（每 30 分钟检查）

```python
def check_functional_health():
    # 1. 检查预测数据
    result = requests.get("https://leo.cpolar.cn/result")
    if result.status_code == 404:
        return "WARNING: 无预测数据"

    # 2. 检查订单统计
    stats = requests.get("https://leo.cpolar.cn/stats").json()

    # 检查最新订单时间
    latest_order_time = stats["latest_orders"][0]["created"]
    if parse_time(latest_order_time) < now() - timedelta(hours=6):
        return "WARNING: 订单系统停滞 >6小时"

    # 检查 PENDING 订单堆积
    pending_count = next(
        (item["count"] for item in stats["status_summary"]
         if item["status"] == "PENDING"),
        0
    )
    if pending_count > 50:
        return "WARNING: PENDING 订单堆积"

    return "OK"
```

**指标**:
- ✅ `/result` 返回有效数据
- ✅ 最新订单时间 < 6小时前
- ✅ PENDING 订单 < 50

---

#### 🔴 Level 3: 数据质量（每 2 小时检查）

```python
def check_data_quality():
    validate = requests.get("https://leo.cpolar.cn/validate").json()

    if validate["overall_status"] == "FAIL":
        failed_tests = [
            test["name"] for test in validate["tests"]
            if test["status"] == "FAIL"
        ]
        return f"ERROR: 验证失败 - {', '.join(failed_tests)}"

    # 检查执行率
    stats = requests.get("https://leo.cpolar.cn/stats").json()
    for item in stats["execution_rate_7d"]:
        if item["exec_rate"] < 50 or item["exec_rate"] > 95:
            return f"WARNING: {item['currency']}-{item['period']}d 执行率异常: {item['exec_rate']}%"

    return "OK"
```

**指标**:
- ✅ 验证测试全部通过
- ✅ 7天执行率在 50%-95% 之间
- ✅ 采样覆盖率 > 80%

---

## ⚠️ 异常检测规则

### 故障类型与响应

| 异常类型 | 检测方法 | 严重程度 | 建议操作 |
|---------|---------|---------|---------|
| **API 无响应** | `/status` 超时 (>10秒) | 🔴 CRITICAL | 立即告警,检查服务器 |
| **服务报错** | `status == "error"` | 🔴 CRITICAL | 查看日志,可能需重启 |
| **无预测数据** | `/result` 返回 404 | 🟡 WARNING | 触发 `POST /update` |
| **订单停滞** | 最新订单 >6小时前 | 🟡 WARNING | 检查定时任务,触发 `/update` |
| **执行率异常** | < 50% 或 > 95% | 🟡 WARNING | 检查市场数据源 |
| **验证失败** | `/validate` 返回 FAIL | 🔴 CRITICAL | 检查数据完整性 |
| **PENDING 堆积** | PENDING 订单 > 50 | 🟡 WARNING | 检查订单处理逻辑 |

---

### 典型异常场景

#### Scenario 1: 系统刚启动（首次运行）

```json
// GET /status
{"api_online": true, "service_info": {"status": "online", ...}}

// GET /result
{"error": "No predictions found.", ...}  // HTTP 404
```

**判断**: ✅ 正常（首次启动）
**操作**: 触发 `POST /update` 生成初始数据

---

#### Scenario 2: 后台任务运行中

```json
// GET /status
{
  "api_online": true,
  "service_info": {
    "status": "processing",
    "current_step": "正在训练模型...",
    "last_update": "2025-01-15 10:45:30"
  }
}
```

**判断**: ✅ 正常（任务执行中）
**监控**: 如果 `processing` 持续 >30分钟,检查是否卡死

---

#### Scenario 3: 验证失败

```json
// GET /validate
{
  "overall_status": "FAIL",
  "passed": 4,
  "failed": 2,
  "tests": [
    {"name": "采样覆盖率", "status": "FAIL", "details": "仅 60% 组合覆盖"},
    {"name": "执行率真实性", "status": "FAIL", "details": "执行率 97%，超过阈值"}
  ]
}
```

**判断**: 🔴 异常（数据质量问题）
**操作**:
1. 记录失败详情
2. 检查 `ml_engine/` 日志文件
3. 可能需要人工介入修复

---

## 📊 监控最佳实践

### 推荐监控流程

```python
import requests
from datetime import datetime, timedelta
import time

class MLSystemMonitor:
    BASE_URL = "https://leo.cpolar.cn"

    def run_health_check(self):
        """完整健康检查流程"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }

        # 1. 基础可用性检查（必需）
        results["checks"]["api_status"] = self.check_api_status()
        if results["checks"]["api_status"]["status"] != "OK":
            results["overall"] = "CRITICAL"
            return results

        # 2. 功能健康检查（必需）
        results["checks"]["prediction_data"] = self.check_prediction_data()
        results["checks"]["order_system"] = self.check_order_system()

        # 3. 数据质量检查（可选,每2小时）
        if self.should_run_deep_check():
            results["checks"]["validation"] = self.check_validation()
            results["checks"]["execution_rate"] = self.check_execution_rate()

        # 综合判断
        results["overall"] = self.calculate_overall_status(results["checks"])
        return results

    def check_api_status(self):
        try:
            resp = requests.get(f"{self.BASE_URL}/status", timeout=10)
            data = resp.json()

            if not data.get("api_online"):
                return {"status": "CRITICAL", "reason": "API offline"}

            service_status = data.get("service_info", {}).get("status")
            if service_status == "error":
                return {"status": "ERROR", "reason": "Service error"}

            return {"status": "OK", "service_status": service_status}
        except Exception as e:
            return {"status": "CRITICAL", "reason": f"Connection failed: {e}"}

    def check_prediction_data(self):
        try:
            resp = requests.get(f"{self.BASE_URL}/result", timeout=10)
            if resp.status_code == 404:
                return {"status": "WARNING", "reason": "No prediction data"}
            if resp.status_code != 200:
                return {"status": "ERROR", "reason": f"HTTP {resp.status_code}"}

            data = resp.json()
            if "optimal_combination" not in data:
                return {"status": "ERROR", "reason": "Invalid data format"}

            return {"status": "OK", "data": data}
        except Exception as e:
            return {"status": "ERROR", "reason": str(e)}

    def check_order_system(self):
        try:
            resp = requests.get(f"{self.BASE_URL}/stats", timeout=10)
            stats = resp.json()

            # 检查最新订单时间
            if stats.get("latest_orders"):
                latest_time = datetime.fromisoformat(
                    stats["latest_orders"][0]["created"]
                )
                hours_ago = (datetime.now() - latest_time).total_seconds() / 3600

                if hours_ago > 6:
                    return {
                        "status": "WARNING",
                        "reason": f"Order system stalled for {hours_ago:.1f} hours"
                    }

            # 检查 PENDING 堆积
            pending = next(
                (item["count"] for item in stats.get("status_summary", [])
                 if item["status"] == "PENDING"),
                0
            )
            if pending > 50:
                return {
                    "status": "WARNING",
                    "reason": f"PENDING orders backlog: {pending}"
                }

            return {"status": "OK", "stats": stats}
        except Exception as e:
            return {"status": "ERROR", "reason": str(e)}

    def check_validation(self):
        try:
            resp = requests.get(f"{self.BASE_URL}/validate", timeout=30)
            data = resp.json()

            if data.get("overall_status") == "FAIL":
                failed_tests = [
                    test["name"] for test in data.get("tests", [])
                    if test.get("status") == "FAIL"
                ]
                return {
                    "status": "ERROR",
                    "reason": f"Validation failed: {', '.join(failed_tests)}"
                }

            return {"status": "OK", "validation": data}
        except Exception as e:
            return {"status": "ERROR", "reason": str(e)}

    def check_execution_rate(self):
        try:
            resp = requests.get(f"{self.BASE_URL}/stats", timeout=10)
            stats = resp.json()

            warnings = []
            for item in stats.get("execution_rate_7d", []):
                rate = item["exec_rate"]
                if rate < 50 or rate > 95:
                    warnings.append(
                        f"{item['currency']}-{item['period']}d: {rate}%"
                    )

            if warnings:
                return {
                    "status": "WARNING",
                    "reason": f"Abnormal execution rates: {', '.join(warnings)}"
                }

            return {"status": "OK"}
        except Exception as e:
            return {"status": "ERROR", "reason": str(e)}

    def should_run_deep_check(self):
        """每2小时运行一次深度检查"""
        return datetime.now().minute < 10  # 简化示例

    def calculate_overall_status(self, checks):
        """综合判断整体状态"""
        if any(c.get("status") == "CRITICAL" for c in checks.values()):
            return "CRITICAL"
        if any(c.get("status") == "ERROR" for c in checks.values()):
            return "ERROR"
        if any(c.get("status") == "WARNING" for c in checks.values()):
            return "WARNING"
        return "OK"

# 使用示例
if __name__ == "__main__":
    monitor = MLSystemMonitor()

    while True:
        result = monitor.run_health_check()
        print(f"[{result['timestamp']}] Overall: {result['overall']}")

        # 根据状态采取行动
        if result["overall"] in ["CRITICAL", "ERROR"]:
            print("🚨 ALERT:", result)
            # 发送告警通知
        elif result["overall"] == "WARNING":
            print("⚠️  WARNING:", result)
            # 记录警告日志

        time.sleep(300)  # 每5分钟检查一次
```

---

### 监控频率建议

| 检查类型 | 频率 | 超时设置 | 告警条件 |
|---------|------|---------|---------|
| `/status` | 每 5 分钟 | 10 秒 | 连续 2 次失败 |
| `/result` | 每 30 分钟 | 10 秒 | 连续 3 次 404 |
| `/stats` | 每 30 分钟 | 10 秒 | 订单停滞 >6小时 |
| `/validate` | 每 2 小时 | 30 秒 | 任意测试失败 |

---

## 🔧 故障排查流程

### Step 1: API 无响应

```bash
# 1. 检查网络连通性
ping leo.cpolar.cn

# 2. 检查端口是否开放
nc -zv leo.cpolar.cn 443

# 3. 检查服务器上的进程
ssh user@server
ps aux | grep uvicorn
```

**可能原因**:
- cpolar 隧道断开
- uvicorn 进程崩溃
- 服务器资源耗尽

---

### Step 2: 服务报错 (`status == "error"`)

```bash
# 查看最新日志
tail -n 100 /home/bumblebee/Project/optimize/log/ml_optimizer.log

# 检查状态文件
cat /home/bumblebee/Project/optimize/data/service_status.json
```

**可能原因**:
- 数据下载失败（API 限流、网络错误）
- 模型训练异常（数据不足、内存溢出）
- 数据库损坏

---

### Step 3: 验证失败

```bash
# 运行单个验证测试
curl https://leo.cpolar.cn/validate/timestamp_correctness
curl https://leo.cpolar.cn/validate/sampling_coverage

# 检查数据库内容
sqlite3 /home/bumblebee/Project/optimize/data/lending_history.db
> SELECT status, COUNT(*) FROM virtual_orders GROUP BY status;
> SELECT * FROM virtual_orders ORDER BY order_timestamp DESC LIMIT 10;
```

**常见问题**:
- **采样覆盖率低**: 市场数据源不稳定
- **执行率异常高**: 验证逻辑错误或数据回填问题
- **时间戳错误**: 系统时钟偏移

---

## 📈 健康报告示例

### 正常状态报告

```json
{
  "timestamp": "2025-01-15T10:50:00",
  "overall": "OK",
  "checks": {
    "api_status": {
      "status": "OK",
      "service_status": "online"
    },
    "prediction_data": {
      "status": "OK",
      "data": {
        "optimal_combination": {
          "currency": "fUSD",
          "period": 30,
          "predicted_rate": 15.23
        }
      }
    },
    "order_system": {
      "status": "OK",
      "stats": {
        "status_summary": [
          {"status": "PENDING", "count": 8},
          {"status": "EXECUTED", "count": 245}
        ]
      }
    },
    "validation": {
      "status": "OK",
      "validation": {
        "overall_status": "PASS",
        "passed": 6,
        "failed": 0
      }
    }
  }
}
```

---

### 异常状态报告

```json
{
  "timestamp": "2025-01-15T11:00:00",
  "overall": "ERROR",
  "checks": {
    "api_status": {
      "status": "OK",
      "service_status": "online"
    },
    "prediction_data": {
      "status": "WARNING",
      "reason": "No prediction data"
    },
    "order_system": {
      "status": "WARNING",
      "reason": "Order system stalled for 7.2 hours"
    },
    "validation": {
      "status": "ERROR",
      "reason": "Validation failed: 采样覆盖率, 执行率真实性"
    }
  },
  "recommendations": [
    "触发 POST /update 生成新预测数据",
    "检查定时任务是否正常运行",
    "查看日志文件排查验证失败原因"
  ]
}
```

---

## 🎯 快速决策表

| 场景 | `/status` | `/result` | `/validate` | 判断 | 操作 |
|------|----------|----------|------------|------|------|
| 正常运行 | ✅ online | ✅ 200 | ✅ PASS | 🟢 健康 | 继续监控 |
| 首次启动 | ✅ online | ❌ 404 | - | 🟡 正常 | 触发 `/update` |
| 任务运行中 | ✅ processing | ✅ 200 | - | 🟢 健康 | 等待完成 |
| 任务卡死 | ⚠️ processing >30min | ✅ 200 | - | 🟡 警告 | 检查日志 |
| 服务报错 | ❌ error | - | - | 🔴 异常 | 查看日志,可能需重启 |
| 数据质量差 | ✅ online | ✅ 200 | ❌ FAIL | 🔴 异常 | 运行修复脚本 |
| API 无响应 | ❌ 超时 | - | - | 🔴 严重 | 检查服务器 |

---

## 📞 联系方式

- **日志位置**: `/home/bumblebee/Project/optimize/log/ml_optimizer.log`
- **数据库位置**: `/home/bumblebee/Project/optimize/data/lending_history.db`
- **状态文件**: `/home/bumblebee/Project/optimize/data/service_status.json`

---

## 📚 附录

### 完整 API 端点列表

| 端点 | 方法 | 用途 | 监控优先级 |
|------|------|------|-----------|
| `/status` | GET | 健康检查 | ⭐⭐⭐ |
| `/result` | GET | 获取预测结果 | ⭐⭐⭐ |
| `/stats` | GET | 订单统计 | ⭐⭐ |
| `/validate` | GET | 系统验证 | ⭐⭐ |
| `/orders` | GET | 订单列表 | ⭐ |
| `/execution_stats` | GET | 执行统计 | ⭐ |
| `/update` | POST | 触发完整更新 | - |
| `/download_data` | POST | 触发数据下载 | - |
| `/train` | POST | 触发模型训练 | - |
| `/predict` | POST | 触发预测生成 | - |
| `/validate_orders` | POST | 触发订单验证 | - |

### 状态码说明

| HTTP Code | 含义 | 常见场景 |
|-----------|------|---------|
| 200 | 成功 | 正常响应 |
| 404 | 未找到 | 预测数据不存在（首次启动） |
| 409 | 冲突 | 任务已在运行（调用 `/update` 时） |
| 500 | 服务器错误 | 内部异常（文件读取失败、数据库错误） |

---

**文档版本**: v1.0
**最后更新**: 2025-01-15
**适用系统**: ML Lending Optimization API v3.0
