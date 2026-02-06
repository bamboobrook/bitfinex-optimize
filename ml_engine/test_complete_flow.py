#!/usr/bin/env python3
"""
完整的 ML Engine 流程测试脚本
测试 API 端点并验证数据准确性
"""
import requests
import time
import json
from datetime import datetime

# API 配置
API_BASE_URL = "https://leo.cpolar.cn"

def log_step(step_name):
    """打印步骤信息"""
    print(f"\n{'='*60}")
    print(f"步骤: {step_name}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

def wait_for_idle(max_wait=300):
    """等待 API 空闲"""
    print("等待 API 空闲...")
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"{API_BASE_URL}/status", timeout=10)
            response.raise_for_status()
            data = response.json()
            current_step = data.get('service_info', {}).get('current_step', 'Unknown')

            if current_step == 'Idle':
                print(f"✓ API 已空闲")
                return True
            else:
                print(f"  当前状态: {current_step}，等待中...")
                time.sleep(5)
        except Exception as e:
            print(f"  检查状态时出错: {e}")
            time.sleep(5)

    print(f"✗ 等待超时 ({max_wait}秒)")
    return False

def check_status():
    """检查 API 状态"""
    log_step("1. 检查 API 状态")
    try:
        response = requests.get(f"{API_BASE_URL}/status", timeout=10)
        response.raise_for_status()
        data = response.json()
        print(f"✓ API 状态: {json.dumps(data, indent=2, ensure_ascii=False)}")
        return True
    except Exception as e:
        print(f"✗ API 状态检查失败: {e}")
        return False

def download_data():
    """下载数据"""
    log_step("2. 下载数据")
    try:
        # 等待空闲
        if not wait_for_idle():
            return False

        payload = {
            "days": 30,  # 下载最近30天的数据
            "force_refresh": False
        }
        response = requests.post(
            f"{API_BASE_URL}/download_data",
            json=payload,
            timeout=300
        )
        response.raise_for_status()
        data = response.json()
        print(f"✓ 数据下载任务已启动:")
        print(f"  - 状态: {data.get('status')}")
        print(f"  - 消息: {data.get('message')}")

        # 等待下载完成
        if not wait_for_idle(max_wait=300):
            return False

        return True
    except Exception as e:
        print(f"✗ 数据下载失败: {e}")
        return False

def process_features():
    """处理特征"""
    log_step("3. 处理特征")
    try:
        # 等待空闲
        if not wait_for_idle():
            return False

        payload = {
            "lookback_days": 30
        }
        response = requests.post(
            f"{API_BASE_URL}/process_features",
            json=payload,
            timeout=300
        )
        response.raise_for_status()
        data = response.json()
        print(f"✓ 特征处理任务已启动:")
        print(f"  - 状态: {data.get('status')}")
        print(f"  - 消息: {data.get('message')}")

        # 等待处理完成
        if not wait_for_idle(max_wait=300):
            return False

        return True
    except Exception as e:
        print(f"✗ 特征处理失败: {e}")
        return False

def train_model():
    """训练模型"""
    log_step("4. 训练模型")
    try:
        # 等待空闲
        if not wait_for_idle():
            return False

        payload = {
            "model_type": "xgboost",
            "test_size": 0.2
        }
        response = requests.post(
            f"{API_BASE_URL}/train",
            json=payload,
            timeout=600
        )
        response.raise_for_status()
        data = response.json()
        print(f"✓ 模型训练任务已启动:")
        print(f"  - 状态: {data.get('status')}")
        print(f"  - 消息: {data.get('message')}")

        # 等待训练完成
        if not wait_for_idle(max_wait=600):
            return False

        # 获取训练结果
        result_response = requests.get(f"{API_BASE_URL}/result", timeout=30)
        if result_response.status_code == 200:
            result_data = result_response.json()
            if 'model_metrics' in result_data:
                print(f"  - 模型性能指标:")
                for key, value in result_data.get('model_metrics', {}).items():
                    print(f"    * {key}: {value}")

        return True
    except Exception as e:
        print(f"✗ 模型训练失败: {e}")
        return False

def predict():
    """执行预测"""
    log_step("5. 执行预测")
    try:
        # 等待空闲
        if not wait_for_idle():
            return False

        payload = {
            "lookback_hours": 24
        }
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=300
        )
        response.raise_for_status()
        data = response.json()
        print(f"✓ 预测任务已启动:")
        print(f"  - 状态: {data.get('status')}")
        print(f"  - 消息: {data.get('message')}")

        # 等待预测完成
        if not wait_for_idle(max_wait=300):
            return False

        return True
    except Exception as e:
        print(f"✗ 预测执行失败: {e}")
        return False

def validate_orders():
    """验证订单"""
    log_step("6. 验证订单")
    try:
        # 等待空闲
        if not wait_for_idle():
            return False

        payload = {
            "lookback_hours": 48
        }
        response = requests.post(
            f"{API_BASE_URL}/validate_orders",
            json=payload,
            timeout=300
        )
        response.raise_for_status()
        data = response.json()
        print(f"✓ 订单验证任务已启动:")
        print(f"  - 状态: {data.get('status')}")
        print(f"  - 消息: {data.get('message')}")

        # 等待验证完成
        if not wait_for_idle(max_wait=300):
            return False

        return True
    except Exception as e:
        print(f"✗ 订单验证失败: {e}")
        return False

def get_execution_stats():
    """获取执行统计"""
    log_step("7. 获取执行统计")
    try:
        response = requests.get(f"{API_BASE_URL}/execution_stats", timeout=30)
        response.raise_for_status()
        data = response.json()
        print(f"✓ 执行统计:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return True
    except Exception as e:
        print(f"✗ 获取执行统计失败: {e}")
        return False

def get_stats():
    """获取系统统计"""
    log_step("8. 获取系统统计")
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=30)
        response.raise_for_status()
        data = response.json()
        print(f"✓ 系统统计:")
        print(json.dumps(data, indent=2, ensure_ascii=False))

        # 验证数据准确性
        print("\n数据验证:")
        status_summary = data.get('status_summary', [])
        total_orders = sum(item['count'] for item in status_summary)
        print(f"  - 总订单数: {total_orders}")
        for item in status_summary:
            percentage = (item['count'] / total_orders * 100) if total_orders > 0 else 0
            print(f"  - {item['status']}: {item['count']} ({percentage:.1f}%)")

        return True
    except Exception as e:
        print(f"✗ 获取系统统计失败: {e}")
        return False

def get_result():
    """获取最新结果"""
    log_step("9. 获取最新结果")
    try:
        response = requests.get(f"{API_BASE_URL}/result", timeout=30)
        response.raise_for_status()
        data = response.json()
        print(f"✓ 最新结果:")
        print(f"  - 时间戳: {data.get('timestamp')}")
        print(f"  - 状态: {data.get('status')}")
        print(f"  - 策略: {data.get('strategy_info')}")

        recommendations = data.get('recommendations', [])
        if recommendations:
            print(f"\n  推荐列表 (共 {len(recommendations)} 个):")
            for rec in recommendations[:3]:  # 只显示前3个
                print(f"    {rec['rank']}. {rec['currency']}-{rec['period']}d @ {rec['rate']:.2f}%")
                print(f"       执行概率: {rec['details']['execution_probability']:.2%}")
                print(f"       当前市场利率: {rec['details']['current']:.2f}%")

        return True
    except Exception as e:
        print(f"✗ 获取最新结果失败: {e}")
        return False

def main():
    """主流程"""
    print("\n" + "="*60)
    print("ML Engine 完整流程测试")
    print(f"API 地址: {API_BASE_URL}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # 执行测试步骤
    steps = [
        ("检查 API 状态", check_status),
        ("下载数据", download_data),
        ("处理特征", process_features),
        ("训练模型", train_model),
        ("执行预测", predict),
        ("验证订单", validate_orders),
        ("获取执行统计", get_execution_stats),
        ("获取系统统计", get_stats),
        ("获取最新结果", get_result),
    ]

    results = []
    for step_name, step_func in steps:
        success = step_func()
        results.append((step_name, success))
        if not success:
            print(f"\n⚠️  步骤 '{step_name}' 失败，继续执行后续步骤...")

    # 打印总结
    log_step("测试总结")
    print("步骤执行结果:")
    for step_name, success in results:
        status = "✓ 成功" if success else "✗ 失败"
        print(f"  {status}: {step_name}")

    success_count = sum(1 for _, success in results if success)
    total_count = len(results)
    print(f"\n总计: {success_count}/{total_count} 步骤成功")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")

    return success_count == total_count

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
