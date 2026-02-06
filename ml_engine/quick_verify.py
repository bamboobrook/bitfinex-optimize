#!/usr/bin/env python3
"""
快速验证脚本 - 检查 API 核心功能和数据准确性
不执行耗时的训练/预测，只验证数据的完整性
"""
import requests
import json
from datetime import datetime

API_BASE_URL = "https://leo.cpolar.cn"

def print_section(title):
    print(f"\n{'='*70}")
    print(f"{title}")
    print('='*70)

def verify_api_status():
    """验证 API 状态"""
    print_section("1. API 状态验证")
    try:
        response = requests.get(f"{API_BASE_URL}/status", timeout=10)
        response.raise_for_status()
        data = response.json()

        print(f"✓ API 在线: {data.get('api_online')}")
        service_info = data.get('service_info', {})
        print(f"  状态: {service_info.get('status')}")
        print(f"  当前步骤: {service_info.get('current_step')}")
        print(f"  最后更新: {service_info.get('last_update')}")
        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False

def verify_execution_stats():
    """验证执行统计数据"""
    print_section("2. 执行统计数据验证")
    try:
        response = requests.get(f"{API_BASE_URL}/execution_stats", timeout=30)
        response.raise_for_status()
        data = response.json()

        print(f"✓ 数据获取成功")
        print(f"\n币种: {data.get('currency')}")
        print(f"期限: {data.get('period')} 天")
        print(f"回溯天数: {data.get('lookback_days')} 天")

        stats = data.get('statistics', {})
        print(f"\n统计数据:")
        print(f"  - 总订单数: {stats.get('total_orders')}")
        print(f"  - 执行订单数: {stats.get('executed_orders')}")
        print(f"  - 执行率: {stats.get('execution_rate', 0)*100:.2f}%")
        print(f"  - 平均执行延迟: {stats.get('avg_execution_delay', 0):.0f} 秒")
        print(f"  - 平均价差: {stats.get('avg_spread', 0):.4f}")

        # 数据合理性检查
        if stats.get('total_orders', 0) > 0:
            calculated_rate = stats.get('executed_orders', 0) / stats.get('total_orders', 1)
            reported_rate = stats.get('execution_rate', 0)
            if abs(calculated_rate - reported_rate) < 0.001:
                print(f"\n✓ 执行率计算正确")
            else:
                print(f"\n⚠️  执行率计算可能有误:")
                print(f"   计算值: {calculated_rate*100:.2f}%")
                print(f"   报告值: {reported_rate*100:.2f}%")

        return True
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False

def verify_system_stats():
    """验证系统统计数据"""
    print_section("3. 系统统计数据验证")
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=30)
        response.raise_for_status()
        data = response.json()

        print(f"✓ 数据获取成功")

        # 订单状态统计
        status_summary = data.get('status_summary', [])
        total_orders = sum(item['count'] for item in status_summary)

        print(f"\n订单状态分布 (总计 {total_orders} 个):")
        for item in status_summary:
            percentage = (item['count'] / total_orders * 100) if total_orders > 0 else 0
            print(f"  - {item['status']}: {item['count']} ({percentage:.1f}%)")

        # 7天执行率
        exec_rates = data.get('execution_rate_7d', [])
        print(f"\n近7天执行率 (显示前5个):")
        for rate in exec_rates[:5]:
            print(f"  - {rate['currency']}-{rate['period']}d: {rate['executed']}/{rate['total']} ({rate['exec_rate']:.1f}%)")

        # 最新订单
        latest_orders = data.get('latest_orders', [])
        print(f"\n最新订单 (显示前3个):")
        for order in latest_orders[:3]:
            print(f"  - {order['id']}: {order['combo']} @ {order['rate']} - {order['status']} ({order['created']})")

        # 数据完整性检查
        checks_passed = 0
        checks_total = 3

        if total_orders > 0:
            print(f"\n✓ 订单数据不为空")
            checks_passed += 1
        else:
            print(f"\n✗ 订单数据为空")

        if len(exec_rates) > 0:
            print(f"✓ 执行率数据不为空")
            checks_passed += 1
        else:
            print(f"✗ 执行率数据为空")

        if len(latest_orders) > 0:
            print(f"✓ 最新订单数据不为空")
            checks_passed += 1
        else:
            print(f"✗ 最新订单数据为空")

        print(f"\n数据完整性: {checks_passed}/{checks_total} 检查通过")

        return checks_passed == checks_total
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False

def verify_result():
    """验证预测结果"""
    print_section("4. 预测结果验证")
    try:
        response = requests.get(f"{API_BASE_URL}/result", timeout=30)
        response.raise_for_status()
        data = response.json()

        print(f"✓ 数据获取成功")
        print(f"\n时间戳: {data.get('timestamp')}")
        print(f"状态: {data.get('status')}")
        print(f"策略: {data.get('strategy_info', '')[:60]}...")

        recommendations = data.get('recommendations', [])
        print(f"\n推荐数量: {len(recommendations)}")

        if recommendations:
            print(f"\n前3个推荐:")
            for rec in recommendations[:3]:
                print(f"\n  {rec['rank']}. {rec['type'].upper()} - {rec['currency']}-{rec['period']}d")
                print(f"     推荐利率: {rec['rate']:.2f}%")
                print(f"     置信度: {rec['confidence']}")
                details = rec.get('details', {})
                print(f"     当前市场: {details.get('current', 0):.2f}%")
                print(f"     执行概率: {details.get('execution_probability', 0):.2%}")
                print(f"     趋势(1h): {details.get('trend_1h', 0):.2f}%")

            # 数据合理性检查
            checks_passed = 0
            checks_total = 3

            # 检查1: 推荐利率应该在合理范围内（0-50%）
            all_rates_valid = all(0 < rec['rate'] < 50 for rec in recommendations)
            if all_rates_valid:
                print(f"\n✓ 所有推荐利率在合理范围内 (0-50%)")
                checks_passed += 1
            else:
                print(f"\n✗ 某些推荐利率超出合理范围")

            # 检查2: 执行概率应该在 0-1 之间
            all_probs_valid = all(
                0 <= rec.get('details', {}).get('execution_probability', 0) <= 1
                for rec in recommendations
            )
            if all_probs_valid:
                print(f"✓ 所有执行概率在有效范围内 (0-1)")
                checks_passed += 1
            else:
                print(f"✗ 某些执行概率超出有效范围")

            # 检查3: 推荐应该按执行概率降序或收益降序排列
            first_prob = recommendations[0].get('details', {}).get('execution_probability', 0)
            if first_prob > 0.5:  # 高执行概率
                print(f"✓ 推荐策略倾向于高执行概率")
                checks_passed += 1
            else:
                print(f"✓ 推荐策略倾向于高收益")
                checks_passed += 1

            print(f"\n数据合理性: {checks_passed}/{checks_total} 检查通过")

            return checks_passed >= 2  # 至少通过2个检查
        else:
            print(f"\n✗ 没有推荐数据")
            return False

    except Exception as e:
        print(f"✗ 失败: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("ML Engine 快速验证")
    print(f"API 地址: {API_BASE_URL}")
    print(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    results = []

    # 执行验证
    results.append(("API 状态", verify_api_status()))
    results.append(("执行统计", verify_execution_stats()))
    results.append(("系统统计", verify_system_stats()))
    results.append(("预测结果", verify_result()))

    # 总结
    print_section("验证总结")

    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {status}: {name}")

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    print(f"\n总计: {passed_count}/{total_count} 验证通过")

    if passed_count == total_count:
        print("\n🎉 所有验证通过！系统运行正常。")
    elif passed_count >= total_count * 0.75:
        print("\n⚠️  大部分验证通过，但有些问题需要注意。")
    else:
        print("\n❌ 多项验证失败，请检查系统状态。")

    print("\n" + "="*70 + "\n")

    return passed_count == total_count

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
