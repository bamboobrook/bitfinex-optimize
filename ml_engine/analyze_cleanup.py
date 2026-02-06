#!/usr/bin/env python3
"""
分析 ml_engine 目录中的文件，确定哪些可以删除
"""
import os
from pathlib import Path
from collections import defaultdict

# 文件分类
FILE_CATEGORIES = {
    "核心模块": [
        "api_server.py",
        "data_processor.py",
        "execution_features.py",
        "execution_validator.py",
        "metrics.py",
        "model_trainer.py",
        "order_manager.py",
        "predictor.py",
        "__init__.py"
    ],
    "初始化脚本": [
        "init_execution_tables.py",  # 数据库表初始化（一次性）
    ],
    "维护脚本": [
        "backfill_expired_orders.py",  # 回填过期订单
        "cleanup_historical_data.py",  # 清理历史数据
        "revalidate_all_orders.py",  # 重新验证所有订单
        "validate_old_pending.py",  # 验证旧的待处理订单
    ],
    "分析工具": [
        "analyze_execution_rates.py",  # 分析执行率
        "evaluate_system_impact.py",  # 评估系统影响
    ],
    "测试脚本": [
        "test_complete_flow.py",  # 完整流程测试（刚创建的）
    ]
}

def analyze_file(filepath):
    """分析文件的基本信息"""
    stat = os.stat(filepath)
    size = stat.st_size
    mtime = stat.st_mtime

    # 读取文件头部注释
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            docstring = ""
            in_docstring = False
            for line in lines[:20]:  # 只读前20行
                if '"""' in line or "'''" in line:
                    if not in_docstring:
                        in_docstring = True
                        docstring = line.strip()
                    else:
                        docstring += " " + line.strip()
                        break
                elif in_docstring:
                    docstring += " " + line.strip()
    except:
        docstring = "无法读取"

    return {
        "size": size,
        "size_kb": f"{size/1024:.1f}KB",
        "mtime": mtime,
        "docstring": docstring
    }

def main():
    ml_engine_dir = Path(__file__).parent

    print("="*80)
    print("ML Engine 文件清理分析")
    print("="*80)

    all_files = set()

    # 分析每个分类
    for category, files in FILE_CATEGORIES.items():
        print(f"\n【{category}】")
        print("-" * 80)

        for filename in files:
            filepath = ml_engine_dir / filename
            if filepath.exists():
                info = analyze_file(filepath)
                all_files.add(filename)

                print(f"\n  文件: {filename}")
                print(f"  大小: {info['size_kb']}")
                print(f"  说明: {info['docstring'][:100]}")

                # 对于非核心文件，给出建议
                if category != "核心模块":
                    if category == "初始化脚本":
                        print(f"  建议: ⚠️  保留（但只需运行一次）")
                    elif category == "维护脚本":
                        print(f"  建议: 🗑️  可以删除（如果不再需要维护）或移到 scripts/ 目录")
                    elif category == "分析工具":
                        print(f"  建议: 🗑️  可以删除（临时分析工具）或移到 scripts/ 目录")
                    elif category == "测试脚本":
                        print(f"  建议: ✅ 保留（用于测试）或移到 tests/ 目录")
            else:
                print(f"\n  文件: {filename} (不存在)")

    # 查找未分类的文件
    print(f"\n{'='*80}")
    print("未分类的文件")
    print("-" * 80)

    uncategorized = []
    for file in ml_engine_dir.glob("*.py"):
        if file.name not in all_files and file.name != "analyze_cleanup.py":
            uncategorized.append(file.name)
            info = analyze_file(file)
            print(f"\n  文件: {file.name}")
            print(f"  大小: {info['size_kb']}")
            print(f"  说明: {info['docstring'][:100]}")

    # 总结建议
    print(f"\n{'='*80}")
    print("清理建议总结")
    print("="*80)

    print("\n✅ 必须保留的核心模块:")
    for f in FILE_CATEGORIES["核心模块"]:
        print(f"   - {f}")

    print("\n⚠️  可以保留但不常用:")
    print("   - init_execution_tables.py (数据库初始化，只需运行一次)")
    print("   - test_complete_flow.py (测试脚本，建议移到 tests/ 目录)")

    print("\n🗑️  建议删除或归档到 scripts/ 目录:")
    for category in ["维护脚本", "分析工具"]:
        if category in FILE_CATEGORIES:
            for f in FILE_CATEGORIES[category]:
                print(f"   - {f}")

    print("\n💡 建议操作:")
    print("   1. 创建 scripts/ 目录用于存放维护和分析脚本")
    print("   2. 将维护脚本和分析工具移到 scripts/ 目录")
    print("   3. 保留核心模块和测试脚本在 ml_engine/ 目录")
    print("   4. 如果确定不再需要某些脚本，可以直接删除")

    print("\n" + "="*80)

    # 生成清理命令
    print("\n如果确定要删除，可以执行以下命令:")
    print("-" * 80)
    print("\n# 创建 scripts 目录（如果不存在）")
    print("mkdir -p scripts")

    print("\n# 移动维护脚本到 scripts 目录")
    for f in FILE_CATEGORIES.get("维护脚本", []) + FILE_CATEGORIES.get("分析工具", []):
        print(f"mv ml_engine/{f} scripts/")

    print("\n# 或者直接删除（谨慎！）")
    for f in FILE_CATEGORIES.get("维护脚本", []) + FILE_CATEGORIES.get("分析工具", []):
        print(f"# rm ml_engine/{f}")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
