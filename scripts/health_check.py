#!/usr/bin/env python3
"""
Bitfinex 借贷利率预测程序 - 每日健康检查

检查项：
1. systemd 服务是否 active
2. 当前进程是否持有调度器锁（pipeline 循环是否在跑）
3. 最近一次 pipeline 是否在合理时间内跑过（防止调度器卡死）
4. 最近一轮 pipeline 的各步骤是否有错误（下载/重训/预测）
5. 最近的预测是否成功（optimal_combination.json 是否更新）

输出：打印到 stdout（供 cron 邮件/日志），同时追加到 log/health_check.log
退出码：0=健康，1=发现问题
"""
import os
import sys
import json
import subprocess
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "log")
APP_LOG = os.path.join(LOG_DIR, "ml_optimizer.log")
HEALTH_LOG = os.path.join(LOG_DIR, "health_check.log")
DATA_DIR = os.path.join(BASE_DIR, "data")
SERVICE_NAME = "optimize-api"

# 健康阈值
PIPELINE_STALE_HOURS = 5      # pipeline 超过 5h 没跑视为调度器卡死（正常 2h 一轮）
PREDICTION_STALE_HOURS = 12   # optimal_combination.json 超过 12h 未更新视为预测异常


def log(msg, level="INFO"):
    """打印并追加到健康检查日志。"""
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{level}] {msg}"
    print(line)
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(HEALTH_LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def run_cmd(cmd):
    """运行 shell 命令，返回 (stdout, returncode)。"""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return r.stdout.strip(), r.returncode
    except Exception as e:
        return f"<cmd failed: {e}>", -1


def check_service():
    """1. 检查 systemd 服务状态。"""
    out, rc = run_cmd(["systemctl", "--user", "is-active", SERVICE_NAME])
    if rc == 0 and out == "active":
        log(f"✅ 服务 {SERVICE_NAME} 正常运行 (active)")
        return True
    log(f"❌ 服务 {SERVICE_NAME} 异常: is-active={out} (rc={rc})", "ERROR")
    return False


def check_scheduler_lock():
    """2. 检查当前是否有进程持有调度器锁。"""
    lock_path = os.path.join(BASE_DIR, ".scheduler.lock")
    if not os.path.exists(lock_path):
        log(f"⚠️  锁文件 {lock_path} 不存在（首次启动前正常）", "WARN")
        return True
    # 用 fcntl 非阻塞尝试：能拿到说明无进程持锁（异常，因为服务应在运行）
    code = (
        "import fcntl, sys\n"
        f"fd = open('{lock_path}', 'w')\n"
        "try:\n"
        "    fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
        "    fcntl.lockf(fd, fcntl.LOCK_UN)\n"
        "    print('UNLOCKED')\n"
        "except (IOError, OSError):\n"
        "    print('LOCKED')\n"
        "fd.close()\n"
    )
    py = os.path.join("/home/bumblebee/anaconda3/envs/optimize/bin/python")
    if not os.path.exists(py):
        py = sys.executable
    out, _ = run_cmd([py, "-c", code])
    if "LOCKED" in out:
        log("✅ 调度器锁被持有（pipeline 循环应正在运行）")
        return True
    log("⚠️  调度器锁未被持有 —— 服务可能在运行但未启动调度器循环", "WARN")
    return False


def _tail_lines(path, n=500):
    """读取日志最后 n 行（容错）。"""
    try:
        with open(path, "rb") as f:
            data = f.read()
        text = data.decode("utf-8", errors="replace")
        return text.splitlines()[-n:]
    except Exception:
        return []


def check_pipeline_freshness():
    """3. 检查最近一次 pipeline 触发时间是否在合理窗口内。"""
    lines = _tail_lines(APP_LOG, 1000)
    last_trigger = None
    for line in reversed(lines):
        if "Scheduled pipeline trigger" in line or "Starting CLOSED-LOOP Optimization Pipeline" in line:
            # 提取行首时间戳 "2026-06-15 10:53:56.764"
            try:
                ts_str = line.split(" | ")[0]
                last_trigger = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
            except Exception:
                pass
            break
    if last_trigger is None:
        log("⚠️  日志中未找到任何 pipeline 触发记录", "WARN")
        return False
    age = datetime.now() - last_trigger
    if age > timedelta(hours=PIPELINE_STALE_HOURS):
        log(f"❌ pipeline 已 {age} 未触发（超过 {PIPELINE_STALE_HOURS}h），调度器可能卡死", "ERROR")
        return False
    log(f"✅ 最近 pipeline 触发于 {last_trigger}（{age} 前）")
    return True


def check_recent_errors():
    """4. 检查最近一轮 pipeline 各步骤的错误。"""
    lines = _tail_lines(APP_LOG, 1500)
    # 找最近一轮 pipeline 的起点
    start_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if "Starting CLOSED-LOOP Optimization Pipeline" in lines[i]:
            start_idx = i
            break
    if start_idx is None:
        log("⚠️  日志中未找到完整 pipeline 记录", "WARN")
        return True
    recent = lines[start_idx:]

    issues = []
    for line in recent:
        # 关键错误信号
        if "Prediction failed" in line and ("exit code -1" in line or "exit code 1" in line):
            issues.append(f"预测失败: {line.split(' | ')[-1][:120]}")
        elif "Retraining failed (exit code" in line or "Retraining timed out" in line:
            issues.append(f"重训失败/超时: {line.split(' | ')[-1][:120]}")
        elif "TIMEOUT:" in line and "exceeded" in line:
            issues.append(f"子进程超时: {line.split(' | ')[-1][:120]}")
        elif "❌ Data download failed and DB data is stale" in line:
            issues.append("下载失败且 DB 数据 stale")

    if issues:
        log(f"❌ 最近一轮 pipeline 发现 {len(issues)} 个问题:", "ERROR")
        for iss in issues[:5]:
            log(f"     - {iss}", "ERROR")
        return False
    log("✅ 最近一轮 pipeline 无关键错误")
    return True


def check_prediction_output():
    """5. 检查预测输出文件是否近期更新。"""
    combo_path = os.path.join(DATA_DIR, "optimal_combination.json")
    if not os.path.exists(combo_path):
        log(f"⚠️  预测输出 {combo_path} 不存在", "WARN")
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(combo_path))
    age = datetime.now() - mtime
    if age > timedelta(hours=PREDICTION_STALE_HOURS):
        log(f"❌ 预测输出 {age} 未更新（超过 {PREDICTION_STALE_HOURS}h）", "ERROR")
        return False
    log(f"✅ 预测输出最近更新于 {mtime}（{age} 前）")
    return True


def main():
    log("=" * 60)
    log(f"Bitfinex 预测程序健康检查 @ {datetime.now()}")
    log("=" * 60)

    results = [
        ("服务状态", check_service()),
        ("调度器锁", check_scheduler_lock()),
        ("Pipeline 时效", check_pipeline_freshness()),
        ("近期错误", check_recent_errors()),
        ("预测输出", check_prediction_output()),
    ]

    log("-" * 60)
    all_ok = True
    for name, ok in results:
        status = "✅" if ok else "❌"
        log(f"{status} {name}")
        if not ok:
            all_ok = False

    log("=" * 60)
    if all_ok:
        log("🎉 总体健康")
        log("=" * 60)
        return 0
    else:
        log("⚠️  发现问题，需人工介入检查（见上方详情）", "ERROR")
        log("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
