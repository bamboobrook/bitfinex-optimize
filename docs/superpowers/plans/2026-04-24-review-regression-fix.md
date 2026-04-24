# Review Regression Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复评审指出的重训练健康 no-op 退出码误报、C3 anchor 排序覆盖收益优先级问题，并验证训练数据 C3 回归批次。

**Architecture:** 保持现有调度器接口不扩实体：`run()` 继续表达“是否部署新模型”，CLI `main()` 单独区分健康 no-op 与失败。C3 beam 排序只调整 hard key 顺序，把 `anchor_backed` 降为收益、成交、周期、fUSD 之后的同档 tie-breaker。

**Tech Stack:** Python 3、pytest、现有 `ml_engine` 模块、后台 API supervisor 脚本。

---

## File Structure

- Modify: `ml_engine/retraining_scheduler.py` — 调整 CLI 退出码，no-op 返回 0，真实训练/部署失败返回 1。
- Modify: `ml_engine/c3_combo_optimizer.py` — 调整 beam hard-sort key，保持“收益 > 成交 > 长周期 > fUSD”，anchor 仅同档优先。
- Modify: `ml_engine/training_data_builder.py` — 恢复订单级 C3 闭环训练样本，避免 dense market 左表导致执行结果 0 匹配。
- Modify: `tests/test_retraining_exit_status.py` — 增加 no-op CLI 回归测试。
- Modify: `tests/test_c3_combo_search.py` — 增加 anchor 不可越过更高收益桶的回归测试，并更新旧 anchor 测试语义。
- Create: `docs/superpowers/plans/2026-04-24-review-regression-fix.md` — 本计划。

### Task 1: Retraining CLI Exit Semantics

**Files:**
- Modify: `tests/test_retraining_exit_status.py`
- Modify: `ml_engine/retraining_scheduler.py`

- [ ] **Step 1: Write the failing no-op test**

Add a subprocess test that patches `RetrainingScheduler.should_retrain()` to return `(False, None)` and `RetrainingScheduler.run()` to return `False`, then runs `python -m ml_engine.retraining_scheduler` without `--force`.

Expected assertion:

```python
assert result.returncode == 0
```

- [ ] **Step 2: Run the no-op test to verify RED**

Run: `pytest tests/test_retraining_exit_status.py::test_retraining_main_returns_zero_when_no_retraining_needed -q`

Expected: FAIL because current CLI maps `run(False)` to exit code 1.

- [ ] **Step 3: Implement minimal CLI status split**

In `main()`, before `scheduler.run(force=args.force)`, call `scheduler.should_retrain()` when not forced. If false, print the healthy no-op message and exit 0. If true or forced, call `scheduler.run(...)` and keep `0 if ok else 1`.

- [ ] **Step 4: Run targeted retraining tests**

Run: `pytest tests/test_retraining_exit_status.py -q`

Expected: PASS, including existing force-failure test returning non-zero.

### Task 2: C3 Anchor Tie-Breaker Ordering

**Files:**
- Modify: `tests/test_c3_combo_search.py`
- Modify: `ml_engine/c3_combo_optimizer.py`

- [ ] **Step 1: Write the failing priority test**

Add a test where only five pairs can be selected, one anchor-backed low-EV pair competes with one non-anchor-backed higher-EV pair in a higher revenue bucket.

Expected assertion:

```python
assert ("fUST", 14) in combo_keys
assert ("fUSD", 14) not in combo_keys
```

- [ ] **Step 2: Run the C3 test to verify RED**

Run: `pytest tests/test_c3_combo_search.py::test_choose_combo_beam_keeps_revenue_bucket_ahead_of_anchor_backed_status -q`

Expected: FAIL because `anchor_backed` is currently first in the sort key.

- [ ] **Step 3: Move anchor after fUSD tie-breaker**

Change `candidate_key` order from:

```python
(anchor_backed, revenue_bucket, fill_bucket, tenor, currency_priority, ...)
```

to:

```python
(revenue_bucket, fill_bucket, tenor, currency_priority, anchor_backed, ...)
```

- [ ] **Step 4: Run targeted C3 tests**

Run: `pytest tests/test_c3_combo_search.py -q`

Expected: PASS, with anchor-backed candidates still preferred only within the same priority tier.

### Task 3: Regression Batch And Runtime Validation

**Files:**
- No source changes unless tests expose directly related failures.

- [ ] **Step 1: Run review regression batch**

Run: `pytest tests/test_retraining_exit_status.py tests/test_c3_combo_search.py tests/test_training_data_builder_c3.py -q`

Expected: PASS or report exact unrelated failures if remaining failures pre-exist outside this fix.

- [ ] **Step 2: Restart backend API**

Run: `pkill -f ml_engine/api_server.py || true` then `nohup scripts/run_api_server.sh > /tmp/optimize-api.log 2>&1 &`.

Expected: API restarts in the background on port 5000.

- [ ] **Step 3: Verify API health and C3 effect**

Run: `curl -sS http://127.0.0.1:5000/status`.

Expected: status endpoint responds successfully after restart.

### Task 4: Commit And Sync

**Files:**
- Commit only reviewed fix files and this plan document.

- [ ] **Step 1: Review diff**

Run: `git diff -- ml_engine/retraining_scheduler.py ml_engine/c3_combo_optimizer.py ml_engine/training_data_builder.py tests/test_retraining_exit_status.py tests/test_c3_combo_search.py docs/superpowers/plans/2026-04-24-review-regression-fix.md`

Expected: diff contains only plan, tests, and two logic fixes.

- [ ] **Step 2: Commit with required log content**

Run:

```bash
git add ml_engine/retraining_scheduler.py ml_engine/c3_combo_optimizer.py ml_engine/training_data_builder.py tests/test_retraining_exit_status.py tests/test_c3_combo_search.py docs/superpowers/plans/2026-04-24-review-regression-fix.md
git commit -m "fix: repair retraining exit and c3 priority" -m "问题描述: no-op retraining was reported as failure; anchor_backed outranked higher C3 revenue buckets; C3 training-data builder produced zero execution matches. 修复思路: split CLI no-op exit handling, demote anchor_backed to same-tier tie-breaker after revenue/fill/tenor/fUSD, and build order-level closed-loop training rows from execution results."
```

- [ ] **Step 3: Push current branch**

Run: `git push`

Expected: remote branch receives the commit.

---

## Self-Review

- Spec coverage: covers two P2 review comments, targeted tests, reported C3 training-data batch, API restart, remote sync.
- Placeholder scan: no TBD/TODO placeholders; every command and expected behavior is explicit.
- Type consistency: uses existing `RetrainingScheduler`, `choose_combo_beam`, `RateCandidate`, and pytest patterns.
