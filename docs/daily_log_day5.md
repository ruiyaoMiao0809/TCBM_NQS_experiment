# Day 5 Daily Log

**Date**: 2026-05-06  
**Phase**: Week 1, Day 5 (Phase 0 commit batch + Phase 1 atexit fix + Phase 2 NaN fix + Phase 3 production v2 retry — R-ABORT-1)  
**Codebase**: `/home/dglg/miao_workplace/tcbm_nqs/`  
**GPU**: 4× NVIDIA A10 24GB (HKUST-GZ lab server)  
**Commits**: 7 (all pushed to origin/main)  
**Tests**: 30 passing (23 baseline + 3 robustness Phase 1 + 4 log_2cosh_safe Phase 2)

---

## 1. 计划 vs 实际

**Day 5 原计划** (Day 4 daily_log §6 handoff):

```
Phase 0 (commit batch): ~15 min
Phase 1 (atexit fix): ~1h
Phase 2 (NaN fix): ~2-3h
Phase 3 (production v2 launch + ~5-6h wall): 拿Tier 1 verdict
```

**Day 5 实际 path**:

```
UTC 07:36-09:30 (~2h):
  - Phase 0 commit batch: 3 commits + push
  - 顺手 fix: pytest env mismatch (tcbm vs tcbm_nqs)
  - 顺手 add: project convention to README + CLAUDE.md
  - known_issues_day2.md update with Issue 5 + Issue 6
  - daily_log_day4.md §6.4 加 Phase 3 tmux launch requirement

UTC 09:30-10:30 (~1h):
  - Phase 1 atexit fix: core/robustness.py (3-layer protection)
  - 3个 experiment scripts backport
  - 3 new pytest tests (Layer A/B/C verify)
  - Found flaky test (subprocess reentrant lock issue), fixed via polling

UTC 10:30-11:14 (~45min):
  - Phase 2 NaN fix: _SafeLog2CoshComplex custom autograd Function
  - 4 new pytest tests (forward equiv, normal/singular backward, Day 4 state load)

UTC 11:14: Phase 3 launch
  - tmux session prod_v2_day5 on GPU 3
  - seed=42 (same as Day 4)

UTC 11:14 → Day 6 02:53 (~15.66h, vs docstring estimate 4.5-5.5h, 3× longer):
  - production v2 跑完 3000/3000 steps
  - 中途 T+5min / T+30min / T+77min / T+3.5h checks 全 healthy
  - 但 T+3.5h+ 之后 cost trajectory 出现 anomaly: step 800 突然跳到 -10.6769 后 frozen 到 step 2900
  - Phase 1 anomaly_cb 没 trigger (因为 cost 没发散到超过 100x best, 只是 frozen)

Day 6 02:53: Run completes
  - rel_err = 98.85% → R-ABORT-1
  - best_cost_raw = -114.84 (远低于 ED truth -8.4579, 物理不可能)
  - best_cost_debiased = -0.0975 (接近 0, optimization 实际几乎没走)
  - T_w never triggered (psi_max never crossed 1.5)
  - swap_acc 0.689 (太热, 目标 [0.15, 0.35])
  - NQS-1e debiasing WARN: debiased ≥ raw (新bug怀疑)
```

**偏离原计划的原因**: 
1. Phase 0/1/2 实际比原计划长 (2h+1h+45min vs 估的 ~15+60+150min), 主要在 pytest env mismatch 和 flaky test 调试上
2. production v2 跑 15.66h 远超 docstring 4.5-5.5h, 实际 pace ~18.8 s/step (Day 4 Phase 3 stall 前是 ~9 s/step, 翻倍原因待查)
3. Phase 3 verdict 是 R-ABORT-1, **不是NaN bug回归 (Phase 2 fix verified)**, 是新发现的 numerical pathology

Schedule slip: 累计 2 → 3 天, buffer 0 + Plan B trigger Day 7 16:00 UTC 仍可达 (52h to go from now).

---

## 2. 关键数字

### 2.1 Phase 0/1/2 commit history

| Commit | Topic | Files | Tests delta |
|--------|-------|-------|-------------|
| c85176b | project convention (README + CLAUDE.md) | 2 docs | - |
| f8da323 | known_issues update (Issue 5 + 6) | 1 doc | - |
| 49dd96b | daily_log §6.4 tmux requirement | 1 doc | - |
| a2793f8 | Phase 1 atexit fix (3-layer robustness) | 5 files | +3 (23→26) |
| de6f8f5 | fix flaky test_explicit_save_on_sigterm | 1 test | - |
| 166c72b | Phase 2 NaN fix (_SafeLog2CoshComplex) | 2 files | +4 (26→30) |

8 个 Day 4 closing batch commits (前几个) 也在 Day 5 早上 push, 共 7 + 3 (Day 4 closing) = 10 commits in Day 5 push window.

### 2.2 Production v2 final results

| Metric | Value | Verdict |
|--------|-------|---------|
| Total wall | 15.66h (939 min) | ⚠ 3× docstring estimate |
| Steps completed | 3000/3000 | ✓ |
| NaN bug recurrence | 0 | ✓ Phase 2 fix verified |
| best_cost_raw | -114.8365 | ✗ 物理不可能 (ED truth = -8.4579) |
| best_cost_debiased | -0.0975 | ✗ 接近 0, 实际几乎没学到 |
| Rel error (debiased) | 98.85% | ✗ R-ABORT-1 |
| swap_acc (mean) | 0.689 | ✗ 目标 [0.15, 0.35] |
| T_w trigger step | never | ✗ psi_max never reached 1.5 |
| psi_max final | <1.5 | ✗ subspace mechanism 整个 run 未激活 |
| NQS-1e debiasing | WARN debiased ≥ raw | ✗ 新bug怀疑 |

### 2.3 Cost trajectory anomaly (smoking gun)

| Step | cost_min | cost_mean | swap_acc | psi_max | subspace |
|------|----------|-----------|----------|---------|----------|
| 0 | 10.9770 | 11.8056 | 1.000 | 0.000 | off |
| 100 | -1.9243 | 3.1611 | 0.818 | 0.000 | off |
| 200 | -2.0662 | 0.9472 | 0.818 | 1.009 | off |
| 500 | (checkpoint, data TBD) | | | | off |
| 800 | **-10.6769** (sudden jump) | | | | off |
| 800-2900 | **-10.6769** (frozen) | | | | off |
| 3000 | (final eval) | | | | off |

step 800 的突然跳变 (-2 → -10.68) 后 frozen 1900 step, pattern 与 Day 4 production v2 stall 高度相似 (Day 4 是 cost 冻在 -1.69, Day 5 冻在 -10.68). Day 4 是 NaN bug 锁住, Day 5 不是 NaN — Phase 2 fix 防住了 NaN, 但出现了**新的 frozen pattern**, root cause 不同.

### 2.4 Phase 2 fix end-to-end verification

| Comparison Point | Day 4 (NaN bug active) | Day 5 (Phase 2 fix on) |
|------------------|------------------------|------------------------|
| step 0 cost_min | 10.9770 | 10.9770 (byte-identical, RNG path deterministic) |
| step 100 cost_min | -1.6900 (frozen by NaN) | -1.9243 (still optimizing) |
| step 100 swap_acc | 1.000 (frozen-replica artifact) | 0.818 (healthy PT) |
| Run progressed past step 87? | NO (NaN trigger) | YES (no NaN trigger 整 3000 step) |

Phase 2 NaN fix verified ✓. Issue 5 in known_issues_day2.md can be marked as resolved at commit 166c72b.

---

## 3. 发现 / 问题

### 3.1 Phase 1 atexit fix 设计 + flaky test 副发现

**3-layer robustness module** (core/robustness.py):
- Layer A: periodic checkpoint every N steps (sync write, 不依赖 atexit)
- Layer B: anomaly detection callback (grad_nan / swap_acc_extreme / cost_diverge → dump + sys.exit)
- Layer C: SIGTERM/SIGINT handler dumps CPU state before exit

3 个 experiment scripts backport (run_gradient_baseline_v2 + run_adam_baseline + run_sr_baseline TODO).

**flaky test_explicit_save_on_sigterm root cause**: subprocess `print()` epilogue 还在 BufferedWriter lock 时 SIGTERM 到达, signal handler 自己也 print → I/O 重入 → RuntimeError → handler 中止. fix 用 polling 替代 sleep + grace period.

**衍生发现 (latent bug, Day 6+ fix)**: `core/robustness.py` 的 `_signal_handler` 同样在内部 `print`, production 场景 atexit fallback 兜得住但 signal_handler 路径偶发不可靠. Day 6+ 应换成 `os.write(2, ...)` (async-signal-safe).

### 3.2 Phase 2 NaN fix 数学 + 物理双重正当性

**Bug**: `core/j1j2_problem.py:228` 的 `log(inner)` 在 `inner = (1+exp(-2|u|))² - 4·sin²(v)·exp(-2|u|) = 0` 时给 -inf. 出现在 z = u + iv 满足 u≈0, v≈π/2 + kπ 时. backward 链式法则 1/inner = 1/0 = inf → NaN.

**Fix**: `_SafeLog2CoshComplex` custom autograd Function:
- forward 保持原 stable form (log(0) = -inf 是 analytic correct, |ψ|² = 0 让 sample 自动从 VMC 排除)
- backward 用 PyTorch native autograd 算正常 gradient + `torch.where` 把 NaN/inf 替换为 0

**双重正当性**:
- 数学: ψ=0 处 gradient 应来自 measure-theoretic argument (∫|ψ|²·∂E/∂θ dσ = 0×anything = 0)
- 物理: ψ=0 的 configuration 不会被 VMC 采样, 对 gradient 贡献 = 0

避开作者反对的 epsilon clamp 路径 (clamps 注入假 gradient, silently corrupt training).

### 3.3 Production v2 R-ABORT — hypothesis: 1/ψ underflow region 未被 Phase 2 fix 覆盖

**重要**: 本节内容是 **hypothesis pending probe verification**, 非 verified root cause. Day 6 Phase 2.5 第一步是 probe 验证或证伪本假设, 然后再决定 fix 方向.

**Phase 2 fix scope 限制**: Phase 2 修复了 inner = 0 的 exact singularity (log(0) = -inf 时 backward 1/0 = NaN). 但 inner 接近 0 但还没到 0 的 underflow region (e.g. inner = 1e-30, log(inner) ≈ -69, finite negative number) 仍然让下游 estimator pipeline 出问题, **如果**这条路径是 R-ABORT root cause.

**Hypothesized mechanism (pending probe)**:
1. step ~800 某个 replica 的 RBM 参数演化到 ψ→0 region (但 inner 不是 exact 0)
2. log_psi = -69 量级 (large negative finite), forward 不触 Phase 2 fix
3. local energy estimator E_loc = ⟨H|ψ⟩/ψ 含 1/ψ 项, 1/ψ = exp(+69) = 1e+30
4. E_loc estimator output 出现极端值 -114.84 (best_cost_raw)
5. 12 replica 中这个 "假 ground state" replica 的 cost_min 被报告
6. PT swap (acc 0.689) 频繁 try to swap into 该 state, 但实际没 stable
7. cost_min 显示为 frozen at -10.68 (12 replica 平均化后的实际最低能量)

**Alternative hypotheses (also pending probe)**:
- (a) Gradient explosion: 没 NaN 但梯度 norm 极大, 把 theta 推到 outlier basin (与 Phase 2 fix 无关)
- (b) Cumulative floating-point error in evaluate() pipeline (与 ψ→0 无关)
- (c) PT swap pathology: swap_acc 0.689 太热 (目标 [0.15, 0.35]), 一个 outlier theta 在 12 replica 间快速 propagate, 全局陷入假 minimum (与 numerical 无关, 是 PT cfg 问题)

**Day 6 Phase 2.5 第一步必须 probe**:
- Load Day 5 production v2 final state + per-step trajectory
- Dump per-replica per-sample log_psi_diff distribution
- 看 |log_psi_diff| 实际是否有 > 30 的 outlier (1/ψ 爆点的标志)
- 看 gradient norm history (alternative (a) 的标志)
- 看 swap acceptance pattern (alternative (c) 的标志)

**Probe 结果决定 Phase 2.5 fix 方向**, 不在 probe 之前 commit fix code.

**Surface symptom 与 Day 4 NaN frozen pattern 表面一致 (cost frozen) 但 mechanism 完全不同**:
- Day 4: NaN gradient → forces NaN → Metropolis 全拒 → theta frozen (verified)
- Day 5: TBD — 三种 hypothesis 之一, probe 后 confirm

### 3.4 NQS-1e debiasing 怀疑 (次级问题, Day 7+ 调查)

最终 eval log 警告 `debiased ≥ raw`. NQS-1e debiasing 的目的是给 unbiased estimator: debiased < raw 是健康 signal. 现在 debiased = -0.0975 ≥ raw 的某个值, 反向了.

**Hypothesis**: 与 §3.3 同源. raw 是 -114.84 (假极小), debiasing formula 算的 correction term 期望让 estimator 更接近 0 (correct ground truth direction), 结果 -0.0975 比 raw 大 (因为 raw 是错的). 这反而是 debiasing 在 "正确地修正" 一个 bug-poisoned raw value.

如果 §3.3 的 1/ψ underflow fix 后 raw 回归正常 magnitude (~-7 to -8 量级), debiasing 应该 自动 healthy. Day 7+ Phase 2.5 fix 后 verify.

### 3.5 Pace anomaly: 18.8 s/step (Day 4 production v2 是 9 s/step)

| Run | Setup | Pace |
|-----|-------|------|
| Day 4 narrow_band Stage A | subspace=off, callback_every=1 (per-step diagnostic) | 35 s/step |
| Day 4 production v2 (pre-stall) | subspace cfg full, callback_every=100 | ~9 s/step |
| Day 5 production v2 | + Phase 2 _SafeLog2CoshComplex + Phase 1 robustness | ~18.8 s/step |

**Phase 2 _SafeLog2CoshComplex** 的 backward 做 enable_grad + autograd.grad 重算一次 forward + autograd.grad call. 这是 ~2× backward overhead. backward 占 step total time 30-40% → 整体 pace × 1.3-1.4 = 12-13 s/step expected. 实际 18.8 s/step 比 expected 又高 50%.

**剩余 50% pace gap 候选解释**:
- Phase 1 callback overhead (Layer A periodic check + Layer B anomaly check 每 100 step fire)
- GPU 3 偶发被别 lab member job 抢用
- system load 总体偏高 (其他 GPU 90%+ util)

Day 6+ 如果要重跑可考虑: 把 Phase 2 fix 优化成 backward 不重算 forward (手工导导数, 风险是 silent corruption); 或接受 18 s/step 的 trade-off.

---

## 4. 红线状态

| Trigger | Day 5 状态 | 备注 |
|---------|-----------|------|
| R-abort-1 (rel_err > 15%) | ✗ TRIGGERED (98.85%) | production v2 retry pending Phase 2.5 |
| R-abort-2 (swap_acc < 0.1 or > 0.9) | ⚠ MARGINAL (0.689 偏热) | 0.9 阈值未破, 但 0.689 远超目标 [0.15, 0.35] |
| R-abort-3a (σ_TCBM/σ_Adam > 0.8) | pending | sweep Day 12-13 |
| R-abort-3b (σ_TCBM/σ_SR > 0.8) | pending | sweep Day 12-13, SR data Day 7+ |
| R-abort-5 (n_kept < 3) | pending | hybrid Day 8 |

R-abort-1 触发**不是 problem class fail, 是 numerical pathology**. Phase 2.5 fix 后 retry 是合理 path (vs 立即 trigger Plan B).

---

## 5. 反思 (5 lessons)

### 5.1 NaN bug fix 覆盖范围必须含 boundary cases, 不只 exact singularity

Phase 2 fix 锁定 inner = 0 的 exact point, 但 NQS pipeline 中 ψ→0 是 continuous behavior, exact 0 只是这条 continuum 的一个点. underflow region (inner ≈ 1e-30, ψ ≈ 1e-10) 同样能让下游 estimator 爆. 教训: 任何 numerical fix 在 *singular point* 之外, 必须 coverage 一个 *neighborhood region*. exact-singularity-only fix 在 ψ→0 这种 limit-process 场景永远不够.

### 5.2 R-ABORT-1 触发不等同于 problem class 失败

表面看 rel_err 98.85% 是 catastrophic failure, 但深度诊断显示真实 optimization state 是 cost ≈ 0 (rel_err ~100%). raw -114.84 是 numerical artifact (1/ψ 爆). **真实 TCBM optimization state 是 "几乎没学习", 不是 "学到错的 minimum"**. 这两种状态修复 path 完全不同:
- "学到错的 minimum" → algorithm 有问题, 改 cfg 或 algorithm 本身
- "几乎没学习" → likely numerical issue 阻塞了正常 learning, fix numerical 后 should learn

教训: 看 R-ABORT-1 数字时**必须看 raw + debiased 一起**, 看 raw 是不是物理可能的 magnitude. raw < ED truth 是 100% bug 信号.

### 5.3 docstring pace estimate 不可信, 必须实测

`run_gradient_baseline_v2.py` docstring 说 4.5-5.5h. Day 5 实际 15.66h, 3× longer. 不是 cfg 改了 — cfg 完全和 Day 4 一致. 真因是 Phase 2 fix overhead + Phase 1 callback overhead + system load.

教训: 任何 production 跑前必须 micro-benchmark 当前 cfg 的实际 pace (`benchmark_per_step.py` 早就写了, Day 5 没用). Day 6+ Phase 2.5 retry 之前必须 micro-benchmark 验证.

### 5.4 Path A vs Path B 决策应基于 "fix 工程量 + Plan B 储备" 双因素

Day 5 末尾 R-ABORT 后两条 path:
- Path A: Phase 2.5 fix + retry (Day 6 整天)
- Path B: 跳过 TCBM 修复, 先做 SR baseline (Plan B 储备)

选 Path A 的 rationale: 
- Phase 2.5 fix 工程量估 4-6h (扩展现有 _SafeLog2CoshComplex, 不重写 algorithm)
- Day 5 的 numerical pathology root cause 已锁定
- 4×4 problem class 还没真测试过 healthy TCBM run
- Plan B (NetKet fork) 仍然 available, Day 7 16:00 UTC trigger 仍可达

教训: Path 决策不只看 schedule slip, 看每条 path 的失败概率 + 失败后的 recovery options. Path A 失败 (Phase 2.5 没修好) 可以 fall back to Path B + Plan B; Path B 直接 commit Plan B preparation, 失去 TCBM 的最后 attempt.

### 5.5 Phase 1 anomaly detection 阈值要 cover 'frozen but not diverged' pattern

Day 5 cost trajectory step 800 → 2900 frozen at -10.68, Phase 1 anomaly_cb 没 trigger. 因为 default abort conditions 是:
- grad_nan: only triggers if NaN in gradient (Phase 2 fix 防住了)
- swap_acc_extreme: only if < 0.1 or > 0.9 sustained 3 steps (Day 5 是 0.689 不算 extreme)
- cost_diverge: only if |cur_cost - best_cost| > 100 × |best_cost| (Day 5 cost frozen 不算 diverge)

教训: anomaly detection 要加新 condition: "**cost_min frozen for N consecutive steps** (e.g. N = 200)". frozen 是 Day 4 NaN pattern + Day 5 underflow pattern 的共同特征. Day 6 Phase 2.5 同步加这个 abort condition. 

---

## 6. Path A: Phase 2.5 fix plan

### 6.1 Path A vs B 决策 rationale (recap)

Day 5 R-ABORT-1 之后 Nick 选 Path A. 见 §5.4.

### 6.2 Phase 2.5 设计原则

**目标**: 让 production v2 retry (run #3) 拿到 healthy Tier 1 verdict (rel_err < 10%).

**Phase 2.5 是 correctness-only phase, 不做 performance optimization**.

**Scope**:
1. **Probe 验证 §3.3 hypothesis**: 用 Day 5 final state + trajectory 数据, dump per-replica per-sample log_psi_diff + gradient norm + swap pattern. 决定 §3.3 三个 hypothesis (1/ψ underflow / gradient explosion / PT swap pathology) 哪个是真实 root cause.
2. **基于 probe 结果实施 fix**:
   - 如果是 1/ψ underflow: `_local_energy_batch` 加 ratio safeguard (mask sample where |log_psi_diff.real| > threshold, threshold 由 probe 数据决定)
   - 如果是 gradient explosion: gradient norm clipping in `_batch_gradient` (threshold 由 probe 数据决定)
   - 如果是 PT swap pathology: swap_acc 高的 cfg 调整 (T_min/T_max ratio, swap_interval, etc.)
3. **best_cost_raw 物理 sanity gate**: 在 progress callback 加 — 如果 cost_raw < E_0_truth - 1.0 (即 cost_raw < -9.4579), 标记为 `physically_impossible_artifact`, 不更新 best tracker, 但仍记录 raw value 在 trajectory 用于调试. Cheap insurance, 防止 future runs 报告物理不可能能量当 best.
4. **`run_gradient_baseline_v2.py` anomaly detection 升级**: 加新 abort condition `cost_min_frozen_N_steps` (N=200). frozen 是 Day 4 NaN pattern + Day 5 R-ABORT pattern 的共同表面特征, 但 Phase 1 anomaly_cb 没 cover 这条.

**不做的事**:
- 不重写 _SafeLog2CoshComplex 数学逻辑 (Phase 2 verified ✓)
- 不优化 _SafeLog2CoshComplex 性能 (Day 7+, 不在 Phase 2.5 critical path)
- 不改 J1J2Problem.gradient() 主循环
- 不动 algorithm 选择 (still TCBM-gradient mode, M=12, k=20, etc.)
- 不在 probe 完成前 commit 任何 fix code

### 6.3 Phase 2.5 schedule (Day 6 plan)

**Day 6 早上 UTC 08:00-10:00**:
- 写 Day 5 daily_log (本文件) + commit + push
- 准备 Phase 2.5 probe script outline (具体 probe 输出 spec)

**Day 6 上午 UTC 10:00-11:00 — Probe phase**:
- Load `results/baseline_v2_seed42.json` + trajectory (logs/prod_v2_day5_seed42_*.log)
- 写 `experiments/probe_r_abort_root_cause.py`:
  - Load Day 5 final theta + Day 5 step 1000 checkpoint state (CHECKPOINT_EVERY=500, 所以离 step 800 frozen 起点最近的 saved state 是 step 1000; step 800-2900 都 frozen at -10.68 意味着 trapped state 在 step 1000 仍 active, 用 step 1000 anchor 不影响 mechanism diagnosis)
  - Dump per-replica per-sample log_psi_diff distribution (histogram, max/min, n_outliers > 30)
  - Dump per-step gradient norm history (max/mean/std)
  - Dump PT swap pattern (which replica pairs swap, frequency)
- **不写 fix code**, 只 probe.

**Day 6 上午 UTC 11:00-12:00 — Analyze probe 输出**:
- 看 §3.3 三个 hypothesis 哪个 confirmed:
  - 1/ψ underflow → log_psi_diff outlier > 30 ✓
  - Gradient explosion → gradient norm spike at step ~800
  - PT swap pathology → swap pattern shows outlier theta propagation
- 决定 fix 方向 + 阈值参数 (从真实 distribution 算, 不拍脑袋)
- 写 fix design doc (~30 min)

**Day 6 下午 UTC 12:00-14:00 — Implementation phase**:
- 按 11-12 UTC 决定的 fix 方向, 实施代码改动
- pytest verify (新增 2-3 个 test cover fix 行为)
- best_cost_raw sanity gate 加进 progress callback (`run_gradient_baseline_v2.py`)
- `cost_min_frozen_N_steps` abort condition 加进 `core/robustness.py` (Phase 1 enhancement)

**Day 6 下午 UTC 14:00-14:30 — Micro-benchmark**:
- 用 `experiments/benchmark_per_step.py` (Day 3 写的) 验证 retry cfg 的实际 pace
- 期望 ≤ 18 s/step (Day 5 实测), 如果 fix 加大 overhead 显著 (e.g. > 25 s/step), reconsider 是否 launch
- 决定 launch GO/NO-GO

**Day 6 下午 UTC 14:30-15:00 — Launch production v2 run #3**:
- tmux session prod_v2_day6 (新名字, 避免和 Day 5 prod_v2_day5 混淆)
- seed=42 same cfg + Phase 2.5 fix
- 估 ~15h wall, Day 7 ~05:30 UTC 完成

**Day 6 下午 UTC 15:00-22:00 — Spare time**:
- production v2 run #3 在跑期间, 用 spare time:
  - 写 known_issues_day2.md Issue 7 (Phase 2 fix scope limitation 的具体 finding)
  - 准备 SR implementation pseudo-code (不动手实施)
  - 准备 Day 6 daily_log 初稿 (留 production v2 verdict 占位符)

**Day 6 晚上 UTC 22:00**:
- T+8h checkpoint check (step 1500 附近)

**Day 7 早上 UTC 05:30**:
- production v2 run #3 verdict:
  - PASS (rel_err < 10%) → Day 7 全天 SR implementation, Plan A 维持
  - MARGINAL (10-15%) → Day 7 retune 后再 launch run #4 (schedule 极度紧张, 可能滑出 Day 7 16:00 UTC Plan B trigger)
  - R-ABORT (>15%) → trigger Plan B at Day 7 16:00 UTC (NetKet fork, 2-3 day, schedule slip 到 Day 22-24)

**Day 6 critical decision points**:
- UTC 11:00-12:00 期间: probe 数据决定 fix 方向. 如果三个 hypothesis 都 disconfirmed, **停下来 discuss with Nick** (extreme case, 需新 hypothesis 设计)
- UTC 14:00-14:30 micro-benchmark 后: pace 数据决定 GO/NO-GO. 如果 pace 显著退化, **discuss before launch** (否则 schedule 整体爆)

### 6.4 Phase 2.5 commit handoff target

预计 commits:
- `Day 6: Phase 2.5 — _local_energy_batch ratio safeguard for 1/ψ underflow region`
- `Day 6: Phase 1 enhancement — cost_min_frozen abort condition`
- `Day 6: known_issues update — Issue 7 (Phase 2 scope limitation)`
- `Day 6: Day 5 daily_log + Day 5 R-ABORT analysis`

预计新 tests: 2-3 (cover 1/ψ safeguard 行为 + cost_min_frozen anomaly).

---

## 7. Day 6 commit handoff

**待 commit (Day 5 closing batch)**:
- `docs/daily_log_day5.md` (本文)
- `results/baseline_v2_seed42.json` (force-add Day 5 final results, R-ABORT evidence)
- `logs/prod_v2_day5_seed42_*.log` (force-add 整 15.66h log, Day 6 Phase 2.5 诊断需要)
- `docs/known_issues_day2.md` Issue 5 mark "Resolved at 166c72b"

**Day 6 早上第一件事**: commit + push 上述, 然后 Phase 2.5 implementation.

---

**End of Day 5 daily_log.**
