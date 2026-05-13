# Day 7 Daily Log — NQS Sub-Project Phase A Closure + Plan C Launch

**Date**: 2026-05-12 (calendar 2026-05-12; Day 4-7 NQS sub-project Phase A completion)
**Phase**: Week 1, Day 7 (Phase 2.5 forensic + Option β fix + Option X verify + Plan C decision + Phase A retrospective)
**Codebase**: `/home/dglg/miao_workplace/tcbm_nqs/`
**GPU**: 4× NVIDIA A10 24GB (HKUST-GZ lab server)
**Commits**: Day 7 bundle expected
**Tests**: 31 passing (unchanged from Day 6)
**Scope**: This daily_log is structured as a Phase A retrospective rather than chronological log, because:
  1. Schedule constraint released by advisor (Prof. Hui Xiong)
  2. Plan C (NetKet+TCBM integration) starts Day 8, lessons must transfer
  3. Day 4-7 5-day arc forms a coherent "self-built framework attempt + fail + diagnosis" 
     narrative; lessons are valuable as a unit for Phase B planning

---

## 1. Phase A 完整 narrative — Day 4 → Day 7

NQS sub-project Phase A 从 Day 4 启动, Day 7 收尾, 共 4 天集中 work + 5 天 calendar gap (Day 5→Day 6). 整体 narrative 是 "self-built TCBM-NQS framework attempt + 多 failure mode 诊断 + 用 NetKet 重做的决策".

### 1.1 Day 4: NaN bug discovery + Phase 1/2 fix
- 启动 production v2 run, step 87 撞 NaN
- 三阶段诊断 (narrow_band + probe_grad_nan + inline anomaly check) 定位 `core/j1j2_problem.py:228` `log(inner)` 在 `inner=0` 处给 -inf, backward 1/0 → NaN
- 同时发现 atexit defect (deep CUDA call 时 Python atexit chain bypass)
- Phase 1 fix: `core/robustness.py` 3-layer 保护 (periodic checkpoint + anomaly detection + signal handlers)
- Phase 2 fix: `_SafeLog2CoshComplex` custom autograd Function (forward 保持 analytic, backward NaN/inf → 0)
- Day 4 末: 两 fix 实施 + 31 tests pass, Phase 3 (production v2 retry) 推到 Day 5

### 1.2 Day 5: Production v2 retry → R-ABORT-1
- Phase 3 launch production v2, 跑 15.66h 完成
- final rel_err 98.85%, best_cost_raw -114.84 (物理不可能, ED truth = -8.46)
- Phase 2 NaN fix verified ✓ (0 NaN recurrence), Phase 1 atexit fix verified ✓ (15.66h 跑完)
- 但**新 pathology**: step 800 cost_min 突然跳 -3.67 → -10.68, frozen 1900 steps
- 当时 hypothesis: 1/ψ underflow (Phase 2 fix scope 只 covers inner=0 singularity, 但 inner ≈ 0 但还没到 0 的 underflow region 仍让下游 1/ψ blow up)
- Day 5 closing: daily_log + commit + Path A (Phase 2.5) plan, Day 5→Day 6 calendar 5-day gap (Nick mental reset 时间)

### 1.3 Day 6 上午: Path δ Layer A bug + κ1 fix
- Fresh start, 准备 Phase 2.5 probe
- 发现 Phase 1 Layer A latent bug: `_state` dict 只有 scalars/lists, 0 tensor, Layer A periodic checkpoint 写 0 个 .pt files (Day 5 production v2 跑完 0 saved theta)
- Path δ design: 加 `update_capture_with_optimizer_state` helper, caller 调 helper 把 theta + best_x 加进 dict
- Path δ Step 1 实施 → mock-based pytest 通过, 但 production run 用 TCBMOptimizer 时 `hasattr(optimizer, 'x')` silent return False (TCBMOptimizer 没 .x attr, replica state 是 local var `positions`)
- Path κ1: core/tcbm_optimizer_NQS.py 加 self._latest_positions + self._best_positions instance attrs (2-line surgical, Category H Nick approved)
- pytest 改成 real-TCBMOptimizer integration test (M=2/n_steps=2/n_vmc=50, 308s wall, catches silent fail pattern)

### 1.4 Day 6 中午: probe-run #1 reproduces Day 5
- Launch probe-run with κ1 fix, seed=42, n_steps=1100
- 6.03h wall, byte-identical reproduces Day 5 step 800 anomaly (cost_min -10.6769, swap_acc cliff 0.727→0.364)
- 但 probe-run #1 没 saved theta (κ1 fix 加入时 probe-run 已 import 旧 module, 不 pick up patch)
- Day 5 final results 被 probe-run 覆盖 (FINAL_PATH 漏 env var), git checkout HEAD 恢复
- hypothesis C STRONG verdict from swap_acc + cost trajectory analysis

### 1.5 Day 6 下午-晚上: 4-variant sweep
- 决策: 不只 fix 单 variant, 4 个 cfg variant 并行测试
- Variant A (T_min=0.005, T_max=1.0, ratio=200): 单调降 T_max
- Variant B (T_min=0.05, T_max=2.0, ratio=40): 单调提 T_min
- Variant C (T_min=0.05, T_max=1.0, ratio=20): 双调 (中等紧)
- Variant D (T_min=0.1, T_max=1.0, ratio=10): 激进双调
- 实施 TCBM_T_MIN / TCBM_T_MAX env var support + physical sanity gate
- 4-variant launch GPU 0-3 parallel, ~15h wall each

### 1.6 Day 7 早上: 4-variant sweep verdict + forensic
- 4-variant sweep 全 R-ABORT-1:
  - A: rel_err 81.59%, cost_min frozen -2.69 (stalled)
  - B: rel_err 84.72%, best_cost_raw -8.04e+8 (catastrophic blowup)
  - C: rel_err 117.73%, best_cost_raw -2.66e+18 (worst catastrophic)
  - D: rel_err 91.53%, cost_min frozen -1.92 (stalled, learned even less than A)
- Hypothesis C (PT swap pathology) revised: 是 visible symptom, 不是 dominant root cause
- Plan B trigger line approached (Day 7 16:00 UTC)
- 决策 Path γ: forensic + parallel SR implementation prep

### 1.7 Day 7 forensic: Hypothesis A 实证 confirmed for B/C
- experiments/forensic_log_psi_diff.py: 24 .pt snapshots × 1000 sample/replica × 12 replicas = ~288K forward
- 0.6s wall actual (GPU batched 极快)
- Variant A/D: 0 outliers (max|log_psi_diff| ≤ 16, healthy)
- Variant B step 1500+: 20%+ outliers, max|x|=14672 (mathematical infinity)
- Variant C atexit: 8% outliers, max|x|=1485
- → Hypothesis A confirmed for B/C 类 catastrophic blowup
- → A/D stall is **separate pathology** (no numerical issue, still fails)

### 1.8 Day 7 中午: Phase 2.5 Option β fix + verify
- Claude Code 4 flag 推 spec design 更严谨 (per-bond vs per-sample, normalization, scope, tracking)
- Option β chosen: per-sample filter via safe_mask, _local_energy_batch returns tuple
- Threshold 30 first try
- Verify on Variant B step 2000 saved theta: 3 replicas 100% unsafe (NaN), 但 filtered E_loc 仍 1e+4 ~ 1e+10 magnitude
- Replica 4 critical: 0 unsafe samples but E_loc=-13582 → cumulative blowup over 64 bonds (单 bond ratio 都 "safe" 但 sum 爆)
- Threshold tightened 30 → 15

### 1.9 Day 7 下午: Option X verify + Plan C decision
- Option X: load Variant A step 500 healthy theta, threshold=15, 100 step short trajectory
- 30.1 min wall, 18 s/step
- Result: cost_min frozen at -2.77 全程 100 步, 0 cliff events, unsafe_pct stable 1.17%
- → Numerical fix WORK (no blowup, no cliff)
- → Optimization stall PERSISTS (Mode 2 separate pathology)
- Plan C 决策: 用 NetKet 重新实现 TCBM, schedule 延到 6-8 weeks, 投稿目标 Day ~55
- Advisor (Prof. Hui Xiong) 同意延后投稿

---

## 2. Two Failure Modes — 最重要的 Phase A 发现

Phase A 最大科学发现: **TCBM-NQS 在 4×4 J1-J2 上有 TWO distinct failure modes, fix 一个不能 cover 另一个**.

### 2.1 Mode 1: Numerical blowup (1/ψ underflow)

**Mechanism**:
- RBM 参数演化到边缘区域, 某些 spin configuration σ 的 ψ(σ) 接近 0 (numerical underflow, but not exact 0)
- VMC local energy estimator: `E_loc = ⟨H|ψ⟩/ψ`, contains `1/ψ` factor
- 1/ψ in underflow region → exp(+30 to +14000) magnitude (mathematically infinity)
- 64-bond accumulation of these terms → E_loc reaches ±1e+18 magnitude
- best_cost_raw 报告 outlier "ground state" energy (-114, -1e+8, -1e+18)
- PT swap 看到 outlier "ground state", swap_acc behavior 异常

**Observed in**:
- Day 5 production v2 (final best_cost_raw = -114.84)
- Day 6 Variant B (best_cost_raw = -8.04e+8, step 1500+ outliers)
- Day 6 Variant C (best_cost_raw = -2.66e+18, atexit outliers)
- Forensic data: Variant B step 1500+ has 20% samples with |log_psi_diff| > 30

**Fixed by Phase 2.5 Option β**:
- per-sample safe_mask (any bond with |log_psi_diff.real| > UNSAFE_THRESHOLD → sample unsafe)
- threshold 15 (forensic-justified, ~3× healthy margin over A/D max ≈ 12)
- _evaluate_vmc filters E_loc[safe_mask] before mean
- Verified on Variant B step 2000: catches 3 replicas at 100% unsafe (NaN signal, correct response)

**But fix has limitation**:
- Threshold tightening is cosmetic, not mechanistic
- Cumulative blowup over 64 bonds can produce unphysical E_loc even when every single bond is "safe"
- Replica 4 example: 0 unsafe samples, E_loc=-13582
- → True fix layer is mechanistic (eliminate 1/ψ from estimator via logsumexp etc.) — this is NetKet's approach

### 2.2 Mode 2: Optimization stall (separate pathology)

**Observed in**:
- Day 6 Variant A (cost_min frozen -2.69 entire run)
- Day 6 Variant D (cost_min frozen -1.92 entire run)
- Day 7 Option X 100-step run (cost_min frozen -2.77, byte-identical 100 steps)

**Characteristics**:
- No numerical pathology (no unsafe samples, max|log_psi_diff| ≤ 16 healthy)
- Optimizer makes ZERO progress (cost truly frozen, not slow improvement)
- swap_acc shows behavior consistent with healthy PT (0.36-0.45 range, near target)
- Subspace mechanism (psi_star=1.5) never activates (psi_max ~1.04 narrow band stuck)
- 与 numerical fix 完全无关

**Candidate causes (待 Plan C 诊断)**:
1. **RBM expressivity insufficient**: α=2 (1120 params) too small for 4×4 J1-J2 frustrated landscape
2. **TCBM subspace mechanism dormant**: psi_star=1.5 trigger threshold too high, never reaches activation
3. **cfg space unexplored**: lr / noise schedule / k / n_vmc_samples 还有 untested region
4. **PT temperature ladder structure**: 即使 numerical-safe cfg, 12 replicas 之间 communication 不够
5. **Algorithm fundamental**: TCBM PT+clamping mechanism 在 frustrated landscape 上需要特定 condition trigger, 当前 cfg 没满足

**NOT fixed by Phase 2.5**:
- Numerical safeguard 不 touch optimization landscape exploration
- Mode 2 在 Plan A path 下 unsolved

### 2.3 Mode 1 vs Mode 2 — 为什么两者都重要

Plan A path 必须解决两者才能 Tier 1 PASS:
- 只 fix Mode 1: 没有 blowup, 但 cost frozen → R-ABORT-1 (rel_err > 15%)
- 只 fix Mode 2: 没有 stall, 但仍 blowup → R-ABORT-1 (rel_err > 15%)

**Phase 2.5 只 covers Mode 1**. Mode 2 在 self-built framework 下需要 algorithm-level work, 1-2 weeks effort 还可能 fail.

**Plan C (NetKet) 的价值**:
- Mode 1 在 NetKet framework 下自动消失 (logsumexp + 内置 numerical stability)
- Mode 2 仍然存在, 但**可以 isolate 出来研究**, 不被 Mode 1 干扰
- Plan C 让我们专注 Mode 2 这个真正的科学问题

---

## 3. 关键诊断数据 — Phase A complete evidence chain

### 3.1 Day 5 production v2 (R-ABORT-1)

| Metric | Value |
|--------|-------|
| Total wall | 15.66h (3000 steps) |
| Phase 2 NaN bug | not triggered ✓ |
| Phase 1 atexit | fired cleanly ✓ |
| rel_err_debiased | 98.85% (R-ABORT-1) |
| best_cost_raw | -114.84 |
| best_cost_debiased | -0.0975 |
| swap_acc mean | 0.689 (target [0.15, 0.35]) |
| T_w trigger | never (psi_max max ~1.04, threshold 1.5) |
| Step 800 cliff | cost_min -3.67 → -10.68 (Δ=-7.01), frozen 1900 steps |

### 3.2 Day 6 probe-run #1 (Day 5 reproduction)

| Metric | Day 5 | Day 6 probe-run |
|--------|-------|------------------|
| step 0 cost_min | 10.9770 | 10.9770 ✓ byte-identical |
| step 800 cost_min | -10.6769 | -10.6769 ✓ byte-identical |
| swap_acc cliff at 800 | 0.727→0.364 | 0.727→0.364 ✓ byte-identical |
| best_cost_raw | -114.84 | -114.84 ✓ byte-identical |
| Wall | 15.66h (3000) | 6.03h (1100) |

**Deterministic confirm**: bug 给 seed=42 + cfg 完全确定性, 排除 CUDA race / floating-point non-determinism / hardware glitch.

### 3.3 Day 6 4-variant sweep verdicts (all R-ABORT-1)

| V | T_min | T_max | ratio | rel_err | best_raw | best_deb | swap_acc | failure mode |
|---|-------|-------|-------|---------|----------|----------|----------|--------------|
| ref Day 5 | 0.005 | 2.0 | 400 | 98.85% | -114.84 | -0.0975 | 0.689 | Mode 1 |
| **A** | 0.005 | 1.0 | 200 | 81.59% | -15.70 | -1.5567 | 0.442 | Mode 2 stalled |
| **B** | 0.05 | 2.0 | 40 | 84.72% | -8.04e+8 | -1.2926 | 0.742 | Mode 1 blowup |
| **C** | 0.05 | 1.0 | 20 | 117.73% | -2.66e+18 | +1.5000 | 0.774 | Mode 1 worst blowup |
| **D** | 0.1 | 1.0 | 10 | 91.53% | -2.33 | -0.7162 | 0.811 | Mode 2 stalled |

**Cross-variant pattern**:
| | T_max=2.0 | T_max=1.0 |
|---|---|---|
| T_min=0.005 | Day 5 Mode 1 | A Mode 2 |
| T_min=0.05 | B Mode 1 | C Mode 1 (worst) |
| T_min=0.1 | (not run) | D Mode 2 |

**Insight**: cfg space 没有 simple "lower T_max" 或 "raise T_min" 关系. T_max=1.0 + T_min=0.05 (Variant C, 中等紧 ladder) 反而 worst case.

### 3.4 Day 7 forensic (24 snapshots, |log_psi_diff.real| > 30 count out of 12000)

| variant | step500 | step1000 | step1500 | step2000 | step2500 | atexit |
|---------|---------|----------|----------|----------|----------|--------|
| **A** | 0 (max=11.5) | 0 (11.2) | 0 (10.8) | 0 (12.1) | 0 (12.0) | 0 (12.0) |
| **B** | 0 (19.4) | 1 (32.9) | **2424 (13644)** | **2413 (14672)** | **2402 (13241)** | **2458 (13464)** |
| **C** | 0 (15.6) | 0 (22.8) | 0 (21.5) | 0 (24.2) | 0 (24.9) | **945 (1485)** |
| **D** | 0 (11.4) | 0 (11.9) | 0 (11.7) | 0 (11.8) | 0 (15.7) | 0 (11.8) |

### 3.5 Day 7 Phase 2.5 Option β verify on Variant B step 2000

| Replica | n_unsafe / 1000 | E_safe (filtered) | E_all (unfiltered) |
|---------|-----------------|--------------------|---------------------|
| 0, 1, 11 | 1000 (100%) | NaN | NaN |
| 10 | 444 (44%) | -4.3e+10 | +2.0e+17 |
| 8 | 120 (12%) | -5.0e+09 | -5.2e+13 |
| **4** | **0 (0%)** | **-13582** | **-13582** |
| 2,3,5,6,7,9 | 0-15 (< 2%) | 1e+5 ~ 1e+9 | similar |

**Replica 4 是 critical insight**: 0 unsafe but cumulative blowup over 64 bonds → 单 threshold 不能 fully fix Mode 1.

### 3.6 Day 7 Option X (threshold 15, A step 500 healthy start, 100 steps)

| Metric | Value |
|--------|-------|
| Wall | 30.1 min (18.0 s/step) |
| Steps | 100/100 ✓ |
| final cost_min | -2.7686 (frozen) |
| max\|cost_min\| | 2.77 (physical) |
| cliff events | 0 ✓ |
| cost_below_-10 ever | False ✓ |
| final unsafe_pct | 1.17% (stable) |
| swap_acc range | 0.36-0.45 (Variant A pattern) |

**Verdict**: Numerical fix ✓, Optimization stall persists → Mode 2 separate.

---

## 4. Phase A LESSONS — 10 条 (transfer 到 Plan C)

这些是 Plan C 启动前必须 internalize 的教训. 每条标 [适用 Phase] 让 Phase B implementation 时直接照查.

### 4.1 [Architecture] Self-built framework attempt 工程成本 >> 科学价值

**Observation**: Day 4-7 共 ~50 hours active engineering, ~30 hours wall (runs), 5 个 failure mode 解决 1 个 (Mode 1 partial).

**Cost breakdown** (rough estimate):
- Phase 1 atexit robustness: ~3h design + 4h implementation + 1h test
- Phase 2 NaN fix: ~2h diagnostic + 2h implementation + 1h test
- Phase 2.5 Option β: ~2h spec + 1h impl + 1h verify
- Path δ Layer A + κ1: ~2h diagnostic + 2h impl + 1h test
- 4-variant sweep: ~1h design + 0.5h impl + 60h wall (cheap CPU)
- Forensic + Option X: ~1h design + 0.6s + 30min runs
- Daily logs: ~3h × 4 days = 12h
- Bug discoveries lost time: ~5-8h (sleep / mental reset over 5-day gap)

**Total ~40-50h engineering, 1 fix delivered, 4 still open**.

**Lesson**: For NQS framework, "build it yourself" is **always more expensive than community-validated alternative** (NetKet). Self-build is justified only if:
- We're contributing framework-level innovation (we're not; we're using standard RBM + VMC + Metropolis)
- No suitable alternative exists (NetKet exists)
- Required customization impossible in alternative (TCBM driver IS customizable in NetKet)

**Plan C application**: 
- 不再写任何 framework-level code (lattice, Hamiltonian, sampler, autograd)
- 只在 NetKet 框架内 implement TCBM-specific code
- 工程量预估 ~5-7 days vs 当前 attempt 的 ~40-50h still incomplete

### 4.2 [Methodology] Bug fix 必须区分 layer: symptomatic vs mechanistic vs algorithmic

**Observation**: Day 4 NaN fix 是 symptomatic (catch the NaN). Day 5 R-ABORT 暴露 fix 不够, 因为 underflow region 不在 Day 4 fix scope. Day 7 Phase 2.5 fix 又是 symptomatic (mask unsafe samples). 当前 Replica 4 cumulative blowup 表明 single-threshold mask 仍不够.

**Three layers**:
- **Symptomatic** (mask bad samples / catch NaN): fixes visible failures, root cause untouched
- **Mechanistic** (reformulate computation to avoid 1/ψ): fixes computation form
- **Algorithmic** (prevent theta walk to corrupted region): fixes upstream cause

**Phase A fix approach**: Symptomatic only. Every layer revealed deeper layer.

**Plan C application**:
- NetKet's logsumexp + complex-valued ψ handling is **mechanistic layer** built-in
- Don't try to symptomatic-fix NetKet (it doesn't need)
- Focus our work on **algorithmic layer**: prevent TCBM dynamics walking to bad theta region

### 4.3 [Diagnosis] Forensic 数据 must inform fix, not the other way

**Observation**: Day 5 R-ABORT 我们 hypothesize Hypothesis A (1/ψ underflow). Day 6 probe-run (without saved theta) gave swap_acc trajectory only, generated WEAK confirm of Hypothesis C, but missed Mode 2 split.

Day 7 forensic (with saved theta) gave **direct distribution data** for log_psi_diff, immediately revealed:
- Hypothesis A confirmed for B/C only
- A/D have no outliers but still fail → Mode 2 exists

**Lesson**: Hypothesis-driven debugging without distribution data is dangerous. The right diagnostic sequence:
1. **Save raw data** (theta snapshots, gradient norms, sample distributions)
2. **Generate distributions** on saved data (forensic-style scripts)
3. **Then** formulate hypothesis from distribution shape
4. Verify hypothesis by targeted experiment

**Phase A inverted this**: hypothesized first, ran fixes blind, then forensic showed hypothesis was incomplete.

**Plan C application**:
- Every NetKet+TCBM production run saves theta snapshots by default
- Forensic-style distribution analysis is **first diagnostic step**, not last
- Don't formulate hypothesis until you have distribution data

### 4.4 [Discipline] Mock-based unit test 不能替代 integration test

**Observation**: Day 6 Path δ Step 1 helper had `hasattr(optimizer, 'x')` silent fail pattern. Mock-based pytest passed; real TCBMOptimizer integration broke. Caused κ1 fix detour, ~1.5h lost.

**Lesson**: 任何 helper function 涉及 external object API, 必须 ≥ 1 real-object integration test, 即使 slow (Day 6 κ1 integration test 308s wall). 

**Plan C application**:
- TCBM-NetKet driver tests use **real NetKet vstate**, not mocks
- Even if test takes minutes, must verify against real API
- Mock test answers "function logic correct" (necessary), integration test answers "function works with real system" (sufficient)

### 4.5 [Process] Spec design must read existing code before writing prescriptions

**Observation**: Phase 2.5 fix spec had 4 errors caught by Claude Code flags:
- Flag 1: spec assumed _local_energy_batch averages internally (it doesn't)
- Flag 2: spec didn't address per-bond vs per-sample mask choice
- Flag 3: spec didn't acknowledge Mode 2 separate pathology in scope
- Flag 4: spec ambiguous on instance attr vs per-call return

All 4 caused by spec author (me/Claude AI) grep-ing keywords but not viewing function bodies.

**Lesson**: Spec design phase requires:
- `view` actual function/class body before prescribing changes
- Trace function signature + entry/exit + key transformations
- 不能 "想当然" external API exists or has specific shape

**Plan C application**:
- Before writing TCBM-NetKet driver design, must `view` NetKet's VMC driver source
- Trace NetKet's gradient flow, callback API, vstate interface
- 不能 grep keyword + 凭印象 prescribe integration

### 4.6 [Schedule] Constraint should not dictate fix scope

**Observation**: Day 5-7 多次 decisions 是 "what fits in remaining hours before Plan B trigger" 而非 "what is the right fix". 例如:
- Day 7 上午 forensic 之后, "trigger Plan B at 16:00" 强制选 Option X 30-min verify 而非 deeper diagnostic
- Day 6 sweep 4 variants 决策受 "Day 7 16:00 trigger" 时间窗约束

If schedule had been more relaxed Day 5 onwards, fix sequence likely different:
- Day 5 R-ABORT → 立即 forensic + theta capture infrastructure (Path δ before Phase 2.5)
- Day 6 forensic data complete first, then design fix from distribution
- Day 7 fix implementation properly mechanistic

**Lesson**: When fix quality 取决于 schedule hours, 而 hours 取决于 deadline, 这是 **misaligned constraint chain**. If deadline can be negotiated (as Phase A → Plan C transition), do it before scope decisions, not after.

**Plan C application**:
- Plan C 不强求 Day 21, 投稿目标 Day ~55
- Each Phase 决策 prioritize correctness + impact over speed
- If new bug discovered Day 30, can take 1 week to diagnose properly, not 30 minutes

### 4.7 [Delegation] Standing policy 显著 reduce 来回成本

**Observation**: Standing delegation policy (Category A/B/C/D autonomous, E/F/G/H ask) 在 Day 6-7 显著 reduce decision overhead.

Examples of autonomous decisions that worked well:
- Path Y env var pattern (Day 6, Category B): minimal touch, backward-compat default
- κ1 integration test parameter tweak (Day 6, Category I): M=2/n_steps=2 vs spec M=4/n_steps=10, autonomous + reported
- tmux orphan cleanup (Day 6 + Day 7, Category A): pure plumbing

Examples of correct STOP+ask escalations:
- Step 4.1 launch path (Day 6): spec's two fallbacks both infeasible, Claude Code STOP, Nick decided Path Y
- Layer A bug discovery (Day 6): Layer A fix failed, Claude Code STOP per standing instruction
- Phase 2.5 spec 4 flags (Day 7): Claude Code STOP rather than silently proceed with multiple errors

**Lesson**: Explicit standing policy + categorized authority levels works. Mock test passing then silent integration fail (Day 6 hasattr) shows that **standing policy must include integration test requirement**, not just unit test.

**Plan C application**:
- Continue standing policy
- Add explicit rule: Category H (touching algorithm code) requires integration test before Nick approval, not just pytest

### 4.8 [Data integrity] Saved evidence is paper SI primary source

**Observation**: Day 5 final results 被 probe-run 覆盖 (FINAL_PATH 漏 env var), recovered via git checkout HEAD. Day 6 R-ABORT-1 evidence (62 files, 1.8 MB) committed to git ensures Day 21 paper SI 有 primary source.

**Lesson**: All evidence files (logs, JSONs, theta snapshots) must be in git, not just on disk. Path/name conflict 是 silent corruption risk.

**Plan C application**:
- NetKet runs save to clearly-named output directories
- All evidence force-add to git immediately after run
- File path conflicts impossible by design (use timestamp + run_id naming)

### 4.9 [Communication] Daily log 是 Future Self 的 institutional memory

**Observation**: Day 5 → Day 6 5-day calendar gap, mental anchor lost. Day 6 fresh start 100% rely on daily_log_day5.md re-read.

Day 6 daily_log 提到 hypothesis priorities (~30%/25%/20%/15%/10%) as informal estimate. Day 7 forensic 后这些数字调整 (~A confirmed for B/C, Mode 2 separate). 

Without daily_log capturing both initial estimate + later revision, **reasoning evolution lost**. Reviewer 看 Day 21 paper SI 可能 question "your decision sequence wasn't documented".

**Lesson**: Daily logs must capture:
- Decisions + rationale at time of decision
- Initial estimates + later revisions
- Caveat / informal-estimate markers ("based on intuition, will refine with data")

**Plan C application**:
- Daily log discipline continues
- Plan C Phase 1/2/3/4/5 each has end-of-phase summary log
- Decision evolution explicit (e.g. "Day 30 hypothesized X, Day 35 data confirmed/refuted")

### 4.10 [Strategy] Schedule constraint release 让 path 选择本质改变

**Observation**: Day 5-7 我们 trapped in Plan A vs Plan B binary:
- Plan A: 修自建 TCBM-NQS, success ~15-25%
- Plan B: 放弃 + NetKet baseline, narrative weak

But under no schedule constraint, **Plan C (NetKet+TCBM integration)** emerges:
- Success ~70-80%
- Full 3-domain narrative
- 6-8 weeks vs 2 weeks
- Long-term PhD value (NetKet skill, follow-up papers)

Plan C 之前不可见因为 schedule lens 把它 filter out.

**Lesson**: When stuck in binary choice with bad options, **question the constraint that created the binary**. Sometimes constraint is real (hardware, IRB), sometimes negotiable (deadline). Negotiate when possible.

**Plan C application**:
- Plan C 解锁因为 advisor agreed to schedule extension
- Phase B 中如果遇到类似 trapped binary, question constraints first

---

## 5. 红线状态 (Phase A 结束)

| Trigger | Day 5 | Day 6 | Day 7 | Phase A Final Status |
|---------|-------|-------|-------|----------------------|
| R-abort-1 (rel_err > 15%) | 98.85% | 4 variants 81-117% | Option X stall persists | **TRIGGERED 3 times, Plan C reset** |
| R-abort-2 (swap_acc < 0.1 or > 0.9) | 0.689 MARGINAL | 0.811 NEAR BREACH | 0.36-0.45 (Option X healthy) | resolved (in Plan A path) |
| R-abort-3a (σ_TCBM/σ_Adam > 0.8) | pending | pending | pending | Plan C deferred |
| R-abort-3b (σ_TCBM/σ_SR > 0.8) | pending | pending | pending | Plan C deferred |
| R-abort-5 (n_kept < 3) | pending | pending | pending | Plan C deferred |
| Plan B trigger line (Day 7 16:00) | -- | -- | RELEASED (advisor approval) | Plan C activated |

**Phase A R-abort triple-trigger** indicates self-built TCBM-NQS framework not viable for paper-quality data. Plan C reset all metrics; will re-evaluate under NetKet+TCBM framework starting Day 8.

---

## 6. Plan C 详细 schedule + 风险评估

### 6.1 Plan C overview

**Goal**: NetKet 框架内实现 TCBM optimizer, 在 4×4 J1-J2 上达到 Tier 1 PASS (rel_err < 10%) + 完整 SR/Adam baseline 对比 + 多 problem class extension.

**Timeline**: 6-8 weeks (Day 8 → Day ~55-60), 投稿 NMI 目标 Day ~55-60.

**Acceptance criteria**:
- Tier 1: best TCBM rel_err < 10% on 4×4 J1-J2 (matches Plan A target)
- Tier 2: rel_err_TCBM / rel_err_SR ≤ 2.0 (NetKet's SR baseline as reference)
- Tier 3a/3b: σ ratios statistical (15 seeds × 4 methods sweep)
- Tier 4: TCBM mechanism evidence (subspace activation, PT swap statistics)

### 6.2 Phase 1: NetKet learning + standard SR baseline (Day 8-13, ~1 week)

**Goals**:
- Install NetKet + verify on lab hardware
- Reproduce 4×4 J1-J2 SR baseline (target rel_err ≤ 10⁻²; Bukov 2021 reference ~10⁻³)
- Learn NetKet API for Phase 2 (TCBM integration)

**Specific tasks**:
- pip install netket jax optax
- 跑 NetKet tutorial notebooks
- 写 4×4 J1-J2 SR baseline script (~50 lines)
- Single seed verify run (~2h GPU)
- 5-10 seeds sweep for initial statistics

**Risks** (informal estimate):
- Low: NetKet works on our hardware (~95% probability)
- Medium: 4×4 J1-J2 SR baseline reaches ~10⁻³ (Bukov 2021 reference)
- Low: API learning curve manageable in 1 week

**Exit condition**: SR baseline rel_err < 5% confirmed on 4×4 J1-J2, NetKet API understood enough to design TCBM integration.

### 6.3 Phase 2: TCBM-NetKet integration design + implementation (Day 14-25, ~2 weeks)

**Goals**:
- Design TCBM as a NetKet "driver" (replacing VMC's SGD step with TCBM step)
- Implement TCBM driver in NetKet ecosystem
- Verify with pytest + 2×2 trivial case

**Specific tasks**:
- View NetKet's VMC driver source (`netket.driver.VMC`)
- Design TCBM driver class with:
  - 12 replica PT swap mechanism
  - Subspace clamping (psi_max-triggered)
  - Langevin step with noise schedule
  - Compatibility with NetKet's vstate, sampler, hamiltonian interfaces
- Implement TCBM driver in Python (use NetKet's JAX-based gradient + Hamiltonian)
- pytest:
  - 2×2 trivial J1-J2 (J1=1, J2=0) ED 对照, TCBM and SR should both reach ground state
  - 4×4 J1-J2 TCBM vs SR single-seed run, verify TCBM doesn't fail Mode 1 (numerical) thanks to NetKet's stable backend
  - QGT-related sanity if TCBM uses gradient information

**Risks**:
- Medium-Low: NetKet JAX vs our PyTorch TCBM impl, need re-implementation of TCBM logic in JAX
- Low: NetKet's driver API flexible enough to accommodate TCBM (it's designed for this)
- Medium: TCBM behavior in NetKet might be slightly different (RNG path different, sampler subtleties)

**Exit condition**: TCBM driver in NetKet runs without numerical errors on 4×4 J1-J2, single seed reaches at least cost_min < -5 (better than Plan A best -2.77).

### 6.4 Phase 3: TCBM-NetKet 4×4 J1-J2 production + statistics (Day 26-35, ~1.5 weeks)

**Goals**:
- Single seed TCBM-NetKet production run (target rel_err < 10%)
- Multi-seed (15 seeds) sweep for statistical claim
- Compare TCBM vs SR vs Adam (3 methods × 15 seeds = 45 runs)

**Specific tasks**:
- Single seed run, 3000-5000 steps (~3-5h GPU)
- Diagnose if Tier 1 NOT reached (likely Mode 2 still present)
  - If Mode 2 persists: investigate RBM expressivity (try α=4, 8), TCBM cfg tune
  - If Mode 2 resolved: proceed to sweep
- Multi-seed sweep on 4 GPUs parallel (~10-15h)
- Compute σ_TCBM, σ_SR, σ_Adam, ratios

**Risks** (highest in Plan C):
- High: Mode 2 (optimization stall) might persist even in NetKet (since it's algorithm not framework issue)
  - If stall persists: this is **a real scientific result** - "TCBM mechanism not suitable for 4×4 J1-J2 frustrated landscape"
  - Honest paper: narrow claim "TCBM works in DC-OPF, CEC2017, hardware HIL; NQS analysis identifies algorithm limitation in frustrated landscape, future work"
- Medium: Tier 3a/3b ratios might fail (σ_TCBM > 0.8 × σ_SR)

**Exit condition**: 
- Best case: Tier 1 PASS, statistical claims valid → full paper
- Middle: Tier 1 MARGINAL (10-15%), narrow paper
- Worst: Tier 1 fail with mechanistic insight → "TCBM is for these problem classes, NOT this one" paper

### 6.5 Phase 4: Multi-problem class extension (optional, Day 36-45, ~1.5 weeks)

**Conditional on Phase 3 results**:
- If Phase 3 4×4 J1-J2 PASS: extend to 6×6 J1-J2 (target Bukov 2021 scale comparison)
- If Phase 3 MARGINAL: try 2D Heisenberg (no frustration, easier landscape) to confirm TCBM works without J2
- If Phase 3 FAIL: don't extend, accept narrow claim

**Risk**: Time burn. Manage by hard deadline UTC Day 45.

### 6.6 Phase 5: Paper writing + internal review (Day 46-55, ~2 weeks)

**Goals**:
- Complete NQS chapter in main paper
- 3-domain narrative integration (DC-OPF + CEC2017 + NQS)
- SI completion (Day 4-7 Phase A retrospective as SI evidence)
- Internal review with Prof. Hui Xiong + collaborators

**Specific tasks**:
- Section: TCBM-NQS introduction + 4×4 J1-J2 setup
- Section: Phase A diagnostic methodology (theta forensic, hypothesis layered analysis) → contribution
- Section: Phase B NetKet+TCBM implementation + results
- Section: comparison with SR/Adam baselines
- Figures: cost trajectory, swap_acc evolution, distribution analyses
- SI: full Phase A bug discoveries + Mode 1/2 split

**Risk**: Writing scope creep. Manage by Phase 4 hard deadline + figure budget.

### 6.7 Phase 6: Submission polish + submit (Day 56-60, ~1 week)

- Methods section finalize
- SI peer-readable
- Cover letter draft
- NMI submission

### 6.8 Total Plan C schedule
Phase 1 (Day 8-13):  NetKet learning + SR baseline
Phase 2 (Day 14-25): TCBM-NetKet integration + pytest
Phase 3 (Day 26-35): 4×4 J1-J2 production + statistics
Phase 4 (Day 36-45): optional multi-problem extension
Phase 5 (Day 46-55): paper writing + internal review
Phase 6 (Day 56-60): polish + submit
Investment: 7-8 weeks calendar, ~30-40 hours/week effective
NMI submission target: ~Day 55-60 (vs original Day 21)
Buffer: ~1 week between Phase 5 finish + actual submission

---

## 7. Day 8 immediate next steps

### 7.1 Day 8 早上 (Phase 1 launch)

1. `pip install netket jax optax`
2. Verify install: run a NetKet "Getting started" example
3. Read NetKet's 4×4 J1-J2 example (community-provided)
4. 估 ~3-4h hands-on with NetKet API

### 7.2 Day 8 下午

1. 写 4×4 J1-J2 SR baseline script (`experiments/netket_sr_baseline_4x4_j1j2.py`)
2. Single seed verify run (~2h GPU on cuda:0)
3. Compare result with Bukov 2021 reference (target rel_err ~ 10⁻³)

### 7.3 Day 8 末

1. Day 8 daily_log: Phase 1 launch summary, NetKet learnings, SR baseline initial results
2. Commit Day 7 bundle (this daily_log + Day 7 evidence) + Day 8 NetKet scripts

---

## 8. Claude Code 自主决定汇总 (Day 7)

**Category A** (pure plumbing):
1. _init_replicas monkey-patch in Option X (follow optimize_from pattern)
2. tmux session cleanup post-runs

**Category B** (backward-compat infrastructure):
1. reset_unsafe_count() method added to J1J2Problem

**Category G** (interpretation, asked Nick):
1. Forensic verdict interpretation (Hypothesis A confirmed for B/C)
2. Phase 2.5 fix design (Option β chosen over α, threshold 30 then 15)
3. Option X verdict interpretation (Mode 2 stall identified)
4. Plan A vs B vs C decision (Plan C chosen, schedule constraint released)

**Category H** (Nick explicit approval):
1. Phase 2.5 numerical safeguard scope (Option β, _local_energy_batch + _evaluate_vmc + _evaluate_exact)

**STOPs + ask 触发**:
1. Pre-step forensic flagged 4 spec errors (normalization, per-bond vs per-sample, scope, tracking)
2. Daily_log Step 1 placeholder issue: Claude Code STOP per fabrication-free, Nick repaste final
3. Option X: TCBMOptimizer API doesn't accept init_positions, autonomous monkey-patch (Category A)

---

## 9. Day 7 commit handoff

**待 commit (Day 7 bundle)**:

| File / Pattern | Count | Size | Note |
|---|---|---|---|
| `docs/daily_log_day7.md` | 1 | - | new (this file) |
| `core/j1j2_problem.py` | 1 modified | - | Phase 2.5 Option β + threshold 15 + reset_unsafe_count |
| `experiments/forensic_log_psi_diff.py` | 1 new | ~6 KB | Day 7 forensic script |
| `experiments/verify_safeguard_with_saved_theta.py` | 1 new | ~3 KB | Day 7 verify Variant B step 2000 |
| `experiments/verify_option_x_short_trajectory.py` | 1 new | ~5 KB | Day 7 Option X 100-step verify |
| `results/forensic_log_psi_diff.json` | 1 new | 103 KB | per-(variant, ckpt, replica) stats |
| `results/safeguard_verify.json` | 1 new | ~3 KB | Phase 2.5 fix verify |
| `results/option_x_verify.json` | 1 new | ~20 KB | Option X trajectory metadata |
| `logs/forensic_20260513_0843.log` | 1 new | ~3 KB | forensic stdout |
| `logs/safeguard_verify_20260513_0947.log` | 1 new | ~3 KB | safeguard verify stdout |
| `logs/option_x_verify_20260513_0956.log` | 1 new | ~5 KB | Option X stdout |

**TOTAL**: 1 daily_log + 10 evidence files ≈ **~150 KB**

**Day 8 早上第一件事**: commit + push 上述, then 启动 Plan C Phase 1 (NetKet install + tutorial).

---

**End of Day 7 daily_log — Phase A closure, Plan C launch ready.**
