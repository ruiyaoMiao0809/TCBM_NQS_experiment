# Day 6 Daily Log

**Date**: 2026-05-11 (5-day gap after Day 5; calendar 2026-05-06 → 2026-05-11)
**Phase**: Week 1, Day 6 (Phase 2.5 — Path δ Layer A fix + probe-run + 4-variant sweep)
**Codebase**: `/home/dglg/miao_workplace/tcbm_nqs/`
**GPU**: 4× NVIDIA A10 24GB (HKUST-GZ lab server)
**Commits**: 3 (098f906, f6af560, 087d66f; prior Day 5 closing was 877edb4)
**Tests**: 31 passing (30 baseline + 1 new κ1 real-TCBMOptimizer integration test replacing mock test)

---

## 1. 计划 vs 实际

**Day 6 原计划** (Day 5 daily_log §6.3):
UTC 08:00-10:00: Day 5 daily_log commit + Phase 2.5 probe script design
UTC 10:00-11:00: Probe phase (load Day 5 state + run hypothesis A/B/C diagnostic)
UTC 11:00-12:00: Analyze probe → decide fix 方向
UTC 12:00-14:00: Phase 2.5 implementation + pytest
UTC 14:00-14:30: Micro-benchmark
UTC 14:30-15:00: Launch production v2 run #3
Day 7 ~05:30: Run #3 verdict

**Day 6 实际 path**:
UTC 07:30-08:00 (~30 min): Day 6 fresh start (5-day gap, mental anchor 通过 daily_log_day5 重建); Phase 2.5 启动决策 (Path α/β/γ/δ/ε 五选项, Nick 选 Path δ)
UTC 08:00-09:30 (~1.5h): Path δ Step 1 — Patch Layer A

core/robustness.py 新 helper update_capture_with_optimizer_state (+37 lines)
tests/test_robustness.py mock-based test (+34 lines, later replaced)
py_compile + pytest 31/31 pass

UTC 09:30-10:00 (~30 min): Path δ Step 2 — experiment script instrumentation

run_gradient_baseline_v2.py: import helper + callback 调用 (+4 lines)
Sub-option 1 (theta+best_x capture only, 不 touch core)

UTC 10:00-10:45 (~45 min): Step 3 micro-benchmark (13.21 s/step verdict GO);
Step 4: Path Y env var override (按 Claude Code STOP 给的真实 path) — TCBM_N_STEPS + TCBM_RUN_NAME env var
UTC 10:45-12:30 (~1.75h): Step 4 launch probe-run (tmux probe_day6 GPU 3)

seed=42, n_steps=1100, TCBM_RUN_NAME=baseline_v2_probe_seed42
T+5min: byte-identical Day 5 (step 0 cost=10.9770)
Commit 098f906

UTC 09:25 内部 self-monitor 触发 BLOCKER:

Layer A 期望写 step 500 .pt
实际 0 .pt files (Phase 1 设计 + Path δ Step 1 helper 双 bug)
hasattr(optimizer, 'x') silent return False, theta 0 captures

UTC 10:30-12:00 (~1.5h): Path κ1 decision (Nick approved Category H core touch)

core/tcbm_optimizer_NQS.py: 2-line surgical patch (expose _latest_positions, _best_positions)
core/robustness.py helper update (read _latest_positions instead of .x)
tests/test_robustness.py: mock test → real-TCBMOptimizer integration test (~50 lines, 308s wall)
pytest 31/31 pass — integration test catches the silent fail pattern
Commit f6af560
probe-run #1 不 kill, 仍跑完拿 swap_acc trajectory for hypothesis C

UTC 13:38 probe-run #1 completed (6.03h wall):

Day 5 anomaly bit-exact 复现
probe-run final 覆盖 Day 5 final results 文件 (FINAL_PATH 漏 env var)
Recovery: git checkout HEAD → Day 5 restored, probe rename to baseline_v2_probe_seed42.json
FINAL_PATH env var patch
Cleanup CHECKPOINT_PATH dead code + DRY refactor (Category A 自主)
tmux 4 orphan session kill (Day 4-5 残留)

UTC 13:45-14:30 (~45 min): Step 6.1 hypothesis C quick analysis

experiments/analyze_probe_metadata.py (load probe + Day 5 trajectory)
5 metric: swap_acc / cost separation / sigma_E / psi_max / bit-exact
Verdict at the time: hypothesis C STRONG (6 supporting signals, 1 qualifier on hypothesis A)

UTC 14:30-15:30 (~1h): Step 6.2 decision — 4-variant parallel sweep

Nick 提议并行测试 cfg variants
Variant A/B/C/D 覆盖 T ratio 200/40/20/10
GPU 0/1/2/3 全空, launch sweep
TCBM_T_MIN + TCBM_T_MAX env var support
Physical sanity gate (E_0_TRUTH - 1.0 = -9.4579 lower bound)
experiments/run_sweep_day6_variants.sh launch script
py_compile + pytest 31/31 pass

UTC ~15:30 (Day 6) → UTC ~07:00 (Day 7) sweep parallel ~15-15.7h wall each

Commit 087d66f (sweep launch + sanity gate + env var)
Day 7 ~07:00 UTC verdict: 4-variant sweep 全 fail (no Tier 1 PASS)


**偏离原计划的原因**:

1. Path δ 决策: 不只走 fix, 顺手解决 Phase 1 Layer A latent bug 是合理 infrastructure investment
2. Layer A latent bug 发现: Phase 1 设计漏洞 + Path δ Step 1 helper hasattr silent fail. 占 ~1.5h 重新设计 κ1 fix
3. probe-run #1 拿不到 theta: hasattr 漏在 production run 时 silent. Path κ1 fix 后续 run 才能拿
4. 4-variant sweep 代替 single fix retry: Nick 提议, 数据密度更高. 但 4-variant sweep 全 fail, hypothesis C 不是 dominant root cause
5. Day 5 final 文件被 probe-run 覆盖: FINAL_PATH 漏 env var, design bug. git checkout 恢复 + patch

Schedule slip: 累计 3 → 4 天, buffer 0. Plan B trigger line Day 7 16:00 UTC 尚未触发 (Day 7 早上重新评估).

---

## 2. 关键数字

### 2.1 Day 6 commit history

| Commit | Topic | Files | Tests | Auto-decided |
|--------|-------|-------|-------|--------------|
| 877edb4 (Day 5 closing) | daily_log + R-ABORT evidence + Issue 5 resolved | 4 docs/evidence | - | - |
| 098f906 | Path δ Step 1+2 (Layer A helper + instrumentation + env var) | 3 source/test | +1 (31) | Cat. B env var |
| f6af560 | Path κ1 (core touch: expose _latest_positions) | 3 source/test | unchanged (31, real integration) | Cat. H Nick-approved |
| 087d66f | Step 6.2 (4-variant sweep launch + sanity gate + T_min/T_max env var) | 4 source/test/script + analysis evidence | unchanged (31) | Cat. B sanity gate + env var |

### 2.2 probe-run #1 数据 (Day 5 reproduction)

| Metric | Day 5 reference | Day 6 probe-run | Match |
|--------|----------------|------------------|-------|
| step 0 cost_min | 10.9770 | 10.9770 | ✓ byte-identical |
| step 800 cost_min | -10.6769 | -10.6769 | ✓ byte-identical |
| step 800 swap_acc cliff | 0.727 → 0.364 | 0.727 → 0.364 | ✓ byte-identical |
| best_cost_raw | -114.84 | -114.84 | ✓ byte-identical |
| swap_acc 全程 mean | 0.689 | 0.689 | ✓ byte-identical |
| Run wall | 15.66h (3000 steps) | 6.03h (1100 steps) | ✓ pace consistent |

**含义**: Day 5 R-ABORT pathology 是 fully deterministic given seed=42 + cfg. Bug 不是 flaky, 可重复诊断.

### 2.3 4-variant sweep verdicts (4-variant sweep 全 fail)

| Variant | T_min | T_max | T ratio | rel_err_deb | best_cost_raw | best_cost_deb | swap_acc | SANITY GATE | verdict |
|---------|-------|-------|---------|-------------|---------------|---------------|----------|-------------|---------|
| Day 5 ref | 0.005 | 2.0 | 400 | 98.85% | -114.84 | -0.0975 | 0.689 | (pre-fix) | R-ABORT-1 |
| **A** | 0.005 | 1.0 | 200 | **81.59%** | -15.70 | -1.5567 | 0.442 | 0 | R-ABORT-1 (best) |
| **B** | 0.05 | 2.0 | 40 | 84.72% | -8.04×10⁸ | -1.2926 | 0.742 | 22 | R-ABORT-1 (blowup) |
| **C** | 0.05 | 1.0 | 20 | 117.73% | -2.66×10¹⁸ | +1.5000 | 0.774 | 14 | R-ABORT-1 (blowup, worst) |
| **D** | 0.1 | 1.0 | 10 | 91.53% | -2.33 | -0.7162 | 0.811 | 0 | R-ABORT-1 (stalled) |

**含义**: 4-variant sweep 全 fail, 4 种 distinct failure modes — none converge to ground state. T_max=2.0 是 catastrophic blowup 的必要条件 (Day 5 + B). 但 T_max=1.0 + 中等 T_min (C 双调 0.05/1.0) 也 catastrophic. T_max=1.0 + 极端 T_min (A 0.005, D 0.1) 都 stalled. 没有简单的 "lower T_max" 关系 — pathology landscape 复杂.

### 2.4 Phase 1 / Path δ / κ1 verification

| Phase | What | Verified | Evidence |
|-------|------|----------|----------|
| Phase 1 atexit fix | atexit/SIGTERM handlers | ✓ | Day 5 + probe-run + 4× sweep all completed cleanly |
| Phase 2 NaN fix | _SafeLog2CoshComplex backward NaN→0 | ✓ | 0 NaN bug recurrence Day 6 |
| Path δ Layer A theta capture | _state['theta_replicas'] + ['best_x'] dump to .pt | ✓ | 20 ckpt .pt + 4 atexit .pt = 24 .pt files total |
| Path κ1 core touch | self._latest_positions / _best_positions instance attrs | ✓ | sweep .pt files 含 theta tensors |

### 2.5 SANITY GATE behavior

| Variant | Triggers | Final best_cost_raw still anomalous? | Reason |
|---------|----------|--------------------------------------|--------|
| A | 0 | -15.70 (mild) | step ~800 dip 发生在 callback 间 (every 100 steps), gate 没 see |
| B | 22 | -8.04×10⁸ (extreme) | optimizer-internal best_cost_raw at every step (not just callback); _state side gate works, optimizer side 没 gate |
| C | 14 | -2.66×10¹⁸ (extreme) | 同 B, optimizer-internal scope limitation |
| D | 0 | -2.33 (clean) | stalled at healthy magnitude, no anomaly to gate |

**Implication**: SANITY GATE 在 callback _state 层 work, optimizer 内部 result['best_cost_raw'] (core/tcbm_optimizer_NQS.py:1078 every step) 没 gate. Category H follow-up: 加 sanity gate to optimizer-internal best tracking.

---

## 3. 发现 / 问题

### 3.1 Layer A latent bug 发现 + Path κ1 fix

**Bug 性质**: Phase 1 (Day 5) 实施时, robustness module Layer A periodic checkpoint 设计 "if isinstance(v, torch.Tensor): dump as .pt". Caller (run_gradient_baseline_v2.py) `_state` dict 从未塞入 tensor (只 scalars + lists). Layer A 写 0 个 .pt files, metadata.json 正常 update.

**Day 6 探测**: Path δ Step 1 设计 `update_capture_with_optimizer_state` helper, 加 `hasattr(optimizer, 'x')` guard. TCBMOptimizer 的 replica state 是 local var `positions` 不是 instance attr → hasattr return False → silent 0 captures. helper 设计本意 "safe fallback", 实际 "broken silently".

**Symptomatic trace**:
- last_step_captured 字段在 _state 中正常 update (那行 unconditional)
- theta_replicas / best_x 字段从未 created (hasattr 失败 skip)
- Layer A 写 .pt 检查 tensor count = 0 → 不写 (correct behavior)
- Result: 6 metadata.json + 0 .pt files (Phase 1 + Path δ Step 1 双 bug 一致表现)

**Path κ1 fix** (Category H, Nick-approved):
- core/tcbm_optimizer_NQS.py: callback fire 前 (line 1109) set `self._latest_positions = positions` + `self._best_positions = best_x`. 2-line surgical, 0 algorithm side effect
- core/robustness.py helper: read `_latest_positions` instead of `.x`
- tests/test_robustness.py: replace MockOptimizer test with real-TCBMOptimizer mini integration test (M=2/n_steps=2/n_vmc=50, ~5 min CPU wall)

**Lesson**: mock-based unit test 通过不等于 integration correct. helper / wrapper / adapter function 涉及 external object API 必须 ≥ 1 real-object integration test.

### 3.2 probe-run #1 bit-identical reproduction (positive signal)

probe-run seed=42 same cfg same RNG path, step 0/100/200/300 byte-identical Day 5 verify. step 800 anomaly + best_cost_raw=-114.84 完美复现.

**Implication**:
- Bug 是 deterministic numerical pathology, 不是 flaky stochastic glitch
- 排除 hypothesis 子场景: CUDA race condition / floating-point non-determinism / hardware glitch
- 任何 fix 之后, Day 5/probe-run 状态下 reset RNG, 应能 byte-identical verify fix 是否 work

### 3.3 hypothesis C STRONG verdict (但后被 sweep 推翻)

probe-run swap_acc + cost trajectory 分析:
- 99% time swap_acc 出 target zone [0.15, 0.35]
- 88% time swap_acc > 0.5 (chronically over-hot)
- step 800 swap_acc cliff drop (0.727 → 0.364) synchronous with cost_min cliff (-3.67 → -10.68)
- cost_min vs cost_mean separation step 800 = 8.55 (single replica outlier signature)
- bit-exact deterministic confirm

**Verdict at the time**: hypothesis C STRONG (PT swap pathology + outlier theta propagation 机制与数据一致)

**Sweep 后 retrospective**: hypothesis C 描述的是 visible symptom (swap pattern), **不是 dominant root cause**. 真实 root cause 更可能 hypothesis A (1/ψ underflow). swap_acc cliff 是 numerical artifact 产生后 PT swap 反应的下游表现.

### 3.4 4-variant sweep 全 fail (大发现, 改变 Phase 2.5 fix direction)

**预期**: hypothesis C STRONG verdict + 4-variant cfg sweep 至少 1 个 variant 应通过 Tier 1 (rel_err < 10%).

**实际**: 0/4 PASS. 最好 A 81.59%, 远超 10% 红线.

**Variant 失败模式归类**:

1. **Catastrophic numerical blowup** (B, C): best_cost_raw 跑到 -10⁸ ~ -10¹⁸ 量级, 22+14 SANITY GATE triggers. 物理不可能能量, 显然是 numerical artifact 严重.

2. **Stalled at healthy magnitude** (A, D): cost_min 冻结 -2 ~ -3, 没数值爆炸但 optimizer 没 progress. swap_acc 仍 0.4 ~ 0.8 (太热), psi_max 卡在 1.04 narrow band.

**Cross-variant pattern**:

| | T_max=2.0 | T_max=1.0 |
|---|---|---|
| T_min=0.005 | Day 5 catastrophe | A stalled |
| T_min=0.05 | B catastrophe | C catastrophe |
| T_min=0.1 | (not run) | D stalled |

T_max=1.0 + T_min=0.05 (C 双调) 出 worst outcome 完全超出预测. Single "降 T_max" 不是 magic bullet.

**修正 hypothesis priority** (基于 sweep, informal estimate):
- ~30%: hypothesis A (1/ψ underflow) dominant — fix 是 _local_energy_batch ratio safeguard
- ~25%: hypothesis A + C 组合 — numerical fix + 适合的 PT cfg
- ~20%: hypothesis A + 别的 mechanism (e.g. gradient explosion + numerical)
- ~15%: TCBM-gradient method 在 4×4 J1-J2 上 inherent insufficient — Plan B
- ~10%: cfg space 还有 untested region work — 但 sweep 已覆盖大部分 reasonable cfg

**Caveat**: 上述 % 是 Day 6 末作者基于 sweep + probe 数据的 informal estimate, 不是 statistical inference. Day 7 forensic 数据将提供更精确的 prior update.

**预测重估**: Day 5 估 70% Phase 2.5 fix 后 PASS, Day 6 sweep 后修正到 **40-50%**. 主因: cfg tune 不能 alone 解决 numerical artifact source.

### 3.5 Physical sanity gate scope limitation

SANITY GATE 实施在 progress_callback 内, gate _state['best_cost_so_far'] 在 callback 时 update. 但 optimizer 内部 result['best_cost_raw'] 在 core/tcbm_optimizer_NQS.py 每 step update 时无 gate. Variant B/C 的 -10⁸/-10¹⁸ 通过 internal 路径 leaked 到 final result.

**Day 7 follow-up** (Category H): 加 sanity gate to optimizer-internal best tracking. 不影响主 loop, 只 gate result['best_cost_raw'].

---

## 4. 红线状态

| Trigger | Day 5 | Day 6 | Status |
|---------|-------|-------|--------|
| R-abort-1 (rel_err > 15%) | ✗ TRIGGERED 98.85% | ✗ TRIGGERED again (all 4 variants 81-117%) | Day 7 reassess |
| R-abort-2 (swap_acc < 0.1 or > 0.9) | ⚠ MARGINAL 0.689 | ⚠⚠ NEAR BREACH (D 0.811, C 0.774, B 0.742; only 0.09-0.16 from 0.9 red line) | escalating |
| R-abort-3a (σ_TCBM/σ_Adam > 0.8) | pending | pending | Day 12-13 sweep |
| R-abort-3b (σ_TCBM/σ_SR > 0.8) | pending (no SR data) | pending (no SR data) | Day 7+ SR baseline |
| R-abort-5 (n_kept < 3) | pending | pending | Day 8 hybrid |

**R-abort-1 二次触发** 不是 "problem class fail", 是 numerical pathology 在 cfg space 持续. Path γ Branch 1 forensic verify hypothesis A 决定下一步.

**Plan B trigger line**: Day 7 16:00 UTC. 决策 criteria:
- Forensic 数据 + SR implementation 进度 (Day 7 上午-下午)
- Plan A path 是否 viable (numerical fix 是否 well-defined + 工程量 < 1 day)
- Plan B preparation 进度 (SR implementation 是否 close to working baseline)

---

## 5. 反思 (5 lessons)

### 5.1 Mock-based unit test 不能替代 real-object integration test (κ1 教训)

Path δ Step 1 helper 通过 MockOptimizer test, 实际 production hasattr silent fail. Step 1 design 没 verify TCBMOptimizer 真实 attribute API.

**校正**: Any helper / wrapper / adapter function 涉及 external object API, 必须 ≥ 1 real-object integration test (即使慢 200×). Mock test 测 unit, integration test 测 contract.

### 5.2 Micro-benchmark 必须覆盖 subspace_warmup 之后

Day 6 micro-benchmark 跑 100 step (subspace_warmup=150 还没到), 给 13.21 s/step 假信号. 实际 sweep run 18-22 s/step (SVD overhead 加回来).

**校正**: Benchmark 必须 ≥ subspace_warmup + 100 step. 任何短 benchmark 标 "preliminary, may underestimate by SVD overhead amount" disclaimer.

### 5.3 Multi-replica system 单 swap_acc 信号不够, 需要 per-replica state monitoring

probe-run 拿到 swap_acc trajectory 但没有 per-replica theta. swap_acc cliff 只告诉我们 "12 replicas 整体 swap behavior 变化", 不告诉我们 "哪个 replica 撞 outlier basin + how does that propagate".

**校正**: 4×4 J1-J2 这种 multi-replica system 必须 capture per-replica state (κ1 fix 之后 sweep 24 .pt 文件 ready for Day 7 forensic).

### 5.4 Hypothesis verdict STRONG 不等于 fix direction sufficient

hypothesis C STRONG verdict + 4-variant cfg sweep = 期望至少 1 个 PASS. 实际 0/4 PASS.

**校正**: Hypothesis confirm 只意味着 "数据支持 hypothesis"; fix direction work 还需要 "hypothesis 描述的 mechanism 是 dominant + fix scope 完整覆盖 mechanism". hypothesis C 描述的是 propagation 机制, 不解决 outlier basin source.

### 5.5 Spec design 必须先 verify assumed APIs (env var, hasattr) 实际存在

Path δ Step 4.1 spec 假设 env var override 已存在 (实际不存在, 要 patch); Path δ Step 1 假设 optimizer.x attribute 存在 (实际不存在, 要 κ1 patch). 两次 spec violation 都是 design 时没去 read source code.

**校正**: 任何 spec 涉及 external API 假设 (env var, instance attr, method signature), design 阶段 grep + read 验证, 不能 "想当然".

---

## 6. Day 7 Path γ plan

### 6.1 Path γ rationale

Day 6 sweep 数据排除 hypothesis C 单独 dominant. 三个 path:

- Path α: Forensic + numerical fix (赌 hypothesis A)
- Path β: 立刻 trigger Plan B (NetKet fork)
- Path γ: Hybrid — forensic + parallel SR implementation start

Nick 选 Path γ. Rationale:
1. 24 .pt files 是 cheap diagnostic resource (~2.5h forensic ROI 高)
2. hypothesis A confirm 的话 fix well-defined (_local_energy_batch ratio safeguard, ~3h 工程)
3. SR implementation 不管 Plan A/B 都要做 (Day 12-13 baseline 必须), 今天启动 = 一举两得
4. Plan β 立刻 trigger Plan B 太悲观, TCBM-NQS 还没真测试 numerical fix

### 6.2 Branch 1 forensic scope

- experiments/forensic_log_psi_diff.py: load 24 .pt snapshots
- 每个 snapshot, 1000 random sample, compute log_psi(σ) + log_psi(σ_flipped), derive log_psi_diff distribution
- 统计 |log_psi_diff.real| > 30 outlier count per variant per checkpoint
- 不解读, 只 data dump

**Expected outcomes**:
- |log_psi_diff| > 30 outlier 大量出现 → hypothesis A confirmed → Phase 2.5 fix 方向是 _local_energy_batch ratio safeguard
- outlier count 低 → hypothesis A 不 dominant → 需要新 hypothesis

### 6.3 Branch 2 SR implementation scope

- 5 function 顺序实施: compute_log_derivatives → compute_qgt → regularize_qgt → solve_sr_update → SROptimizer.optimize()
- Regularization: Bukov-style proportional (S + ε·diag(S))
- Solver: pseudoinverse (torch.linalg.pinv)
- Trajectory: single trajectory (not PT, standard SR)
- pytest: 2×2 trivial ED 对照 + QGT Hermitian/PSD + log_derivatives shape

### 6.4 Day 7 decision points

**UTC ~09:30 (Day 6 daily_log commit + push 完成 → Path γ 启动)**

**UTC ~12:00 (~2.5h post-start, forensic + SR design 完成)**:
- Forensic verdict: hypothesis A confirmed?
- SR design: 5 function pseudo-code ready?

**UTC 12:00-15:30 (forensic 解读 + SR implementation)**:
- Forensic verdict 解读 (~30 min, Nick+AI)
- 如 A confirmed: launch Phase 2.5 numerical fix design (~30 min) + 实施 (~2.5h)
- SR implementation 并行 (~3h)

**UTC 16:00 (Plan B trigger line)**:
- 评估: numerical fix 工程 + SR baseline 进度
- numerical fix 完成 + SR 70%+ done: continue Plan A, launch run #4 with fix
- numerical fix 复杂 (> 4h) or SR 卡住: trigger Plan B (NetKet fork)

**Schedule risk**: 如果 Day 7 daily_log + Path γ 全跑通需要 ~7-8h, UTC 16:00 trigger line 之前完成所有工作 viable but tight. 任何 mid-day intervention (e.g. 又一个 latent bug) 可能 push 决策到 Day 8.

---

## 7. Claude Code 自主决定汇总

(Standing delegation policy 要求每日记录)

**Category A** (pure plumbing, autonomous):
1. tmux orphan cleanup (Day 4-5 残留 session + Day 6 sweep session post-completion)
2. Test parameter tweak in κ1 integration test (M=2/n_steps=2/n_vmc=50 vs spec M=4/n_steps=10/n_vmc=100, 加速 CPU test)
3. File path naming for probe results (baseline_v2_probe_seed42.json)
4. CHECKPOINT_PATH dead code removal
5. DRY refactor: TCBM_RUN_NAME 统一从 module-level _RUN_NAME constant 读

**Category B** (backward-compat infrastructure extensions):
1. TCBM_N_STEPS env var (098f906)
2. TCBM_RUN_NAME env var (098f906)
3. TCBM_T_MIN env var (087d66f)
4. TCBM_T_MAX env var (087d66f)
5. update_capture_with_optimizer_state helper function (f6af560)
6. _latest_positions / _best_positions instance attributes (f6af560)
7. Physical sanity gate (E_0_TRUTH - 1.0 lower bound, 087d66f)
8. Background task scheduling (T+5min / T+30min / step 800 check)

**Category I** (spec 小调整 + 自报):
1. Path Y env var pattern (Step 4.1 spec 给的两个 fallback 都不可行, 走 minimal touch env var add 而非 wrapper script)
2. probe-run #1 不 kill 让它跑完 (拿 partial value: swap_acc trajectory for hypothesis C)

**Category H** (Nick explicit approval required, Nick 批准了):
1. Path κ1: core/tcbm_optimizer_NQS.py 加 self._latest_positions + self._best_positions instance attributes

**STOPs + ask 触发**:
1. Step 4.1 launch path 实施: spec 内不可行, Claude Code STOP ask, Nick decided Path Y
2. UTC 09:25 Layer A bug discovery: Layer A fix failed at step 500 checkpoint, Claude Code STOP per Nick's standing instruction
3. Daily_log finalization: Step 1 placeholder 没真 paste, Claude Code STOP ask (fabrication-free)

---

## 8. Day 7 commit handoff

**待 commit (Day 7 morning closing batch)**:

| File / Pattern | Count | Size | Note |
|---|---|---|---|
| `docs/daily_log_day6.md` | 1 | - | new |
| `results/sweep_v_{a,b,c,d}_day6_seed42.json` | 4 | 108 KB | force-add (sweep final results) |
| `results/sweep_v_*_day6_seed42_checkpoint_step{500,1000,1500,2000,2500}_metadata.json` | 20 | 92 KB | force-add |
| `results/sweep_v_*_day6_seed42_checkpoint_step{500,1000,1500,2000,2500}_state.pt` | 20 | 1.2 MB | force-add (Path δ+κ1 first theta capture) |
| `results/sweep_v_*_day6_seed42_atexit_metadata.json` | 4 | (part of 272 KB) | force-add |
| `results/sweep_v_*_day6_seed42_atexit_state.pt` | 4 | (part of 272 KB) | force-add |
| `results/baseline_v2_probe_seed42*.json` | 4 | 17 KB | force-add (probe-run #1, pre-κ1 so no .pt) |
| `logs/sweep_{a,b,c,d}_day6_20260512_0813.log` | 4 | 36 KB | force-add |
| `logs/probe_day6_seed42_20260511_0736.log` | 1 | 3.5 KB | force-add |
| `logs/probe_analysis_20260512_0725.log` | 1 | 3.3 KB | force-add |

**TOTAL**: 1 daily_log + 36 evidence files ≈ **1.8 MB**

Note: `results/probe_analysis_hypothesis_c.json` 已在 087d66f commit, 不重 add.

**Day 7 早上第一件事**: commit + push 上述, 然后启动 Path γ Branch 1 + Branch 2.

---

**End of Day 6 daily_log.**
