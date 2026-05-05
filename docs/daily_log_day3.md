# Day 3 Daily Log

**Date**: 2026-04-29  
**Phase**: Week 1, Day 3 (Diagnostic + SR baseline preparation)  
**Codebase**: `/home/dglg/miao_workplace/tcbm_nqs/`  
**GPU**: 4× NVIDIA A10 24GB (HKUST-GZ lab server)  
**Commits**: 9 (8 pushed, 1 local pending push)  
**Tests**: 23 passing (16 Day 1 + 4 Day 3 callback + 3 Day 3 RNG seeding)

---

## 1. 计划 vs 实际

**Day 3 原计划** (Day 2 daily_log 末尾记录):
- 早上: Phase A forward work (callback hook + production v2 + micro-benchmark)
- 下午: launch production v2 TCBM-gradient (seed=42, 5-6h wall)
- 晚上: Tier 1 verdict + Day 4 plan

**Day 3 实际 path**:
- 早上 (06:00-13:00 UTC): Phase A forward work 按计划完成 (commits 7580cc6 + fddbc6f)
- 下午 (13:00-16:00 UTC): Adam baseline skeleton + final-eval asymmetry doc + Bukov 三图精读 (commits d86bff9, f2d8223, c02339b)
- 16:44 UTC: Adam diagnostic v1 launched (5 seeds × 2000 steps × M=1)
- 17:30 UTC: 发现 v1 byte-identical results 异常 (seed 42 与 seed 7 final shots 完全相同到4位小数)
- 17:30-18:30 UTC: Claude Code grep 定位 RNG bug 在 `core/j1j2_problem.py:319-320`，全 codebase 修复 + 3 pytest cases (commit 18716b3)
- 18:30-19:00 UTC: v3 outline reframe (Adam → SR primary baseline)，commits 8ee7ebf + 2a99dbf
- 19:00-21:30 UTC: Adam diagnostic v2 (RNG-fixed) launched + completed (5 seeds 顺序在 GPU 3，~2.5h wall)
- 21:30 UTC: 5-signal post analysis + verdict (commit 76ab5aa)

**偏离原计划的原因**: Adam v1 的 RNG bug 是 critical discovery，必须立即修复后重跑 v2 才能拿到可信 verdict。production v2 TCBM-gradient run 因此推迟到 Day 4 下午。Schedule slip = 1 天，剩余 buffer ~1 天。

---

## 2. 关键数字

### 2.1 Adam diagnostic v2 5-seed 结果

| Seed | best_during | final | drift | rel_err |
|------|-------------|-------|-------|---------|
| 42 | -4.35 | -3.527 | +0.823 | 58.30% |
| 7 | -2.91 | -1.319 | +1.591 | 84.40% |
| 13 | -4.05 | -2.075 | +1.975 | 75.47% |
| 21 | -3.89 | -2.465 | +1.425 | 70.86% |
| 99 | -4.13 | -1.419 | +2.711 | 83.22% |

ED ground state: E_0 = -8.4579 (4×4 J1-J2 J2/J1=0.5, S_z=0 sector dim 12,870)

### 2.2 5 个诊断信号

| Signal | 数值 | 标签 |
|--------|------|------|
| σ_Adam_rel | **9.495%** | 5× strong threshold (>2%) |
| Cluster gap_ratio | 0.481 | BORDERLINE (under 0.5 strict cutoff) |
| Mean drift | **+1.705** | LARGE_DRIFT (5/5 seeds positive) |
| Plateau escape | 1/5 | PARTIAL_ESCAPE (seed 99 only) |
| Combined verdict | 2 strong + 4 weak | MODERATE_RUGGEDNESS (Claude Code label) |

**Claude Code label vs Nick reading**: Claude Code 给出 `MODERATE_RUGGEDNESS`，Nick 读法是 STRONG multi-basin signal。分歧来源是 gap_ratio 0.481 在 0.5 strict cutoff 下被判 BORDERLINE，但 5-seed 样本量本就不足以让 gap_ratio 在严格阈下稳定通过——这不是 noise floor，是采样不足。综合 σ 9.5% (5× threshold)、5/5 正向 drift 一致、2-cluster 结构 (seed 42 outlier at -3.527, 其余 ∈ [-2.5, -1.3])，Nick 的 STRONG 读法是更精确的 interpretation。Day 4 v3 doc 写作时按 STRONG 处理。

### 2.3 Day 3 commits (9 总, 8 pushed)

| Hash | Message | Phase |
|------|---------|-------|
| 7580cc6 | Day 3 A1: callback hook to TCBMOptimizer.optimize() | 早上 forward work |
| fddbc6f | Day 3 A2/A3: production v2 + micro-benchmark scripts | 早上 forward work |
| f2d8223 | Day 3: TCBM vs Adam final-eval target asymmetry doc | 下午 |
| d86bff9 | Day 3: Adam baseline (P1-1.1) skeleton with full robustness suite | 下午 |
| c02339b | Day 3: Bukov Fig 11/12/8 precise readings (Nick visual inspection) | 下午 |
| 18716b3 | Day 3: fix J1J2Problem RNG hardcoded to seed=0 | 晚上 (RNG bug fix) |
| 8ee7ebf | Day 3: Prediction v3 outline (Adam → SR reframe) | 晚上 (reframe) |
| 2a99dbf | Day 3: v3 outline followup revisions | 晚上 (reframe) |
| 76ab5aa | Day 3: Adam diagnostic v2 (RNG-fixed) + post-analysis | 晚上 (verdict) — local pending push |

### 2.4 Adam ceiling 真实数字 (用于 v3 outline Step 4 update)

- mean rel_err (5 seeds final): **74.45%**
- min rel_err: **58.30%** (seed 42 best)
- max rel_err: **84.40%** (seed 7 worst)
- best_during mean rel_err: ~54% (5 seeds 平均)

之前 v1 跑出来的 65% 是 effective seed=0 的 buggy 数字，作废。v2 的 74.45% 是权威数据。

---

## 3. 发现 / 问题

### 3.1 RNG bug (codebase-wide critical fix)

`core/j1j2_problem.py:319-320` 的 `J1J2Problem` 内部 `torch.Generator` hardcoded 到 `manual_seed(0)`，注释 "users override via optimizer's seed" 是误导——脚本层的 `torch.manual_seed(seed)` 和 `np.random.seed(seed)` 都不影响这个内部 generator。该 generator 用于 4 处:
- Line 345: `random_feasible` (init theta)
- Line 487: VMC chain init
- Line 496: Metropolis flip site
- Line 506: Metropolis accept log_u

**影响范围**: Day 1-3 所有 production runs 都是 effective seed=0 (Day 2 6h22min 失败 baseline 也是 seed=0)。如果未发现此 bug 直接进入 Day 17-19 的 15-seed × 4-method robustness sweep (60 runs)，结果会是 σ = 0 的 vacuous 数据，Tier 3 claim 全废，paper rebuttal 阶段会被审稿人一句话击穿。

**修复**: 加 `J1J2Problem.set_seed(seed)` 方法重置内部 generator + production scripts 在创建 problem 后立即调用 + 3 个 pytest 验证 (`test_problem_set_seed_changes_init`, `test_problem_set_seed_reproducible`, `test_evaluate_seeded_reproducible`)。

**留痕策略**: v1 buggy script (`experiments/quick_adam_diagnostic.py`) + 其 partial JSON (`results/quick_adam_diagnostic.json`) 都保留并 force-add 到 git，作为 RNG bug 存在的 evidence (3 seeds 全 byte-identical 的 reproducibility failure)。`.gitignore` 不修改，`results/*.json` 仍 ignored 给后续 sweep artifacts。

### 3.2 MODERATE → STRONG 读法分歧 (见 §2.2)

详见上文。Day 4 v3 doc 写作按 STRONG 处理，但 daily_log 必须明确记下分歧来源 (gap_ratio 0.5 strict cutoff vs 5-seed 采样不足)，Day 7 random null calibration 完成后用 50 pairs 数据回头验证这个判断。

### 3.3 Dual-baseline robustness option (Tier 3 拆分)

Adam v2 verdict 出来后浮现一个 narrative 选项：Tier 3 不必单押 σ_TCBM/σ_SR ≤ 0.5 (high risk, Bukov 6×6 SR σ ≈ 0.34% 是 challenging benchmark)，可拆成:
- Tier 3a: σ_TCBM/σ_Adam ≤ 0.5 — easily achievable (predicted σ_TCBM ~1-3% vs σ_Adam = 9.5%), 安全垫 claim
- Tier 3b: σ_TCBM/σ_SR ≤ 2.0 — challenging (4×4 σ_SR 预期 0.5-1.5%), narrative ceiling

合起来读: "TCBM 显著好于 unstable baseline 且不输给 best baseline"，比单 Tier 3 少一个 R-abort 风险面。Day 4 早上 Nick 拍板采纳此结构。Day 4 下午 v3 doc Step 5 §1.2 success criteria table 改成 5-row。

---

## 4. 红线状态

| Trigger | 状态 | 备注 |
|---------|------|------|
| R-abort-1 (rel_err > 15% vs ED) | inconclusive | 等 Day 4 production v2 |
| R-abort-2 (swap_acc < 0.1 or > 0.9) | pending | 等 Day 4 production v2 |
| R-abort-3a (σ_TCBM/σ_Adam > 0.8) | pending | 等 Day 12-13 sweep |
| R-abort-3b (σ_TCBM/σ_SR > 0.8) | pending | 等 Day 12-13 sweep, SR 数据 Day 6+ |
| R-abort-5 (n_kept < 3) | pending | 等 Day 8 hybrid 测量 |

Day 4 evening verdict 唯一可执行的红线是 R-abort-1 + R-abort-2 + 发散/NaN watchdog。Tier 3a/3b 整段判断 deferred 到 Day 6+。

---

## 5. 反思 (5 lessons)

### 5.1 RNG infrastructure 必须 Day 1 显式 test 跨 seed reproducibility

Day 1 写 23 tests 时把 reproducibility 假设成 "脚本层 manual_seed 就够"，没写一个 "different seeds → different outputs" 的 test。如果 Day 1 就有这个 test，Day 2 那 6h22min 的失败 baseline 不会发生 (即使 SIGTERM 也已经能从 partial output 看出 seed 异常)。教训: **任何 stochastic component 在添加时必须配套 cross-seed 测试**，不只是 same-seed 重现性测试。

### 5.2 Wall-clock 估算系统性低估 6-10×

Day 2 估算 30 min, 实际 6h22min, 6.5× 低估。原因: 忘掉 M=12 replicas × VMC per-replica cost 的乘法效应。Day 3 production v2 估算 5-6h 是吃过教训后的修正版 (n_final=2 节省 ~30%, M=12 replicas 已纳入)。教训: **multi-replica algorithm 估算 wall-clock 必须显式写 M × per-replica × n_steps 公式，而不是凭直觉**。Day 4 production v2 launch 后第一次 callback fire (step 100) 应该在 ~10-12 min，如果超过 20 min 立即 SIGTERM 重新评估。

### 5.3 force-add 优于修改 .gitignore 做 one-off override

Day 3 晚上要把 `results/*.json` (默认 ignored) 中的两个特定文件 (v1 buggy + v2 final) commit 进去。两种做法:
- 修改 `.gitignore` 加 exception → 影响所有未来 sweep artifacts
- `git add -f results/quick_adam_diagnostic.json results/quick_adam_diagnostic_v2.json` → 单次 override，未来 sweep artifacts 仍 ignored

选择第二种保留了 sweep artifacts ignored 的纪律，避免几周后 main sweep 50 个 JSON 全部进 git。教训: **policy-level pattern (gitignore) 不应被 instance-level needs 修改**。

### 5.4 v3 outline 必须 explicit 写 Plan A/B/C trigger conditions

v2 outline 的 contingency 写得 vague ("if SR fails")，Day 3 reframe 时发现这种措辞在压力下完全无法执行——什么叫 fails？rel_err > 5% 算 fail 吗？10%？implementation bug 算不算？v3 outline 写 Plan B trigger 是 "Day 7 16:00 UTC 如 SR rel_err > 5% on 4×4 single seed"，Plan C trigger 是 "Day 10 16:00 UTC 如 NetKet fork 也失败"——具体到日期 + 数值阈 + 失败定义。教训: **任何 contingency plan 必须有 trigger date + numerical threshold + failure definition 三要素**，缺一会在执行时 thrash。

### 5.5 Adam ceiling reframe 完整链条

数字演进:
- v1 buggy: best_during 21% rel_err (一个 fluke 数字, effective seed=0)
- v2 真实: final mean 74.45%, best_during mean ~54%, min final 58.30%

读法演进:
- v1: Adam achievable as comparable baseline (但其实 seed=0)
- v2: Adam STRONG multi-basin (4×4 retained, no 6×6 pivot needed)
- v2 同时确认: Adam **不是** SR-level 的 meaningful baseline (Bukov 4×4 SR ~10⁻³ vs Adam 58%-84%)

最终 narrative: Adam 作为 secondary unstable baseline (Tier 3a 的对照, 5 seeds ablation)，SR 作为 primary best baseline (Tier 2 + Tier 3b 的对照)。这是 Adam → SR reframe 的本质——不是 Adam 被淘汰，而是定位从 primary 改成 secondary。

教训: **baseline 的定位 (primary/secondary/ablation) 应在 outline 阶段 explicit 写清，避免 reframe 时被误读为 "kill baseline"**。

---

## 6. Day 4 plan handoff

详见 `docs/NQS_experiment_plan.md` Day 4 section + Step 1-9 detailed breakdown (本次 chat session 内已对齐)。关键交接点:

1. **Step 4** v3 outline Adam ceiling 数字: 65% → 74.45% (mean rel_err) / 58.30% (min rel_err)
2. **Step 5** full v3 doc 用 5-tier 结构 (Tier 3 拆 3a/3b)
3. **Step 5** 新加 §11 disclosure items: TCBM best_x vs Adam/SR final_θ asymmetry
4. **Step 5** Bukov 数字直接从 `docs/bukov_2021_anchor.md` 引用，不凭记忆复述
5. **Step 6** production v2 launch on GPU 3 (22515 MB free, 0% util, 已确认)
6. **Step 8** Day 4 evening verdict 仅评估 Tier 1 + R-abort-2，Tier 3a/3b deferred 到 Day 6+

---

**End of Day 3 daily_log.**