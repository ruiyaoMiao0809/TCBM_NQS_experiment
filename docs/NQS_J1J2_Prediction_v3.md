# NQS J1-J2 (4×4) 任务的 TCBM 适用性数学证明 v3

**Status**: SKELETON (Day 4 Step 5.i, 2026-05-05). Section structure + per-section
descriptions only. Content fill = Step 5.ii (Day 4 afternoon).

**Supersedes**: `NQS_J1J2_Prediction_v2.md` (1098 lines, kept as historical artifact).

**Reframe**: Adam → SR primary baseline (per Day 3 quick_adam_diagnostic_v2 verdict
+ Day 4 Step 4 v3 outline update). Single Tier 3 with sub-criteria (a)/(b),
both must hold for Tier 3 PASS.

**Section IDs preserved from v2 for diff clarity** (§1.1, §1.3, §2, §4.1-4.3,
§7.1, §8.3 are near-verbatim from v2). New sections: §9, §10, §11.

**Length target**: ~900-1100 lines (close to v2's 1098).

---

## §0 Document metadata

- ~30 lines
- Status block, version, date, supersedes-v2 note, key reframe summary,
  Day 3 v2 verified data anchor (mean rel_err 74.45%, σ_Adam=9.5%)
- Sources: `docs/NQS_J1J2_Prediction_v3_outline.md` Trigger paragraph (lines 6-21);
  `results/quick_adam_diagnostic_v2.json` summary block

---

## §1. Stage I: 问题形式化

### §1.1 标准化优化形式

- ~50 lines, **PRESERVE v2 verbatim** (lines 41-63 of v2 doc)
- 4×4 J1-J2 PBC, J2/J1=0.5, RBM α=2, D=1120 real params, S_z=0 sector dim 12,870
- Hamiltonian + RBM ansatz definition + parameter layout
- Sources: v2 §1.1; `core/j1j2_problem.py` rbm_param_layout()

### §1.2 P0-1.2 Success criteria (4-tier replacement)

- ~80 lines, **REWRITE** (replaces v2 §1.2's single-criterion definition)
- 4-tier table with single Tier 3 dual-baseline structure:
  - Tier 1: rel_err < 10% (claim A: feasibility)
  - Tier 2: rel_err_TCBM/rel_err_SR ≤ 2.0 (NQS comparable)
  - Tier 3 (single tier, dual sub-criteria, BOTH must hold): (a) σ_TCBM/σ_Adam ≤ 0.5
    AND (b) σ_TCBM ≤ 2× σ_SR (claim B: robustness)
  - Tier 4: n_kept ≥ 5/20 (claim C: mechanism, TCBM ⊥ SR)
- Paper outcome dependency table (which tiers PASS → which paper strength)
- Sources: outline Change 1 (lines 26-46); outline Implications (lines 169-189)

### §1.3 量级分析

- ~30 lines, **PRESERVE v2 verbatim** (lines 96-106 of v2)
- Energy scales, gap analysis, L_basin estimate
- Sources: v2 §1.3; `docs/daily_log.md` Day 1 entry (L_basin = 20.45)

### §1.4 约束处理

- ~10 lines, **PRESERVE v2 verbatim**
- NQS has no hard constraint (RBM is unconstrained over R^D)

### §1.5 Stage I 交付物

- ~15 lines, **PRESERVE v2 with minor updates**
- J1J2Problem class, ED reference, unit tests (23 → mention RNG fix tests)

---

## §2. Stage II: 能量势垒结构分析

### §2.1-2.6 全段 PRESERVE v2 verbatim

- ~150 lines total, 6 subsections preserved
- Bukov anchor numbers stay load-bearing (SR also lives on this landscape)
- Sources: v2 §2.1-2.6; `docs/bukov_2021_anchor.md` Fig 11/12 readings;
  Nick visual inspection (commit c02339b)
- Note: §2.4 WKB Condition + §2.5 Stage II Gate kept verbatim — v3 reframe
  doesn't touch energy-barrier topography analysis.

---

## §3. Hypotheses (REWRITE)

### §3.1 H1, H2, H3 reframed

- ~80 lines, **REWRITE** (v2 §3 was algebraic structure; v3 §3 is hypotheses)
  - H1: TCBM σ_seed ≤ 0.5 × SR σ_seed (primary claim B)
  - H2 (NEW): TCBM-hybrid n_kept ≥ 5/20 SR-orthogonal (claim C)
  - H3 (reframed): TCBM clamping subspace ≠ QGT principal subspace
- Sources: outline Change 2 (lines 50-70)

### §3.2 H2 measurement procedure (4-step Day 7-8)

- ~60 lines, **NEW**
- Step 1: random null calibration (50 pairs random k=20 in D=1120)
- Step 2: cosine threshold calibration (random_null_p95 ≈ 5)
- Step 3: TCBM vs SR n_kept measurement
- Step 4: statistical test against null distribution
- Sources: outline H2 measurement procedure (lines 71-89)

### §3.3-3.5 v2 algebraic structure analysis (RELOCATE / TRIM)

- v2 §3.1-3.5 (algebraic decomposition, A_1 representation, emergent
  low-dim) — relocate to §3.4 as "supporting algebraic argument";
  trim from ~94 → ~30 lines (load-bearing for ψ probe theory in §4)
- Sources: v2 §3.1-3.5

---

## §4. Stage III-B: ψ Warmup Probe

### §4.1-4.3 全段 PRESERVE v2 verbatim

- ~80 lines total (v2 lines 426-501)
- ψ evolution physical mechanism + quantitative prediction + v2 case comparison
- Sources: v2 §4.1, 4.2, 4.3

### §4.4 ψ probe protocol (Day 7 execution) — PRESERVE v2

- ~30 lines, v2 §4.4 (lines 502-531)
- Sources: v2 §4.4; `experiments/run_t4a_diagnostic.py`

### §4.5-4.6 PRESERVE v2 with minor edits

- ~30 lines
- Stage III-B deliverables + Stage III total Gate
- Edits: update "Adam baseline" references → "SR primary + Adam ablation"

---

## §5. Stage IV: 同构映射构造

### §5.1-5.3 PRESERVE v2 verbatim

- ~70 lines (v2 lines 574-628)
- Candidate target physics, RBM Free Energy isomorphism, identity mapping limits

### §5.4 Bukov 2021 quantitative anchor — PRESERVE v2

- ~20 lines (v2 lines 629-646)
- Direct number embedding per D3.A: σ_seed ≈ 0.34% (Bukov Fig 12),
  Hessian ratio 4×10⁻⁷ (Fig 11), partial learning ~10⁻⁴ (Fig 8)
- Cite by reference: "see `docs/bukov_2021_anchor.md` for verbatim quotes"
- Sources: v2 §5.4; `docs/bukov_2021_anchor.md`

### §5.5 Stage IV Gate — PRESERVE v2

- ~20 lines (v2 lines 647-665)

### §5.6 Stage IV 交付物 (EXPAND with SR list)

- ~30 lines, **REWRITE** (expand v2's list)
- Add: `core/sr_optimizer.py` (Day 5), `experiments/run_sr_baseline.py` (Day 6),
  `experiments/run_sr_robustness_sweep.py` (Day 17-19),
  `analyze_subspace_overlap.py` (Day 8+)
- Sources: outline Change 6 (lines 105-110)

### §5.7 Success thresholds table (RELOCATE from v2 §6.7 area)

- ~30 lines, **REWRITE** as 5-row table per outline Change 3
- Rows: rel_err_debiased, rel_err_TCBM/rel_err_SR, σ_TCBM dual-baseline,
  n_kept, swap_acc
- Note: Tier 3 row presents (a) AND (b) sub-criteria in single row
  (NOT split — per Step 4 ruling, single Tier 3 with both must hold)
- Sources: outline Change 3 (lines 73-87); outline Implications (lines 169-189)

---

## §6. Stage V: T1-T5 量化验证 + R-aborts

### §6.1-6.6 PRESERVE v2 verbatim

- ~140 lines (v2 lines 696-825)
- T1, T2, T3, T4a, T4b, T5 conditions all preserved
- Sources: v2 §6.1-6.6

### §6.7 Stage V judgment matrix (Adam → SR reflow)

- ~20 lines, **REWRITE** (v2 lines 826-845)
- Replace Adam columns with SR primary + Adam secondary (ablation)
- Single Tier 3 dual-baseline structure reflected
- Sources: v2 §6.7; outline Change 5 (line 102)

### §6.8 R-abort triggers table (REWRITE)

- ~30 lines, **REWRITE**
- Full R-abort table (R-abort-1, -2, -3 sub-criterion (a), -3 sub-criterion (b),
  -4, -5)
- R-abort-3 sub-criterion (a): σ_TCBM/σ_Adam > 0.8
- R-abort-3 sub-criterion (b): σ_TCBM > 4× σ_SR (i.e., 2× over PASS bar)
  ← **threshold recalibration vs daily_log historical 0.8**
- R-abort-4: T4a ψ < 1.1 by step 1000 (preserved from v2.1 protocol)
- R-abort-5 (NEW): n_kept < 3 → mechanism claim demoted
- Sources: outline Change 4 (lines 91-99); daily_log Step 4 deferred-fix flag

---

## §7. 完整决策推导

### §7.1 应用 v2 §8.3 最终判定规则 — PRESERVE v2 with baseline edits

- ~20 lines (v2 lines 870-885)
- Final judgment rule, "Adam baseline" → "SR primary + Adam ablation" updates
- Sources: v2 §7.1

### §7.2 与 v2 §9 已知案例的最终对比 — PRESERVE v2

- ~20 lines (v2 lines 886-901)
- Sources: v2 §7.2

### §7.3 Symptom 分类 — PRESERVE v2

- ~20 lines (v2 lines 902-919)
- Sources: v2 §7.3

---

## §8. 核心证明结论与 ex ante 预测

### §8.1 总判定 — UPDATE

- ~15 lines, minor edit
- v2 said "STRONG GREEN with Day 7 verification"; v3 retains GREEN judgment +
  add "Day 3 Adam multi-basin verdict supports 4×4 retention"

### §8.2 Ex Ante 可证伪预测 — UPDATE

- ~25 lines
- Add Day 3 v2 anchored predictions: σ_Adam = 9.5%, σ_TCBM ≤ 4.7% (sub-criterion
  (a)); σ_TCBM ≤ 2× σ_SR pending Day 6
- Sources: v2 §8.2; outline Implications

### §8.3 与方法论 v2 §1.1 哲学呼应 — PRESERVE v2 verbatim

- ~20 lines (v2 lines 952-967)
- Sources: v2 §8.3

### §8.4 Day 7 决策依据 — PRESERVE v2

- ~15 lines, minor edits for actionable spec

---

## §9. NEW: Adam → SR reframe rationale

- ~80 lines, **NEW SECTION**
- Why this reframe was triggered (Day 3 v2 verdict)
  - Sub-§9.1: Day 3 v2 verified data — 5-seed table, σ_Adam=9.5%, mean rel_err
    74.45%, drift +1.705 (5/5 positive)
  - Sub-§9.2: Why Adam ceiling matters — Bukov 4×4 SR achieves ~10⁻³ rel_err
    vs Adam's 58-84%, 3-4 orders of magnitude gap
  - Sub-§9.3: Why SR is NQS field standard — Bukov 2021, NetKet/jVMC use SR;
    SR is the natural-gradient baseline for any "TCBM vs NQS-standard" comparison
  - Sub-§9.4: Why Adam is retained as ablation — SI demonstrates vanilla
    gradient descent failure mode (74.45% rel_err) → highlights PT value
- Sources: `results/quick_adam_diagnostic_v2.json` summary; outline Trigger
  paragraph (lines 6-21); `docs/bukov_2021_anchor.md`; daily_log §3.1 RNG bug
  + §3.2 Adam verdict reading

---

## §10. NEW: Reframe risks + Plan A/B/C hedges

- ~70 lines, **NEW SECTION**
- Sub-§10.1: Reframe risks
  - Risk 1: SR may be too good (Tier 2 fail) → narrative defense (4 bullets) +
    Day 3 verified anchor (σ_Adam=9.5%, σ_TCBM ≤ 4.7% sub-criterion (a) safety net)
  - Risk 2: n_kept = 0 → mechanism claim collapses → switch to J1-J2-J3
  - Risk 3: SR implementation cost → Plan A/B/C hedges (below)
- Sub-§10.2: Plan A/B/C trigger rules + execution
  - Plan A (default): SR runs by Day 6, full reframe schedule on track
  - Plan B (Day 7 16:00 trigger if SR rel_err > 5%): NetKet fork, Day 8-12,
    schedule slips to Day 22-24
  - Plan C (Day 10 16:00 trigger if NetKet also fails): Adam-only fallback paper
- Sources: outline Reframe risks (lines 269-322); outline Workflow
  implications (lines 196-260); daily_log §3.3 dual-baseline narrative

---

## §11. NEW: Disclosures and asymmetries

- ~80 lines, **NEW SECTION** (D2.B scope: 5 items)
- Sub-§11.1: TCBM vs Adam/SR final-eval target asymmetry
  - TCBM: best_x (PT replica with lowest single-shot cost) — selection bias
    intrinsic to PT; mitigated by N=2 evaluations averaged at termination
  - Adam: final_θ (last optimizer step) — assumes monotonic convergence;
    selection bias would inflate apparent performance
  - SR: final_θ (matches Adam, no PT replicas to select among)
  - Sources: `docs/known_issues_day2.md` final-eval asymmetry section
- Sub-§11.2: NQS-1e debiasing (TCBM only)
  - VMC noise → best-observed energy biased low (selection effect)
  - TCBM re-evaluates best_x with n_vmc_samples_final samples for debiasing
  - Adam / SR don't have this (M=1, no replica selection)
  - Sources: `core/tcbm_optimizer_NQS.py` NQS-1e module docstring
- Sub-§11.3: n_vmc_samples_final asymmetry (verified Day 4)
  - TCBM v2: n_final=2 (Day 3 reduced from 4 for ~30% wall savings)
  - TCBM v1 (historical): n_final=4 (Day 2 6h22min run)
  - Adam: N_FINAL_AVERAGES=4
  - Adam diagnostic v2: N_FINAL_SHOTS=4
  - SR (Day 6+ skeleton): N_FINAL_AVERAGES=4
  - 2-vs-4 shot asymmetry: TCBM debiased estimate has higher per-replica
    variance vs Adam/SR's 4-shot mean. Mitigated by TCBM's M=12 PT structure
    (more total exploration before final eval). Disclosed for SI methods.
  - Sources: `experiments/run_gradient_baseline_v2.py:170` (n_final=2);
    `experiments/run_adam_baseline.py:55` (N_FINAL_AVERAGES=4);
    `experiments/quick_adam_diagnostic_v2.py:47` (N_FINAL_SHOTS=4);
    `experiments/run_sr_baseline.py:58` (skeleton matches Adam)
- Sub-§11.4: RNG bug timeline + fix
  - Day 1-3 production runs effectively seed=0 (J1J2Problem._gen hardcoded)
  - Day 3 evening discovery via Adam diagnostic v1 byte-identical results
  - Fix commit 18716b3: J1J2Problem.set_seed() method + 3 pytest cases
  - All Day 4+ work assumes RNG-fixed codebase
  - Sources: daily_log §3.1; `core/j1j2_problem.py` set_seed() method;
    `tests/test_rng_seeding.py`
- Sub-§11.5: Seed sequencing
  - Adam diagnostic v2 seeds: [42, 7, 13, 21, 99] (5 seeds, Day 3)
  - Production v2 seed: 42 (Day 4 single seed before sweep)
  - Robustness sweep seeds: TBD Day 12-13 (15 seeds × 4 methods)
  - No seed asymmetry between methods at sweep stage (same 15 seeds × all methods)
  - Sources: `experiments/quick_adam_diagnostic_v2.py:44`;
    `docs/NQS_experiment_plan.md` Day 12-13

---

## Skeleton self-check

- 16 sections (§0 + §1.1-1.5 + §2 + §3.1-3.4 + §4.1-4.6 + §5.1-5.7 + §6.1-6.8 +
  §7.1-7.3 + §8.1-8.4 + §9 + §10 + §11)
- Estimated total: ~30 + 50 + 80 + 30 + 10 + 15 + 150 + 80 + 60 + 30 + 80 + 30 +
  30 + 30 + 30 + 30 + 140 + 20 + 30 + 20 + 20 + 20 + 15 + 25 + 20 + 15 + 80 +
  70 + 80 = ~1378 lines
- Over D5 budget (1100 cap). Need trim.
- Trim candidates:
  - §2 Stage II (150 → 100, drop redundant prose)
  - §6.1-6.6 T1-T5 (140 → 100, condense T-conditions if Stage II covers similar)
  - §3.3-3.5 algebraic structure (already trimmed 94 → 30; could go to 20)
  - §11 (80 → 70, tighten sub-§11.4 RNG timeline since covered in daily_log)
- After trim: ~1378 - 90 = ~1290 lines. Still over.
- Decision needed: accept 1300-line target OR aggressive trim to 1100.

## D2.B coverage check

- §11.1: TCBM best_x vs Adam/SR final_θ ✓
- §11.2: NQS-1e debiasing TCBM-only ✓
- §11.3: n_vmc_samples_final TCBM=2 vs Adam/SR=4 ✓
- §11.4: RNG bug timeline ✓
- §11.5: seed sequencing ✓
- 5/5 D2.B disclosures covered.

## Open decision points for review pass

- (Q1) Length: aggressive trim to 1100 (risk over-compression of preserved
  v2 sections) OR accept 1200-1300 line target (slightly over D5)?
- (Q2) §3.3-3.5 v2 algebraic structure: keep at 30 lines (load-bearing for §4
  ψ probe theory) OR trim to 20 lines (acknowledge ψ probe self-contained)?
- (Q3) §6.8 R-abort threshold for sub-criterion (b): use σ_TCBM > 4× σ_SR
  (2× over PASS bar) OR pick numerically once Day 6 SR data in?
- (Q4) §11.5 seed sequencing: include "Adam diagnostic v2 had buggy v1 seeds"
  paragraph OR keep clean by referencing daily_log §3.1?

---

**End of skeleton.** Awaiting Nick review pass before content fill (Step 5.ii).
