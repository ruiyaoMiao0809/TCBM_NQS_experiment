# NQS J1-J2 Prediction v3 — Outline (Reframe Adam → SR)

**Status**: Draft outline (Day 3, 2026-05-04). NOT a full v3 doc.
Awaiting Adam diagnostic verdict + Nick reframe approval before promoting to full v3.

**Trigger**: Day 3 quick_adam_diagnostic showed vanilla Adam ceiling at rel_err ≈ 65% on 4×4 J1-J2 α=2 RBM. Bukov 2021 4×4 SR achieves ~10⁻³ rel_err. Adam cannot serve as a meaningful baseline for "TCBM finds ground state" claims. Reframe: SR becomes primary baseline; Adam demoted to ablation.

---

## What changes (vs v2)

### Change 1 — §1.2 P0-1.2 success criteria (4-tier replacement)

v2: 单一 rel_err < 10%

v3: 4 个 layered tiers, paper 强度依赖几 tier 通过:

| Tier | Metric | Threshold | Claim Level |
|------|--------|-----------|-------------|
| 1 (基础 feasibility) | rel_err_debiased vs ED truth | < 10% | TCBM works (claim A) |
| 2 (NQS 可比性) | rel_err_TCBM / rel_err_SR | ≤ 2.0 | competitive accuracy |
| 3 (核心 robustness) | σ_TCBM / σ_SR (15 seeds) | ≤ 0.5 | claim B (primary) |
| 4 (核心 mechanism) | n_kept (TCBM ⊥ SR) | ≥ 5 / 20 | claim C (mechanism) |

Paper outcome dependency:
- Tier 1+2+3+4 全 pass: NMI strong paper
- Tier 1+3+4 pass (Tier 2 fail): publishable, narrower narrative ("TCBM σ 更小但 absolute 弱")
- Tier 1+2+3 pass (Tier 4 fail): narrower claim ("σ 更小, mechanism 不清楚")
- 仅 Tier 1 pass: re-evaluation needed (probably 6×6 pivot)

### Change 2 — §3 Hypotheses (rename + add)

| Old (v2) | New (v3) |
|----------|----------|
| H1: TCBM converges faster than Adam on J1-J2 4×4 | H1 (renamed): TCBM σ_seed ≤ 0.5 × SR σ_seed (robustness, primary claim) |
| H2: TCBM final E ≤ Adam final E | H2 (NEW): TCBM-hybrid (gradient + QGT subspace) learns directions n_kept ≥ 5 out of k=20 that are SR-orthogonal (mechanism, paper claim C) |
| H3: TCBM clamping subspace catches SR-missed barriers | H3 (reframed): clamping subspace ≠ QGT principal subspace; TCBM and SR explore complementary directions on the variational manifold |

**Removed**: implicit assumption "Adam is the optimizer to beat" — it isn't, on this problem class.

**New H2 quantification**:
- TCBM hybrid mode tracks `subspace_TCBM` (SVD top-k of clamped gradient buffer)
- SR optimizer (when run with `record_qgt_eigvals=True`) gives `subspace_SR` (top-k QGT eigvecs)
- `n_kept = number of TCBM directions with cosine similarity < calibrated_threshold to all SR directions`
- Claim C passes if `n_kept ≥ 5` (≥25% of TCBM subspace is SR-orthogonal)

**H2 measurement procedure (Day 7-8)**:

Step 1 (Day 7): Random null calibration
  - Generate 50 pairs of random k=20 subspaces in D=1120 ambient space
  - Compute n_kept for each pair (using cosine threshold to be calibrated below)
  - Get random_null_n_kept distribution; record p5, p50, p95

Step 2 (Day 7): Calibrate cosine threshold
  - Vary cosine threshold from 0.1 to 0.5 (steps of 0.05)
  - Find threshold where random_null_p95 ≈ 5 (i.e., expected n_kept under null is ≤5)
  - Use this calibrated threshold for actual TCBM vs SR measurement

Step 3 (Day 8): TCBM vs SR n_kept measurement
  - subspace_TCBM = SVD top-20 of TCBM hybrid clamped buffer (latest update)
  - subspace_SR = top-20 QGT eigvectors (from SR run with record_qgt_eigvals=True)
  - Compute cosine similarity matrix
  - n_kept = TCBM directions with cosine < calibrated_threshold to ALL SR dirs

Step 4 (Day 8): Statistical test
  - n_kept > random_null_p95: significant orthogonality (claim C strong)
  - n_kept in [random_null_p5, random_null_p95]: indistinguishable from null (claim C weak)
  - n_kept < random_null_p5: anti-correlated (TCBM stays close to SR, claim C fails)

### Change 3 — §5.4 success thresholds (table replacement)

**v2 table** (3 rows, σ_Adam-based):
| Metric | Threshold | R-abort |
|--------|-----------|---------|
| rel_err_debiased | < 10% | > 15% |
| σ_TCBM / σ_Adam | ≤ 0.5 | > 0.8 |
| swap_acc avg | ∈ [0.2, 0.6] | < 0.1 or > 0.9 |

**v3 table** (5 rows, σ_SR-based + n_kept):
| Metric | Threshold (PASS) | R-abort | Source |
|--------|------------------|---------|--------|
| rel_err_debiased (vs ED) | < 10% | > 15% | TCBM final-eval mean |
| rel_err_TCBM / rel_err_SR | ≤ 2.0 | > 5.0 | TCBM vs SR final-eval |
| σ_TCBM / σ_SR (15 seeds) | ≤ 0.5 | > 0.8 | Week 3 robustness sweep |
| n_kept (SR-orthogonal directions) | ≥ 5 / 20 | < 3 / 20 | Day 8+ subspace overlap analysis |
| swap_acc avg | ∈ [0.2, 0.6] | < 0.1 or > 0.9 | TCBM PT diagnostic |

### Change 4 — §6 R-abort triggers (revisions)

| Trigger | v2 wording | v3 wording |
|---------|-----------|-----------|
| R-abort-1 | rel_err > 15% (vs ED) | unchanged |
| R-abort-2 | swap_acc < 0.1 or > 0.9 | unchanged |
| R-abort-3 | σ_TCBM / σ_Adam > 0.8 | **σ_TCBM / σ_SR > 0.8** |
| R-abort-4 | (n/a in v2) | (n/a in v3) |
| R-abort-5 (NEW) | — | **n_kept < 3 → mechanism claim C fails; demote paper to "TCBM matches SR robustness" only** |

R-abort-5 is the killer for the mechanism story: if TCBM's clamping subspace is fully contained within SR's curvature subspace, TCBM is just SR with extra steps, not a complementary technique. Day 8+ measurement decides.

### Change 5 — §6.7 Stage V judgment matrix (reflow)

Unchanged structure. Just update Adam columns → SR columns. Day 6+ deliverable.

### Change 6 — §5.6 Stage IV deliverables (additions)

Add to existing list:
- Implementation of `core/sr_optimizer.py` (Day 5)
- `experiments/run_sr_baseline.py` single-seed run (Day 6)
- `experiments/run_sr_robustness_sweep.py` 15-seed run (Day 17-19, scaled from Adam sweep design)
- New diagnostic: `analyze_subspace_overlap.py` — computes `n_kept` between TCBM and SR principal subspaces

---

## What does NOT change

- §1.1 problem formalization (4×4 J1-J2 PBC, J2/J1=0.5, ED truth E_0=-8.4579)
- §1.3 magnitude analysis
- §2 Stage II energy-barrier topography (Bukov Fig 11/12 evidence stays load-bearing — SR also lives on this landscape)
- §4.1-4.3 ψ probe theory (mechanism narrative)
- §5.4 Stage IV gate (TCBM-feasibility check)
- §7.1 §8.3 final decision tree (just swap "Adam" → "SR" in baseline references)
- All Bukov 2021 anchor numbers (Fig 11 Hessian spectrum, Fig 12 σ_seed ≈ 0.34%, Fig 8 partial learning ≈ 10⁻⁴)

---

## Open questions for Nick

1. **n_kept threshold (5 of 20)**: arbitrary cutoff, sensitivity analysis needed Day 8+
2. **rel_err_TCBM / rel_err_SR ≤ 2.0**: 2× headroom is generous; if Bukov 4×4 SR achieves 0.1% then 2× = 0.2% rel_err, much tighter than 10% absolute. Worth re-evaluating once SR baseline number is in.
3. **Adam ablation**: RESOLVED — keep, reduce to 5 seeds (was 15 in v2).
   - Value: SI 演示 vanilla gradient descent fails on NQS (rel_err 65%) → 衬托 PT 价值
   - Cost: 5 seeds × 30 min = 2.5h (60% reduction from v2 plan)
   - Implementation: existing run_adam_baseline.py + already-collected Day 3 data

---

## Workflow implications

**Day 5** (was: TCBM tuning):
- Now: implement `core/sr_optimizer.py` (4 helpers + SROptimizer.optimize)
- Pytest: SR ground state on 2×2 J1-J2 within 5% of ED truth (cheap correctness check)

**Day 6** (was: TCBM run):
- Now: run_sr_baseline.py single-seed; verify rel_err ≤ 1% (Bukov-anchored)
- If rel_err > 5%: SR impl bug, fix before any TCBM comparison

**Day 7-9** (was: TCBM rerun + variants):
- Add: subspace overlap analyzer
- Add: TCBM-hybrid mode wired to record subspace evolution

**Day 17-19** (was: 15-seed Adam sweep + 15-seed TCBM sweep):
- Now: 15-seed SR sweep + 15-seed TCBM sweep + 5-seed Adam ablation (60% of original cost since Adam dropped from 15 to 5)

---

## Reframe risks (be explicit)

1. **SR may be too good (Tier 2 FAIL)**:

   Empirical scenario: SR achieves rel_err 0.1%, TCBM achieves 0.5% (5× worse) → Tier 2 fail
   But TCBM may still pass Tier 3 (σ ratio < 0.5).

   Narrative defense:
   - Empirical: rel_err_TCBM 增加 X% but σ_TCBM 降低 50% — accuracy/robustness trade-off
   - Theoretical: parallel tempering 引入的 inter-replica diversity 必然以一定 convergence
     accuracy 换取 variance reduction (well-known in optimization theory)
   - Practical use case: production NQS simulations (quantum chemistry, materials design)
     value reliability over single-shot accuracy (e.g., uncertainty quantification needs
     low σ over multiple runs, not lowest single E)
   - Field precedent: simulated annealing vs gradient descent in classical optimization
     established this trade-off; TCBM brings it to NQS

   Required citation: find one quantum chem / materials NQS paper that emphasizes
   "reliability over accuracy" framing. Day 6 SI prep includes this lit search.
2. **n_kept may be 0**: if TCBM clamping subspace is fully contained in SR's QGT principal subspace, the mechanism story collapses. Day 8 measurement is binary signal — if n_kept = 0, immediately consider switching to a different problem class (e.g., longer-range J1-J2-J3) where the two subspaces could plausibly differ.
3. **SR implementation cost (Plan A/B/C hedge)**:

   Plan A (default): SR runs by Day 6, full reframe schedule on track.

   Plan B (Day 7 SR not converging): Switch to NetKet fork for SR baseline.
   - Day 8 install NetKet, fork Examples/HeisenbergJ1J2/heisenbergJ1J2.py
   - Day 9 adapt L=4, J2=0.5, run baseline in jax
   - Day 10 extract data, compare with PyTorch TCBM (SI explains hybrid setup)
   - Cost: +2-3 days, schedule slips to Day 22-24
   - Acceptable: paper still clean, just longer

   Plan C (Day 10 NetKet also fails): Fallback to Adam-only paper.
   - Reframe Adam ceiling as "negative result + mechanism contribution"
   - Paper narrower: "TCBM PT mechanism studied, comparison with NQS-standard SR
     deferred to follow-up"
   - Don't switch to this until Day 10 Plan B is provably stuck.

   Decision rule: Plan A → B at Day 7 16:00 if SR rel_err > 5% on 4×4 single seed.
                  Plan B → C at Day 10 16:00 if NetKet not running cleanly.

---

## Decision needed from Nick

- (A) Approve outline → I write full v3 Day 4 evening
- (B) Modify outline → revise then write full v3
- (C) Reject reframe → keep v2 Adam-baseline, find different way to frame Adam ceiling
