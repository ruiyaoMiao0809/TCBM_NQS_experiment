# TCBM-NQS POC Day 3 → Day 4 Context Patch

**Patch version**: v6 (TCBM Project, NQS sub-project)
**Generated**: 2026-05-04 evening, end of Day 3
**Status**: Day 3 closed (9 commits), Day 4 ready to start
**Codebase**: `/home/dglg/miao_workplace/tcbm_nqs/` on 4× A10 24GB server (HKUST-GZ lab)

---

## Part 1: Project context (necessary background)

### 1.1 NQS POC overall positioning

This is a 3-week POC sub-project of the main TCBM project (DC-OPF success on IEEE 118-bus + CEC2017 synthetic benchmarks).

**Goal**: validate TCBM optimizer on quantum many-body variational problem (4×4 J1-J2 frustrated Heisenberg AFM, RBM ansatz) for inclusion in NMI submission as a third benchmark domain.

**Original schedule**: Day 0-21, target NMI submission end of Day 21.

**Three weeks budget**:
- Week 1 (Day 1-7): prep, diagnostic, baseline implementation
- Week 2 (Day 8-14): main runs (TCBM + SR + Adam)
- Week 3 (Day 15-21): robustness sweep + SI writing

### 1.2 Physical system

**4×4 J1-J2 Heisenberg AFM, J2/J1=0.5, periodic boundary conditions**:
- 16 spin-1/2 sites
- 32 nearest-neighbor bonds (J1=1.0)
- 32 next-nearest-neighbor diagonal bonds (J2=0.5)
- S_z=0 sector dim = 12,870 (Hilbert subspace)
- ED ground state energy E_0 = -8.4579 (verified Day 1, was hallucinated as -0.497 earlier)
- Maximally frustrated (Marshall-Peierls trap basin exists)

### 1.3 Algorithm setup

**RBM ansatz, α=2 (complex)**:
- N = 16 visible (spin sites)
- M_hidden = 32 (αN)
- Total complex params = 16·32 + 16 + 32 = 560
- Total real params D = 1,120 (TCBM optimizes over this)

**TCBM optimizer cfg (production)**:
- M = 12 replicas (parallel tempering)
- n_steps = 3000
- k = 20 (clamping subspace dim)
- T_min = 0.005, T_max = 2.0
- subspace_warmup = 150, subspace_update_freq = 80
- psi_star = 1.5, min_warmup = 120
- grad_buffer_size = 400, grad_buffer_use = 200
- subspace_source = 'gradient' (P0-1.2) or 'hybrid' (Day 5+)
- n_vmc_samples = 2000
- n_vmc_samples_final = 2 (was 4 in v1, reduced Day 3 for ~30% wall-clock saving)

### 1.4 Bukov 2021 anchor (verbatim from arXiv 2011.11214)

**Critical reference for paper SI**, three precise readings from Nick visual inspection (Day 3 c02339b):

**Fig 11 Hessian spectrum**:
- 4×4 (iter 499, E_gs=-8.457917): |λ_min|≈10⁻², |λ_max|≈10⁵, ratio ≈ 4×10⁻⁷ (flat directions)
- 6×6 (iter 675, E_gs=-18.073818): |λ_min|≈10², |λ_max|≈10⁹, ratio ≈ 10⁻⁷ (sharp narrow valleys)
- Same ratio at both sizes BUT absolute |λ_min| differs 10⁴×
- **Implication for TCBM mechanism**: 4×4 lacks sharp directions, so TCBM advantage is "diversity injection" not "narrow valley detection"

**Fig 12 (4 seeds, N=6×6)**:
- Final |E-EGS|/N range: 3-7×10⁻³
- Mean ≈ 4×10⁻³, std ≈ 1.7×10⁻³
- σ/|E| ≈ 0.34% (NOT 5-10% as previously hallucinated)

**Fig 8 left (N=4×4 partial learning)**:
- log|ψ| optimization final: ≈ 3×10⁻⁴
- φ optimization final: ≈ 5×10⁻⁵
- Full learning 4×4 expected ~10⁻³

### 1.5 Schedule constraint

**Hard cap: Day 21 NMI submission**

Current slip: 1 day (Day 3 production v2 not launched, used for Adam diagnostic).
Buffer remaining: ~1 day.
Risk: if SR implementation Day 5-7 overruns, will tighten to Day 22-24 (Plan B).

---

## Part 2: All work done in this chat (Day 2 → Day 3)

### 2.1 Day 2 (2026-04-28) — recap

**Catastrophic baseline failure but 4 forward-work commits**.

Day 2 baseline run launched 09:06, SIGTERM at 15:28 (6h22min wall), **0 bytes data salvaged**.

Three engineering defects caused 100% data loss:
1. Python stdout block-buffered through tee (no flush)
2. optimize() internal loop had no progress print
3. SIGTERM doesn't trigger atexit by default; no intermediate checkpoint

GPU usage during this run was completely fine (you owned GPU 3 alone). The 6h22min was algorithm reality (estimate was 30 min, 6.5× underestimate due to forgetting M=12 replica multiplier × VMC per-replica cost).

**Day 2 commits (4)**:
- `10d4981` — L_basin caveat to Day 1 log
- `07e005e` — Bukov 2021 anchor + 3 hallucination fixes (5-10% σ → 0.34%, hump → marked hallucination, λ ratio → marked hallucination)
- `b8908c7` — failed baseline run + heartbeat data salvage
- `09d29be` — Day 2 daily_log entry

**Day 2 known_issues_day2.md**: 3 items recorded:
1. VMC evaluate cost severely underestimated (Day 8-9 batched evaluate fix path)
2. record_every vs swap_interval LCM stale-value
3. ptrace_scope=1 limits stack inspection

### 2.2 Day 3 (2026-04-29) — full breakdown

**9 commits, 1 critical bug discovery, 1 major reframe**.

#### 2.2.1 Phase A: Forward work (no GPU, 06:00-13:00 UTC)

**A0: GPU watcher** (PID 2700907)
- Bash bg loop, every 5 min check 4 GPUs free mem
- Logs to `logs/gpu_watch_day3.log`
- Triggers `*** READY ***` when free ≥ 22 GB

**A1: Callback hook in TCBMOptimizer** (commit `7580cc6`)
- Added `callback: Optional[Callable[[int, Dict], None]]` parameter to `optimize()`
- Calls every `callback_every` steps (default 100)
- Wrapped in try-except (callback failure doesn't crash optimizer)
- 4 new pytest cases verify callback fires + info dict has 8 keys + backward compat
- All 16 Day 1 tests still pass (no regressions)

**A2: Production v2 script** (commit `fddbc6f`)
- `experiments/run_gradient_baseline_v2.py` with three robustness fixes:
  - `sys.stdout.reconfigure(line_buffering=True)`
  - `signal.signal(SIGTERM, ...)` triggering `sys.exit(0)`
  - `atexit.register(...)` partial JSON dump
- Hard checkpoint every 500 steps to `results/baseline_v2_checkpoint.json`
- `n_vmc_samples_final = 2` (reduced from 4 to save ~1.3h)
- `TCBM_DEVICE` env var for GPU selection (replaces hardcoded `cuda:3`)
- `assert cfg.subspace_source == 'gradient'` to prevent silent drift

**A3: Micro-benchmark script** (commit `fddbc6f`)
- `experiments/benchmark_per_step.py` runs 100 steps + final eval
- Decision rule: <4h projected GO, 4-8h ASK, >8h ESCALATE
- Was meant to be run before production v2, but production was never launched (RNG bug discovered first)

#### 2.2.2 Adam baseline + Bukov anchor (afternoon Day 3)

**Adam baseline skeleton** (commit `d86bff9`)
- `experiments/run_adam_baseline.py` ~270 lines
- Same robustness three-pack as v2
- Three design choices Claude Code caught:
  1. `evaluate()` returns `(cost, penalty, sigma_E)` not `(cost, sigma_E, _)`
  2. final_cost = `mean(4 shots)` NOT `min` (avoids selection bias)
  3. final eval target = final theta NOT best_x (Adam single-replica, no replica selection bias)

**Final-eval asymmetry note** (commit `f2d8223`)
- `docs/known_issues_day2.md` updated: TCBM uses best_x, Adam uses final_θ — paper SI must disclose
- TCBM/Adam both use `mean()` for `best_cost_debiased` — verified at `core/tcbm_optimizer_NQS.py:1139`

**Bukov Fig 11/12/8 precise readings** (commit `c02339b`)
- `docs/bukov_2021_anchor.md` updated with Nick visual inspection readings
- 3 placeholder paragraphs replaced with actual figures
- Critical insight: 4×4 vs 6×6 |λ_min| differs 10⁴× in absolute magnitude

#### 2.2.3 Adam diagnostic v1 (broken, evening Day 3)

**Why we ran Adam diagnostic instead of TCBM production**: 

Nick raised valid scientific concern — TCBM previously succeeded on DC-OPF (penalty 10¹¹) and CEC2017 (sharp narrow peaks). Bukov Fig 11 showed 4×4 has flat negative curvature ("bowl-shape"). If 4×4 is too smooth, σ_TCBM/σ_Adam ≈ 1, R-abort-3 triggers.

Need to test 4×4 ruggedness via 5-seed Adam baseline before committing 3 weeks.

**Initial v1 launch** (16:44 UTC):
- `experiments/quick_adam_diagnostic.py` — 5 seeds [42, 7, 13, 21, 99]
- Adam M=1, n_steps=2000, n_samples=2000, n_final=4 (mean)
- Cluster analysis (gap_ratio threshold 0.5)
- Drift analysis (final - best_during)
- TCBM_DEVICE=cuda:3 (the only free GPU)

**v1 ran seed 42 + seed 7, both finished with byte-identical results**:
- Seed 42 final_cost = -2.4822, rel_err 70.65%
- Seed 7 final_cost = -2.4822, rel_err 70.65% (same to 4 decimal places)
- Final shots `[-2.5053, -2.5002, -2.4331, -2.4904]` byte-identical for both seeds

**This was a critical discovery.**

#### 2.2.4 RNG bug — codebase-wide critical fix (commit `18716b3`)

**Root cause** (Claude Code grep):
```
core/j1j2_problem.py:319-320
    self._gen = torch.Generator(device=device)
    self._gen.manual_seed(0)   # users override via optimizer's seed
```

`J1J2Problem` has internal `torch.Generator` hardcoded to seed=0. Used in 4 places:
- Line 345: `random_feasible` (init theta)
- Line 487: VMC chain init
- Line 496: Metropolis flip site
- Line 506: Metropolis accept log_u

`torch.manual_seed(seed)` and `np.random.seed(seed)` at script level **don't affect this internal generator**. The comment "users override via optimizer's seed" was a lie — no optimizer reset it.

**Impact scope (CRITICAL)**:
- All Day 1-3 production runs were effectively seed=0
- Day 17-19 robustness sweep (15 seeds × 4 methods × 4×4 = 60 runs) would have produced **identical results** → σ = 0 → entire sweep vacuous
- Day 2 baseline 6h22min run was also seed=0 effective

**Fix** (1 method to J1J2Problem + 2-line patch to scripts):
```python
def set_seed(self, seed: int):
    """Reset internal RNG to given seed."""
    self._gen.manual_seed(seed)
    return self
```

Production scripts updated:
```python
problem = J1J2Problem(...)
problem.set_seed(SEED)  # NEW LINE
theta = problem.random_feasible(1).requires_grad_(True)
```

3 new pytest tests in `tests/test_rng_seeding.py`:
- `test_problem_set_seed_changes_init` (different seeds → different theta)
- `test_problem_set_seed_reproducible` (same seed twice → identical theta)
- `test_evaluate_seeded_reproducible` (same seed + theta → same evaluate result)

All 23 tests pass (16 Day 1 + 4 Day 3 callback + 3 RNG seeding).

#### 2.2.5 v3 outline reframe (commits `8ee7ebf`, `2a99dbf`)

**Trigger**: v1 Adam ceiling was rel_err 65% (later confirmed 58% with proper RNG). Bukov 2021 4×4 SR achieves ~10⁻³ rel_err. Vanilla Adam not a meaningful baseline for "TCBM finds ground state" claim.

**Decision**: Reframe Adam → SR primary baseline.

**v3 outline (~250 lines)** in `docs/NQS_J1J2_Prediction_v3_outline.md`:

**6 Changes vs v2**:
1. **§1.2 Success criteria → 4-tier**:
   - Tier 1 (feasibility): rel_err < 10% vs ED
   - Tier 2 (NQS comparable): rel_err_TCBM/rel_err_SR ≤ 2.0
   - Tier 3 (robustness): σ_TCBM/σ_SR ≤ 0.5 (15 seeds)
   - Tier 4 (mechanism): n_kept ≥ 5/20 (TCBM ⊥ SR)
2. **§3 Hypotheses renamed**:
   - H1: TCBM σ_seed ≤ 0.5 × SR σ_seed (primary claim B)
   - H2 NEW: n_kept ≥ 5/20 directions SR-orthogonal (mechanism claim C)
   - H3: TCBM and SR explore complementary directions
3. **§5.4 thresholds**: 5-row table replacing 3-row, σ_SR-based + n_kept
4. **§6 R-aborts**: R-abort-3 σ_TCBM/σ_SR > 0.8, R-abort-5 NEW (n_kept < 3 demotes mechanism claim)
5. **§6.7 reflow**: Adam columns → SR columns
6. **§5.6 deliverables**: SR optimizer, SR baseline, subspace overlap analyzer, sweep

**H2 measurement procedure (Day 7-8)**:
- Step 1: Random null calibration (50 pairs of random k=20 subspaces in D=1120 ambient)
- Step 2: Calibrate cosine threshold (find threshold where random_null_p95 ≈ 5)
- Step 3: TCBM vs SR n_kept measurement
- Step 4: Statistical test against null distribution

**3 Reframe Risks with hedges**:
1. SR may be too good (Tier 2 fail) → narrative defense: "accuracy/robustness trade-off" + 4 bullet points (empirical/theoretical/practical/field precedent) + Day 6 lit search task
2. n_kept may be 0 → if so, switch to longer-range J1-J2-J3 problem class
3. SR implementation cost → Plan A/B/C hedge:
   - **Plan A (default)**: SR runs by Day 6
   - **Plan B (Day 7 16:00 trigger)**: NetKet fork (Day 8-12, +2-3 days slip to Day 22-24)
   - **Plan C (Day 10 16:00 trigger)**: Adam-only fallback paper

**3 Open Questions for Nick**:
1. n_kept threshold (5/20) — sensitivity analysis Day 8+
2. rel_err_TCBM/rel_err_SR ≤ 2.0 — re-evaluate once SR baseline number in
3. Adam ablation: RESOLVED (keep, reduce 15→5 seeds, reuse Day 3 data)

**SR optimizer skeleton** in `core/sr_optimizer.py` (197 lines, Day 5 implement):
- `SRConfig` dataclass: lr, epsilon, epsilon_mode, solver, record options
- `SROptimizer` class with `NotImplementedError`
- 4 helper functions (free): `compute_log_derivatives`, `compute_qgt`, `regularize_qgt`, `solve_sr_update`

**SR baseline script skeleton** in `experiments/run_sr_baseline.py` (287 lines):
- 100% JSON schema parity with `run_adam_baseline.py` (PT-only fields = None)
- 4 SR-specific fields in progress_history sub-tree (qgt_cond_history, sr_residual_history, qgt_eigvals_final)

#### 2.2.6 Adam diagnostic v2 (RNG-fixed) — final scientific output

**Launch**: 16:44 UTC, completed 19:21 UTC (~2.5h, 5 seeds sequential on GPU 3).

**Per-seed results**:

| Seed | best_during | final | drift | rel_err |
|------|-------------|-------|-------|---------|
| 42 | -4.35 | -3.53 | +0.82 | 58.30% |
| 7 | -2.91 | -1.32 | +1.59 | 84.40% |
| 13 | -4.05 | -2.08 | +1.98 | 75.47% |
| 21 | -3.89 | -2.46 | +1.43 | 70.86% |
| 99 | -4.13 | -1.42 | +2.71 | 83.22% |

**5 signals (analyze_diagnostic_post.py)**:

| Signal | Value | Interpretation |
|--------|-------|----------------|
| σ_Adam_rel | **9.495%** | 5× strong threshold, big spread |
| Cluster (gap_ratio) | **0.48** | BORDERLINE (under 0.5 strict cutoff) |
| Mean drift | **+1.705** | LARGE_DRIFT (5/5 seeds positive) |
| Plateau escape | 1/5 | seed 99 only |
| Combined | 2/4 strong + 4/4 weak | MODERATE_RUGGEDNESS |

**Nick's reading (more accurate than Claude Code's MODERATE label)**: STRONG multi-basin signal.

Reasoning:
- σ 9.5% is 5× threshold — very large
- 5/5 seeds positive drift, mean +1.7 — consistent across seeds (Adam crosses saddle into worse basin)
- 2-cluster structure: seed 42 outlier at -3.53, rest in [-2.5, -1.3]
- gap_ratio 0.48 is borderline due to **5-seed sample insufficient** for 0.5 strict threshold, not because it's noise floor
- Adam reaches better basin (best_during ~-4.0 across 4/5 seeds) but cannot hold (drift)

**Implication for Day 4+**:
- 4×4 retained (no 6×6 pivot)
- v3 outline Plan A path confidence: 60% → 80%
- σ_TCBM/σ_Adam ≤ 0.5 easily achievable (predicted σ_TCBM ~1-3%, σ_Adam = 9.5%)
- σ_TCBM/σ_SR ≤ 0.5 may be challenging (Bukov 6×6 SR σ ≈ 0.34%, our 4×4 SR σ likely 0.5-1.5%)
- v3 outline Tier 3 may need **dual-baseline robustness claim**:
  - σ_TCBM/σ_Adam ≤ 0.5 (against unstable baseline)
  - σ_TCBM ≤ 2× σ_SR (within 2× of best baseline)

**Day 3 final commit (`<NEW>`, force-add JSONs)**:
- `experiments/quick_adam_diagnostic_v2.py`
- `experiments/quick_adam_diagnostic.py` (v1 buggy, kept as RNG bug evidence — non-reproducible)
- `experiments/analyze_diagnostic_post.py`
- `results/quick_adam_diagnostic.json` (force-added, v1 partial)
- `results/quick_adam_diagnostic_v2.json` (force-added, ~200 KB with full trajectories)

`.gitignore` not modified — `results/*.json` still ignored for future sweep artifacts.

### 2.3 Day 3 commit log (9 commits, all local; 8 on GitHub)

```
<NEW>   Day 3: Adam diagnostic v2 (RNG-fixed) + post-analysis  ← LATEST, not pushed
2a99dbf Day 3: v3 outline followup revisions                   ← not pushed
8ee7ebf Day 3: Prediction v3 outline (Adam → SR reframe)      ← pushed
18716b3 Day 3: RNG fix (4 files, 87 ins)                      ← pushed
c02339b Day 3: Bukov Fig 11/12/8 precise readings             ← pushed
d86bff9 Day 3: Adam baseline (P1-1.1) skeleton                ← pushed
f2d8223 Day 3: TCBM vs Adam final-eval target asymmetry       ← pushed
fddbc6f Day 3 A2/A3: production v2 + micro-benchmark scripts  ← pushed
7580cc6 Day 3 A1: callback hook to TCBMOptimizer.optimize()   ← pushed
```

GitHub origin/main: `8ee7ebf` (last pushed). Local 2 commits ahead.

---

## Part 3: Latest NQS experiment plan (post-Day 3 reframe)

### 3.1 v3 outline summary (full doc TBD Day 4 evening)

**Primary claim B**: σ_TCBM ≤ 0.5 × σ_SR (robustness)
**Secondary claim C**: TCBM-hybrid learns directions n_kept ≥ 5/20 SR-orthogonal (mechanism)

**Baseline restructure**:
- **Primary baseline**: SR (NaturalGradient + QGT, NQS field standard)
- **Secondary baseline**: Adam (5 seeds, ablation only)
- **Comparison method**: TCBM-hybrid (subspace_source='hybrid', mechanism showcase)

### 3.2 4-tier success criteria

| Tier | Metric | Threshold | Claim Level |
|------|--------|-----------|-------------|
| 1 (basic) | rel_err_debiased vs ED | < 10% | TCBM works |
| 2 (NQS comp) | rel_err_TCBM/rel_err_SR | ≤ 2.0 | competitive |
| 3 (robustness) | σ_TCBM/σ_SR (15 seeds) | ≤ 0.5 | claim B |
| 4 (mechanism) | n_kept (TCBM ⊥ SR) | ≥ 5/20 | claim C |

**Paper outcome dependency**:
- Tier 1+2+3+4 all pass: NMI strong paper
- Tier 1+3+4 pass (Tier 2 fail): publishable narrower claim
- Tier 1+2+3 pass (Tier 4 fail): narrower mechanism claim
- Only Tier 1: re-evaluate (likely 6×6 pivot)

### 3.3 R-abort triggers

| Trigger | v3 condition |
|---------|--------------|
| R-abort-1 | rel_err > 15% vs ED |
| R-abort-2 | swap_acc < 0.1 or > 0.9 |
| R-abort-3 | σ_TCBM/σ_SR > 0.8 (was σ_TCBM/σ_Adam in v2) |
| R-abort-5 (NEW) | n_kept < 3 → demote mechanism claim |

### 3.4 Reframe risks + hedges

**Risk 1: SR may be too good (Tier 2 fail)**
- Narrative defense: accuracy/robustness trade-off
- 4 bullet points: empirical, theoretical, practical (production NQS values reliability), field precedent (simulated annealing vs gradient descent)
- Day 6 SI prep: lit search for "reliability over accuracy" citation in quantum chem / materials NQS

**Risk 2: n_kept = 0**
- If TCBM clamping fully contained in SR's QGT subspace
- Switch to longer-range J1-J2-J3 problem class
- Day 8 measurement is binary signal

**Risk 3: SR implementation cost — Plan A/B/C**
- **Plan A (default)**: SR runs by Day 6
- **Plan B (trigger Day 7 16:00 if SR rel_err > 5%)**: Fork NetKet `Examples/HeisenbergJ1J2/heisenbergJ1J2.py`
  - Day 8: install NetKet, fork
  - Day 9: adapt L=4, J2=0.5, run baseline in jax
  - Day 10: extract data, compare with PyTorch TCBM
  - Cost: +2-3 days, slip to Day 22-24
- **Plan C (trigger Day 10 16:00 if NetKet fails)**: Adam-only fallback paper

### 3.5 Resource constraints

**4× A10 24GB (HKUST-GZ lab, currently available)**:
- 4×4 NQS production: 1 GPU per run (Day 2 实测 22 GB peak with n_final=4, ~12 GB with n_final=2)
- 4×4 sweep parallelism: 4 runs simultaneous
- 6×6 NQS: NOT viable on A10 (single-run mem ~150 GB needed)

**6×6 viability** (если 4×4 fails or NMI revision requires):
- Plan: ~$50-150 USD on vast.ai H100 80GB
- Alternative: ask Hui Xiong professor for HKUST/HKUST-GZ A100/H100 cluster
- 4× A100 80GB cluster: scenario B (M=8, n=1000) ~10 days for 6 runs
- Strategy: 4×4 first, 6×6 deferred to NMI revision

---

## Part 4: Day 4 detailed plan (next chat starts here)

### 4.1 Day 4 morning tasks (~2-3 hours)

**Step 1: Review Day 3 verdict + current state** (15 min)
```bash
cd /home/dglg/miao_workplace/tcbm_nqs
git log --oneline | head -10                           # confirm 9 commits
cat results/quick_adam_diagnostic_v2.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
print('summary:', data.get('summary', 'N/A'))
"                                                       # re-read 5 signals
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits
                                                        # check GPU availability
```

**Expected state**:
- 9 Day 3 commits (last is force-added JSONs + verdict)
- Adam v2 verdict: STRONG multi-basin signal (5 signals confirmed)
- GPU 3 likely freed up, mem free > 22 GB

**Step 2: Write Day 3 daily_log entry** (~30 min)
- New file `docs/daily_log_day3.md` (will merge into main `docs/daily_log.md`)
- Required sections:
  - 计划 vs 实际 (Phase A complete, Phase B replaced by Adam diagnostic, RNG fix detour)
  - 关键数字 table (5 signals, 5 seed costs, 9 commits)
  - 发现/问题 (RNG bug discovery, MODERATE→STRONG ruggedness reading, dual-baseline robustness option)
  - 红线状态: R-abort-1 still inconclusive, R-abort-2/3/5 pending
  - Day 3 反思 (5 lessons learned):
    1. RNG infrastructure 必须在 day 1 显式 test 跨 seed reproducibility
    2. Wall-clock 估算系统性低估 (Day 2 6h22min 教训仍 actionable)
    3. force-add 优于 修改 .gitignore 做 one-off override
    4. v3 outline 应当 explicit document Plan A/B/C trigger conditions, 不能是 vague "if SR fails"
    5. Adam ceiling reframe: best_during 21% (v1 fluke) → 49% (v2 真实) → 4×4 仍 multi-basin 但 Adam 真不能 reach SR-level
- Commit: `git add docs/daily_log.md && git commit -m "Day 3: daily_log entry"`

**Step 3: Push pending commits to GitHub** (5 min)
```bash
git push origin main                                   # pushes 2a99dbf, <new>, daily_log
                                                        # 3 commits ahead → 0 ahead
```

**Step 4: v3 outline Reframe Risk 1 数字 update** (~5 min)
- Current outline says "Adam ceiling 65%"
- v2 实测: final mean ~ -2.16 → rel_err 74% (across 5 seeds), best_during ~-3.87 → rel_err 54%
- Update Reframe Risk 1 narrative defense Adam ceiling reference
- Commit: `git add docs/NQS_J1J2_Prediction_v3_outline.md && git commit -m "Day 4: v3 outline Adam ceiling number update with v2 verified data"`

### 4.2 Day 4 afternoon tasks (~3-4 hours)

**Step 5: Promote v3 outline to full v3 doc** (~2 hours)

Create `docs/NQS_J1J2_Prediction_v3.md` (target ~800-1000 lines, expand from 250-line outline):

Required sections (preserve v2 section IDs for diff clarity):
- §1.1 Problem formalization (unchanged from v2)
- §1.2 Success criteria — 4-tier replacement (from outline Change 1)
- §1.3 Magnitude analysis (unchanged from v2)
- §2 Stage II energy-barrier topography (unchanged from v2, Bukov anchors stay)
- §3 Hypotheses (renamed H1/H2 NEW/H3 reframed from outline Change 2)
- §3.1 H2 measurement procedure detailed (4 steps with code-level spec)
- §4.1-4.3 ψ probe theory (unchanged from v2)
- §5.4 success thresholds — 5-row table (from outline Change 3)
- §5.6 deliverables list expanded (from outline Change 6)
- §6 R-aborts — full revisions table (from outline Change 4)
- §6.7 Stage V judgment matrix reflow (from outline Change 5)
- §7.1, §8.3 final decision tree updated
- New §9: Adam → SR reframe rationale (cite Day 3 v2 verdict)
- New §10: Reframe risks + Plan A/B/C hedges (from outline Reframe risks)

Commit: `git add docs/NQS_J1J2_Prediction_v3.md && git commit -m "Day 4: full v3 doc (Adam → SR reframe complete)"`

**Step 6: Launch production v2 TCBM-gradient run** (~5-6 hours wall, parallel with Step 7)

```bash
cd /home/dglg/miao_workplace/tcbm_nqs
mkdir -p logs

# Verify GPU 3 free
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | grep '^3,'
# Expect ≥ 22 GB

# Launch production v2 with RNG fix
TCBM_DEVICE=cuda:3 python -u experiments/run_gradient_baseline_v2.py \
  2>&1 | tee logs/production_v2_seed42_$(date +%Y%m%d_%H%M).log
```

Expected wall: ~5-6h (跟 Day 2 类似 main loop, n_final=2 saves ~30%)
Expected verdict: rel_err < 10% (Tier 1 PASS) — 4×4 partial learning achievable

Watchdog plan:
- Progress prints every 100 steps (callback hook fires)
- Checkpoint every 500 steps (atexit safety net)
- nvidia-smi heartbeat manual check every 30 min
- If `cost diverges/NaN`: SIGTERM, restart with smaller lr
- If `swap_acc < 0.1 or > 0.9`: R-abort-2 trigger

**Step 7: Day 5 SR implementation prep** (~1 hour, parallel with Step 6 production run)

If Claude Code idle while production runs:
- Read `core/sr_optimizer.py` skeleton (already 197 lines, F1 forward work from Day 3)
- Identify implementation order:
  1. `compute_log_derivatives(problem, theta, samples)` — use `torch.autograd.grad` with `retain_graph=True`
  2. `compute_qgt(log_derivatives)` — `S_ij = E[O_i^* O_j] - E[O_i^*] E[O_j]`, einsum
  3. `regularize_qgt(qgt, eps, mode='scale-aware')` — Bukov-style ε·diag(S)
  4. `solve_sr_update(qgt_reg, gradient, method='pinv')` — `torch.linalg.pinv(S) @ g`
  5. `SROptimizer.optimize(callback)` — main loop with QGT recording
- Write `tests/test_sr_optimizer.py` skeleton (3-5 tests):
  - `test_sr_ground_state_2x2_J2_0`: trivial unfrustrated 2×2 case, |E - E_GS| < 5%
  - `test_qgt_symmetric`: S should be Hermitian
  - `test_qgt_pos_semidef`: eigenvalues ≥ -1e-6 (numerical PSD)
  - `test_log_derivatives_shape`: O matrix shape (n_samples, D)
  - `test_pinv_solver_consistent`: small linear system reproduce truth
- Don't implement yet (Day 5), only design + tests

### 4.3 Day 4 evening tasks

**Step 8: Production v2 verdict** (when run completes ~Day 4 evening or Day 5 morning)
- Read `results/baseline_v2_seed42.json`
- Tier 1 verdict: rel_err < 10% PASS / 10-15% MARGINAL / >15% R-ABORT-1
- Diagnostics check:
  - swap_acceptance early/mid/late (target [0.2, 0.6])
  - T_w trigger (None acceptable, step number better)
  - psi_max cumulative (should reach ~1.5-2.5)
  - NQS-1e debiasing sanity (debiased < raw)

**Step 9: Decision on Day 5 path**
- If Tier 1 PASS: Day 5 launch SR implementation (Plan A on track)
- If Tier 1 MARGINAL: Day 5 retune (n_vmc_samples=4000, subspace_warmup=300, psi_star=1.3) before SR
- If Tier 1 R-ABORT: discuss with Nick — TCBM may need different cfg or 4×4 problem class wrong
- Commit verdict to `docs/daily_log_day4.md`

### 4.4 Day 4 schedule summary

```
Morning (3h):
  - Review Day 3 state + Adam verdict
  - Write Day 3 daily_log entry + commit
  - Push pending commits to GitHub
  - Update v3 outline Adam ceiling number
  
Afternoon (4h):
  - Promote v3 outline → full v3 doc
  - Launch production v2 TCBM-gradient (5-6h wall, async)
  - Day 5 SR impl prep (parallel)

Evening (1h+):
  - Production v2 verdict (when ready)
  - Day 5 path decision
  - Day 4 daily_log entry
```

### 4.5 Day 4 commit budget (target 5-7 commits)

```
Day 4 morning:
  - Day 3 daily_log entry
  - v3 outline Adam ceiling update

Day 4 afternoon:
  - Full v3 doc promotion
  - Day 5 SR test skeleton (optional)

Day 4 evening:
  - Production v2 result + verdict
  - Day 4 daily_log entry
```

---

## Part 5: Workflow notes for next chat

### 5.1 Critical context to preserve

**Always remember**:
1. **RNG bug fixed but historical artifact**: All Day 1-3 production runs (except v2 diagnostic) effectively seed=0
2. **TCBM/Adam final-eval asymmetry**: TCBM uses `best_x`, Adam uses `final_θ` — paper SI must disclose
3. **n_vmc_samples_final = 2**: production v2 uses 2 (not 4), saves ~1.3h on final eval
4. **TCBM_DEVICE env var**: production scripts read `cuda:0` default, override with `TCBM_DEVICE=cuda:N`
5. **GPU peak mem**: 4×4 + n_final=2 ≈ 12-15 GB peak; 4×4 + n_final=4 ≈ 22 GB peak
6. **Schedule cap**: Day 21 NMI submission, currently 1 day slip
7. **Plan B trigger**: Day 7 16:00 UTC if SR rel_err > 5% on 4×4 single seed
8. **Plan C trigger**: Day 10 16:00 UTC if NetKet fork also fails

### 5.2 Engineering principles reinforced (Day 0-3)

1. **Wall-clock estimation**: LLM systematically underestimates 6-10× when forgetting M=replica multiplier
2. **Long-run robustness three-pack mandatory**: `python -u` + `signal.signal(SIGTERM, ...)` + `atexit.register(...)` + intermediate checkpoint every N steps
3. **Cross-seed reproducibility**: every new production code must have a pytest verifying seed reset works
4. **Literature numbers**: always fetch original PDF, never trust LLM prior knowledge for quantitative claims
5. **Force-add for one-off git override**: `git add -f` better than modifying `.gitignore`
6. **Don't amend pushed commits**: use new commit + `git push origin main` instead of force-push
7. **Spot-check Claude Code reports**: `MODERATE` vs `STRONG` verdict labels matter for downstream decisions

### 5.3 File inventory at end of Day 3

**Production code (committed)**:
- `core/j1j2_problem.py` — added `set_seed()` method
- `core/tcbm_optimizer_NQS.py` — added `callback` parameter to `optimize()`
- `experiments/run_gradient_baseline_v2.py` — production with three-pack robustness
- `experiments/run_adam_baseline.py` — Adam M=1 baseline
- `experiments/benchmark_per_step.py` — micro-benchmark
- `experiments/quick_adam_diagnostic_v2.py` — RNG-fixed 5-seed Adam diagnostic
- `experiments/quick_adam_diagnostic.py` — v1 buggy (RNG bug evidence)
- `experiments/analyze_diagnostic_post.py` — 5-signal post-analysis

**Skeletons (NOT committed, Day 5+)**:
- `core/sr_optimizer.py` (197 lines, NotImplementedError, F1 forward work)
- `experiments/run_sr_baseline.py` (287 lines, schema parity, F2 forward work)

**Documentation (committed)**:
- `docs/bukov_2021_anchor.md` — verbatim quotes + Fig 11/12/8 readings
- `docs/known_issues_day2.md` — VMC cost, swap stale, ptrace, Adam asymmetry
- `docs/NQS_J1J2_Prediction_v2.md` — original prediction (pre-reframe)
- `docs/NQS_J1J2_Prediction_v3_outline.md` — Adam → SR reframe outline (250 lines)
- `docs/daily_log.md` — Day 0-2 entries (Day 3 entry pending Day 4 morning)

**Tests (committed)**:
- `tests/test_j1j2_problem.py` — 16 Day 1 tests
- `tests/test_callback.py` — 4 Day 3 callback tests
- `tests/test_rng_seeding.py` — 3 Day 3 RNG tests
- Total: 23 tests, all pass

**Results (force-added JSONs)**:
- `results/quick_adam_diagnostic.json` — v1 partial (3 seeds, RNG bug evidence)
- `results/quick_adam_diagnostic_v2.json` — v2 full (5 seeds, ~200 KB with trajectories)

**Logs (gitignored, ephemeral)**:
- `logs/heartbeat_baseline_20260428_0919.log` — Day 2 heartbeat (already committed as evidence)
- `logs/baseline_seed42_20260428_*.log` — Day 2 stdout (0 bytes)
- `logs/adam_diagnostic_20260429_*.log` — Day 3 v1 stdout
- `logs/adam_v2_20260429_*.log` — Day 3 v2 stdout
- `logs/gpu_watch_day3.log` — GPU watcher heartbeat

### 5.4 Resources & references

- **Codebase**: `/home/dglg/miao_workplace/tcbm_nqs/`
- **Reference codebases**:
  - `/home/dglg/miao_workplace/netket_reference/` (jax, J1-J2 example for Day 8 SR fork)
  - `/home/dglg/miao_workplace/vmc_jax/` (jVMC, no J1J2 example, SR in `util/{tdvp,minsr}.py`)
- **GPU**: 4× NVIDIA A10 24GB, HKUST-GZ lab server
- **Compute provider for 6×6 (if needed)**: vast.ai (~$50-150 USD)
- **Bukov 2021 anchor**: arXiv:2011.11214 (SciPost Phys 10, 147)
- **NMI submission target**: Day 21 (or Day 22-24 if Plan B triggered)

### 5.5 Open decisions for Nick (pending Day 4+)

1. **n_kept threshold**: 5/20 is arbitrary, will calibrate vs random null Day 7
2. **rel_err_TCBM/rel_err_SR ≤ 2.0**: re-evaluate once SR baseline data in (Day 6)
3. **Adam ablation 5 vs 10 seeds**: currently 5, may bump to 10 if robustness comparison thin
4. **Dual-baseline Tier 3**: σ_TCBM/σ_Adam ≤ 0.5 AND σ_TCBM ≤ 2× σ_SR — should be added to v3 doc
5. **6×6 future work**: defer to NMI revision response, prepare claim narrative

### 5.6 Quick start commands for Day 4 (next chat)

```bash
# 1. Verify state
cd /home/dglg/miao_workplace/tcbm_nqs
git log --oneline | head -10
git status
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits

# 2. Re-read Adam verdict
cat results/quick_adam_diagnostic_v2.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(json.dumps(data.get('summary', {}), indent=2))
"

# 3. Read v3 outline
cat docs/NQS_J1J2_Prediction_v3_outline.md | head -50

# 4. If GPU 3 free ≥ 22 GB, ready for production v2 launch
TCBM_DEVICE=cuda:3 python -u experiments/run_gradient_baseline_v2.py \
  2>&1 | tee logs/production_v2_seed42_$(date +%Y%m%d_%H%M).log
```

---

## Part 6: Patch summary

**This patch contains**:
- Project context (Section 1) — physical system, algorithm, schedule
- Work done in this chat (Section 2) — Day 2 recap + full Day 3 detail
- Latest experiment plan (Section 3) — v3 outline summary, 4-tier criteria, Plan A/B/C
- Day 4 detailed plan (Section 4) — morning/afternoon/evening tasks
- Workflow notes (Section 5) — critical context, principles, file inventory, open decisions

**Day 3 final state**:
- 9 commits total (8 on GitHub, 1 local pending push)
- 23 tests passing
- v2 Adam diagnostic verdict: STRONG multi-basin signal (4×4 retained)
- v3 outline complete (250 lines)
- SR optimizer + baseline skeletons in place (Day 5 implement)
- Production v2 ready to launch (RNG-fixed, three-pack robustness)

**Day 4 critical path**:
1. Morning: daily_log + push commits + outline update
2. Afternoon: full v3 doc + launch production v2 TCBM-gradient
3. Evening: production v2 verdict + Day 5 path decision

**Schedule status**:
- Original Day 21 NMI cap
- Currently 1 day slip (production v2 not launched Day 3)
- Buffer: ~1 day remaining
- Plan B available (Day 22-24 cap if SR Day 7 fails, NetKet fork)
- Plan C available (Adam-only fallback if NetKet Day 10 fails)

---

**End of Context Patch v6.**

For Day 4 start: paste this entire markdown into next chat, plus tell me "Day 4 morning, ready to start with Step 1: review Day 3 state and write daily_log entry."
