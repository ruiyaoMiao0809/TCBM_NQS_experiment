# Plan C · Day 8 — Experiment line: NetKet bring-up + 4×4 J1-J2 + ED/SR/QGT smoke

> **Status: FINAL** — finalized after Nick+AI review (3 corrections applied).
> Experiment line only; theory line (Stage III-A) ran in parallel in another
> window and is not covered here.

## Goal (today, experiment line only)

Install NetKet, stand up the 4×4 J1-J2 system, cross-check ED against Phase A
truth (E0 = −8.4579), confirm SR runs and is free of Phase A's Mode-1 numerical
pathology, and confirm the QGT is reachable as a dense matrix — plus deliver the
RBM parameter-layout / P_Im interface doc for the theory line.

## Result: 5/5 STEP GO (ALL-GO = True)

| STEP | verdict | key numbers |
|------|---------|-------------|
| 0 install | GO | netket 3.21.0, jax→0.8.3 (see decisions), torch/numpy/scipy untouched |
| 1 graph/H | GO | NN(color0)=32, NNN(color1)=32, total=64; Hilbert dim(Sz=0)=12870 |
| 2 ED | GO | E0_netket = **−8.457923**, |Δ|=2.3e−5 < 1e−3 (S·S convention) |
| 3 ansatz/layout | GO | n_parameters = **560 complex = 1120 real** (= Phase A D=1120) |
| 4 SR smoke + Mode-1 | GO | +11.995 → −7.127 in 200 steps (rel_err ≈ 15.7%, **smoke only, NOT converged**), monotone; **0 NaN, no E<E0** |
| 5 QGT | GO | dense (1120,1120) f64, finite, herm_err=0, min_eig≈−4.4e−16 (PSD) |

### 🎯 Headline (validates Plan C 方案2 premise)
The smoke's value is **NOT the energy number** (−7.127 at 200 untuned steps is
rel_err ≈ 15.7% — a deliberately short, un-tuned run, far from converged). Its
value is two qualitative facts:
- **(a) No Mode-1 pathology**: zero NaN across all 200 steps, and the energy never
  dipped below E0 (no impossible "−1e18"-type readings). NetKet's logsumexp-based
  estimator eliminates Phase A's 1/ψ-underflow blowup at the source.
- **(b) Landscape is learnable**: SR drove the energy down smoothly and monotonically
  (+11.995 → −7.127), i.e. the 4×4 J1-J2 frustrated landscape is optimizable here.

Premise of Plan C 方案2 (NetKet as physics oracle removes the numerical failure mode)
is confirmed.

### Implication for Phase 0
With Mode-1 removed and the landscape demonstrated learnable by a standard
optimizer (SR), the remaining open question for TCBM **narrows to a pure
T4a / subspace question** (does the gradient buffer develop low-rank structure;
is the clamping subspace useful), no longer entangled with "can anything even
optimize this system." Concretely, this re-attributes the Phase A failure to
**(Mode-1 numerical blowup) + (a TCBM-specific Mode-2 stall)** — *not* to the
landscape being unsolvable. Consequently the upcoming **Phase 0 ψ-probe verdict is
no longer confounded** by the "is the landscape learnable" confound: a RED/GREEN ψ
reading can be read as a clean statement about TCBM's subspace mechanism itself.

## Decisions taken today

### D1 (F-level, ratified Nick+AI) — jax downgrade to fix netket↔jax drift
- `pip install netket` pulled netket 3.21.0 (2025-12-15) **+ jax 0.10.1** (netket
  only pins `jax>=0.7.0`, no ceiling).
- netket 3.21.0 uses `jax.lax.pvary` / `jax.typeof(x).vma`, **removed in jax 0.10.x**
  → `lanczos_ed` crashed (`AttributeError: ... 'vma'`).
- Decision: pin **jax==jaxlib==0.8.3** (netket-3.21.0 era; has pvary/vma) in the
  shared `tcbm_nqs` env. Verified: pvary/vma present; torch 2.5.1 / numpy 2.4.3 /
  scipy 1.17.1 unchanged. Transitive: **flax 0.12.7 → 0.12.6** (0.12.7 requires
  jax≥0.10). Re-resolved cleanly. Frozen to `plan_c/environment_planc.txt`.

### D2 (F-level, ratified Nick+AI) — energy convention σ·σ → S·S
- NetKet `nk.operator.Heisenberg` is **σ·σ (Pauli)**; Phase A truth is **S·S**
  (S=σ/2), so E_netket = 4·E_phaseA (2-site singlet = −3.0 confirms; ratio exactly 4).
- Decision: unify the whole experiment line to **S·S** (native), so Phase A config
  + methodology thresholds (T1 rel_err, gradient/step scale) carry over unchanged.
  Single entry `plan_c.core.j1j2_system.build_j1j2_hamiltonian` passes **J/4**; the
  `/4` lives ONLY there; callers use physical J1=1.0, J2=0.5. Guarded by
  `plan_c/tests/test_energy_convention.py` (3/3 pass).
- Rejected option "÷4 only at display": would run the optimizer on 4× gradients,
  inconsistent with reported scale.

### D3 (engineering, reviewed & accepted) — QGTJacobianDense instead of on-the-fly
- Under jax 0.8.3 the **on-the-fly** QGT SR solve crashes with
  `plum.AmbiguousLookupError: is_scalar(JitTracer)` (a 2nd netket↔jax dispatch drift),
  and on-the-fly **cannot densify** non-holomorphic complex params (STEP 5 needs
  `to_dense`). `QGTJacobianDense` works for both and gives the (1120,1120) real-split
  dense QGT the project needs anyway. Mathematically the same QGT. Probe:
  `plan_c/experiments/day8_qgt_probe.py` (A=on-the-fly CRASH, B=Dense OK, C=PyTree
  CRASH, D=Dense/PyTree to_dense OK).

## Deliverable: parameter layout / P_Im interface (for theory line)
`plan_c/docs/netket_param_layout.md`. Ravel (complex, 560): `Dense/bias`[0:32]
(hidden bias), `Dense/kernel`[32:544], `visible_bias`[544:560]. Convention
`θ_real = [Re(flat), Im(flat)]` (1120) ⇒ **P_Im = θ_real[560:1120]** (last 560).

## Resolved & remaining items

- ✅ **QGT real-axis layout — RESOLVED (commit d5622eb)**: **BLOCK `[Re;Im]`** (Re
  first); **P_Im = θ_real[560:1120]**, **zero conversion** (QGT axis i aligns 1:1
  with θ_real[i]). Verified empirically across boundary indices on all 3 leaves
  (`day8_qgt_axis_probe.py`); realification = `nk.jax.tree_to_real` → `ravel_pytree`,
  the QGTJacobianDense param-axis path (tree_to_real is an ordered Re-before-Im
  container, not an imag-first dict — checked, not assumed).
- **holomorphic flag**: **TCBM / QGT-source line locked to `holomorphic=False`**
  (we want the real 1120×1120 QGT for Re/Im projection). The **SR-baseline** setting
  is still open — pending Day 9 lookup of NetKet's recommended value for a complex
  non-holomorphic RBM (E-level).
- Day 9–10: wrappers touching NetKet APIs → must add real-object integration tests
  (Lesson 6). No wrappers written today. Complex↔real in wrappers MUST reuse the same
  `tree_to_real`/`ravel_pytree` path so θ_real aligns with the QGT axes.

## Files added (plan_c/)
- `core/j1j2_system.py` (+`__init__.py`), `__init__.py`
- `experiments/day8_smoke.py`, `day8_introspect_api.py`, `day8_qgt_probe.py`, `day8_qgt_axis_probe.py`
- `tests/test_energy_convention.py`
- `docs/netket_param_layout.md`, `docs/daily_log_day8.md` (this file)
- `results/day8_smoke_summary.json`, `logs/day8_*.log`, `environment_planc.txt`

## Lessons
- **Lesson 3 reinforced** (verify API against installed package, not snippets):
  introspection caught Heisenberg J-by-color 0-indexing, RBM `param_dtype`, on-the-fly
  QGT limits; the axis probe caught BLOCK [Re;Im] (a docstring-pseudocode dict would
  have raveled imag-first).
- **Lesson 11 (NEW) — pin the ecosystem to the library's release era.** A library
  with unbounded transitive pins (`netket` → `jax>=0.7.0`, no ceiling) lets `pip`
  pull a far-future jax/flax that silently breaks it (two distinct incompats hit:
  `pvary/vma` removal, `is_scalar(JitTracer)` dispatch). Resolve by pinning the whole
  ecosystem to versions current at the library's release date, then `pip freeze` to
  lock it. **How to apply:** for any community tool, after install run a real
  end-to-end smoke (not just `import`); on breakage, pin to the release-era versions
  and freeze, don't chase latest.
