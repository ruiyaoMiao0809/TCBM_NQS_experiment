"""Day 8 — Plan C experiment line smoke test (NetKet as physics oracle).

Scope (experiment line only; theory line runs in another window):
  STEP 1  build 4x4 J1-J2 graph / Hilbert / Heisenberg Hamiltonian
  STEP 2  ED cross-check  |E0_netket - (-8.4579)| < 1e-3   [HARD GATE]
  STEP 3  RBM(alpha=2, complex) + MetropolisExchange + MCState
          + parameter-layout documentation (interface deliverable)
  STEP 4  SR 200-step smoke + Mode-1 numerical-pathology check
  STEP 5  QGT reachability + ordering doc

All NetKet API verified against installed netket 3.21.0 (see
plan_c/logs/day8_introspect.log) and the official gs-j1j2 tutorial.
CPU only (4x4 is tiny; avoid jax/torch CUDA contention).

Run:  python -u plan_c/experiments/day8_smoke.py
"""
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"   # 4x4 tiny; keep jax off CUDA (Lesson: no GPU contention)

import sys
# make repo root importable so `plan_c.core...` resolves
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import json
import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import netket as nk

from plan_c.core.j1j2_system import (
    build_j1j2_graph, build_j1j2_hilbert, build_j1j2_hamiltonian,
)

import warnings
# Quiet known-benign upstream noise (jax 0.8.3 deprecations; NetKet defaults to
# holomorphic=False for complex RBM — accepted for the smoke). Real errors/NaN
# are still surfaced by the STEP-4 Mode-1 checks.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*[Hh]olomorphic.*")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
L = 4
N = L * L                       # 16 sites
J1, J2 = 1.0, 0.5
E0_TRUTH = -8.4579              # Phase A ED reference (4x4 PBC, J2/J1=0.5)
SEED = 42

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

summary = {"versions": {"netket": nk.__version__, "jax": jax.__version__,
                        "backend": jax.default_backend()}}
print(f"netket {nk.__version__} | jax {jax.__version__} | backend {jax.default_backend()}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — graph / Hilbert / Hamiltonian (via plan_c.core single source of truth)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70 + "\nSTEP 1 — graph / Hilbert / Hamiltonian\n" + "=" * 70)

# Edge color convention (0-indexed for nk.operator.Heisenberg J[color]):
#   color 0 -> NN -> J1 ;  color 1 -> NNN -> J2.
# Energy convention: build_j1j2_hamiltonian passes J/4 internally so NetKet
# (sigma.sigma) emits energies in the Phase A S.S scale (E0=-8.4579). See
# plan_c/core/j1j2_system.py. Callers pass physical J1=1.0, J2=0.5.
g = build_j1j2_graph(L=L)
hi = build_j1j2_hilbert(g)
ha = build_j1j2_hamiltonian(hi, g, J1=J1, J2=J2)

edges = list(g.edges(return_color=True))
n_nn = sum(1 for e in edges if e[2] == 0)
n_nnn = sum(1 for e in edges if e[2] == 1)
print(f"edges by color: NN(color0)={n_nn}  NNN(color1)={n_nnn}  total={len(edges)}")
print(f"graph n_nodes={g.n_nodes}  hilbert.size={hi.size}  hilbert dim(Sz=0)={hi.n_states}")

step1_ok = (n_nn == 32 and n_nnn == 32 and len(edges) == 64)
summary["step1"] = {"n_nn": n_nn, "n_nnn": n_nnn, "total_bonds": len(edges),
                    "n_nodes": int(g.n_nodes), "hilbert_dim": int(hi.n_states),
                    "go": bool(step1_ok)}
print(f"STEP 1 verdict: {'GO' if step1_ok else 'NO-GO (expected NN=32 NNN=32 total=64)'}")
if not step1_ok:
    print("STOP: bond counts wrong; graph construction error (NNN direction or PBC).")
    with open(os.path.join(RESULTS_DIR, "day8_smoke_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    raise SystemExit(1)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — ED cross-check (HARD GATE)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70 + "\nSTEP 2 — ED cross-check (lanczos_ed)\n" + "=" * 70)
evals = nk.exact.lanczos_ed(ha)            # k=1, compute_eigenvectors=False
E0_netket = float(np.asarray(evals).ravel()[0])
delta = abs(E0_netket - E0_TRUTH)
step2_ok = delta < 1e-3
print(f"E0_netket = {E0_netket:.6f}   E0_truth = {E0_TRUTH}   |delta| = {delta:.3e}")
print(f"E0/N = {E0_netket / N:.6f}  (Phase A: -0.5286)")
summary["step2"] = {"E0_netket": E0_netket, "E0_truth": E0_TRUTH,
                    "delta": delta, "E0_per_site": E0_netket / N, "go": bool(step2_ok)}
print(f"STEP 2 verdict: {'GO' if step2_ok else 'NO-GO'}")
if not step2_ok:
    ratio = E0_netket / E0_TRUTH
    print(f"STOP: ED mismatch. ratio E0_netket/E0_truth = {ratio:.4f}")
    print("      (ratio≈4 => Pauli vs spin-1/2 convention; ratio≈2 => J double count;")
    print("       else => coloring/bond/PBC error). Reporting and halting.")
    with open(os.path.join(RESULTS_DIR, "day8_smoke_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    raise SystemExit(1)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — RBM ansatz + sampler + MCState + parameter layout doc
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70 + "\nSTEP 3 — RBM + sampler + MCState + layout\n" + "=" * 70)
ma = nk.models.RBM(alpha=2, param_dtype=complex)        # complex RBM, alpha=2
sa = nk.sampler.MetropolisExchange(hi, graph=g, d_max=2)  # Exchange preserves Sz=0 (tutorial uses d_max=2)
vs = nk.vqs.MCState(sa, ma, n_samples=4096, seed=SEED, sampler_seed=SEED)

n_params = vs.n_parameters
flat, unravel = ravel_pytree(vs.parameters)
n_complex = int(flat.shape[0])
n_real = n_complex * 2
print(f"vs.n_parameters = {n_params}")
print(f"ravel flat: shape={flat.shape} dtype={flat.dtype}  -> {n_complex} complex = {n_real} real")
print(f"expected: 560 complex = 1120 real (Phase A D=1120)")

step3_ok = (n_complex == 560)

# Build per-leaf index intervals in ravel order (ravel_pytree uses tree_flatten order).
leaves_with_path = jax.tree_util.tree_flatten_with_path(vs.parameters)[0]
layout_rows = []
idx = 0
for path, leaf in leaves_with_path:
    keystr = "/".join(str(getattr(k, "key", k)) for k in path)
    size = int(np.prod(leaf.shape))
    layout_rows.append((keystr, tuple(leaf.shape), str(leaf.dtype), idx, idx + size))
    idx += size
print("\nparameter leaves (ravel order):")
for keystr, shape, dt, lo, hi_ in layout_rows:
    print(f"  {keystr:28s} shape={str(shape):12s} dtype={dt:12s} complex_idx[{lo}:{hi_}]")

summary["step3"] = {"n_parameters": int(n_params), "n_complex": n_complex,
                    "n_real": n_real, "go": bool(step3_ok),
                    "leaves": [{"key": k, "shape": list(s), "dtype": d,
                                "complex_idx": [lo, hi_]}
                               for (k, s, d, lo, hi_) in layout_rows]}
print(f"STEP 3 verdict: {'GO' if step3_ok else 'NO-GO (n_complex != 560; affects P_Im convention)'}")

# Write the interface doc for the theory line (T4b P_Im convention).
doc = []
doc.append("# NetKet RBM parameter layout (Plan C, Day 8)\n")
doc.append("> Status: **待 Nick / AI review**. Auto-generated by `day8_smoke.py`.\n")
doc.append("\n## Energy convention (F-level, ratified Nick+AI Day 8)\n\n")
doc.append("- NetKet `nk.operator.Heisenberg` = **sigma.sigma (Pauli)** convention.\n")
doc.append("- Phase A ED / truth `E0 = -8.4579` = **S.S (spin-1/2, S=sigma/2)**; "
           "`S.S = (1/4) sigma.sigma`.\n")
doc.append("- Whole experiment line unified to **S.S** scale: the single entry "
           "`plan_c.core.j1j2_system.build_j1j2_hamiltonian` passes **J/4** to NetKet "
           "so energies are native S.S. `/4` appears ONLY there; callers use physical "
           "J1=1.0, J2=0.5.\n")
doc.append("- `E0_target = -8.4579`; guarded by `plan_c/tests/test_energy_convention.py`.\n")
doc.append(f"\n- netket {nk.__version__}, jax {jax.__version__}, backend {jax.default_backend()}\n")
doc.append(f"- model: `nk.models.RBM(alpha=2, param_dtype=complex)` on 4x4 J1-J2, Sz=0\n")
doc.append(f"- `vs.n_parameters = {n_params}`; ravel flat = **{n_complex} complex = {n_real} real**\n")
doc.append("\n## pytree structure (ravel_pytree order)\n\n")
doc.append("`jax.flatten_util.ravel_pytree(vs.parameters)` flattens leaves in "
           "`tree_flatten` order, row-major within each leaf:\n\n")
doc.append("| leaf key | shape | dtype | complex index range |\n")
doc.append("|----------|-------|-------|---------------------|\n")
for keystr, shape, dt, lo, hi_ in layout_rows:
    doc.append(f"| `{keystr}` | {shape} | {dt} | `[{lo}:{hi_}]` |\n")
doc.append(f"\n## complex -> real mapping convention (interface for theory line)\n\n")
doc.append("Define the real parameter vector as:\n\n")
doc.append("```\n")
doc.append("flat_complex = ravel_pytree(vs.parameters)[0]   # shape (560,), complex128\n")
doc.append("theta_real   = concat([ Re(flat_complex), Im(flat_complex) ])  # shape (1120,)\n")
doc.append("```\n\n")
doc.append(f"Under this convention (Re block first, Im block second):\n\n")
doc.append(f"- **Re block** = `theta_real[0:{n_complex}]`\n")
doc.append(f"- **P_Im (imaginary block)** = `theta_real[{n_complex}:{n_real}]`  "
           f"= the **last {n_complex} dims**\n\n")
doc.append("This is the unambiguous P_Im definition for the theory-line T4b projection.\n")
doc.append("\n## QGT ordering (filled by STEP 5 below)\n\n")
doc.append("PLACEHOLDER — see STEP 5 output appended at end of run.\n")
DOC_PATH = os.path.join(DOCS_DIR, "netket_param_layout.md")
with open(DOC_PATH, "w") as f:
    f.write("".join(doc))
print(f"layout doc written: {DOC_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — SR 200-step smoke + Mode-1 check
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70 + "\nSTEP 4 — SR 200-step smoke + Mode-1 check\n" + "=" * 70)
# QGT representation = QGTJacobianDense (NOT the box's default on-the-fly).
# Reason (flagged for Nick): under jax 0.8.3 the on-the-fly QGT solve crashes
# with plum AmbiguousLookupError on is_scalar(JitTracer) (netket<->jax version
# drift); and on-the-fly *cannot* be densified for non-holomorphic complex
# params (STEP 5 needs to_dense). QGTJacobianDense works for both and yields the
# (1120,1120) real-split QGT the project needs. Mathematically the same QGT.
op = nk.optimizer.Sgd(learning_rate=0.01)
sr = nk.optimizer.SR(qgt=nk.optimizer.qgt.QGTJacobianDense, diag_shift=0.01)
driver = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)

energy_log = []           # (step, E_mean_real, E_error)
any_nan = False
impossible_energy = False  # E < E0_TRUTH (variational principle violation -> Mode-1)
min_E = float("inf")

N_ITER = 200
for it in range(N_ITER):
    driver.advance()
    if it % 10 == 0 or it == N_ITER - 1:
        st = vs.expect(ha)
        e_mean = float(np.real(np.asarray(st.mean)))
        e_err = float(np.asarray(st.error_of_mean))
        energy_log.append((it, e_mean, e_err))
        if not np.isfinite(e_mean) or not np.isfinite(e_err):
            any_nan = True
        # NaN in parameters too
        pflat, _ = ravel_pytree(vs.parameters)
        if not bool(jnp.all(jnp.isfinite(pflat))):
            any_nan = True
        min_E = min(min_E, e_mean)
        if e_mean < E0_TRUTH - 1e-2:    # below ED ground = impossible (Mode-1 signature)
            impossible_energy = True
        print(f"  it={it:4d}  E={e_mean:+.5f} ± {e_err:.5f}"
              f"{'  <-- NaN!' if any_nan else ''}"
              f"{'  <-- IMPOSSIBLE(<E0)!' if e_mean < E0_TRUTH - 1e-2 else ''}")

final_E = energy_log[-1][1]
# Monotone-ish check: 10-step moving average should trend down (first vs last)
first_E = energy_log[0][1]
descended = final_E < first_E - 0.5
final_below_m5 = final_E < -5.0
mode1_clean = (not any_nan) and (not impossible_energy)

step4_ok = descended and final_below_m5 and mode1_clean
summary["step4"] = {
    "n_iter": N_ITER, "first_E": first_E, "final_E": final_E, "min_E": min_E,
    "descended": bool(descended), "final_below_-5": bool(final_below_m5),
    "any_nan": bool(any_nan), "impossible_energy_below_E0": bool(impossible_energy),
    "mode1_clean": bool(mode1_clean), "go": bool(step4_ok),
    "energy_log": [[int(s), e, er] for (s, e, er) in energy_log],
}
print(f"\nfirst_E={first_E:+.5f}  final_E={final_E:+.5f}  min_E={min_E:+.5f}")
print(f"descended={descended}  final<-5={final_below_m5}  "
      f"MODE-1 clean (0 NaN, no E<E0)={mode1_clean}")
print(f"STEP 4 verdict: {'GO' if step4_ok else 'NO-GO'}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — QGT reachability + ordering
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70 + "\nSTEP 5 — QGT reachability + ordering\n" + "=" * 70)
# QGTJacobianDense: required because on-the-fly QGT cannot densify non-holomorphic
# complex params (see STEP 4 note). Yields real-split (1120,1120) dense QGT.
qgt = vs.quantum_geometric_tensor(nk.optimizer.qgt.QGTJacobianDense)
S = np.asarray(qgt.to_dense())
print(f"QGT dense shape={S.shape} dtype={S.dtype}")
finite = bool(np.all(np.isfinite(S)))
# symmetry / hermiticity
herm_err = float(np.max(np.abs(S - S.conj().T))) if S.size else float("nan")
# singular values (SVD), top 20
svals = np.linalg.svd(S, compute_uv=False)
top20 = svals[:20]
min_eig = None
try:
    # near-PSD: smallest eigenvalue of Hermitian part
    Hpart = 0.5 * (S + S.conj().T)
    eigs = np.linalg.eigvalsh(Hpart)
    min_eig = float(eigs[0])
except Exception as e:
    print(f"  eigvalsh failed: {e}")

near_psd = (min_eig is not None) and (min_eig > -1e-6)
step5_ok = finite and (herm_err < 1e-5) and near_psd and len(top20) >= 20
print(f"finite={finite}  hermiticity_err={herm_err:.3e}  min_eig(Hermitian part)={min_eig}")
print(f"top-20 singular values:\n{np.array2string(top20, precision=4)}")
summary["step5"] = {"qgt_shape": list(S.shape), "qgt_dtype": str(S.dtype),
                    "finite": finite, "hermiticity_err": herm_err,
                    "min_eig_hermitian": min_eig, "near_psd": bool(near_psd),
                    "top20_svals": [float(v) for v in top20], "go": bool(step5_ok)}
print(f"STEP 5 verdict: {'GO' if step5_ok else 'NO-GO'}")

# QGT ordering note: dense QGT row/col i corresponds to ravel_pytree leaf order
# (NetKet builds the jacobian in the same pytree flatten order as ravel_pytree).
qgt_dim = S.shape[0]
qgt_matches_ravel = (qgt_dim == n_complex) or (qgt_dim == n_real)
ordering_note = (
    f"\n## QGT ordering (STEP 5 result) — RESOLVED (Day 8 closeout)\n\n"
    f"- QGT dense shape = `{S.shape}`, dtype = `{S.dtype}` (via QGTJacobianDense).\n"
    f"- ravel complex dim = {n_complex} (real = {n_real}); QGT dim {qgt_dim} "
    f"{'== n_real -> real-split QGT' if qgt_dim == n_real else ('== n_complex -> complex QGT' if qgt_dim == n_complex else '!! UNEXPECTED')}.\n"
    f"- Real-axis layout = **BLOCK `[Re_all(0:{n_complex}); Im_all({n_complex}:{n_real})]`, "
    f"Re first** — confirmed by `day8_qgt_axis_probe.py` (value at complex idx k -> "
    f"real idx k for Re, k+{n_complex} for Im). Realification = nk.jax.tree_to_real "
    f"then ravel_pytree (the QGTJacobianDense param-axis path); tree_to_real is ordered "
    f"(Re before Im), not a plain dict.\n"
    f"- Identical to `theta_real=[Re(flat),Im(flat)]` => **P_Im = theta_real[{n_complex}:{n_real}]**, "
    f"QGT axis aligns 1:1 with theta_real, no conversion for Day13 projection.\n"
)
# Replace placeholder in doc
with open(DOC_PATH) as f:
    doc_txt = f.read()
doc_txt = doc_txt.replace(
    "## QGT ordering (filled by STEP 5 below)\n\nPLACEHOLDER — see STEP 5 output appended at end of run.\n",
    ordering_note.lstrip("\n"))
with open(DOC_PATH, "w") as f:
    f.write(doc_txt)
print(f"layout doc QGT section updated: {DOC_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
summary["all_go"] = all(summary[s]["go"] for s in
                        ["step1", "step2", "step3", "step4", "step5"])
SUMMARY_PATH = os.path.join(RESULTS_DIR, "day8_smoke_summary.json")
with open(SUMMARY_PATH, "w") as f:
    json.dump(summary, f, indent=2)
print("\n" + "=" * 70)
print(f"ALL-GO = {summary['all_go']}")
for s in ["step1", "step2", "step3", "step4", "step5"]:
    print(f"  {s}: {'GO' if summary[s]['go'] else 'NO-GO'}")
print(f"summary written: {SUMMARY_PATH}")
print("=" * 70)
