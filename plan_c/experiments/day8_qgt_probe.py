"""Day 8 — diagnostic probe for the STEP-4 plum is_scalar(JitTracer) crash.

The on-the-fly QGT SR path crashes with plum AmbiguousLookupError under
jax 0.8.3. This probe checks whether the *dense / Jacobian* QGT path (which is
what the project wants for TCBM subspace extraction anyway) avoids it, for both
(a) the SR solve step and (b) QGT to_dense() extraction (STEP 5).

Read-only diagnostic (no env changes). Run:
  python -u plan_c/experiments/day8_qgt_probe.py
"""
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import sys
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import netket as nk
from plan_c.core.j1j2_system import build_j1j2_graph, build_j1j2_hilbert, build_j1j2_hamiltonian

print("nk", nk.__version__)
print("qgt module members:", [x for x in dir(nk.optimizer.qgt) if x.startswith("QGT")])

g = build_j1j2_graph(4)
hi = build_j1j2_hilbert(g)
ha = build_j1j2_hamiltonian(hi, g)
ma = nk.models.RBM(alpha=2, param_dtype=complex)
sa = nk.sampler.MetropolisExchange(hi, graph=g, d_max=2)
vs = nk.vqs.MCState(sa, ma, n_samples=1024, seed=42, sampler_seed=42)

# (A) SR step with on-the-fly (reproduce crash) — wrapped
print("\n[A] on-the-fly SR step:")
try:
    op = nk.optimizer.Sgd(learning_rate=0.01)
    sr = nk.optimizer.SR(diag_shift=0.01)
    drv = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)
    drv.advance()
    print("   OK (no crash)")
except Exception as e:
    print(f"   CRASH: {type(e).__name__}: {str(e)[:120]}")

# (B) SR step with QGTJacobianDense
print("\n[B] QGTJacobianDense SR step:")
try:
    vs_b = nk.vqs.MCState(sa, ma, n_samples=1024, seed=42, sampler_seed=42)
    op = nk.optimizer.Sgd(learning_rate=0.01)
    sr = nk.optimizer.SR(qgt=nk.optimizer.qgt.QGTJacobianDense, diag_shift=0.01)
    drv = nk.VMC(ha, op, variational_state=vs_b, preconditioner=sr)
    drv.advance()
    e = vs_b.expect(ha)
    print(f"   OK  E={float(np.real(np.asarray(e.mean))):+.5f}")
except Exception as e:
    print(f"   CRASH: {type(e).__name__}: {str(e)[:160]}")

# (C) SR step with QGTJacobianPyTree
print("\n[C] QGTJacobianPyTree SR step:")
try:
    vs_c = nk.vqs.MCState(sa, ma, n_samples=1024, seed=42, sampler_seed=42)
    op = nk.optimizer.Sgd(learning_rate=0.01)
    sr = nk.optimizer.SR(qgt=nk.optimizer.qgt.QGTJacobianPyTree, diag_shift=0.01)
    drv = nk.VMC(ha, op, variational_state=vs_c, preconditioner=sr)
    drv.advance()
    e = vs_c.expect(ha)
    print(f"   OK  E={float(np.real(np.asarray(e.mean))):+.5f}")
except Exception as e:
    print(f"   CRASH: {type(e).__name__}: {str(e)[:160]}")

# (D) QGT to_dense extraction (STEP 5) — on-the-fly vs Jacobian
print("\n[D] QGT to_dense():")
for name, qgt_cls in [("default", None),
                      ("QGTJacobianDense", nk.optimizer.qgt.QGTJacobianDense),
                      ("QGTJacobianPyTree", nk.optimizer.qgt.QGTJacobianPyTree)]:
    try:
        if qgt_cls is None:
            S = vs.quantum_geometric_tensor()
        else:
            S = vs.quantum_geometric_tensor(qgt_cls)
        D = np.asarray(S.to_dense())
        print(f"   {name}: OK  shape={D.shape} dtype={D.dtype} finite={np.all(np.isfinite(D))}")
    except Exception as e:
        print(f"   {name}: CRASH {type(e).__name__}: {str(e)[:120]}")

print("\nprobe done")
