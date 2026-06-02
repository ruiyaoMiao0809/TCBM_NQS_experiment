"""Day 8 closeout — QGT real-axis layout probe (fix P_Im <-> QGT-axis mapping).

QGTJacobianDense realifies the *parameter* axis via the EXACT path
  nk.jax.tree_to_real(pars)  ->  ravel_pytree(...)
(see netket/jax/_jacobian/logic.py:256 and jacobian_dense.py:31). So the 1120
real QGT axis order == ravel_pytree order of tree_to_real(pars).

We do NOT trust the docstring pseudocode (a plain {'real':,'imag':} dict would
ravel imag-first under jax's alphabetical key sort). Instead we drive a known
Re-only / Im-only value through that real function and read the landing index.

Run:  python -u plan_c/experiments/day8_qgt_axis_probe.py
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
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import netket as nk
from plan_c.core.j1j2_system import build_j1j2_graph, build_j1j2_hilbert, build_j1j2_hamiltonian

print("nk", nk.__version__, "| jax", jax.__version__)
print("nk.jax.tree_to_real present:", hasattr(nk.jax, "tree_to_real"))

g = build_j1j2_graph(4)
hi = build_j1j2_hilbert(g)
ma = nk.models.RBM(alpha=2, param_dtype=complex)
sa = nk.sampler.MetropolisExchange(hi, graph=g, d_max=2)
vs = nk.vqs.MCState(sa, ma, n_samples=256, seed=42, sampler_seed=42)

pars = vs.parameters
flat_c, unravel_c = ravel_pytree(pars)          # (560,) complex, the STEP-3 ravel order
n_c = flat_c.shape[0]
print(f"n_complex = {n_c}")

RE_VAL, IM_VAL = 100.0, 7.0   # distinctive, distinguishable magnitudes

def probe_index(k):
    """Put RE_VAL+IM_VAL*1j at complex ravel index k, push through NetKet's
    tree_to_real + ravel, return (re_idx, im_idx) in the 1120 real vector."""
    z = jnp.zeros((n_c,), dtype=flat_c.dtype).at[k].set(RE_VAL + 1j * IM_VAL)
    pars_k = unravel_c(z)
    pars_real, _reconstruct = nk.jax.tree_to_real(pars_k)
    flat_r, _ = ravel_pytree(pars_real)
    flat_r = np.asarray(flat_r)
    re_hits = np.where(np.isclose(flat_r, RE_VAL))[0]
    im_hits = np.where(np.isclose(flat_r, IM_VAL))[0]
    n_real = flat_r.shape[0]
    return n_real, (re_hits.tolist(), im_hits.tolist())

# Probe one index per leaf (Dense/bias=0, Dense/kernel=32, visible_bias=544)
# plus a couple extra to confirm the rule holds across the whole vector.
test_ks = [0, 1, 32, 200, 543, 544, 559]
print("\n  k (complex idx) ->  Re lands at | Im lands at   (real vector dim shown once)")
results = {}
n_real = None
for k in test_ks:
    n_real, (re_idx, im_idx) = probe_index(k)
    results[k] = (re_idx, im_idx)
    print(f"  k={k:4d}  ->  Re@{re_idx}  | Im@{im_idx}")

# Decide layout: block [Re_all(0:560); Im_all(560:1120)] predicts Re@k, Im@k+560.
# imag-first block predicts Im@k, Re@k+560. interleaved predicts {2k,2k+1}.
def classify(k, re_idx, im_idx):
    if re_idx == [k] and im_idx == [k + n_c]:
        return "BLOCK [Re;Im] (Re first)"
    if im_idx == [k] and re_idx == [k + n_c]:
        return "BLOCK [Im;Re] (Im first)"
    if set(re_idx + im_idx) == {2 * k, 2 * k + 1}:
        return f"INTERLEAVED (Re@{re_idx}, Im@{im_idx})"
    return "UNKNOWN/other"

verdicts = {classify(k, *results[k]) for k in test_ks}
print(f"\nreal vector dim = {n_real}")
print("per-k classification:", verdicts)
assert len(verdicts) == 1, f"INCONSISTENT layout across leaves: {verdicts}"
layout = verdicts.pop()
print(f"LAYOUT VERDICT: {layout}")

# Emit the definitive P_Im index set.
if layout.startswith("BLOCK [Re;Im]"):
    p_im = f"theta_real[{n_c}:{2*n_c}]  (last {n_c} dims; contiguous)"
    re_block = f"theta_real[0:{n_c}]"
elif layout.startswith("BLOCK [Im;Re]"):
    p_im = f"theta_real[0:{n_c}]  (first {n_c} dims; contiguous)"
    re_block = f"theta_real[{n_c}:{2*n_c}]"
elif layout.startswith("INTERLEAVED"):
    p_im = "odd indices theta_real[1::2]"
    re_block = "even indices theta_real[0::2]"
else:
    p_im = re_block = "UNRESOLVED"
print(f"\nP_Im (imaginary-part axes) = {p_im}")
print(f"Re axes                    = {re_block}")
print("This ordering == ravel_pytree(tree_to_real(pars)) == QGTJacobianDense param axis.")
print("\nprobe done")
