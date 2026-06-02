"""Regression test for the Plan C energy convention (Day 8, F-level decision).

If anyone changes the sigma.sigma -> S.S (/4) convention in
build_j1j2_hamiltonian, this test fails immediately (Phase-A Lesson 10 spirit:
make silent assumption drift loud).

Run:  conda activate tcbm_nqs && pytest plan_c/tests/test_energy_convention.py -v
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")   # 4x4 tiny; keep jax off CUDA

# Make the repo root importable so `plan_c.core...` resolves when pytest is
# invoked from anywhere.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import netket as nk
from plan_c.core.j1j2_system import (
    build_j1j2_graph,
    build_j1j2_hilbert,
    build_j1j2_hamiltonian,
    E0_TARGET,
)


def test_ground_state_energy_matches_phase_a_ed():
    """E0 from NetKet (S.S convention) must match Phase A ED truth -8.4579."""
    g = build_j1j2_graph(L=4)
    hi = build_j1j2_hilbert(g)
    ha = build_j1j2_hamiltonian(hi, g, J1=1.0, J2=0.5)
    E0 = float(np.asarray(nk.exact.lanczos_ed(ha)).ravel()[0])
    assert abs(E0 - E0_TARGET) < 1e-3, (
        f"E0={E0:.6f} != {E0_TARGET} (delta={abs(E0 - E0_TARGET):.3e}). "
        f"Energy convention drift? NetKet Heisenberg is sigma.sigma; "
        f"build_j1j2_hamiltonian must pass J/4 for the S.S scale."
    )


def test_bond_counts():
    """4x4 PBC: 32 NN (color 0) + 32 NNN (color 1) = 64 bonds."""
    g = build_j1j2_graph(L=4)
    edges = list(g.edges(return_color=True))
    n_nn = sum(1 for e in edges if e[2] == 0)
    n_nnn = sum(1 for e in edges if e[2] == 1)
    assert (n_nn, n_nnn) == (32, 32), f"got NN={n_nn} NNN={n_nnn} (expected 32/32)"


def test_netket_is_sigma_sigma_convention():
    """Sanity: 2-site Heisenberg singlet = -3.0 in raw NetKet (sigma.sigma),
    confirming the factor-4 rationale for the /4 in build_j1j2_hamiltonian."""
    g2 = nk.graph.Graph(edges=[(0, 1)])
    hi2 = nk.hilbert.Spin(s=0.5, N=2, total_sz=0.0)
    h2 = nk.operator.Heisenberg(hi2, g2, J=1.0)   # raw, no /4
    e2 = float(np.asarray(nk.exact.lanczos_ed(h2)).ravel()[0])
    assert abs(e2 - (-3.0)) < 1e-6, f"2-site E0={e2} (expected -3.0 for sigma.sigma)"
