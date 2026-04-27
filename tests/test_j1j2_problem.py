"""tests/test_j1j2_problem.py — Day 1.B pytest suite for core/j1j2_problem.py.

Tests 1-7 are pytest conversions of the inline tests at the bottom of
core/j1j2_problem.py (P0-1.1 success criteria, Protocol v2.1 §1.5).

Tests 8-10 are new (Day 1.B):
  test_8:  ED matrix cross-check on N=4 (2×2 PBC) lattice.
  test_9:  _log_2cosh_complex stable-form parametric over 7 (u,v) edge cases.
  test_10: _log_2cosh_complex realistic batch sanity at typical RBM scales.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.j1j2_problem import (                                # noqa: E402
    J1J2Problem,
    _log_2cosh_complex,
    build_square_lattice_bonds,
    log_psi_rbm,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def problem_4x4():
    """Standard POC setup: 4×4 PBC, J1=1, J2=0.5, α=2 (D=1120). CPU."""
    return J1J2Problem(Lx=4, Ly=4, J1=1.0, J2=0.5, alpha=2, device='cpu')


# ─────────────────────────────────────────────────────────────────────────────
# Tests 1-7: pytest conversions of inline skeleton tests
# ─────────────────────────────────────────────────────────────────────────────

def test_dimensions(problem_4x4):
    """D = 2(N + M + NM) = 1120 for 4×4, α=2."""
    p = problem_4x4
    assert p.N == 16
    assert p.M == 32
    expected_D = 2 * (16 + 32 + 16 * 32)
    assert expected_D == 1120
    assert p.dim == expected_D


def test_bonds(problem_4x4):
    """4×4 PBC: 2L² = 32 NN bonds, 2L² = 32 NNN diagonals."""
    p = problem_4x4
    assert p._nn_bonds.shape == (32, 2)
    assert p._nnn_bonds.shape == (32, 2)


def test_log_psi_stability(problem_4x4):
    """log_psi finite for random θ on a random sigma batch."""
    p = problem_4x4
    theta = p.random_feasible(1)[0]
    sigma = (torch.randint(0, 2, (8, p.N), dtype=torch.long) * 2 - 1).to(torch.float32)
    log_psi = log_psi_rbm(theta, sigma, p.N, p.M)
    assert torch.isfinite(log_psi.real).all()
    assert torch.isfinite(log_psi.imag).all()


def test_evaluate_stochastic(problem_4x4):
    """evaluate returns (cost, penalty=0, std>0) for stochastic VMC mode."""
    p = problem_4x4
    theta_batch = p.random_feasible(3)
    cost, pen, std = p.evaluate(theta_batch, n_samples=500)
    assert cost.shape == (3,)
    assert pen.shape == (3,)
    assert pen.abs().max().item() == 0
    assert std.shape == (3,)
    assert (std > 0).all()


def test_sigma_std_sqrt_scaling(problem_4x4):
    """σ_E ∝ 1/√N_samples. 4× samples should give std ratio ~2 (±50%)."""
    p = problem_4x4
    theta = p.random_feasible(1)
    _, _, std_small = p.evaluate(theta, n_samples=200)
    _, _, std_large = p.evaluate(theta, n_samples=800)
    ratio = (std_small / std_large).item()
    assert 1.3 < ratio < 3.0, f"σ_E sqrt-scaling violated: ratio={ratio:.2f} (expected ~2)"


def test_energy_physical_range(problem_4x4):
    """E ∈ [-2N, 2N]: each Heisenberg bond contributes ≤ |J|/4·(σσ) + |J|/2 in magnitude."""
    p = problem_4x4
    theta_batch = p.random_feasible(5)
    cost, _, _ = p.evaluate(theta_batch, n_samples=500)
    assert (cost > -2 * p.N).all()
    assert (cost < 2 * p.N).all()


def test_gradient_shape(problem_4x4):
    """gradient returns (B, D), all finite, and ||grad||_inf in sane range."""
    p = problem_4x4
    theta_batch = p.random_feasible(2)
    grad = p.gradient(theta_batch)
    assert grad.shape == (2, p.dim)
    assert torch.isfinite(grad).all()
    # Sanity bound: for random init (init_scale=0.01), gradient values should
    # be O(1) or smaller. A loose <1e6 bound catches autograd graph leaks or
    # unstable backward passes without false-positives on legitimate gradients.
    grad_inf = grad.abs().max().item()
    assert grad_inf < 1e6, f"||grad||_inf = {grad_inf} suggests numerical instability"


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: ED matrix cross-check (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def _build_full_hamiltonian_matrix(Lx, Ly, J1, J2):
    """Full (2^N, 2^N) sparse H for cross-check.

    Uses the same bond list as J1J2Problem (build_square_lattice_bonds), so
    any PBC degeneracy (e.g. Lx=2 or Ly=2 wrapping) appears identically on
    both sides. The cross-check thus validates internal consistency of the
    local-energy code path, NOT a literature benchmark.
    """
    N = Lx * Ly
    nn_bonds, nnn_bonds = build_square_lattice_bonds(Lx, Ly)
    bond_specs = [(nn_bonds, J1), (nnn_bonds, J2)]

    dim = 2 ** N
    rows, cols, data = [], [], []

    for state in range(dim):
        diag = 0.0
        for bonds, J in bond_specs:
            for (i, j) in bonds:
                si = 2 * ((state >> i) & 1) - 1                  # ±1
                sj = 2 * ((state >> j) & 1) - 1
                diag += J * 0.25 * si * sj                       # diagonal Sz·Sz
                if si * sj < 0:                                  # off-diagonal
                    flipped = state ^ ((1 << i) | (1 << j))
                    rows.append(state)
                    cols.append(flipped)
                    data.append(J * 0.5)
        rows.append(state)
        cols.append(state)
        data.append(diag)

    H = sp.coo_matrix((data, (rows, cols)), shape=(dim, dim)).tocsr()
    return 0.5 * (H + H.T)                                       # enforce symmetry


def test_evaluate_exact_matches_ed_small():
    """Cross-check J1J2Problem.evaluate (exact mode) vs explicit H matrix on N=4.

    Builds H as a (2^N, 2^N) sparse matrix and computes <ψ|H|ψ>/<ψ|ψ> directly
    via matrix multiplication, then compares to evaluate(theta, n_samples=0)
    which uses the local-energy code path.

    Lattice 2×2 PBC: bonds are double-counted due to short-direction wraparound
    (4 unique NN pairs ×2, 2 unique NNN pairs ×4). This is consistent on both
    sides of the cross-check — the test validates implementation consistency,
    not literature ground-state energy.
    """
    Lx, Ly = 2, 2
    N = Lx * Ly
    J1, J2 = 1.0, 0.5

    prob = J1J2Problem(Lx=Lx, Ly=Ly, J1=J1, J2=J2, alpha=2, device='cpu')
    torch.manual_seed(42)
    theta = prob.random_feasible(1)                              # (1, D)

    # Path A: J1J2Problem.evaluate exact mode (uses _local_energy_batch)
    cost, _, _ = prob.evaluate(theta, n_samples=0)
    E_via_local = float(cost[0].item())

    # Path B: full H matrix · ψ vector
    H = _build_full_hamiltonian_matrix(Lx, Ly, J1, J2)

    all_sigma = torch.zeros(2 ** N, N, dtype=torch.float32)
    for state in range(2 ** N):
        for k in range(N):
            all_sigma[state, k] = 2 * ((state >> k) & 1) - 1
    log_psi = log_psi_rbm(theta[0], all_sigma, N, prob.M)        # (2^N,) complex
    psi = torch.exp(log_psi).detach().numpy().astype(np.complex128)

    Hpsi = H @ psi
    numer = np.vdot(psi, Hpsi).real
    denom = np.vdot(psi, psi).real
    E_via_matrix = float(numer / denom)

    err = abs(E_via_local - E_via_matrix)
    print(f"\n  E_via_local  = {E_via_local:.10f}")
    print(f"  E_via_matrix = {E_via_matrix:.10f}")
    print(f"  abs error    = {err:.3e}")
    assert err < 1e-3, (
        f"ED cross-check failed: E_local={E_via_local}, E_matrix={E_via_matrix}, err={err}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: _log_2cosh_complex parametric over 7 (u,v) edge cases (NEW)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("u,v", [
    (-100.0, 0.0),
    (-30.0,  math.pi / 2),
    (0.0,    0.0),
    (0.0,    math.pi),
    (30.0,   -math.pi / 2),
    (50.0,   50 * math.pi),
    (100.0,  math.pi),
])
def test_log_2cosh_stability(u, v):
    """Each of the 7 edge cases must produce finite (Re, Im) output.

    Zero-point (0, 0) additionally checks the analytic value log(2·cosh(0)) = log 2.
    """
    z = torch.complex(torch.tensor(u, dtype=torch.float32),
                      torch.tensor(v, dtype=torch.float32))
    res = _log_2cosh_complex(z)
    assert torch.isfinite(res.real), f"Re part not finite at ({u}, {v})"
    assert torch.isfinite(res.imag), f"Im part not finite at ({u}, {v})"

    if u == 0.0 and v == 0.0:
        err_re = abs(res.real.item() - math.log(2.0))
        assert err_re < 1e-5, (
            f"Zero-point: Re result={res.real.item()}, expected log(2)={math.log(2.0)}, err={err_re:.3e}"
        )
        assert abs(res.imag.item()) < 1e-5, f"Zero-point Im should be 0, got {res.imag.item()}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: realistic batch sanity (NEW, from Supplement 2)
# ─────────────────────────────────────────────────────────────────────────────

def test_log_2cosh_realistic_batch():
    """Verify _log_2cosh_complex on a batch of typical RBM hidden activations."""
    torch.manual_seed(42)
    # Simulate hidden_act for M=32, batch=100, with magnitudes typical of
    # post-warmup RBM training (|θ| ~ 1-5)
    u_batch = torch.randn(100, 32) * 3.0           # typical mid-training scale
    v_batch = torch.randn(100, 32) * 0.5           # imag stays small
    z = torch.complex(u_batch, v_batch)

    result = _log_2cosh_complex(z)

    # All finite
    assert torch.isfinite(result.real).all()
    assert torch.isfinite(result.imag).all()

    # Magnitude check: log|2·cosh(u+iv)| ≈ |u| + O(1).
    # Mathematical justification:
    #   log|2·cosh(u+iv)| = |u| + (1/2)·log[(1 + e^(-2|u|))² - 4·sin²(v)·e^(-2|u|)]
    # The bracketed term is bounded by [(1)² - 0, (1+1)²] = [1, 4] for any u, v,
    # so its log is bounded by [0, 2·log(2)] ≈ [0, 1.39].
    # Thus log|2·cosh| ≤ |u| + log(2) < |u| + 1.0 (with margin).
    # log|2·cosh| ≥ log(2) > 0 always, so we test against result.real (not abs).
    assert (result.real <= u_batch.abs() + 1.0).all(), (
        f"Magnitude bound violated: max overflow = "
        f"{(result.real - u_batch.abs()).max().item():.4f}"
    )

    # Sum over hidden dim is the actual NQS h_contrib
    h_contrib = result.sum(dim=-1)
    assert torch.isfinite(h_contrib.real).all()
    assert torch.isfinite(h_contrib.imag).all()
