"""tests/test_rng_seeding.py — Day 3 RNG fix.

Discovered Day 3: J1J2Problem._gen was hardcoded to manual_seed(0); module-level
torch.manual_seed/np.random.seed had no effect on internal sampling. Two
different script-level seeds produced byte-identical training trajectories.

Fix: J1J2Problem.set_seed(seed) method (core/j1j2_problem.py).

Verifies:
  test_set_seed_changes_init       — different seeds → different random_feasible
  test_set_seed_reproducible       — same seed twice → identical random_feasible
  test_evaluate_seeded_reproducible — same seed + same theta → identical evaluate
"""
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.j1j2_problem import J1J2Problem


def test_set_seed_changes_init():
    """Different seeds should give meaningfully different random_feasible output."""
    problem = J1J2Problem(Lx=2, Ly=2, alpha=2, J2=0.5, device='cpu')

    problem.set_seed(42)
    theta_42 = problem.random_feasible(1)

    problem.set_seed(7)
    theta_7 = problem.random_feasible(1)

    diff = (theta_42 - theta_7).abs().max().item()
    assert diff > 0.001, (
        f"theta_42 and theta_7 too close: max diff {diff} "
        f"(RNG fix may not be working — check J1J2Problem.set_seed)"
    )


def test_set_seed_reproducible():
    """Same seed twice should give identical random_feasible output."""
    problem = J1J2Problem(Lx=2, Ly=2, alpha=2, J2=0.5, device='cpu')

    problem.set_seed(42)
    theta_a = problem.random_feasible(1)

    problem.set_seed(42)
    theta_b = problem.random_feasible(1)

    assert torch.allclose(theta_a, theta_b), (
        f"Same seed gave different theta: max diff "
        f"{(theta_a - theta_b).abs().max().item()}"
    )


def test_evaluate_seeded_reproducible():
    """Same seed + same theta should give same evaluate result (VMC chain reproducible)."""
    problem = J1J2Problem(Lx=2, Ly=2, alpha=2, J2=0.5, device='cpu',
                          thermalization_steps=10)

    problem.set_seed(42)
    theta = problem.random_feasible(1)

    problem.set_seed(100)
    cost_a, _, _ = problem.evaluate(theta, n_samples=100)

    problem.set_seed(100)
    cost_b, _, _ = problem.evaluate(theta, n_samples=100)

    assert torch.allclose(cost_a, cost_b), (
        f"Same seed evaluate gave different costs: "
        f"{cost_a.item():.6f} vs {cost_b.item():.6f}"
    )
