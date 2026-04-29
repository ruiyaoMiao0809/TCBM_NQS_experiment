"""tests/test_callback.py — Day 3 A1: progress callback hook.

Verifies:
  test_callback_called          — callback fires at expected cadence
  test_callback_info_keys       — info dict contains required snapshot keys
  test_callback_default_none    — Day 1 backward compat (no callback arg)
  test_callback_failure_caught  — callback exception does not abort optimize()
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.j1j2_problem import J1J2Problem
from core.tcbm_optimizer_NQS import TCBMOptimizer, TCBMConfig


def _small_setup():
    """Cheapest possible 2×2 J1-J2 + minimal TCBMConfig.

    Empirically each optimize() call here is ~7s/step on CPU; 6 steps + final
    eval ≈ 45 s, so a 4-test suite fits well under a 300 s pytest timeout.
    """
    torch.manual_seed(0)
    problem = J1J2Problem(
        Lx=2, Ly=2, alpha=2, J2=0.5, device='cpu',
        thermalization_steps=10,
    )
    cfg = TCBMConfig(
        M=2, n_steps=6, k=4,
        T_min=0.005, T_max=2.0,
        subspace_warmup=3, subspace_update_freq=4,
        psi_star=1.5, min_warmup=3,
        grad_buffer_size=20, grad_buffer_use=10,
        subspace_source='gradient',
        n_vmc_samples=10,
        n_vmc_samples_final=2,
        seed=42,
        record_trajectory=False,
    )
    return problem, cfg


def test_callback_called():
    problem, cfg = _small_setup()
    opt = TCBMOptimizer(problem, cfg)

    calls = []

    def cb(step, info):
        calls.append((step, info))

    result = opt.optimize(callback=cb, callback_every=2)

    assert len(calls) >= 3, f"expected >=3 callbacks (steps 0,2,4), got {len(calls)}"
    steps_seen = [s for s, _ in calls]
    assert steps_seen[0] == 0, f"first callback should be at step 0, got {steps_seen[0]}"
    assert all(s % 2 == 0 for s in steps_seen), f"non-multiple-of-2 step in {steps_seen}"
    # result still complete
    assert 'best_cost_debiased' in result
    assert 'trajectory' in result


def test_callback_info_keys():
    problem, cfg = _small_setup()
    opt = TCBMOptimizer(problem, cfg)

    seen_keys = set()

    def cb(step, info):
        seen_keys.update(info.keys())

    opt.optimize(callback=cb, callback_every=2)

    required = {
        'cost_min', 'cost_mean',
        'sigma_E_max', 'sigma_E_mean',
        'swap_acc_recent', 'subspace_active',
        'psi_max', 'T_w_step',
    }
    missing = required - seen_keys
    assert not missing, f"callback info missing keys: {missing}"


def test_callback_default_none():
    """Day 1 backward compat: optimize() with no arg still works."""
    problem, cfg = _small_setup()
    opt = TCBMOptimizer(problem, cfg)
    result = opt.optimize()
    assert 'best_cost_debiased' in result
    assert 'trajectory' in result


def test_callback_failure_caught():
    """A failing callback must warn but not abort optimize()."""
    problem, cfg = _small_setup()
    opt = TCBMOptimizer(problem, cfg)

    n_called = [0]

    def bad_cb(step, info):
        n_called[0] += 1
        raise RuntimeError("intentional callback failure for test")

    with pytest.warns(UserWarning, match="callback at step .* raised RuntimeError"):
        result = opt.optimize(callback=bad_cb, callback_every=2)

    assert n_called[0] >= 3, "bad callback should still be called multiple times"
    assert 'best_cost_debiased' in result, "optimize() must still return a complete result"
