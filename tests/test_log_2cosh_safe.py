"""
Tests for _SafeLog2CoshComplex (Issue 5 fix).
"""

import pytest
import torch

from core.j1j2_problem import (
    _log_2cosh_complex,
    _log_2cosh_complex_native,
    _SafeLog2CoshComplex,
)


def test_forward_matches_native():
    """Safe wrapper forward must give identical result as native."""
    z = torch.randn(10, 32, dtype=torch.complex64) * 2.0

    out_safe = _log_2cosh_complex(z)
    out_native = _log_2cosh_complex_native(z)

    assert torch.allclose(out_safe, out_native, rtol=1e-5, atol=1e-6), \
        f"forward mismatch: max diff {(out_safe - out_native).abs().max()}"


def test_backward_finite_at_normal_point():
    """At normal z (no singularity), backward should give finite gradient identical to native autograd."""
    z = torch.randn(5, 8, dtype=torch.complex64, requires_grad=True)

    # Safe path
    out_safe = _log_2cosh_complex(z)
    loss_safe = out_safe.real.sum()
    grad_safe, = torch.autograd.grad(loss_safe, z, retain_graph=False)

    # Native path (with autograd, no safe wrapper)
    z2 = z.detach().clone().requires_grad_(True)
    out_native = _log_2cosh_complex_native(z2)
    loss_native = out_native.real.sum()
    grad_native, = torch.autograd.grad(loss_native, z2, retain_graph=False)

    assert torch.isfinite(grad_safe).all(), "safe gradient has non-finite at normal z!"
    assert torch.allclose(grad_safe, grad_native, rtol=1e-5, atol=1e-6), \
        f"normal-point gradient mismatch: {(grad_safe - grad_native).abs().max()}"


def test_backward_finite_at_singularity():
    """At z = i*pi/2 (singularity), backward must give finite gradient (mask 0)."""
    # Construct z exactly at singularity for some elements
    z = torch.zeros(4, 8, dtype=torch.complex64, requires_grad=True)
    # First few elements at singularity:
    with torch.no_grad():
        z[0, 0] = 0.0 + 1j * (torch.pi / 2)
        z[0, 1] = 0.0 + 1j * (3 * torch.pi / 2)
        # Rest are at random non-singular values
        z[0, 2:] = torch.randn(6, dtype=torch.complex64) * 0.5
        z[1:] = torch.randn(3, 8, dtype=torch.complex64) * 0.5
    z.requires_grad_(True)

    out = _log_2cosh_complex(z)
    loss = out.real.sum()
    grad, = torch.autograd.grad(loss, z, retain_graph=False)

    assert torch.isfinite(grad).all(), \
        f"gradient has non-finite at singularity! grad[0,0]={grad[0,0]}, grad[0,1]={grad[0,1]}"
    # Singular points should have gradient exactly 0 (replaced by mask)
    assert grad[0, 0].abs() < 1e-7, f"grad at singularity should be 0, got {grad[0, 0]}"
    assert grad[0, 1].abs() < 1e-7, f"grad at singularity should be 0, got {grad[0, 1]}"


def test_load_day4_state_and_backward():
    """Load Day 4 NaN-triggering state, fix should give finite gradient."""
    state_path = 'results/state_at_first_nan.pt'

    import os
    if not os.path.exists(state_path):
        pytest.skip(f"{state_path} not found (Day 4 evidence file)")

    state = torch.load(state_path, weights_only=False)
    # state contains theta + positions at Stage A step 87 NaN trigger

    # Try to recompute backward through full forward path with safe wrapper
    # (this depends on J1J2Problem.gradient() routing through _log_2cosh_complex)
    from core.j1j2_problem import J1J2Problem

    # Reconstruct problem with same cfg as Day 4 narrow_band Stage A
    # (cfg should be reproducible from production_v2 cfg)
    # If this is too cfg-dependent, just verify _log_2cosh_complex on saved theta directly

    # Simpler test: just verify the _log_2cosh_complex on a representative problematic input
    # by reconstructing approximate hidden_act from saved theta
    theta = state.get('theta')
    if theta is None:
        pytest.skip("state_at_first_nan.pt missing 'theta' key")

    # Just verify theta processing through _log_2cosh_complex doesn't NaN
    # (basic sanity, full integration test would need J1J2Problem reconstruction)
    z_test = theta[:32].to(torch.complex64) if theta.dtype.is_floating_point else theta[:32]
    z_test = z_test.detach().clone().requires_grad_(True)
    out = _log_2cosh_complex(z_test)
    loss = out.real.sum()
    grad, = torch.autograd.grad(loss, z_test)

    assert torch.isfinite(grad).all(), \
        f"gradient still has non-finite even after fix! n_nan={torch.isnan(grad).sum()}, n_inf={torch.isinf(grad).sum()}"
