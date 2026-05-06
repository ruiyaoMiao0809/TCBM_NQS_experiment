"""
experiments/probe_grad_nan.py
==============================
Day 4 Stage 2 probe: locate the precise intermediate where J1J2Problem.gradient()
produces NaN, given that Stage A (subspace OFF) reproduces the production v2
frozen state with grad_nan starting at step 87 (||theta|| ~18-19).

Strategy
--------
Monkey-patch `core.j1j2_problem._log_2cosh_complex` and `core.j1j2_problem.log_psi_rbm`
from this script so that every intermediate quantity in the forward path is
captured to a global log. The functions themselves are not modified — patches
are install-only and live in this process; core/ files untouched.

Then call `problem.evaluate(x)` + `total.backward()` — the production gradient
path — so we observe exactly what production hits, with full intermediate
visibility.

Theta source
------------
Default mode (cheap, ~30 sec total): synthesize thetas at progressive ||θ|| levels
[1.15, 5.0, 10.0, 15.0, 17.0, 18.0, 19.0, 20.0, 22.0, 25.0, 30.0]. This brackets
the observed NaN onset region (||θ|| step50=15.37 finite → step100=19.39 NaN)
and lets us pinpoint the threshold without paying for a 30-min rerun.

Optional mode (--load PATH): load theta from a torch.save dump (exact step-86
state from rerun, if we choose to invest the wall-clock).

Sources read for monkey-patch design (no modifications):
  core/j1j2_problem.py:114-149  log_psi_rbm
  core/j1j2_problem.py:152-230  _log_2cosh_complex
  core/j1j2_problem.py:419-436  gradient
  core/j1j2_problem.py:530-547  _evaluate_vmc
  core/j1j2_problem.py:577-668  _local_energy_batch (esp. line 659: ratio = exp(...))

Run:
    TCBM_DEVICE=cuda:3 python -u experiments/probe_grad_nan.py
    TCBM_DEVICE=cuda:3 python -u experiments/probe_grad_nan.py --load results/probe_state_step86.pt

Output: results/probe_grad_nan_intermediates.json (one entry per probe theta).
"""
import sys
import os
import json
import argparse
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

import core.j1j2_problem as j1j2_mod
from core.j1j2_problem import J1J2Problem

DEVICE = os.environ.get('TCBM_DEVICE', 'cuda:0')
SEED = 42
N_VMC_SAMPLES = 2000

OUTPUT_PATH = Path('results/probe_grad_nan_intermediates.json')

# Default ||theta|| ladder. Brackets observed NaN onset (~18) with margins.
DEFAULT_NORM_LADDER = [1.15, 5.0, 10.0, 15.0, 17.0, 18.0, 19.0, 20.0, 22.0, 25.0, 30.0]


# ─────────────────────────────────────────────────────────────────────────────
# Intermediate capture (per-probe scope; reset between probes)
# ─────────────────────────────────────────────────────────────────────────────

# List of {'point': str, 'call_index': int, 'tensor_stats': dict}
# `call_index` distinguishes repeated calls (e.g. log_psi_rbm called once for
# samples + N_bonds times for sigma_flipped inside _local_energy_batch).
_intermediates: List[Dict[str, Any]] = []
_call_counter: Dict[str, int] = {}


def _reset_capture():
    _intermediates.clear()
    _call_counter.clear()


def _stats(t: torch.Tensor, name: str) -> Dict[str, Any]:
    """Compute compact summary stats for a (possibly complex) tensor."""
    out: Dict[str, Any] = {
        'name': name,
        'shape': list(t.shape),
        'dtype': str(t.dtype),
        'numel': int(t.numel()),
    }
    if t.numel() == 0:
        return out

    detached = t.detach()
    if detached.is_complex():
        real = detached.real
        imag = detached.imag
        out['real_min'] = float(real.min().item())
        out['real_max'] = float(real.max().item())
        out['real_abs_max'] = float(real.abs().max().item())
        out['imag_min'] = float(imag.min().item())
        out['imag_max'] = float(imag.max().item())
        out['imag_abs_max'] = float(imag.abs().max().item())
        out['all_finite_real'] = bool(torch.isfinite(real).all().item())
        out['all_finite_imag'] = bool(torch.isfinite(imag).all().item())
        out['n_inf_real'] = int(torch.isinf(real).sum().item())
        out['n_nan_real'] = int(torch.isnan(real).sum().item())
        out['n_inf_imag'] = int(torch.isinf(imag).sum().item())
        out['n_nan_imag'] = int(torch.isnan(imag).sum().item())
    else:
        out['min'] = float(detached.min().item())
        out['max'] = float(detached.max().item())
        out['abs_max'] = float(detached.abs().max().item())
        out['all_finite'] = bool(torch.isfinite(detached).all().item())
        out['n_inf'] = int(torch.isinf(detached).sum().item())
        out['n_nan'] = int(torch.isnan(detached).sum().item())
    return out


def _record(point: str, t: torch.Tensor, name: str = '') -> None:
    idx = _call_counter.get(point, 0)
    _call_counter[point] = idx + 1
    _intermediates.append({
        'point': point,
        'call_index': idx,
        'tensor_stats': _stats(t, name or point),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Monkey-patches: drop-in replacements for j1j2_mod._log_2cosh_complex and
# j1j2_mod.log_psi_rbm. Logic is identical to source (j1j2_problem.py:152-230,
# 114-149); only added _record() probes.
# ─────────────────────────────────────────────────────────────────────────────

def patched_log_2cosh_complex(z: torch.Tensor) -> torch.Tensor:
    _record('log_2cosh.INPUT_z', z)
    u, v = z.real, z.imag
    _record('log_2cosh.u_real', u)
    _record('log_2cosh.v_imag', v)
    abs_u = torch.abs(u)
    _record('log_2cosh.abs_u', abs_u)
    a = torch.exp(-2.0 * abs_u)
    _record('log_2cosh.a=exp(-2|u|)', a)
    one_plus_a_sq = (1.0 + a) ** 2
    _record('log_2cosh.(1+a)^2', one_plus_a_sq)
    sin_v = torch.sin(v)
    _record('log_2cosh.sin(v)', sin_v)
    sin_v_sq = sin_v ** 2
    _record('log_2cosh.sin(v)^2', sin_v_sq)
    inner = one_plus_a_sq - 4.0 * sin_v_sq * a
    _record('log_2cosh.inner', inner)
    log_inner = torch.log(inner)
    _record('log_2cosh.log(inner)', log_inner)
    log_magnitude = abs_u + 0.5 * log_inner
    _record('log_2cosh.log_magnitude', log_magnitude)
    tanh_u = torch.tanh(u)
    _record('log_2cosh.tanh(u)', tanh_u)
    cos_v = torch.cos(v)
    _record('log_2cosh.cos(v)', cos_v)
    phase = torch.atan2(tanh_u * sin_v, cos_v)
    _record('log_2cosh.phase', phase)
    result = torch.complex(log_magnitude, phase)
    _record('log_2cosh.OUTPUT', result)
    return result


def patched_log_psi_rbm(
    theta: torch.Tensor, sigma: torch.Tensor, N: int, M: int,
) -> torch.Tensor:
    _record('log_psi_rbm.INPUT_theta', theta)
    _record('log_psi_rbm.INPUT_sigma', sigma)
    a, b, W = j1j2_mod.unpack_rbm(theta, N, M)
    _record('log_psi_rbm.unpacked_a', a)
    _record('log_psi_rbm.unpacked_b', b)
    _record('log_psi_rbm.unpacked_W', W)
    sigma_c = sigma.to(a.dtype)
    v_contrib = sigma_c @ a
    _record('log_psi_rbm.v_contrib=sigma·a', v_contrib)
    hidden_act = sigma_c @ W + b.unsqueeze(0)
    _record('log_psi_rbm.hidden_act', hidden_act)
    log_2cosh_vals = j1j2_mod._log_2cosh_complex(hidden_act)
    _record('log_psi_rbm.log_2cosh_vals', log_2cosh_vals)
    h_contrib = log_2cosh_vals.sum(dim=-1)
    _record('log_psi_rbm.h_contrib', h_contrib)
    result = v_contrib + h_contrib
    _record('log_psi_rbm.OUTPUT_log_psi', result)
    return result


def install_patches():
    global _orig_log_2cosh, _orig_log_psi_rbm
    _orig_log_2cosh = j1j2_mod._log_2cosh_complex
    _orig_log_psi_rbm = j1j2_mod.log_psi_rbm
    j1j2_mod._log_2cosh_complex = patched_log_2cosh_complex
    j1j2_mod.log_psi_rbm = patched_log_psi_rbm


def uninstall_patches():
    j1j2_mod._log_2cosh_complex = _orig_log_2cosh
    j1j2_mod.log_psi_rbm = _orig_log_psi_rbm


_orig_log_2cosh = None
_orig_log_psi_rbm = None


# ─────────────────────────────────────────────────────────────────────────────
# Probe one theta
# ─────────────────────────────────────────────────────────────────────────────

def probe_one_theta(
    problem: J1J2Problem,
    theta_1d: torch.Tensor,
    label: str,
) -> Dict[str, Any]:
    """Run forward + backward on theta with intermediate capture; return summary."""
    _reset_capture()

    theta_norm = float(theta_1d.norm().item())
    print(f"\n{'─' * 70}")
    print(f"Probe: {label}  ||theta||={theta_norm:.4f}")
    print(f"{'─' * 70}", flush=True)

    # Match production gradient call: problem.gradient does the same as evaluate+backward.
    # We replicate it inline to control intermediate capture and surface the gradient stats.
    x = theta_1d.unsqueeze(0).detach().requires_grad_(True)  # (1, D)

    t0 = time.time()
    cost, pen, std = problem.evaluate(x, n_samples=N_VMC_SAMPLES)
    eval_secs = time.time() - t0

    cost_stats = _stats(cost, 'cost (after evaluate)')
    pen_stats = _stats(pen, 'pen (after evaluate)')
    std_stats = _stats(std, 'std (after evaluate)')
    print(f"  evaluate()  ({eval_secs:.1f}s)  cost={cost_stats}", flush=True)

    # Backward
    total = (cost + pen).sum()
    total_stats = _stats(total, 'total = (cost+pen).sum()')

    grad_stats = None
    grad_failed = False
    try:
        t1 = time.time()
        total.backward()
        bwd_secs = time.time() - t1
        grad = x.grad.detach()
        grad_stats = _stats(grad, 'd(total)/d(theta)')
        print(f"  backward()  ({bwd_secs:.1f}s)  grad_stats={grad_stats}", flush=True)
    except Exception as e:
        grad_failed = True
        grad_stats = {'error': f'{type(e).__name__}: {e}'}
        print(f"  backward() FAILED: {grad_stats['error']}", flush=True)

    # Find first non-finite intermediate
    first_bad = None
    for entry in _intermediates:
        ts = entry['tensor_stats']
        is_complex = 'real_min' in ts
        if is_complex:
            bad = (not ts.get('all_finite_real', True)) or (
                not ts.get('all_finite_imag', True))
        else:
            bad = not ts.get('all_finite', True)
        if bad:
            first_bad = entry
            break

    if first_bad is not None:
        print(f"  *** FIRST NON-FINITE INTERMEDIATE: {first_bad['point']} "
              f"(call_index={first_bad['call_index']}) ***", flush=True)
        print(f"      stats: {first_bad['tensor_stats']}", flush=True)
    else:
        print(f"  All intermediates finite. Gradient finite: "
              f"{grad_stats.get('all_finite', '(failed)') if grad_stats else 'N/A'}",
              flush=True)

    return {
        'label': label,
        'theta_norm': theta_norm,
        'eval_seconds': eval_secs,
        'cost_stats': cost_stats,
        'pen_stats': pen_stats,
        'std_stats': std_stats,
        'total_stats': total_stats,
        'grad_stats': grad_stats,
        'grad_failed': grad_failed,
        'first_non_finite_intermediate': first_bad,
        'n_intermediates_captured': len(_intermediates),
        'all_intermediates': list(_intermediates),  # full per-call detail
    }


# ─────────────────────────────────────────────────────────────────────────────
# Theta sources
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_thetas(problem: J1J2Problem) -> List[Dict[str, Any]]:
    """Generate theta tensors at progressive ||theta|| levels by scaling a fresh init."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    problem.set_seed(SEED)
    base = problem.random_feasible(1)[0]   # (D,)
    base_norm = float(base.norm().item())
    print(f"Base random_feasible theta: ||theta||={base_norm:.4f}, D={base.shape[0]}",
          flush=True)

    out = []
    for target in DEFAULT_NORM_LADDER:
        scale = target / max(base_norm, 1e-10)
        theta = base.clone() * scale
        out.append({'theta': theta, 'label': f'synthetic_norm={target:.2f}'})
    return out


def load_theta_from_disk(path: Path) -> List[Dict[str, Any]]:
    """Load a theta state previously saved via torch.save."""
    state = torch.load(path, map_location=DEVICE)
    if isinstance(state, dict) and 'theta' in state:
        theta = state['theta']
    elif isinstance(state, torch.Tensor):
        theta = state
    else:
        raise ValueError(
            f"Unrecognized state format in {path}: {type(state)}. "
            f"Expected dict with 'theta' key or torch.Tensor.")
    if theta.dim() == 2 and theta.shape[0] > 1:
        # Batched (M, D); take cold replica
        theta = theta[0]
    elif theta.dim() == 2 and theta.shape[0] == 1:
        theta = theta[0]
    return [{'theta': theta.to(DEVICE), 'label': f'loaded_from={path.name}'}]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default=None,
                        help='Path to torch.save dump of theta (single tensor or '
                             "dict with 'theta' key). Default: synthesize ladder.")
    args = parser.parse_args()

    print(f"probe_grad_nan starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}, seed: {SEED}, n_vmc_samples: {N_VMC_SAMPLES}")
    print(f"Output: {OUTPUT_PATH}", flush=True)

    problem = J1J2Problem(Lx=4, Ly=4, alpha=2, J1=1.0, J2=0.5, device=DEVICE)
    print(f"Problem: 4x4 J1-J2 PBC, J2/J1=0.5, D_params={problem.dim}", flush=True)

    install_patches()

    if args.load is not None:
        thetas = load_theta_from_disk(Path(args.load))
    else:
        thetas = make_synthetic_thetas(problem)

    print(f"\nProbing {len(thetas)} theta(s)...", flush=True)

    results = []
    for entry in thetas:
        try:
            r = probe_one_theta(problem, entry['theta'], entry['label'])
            results.append(r)
        except Exception as e:
            print(f"  PROBE FAILED for {entry['label']}: "
                  f"{type(e).__name__}: {e}", flush=True)
            results.append({
                'label': entry['label'],
                'error': f'{type(e).__name__}: {e}',
            })

    uninstall_patches()

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("=== Probe Summary ===")
    print(f"{'=' * 70}")
    for r in results:
        if 'error' in r:
            print(f"[{r['label']}] PROBE ERROR: {r['error']}")
            continue
        norm = r['theta_norm']
        gs = r['grad_stats'] or {}
        grad_finite = gs.get('all_finite', None)
        first_bad = r.get('first_non_finite_intermediate')
        if first_bad is None and grad_finite:
            verdict = 'CLEAN'
            detail = f"||grad||={gs.get('abs_max', 'n/a')}"
        elif first_bad is not None:
            verdict = 'NON_FINITE_INTERMEDIATE'
            detail = f"first @ {first_bad['point']}"
        else:
            verdict = 'GRAD_FAILED'
            detail = gs.get('error', '')
        print(f"  ||theta||={norm:7.3f}  →  {verdict:30s}  {detail}")

    save = {
        'config': {
            'device': DEVICE,
            'seed': SEED,
            'n_vmc_samples': N_VMC_SAMPLES,
            'norm_ladder': DEFAULT_NORM_LADDER,
            'load_path': args.load,
        },
        'results': results,
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_PATH}")
    print("=== End ===")


if __name__ == '__main__':
    main()
