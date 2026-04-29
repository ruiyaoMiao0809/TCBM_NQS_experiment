"""
experiments/run_adam_baseline.py
=================================
Day 4 P1-1.1: Adam baseline (no PT, no clamping, no NQS-1e debiasing).

Reference comparison: TCBM-gradient (run_gradient_baseline_v2) vs Adam (this).
POC main hypothesis: Adam should converge SLOWER and with HIGHER seed-to-seed
σ than TCBM-gradient on the J2/J1=0.5 frustrated landscape.

Robustness suite identical to v2 (so a SIGTERM at the wall-clock cap does not
lose data, and stdout is line-buffered through tee):
  1. python -u + sys.stdout.reconfigure(line_buffering=True)
  2. signal.signal(SIGTERM, _sigterm_handler)
  3. atexit.register(_atexit_dump)
  4. Hard checkpoint every CHECKPOINT_EVERY steps
  5. Progress print every CALLBACK_EVERY steps

Run:
    TCBM_DEVICE=cuda:N python -u experiments/run_adam_baseline.py 2>&1 \
      | tee logs/adam_$(date +%Y%m%d_%H%M).log

Expected wall: ~30–60 min on a single A10/A100. M=1 vs TCBM M=12 gives a ~12×
theoretical speedup, but PyTorch overhead per VMC sample drives the actual
ratio toward 6–8×.

Note on gradient quality:
We use autograd through evaluate() (the "partial gradient" pattern documented
in J1J2Problem.gradient — missing the score-function term). Bukov 2021 / NetKet
use the log-derivative estimator for unbiased lower-variance gradients; that is
a Week 2+ TODO and not relevant for this Adam-vs-TCBM POC reference.
"""
import sys
import os
import json
import time
import signal
import atexit
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.j1j2_problem import J1J2Problem

E_0_TRUTH = -8.4579
P0_THRESHOLD = 0.10
R_ABORT_THRESHOLD = 0.15

N_STEPS = 3000
N_VMC_SAMPLES = 2000
N_FINAL_AVERAGES = 4         # multi-shot mean as Adam's analog of NQS-1e
LR = 0.001
BETAS = (0.9, 0.999)
CALLBACK_EVERY = 100
CHECKPOINT_EVERY = 500

CHECKPOINT_PATH = Path('results/adam_checkpoint.json')
FINAL_PATH = Path('results/adam_seed42.json')

_state: Dict[str, Any] = {
    'last_step': 0,
    'best_cost_so_far': float('inf'),
    'cost_history': [],
    'sigma_E_history': [],
    'cfg': None,
    'start_time': None,
    'completed': False,
}


def dump_state(path: Path, label: str) -> None:
    """Idempotent snapshot of _state to JSON. Safe from any context."""
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = dict(_state)
    snapshot['dump_label'] = label
    snapshot['dump_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    if _state['start_time'] is not None:
        snapshot['elapsed_seconds'] = time.time() - _state['start_time']
    try:
        with open(path, 'w') as f:
            json.dump(snapshot, f, indent=2, default=str)
        print(f"[{label}] dumped state to {path} (step={_state['last_step']})", flush=True)
    except Exception as e:
        print(f"[{label}] FAILED to dump: {type(e).__name__}: {e}", flush=True)


def _sigterm_handler(signum, frame):
    print(
        f"\n[SIGTERM received at step {_state['last_step']}, attempting graceful exit]",
        flush=True,
    )
    sys.exit(0)


def _atexit_dump():
    if _state['completed']:
        return
    if _state['last_step'] == 0 and _state['start_time'] is None:
        return
    print("\n[atexit] dumping partial state due to interrupted exit", flush=True)
    dump_state(FINAL_PATH.with_suffix('.partial.json'), 'atexit_partial')


# Register at module load (before main()).
signal.signal(signal.SIGTERM, _sigterm_handler)
atexit.register(_atexit_dump)


def main():
    torch.manual_seed(42)
    _state['start_time'] = time.time()

    DEVICE = os.environ.get('TCBM_DEVICE', 'cuda:0')
    problem = J1J2Problem(Lx=4, Ly=4, alpha=2, J1=1.0, J2=0.5, device=DEVICE)
    print(f"Using device: {DEVICE} (set TCBM_DEVICE env var to override)")
    print(f"Problem: 4x4 J1-J2 PBC, J2/J1=0.5, D_params={problem.dim} (RBM α=2 real params)")

    _state['cfg'] = {
        'optimizer': 'Adam',
        'lr': LR,
        'betas': list(BETAS),
        'n_steps': N_STEPS,
        'n_vmc_samples': N_VMC_SAMPLES,
        'n_final_averages': N_FINAL_AVERAGES,
        'seed': 42,
        'M': 1,
    }

    print(f"\nConfig: Adam(lr={LR}, betas={BETAS}), n_steps={N_STEPS}")
    print(f"  n_vmc_samples={N_VMC_SAMPLES}, final_eval={N_FINAL_AVERAGES}× averaged")
    print(f"\nTarget: |E - {E_0_TRUTH}| / {abs(E_0_TRUTH):.4f} < {P0_THRESHOLD*100:.0f}%")
    print(f"Abort:  > {R_ABORT_THRESHOLD*100:.0f}%")
    print(f"Expected wall: ~30-60 min (M=1 vs TCBM M=12)")
    print(f"Progress prints every {CALLBACK_EVERY} steps")
    print(f"Hard checkpoints every {CHECKPOINT_EVERY} steps to {CHECKPOINT_PATH}")
    print(f"{'='*70}\n")

    # Initialize parameters as a single-batch leaf tensor.
    theta = problem.random_feasible(1).requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=LR, betas=BETAS)

    # Main loop
    for step in range(N_STEPS):
        optimizer.zero_grad()
        cost, penalty, sigma_E = problem.evaluate(theta, n_samples=N_VMC_SAMPLES)
        loss = (cost + penalty).sum()
        loss.backward()
        optimizer.step()

        cost_val = float(cost.detach().item())
        sigma_val = float(sigma_E.detach().item())

        if cost_val < _state['best_cost_so_far']:
            _state['best_cost_so_far'] = cost_val

        _state['last_step'] = step

        if step % CALLBACK_EVERY == 0:
            _state['cost_history'].append(cost_val)
            _state['sigma_E_history'].append(sigma_val)
            print(
                f"[step {step:4d}] "
                f"cost={cost_val:.4f} "
                f"sigma_E={sigma_val:.4f} "
                f"best={_state['best_cost_so_far']:.4f}",
                flush=True,
            )

        if step > 0 and step % CHECKPOINT_EVERY == 0:
            dump_state(CHECKPOINT_PATH, f'checkpoint_step{step}')

    # Final eval: N_FINAL_AVERAGES fresh evaluations of the *final* theta.
    # Mean (not min) for unbiased comparison with TCBM's NQS-1e debiasing,
    # which also uses mean. No PT replicas to debias here, just a
    # multi-shot average on the converged parameter.
    print(f"\nFinal eval: {N_FINAL_AVERAGES} fresh evaluations on final theta...")
    final_costs = []
    final_sigmas = []
    with torch.no_grad():
        for i in range(N_FINAL_AVERAGES):
            c, _, s = problem.evaluate(theta, n_samples=N_VMC_SAMPLES)
            final_costs.append(float(c.item()))
            final_sigmas.append(float(s.item()))
            print(
                f"  shot {i+1}/{N_FINAL_AVERAGES}: cost={final_costs[-1]:.4f} "
                f"sigma={final_sigmas[-1]:.4f}",
                flush=True,
            )

    best_cost_raw = _state['best_cost_so_far']        # best single-shot during training (noisy)
    best_cost_debiased = float(np.mean(final_costs))   # multi-shot mean on final theta
    final_sigma_E_mean = float(np.mean(final_sigmas))
    final_sigma_E_max = float(np.max(final_sigmas))

    elapsed = time.time() - _state['start_time']

    rel_error_raw = abs(best_cost_raw - E_0_TRUTH) / abs(E_0_TRUTH)
    rel_error_debiased = abs(best_cost_debiased - E_0_TRUTH) / abs(E_0_TRUTH)

    if rel_error_debiased < P0_THRESHOLD:
        verdict = "PASS"
    elif rel_error_debiased < R_ABORT_THRESHOLD:
        verdict = "MARGINAL (10-15%, needs retune)"
    else:
        verdict = "TRIGGER R-ABORT-1"

    print(f"\n{'='*70}")
    print(f"Run completed in {elapsed:.1f}s ({elapsed/60:.1f} min = {elapsed/3600:.2f}h)")
    print(f"{'='*70}")
    print(f"best_cost_raw       = {best_cost_raw:.4f}  (best single-shot during training, noisy)")
    print(f"best_cost_debiased  = {best_cost_debiased:.4f}  (mean of {N_FINAL_AVERAGES} final shots)")
    print(f"E_0 (ED truth)      = {E_0_TRUTH:.4f}")
    print(f"Relative error (raw)      = {rel_error_raw*100:.2f}%")
    print(f"Relative error (debiased) = {rel_error_debiased*100:.2f}%")
    print(f"\n[{verdict}] P1-1.1 Adam baseline verdict")
    print(f"\nDiagnostics:")
    print(f"  final sigma_E mean = {final_sigma_E_mean:.4f}")
    print(f"  final sigma_E max  = {final_sigma_E_max:.4f}")
    print(f"  final shots cost  = {[f'{c:.4f}' for c in final_costs]}")
    print(f"  final shots sigma = {[f'{s:.4f}' for s in final_sigmas]}")

    save_data = {
        'config': _state['cfg'],
        'best_cost_raw': best_cost_raw,
        'best_cost_debiased': best_cost_debiased,
        'final_costs_per_shot': final_costs,
        'final_sigmas_per_shot': final_sigmas,
        'final_sigma_E_mean': final_sigma_E_mean,
        'final_sigma_E_max': final_sigma_E_max,
        'rel_error_raw': float(rel_error_raw),
        'rel_error_debiased': float(rel_error_debiased),
        'elapsed_seconds': elapsed,
        'E_0_truth': E_0_TRUTH,
        'verdict': verdict,
        'progress_history': {
            'cost_per_callback': _state['cost_history'],
            'sigma_E_per_callback': _state['sigma_E_history'],
        },
        # Adam has no PT — these fields are None for parity with v2's schema
        'T_w_step': None,
        'swap_acceptance_overall': None,
        'swap_acceptance_std': None,
        'swap_acceptance_early': None,
        'swap_acceptance_mid': None,
        'swap_acceptance_late': None,
        'n_accepted_swaps_total': 0,
        'psi_history_len': 0,
        'psi_max_cumulative': 0.0,
    }

    FINAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FINAL_PATH, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nFinal saved: {FINAL_PATH}")

    # Mark completed AFTER final dump so atexit_dump won't double-write.
    _state['completed'] = True

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print(f"Cleaned checkpoint: {CHECKPOINT_PATH}")


if __name__ == '__main__':
    main()
