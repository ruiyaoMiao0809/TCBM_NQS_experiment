"""
experiments/run_sr_baseline.py
================================
Day 6+ Stochastic Reconfiguration (SR) baseline for TCBM-NQS POC.

PRIMARY baseline (replaces Adam in Prediction v3). Adam baseline is now
demoted to an ablation only -- vanilla Adam without natural-gradient
preconditioning cannot find the J1-J2 ground state (Day 3 diagnostic
showed rel_err > 50% across seeds at α=2 RBM, while Bukov 2021 achieves
~10^-3 with SR).

Reference comparison schema (Day 6+):
    TCBM-gradient (run_gradient_baseline_v2)  vs  SR (this)
    POC main hypothesis (v3): σ_TCBM / σ_SR ≤ 0.5

Robustness suite identical to v2 / run_adam_baseline (so SIGTERM at the
wall-clock cap loses no data; stdout is line-buffered through tee):
    1. python -u + sys.stdout.reconfigure(line_buffering=True)
    2. signal.signal(SIGTERM, _sigterm_handler)
    3. atexit.register(_atexit_dump)
    4. Hard checkpoint every CHECKPOINT_EVERY steps
    5. Progress print every CALLBACK_EVERY steps

Run (Day 6+ once core/sr_optimizer.py is implemented):
    TCBM_DEVICE=cuda:N python -u experiments/run_sr_baseline.py 2>&1 \
      | tee logs/sr_$(date +%Y%m%d_%H%M).log

Expected wall: ~30-90 min on a single A10/A100 (D=1120 pinv solve is
fast, dominant cost is QGT build O(N·D²) per step).

Status: skeleton (Day 6+ implementation pending core/sr_optimizer.py).
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
from core.sr_optimizer import SROptimizer, SRConfig

E_0_TRUTH = -8.4579
P0_THRESHOLD = 0.10                  # rel_err target (paper claim A)
R_ABORT_THRESHOLD = 0.15             # if SR can't reach 15% rel_err, abort
SR_TARGET_REL_ERR = 0.001            # Bukov 2021 4x4 SR achievable (Section 5.1)

N_STEPS = 2000
N_VMC_SAMPLES = 2000
N_FINAL_AVERAGES = 4
LR = 0.05                            # Bukov-typical SR learning rate
EPSILON = 1e-3
EPSILON_MODE = 'scale-aware'
SOLVER = 'pinv'
CALLBACK_EVERY = 100
CHECKPOINT_EVERY = 500

CHECKPOINT_PATH = Path('results/sr_checkpoint.json')
FINAL_PATH = Path('results/sr_seed42.json')

_state: Dict[str, Any] = {
    'last_step': 0,
    'best_cost_so_far': float('inf'),
    'cost_history': [],
    'sigma_E_history': [],
    'qgt_cond_history': [],          # SR-specific: log10 condition number per callback
    'sr_residual_history': [],       # SR-specific: ‖S Δθ - g‖/‖g‖ per callback
    'cfg': None,
    'start_time': None,
    'completed': False,
}


def dump_state(path: Path, label: str) -> None:
    """Idempotent snapshot of _state to JSON. Schema parity with v2/Adam."""
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


signal.signal(signal.SIGTERM, _sigterm_handler)
atexit.register(_atexit_dump)


def progress_callback(step: int, info: Dict[str, float]) -> None:
    """Callback fired by SROptimizer every CALLBACK_EVERY steps.

    Records per-callback diagnostics into _state and prints concise progress.
    SR-specific fields (qgt_cond_number, sr_solve_residual) are extracted
    if present; otherwise treated as None.
    """
    cost = info.get('cost_mean', float('nan'))
    sigma = info.get('sigma_E_mean', float('nan'))
    qgt_cond = info.get('qgt_cond_number')
    sr_res = info.get('sr_solve_residual')

    _state['last_step'] = step
    _state['cost_history'].append(float(cost))
    _state['sigma_E_history'].append(float(sigma))
    if qgt_cond is not None:
        _state['qgt_cond_history'].append(float(qgt_cond))
    if sr_res is not None:
        _state['sr_residual_history'].append(float(sr_res))

    if cost < _state['best_cost_so_far']:
        _state['best_cost_so_far'] = float(cost)

    qgt_str = f"qgt_cond={qgt_cond:.2f}" if qgt_cond is not None else "qgt_cond=N/A"
    res_str = f"res={sr_res:.2e}" if sr_res is not None else "res=N/A"
    print(
        f"[step {step:4d}] "
        f"cost={cost:.4f} sigma_E={sigma:.4f} "
        f"best={_state['best_cost_so_far']:.4f} "
        f"{qgt_str} {res_str}",
        flush=True,
    )

    if step > 0 and step % CHECKPOINT_EVERY == 0:
        dump_state(CHECKPOINT_PATH, f'checkpoint_step{step}')


def main():
    _state['start_time'] = time.time()

    DEVICE = os.environ.get('TCBM_DEVICE', 'cuda:0')
    problem = J1J2Problem(Lx=4, Ly=4, alpha=2, J1=1.0, J2=0.5, device=DEVICE)
    print(f"Using device: {DEVICE} (set TCBM_DEVICE env var to override)")
    print(f"Problem: 4x4 J1-J2 PBC, J2/J1=0.5, D_params={problem.dim} (RBM α=2 real params)")

    cfg = SRConfig(
        lr=LR,
        epsilon=EPSILON,
        epsilon_mode=EPSILON_MODE,
        solver=SOLVER,
        n_steps=N_STEPS,
        n_vmc_samples=N_VMC_SAMPLES,
        n_vmc_samples_final=N_FINAL_AVERAGES,
        seed=42,
        record_qgt_eigvals=True,
        record_qgt_every=200,
        record_trajectory=True,
    )
    _state['cfg'] = {
        'optimizer': 'SR',
        'lr': cfg.lr,
        'epsilon': cfg.epsilon,
        'epsilon_mode': cfg.epsilon_mode,
        'solver': cfg.solver,
        'n_steps': cfg.n_steps,
        'n_vmc_samples': cfg.n_vmc_samples,
        'n_vmc_samples_final': cfg.n_vmc_samples_final,
        'seed': cfg.seed,
        'M': 1,
    }

    print(f"\nConfig: SR(lr={LR}, ε={EPSILON} [{EPSILON_MODE}], solver={SOLVER}), n_steps={N_STEPS}")
    print(f"  n_vmc_samples={N_VMC_SAMPLES}, final_eval={N_FINAL_AVERAGES}× averaged")
    print(f"\nTarget: |E - {E_0_TRUTH}| / {abs(E_0_TRUTH):.4f} < {P0_THRESHOLD*100:.0f}%")
    print(f"Bukov anchor (4x4 SR): rel_err ≈ {SR_TARGET_REL_ERR*100:.2f}%")
    print(f"Abort:  > {R_ABORT_THRESHOLD*100:.0f}%")
    print(f"Expected wall: ~30-90 min")
    print(f"Progress prints every {CALLBACK_EVERY} steps")
    print(f"Hard checkpoints every {CHECKPOINT_EVERY} steps to {CHECKPOINT_PATH}")
    print(f"{'='*70}\n")

    # Day 6+ TODO: instantiate optimizer and run.
    # Currently SROptimizer.__init__ raises NotImplementedError.
    optimizer = SROptimizer(problem, cfg)
    result = optimizer.optimize(callback=progress_callback, callback_every=CALLBACK_EVERY)

    # === Final-eval handling (schema parity with v2 / Adam) ===
    best_cost_raw = _state['best_cost_so_far']
    best_cost_debiased = float(result['best_cost_debiased'])
    final_costs = result.get('final_costs_per_shot', [])
    final_sigmas = result.get('final_sigmas_per_shot', [])
    final_sigma_E_mean = float(result.get('final_sigma_E_mean', float('nan')))
    final_sigma_E_max = float(result.get('final_sigma_E_max', float('nan')))

    elapsed = time.time() - _state['start_time']

    rel_error_raw = abs(best_cost_raw - E_0_TRUTH) / abs(E_0_TRUTH)
    rel_error_debiased = abs(best_cost_debiased - E_0_TRUTH) / abs(E_0_TRUTH)

    if rel_error_debiased < P0_THRESHOLD:
        verdict = "PASS"
    elif rel_error_debiased < R_ABORT_THRESHOLD:
        verdict = "MARGINAL (10-15%, SR may need lr/ε retune)"
    else:
        verdict = "TRIGGER R-ABORT-1 (SR diverged or stuck — investigate QGT condition)"

    print(f"\n{'='*70}")
    print(f"Run completed in {elapsed:.1f}s ({elapsed/60:.1f} min = {elapsed/3600:.2f}h)")
    print(f"{'='*70}")
    print(f"best_cost_raw       = {best_cost_raw:.4f}  (best single-shot during training, noisy)")
    print(f"best_cost_debiased  = {best_cost_debiased:.4f}  (mean of {N_FINAL_AVERAGES} final shots)")
    print(f"E_0 (ED truth)      = {E_0_TRUTH:.4f}")
    print(f"Relative error (raw)      = {rel_error_raw*100:.2f}%")
    print(f"Relative error (debiased) = {rel_error_debiased*100:.2f}%")
    print(f"Bukov 4x4 SR achievable  ≈ {SR_TARGET_REL_ERR*100:.2f}%")
    print(f"\n[{verdict}] P1-1.1-SR baseline verdict")

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
            'qgt_cond_per_callback': _state['qgt_cond_history'],
            'sr_residual_per_callback': _state['sr_residual_history'],
        },
        # PT-specific fields are None (schema parity with v2)
        'T_w_step': None,
        'swap_acceptance_overall': None,
        'swap_acceptance_std': None,
        'swap_acceptance_early': None,
        'swap_acceptance_mid': None,
        'swap_acceptance_late': None,
        'n_accepted_swaps_total': 0,
        'psi_history_len': 0,
        'psi_max_cumulative': 0.0,
        # SR-specific final diagnostics
        'qgt_eigvals_final': result.get('qgt_eigvals_final'),
    }

    FINAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FINAL_PATH, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nFinal saved: {FINAL_PATH}")

    _state['completed'] = True

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print(f"Cleaned checkpoint: {CHECKPOINT_PATH}")


if __name__ == '__main__':
    main()
