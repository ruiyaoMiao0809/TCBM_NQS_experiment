"""
experiments/run_gradient_baseline_v2.py
========================================
Day 3 P0-1.2 retry. Same algorithmic config as v1, but fixes Day 2's three
engineering defects that caused 100% data loss on a 6h22min run:

  1. python -u + sys.stdout.reconfigure(line_buffering=True)
       — without this, stdout block-buffers when piped through `tee` and
         flushes nothing until process exit.
  2. signal.signal(SIGTERM, _sigterm_handler) + sys.exit(0)
       — Python's default SIGTERM handling is uncaught termination, which
         skips atexit handlers. We catch and exit cleanly.
  3. atexit.register(_atexit_dump) + hard-checkpoint every 500 steps
       — partial state JSON is written every 500 steps and on any clean exit
         (including SIGTERM-triggered sys.exit).

Plus these v1→v2 deltas:
  4. progress_callback prints one line every CALLBACK_EVERY=100 steps so
     long-running progress is visible in the tee'd log.
  5. cfg.n_vmc_samples_final reduced 4 → 2. Day 2 final-eval round-1 alone
     took ~57 min; n_final=2 halves that and still gives a reasonable
     debiased estimate (σ_E ↑ ~30% which is still well below 5% target).

Run:
    python -u experiments/run_gradient_baseline_v2.py 2>&1 \
      | tee logs/baseline_v2_$(date +%Y%m%d_%H%M).log

Expected wall: 4.5–5.5h based on Day 2 heartbeat phase analysis
  (4h main loop + ~30 min final eval at n_final=2).
Run benchmark_per_step.py first to confirm projection.
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

# Defense-in-depth: even without `python -u`, line-buffer stdout.
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.j1j2_problem import J1J2Problem
from core.tcbm_optimizer_NQS import TCBMOptimizer, TCBMConfig

E_0_TRUTH = -8.4579
P0_THRESHOLD = 0.10
R_ABORT_THRESHOLD = 0.15
CALLBACK_EVERY = 100
CHECKPOINT_EVERY = 500

CHECKPOINT_PATH = Path('results/baseline_v2_checkpoint.json')
FINAL_PATH = Path('results/baseline_v2_seed42.json')

# Module-level state shared across callback / atexit / signal handler.
_state: Dict[str, Any] = {
    'last_step': 0,
    'best_cost_so_far': float('inf'),
    'cost_history': [],
    'sigma_E_history': [],
    'swap_acc_history': [],
    'psi_max': 0.0,
    'T_w_step': None,
    'cfg': None,
    'start_time': None,
    'completed': False,
}


def dump_state(path: Path, label: str) -> None:
    """Idempotent snapshot of _state to JSON. Safe to call from any context."""
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
        # Last-ditch: never let the dump crash the process.
        print(f"[{label}] FAILED to dump: {type(e).__name__}: {e}", flush=True)


def progress_callback(step: int, info: Dict[str, float]) -> None:
    """Called by optimize() every CALLBACK_EVERY steps."""
    _state['last_step'] = step
    _state['cost_history'].append(float(info.get('cost_mean', 0.0)))
    _state['sigma_E_history'].append(float(info.get('sigma_E_max', 0.0)))
    _state['swap_acc_history'].append(float(info.get('swap_acc_recent', 0.0)))

    cost_min = float(info.get('cost_min', float('inf')))
    if cost_min < _state['best_cost_so_far']:
        _state['best_cost_so_far'] = cost_min

    psi_max = float(info.get('psi_max', 0.0))
    if psi_max > _state['psi_max']:
        _state['psi_max'] = psi_max

    if info.get('T_w_step') is not None and _state['T_w_step'] is None:
        _state['T_w_step'] = int(info['T_w_step'])

    print(
        f"[step {step:4d}] "
        f"cost_min={cost_min:.4f} "
        f"cost_mean={info.get('cost_mean', 0.0):.4f} "
        f"sigma_E_max={info.get('sigma_E_max', 0.0):.4f} "
        f"swap_acc={info.get('swap_acc_recent', 0.0):.3f} "
        f"psi_max={_state['psi_max']:.3f} "
        f"subspace={'ON' if info.get('subspace_active') else 'off'}",
        flush=True,
    )

    if step > 0 and step % CHECKPOINT_EVERY == 0:
        dump_state(CHECKPOINT_PATH, f'checkpoint_step{step}')


def _sigterm_handler(signum, frame):
    """Convert SIGTERM into a clean sys.exit so atexit handlers run."""
    print(
        f"\n[SIGTERM received at step {_state['last_step']}, attempting graceful exit]",
        flush=True,
    )
    sys.exit(0)


def _atexit_dump():
    """Final safety net. Writes a partial JSON if main() did not complete."""
    if _state['completed']:
        return
    if _state['last_step'] == 0 and _state['start_time'] is None:
        # Never even started — nothing to save.
        return
    print("\n[atexit] dumping partial state due to interrupted exit", flush=True)
    dump_state(FINAL_PATH.with_suffix('.partial.json'), 'atexit_partial')


# CRITICAL: register at module load time, before main() runs.
signal.signal(signal.SIGTERM, _sigterm_handler)
atexit.register(_atexit_dump)


def main():
    torch.manual_seed(42)
    _state['start_time'] = time.time()

    DEVICE = os.environ.get('TCBM_DEVICE', 'cuda:0')
    problem = J1J2Problem(Lx=4, Ly=4, alpha=2, J1=1.0, J2=0.5, device=DEVICE)
    print(f"Using device: {DEVICE} (set TCBM_DEVICE env var to override)")
    print(f"Problem: 4x4 J1-J2 PBC, J2/J1=0.5, D_params={problem.dim} (RBM α=2 real params)")

    cfg = TCBMConfig(
        M=12, n_steps=3000, k=20,
        T_min=0.005, T_max=2.0, T_min_floor=0.002,
        lambda_min=0.05, lambda_max=2.0, tau_lambda=500,
        subspace_warmup=150, subspace_update_freq=80,
        psi_star=1.5, min_warmup=120,
        grad_buffer_size=400, grad_buffer_use=200,
        subspace_source='gradient',
        n_vmc_samples=2000,
        n_vmc_samples_final=2,        # ← Day 3 change (was 4 in v1)
        noise_temperature_alpha=1.0,
        seed=42,
    )
    _state['cfg'] = {k: v for k, v in cfg.__dict__.items() if not k.startswith('_')}

    print(f"\nConfig: M={cfg.M}, n_steps={cfg.n_steps}, k={cfg.k}")
    print(f"  n_vmc_samples_final={cfg.n_vmc_samples_final} (Day 3 reduced from 4)")
    print(f"  subspace_warmup={cfg.subspace_warmup}, freq={cfg.subspace_update_freq}")
    print(f"  expected ~{(cfg.n_steps - cfg.subspace_warmup) // cfg.subspace_update_freq} SVD updates total")
    print(f"  psi_star={cfg.psi_star} (T_w trigger threshold)")
    print(f"\nTarget: |E - {E_0_TRUTH}| / {abs(E_0_TRUTH):.4f} < {P0_THRESHOLD*100:.0f}%")
    print(f"Abort:  > {R_ABORT_THRESHOLD*100:.0f}%")
    print(f"Expected wall: 4.5-5.5h (run benchmark_per_step.py first to confirm)")
    print(f"Progress prints every {CALLBACK_EVERY} steps")
    print(f"Hard checkpoints every {CHECKPOINT_EVERY} steps to {CHECKPOINT_PATH}")
    print(f"{'='*70}\n")

    assert cfg.subspace_source == 'gradient', f"unexpected subspace_source: {cfg.subspace_source}"

    optimizer = TCBMOptimizer(problem, cfg)
    result = optimizer.optimize(callback=progress_callback, callback_every=CALLBACK_EVERY)

    elapsed = time.time() - _state['start_time']

    rel_error_raw = abs(result['best_cost_raw'] - E_0_TRUTH) / abs(E_0_TRUTH)
    rel_error_debiased = abs(result['best_cost_debiased'] - E_0_TRUTH) / abs(E_0_TRUTH)

    if rel_error_debiased < P0_THRESHOLD:
        verdict = "PASS"
        verdict_emoji = "PASS"
    elif rel_error_debiased < R_ABORT_THRESHOLD:
        verdict = "MARGINAL (10-15%, needs Day 4 retune)"
        verdict_emoji = "MARGINAL"
    else:
        verdict = "TRIGGER R-ABORT-1"
        verdict_emoji = "FAIL"

    T_w_step = result.get('T_w', None)
    T_w_str = f"step {T_w_step}" if T_w_step is not None else (
        "never (psi never reached psi_star=1.5)"
    )

    swap_acc_traj = result.get('trajectory', {}).get('swap_acceptance', [])
    if swap_acc_traj and len(swap_acc_traj) >= 30:
        n_traj = len(swap_acc_traj)
        third = n_traj // 3
        swap_early = float(np.mean(swap_acc_traj[:third]))
        swap_mid = float(np.mean(swap_acc_traj[third:2 * third]))
        swap_late = float(np.mean(swap_acc_traj[2 * third:]))
        swap_overall = float(np.mean(swap_acc_traj))
        swap_std = float(np.std(swap_acc_traj))
    elif swap_acc_traj:
        swap_overall = float(np.mean(swap_acc_traj))
        swap_std = float(np.std(swap_acc_traj))
        swap_early = swap_mid = swap_late = swap_overall
    else:
        swap_overall = swap_std = swap_early = swap_mid = swap_late = float('nan')

    psi_history = result.get('psi_history', [])
    psi_max = max((p for _, p in psi_history), default=0.0) if psi_history else 0.0

    print(f"\n{'='*70}")
    print(f"Run completed in {elapsed:.1f}s ({elapsed/60:.1f} min = {elapsed/3600:.2f}h)")
    print(f"{'='*70}")
    print(f"best_cost_raw       = {result['best_cost_raw']:.4f}")
    print(f"best_cost_debiased  = {result['best_cost_debiased']:.4f}")
    print(f"E_0 (ED truth)      = {E_0_TRUTH:.4f}")
    print(f"Relative error (raw)      = {rel_error_raw*100:.2f}%")
    print(f"Relative error (debiased) = {rel_error_debiased*100:.2f}%")
    print(f"\n[{verdict_emoji}] P0-1.2 Verdict: {verdict}")

    print(f"\nDiagnostics:")
    print(f"  swap_acceptance overall = {swap_overall:.3f} (target [0.15, 0.35])")
    print(f"  swap_acceptance early/mid/late = {swap_early:.3f} / {swap_mid:.3f} / {swap_late:.3f}")
    print(f"  swap_acceptance std = {swap_std:.3f}")
    print(f"  total swap events = {result.get('n_accepted_swaps', 'N/A')}")
    print(f"  final sigma_E mean = {result['final_sigma_E_mean']:.4f}")
    print(f"  final sigma_E max  = {result['final_sigma_E_max']:.4f}")
    print(f"  T_w triggered: {T_w_str}")
    print(f"  psi_history len = {len(psi_history)}")
    print(f"  psi_max cumulative = {psi_max:.4f}")

    if result['best_cost_debiased'] < result['best_cost_raw']:
        print(f"  NQS-1e debiasing: working (debiased < raw)")
    else:
        print(f"  NQS-1e debiasing: WARN debiased >= raw, suspect bug")

    save_data = {
        'config': _state['cfg'],
        'best_cost_raw': float(result['best_cost_raw']),
        'best_cost_debiased': float(result['best_cost_debiased']),
        'final_sigma_E_mean': float(result['final_sigma_E_mean']),
        'final_sigma_E_max': float(result['final_sigma_E_max']),
        'rel_error_raw': float(rel_error_raw),
        'rel_error_debiased': float(rel_error_debiased),
        'elapsed_seconds': elapsed,
        'E_0_truth': E_0_TRUTH,
        'verdict': verdict,
        'T_w_step': T_w_step,
        'swap_acceptance_overall': swap_overall,
        'swap_acceptance_std': swap_std,
        'swap_acceptance_early': swap_early,
        'swap_acceptance_mid': swap_mid,
        'swap_acceptance_late': swap_late,
        'n_accepted_swaps_total': int(result.get('n_accepted_swaps', 0)),
        'psi_history_len': len(psi_history),
        'psi_max_cumulative': float(psi_max),
        'cost_trajectory_full': [float(c) for c in result.get('trajectory', {}).get('cost', [])],
        'sigma_trajectory_full': [float(s) for s in result.get('trajectory', {}).get('sigma_E', [])],
        'swap_acceptance_trajectory_full': [float(s) for s in swap_acc_traj],
        'progress_history': {
            'cost_mean_per_callback': _state['cost_history'],
            'sigma_E_max_per_callback': _state['sigma_E_history'],
            'swap_acc_per_callback': _state['swap_acc_history'],
        },
    }

    FINAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FINAL_PATH, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nFinal saved: {FINAL_PATH}")

    # Mark completed AFTER final dump succeeds, so atexit_dump won't double-write.
    _state['completed'] = True

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print(f"Cleaned checkpoint: {CHECKPOINT_PATH}")


if __name__ == '__main__':
    main()
