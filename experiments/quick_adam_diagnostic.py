"""
experiments/quick_adam_diagnostic.py
=====================================
Day 3 Stage 1: probe 4x4 NQS landscape ruggedness via Adam baseline.

5 seeds x Adam M=1 x n_steps=2000 x n_samples=2000 x n_final=4 (mean)

Outputs sigma_Adam diagnostic. Decides whether 4x4 has enough ruggedness for
TCBM advantage to manifest, before committing 3 weeks to it.

Decision rule:
  sigma_Adam / |E_0| < 0.5%: WARNING (landscape too smooth, abandon 4x4)
  sigma_Adam / |E_0| in [0.5%, 1%]: MARGINAL (proceed but hedged)
  sigma_Adam / |E_0| in [1%, 2%]: GO (rugged enough, commit to 4x4 plan)
  sigma_Adam / |E_0| >= 2%: STRONG GO

Three-piece robustness (sequential 5 seeds, ~2.5h total):
  1. python -u + sys.stdout.reconfigure(line_buffering=True)
  2. signal.signal(SIGTERM, sigterm_handler)
  3. atexit.register(atexit_dump)
  + per-seed checkpoint to results/quick_adam_diagnostic.json after each seed

Run:
    TCBM_DEVICE=cuda:3 python -u experiments/quick_adam_diagnostic.py 2>&1 \
      | tee logs/adam_diagnostic_$(date +%Y%m%d_%H%M).log
"""
import sys
import os
import json
import time
import signal
import atexit
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.j1j2_problem import J1J2Problem

E_0_TRUTH = -8.4579
DEVICE = os.environ.get('TCBM_DEVICE', 'cuda:0')
SEEDS = [42, 7, 13, 21, 99]
N_STEPS_PER_SEED = 2000
N_VMC_SAMPLES = 2000
N_FINAL_SHOTS = 4
LR = 0.001
BETAS = (0.9, 0.999)

OUTPUT_PATH = Path('results/quick_adam_diagnostic.json')

_results = {
    'seeds_done': [],
    'partial': True,
    'config': {
        'seeds': SEEDS,
        'n_steps_per_seed': N_STEPS_PER_SEED,
        'n_vmc_samples': N_VMC_SAMPLES,
        'n_final_shots': N_FINAL_SHOTS,
        'lr': LR,
        'betas': list(BETAS),
        'device': DEVICE,
    },
    'start_time': None,
}


def dump_state(label='partial'):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot = dict(_results)
    snapshot['dump_label'] = label
    snapshot['dump_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    if _results['start_time'] is not None:
        snapshot['elapsed_seconds'] = time.time() - _results['start_time']
    try:
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(snapshot, f, indent=2, default=str)
        print(f"[{label}] dumped to {OUTPUT_PATH} ({len(_results['seeds_done'])}/{len(SEEDS)} seeds done)",
              flush=True)
    except Exception as e:
        print(f"[{label}] FAILED to dump: {type(e).__name__}: {e}", flush=True)


def sigterm_handler(signum, frame):
    print(f"\n[SIGTERM after seed {len(_results['seeds_done'])}/{len(SEEDS)}, graceful exit]",
          flush=True)
    sys.exit(0)


def atexit_dump():
    if not _results['partial']:
        return
    if _results['start_time'] is None:
        return
    print("\n[atexit] dumping partial state due to interrupted exit", flush=True)
    dump_state('atexit_partial')


signal.signal(signal.SIGTERM, sigterm_handler)
atexit.register(atexit_dump)


def run_one_seed(seed):
    """Run single Adam optimization for one seed; return result dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    problem = J1J2Problem(Lx=4, Ly=4, alpha=2, J1=1.0, J2=0.5, device=DEVICE)
    theta = problem.random_feasible(1).requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=LR, betas=BETAS)

    cost_trajectory = []
    sigma_trajectory = []
    t_seed_start = time.time()

    for step in range(N_STEPS_PER_SEED):
        optimizer.zero_grad()
        cost, penalty, sigma_E = problem.evaluate(theta, n_samples=N_VMC_SAMPLES)
        loss = (cost + penalty).sum()
        loss.backward()
        optimizer.step()

        cost_trajectory.append(float(cost.item()))
        sigma_trajectory.append(float(sigma_E.item()))

        if step % 100 == 0:
            print(
                f"  [seed {seed} step {step:4d}] "
                f"cost={cost.item():.4f} sigma_E={sigma_E.item():.4f}",
                flush=True,
            )

    # Final eval: N_FINAL_SHOTS fresh evaluations on final theta, mean.
    final_costs = []
    with torch.no_grad():
        for i in range(N_FINAL_SHOTS):
            c, _, _ = problem.evaluate(theta, n_samples=N_VMC_SAMPLES)
            final_costs.append(float(c.item()))

    final_cost_mean = float(np.mean(final_costs))
    rel_error = abs(final_cost_mean - E_0_TRUTH) / abs(E_0_TRUTH)
    elapsed = time.time() - t_seed_start

    result = {
        'seed': seed,
        'final_cost_mean': final_cost_mean,
        'final_cost_shots': final_costs,
        'rel_error': rel_error,
        'cost_trajectory': cost_trajectory,
        'sigma_trajectory': sigma_trajectory,
        'elapsed_seconds': elapsed,
    }

    print(
        f"[seed {seed} DONE] final_cost={final_cost_mean:.4f}, "
        f"rel_error={rel_error*100:.2f}%, elapsed={elapsed/60:.1f} min",
        flush=True,
    )

    return result


def main():
    print(f"Quick Adam diagnostic: {len(SEEDS)} seeds x {N_STEPS_PER_SEED} steps x Adam M=1")
    print(f"Device: {DEVICE} (set TCBM_DEVICE env var to override)")
    print(f"Seeds: {SEEDS}")
    print(f"n_vmc_samples={N_VMC_SAMPLES}, n_final_shots={N_FINAL_SHOTS}")
    print(f"Adam(lr={LR}, betas={BETAS})")
    print(f"Expected total wall: ~2.5h (single GPU sequential)")
    print(f"{'='*70}\n")

    _results['start_time'] = time.time()

    for i, seed in enumerate(SEEDS):
        print(f"\n--- Seed {i+1}/{len(SEEDS)}: {seed} ---", flush=True)
        result = run_one_seed(seed)
        _results['seeds_done'].append(result)
        dump_state(f'checkpoint_after_seed{seed}')

    elapsed = time.time() - _results['start_time']

    final_costs = [r['final_cost_mean'] for r in _results['seeds_done']]
    rel_errors = [r['rel_error'] for r in _results['seeds_done']]

    sigma_Adam_abs = float(np.std(final_costs))
    sigma_Adam_rel = sigma_Adam_abs / abs(E_0_TRUTH)
    mean_rel_error = float(np.mean(rel_errors))
    max_rel_error = float(np.max(rel_errors))
    min_rel_error = float(np.min(rel_errors))

    # Cluster analysis: real barrier (multi-basin) vs noise floor (single-basin)
    sorted_costs = sorted(final_costs)
    gaps = [sorted_costs[i+1] - sorted_costs[i] for i in range(len(sorted_costs)-1)]
    max_gap = max(gaps) if gaps else 0.0
    total_range = sorted_costs[-1] - sorted_costs[0] if sorted_costs else 0.0
    gap_ratio = max_gap / total_range if total_range > 0 else 0.0

    if gap_ratio > 0.5:
        cluster_signal = "MULTI_BASIN (real barrier evidence)"
        cluster_emoji = "✅"
    elif gap_ratio > 0.35:
        cluster_signal = "BORDERLINE (weak barrier evidence)"
        cluster_emoji = "🤔"
    else:
        cluster_signal = "NOISE_FLOOR (uniform spread, no barrier)"
        cluster_emoji = "⚠️"

    # Combined verdict: σ_Adam (spread magnitude) + cluster_signal (spread real vs noise)
    if sigma_Adam_rel < 0.005:
        verdict = "WARN: σ_Adam < 0.5% (landscape too smooth, abandon 4×4)"
        verdict_emoji = "⚠️"
    elif sigma_Adam_rel < 0.01:
        if "MULTI_BASIN" in cluster_signal:
            verdict = "MARGINAL_GO: σ ∈ [0.5%, 1%] but multi-basin evidence (proceed with hedge)"
            verdict_emoji = "🤔✓"
        else:
            verdict = "MARGINAL: σ ∈ [0.5%, 1%] noise floor only (caution: TCBM advantage uncertain)"
            verdict_emoji = "🤔"
    elif sigma_Adam_rel < 0.02:
        if "MULTI_BASIN" in cluster_signal:
            verdict = "GO: σ ∈ [1%, 2%] + multi-basin (TCBM should win clearly)"
            verdict_emoji = "✅"
        elif "BORDERLINE" in cluster_signal:
            verdict = "GO: σ ∈ [1%, 2%] + borderline cluster (proceed)"
            verdict_emoji = "✅"
        else:
            verdict = "MARGINAL_GO: σ ∈ [1%, 2%] but noise-dominated (TCBM advantage limited)"
            verdict_emoji = "🤔✓"
    else:
        if "MULTI_BASIN" in cluster_signal:
            verdict = "STRONG GO: σ ≥ 2% + multi-basin (very rugged, TCBM clear win)"
            verdict_emoji = "✅✅"
        else:
            verdict = "GO: σ ≥ 2% but noise-dominated (likely noisy convergence, proceed cautiously)"
            verdict_emoji = "✅"

    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC COMPLETE in {elapsed/60:.1f} min ({elapsed/3600:.2f}h)")
    print(f"{'='*70}")
    print(f"5 seed final costs: {[f'{c:.4f}' for c in final_costs]}")
    print(f"5 seed rel errors:  {[f'{r*100:.2f}%' for r in rel_errors]}")
    print(f"Mean rel error: {mean_rel_error*100:.2f}%")
    print(f"Max rel error:  {max_rel_error*100:.2f}%")
    print(f"Min rel error:  {min_rel_error*100:.2f}%")
    print()
    print(f"sigma_Adam (absolute) = {sigma_Adam_abs:.4f}")
    print(f"sigma_Adam / |E_0|    = {sigma_Adam_rel*100:.3f}%")
    print()
    print(f"Cluster analysis:")
    print(f"  Sorted final costs: {[f'{c:.4f}' for c in sorted_costs]}")
    print(f"  Pairwise gaps:      {[f'{g:.4f}' for g in gaps]}")
    print(f"  Max gap:            {max_gap:.4f}")
    print(f"  Total range:        {total_range:.4f}")
    print(f"  Gap ratio:          {gap_ratio:.2f}")
    print(f"  {cluster_emoji} Cluster signal: {cluster_signal}")
    print()
    print(f"{verdict_emoji} Verdict: {verdict}")

    _results['summary'] = {
        'sigma_Adam_abs': sigma_Adam_abs,
        'sigma_Adam_rel': sigma_Adam_rel,
        'mean_rel_error': mean_rel_error,
        'max_rel_error': max_rel_error,
        'min_rel_error': min_rel_error,
        'sorted_final_costs': sorted_costs,
        'pairwise_gaps': gaps,
        'max_gap': max_gap,
        'total_range': total_range,
        'gap_ratio': gap_ratio,
        'cluster_signal': cluster_signal,
        'verdict': verdict,
        'total_elapsed_h': elapsed / 3600,
        'completed': True,
    }
    _results['partial'] = False

    dump_state('final')
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
