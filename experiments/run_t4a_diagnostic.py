"""
experiments/run_t4a_diagnostic.py
==================================
T4a Early-Abort Diagnostic (Day 7 AM, Protocol v2.1 §P0-1.4).

Runs a 1000-step warmup with ψ* = ∞ (T_w never triggers), recording ψ(t)
evolution. Judges GREEN / YELLOW / RED based on cumulative max ψ.

Usage:
    python experiments/run_t4a_diagnostic.py                   # default config
    python experiments/run_t4a_diagnostic.py --retry_level 1   # first retry
    python experiments/run_t4a_diagnostic.py --retry_level 2   # second retry

Retry levels (see Protocol v2.1 §1.2 P0-1.4):
    0: default (grad_buf 400, warmup 150, k 20)
    1: enlarged buffer (grad_buf 600, warmup 300)
    2: reduced k (k 12)

Output:
    results/week1_t4a_diagnostic_retry{level}.json
    results/figures/t4a_psi_evolution_retry{level}.png

Judgment:
    GREEN: max ψ ≥ 1.5 by step 1000   → continue Week 2 plan
    YELLOW: max ψ ∈ [1.1, 1.5)        → try next retry level
    RED: max ψ < 1.1                  → trigger R-abort-4, switch Plan B

GPU time: ~15 minutes on A10.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Project path setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.tcbm_optimizer_NQS import TCBMOptimizer, TCBMConfig
from core.j1j2_problem import J1J2Problem


RETRY_CONFIGS = {
    0: dict(grad_buffer_size=400, grad_buffer_use=200, subspace_warmup=150, k=20,
            name='default'),
    1: dict(grad_buffer_size=600, grad_buffer_use=300, subspace_warmup=300, k=20,
            name='enlarged_buffer'),
    2: dict(grad_buffer_size=600, grad_buffer_use=300, subspace_warmup=300, k=12,
            name='reduced_k'),
}


def run_diagnostic(retry_level: int = 0, seed: int = 42, device: str = 'cuda',
                   output_dir: str = 'results'):
    """Run a single T4a diagnostic."""
    if retry_level not in RETRY_CONFIGS:
        raise ValueError(f"retry_level must be in {list(RETRY_CONFIGS.keys())}")

    retry_cfg = RETRY_CONFIGS[retry_level]
    print(f"\n{'='*70}")
    print(f"T4a Early-Abort Diagnostic — retry_level={retry_level} ({retry_cfg['name']})")
    print(f"Seed: {seed}  |  Device: {device}")
    print(f"Config: grad_buf={retry_cfg['grad_buffer_size']}, "
          f"warmup={retry_cfg['subspace_warmup']}, k={retry_cfg['k']}")
    print(f"{'='*70}\n")

    # Build problem
    problem = J1J2Problem(Lx=4, Ly=4, J1=1.0, J2=0.5, alpha=2, device=device)
    print(f"Problem: J1-J2 4x4, J2/J1=0.5, D={problem.dim}")

    # Build config (key: psi_star=inf, min_warmup=huge, so T_w never triggers)
    cfg = TCBMConfig(
        M=12,
        n_steps=1000,
        k=retry_cfg['k'],
        T_min=0.005, T_max=2.0, T_min_floor=0.002,
        lambda_min=0.05, lambda_max=2.0, tau_lambda=500,
        subspace_warmup=retry_cfg['subspace_warmup'],
        subspace_update_freq=50,
        psi_star=float('inf'),         # NEVER trigger T_w
        min_warmup=10**9,               # NEVER trigger T_w
        grad_buffer_size=retry_cfg['grad_buffer_size'],
        grad_buffer_use=retry_cfg['grad_buffer_use'],
        subspace_source='gradient',
        n_vmc_samples=1500,
        n_vmc_samples_final=4,
        noise_temperature_alpha=1.0,
        record_every=10,
        seed=seed,
    )

    # Run
    t0 = datetime.now()
    opt = TCBMOptimizer(problem, cfg)
    result = opt.optimize()
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\nOptimization complete in {elapsed:.1f}s")

    # Analyze ψ history
    psi_history = result['psi_history']    # list of (step, psi) tuples
    if not psi_history:
        print("⚠️ WARNING: No psi updates recorded. SVD never triggered.")
        print("   Check that subspace_warmup < 1000.")
        return dict(judgment='UNKNOWN', max_psi=0.0, retry_level=retry_level)

    steps = [s for s, p in psi_history]
    psis  = [p for s, p in psi_history]
    psi_max_cumulative = np.maximum.accumulate(psis)
    max_psi_final = float(psi_max_cumulative[-1])

    # Judgment
    if max_psi_final >= 1.5:
        judgment = 'GREEN'
        emoji = '✅'
        action = 'Continue Week 2 plan'
    elif max_psi_final >= 1.1:
        judgment = 'YELLOW'
        emoji = '⚠️'
        if retry_level < 2:
            action = f'Try retry_level={retry_level + 1}'
        else:
            action = 'All retry levels exhausted → RED (R-abort-4)'
    else:
        judgment = 'RED'
        emoji = '❌'
        action = 'Trigger R-abort-4, switch to Plan B'

    # Report
    print(f"\n{'─'*70}")
    print(f"  T4a Diagnostic Result")
    print(f"{'─'*70}")
    print(f"  Cumulative max ψ:     {max_psi_final:.3f}")
    print(f"  ψ final value:        {psis[-1]:.3f}")
    print(f"  N SVD updates:        {len(psi_history)}")
    print(f"  Steps with ψ ≥ 1.5:   {sum(p >= 1.5 for p in psi_max_cumulative)}")
    print(f"  Steps with ψ ≥ 1.1:   {sum(p >= 1.1 for p in psi_max_cumulative)}")
    print(f"\n  {emoji} Judgment: {judgment}")
    print(f"  → Action: {action}")
    print(f"{'─'*70}\n")

    # Save JSON
    out_path = Path(output_dir) / f'week1_t4a_diagnostic_retry{retry_level}.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_data = {
        'retry_level': retry_level,
        'retry_config_name': retry_cfg['name'],
        'retry_config': {k: v for k, v in retry_cfg.items() if k != 'name'},
        'seed': seed,
        'device': device,
        'elapsed_seconds': elapsed,
        'psi_history_steps': steps,
        'psi_history_values': psis,
        'psi_cum_max': psi_max_cumulative.tolist(),
        'max_psi_final': max_psi_final,
        'judgment': judgment,
        'best_cost_raw': result['best_cost_raw'],
        'best_cost_debiased': result['best_cost_debiased'],
        'final_sigma_E_mean': result['final_sigma_E_mean'],
        'timestamp': t0.isoformat(),
    }
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved JSON: {out_path}")

    # Plot
    fig_dir = Path(output_dir) / 'figures'
    fig_dir.mkdir(exist_ok=True)
    fig_path = fig_dir / f't4a_psi_evolution_retry{retry_level}.png'

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(steps, psis, 'o-', alpha=0.5, label=r'$\psi(t) = \sigma_k/\sigma_{k+1}$ (per update)')
    ax.plot(steps, psi_max_cumulative, 'r-', linewidth=2, label=r'$\max_{\tau \leq t} \psi(\tau)$')
    ax.axhline(1.5, color='green', linestyle='--', alpha=0.7, label='GREEN threshold (1.5)')
    ax.axhline(1.1, color='orange', linestyle='--', alpha=0.7, label='YELLOW/RED boundary (1.1)')
    ax.set_xlabel('Optimization step')
    ax.set_ylabel(r'$\psi(t) = \sigma_k / \sigma_{k+1}$')
    title_suffix = f" — {judgment}"
    ax.set_title(f'T4a Diagnostic (retry_level={retry_level}, {retry_cfg["name"]}){title_suffix}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"Saved plot: {fig_path}")

    return dict(judgment=judgment, max_psi=max_psi_final, retry_level=retry_level,
                output_path=str(out_path))


def main():
    parser = argparse.ArgumentParser(description='T4a Early-Abort Diagnostic (Protocol v2.1)')
    parser.add_argument('--retry_level', type=int, default=0, choices=[0, 1, 2],
                        help='Retry configuration level (0=default, 1=enlarged buf, 2=reduced k)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='cuda or cpu')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory')
    args = parser.parse_args()

    summary = run_diagnostic(
        retry_level=args.retry_level,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
    )

    # Exit code reflects judgment (for shell-level automation)
    if summary['judgment'] == 'GREEN':
        print("\n✅ T4a PASS. Ready for Week 2.")
        sys.exit(0)
    elif summary['judgment'] == 'YELLOW':
        print(f"\n⚠️ T4a YELLOW. Consider retry_level={args.retry_level + 1}.")
        sys.exit(1)
    elif summary['judgment'] == 'RED':
        print("\n❌ T4a FAIL. Trigger R-abort-4, prepare Plan B.")
        sys.exit(2)
    else:
        print("\n? T4a UNKNOWN.")
        sys.exit(3)


if __name__ == '__main__':
    main()
