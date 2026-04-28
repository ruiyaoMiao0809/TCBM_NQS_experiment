"""
experiments/run_gradient_baseline_smaller.py
============================================
Day 3 fallback: smaller cfg for faster iteration if Day 2 baseline times out
or fails to converge.

Reduced from default (run_gradient_baseline.py):
- n_steps: 3000 → 1000  (sufficient to see convergence trend)
- n_vmc_samples: 2000 → 1000  (faster per step, larger sigma_E acceptable for diagnostic)
- subspace_warmup: 150 → 50  (proportionally scaled to n_steps/3)
- min_warmup: 120 → 40
- grad_buffer_size: 400 → 200; grad_buffer_use: 200 → 100
- subspace_update_freq: 80 → 40

Expected wall clock: ~30-60 min (3-6× faster than full run, depending on
whether the bottleneck is per-step VMC cost or step count).

Verdict thresholds RELAXED for diagnostic use (smaller n_steps cannot reach
10% in 1000 steps even on a healthy run):
- P0 diagnostic: 20% (vs 10% production)
- R-abort diagnostic: 30% (vs 15% production)

Use cases:
- Day 3 if Day 2 R-ABORT-1 → diagnose root cause cheaply
- Day 3 retune sweep (vary one cfg param at a time) → 3 retunes × 30-60 min

NOT for paper-quality results. Production verdict comes from full
run_gradient_baseline.py with un-relaxed thresholds.
"""

import sys, json, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.j1j2_problem import J1J2Problem
from core.tcbm_optimizer_NQS import TCBMOptimizer, TCBMConfig

E_0_TRUTH = -8.4579
P0_THRESHOLD_DIAGNOSTIC = 0.20    # 20% for smaller diagnostic run
R_ABORT_THRESHOLD_DIAGNOSTIC = 0.30  # 30% for smaller diagnostic run


def main():
    torch.manual_seed(42)

    problem = J1J2Problem(Lx=4, Ly=4, alpha=2, J1=1.0, J2=0.5, device='cuda:3')
    print(f"Problem: 4x4 J1-J2 PBC, J2/J1=0.5, D_params={problem.dim} (RBM α=2 real params)")

    cfg = TCBMConfig(
        M=12, n_steps=1000, k=20,
        T_min=0.005, T_max=2.0, T_min_floor=0.002,
        lambda_min=0.05, lambda_max=2.0, tau_lambda=500,
        subspace_warmup=50, subspace_update_freq=40,
        psi_star=1.5, min_warmup=40,
        grad_buffer_size=200, grad_buffer_use=100,
        subspace_source='gradient',
        n_vmc_samples=1000, n_vmc_samples_final=4,
        noise_temperature_alpha=1.0,
        seed=42,
    )
    print(f"\n[DIAGNOSTIC RUN — relaxed thresholds, do not use for paper claims]")
    print(f"Config: M={cfg.M}, n_steps={cfg.n_steps} (1/3 of production), k={cfg.k}")
    print(f"  n_vmc_samples={cfg.n_vmc_samples} (1/2 of production)")
    print(f"  subspace_warmup={cfg.subspace_warmup}, freq={cfg.subspace_update_freq}")
    print(f"  expected ~{(cfg.n_steps - cfg.subspace_warmup) // cfg.subspace_update_freq} SVD updates total")
    print(f"  psi_star={cfg.psi_star} (T_w trigger threshold)")
    print(f"\nDiagnostic target: |E - {E_0_TRUTH}| / {abs(E_0_TRUTH):.4f} < {P0_THRESHOLD_DIAGNOSTIC*100:.0f}%")
    print(f"Diagnostic abort:  > {R_ABORT_THRESHOLD_DIAGNOSTIC*100:.0f}%")
    print(f"Expected duration: ~30-60 minutes on A10")
    print(f"{'='*70}\n")

    t0 = time.time()
    optimizer = TCBMOptimizer(problem, cfg)
    result = optimizer.optimize()
    elapsed = time.time() - t0

    rel_error_raw = abs(result['best_cost_raw'] - E_0_TRUTH) / abs(E_0_TRUTH)
    rel_error_debiased = abs(result['best_cost_debiased'] - E_0_TRUTH) / abs(E_0_TRUTH)

    if rel_error_debiased < P0_THRESHOLD_DIAGNOSTIC:
        verdict = "DIAGNOSTIC PASS"
        verdict_emoji = "✅"
    elif rel_error_debiased < R_ABORT_THRESHOLD_DIAGNOSTIC:
        verdict = "DIAGNOSTIC MARGINAL (20-30%, retune cfg)"
        verdict_emoji = "⚠️"
    else:
        verdict = "DIAGNOSTIC FAIL (≥30%, mechanism issue suspected)"
        verdict_emoji = "❌"

    T_w_step = result.get('T_w', None)
    if T_w_step is not None:
        T_w_str = f"step {T_w_step}"
    else:
        T_w_str = "never (psi never reached psi_star=1.5; algorithm stayed in pre-clamping phase throughout)"

    swap_acc_traj = result.get('trajectory', {}).get('swap_acceptance', [])
    if swap_acc_traj and len(swap_acc_traj) >= 30:
        n_traj = len(swap_acc_traj)
        third = n_traj // 3
        swap_early = float(np.mean(swap_acc_traj[:third]))
        swap_mid = float(np.mean(swap_acc_traj[third:2*third]))
        swap_late = float(np.mean(swap_acc_traj[2*third:]))
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
    print(f"Run completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*70}")
    print(f"best_cost_raw       = {result['best_cost_raw']:.4f}")
    print(f"best_cost_debiased  = {result['best_cost_debiased']:.4f}")
    print(f"E_0 (ED truth)      = {E_0_TRUTH:.4f}")
    print(f"Relative error (raw)      = {rel_error_raw*100:.2f}%")
    print(f"Relative error (debiased) = {rel_error_debiased*100:.2f}%")
    print(f"\n{verdict_emoji} Diagnostic Verdict: {verdict}")
    print(f"\n[Reminder: relaxed thresholds; NOT paper-quality result]")

    print(f"\nDiagnostics:")
    print(f"  swap_acceptance overall mean = {swap_overall:.3f}  (target [0.15, 0.35])")
    print(f"  swap_acceptance early/mid/late = {swap_early:.3f} / {swap_mid:.3f} / {swap_late:.3f}")
    print(f"  swap_acceptance std (per record) = {swap_std:.3f}  (note: stale repeats underestimate)")
    print(f"  total swap events = {result.get('n_accepted_swaps', 'N/A')}")
    print(f"  final sigma_E mean = {result['final_sigma_E_mean']:.4f}")
    print(f"  final sigma_E max  = {result['final_sigma_E_max']:.4f}")
    print(f"  T_w triggered: {T_w_str}")
    print(f"  psi_history len = {len(psi_history)} (expected ~{(cfg.n_steps - cfg.subspace_warmup) // cfg.subspace_update_freq})")
    print(f"  psi_max cumulative = {psi_max:.4f}")

    if result['best_cost_debiased'] < result['best_cost_raw']:
        print(f"  NQS-1e debiasing: working (debiased < raw)")
    else:
        print(f"  NQS-1e debiasing: ⚠ debiased >= raw, suspect bug")

    assert cfg.subspace_source == 'gradient', f"unexpected subspace_source: {cfg.subspace_source}"
    output_path = Path('results/baseline_gradient_smaller_seed42.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        'config': {k: v for k, v in cfg.__dict__.items() if not k.startswith('_')},
        'best_cost_raw': float(result['best_cost_raw']),
        'best_cost_debiased': float(result['best_cost_debiased']),
        'final_sigma_E_mean': float(result['final_sigma_E_mean']),
        'final_sigma_E_max': float(result['final_sigma_E_max']),
        'rel_error_raw': float(rel_error_raw),
        'rel_error_debiased': float(rel_error_debiased),
        'elapsed_seconds': elapsed,
        'E_0_truth': E_0_TRUTH,
        'verdict': verdict,
        'is_diagnostic_run': True,
        'thresholds': {
            'P0_diagnostic': P0_THRESHOLD_DIAGNOSTIC,
            'R_abort_diagnostic': R_ABORT_THRESHOLD_DIAGNOSTIC,
        },
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
    }
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved: {output_path}")


if __name__ == '__main__':
    main()
