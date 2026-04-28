"""
experiments/run_gradient_baseline.py
=====================================
Day 2 P0-1.2: TCBM-gradient baseline first end-to-end run on J1-J2 4x4.

Targets:
- P0-1.2 success: |E - E_0_truth| / |E_0_truth| < 10%
- R-abort-1 trigger: > 15%

E_0_truth = -8.4579 (from results/ed_reference_j1j2_4x4_J2=0.50.json, Day 1).

Note on n_vmc_samples=2000:
Bukov 2021 (SciPost Phys 10, 147; arXiv:2011.11214) uses NMC=2^15=32768 samples
per iteration (Section 6.1, Fig 12) for their N=6×6 experiments. Our n=2000 is
16x smaller in absolute terms, but our sampling ratio relative to accessible
S_z=0 Hilbert space (12870) is ~16%, comparable to or higher than Bukov's
relative coverage of their 6×6 Hilbert space.

If Day 2 baseline rel_error > 10% AND σ_E persistently > 0.05 throughout training
(noise-dominated optimization), Day 3 retry with n_vmc_samples=4000 is the first
single-variable change to try (vary-one-thing-at-a-time discipline).

Note on swap_acceptance trajectory:
trajectory['swap_acceptance'] is sampled at record_every=10 step intervals, but
swap calls happen at swap_interval=15. Due to LCM(10,15)=30, early records may
contain periodic stale repeats. std of trajectory underestimates true per-swap-call
variance by ~10-30%. mean is unaffected. For paper-level swap statistics,
recompute from raw per-swap-call data; for Day 2 verdict, mean trajectory is sufficient.
"""

import sys, json, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.j1j2_problem import J1J2Problem
from core.tcbm_optimizer_NQS import TCBMOptimizer, TCBMConfig

E_0_TRUTH = -8.4579   # from Day 1 ED reference
P0_THRESHOLD = 0.10    # 10% relative error target
R_ABORT_THRESHOLD = 0.15  # 15% triggers abort


def main():
    torch.manual_seed(42)

    problem = J1J2Problem(Lx=4, Ly=4, alpha=2, J1=1.0, J2=0.5, device='cuda:3')
    print(f"Problem: 4x4 J1-J2 PBC, J2/J1=0.5, D_params={problem.dim} (RBM α=2 real params)")

    cfg = TCBMConfig(
        M=12, n_steps=3000, k=20,
        T_min=0.005, T_max=2.0, T_min_floor=0.002,
        lambda_min=0.05, lambda_max=2.0, tau_lambda=500,
        subspace_warmup=150, subspace_update_freq=80,
        psi_star=1.5, min_warmup=120,
        grad_buffer_size=400, grad_buffer_use=200,
        subspace_source='gradient',
        n_vmc_samples=2000, n_vmc_samples_final=4,
        noise_temperature_alpha=1.0,
        seed=42,
    )
    print(f"\nConfig: M={cfg.M}, n_steps={cfg.n_steps}, k={cfg.k}")
    print(f"  subspace_warmup={cfg.subspace_warmup}, freq={cfg.subspace_update_freq}")
    print(f"  expected ~{(cfg.n_steps - cfg.subspace_warmup) // cfg.subspace_update_freq} SVD updates total")
    print(f"  psi_star={cfg.psi_star} (T_w trigger threshold)")
    print(f"\nTarget: |E - {E_0_TRUTH}| / {abs(E_0_TRUTH):.4f} < {P0_THRESHOLD*100:.0f}%")
    print(f"Abort:  > {R_ABORT_THRESHOLD*100:.0f}%")
    print(f"Expected duration: ~30 minutes on A10")
    print(f"{'='*70}\n")

    t0 = time.time()
    optimizer = TCBMOptimizer(problem, cfg)
    result = optimizer.optimize()
    elapsed = time.time() - t0

    # 关键指标
    rel_error_raw = abs(result['best_cost_raw'] - E_0_TRUTH) / abs(E_0_TRUTH)
    rel_error_debiased = abs(result['best_cost_debiased'] - E_0_TRUTH) / abs(E_0_TRUTH)

    # 判定
    if rel_error_debiased < P0_THRESHOLD:
        verdict = "PASS"
        verdict_emoji = "✅"
    elif rel_error_debiased < R_ABORT_THRESHOLD:
        verdict = "MARGINAL (10-15%, needs Day 3 retune)"
        verdict_emoji = "⚠️"
    else:
        verdict = "TRIGGER R-ABORT-1"
        verdict_emoji = "❌"

    # T_w 处理（None means SVD ran but psi never reached psi_star）
    T_w_step = result.get('T_w', None)
    if T_w_step is not None:
        T_w_str = f"step {T_w_step}"
    else:
        T_w_str = "never (psi never reached psi_star=1.5; algorithm stayed in pre-clamping phase throughout)"

    # swap_acceptance trajectory 分段统计
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

    # psi history
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
    print(f"\n{verdict_emoji} P0-1.2 Verdict: {verdict}")

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

    # NQS-1e debiasing sanity
    if result['best_cost_debiased'] < result['best_cost_raw']:
        print(f"  NQS-1e debiasing: working (debiased < raw)")
    else:
        print(f"  NQS-1e debiasing: ⚠ debiased >= raw, suspect bug")

    # Save
    assert cfg.subspace_source == 'gradient', f"unexpected subspace_source: {cfg.subspace_source}"
    output_path = Path('results/baseline_gradient_seed42.json')
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
        'T_w_step': T_w_step,  # Optional[int]
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
