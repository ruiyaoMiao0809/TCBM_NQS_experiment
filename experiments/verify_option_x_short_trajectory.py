"""Day 7 Path γ Branch 1' Option X verify.

Load Variant A step 500 saved theta_replicas (M=12, D=1120) — healthy state,
forensic showed max|log_psi_diff|≈11 < 15 threshold. Run 100 short steps using
TCBM-gradient optimizer with new threshold=15 fix. Check:
  1. Trajectory stability (no cost cliff > 5 in single step)
  2. unsafe_count stays controlled (healthy state should give very low rate)
  3. No catastrophic cost < -10

Output: results/option_x_verify.json + stdout summary

Run time estimate: 100 step × ~18 s/step = ~30 min on GPU.

Note: TCBMOptimizer.optimize_from(x0, n_steps) only injects x0 as replica 0.
We need all 12 replicas to start from saved state — using a generalized
monkey-patch on _init_replicas (same pattern as optimize_from).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.j1j2_problem import J1J2Problem  # noqa: E402
from core.tcbm_optimizer_NQS import TCBMOptimizer, TCBMConfig  # noqa: E402

DEVICE = os.environ.get('TCBM_DEVICE', 'cuda:0' if torch.cuda.is_available() else 'cpu')
PT_PATH = 'results/sweep_v_a_day6_seed42_checkpoint_step500_state.pt'

print(f"=== Loading {PT_PATH} (healthy starting state) ===")
data = torch.load(PT_PATH, weights_only=False, map_location='cpu')
starting_theta = data['theta_replicas']                     # (12, 1120)
print(f"  shape: {tuple(starting_theta.shape)}, dtype: {starting_theta.dtype}")
print(f"  device target: {DEVICE}")

torch.manual_seed(42)
problem = J1J2Problem(Lx=4, Ly=4, alpha=2, J1=1.0, J2=0.5, device=DEVICE)
problem.set_seed(42)
problem.reset_unsafe_count()

# cfg: identical Variant A (T_min=0.005, T_max=1.0), but n_steps=100
cfg = TCBMConfig(
    M=12, n_steps=100, k=20,
    T_min=0.005, T_max=1.0, T_min_floor=0.002,
    lambda_min=0.05, lambda_max=2.0, tau_lambda=500,
    subspace_warmup=150,  # >100, no SVD during verify
    subspace_update_freq=80,
    psi_star=1.5, min_warmup=120,
    grad_buffer_size=400, grad_buffer_use=200,
    subspace_source='gradient',
    n_vmc_samples=2000,
    n_vmc_samples_final=2,
    noise_temperature_alpha=1.0,
    seed=42,
)

optimizer = TCBMOptimizer(problem, cfg)
starting_theta = starting_theta.to(DEVICE)

# Monkey-patch _init_replicas to inject ALL 12 saved replicas
print("=== Patching _init_replicas to inject all 12 saved replicas ===")
orig_init = optimizer._init_replicas

def patched_init():
    positions, energies, stds = orig_init()
    # Replace all M replicas with saved theta
    M = positions.shape[0]
    assert starting_theta.shape[0] >= M, \
        f"saved theta has {starting_theta.shape[0]} replicas, need {M}"
    positions[:] = starting_theta[:M].to(optimizer.device, dtype=optimizer.dtype)
    # Re-evaluate energies and stds for the injected positions
    energies_new, stds_new = optimizer._eval_total(positions)
    energies[:] = energies_new
    stds[:] = stds_new
    return positions, energies, stds

optimizer._init_replicas = patched_init

trajectory = {
    'steps': [],
    'cost_min': [],
    'cost_mean': [],
    'swap_acc': [],
    'psi_max': [],
    'unsafe_count_cumulative': [],
    'unsafe_total_cumulative': [],
}


def trajectory_callback(step, info):
    trajectory['steps'].append(step)
    trajectory['cost_min'].append(float(info.get('cost_min', float('nan'))))
    trajectory['cost_mean'].append(float(info.get('cost_mean', float('nan'))))
    trajectory['swap_acc'].append(float(info.get('swap_acc_recent', float('nan'))))
    trajectory['psi_max'].append(float(info.get('psi_max', float('nan'))))
    trajectory['unsafe_count_cumulative'].append(int(problem._unsafe_sample_count))
    trajectory['unsafe_total_cumulative'].append(int(problem._unsafe_total_evaluated))

    if step % 10 == 0:
        unsafe_pct = (problem._unsafe_sample_count / max(1, problem._unsafe_total_evaluated)) * 100
        print(
            f"  step {step:3d}: cost_min={info.get('cost_min', float('nan')):.4f}, "
            f"cost_mean={info.get('cost_mean', float('nan')):.4f}, "
            f"swap_acc={info.get('swap_acc_recent', float('nan')):.3f}, "
            f"unsafe={problem._unsafe_sample_count}/{problem._unsafe_total_evaluated} ({unsafe_pct:.2f}%)",
            flush=True,
        )


print("=== Running 100 short steps with threshold=15 ===")
t_start = time.time()
result = optimizer.optimize(callback=trajectory_callback, callback_every=1)
t_wall = time.time() - t_start
print(f"\n=== Wall time: {t_wall:.1f}s ({t_wall/60:.1f} min) ===")

# Cliff detection
cliffs = []
for i in range(1, len(trajectory['cost_min'])):
    a = trajectory['cost_min'][i - 1]
    b = trajectory['cost_min'][i]
    if not np.isnan(a) and not np.isnan(b):
        delta = b - a
        if delta < -5.0:
            cliffs.append({'step': trajectory['steps'][i], 'delta': float(delta), 'before': a, 'after': b})

cost_arr = np.array([c for c in trajectory['cost_min'] if not np.isnan(c)])
final_unsafe = trajectory['unsafe_count_cumulative'][-1]
final_total = trajectory['unsafe_total_cumulative'][-1]
final_unsafe_pct = 100.0 * final_unsafe / max(1, final_total)

verdict_data = {
    'final_step': trajectory['steps'][-1] if trajectory['steps'] else None,
    'final_cost_min': trajectory['cost_min'][-1] if trajectory['cost_min'] else None,
    'min_cost_during_trajectory': float(cost_arr.min()) if len(cost_arr) else None,
    'max_cost_abs_during_trajectory': float(np.abs(cost_arr).max()) if len(cost_arr) else None,
    'final_unsafe_count': final_unsafe,
    'final_unsafe_total': final_total,
    'final_unsafe_pct': final_unsafe_pct,
    'cost_below_minus_10_at_any_step': bool((cost_arr < -10).any()) if len(cost_arr) else False,
    'cost_below_minus_100_at_any_step': bool((cost_arr < -100).any()) if len(cost_arr) else False,
    'cost_cliff_events_count': len(cliffs),
    'cost_cliff_events': cliffs[:10],
    'wall_time_seconds': t_wall,
    'best_cost_raw': float(result.get('best_cost_raw', float('nan'))),
    'best_cost_debiased': float(result.get('best_cost_debiased', float('nan'))),
}

result_dump = {
    'verdict_data': verdict_data,
    'cfg': {
        'n_steps': 100,
        'threshold': 15.0,
        'starting_state': PT_PATH,
        'M': 12,
        'T_min': 0.005,
        'T_max': 1.0,
    },
    'trajectory': trajectory,
}

with open('results/option_x_verify.json', 'w') as f:
    json.dump(result_dump, f, indent=2, default=str)

print("\n=== Verdict data (no interpretation, Nick reads) ===")
for k, v in verdict_data.items():
    if k == 'cost_cliff_events':
        print(f"  {k}: {len(cliffs)} events" + (f", first 3: {cliffs[:3]}" if cliffs else ""))
    else:
        print(f"  {k}: {v}")

print(f"\nOutput: results/option_x_verify.json")
