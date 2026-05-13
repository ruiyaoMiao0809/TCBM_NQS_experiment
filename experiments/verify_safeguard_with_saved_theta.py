"""Day 7 Path γ Branch 1' safeguard verify (Option β, per-sample filter).

Load Variant B step 2000 — forensic showed 20.11% outliers, max|x|=14672.
Re-evaluate _local_energy_batch with new safeguard, confirm:
1. Unsafe sample count > 0 (mask is catching them)
2. E_safe filtered E_loc is finite + healthy magnitude (< 50 → physical, vs
   Day 6 best_cost_raw -8e8)
3. E_all unfiltered remains extreme (verifies the safeguard is meaningfully
   filtering, not a no-op)

Output: results/safeguard_verify.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.j1j2_problem import J1J2Problem  # noqa: E402

PT_PATH = 'results/sweep_v_b_day6_seed42_checkpoint_step2000_state.pt'
N_SAMPLES = 1000

data = torch.load(PT_PATH, weights_only=False, map_location='cpu')
theta_replicas = data['theta_replicas']      # (12, 1120)

problem = J1J2Problem(Lx=4, Ly=4, alpha=2, J1=1.0, J2=0.5, device='cpu')
problem.set_seed(42)

torch.manual_seed(99)
sigma = torch.randint(0, 2, (N_SAMPLES, 16), dtype=torch.float32) * 2 - 1

results = {'per_replica': []}

for r in range(12):
    theta_r = theta_replicas[r]
    problem.reset_unsafe_count()

    with torch.no_grad():
        E_loc, sample_safe = problem._local_energy_batch(theta_r, sigma)

    n_safe = int(sample_safe.sum().item())
    n_unsafe = N_SAMPLES - n_safe

    if n_safe > 0:
        E_mean_safe = float(E_loc.real[sample_safe].mean().item())
        E_loc_safe_max_abs = float(E_loc.real[sample_safe].abs().max().item())
    else:
        E_mean_safe = float('nan')
        E_loc_safe_max_abs = float('nan')

    E_mean_all = float(E_loc.real.mean().item())
    E_loc_all_max_abs = float(E_loc.real.abs().max().item())

    replica_result = {
        'replica': r,
        'n_safe': n_safe,
        'n_unsafe': n_unsafe,
        'pct_unsafe': 100.0 * n_unsafe / N_SAMPLES,
        'E_mean_safe_filtered': E_mean_safe,
        'E_mean_all_unfiltered': E_mean_all,
        'E_loc_max_abs_all': E_loc_all_max_abs,
        'E_loc_max_abs_safe': E_loc_safe_max_abs,
    }
    results['per_replica'].append(replica_result)

    print(
        f"Replica {r}: n_unsafe={n_unsafe:4d}/{N_SAMPLES} ({replica_result['pct_unsafe']:5.1f}%), "
        f"E_safe={E_mean_safe:.4f}, E_all={E_mean_all:.4e}, "
        f"max|E_all|={E_loc_all_max_abs:.4e}, max|E_safe|={E_loc_safe_max_abs:.4f}"
    )

# Aggregate
finite_safe = [r for r in results['per_replica'] if not np.isnan(r['E_mean_safe_filtered'])]
results['summary'] = {
    'pt_source': PT_PATH,
    'n_samples_per_replica': N_SAMPLES,
    'threshold': 30.0,
    'n_replicas_with_unsafe_samples': sum(1 for r in results['per_replica'] if r['n_unsafe'] > 0),
    'mean_pct_unsafe_per_replica': float(np.mean([r['pct_unsafe'] for r in results['per_replica']])),
    'E_safe_max_abs_across_replicas': (
        max(abs(r['E_mean_safe_filtered']) for r in finite_safe) if finite_safe else None
    ),
    'E_all_max_abs_across_replicas': max(abs(r['E_mean_all_unfiltered']) for r in results['per_replica']),
    'E_safe_all_physical_lt_50': all(
        abs(r['E_mean_safe_filtered']) < 50 for r in finite_safe
    ) if finite_safe else False,
    'forensic_reference': 'variant B step2000: pct_above_30 = 20.11%, max|log_psi_diff| = 14672',
    'comment': (
        'If pct_unsafe ~20% per replica matches forensic AND E_safe < 50, fix '
        'verified. addresses B/C blowup only; A/D stall is separate pathology.'
    ),
}

print("\n=== SUMMARY ===")
for k, v in results['summary'].items():
    print(f"  {k}: {v}")

with open('results/safeguard_verify.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved: results/safeguard_verify.json")
