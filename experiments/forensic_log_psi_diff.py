"""
Day 7 Path γ Branch 1 Forensic Analysis.

Load 4 sweep variants × 6 checkpoints = 24 theta snapshots.
For each snapshot, forward-compute log_psi(σ) and log_psi(σ_flipped) on a
sample of spin configurations to derive log_psi_diff distribution.

Goal: verify Hypothesis A (1/ψ underflow) — if |log_psi_diff.real| > 30
outliers exist in any variant's late checkpoints (post step 800 anomaly),
1/ψ underflow is the dominant numerical pathology and Phase 2.5 fix should
target _local_energy_batch ratio safeguard.

Output: results/forensic_log_psi_diff.json + stdout summary
"""

import glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.j1j2_problem import J1J2Problem, log_psi_rbm  # noqa: E402

DEVICE = 'cuda:0'
N_PROBE_SAMPLES = 1000
N_SPINS = 16        # 4x4 lattice
M_HIDDEN = 32       # alpha * N = 2 * 16
THRESHOLDS = [5, 15, 30, 50]
FIX_RELEVANT = 30

# Instantiate J1J2Problem so we can confirm dims (use module-level log_psi_rbm)
problem = J1J2Problem(Lx=4, Ly=4, alpha=2, J1=1.0, J2=0.5, device=DEVICE)
assert problem.N == N_SPINS and problem.M == M_HIDDEN, \
    f"problem dims mismatch: N={problem.N}, M={problem.M}"

variants = ['a', 'b', 'c', 'd']

results = {
    'meta': {
        'n_probe_samples_per_replica': N_PROBE_SAMPLES,
        'n_replicas': 12,
        'problem': '4x4 J1-J2, J1=1, J2=0.5, alpha=2',
        'thresholds_examined': THRESHOLDS,
        'fix_relevant_threshold': FIX_RELEVANT,
        'description': (
            'log_psi_diff.real distribution across 4 sweep variants × '
            'multiple checkpoints; log_psi_diff = log_psi(σ_flipped) - log_psi(σ)'
        ),
        'method': 'log_psi_rbm(theta, sigma, N=16, M=32), flip 1 random site per sample',
        'device': DEVICE,
    },
    'variants': {},
}


def ckpt_sort_key(label):
    """Order: step500 → step1000 → ... → step2500 → atexit."""
    if label == 'atexit':
        return (1, 0)
    m = re.search(r'step(\d+)', label)
    return (0, int(m.group(1))) if m else (2, 0)


t_total_start = time.time()

for v in variants:
    results['variants'][v] = {}
    variant_files = sorted(glob.glob(f'results/sweep_v_{v}_day6_seed42*_state.pt'))
    print(f"\n=== Variant {v}: {len(variant_files)} files ===", flush=True)

    for fpath in variant_files:
        fname = Path(fpath).stem
        if 'atexit' in fname:
            ckpt_label = 'atexit'
        else:
            m = re.search(r'step(\d+)', fname)
            ckpt_label = f'step{m.group(1)}' if m else 'unknown'

        try:
            t0 = time.time()
            data = torch.load(fpath, weights_only=False, map_location=DEVICE)
            theta_replicas = data.get('theta_replicas')

            if theta_replicas is None:
                print(f"  {ckpt_label}: NO theta_replicas, skipping", flush=True)
                continue

            theta_replicas = theta_replicas.to(DEVICE)
            M_replicas, D = theta_replicas.shape
            assert D == 1120, f"unexpected D={D}"

            all_log_psi_diff_real = []
            per_replica = []

            for r in range(M_replicas):
                theta_r = theta_replicas[r]  # (1120,)

                # Deterministic sample for reproducibility across re-runs
                seed = (42 + r + abs(hash(f'{v}_{ckpt_label}')) % 10000)
                gen = torch.Generator(device=DEVICE).manual_seed(seed)

                sigma_int = torch.randint(
                    0, 2, (N_PROBE_SAMPLES, N_SPINS),
                    dtype=torch.int8, device=DEVICE, generator=gen,
                )
                sigma = (sigma_int.to(torch.float32) * 2.0 - 1.0)  # ±1 spins

                flip_site = torch.randint(
                    0, N_SPINS, (N_PROBE_SAMPLES,),
                    device=DEVICE, generator=gen,
                )
                sigma_flipped = sigma.clone()
                arange_idx = torch.arange(N_PROBE_SAMPLES, device=DEVICE)
                sigma_flipped[arange_idx, flip_site] *= -1.0

                with torch.no_grad():
                    log_psi_cur = log_psi_rbm(theta_r, sigma, N_SPINS, M_HIDDEN)
                    log_psi_flipped = log_psi_rbm(theta_r, sigma_flipped, N_SPINS, M_HIDDEN)
                    log_psi_diff = log_psi_flipped - log_psi_cur

                log_psi_diff_real = log_psi_diff.real.cpu().numpy().flatten()
                all_log_psi_diff_real.extend(log_psi_diff_real.tolist())

                abs_real = np.abs(log_psi_diff_real)
                replica_stats = {
                    'replica': r,
                    'mean_real': float(log_psi_diff_real.mean()),
                    'std_real': float(log_psi_diff_real.std()),
                    'max_abs_real': float(abs_real.max()),
                }
                for t in THRESHOLDS:
                    replica_stats[f'count_above_{t}'] = int((abs_real > t).sum())
                per_replica.append(replica_stats)

            all_arr = np.array(all_log_psi_diff_real)
            abs_arr = np.abs(all_arr)
            overall = {
                'total_samples': len(all_arr),
                'mean_real': float(all_arr.mean()),
                'std_real': float(all_arr.std()),
                'max_abs_real': float(abs_arr.max()),
            }
            for t in THRESHOLDS:
                overall[f'count_above_{t}'] = int((abs_arr > t).sum())
            overall['pct_above_30'] = float(overall['count_above_30'] / len(all_arr) * 100)

            elapsed = time.time() - t0
            print(
                f"  {ckpt_label}: mean={all_arr.mean():.3f}, max_abs={abs_arr.max():.3f}, "
                f"count>30={overall['count_above_30']} ({overall['pct_above_30']:.2f}%), "
                f"wall={elapsed:.1f}s",
                flush=True,
            )

            results['variants'][v][ckpt_label] = {
                'per_replica': per_replica,
                'overall_stats': overall,
            }

        except Exception as e:
            print(f"  {ckpt_label}: ERROR {type(e).__name__}: {e}", flush=True)
            results['variants'][v][ckpt_label] = {'error': f'{type(e).__name__}: {str(e)}'}

t_total = time.time() - t_total_start
results['meta']['total_wall_seconds'] = t_total

output_path = 'results/forensic_log_psi_diff.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n=== Forensic analysis complete. Wall {t_total:.1f}s. Output: {output_path} ===\n")
print("=" * 80)
print("SUMMARY: count of |log_psi_diff.real| > 30 per variant/checkpoint")
print("=" * 80)
print(f"{'variant':<10} {'ckpt':<12} {'count>30':<12} {'pct>30':<10} {'max|x|':<14} {'mean':<14}")
print("-" * 80)
for v in variants:
    ckpts = sorted(results['variants'][v].keys(), key=ckpt_sort_key)
    for ckpt in ckpts:
        entry = results['variants'][v][ckpt]
        if 'error' in entry:
            print(f"{v:<10} {ckpt:<12} ERROR: {entry['error']}")
        else:
            s = entry['overall_stats']
            print(
                f"{v:<10} {ckpt:<12} {s['count_above_30']:<12} "
                f"{s['pct_above_30']:<10.2f} {s['max_abs_real']:<14.3f} {s['mean_real']:<14.4f}"
            )
print("=" * 80)
