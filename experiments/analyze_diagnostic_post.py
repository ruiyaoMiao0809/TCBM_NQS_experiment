"""
experiments/analyze_diagnostic_post.py
========================================
Post-process Adam diagnostic JSON to compute landscape ruggedness signals.

Computes 5 signals from the 5-seed Adam diagnostic:
    1. sigma_Adam_abs / sigma_Adam_rel    -- spread magnitude
    2. cluster_signal (gap_ratio)         -- multi-basin vs noise floor
    3. drift_signal (per-seed final - best_during)  -- Adam instability
    4. plateau_evidence (early[100:300] vs late[1500:1900])  -- basin escape
    5. combined verdict                   -- holistic 4×4 ruggedness call

Run after diagnostic completes:
    python experiments/analyze_diagnostic_post.py
    python experiments/analyze_diagnostic_post.py --input results/quick_adam_diagnostic.json
    python experiments/analyze_diagnostic_post.py --write-back   # update JSON summary

Default input: results/quick_adam_diagnostic_v2.json (post-RNG-fix data).
Use --input results/quick_adam_diagnostic.json for v1 (RNG-buggy, all seeds
byte-identical -- useful as control evidence).
"""
import argparse
import json
from pathlib import Path

import numpy as np

E_0_TRUTH = -8.4579
DEFAULT_INPUT = 'results/quick_adam_diagnostic_v2.json'

EARLY_WINDOW = (100, 300)    # plateau window
LATE_WINDOW = (1500, 1900)   # escape detection window


def compute_metrics(seeds_done):
    """Compute all 5 signals from a list of seed result dicts."""
    final_costs = [r['final_cost_mean'] for r in seeds_done]
    rel_errors = [r['rel_error'] for r in seeds_done]

    # Signal 1: spread magnitude
    sigma_abs = float(np.std(final_costs))
    sigma_rel = sigma_abs / abs(E_0_TRUTH)
    mean_rel = float(np.mean(rel_errors))
    max_rel = float(np.max(rel_errors))
    min_rel = float(np.min(rel_errors))

    # Signal 2: cluster (gap_ratio)
    sorted_costs = sorted(final_costs)
    gaps = [sorted_costs[i+1] - sorted_costs[i] for i in range(len(sorted_costs)-1)]
    max_gap = max(gaps) if gaps else 0.0
    total_range = sorted_costs[-1] - sorted_costs[0] if sorted_costs else 0.0
    gap_ratio = max_gap / total_range if total_range > 0 else 0.0

    if gap_ratio > 0.5:
        cluster_signal = "MULTI_BASIN (real barrier evidence)"
    elif gap_ratio > 0.35:
        cluster_signal = "BORDERLINE (weak barrier evidence)"
    else:
        cluster_signal = "NOISE_FLOOR (uniform spread, no barrier)"

    # Signal 3: drift (per-seed final - best_during)
    drifts = []
    bests = []
    for r in seeds_done:
        best = float(min(r['cost_trajectory']))
        drift = r['final_cost_mean'] - best
        drifts.append(drift)
        bests.append(best)
    mean_drift = float(np.mean(drifts))
    median_drift = float(np.median(drifts))
    max_drift = float(np.max(drifts))

    if mean_drift > 0.5:
        drift_signal = "LARGE_DRIFT (Adam unstable, multi-basin landscape)"
    elif mean_drift > 0.1:
        drift_signal = "MODERATE_DRIFT (some basin escape, may be rugged)"
    else:
        drift_signal = "SMALL_DRIFT (Adam converging to floor or single basin)"

    # Signal 4: plateau_evidence
    plateau_per_seed = []
    n_escapes = 0
    n_stays = 0
    for r in seeds_done:
        traj = r['cost_trajectory']
        if len(traj) < LATE_WINDOW[1]:
            plateau_per_seed.append({
                'seed': r['seed'],
                'note': f'trajectory too short ({len(traj)} < {LATE_WINDOW[1]})',
            })
            continue
        early = traj[EARLY_WINDOW[0]:EARLY_WINDOW[1]]
        late = traj[LATE_WINDOW[0]:LATE_WINDOW[1]]
        early_mean = float(np.mean(early))
        late_mean = float(np.mean(late))
        diff = late_mean - early_mean
        plateau_per_seed.append({
            'seed': r['seed'],
            'early_mean': early_mean,
            'late_mean': late_mean,
            'late_minus_early': diff,
        })
        if diff > 0.5:
            n_escapes += 1
        elif diff < -0.5:
            n_stays += 1   # actually improvement; counted as "stayed and improved"

    if n_escapes >= 3:
        plateau_signal = f"BASIN_ESCAPE ({n_escapes}/{len(seeds_done)} seeds left early plateau)"
    elif n_escapes >= 1:
        plateau_signal = f"PARTIAL_ESCAPE ({n_escapes}/{len(seeds_done)} seeds escaped)"
    else:
        plateau_signal = f"PLATEAU_STABLE (no seeds escaped early plateau)"

    # Signal 5: combined verdict
    n_strong_signals = (
        (1 if sigma_rel >= 0.01 else 0) +
        (1 if 'MULTI_BASIN' in cluster_signal else 0) +
        (1 if mean_drift > 0.5 else 0) +
        (1 if n_escapes >= 3 else 0)
    )
    n_weak_signals = (
        (1 if sigma_rel >= 0.005 else 0) +
        (1 if 'BORDERLINE' in cluster_signal else 0) +
        (1 if mean_drift > 0.1 else 0) +
        (1 if n_escapes >= 1 else 0)
    )

    if n_strong_signals >= 3:
        verdict = "STRONG_RUGGEDNESS (4×4 multi-basin, TCBM advantage plausible)"
        decision = "CONTINUE: commit to 4×4 + Plan A SR baseline (Day 5+)"
    elif n_strong_signals >= 2:
        verdict = "MODERATE_RUGGEDNESS (some multi-basin evidence)"
        decision = "CONTINUE with hedge: 4×4 OK, monitor TCBM σ closely Day 17-19"
    elif n_weak_signals >= 3:
        verdict = "WEAK_RUGGEDNESS (signals borderline)"
        decision = "PROCEED CAUTIOUSLY: 4×4 may not show strong TCBM advantage"
    else:
        verdict = "NOISE_FLOOR_ONLY (likely single basin)"
        decision = "PIVOT to 6×6 (4×4 too smooth for clean TCBM-vs-SR demonstration)"

    return {
        'final_costs': final_costs,
        'rel_errors': rel_errors,
        'best_during_per_seed': bests,
        'sorted_costs': sorted_costs,
        'pairwise_gaps': gaps,
        'sigma_Adam_abs': sigma_abs,
        'sigma_Adam_rel': sigma_rel,
        'mean_rel_error': mean_rel,
        'max_rel_error': max_rel,
        'min_rel_error': min_rel,
        'max_gap': max_gap,
        'total_range': total_range,
        'gap_ratio': gap_ratio,
        'cluster_signal': cluster_signal,
        'drifts_per_seed': drifts,
        'mean_drift': mean_drift,
        'median_drift': median_drift,
        'max_drift': max_drift,
        'drift_signal': drift_signal,
        'plateau_per_seed': plateau_per_seed,
        'n_plateau_escapes': n_escapes,
        'plateau_signal': plateau_signal,
        'n_strong_signals': n_strong_signals,
        'n_weak_signals': n_weak_signals,
        'combined_verdict': verdict,
        'decision': decision,
    }


def print_report(metrics, n_seeds):
    print("=" * 70)
    print(f"POST-PROCESSING ANALYSIS  ({n_seeds} seeds)")
    print("=" * 70)

    print("\n[Signal 1] Spread magnitude")
    print(f"  Final costs (sorted): {[f'{c:.4f}' for c in metrics['sorted_costs']]}")
    print(f"  Best-during per seed: {[f'{c:.4f}' for c in metrics['best_during_per_seed']]}")
    print(f"  Rel errors:           {[f'{r*100:.2f}%' for r in metrics['rel_errors']]}")
    print(f"  sigma_Adam (absolute) = {metrics['sigma_Adam_abs']:.4f}")
    print(f"  sigma_Adam / |E_0|    = {metrics['sigma_Adam_rel']*100:.3f}%")
    print(f"  Mean rel_error: {metrics['mean_rel_error']*100:.2f}%")

    print("\n[Signal 2] Cluster analysis (gap_ratio)")
    print(f"  Pairwise gaps: {[f'{g:.4f}' for g in metrics['pairwise_gaps']]}")
    print(f"  Max gap:       {metrics['max_gap']:.4f}")
    print(f"  Total range:   {metrics['total_range']:.4f}")
    print(f"  Gap ratio:     {metrics['gap_ratio']:.2f}")
    print(f"  Signal:        {metrics['cluster_signal']}")

    print("\n[Signal 3] Drift (Adam stability)")
    print(f"  Per-seed drift (final - best_during): "
          f"{[f'{d:+.3f}' for d in metrics['drifts_per_seed']]}")
    print(f"  Mean drift:   {metrics['mean_drift']:+.3f}")
    print(f"  Median drift: {metrics['median_drift']:+.3f}")
    print(f"  Max drift:    {metrics['max_drift']:+.3f}")
    print(f"  Signal:       {metrics['drift_signal']}")

    print(f"\n[Signal 4] Plateau evidence (early {EARLY_WINDOW} vs late {LATE_WINDOW})")
    for p in metrics['plateau_per_seed']:
        if 'note' in p:
            print(f"  Seed {p['seed']}: {p['note']}")
        else:
            print(f"  Seed {p['seed']}: early={p['early_mean']:.3f}, "
                  f"late={p['late_mean']:.3f}, diff={p['late_minus_early']:+.3f}")
    print(f"  Escapes: {metrics['n_plateau_escapes']}/{len(metrics['plateau_per_seed'])} "
          f"(diff > 0.5)")
    print(f"  Signal:  {metrics['plateau_signal']}")

    print("\n[Signal 5] Combined verdict")
    print(f"  Strong signals: {metrics['n_strong_signals']}/4")
    print(f"  Weak signals:   {metrics['n_weak_signals']}/4")
    print(f"  Verdict:  {metrics['combined_verdict']}")
    print(f"  Decision: {metrics['decision']}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=DEFAULT_INPUT,
                        help=f'JSON input path (default: {DEFAULT_INPUT})')
    parser.add_argument('--write-back', action='store_true',
                        help='Update JSON summary section with post-analysis fields')
    args = parser.parse_args()

    input_path = Path(args.input)
    with open(input_path) as f:
        data = json.load(f)

    seeds_done = data.get('seeds_done', [])
    print(f"Loaded {len(seeds_done)} seeds from {input_path}")
    print(f"  partial: {data.get('partial')}, dump_label: {data.get('dump_label')}")

    if len(seeds_done) < 2:
        print(f"\nNot enough seeds for analysis (need ≥2, got {len(seeds_done)}). Exiting.")
        return

    metrics = compute_metrics(seeds_done)
    print_report(metrics, len(seeds_done))

    if args.write_back:
        if 'summary' not in data:
            data['summary'] = {}
        data['summary']['post_analysis'] = {
            'sigma_Adam_abs': metrics['sigma_Adam_abs'],
            'sigma_Adam_rel': metrics['sigma_Adam_rel'],
            'gap_ratio': metrics['gap_ratio'],
            'cluster_signal': metrics['cluster_signal'],
            'mean_drift': metrics['mean_drift'],
            'drift_signal': metrics['drift_signal'],
            'n_plateau_escapes': metrics['n_plateau_escapes'],
            'plateau_signal': metrics['plateau_signal'],
            'n_strong_signals': metrics['n_strong_signals'],
            'n_weak_signals': metrics['n_weak_signals'],
            'combined_verdict': metrics['combined_verdict'],
            'decision': metrics['decision'],
        }
        with open(input_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"\nWrote post_analysis fields to {input_path}")


if __name__ == '__main__':
    main()
