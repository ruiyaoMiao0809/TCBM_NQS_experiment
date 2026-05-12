"""
Quick analysis of Day 6 probe-run trajectory (Phase 2.5 Step 6.1).

Inputs:
  - results/baseline_v2_probe_seed42.json (Day 6 probe-run, 1100 steps, record_every=10)
  - results/baseline_v2_seed42.json (Day 5 production v2, 3000 steps, record_every=10)
  - logs/probe_day6_seed42_20260511_0736.log (per-100-step printouts including psi_max)

Outputs:
  - stdout: 5 metric tables + hypothesis-C verdict
  - results/probe_analysis_hypothesis_c.json (machine-readable summary)

Purpose: probe-run #1 ran without Path κ1 fix → no theta tensor available.
Indirect hypothesis-C verification via:
  - swap_acc trajectory (cliff, % outside target [0.15, 0.35])
  - cost_min vs cost_mean separation (outlier-replica signature)
  - sigma_E trajectory (VMC mixing health)
  - psi_max trajectory (parsed from log; track normalization breach)
  - bit-exact Day-5 reproduction (sanity)
"""

import json
import re
from pathlib import Path

import numpy as np


def load_jsons():
    probe = json.load(open('results/baseline_v2_probe_seed42.json'))
    day5 = json.load(open('results/baseline_v2_seed42.json'))
    return probe, day5


def parse_log_psi_max(log_path):
    """Extract per-100-step psi_max from log: '[step  N00] ... psi_max=X.XXX ...'"""
    psi_traj = []
    cost_min_traj = []
    swap_acc_traj = []
    pat = re.compile(
        r'\[step\s+(\d+)\]\s+cost_min=(-?\d+\.\d+).*?'
        r'swap_acc=(\d+\.\d+)\s+psi_max=(\d+\.\d+)'
    )
    with open(log_path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                step = int(m.group(1))
                cost_min_traj.append((step, float(m.group(2))))
                swap_acc_traj.append((step, float(m.group(3))))
                psi_traj.append((step, float(m.group(4))))
    return psi_traj, cost_min_traj, swap_acc_traj


def metric_1_swap_acc(probe, day5, swap_log_traj):
    """Swap acceptance cliff + % outside target zone."""
    p_traj = np.array(probe['swap_acceptance_trajectory_full'])  # record_every=10
    d_traj = np.array(day5['swap_acceptance_trajectory_full'])

    target_lo, target_hi = 0.15, 0.35
    in_target_probe = ((p_traj >= target_lo) & (p_traj <= target_hi)).sum()
    in_target_day5 = ((d_traj >= target_lo) & (d_traj <= target_hi)).sum()

    above_05_probe = (p_traj > 0.5).sum()
    above_05_day5 = (d_traj > 0.5).sum()

    diffs = np.abs(np.diff(p_traj))
    max_cliff_idx = int(np.argmax(diffs))
    max_cliff_step = (max_cliff_idx + 1) * 10
    max_cliff_value = float(diffs[max_cliff_idx])

    log_swap_arr = np.array([v for _, v in swap_log_traj])
    log_steps = np.array([s for s, _ in swap_log_traj])
    log_diffs = np.abs(np.diff(log_swap_arr))
    log_cliff_idx = int(np.argmax(log_diffs)) if len(log_diffs) else 0
    log_cliff_step = int(log_steps[log_cliff_idx + 1]) if len(log_diffs) else 0
    log_cliff_from = float(log_swap_arr[log_cliff_idx]) if len(log_diffs) else 0
    log_cliff_to = float(log_swap_arr[log_cliff_idx + 1]) if len(log_diffs) else 0

    return {
        'probe_n_in_target_zone': int(in_target_probe),
        'probe_total': int(len(p_traj)),
        'probe_pct_in_target_zone': float(in_target_probe / len(p_traj)),
        'day5_n_in_target_zone': int(in_target_day5),
        'day5_total': int(len(d_traj)),
        'day5_pct_in_target_zone': float(in_target_day5 / len(d_traj)),
        'probe_n_above_0.5': int(above_05_probe),
        'day5_n_above_0.5': int(above_05_day5),
        'probe_max_cliff_step': int(max_cliff_step),
        'probe_max_cliff_delta': max_cliff_value,
        'log_per100_max_cliff_step': log_cliff_step,
        'log_per100_max_cliff_from': log_cliff_from,
        'log_per100_max_cliff_to': log_cliff_to,
    }


def metric_2_cost_separation(probe, day5, cost_log_traj):
    """cost_min vs cost_mean separation per callback (every 100 steps).
    cost_min comes from log parsing; cost_mean from JSON progress_history."""
    cost_mean = probe['progress_history']['cost_mean_per_callback']
    log_steps = [s for s, _ in cost_log_traj]
    log_mins = [v for _, v in cost_log_traj]
    n = min(len(cost_mean), len(log_mins))
    seps = []
    for i in range(n):
        sep = log_mins[i] - cost_mean[i]
        seps.append((log_steps[i], log_mins[i], cost_mean[i], sep))
    max_sep_entry = min(seps, key=lambda t: t[3])  # most negative separation
    return {
        'n_callbacks': n,
        'max_separation_step': max_sep_entry[0],
        'max_separation_cost_min': max_sep_entry[1],
        'max_separation_cost_mean': max_sep_entry[2],
        'max_separation_delta': max_sep_entry[3],
        'cost_min_step_700': next((v for s, v in cost_log_traj if s == 700), None),
        'cost_min_step_800': next((v for s, v in cost_log_traj if s == 800), None),
    }


def metric_3_sigma_e(probe, day5):
    """sigma_E_max trajectory mixing indicator."""
    p_sig = np.array(probe['sigma_trajectory_full'])
    d_sig = np.array(day5['sigma_trajectory_full'])
    return {
        'probe_sigma_E_max_overall': float(p_sig.max()),
        'probe_sigma_E_mean': float(p_sig.mean()),
        'probe_sigma_E_std': float(p_sig.std()),
        'day5_sigma_E_max_overall_first_1100steps': float(d_sig[:110].max()),
        'day5_sigma_E_mean_first_1100steps': float(d_sig[:110].mean()),
    }


def metric_4_psi_trajectory(psi_log_traj):
    """psi_max evolution (from log per-100-step prints)."""
    psi_arr = np.array([v for _, v in psi_log_traj])
    step_arr = np.array([s for s, _ in psi_log_traj])
    above_1_first_step = None
    for s, v in psi_log_traj:
        if v > 1.0:
            above_1_first_step = s
            break
    return {
        'psi_max_cumulative': float(psi_arr.max()),
        'psi_min_in_history': float(psi_arr.min()),
        'first_step_above_1.0': above_1_first_step,
        'pct_steps_above_1.0': float((psi_arr > 1.0).sum() / len(psi_arr)),
        'ever_reached_psi_star_1.5': bool((psi_arr >= 1.5).any()),
        'final_psi_max_value': float(psi_arr[-1]),
    }


def metric_5_bit_exact_check(probe, day5):
    """First 110 entries should be byte-identical (probe = first 1100 steps of Day 5 path)."""
    p_cost = np.array(probe['cost_trajectory_full'])
    d_cost = np.array(day5['cost_trajectory_full'][:110])
    p_swap = np.array(probe['swap_acceptance_trajectory_full'])
    d_swap = np.array(day5['swap_acceptance_trajectory_full'][:110])

    cost_max_diff = float(np.abs(p_cost - d_cost).max())
    swap_max_diff = float(np.abs(p_swap - d_swap).max())

    return {
        'cost_trajectory_first_1100_max_abs_diff': cost_max_diff,
        'swap_trajectory_first_1100_max_abs_diff': swap_max_diff,
        'bit_exact_match': cost_max_diff == 0.0 and swap_max_diff == 0.0,
        'best_cost_raw_probe': probe['best_cost_raw'],
        'best_cost_raw_day5': day5['best_cost_raw'],
        'best_cost_raw_match': probe['best_cost_raw'] == day5['best_cost_raw'],
    }


def render_verdict(m1, m2, m3, m4, m5):
    """Hypothesis C: PT swap pathology (over-hot, outlier propagation)."""
    signals_supporting = []
    signals_against = []

    if m1['probe_pct_in_target_zone'] < 0.1:
        signals_supporting.append(
            f"swap_acc in [0.15, 0.35] only {m1['probe_pct_in_target_zone']:.1%} "
            f"({m1['probe_n_in_target_zone']}/{m1['probe_total']})"
        )
    if m1['probe_n_above_0.5'] > m1['probe_total'] * 0.5:
        signals_supporting.append(
            f"swap_acc > 0.5 in {m1['probe_n_above_0.5']}/{m1['probe_total']} "
            f"({m1['probe_n_above_0.5']/m1['probe_total']:.1%}) — chronically over-hot"
        )
    if m1['log_per100_max_cliff_step'] == 800:
        signals_supporting.append(
            f"swap_acc cliff at step 800: {m1['log_per100_max_cliff_from']:.3f} → "
            f"{m1['log_per100_max_cliff_to']:.3f} (synchronized with cost jump)"
        )
    if m2['cost_min_step_800'] is not None and m2['cost_min_step_800'] < -10:
        signals_supporting.append(
            f"cost_min jump at step 800: {m2['cost_min_step_700']} → "
            f"{m2['cost_min_step_800']} (concurrent with swap_acc cliff)"
        )
    if m2['max_separation_delta'] < -5:
        signals_supporting.append(
            f"cost_min vs cost_mean separation = {m2['max_separation_delta']:.2f} "
            f"at step {m2['max_separation_step']} — outlier replica signature"
        )

    if not m4['ever_reached_psi_star_1.5']:
        signals_against.append(
            "psi_max never reached psi_star=1.5 → subspace clamping mechanism never "
            "kicked in; pure PT-Langevin regime throughout (limits Hypothesis A "
            "scenarios where subspace flip-flops would matter)"
        )

    if m5['bit_exact_match']:
        signals_supporting.append(
            "probe-run trajectory bit-exact match Day 5 first 1100 steps → "
            "anomaly is deterministic given seed=42 + cfg, not flaky"
        )

    n_supporting = len(signals_supporting)
    if n_supporting >= 4:
        verdict = 'STRONG'
    elif n_supporting >= 2:
        verdict = 'MODERATE'
    elif n_supporting >= 1:
        verdict = 'WEAK'
    else:
        verdict = 'INCONCLUSIVE'

    return {
        'verdict': verdict,
        'n_supporting_signals': n_supporting,
        'n_against_signals': len(signals_against),
        'supporting': signals_supporting,
        'against': signals_against,
    }


def main():
    probe, day5 = load_jsons()
    psi_log, cost_log, swap_log = parse_log_psi_max('logs/probe_day6_seed42_20260511_0736.log')

    m1 = metric_1_swap_acc(probe, day5, swap_log)
    m2 = metric_2_cost_separation(probe, day5, cost_log)
    m3 = metric_3_sigma_e(probe, day5)
    m4 = metric_4_psi_trajectory(psi_log)
    m5 = metric_5_bit_exact_check(probe, day5)
    v = render_verdict(m1, m2, m3, m4, m5)

    print("\n" + "=" * 70)
    print("METRIC 1: swap_acc trajectory (Hypothesis C indicator)")
    print("=" * 70)
    for k, val in m1.items():
        print(f"  {k}: {val}")

    print("\n" + "=" * 70)
    print("METRIC 2: cost_min vs cost_mean separation (outlier replica)")
    print("=" * 70)
    for k, val in m2.items():
        print(f"  {k}: {val}")

    print("\n" + "=" * 70)
    print("METRIC 3: sigma_E_max trajectory (VMC mixing)")
    print("=" * 70)
    for k, val in m3.items():
        print(f"  {k}: {val}")

    print("\n" + "=" * 70)
    print("METRIC 4: psi_max trajectory (normalization breach)")
    print("=" * 70)
    for k, val in m4.items():
        print(f"  {k}: {val}")

    print("\n" + "=" * 70)
    print("METRIC 5: bit-exact Day-5 reproduction (sanity)")
    print("=" * 70)
    for k, val in m5.items():
        print(f"  {k}: {val}")

    print("\n" + "=" * 70)
    print(f"HYPOTHESIS C VERDICT: {v['verdict']}")
    print(f"  supporting signals ({v['n_supporting_signals']}):")
    for s in v['supporting']:
        print(f"    - {s}")
    if v['against']:
        print(f"  against / qualifier signals ({v['n_against_signals']}):")
        for s in v['against']:
            print(f"    - {s}")
    print("=" * 70)

    out_path = Path('results/probe_analysis_hypothesis_c.json')
    with open(out_path, 'w') as f:
        json.dump({
            'metric_1_swap_acc': m1,
            'metric_2_cost_separation': m2,
            'metric_3_sigma_e': m3,
            'metric_4_psi_trajectory': m4,
            'metric_5_bit_exact': m5,
            'hypothesis_c_verdict': v,
        }, f, indent=2, default=str)
    print(f"\nMachine-readable summary saved: {out_path}")


if __name__ == '__main__':
    main()
