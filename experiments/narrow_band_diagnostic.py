"""
experiments/narrow_band_diagnostic.py
=====================================
Day 4 stall root-cause diagnostic. Two-stage per-step probe of TCBM/NQS to
locate where production_v2 (cuda:3, killed at step 400 with frozen cost) died.

  Stage A: Langevin-only baseline (subspace forced OFF)
    Tests whether NQS+J1J2 problem can descend without TCBM subspace
    machinery. Same cfg as run_gradient_baseline_v2.py except:
      subspace_warmup=999, min_warmup=999, subspace_update_freq=999
    so _update_subspace() is never called within 200 steps.

  Stage B: TCBM full (production cfg)
    Exact production v2 cfg, n_steps=200. Catches the first SVD update at
    step 160 (subspace_warmup=150, freq=80) — the most likely SVD-warning
    trigger seen in the production stall log.

Per-step records (callback_every=1, dumped to JSON):
  step, theta_norm_overall, theta_norms_per_replica,
  grad_norm_overall, grad_norms_per_replica,
  cost_min, cost_mean, sigma_E_max, sigma_E_mean, swap_acc_recent,
  subspace_active, psi_raw, svd_failed_this_step,
  is_theta_nan, is_grad_nan

`psi_raw` is the unclamped value returned by _update_subspace; None on steps
where _update_subspace was not called. `svd_failed_this_step` is True on
steps where torch.linalg.svd / cusolver emitted a warning during this step.

Non-invasive instrumentation:
  - monkey-patch optimizer._batch_gradient to capture (positions, grads)
  - monkey-patch optimizer._update_subspace to capture raw psi
  - override warnings.showwarning to count cusolver/SVD warnings
No file under core/ is modified.

Robustness three-piece (production v2 had this too but atexit didn't fire —
under investigation; keep all three for the diagnostic):
  1. python -u + sys.stdout.reconfigure(line_buffering=True)
  2. signal.signal(SIGTERM, _sigterm_handler) → sys.exit(0)
  3. atexit.register(_atexit_dump) writing partial JSONs to results/

Run:
    TCBM_DEVICE=cuda:3 python -u experiments/narrow_band_diagnostic.py 2>&1 \
      | tee logs/narrow_band_$(date +%Y%m%d_%H%M).log

Expected wall: ~50-60 min total (Stage A ~25-30 min + 5s pause + Stage B
~25-30 min). Stage A has a 5-second pause before Stage B for manual abort.
"""
import sys
import os
import json
import time
import signal
import atexit
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.j1j2_problem import J1J2Problem
from core.tcbm_optimizer_NQS import TCBMOptimizer, TCBMConfig

E_0_TRUTH = -8.4579
DEVICE = os.environ.get('TCBM_DEVICE', 'cuda:0')
SEED = 42
N_STEPS = 200

OUT_A = Path('results/narrow_band_diagnostic_stageA.json')
OUT_B = Path('results/narrow_band_diagnostic_stageB.json')

# probe-on-nan mode: when env NARROW_BAND_MODE=probe_on_nan is set, the first
# is_grad_nan=True step triggers an inline probe + state dump + sys.exit(0),
# bypassing the atexit defect that bit production v2 and Stage A.
OUT_PROBE = Path('results/probe_at_first_nan.json')
OUT_STATE_NAN = Path('results/state_at_first_nan.pt')
PROBE_ON_NAN_MODE = os.environ.get('NARROW_BAND_MODE') == 'probe_on_nan'
_probe_done = {'fired': False}


# ─────────────────────────────────────────────────────────────────────────────
# Module-level state
# ─────────────────────────────────────────────────────────────────────────────

_state: Dict[str, Any] = {
    'current_stage': None,           # 'A' | 'B' | None
    'stageA_per_step': [],
    'stageB_per_step': [],
    'stageA_cfg': None,
    'stageB_cfg': None,
    'stageA_summary': None,
    'stageB_summary': None,
    'stageA_completed': False,
    'stageB_completed': False,
    'start_time': None,
}

# Holders updated by monkey-patches; read by callback
_capture: Dict[str, Any] = {
    'last_positions': None,    # (M, D) tensor at most-recent _batch_gradient call
    'last_grads': None,        # (M, D) tensor at most-recent _batch_gradient call
    'last_step_seen': -1,      # incremented by patched _batch_gradient
    'psi_per_step': {},        # {step: psi_value} from patched _update_subspace
    'svd_warning_count': 0,    # cumulative count
    'svd_warning_count_last': 0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Warning capture (SVD / cusolver / converge UserWarnings)
# ─────────────────────────────────────────────────────────────────────────────

_orig_showwarning = warnings.showwarning


def _showwarning_patched(message, category, filename, lineno, file=None, line=None):
    msg_str = str(message).lower()
    if ('svd' in msg_str or 'converge' in msg_str or 'cusolver' in msg_str):
        _capture['svd_warning_count'] += 1
    return _orig_showwarning(message, category, filename, lineno, file=file, line=line)


warnings.showwarning = _showwarning_patched
# Don't suppress repeated SVD warnings — we want to count every occurrence.
warnings.filterwarnings('always', message='.*svd.*')
warnings.filterwarnings('always', message='.*cusolver.*')
warnings.filterwarnings('always', message='.*converge.*')


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer monkey-patches (non-invasive)
# ─────────────────────────────────────────────────────────────────────────────

def install_optimizer_patches(optimizer: TCBMOptimizer) -> None:
    """Wrap _batch_gradient and _update_subspace to capture per-step state."""
    orig_batch_gradient = optimizer._batch_gradient
    orig_update_subspace = optimizer._update_subspace

    def patched_batch_gradient(positions: torch.Tensor) -> torch.Tensor:
        grads = orig_batch_gradient(positions)
        # Snapshot for the callback. .detach().clone() so subsequent in-place
        # ops (Langevin step) don't mutate what we recorded.
        _capture['last_positions'] = positions.detach().clone()
        _capture['last_grads'] = grads.detach().clone()
        _capture['last_step_seen'] += 1
        return grads

    def patched_update_subspace(grad_buffer, old_U, positions=None):
        result = orig_update_subspace(grad_buffer, old_U, positions=positions)
        # result = (U_new, sv_new, psi, delta_k_star, diag)
        psi_val = float(result[2])
        # last_step_seen was just set by patched_batch_gradient earlier in
        # this same optimizer step.
        _capture['psi_per_step'][_capture['last_step_seen']] = psi_val
        return result

    optimizer._batch_gradient = patched_batch_gradient
    optimizer._update_subspace = patched_update_subspace


# ─────────────────────────────────────────────────────────────────────────────
# probe-on-nan trigger (only fires when PROBE_ON_NAN_MODE=True)
# ─────────────────────────────────────────────────────────────────────────────

def _trigger_probe_and_exit(
    problem, positions: torch.Tensor, grads: torch.Tensor, step: int,
) -> None:
    """Save state + run probe_one_theta on the NaN-producing replica + sys.exit."""
    sys.path.insert(0, str(Path(__file__).parent))
    from probe_grad_nan import probe_one_theta, install_patches

    nan_mask = torch.isnan(grads).any(dim=1)
    nan_idx = int(nan_mask.nonzero()[0].item()) if nan_mask.any() else 0
    theta_1d = positions[nan_idx].detach().clone()

    OUT_STATE_NAN.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'theta': theta_1d.cpu(),
        'positions': positions.detach().cpu(),
        'grad': grads.detach().cpu(),
        'step': step,
        'nan_replica_idx': nan_idx,
    }, OUT_STATE_NAN)
    print(f"*** State saved to {OUT_STATE_NAN} (replica {nan_idx}) ***", flush=True)

    try:
        install_patches()
        result = probe_one_theta(
            problem, theta_1d.to(problem.device),
            label=f'narrow_band_step{step}_replica{nan_idx}',
        )
        result_lean = {k: v for k, v in result.items() if k != 'all_intermediates'}
        with open(OUT_PROBE, 'w') as f:
            json.dump({'step': step, 'nan_replica_idx': nan_idx,
                       'result': result_lean}, f, indent=2, default=str)
        print(f"*** Probe complete, dumped to {OUT_PROBE} ***", flush=True)
    except Exception as e:
        print(f"*** Probe FAILED: {type(e).__name__}: {e} ***", flush=True)

    print(f"*** First NaN at step {step}, probe complete, exiting ***", flush=True)
    sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# Per-step callback
# ─────────────────────────────────────────────────────────────────────────────

def make_callback(stage: str, problem=None) -> Callable[[int, Dict[str, float]], None]:
    target_list = (_state['stageA_per_step'] if stage == 'A'
                   else _state['stageB_per_step'])

    def cb(step: int, info: Dict[str, float]) -> None:
        positions = _capture['last_positions']
        grads = _capture['last_grads']

        if positions is not None:
            theta_norm_overall = float(positions.norm().item())
            theta_norms_per_replica = positions.norm(dim=1).cpu().tolist()
            is_theta_nan = bool(torch.isnan(positions).any().item())
        else:
            theta_norm_overall = float('nan')
            theta_norms_per_replica = []
            is_theta_nan = False

        if grads is not None:
            grad_norm_overall = float(grads.norm().item())
            grad_norms_per_replica = grads.norm(dim=1).cpu().tolist()
            is_grad_nan = bool(torch.isnan(grads).any().item())
        else:
            grad_norm_overall = float('nan')
            grad_norms_per_replica = []
            is_grad_nan = False

        psi_raw = _capture['psi_per_step'].get(step, None)

        delta = (_capture['svd_warning_count']
                 - _capture['svd_warning_count_last'])
        svd_failed = bool(delta > 0)
        _capture['svd_warning_count_last'] = _capture['svd_warning_count']

        record = {
            'step': step,
            'theta_norm_overall': theta_norm_overall,
            'theta_norms_per_replica': theta_norms_per_replica,
            'grad_norm_overall': grad_norm_overall,
            'grad_norms_per_replica': grad_norms_per_replica,
            'cost_min': float(info.get('cost_min', float('nan'))),
            'cost_mean': float(info.get('cost_mean', float('nan'))),
            'sigma_E_max': float(info.get('sigma_E_max', float('nan'))),
            'sigma_E_mean': float(info.get('sigma_E_mean', float('nan'))),
            'swap_acc_recent': float(info.get('swap_acc_recent', float('nan'))),
            'subspace_active': bool(info.get('subspace_active', False)),
            'psi_raw': psi_raw,
            'svd_failed_this_step': svd_failed,
            'is_theta_nan': is_theta_nan,
            'is_grad_nan': is_grad_nan,
        }
        target_list.append(record)

        # Loud per-step alert on NaN (continues collecting, doesn't abort)
        if is_theta_nan or is_grad_nan:
            print(
                f"  *** NAN at stage {stage} step {step}: "
                f"theta_nan={is_theta_nan} grad_nan={is_grad_nan} ***",
                flush=True,
            )
            # probe-on-nan mode: probe + save + sys.exit on first grad_nan
            if (PROBE_ON_NAN_MODE and is_grad_nan
                    and not _probe_done['fired']
                    and problem is not None
                    and positions is not None and grads is not None):
                _probe_done['fired'] = True
                _trigger_probe_and_exit(problem, positions, grads, step)

        # Loud per-step alert on SVD warning
        if svd_failed:
            print(
                f"  *** SVD WARNING at stage {stage} step {step} "
                f"(cumulative count = {_capture['svd_warning_count']}) ***",
                flush=True,
            )

        # Periodic progress (every 50 steps)
        if step % 50 == 0:
            print(
                f"[stage {stage} step {step:4d}] "
                f"||theta||={theta_norm_overall:.4f} "
                f"||grad||={grad_norm_overall:.4f} "
                f"cost_min={record['cost_min']:.4f} "
                f"cost_mean={record['cost_mean']:.4f} "
                f"psi_raw={psi_raw} "
                f"svd_fail={svd_failed} "
                f"sub_active={record['subspace_active']}",
                flush=True,
            )

    return cb


# ─────────────────────────────────────────────────────────────────────────────
# Summary computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_stage_summary(per_step: List[Dict]) -> Dict[str, Any]:
    """First occurrence of each anomaly type, plus full psi call history."""
    first_theta_nan = None
    first_grad_nan = None
    first_theta_frozen = None
    first_grad_dead = None
    first_svd_fail = None
    psi_calls: List = []

    prev_theta_norm: Optional[float] = None
    for rec in per_step:
        s = rec['step']
        if rec['is_theta_nan'] and first_theta_nan is None:
            first_theta_nan = s
        if rec['is_grad_nan'] and first_grad_nan is None:
            first_grad_nan = s
        if rec['svd_failed_this_step'] and first_svd_fail is None:
            first_svd_fail = s
        if (rec['grad_norm_overall'] < 1e-10
                and not (rec['grad_norm_overall'] != rec['grad_norm_overall'])  # exclude NaN
                and first_grad_dead is None):
            first_grad_dead = s
        if prev_theta_norm is not None:
            d = abs(rec['theta_norm_overall'] - prev_theta_norm)
            if d < 1e-8 and first_theta_frozen is None:
                first_theta_frozen = s
        prev_theta_norm = rec['theta_norm_overall']
        if rec['psi_raw'] is not None:
            psi_calls.append([s, rec['psi_raw']])

    return {
        'first_theta_nan_step': first_theta_nan,
        'first_grad_nan_step': first_grad_nan,
        'first_theta_frozen_step': first_theta_frozen,
        'first_grad_dead_step': first_grad_dead,
        'first_svd_fail_step': first_svd_fail,
        'psi_call_history': psi_calls,
        'n_steps_recorded': len(per_step),
    }


# ─────────────────────────────────────────────────────────────────────────────
# State dump (idempotent, safe to call from atexit / SIGTERM)
# ─────────────────────────────────────────────────────────────────────────────

def dump_state(label: str = 'partial') -> None:
    if _state['stageA_per_step']:
        snapshot = {
            'stage': 'A',
            'config': _state['stageA_cfg'],
            'per_step_data': _state['stageA_per_step'],
            'summary': (_state['stageA_summary']
                        or compute_stage_summary(_state['stageA_per_step'])),
            'completed': _state['stageA_completed'],
            'dump_label': label,
            'dump_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        try:
            OUT_A.parent.mkdir(parents=True, exist_ok=True)
            with open(OUT_A, 'w') as f:
                json.dump(snapshot, f, indent=2, default=str)
        except Exception as e:
            print(f"[{label}] FAILED dump A: {type(e).__name__}: {e}", flush=True)

    if _state['stageB_per_step']:
        snapshot = {
            'stage': 'B',
            'config': _state['stageB_cfg'],
            'per_step_data': _state['stageB_per_step'],
            'summary': (_state['stageB_summary']
                        or compute_stage_summary(_state['stageB_per_step'])),
            'completed': _state['stageB_completed'],
            'dump_label': label,
            'dump_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        try:
            OUT_B.parent.mkdir(parents=True, exist_ok=True)
            with open(OUT_B, 'w') as f:
                json.dump(snapshot, f, indent=2, default=str)
        except Exception as e:
            print(f"[{label}] FAILED dump B: {type(e).__name__}: {e}", flush=True)

    print(
        f"[{label}] dumped (A: {len(_state['stageA_per_step'])} steps, "
        f"B: {len(_state['stageB_per_step'])} steps)",
        flush=True,
    )


def _sigterm_handler(signum, frame):
    print(
        f"\n[SIGTERM at stage {_state['current_stage']}, graceful exit]",
        flush=True,
    )
    sys.exit(0)


def _atexit_dump():
    if _state['stageA_completed'] and _state['stageB_completed']:
        return
    if _state['start_time'] is None:
        return
    print("\n[atexit] dumping partial state due to interrupted exit", flush=True)
    dump_state('atexit_partial')


signal.signal(signal.SIGTERM, _sigterm_handler)
atexit.register(_atexit_dump)


# ─────────────────────────────────────────────────────────────────────────────
# Stage configs
# ─────────────────────────────────────────────────────────────────────────────

def make_cfg_stage_A() -> TCBMConfig:
    """Production v2 cfg with subspace mechanisms forced off."""
    return TCBMConfig(
        M=12, n_steps=N_STEPS, k=20,
        T_min=0.005, T_max=2.0, T_min_floor=0.002,
        lambda_min=0.05, lambda_max=2.0, tau_lambda=500,
        subspace_warmup=999,           # > N_STEPS so _update_subspace never fires
        subspace_update_freq=999,
        psi_star=1.5,
        min_warmup=999,
        grad_buffer_size=400, grad_buffer_use=200,
        subspace_source='gradient',
        n_vmc_samples=2000,
        n_vmc_samples_final=2,
        noise_temperature_alpha=1.0,
        seed=SEED,
    )


def make_cfg_stage_B() -> TCBMConfig:
    """Exact production v2 cfg (subspace_warmup=150, freq=80) with shortened n_steps."""
    return TCBMConfig(
        M=12, n_steps=N_STEPS, k=20,
        T_min=0.005, T_max=2.0, T_min_floor=0.002,
        lambda_min=0.05, lambda_max=2.0, tau_lambda=500,
        subspace_warmup=150, subspace_update_freq=80,
        psi_star=1.5, min_warmup=120,
        grad_buffer_size=400, grad_buffer_use=200,
        subspace_source='gradient',
        n_vmc_samples=2000,
        n_vmc_samples_final=2,
        noise_temperature_alpha=1.0,
        seed=SEED,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run one stage
# ─────────────────────────────────────────────────────────────────────────────

def run_stage(stage: str) -> None:
    cfg = make_cfg_stage_A() if stage == 'A' else make_cfg_stage_B()
    cfg_dict = {k: v for k, v in cfg.__dict__.items() if not k.startswith('_')}

    if stage == 'A':
        _state['stageA_cfg'] = cfg_dict
    else:
        _state['stageB_cfg'] = cfg_dict

    title = ('Langevin-only baseline (subspace OFF)' if stage == 'A'
             else 'TCBM full (production cfg)')
    print(f"\n{'=' * 70}")
    print(f"Stage {stage}: {title}")
    print(f"{'=' * 70}")
    print(f"Device: {DEVICE}, seed: {SEED}, n_steps: {N_STEPS}, M: {cfg.M}")
    print(
        f"subspace_warmup={cfg.subspace_warmup}, "
        f"min_warmup={cfg.min_warmup}, freq={cfg.subspace_update_freq}"
    )
    print(f"{'=' * 70}\n", flush=True)

    # Reset capture state
    _capture['last_positions'] = None
    _capture['last_grads'] = None
    _capture['last_step_seen'] = -1
    _capture['psi_per_step'] = {}
    _capture['svd_warning_count'] = 0
    _capture['svd_warning_count_last'] = 0
    _state['current_stage'] = stage

    # Fresh problem + optimizer per stage
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    problem = J1J2Problem(Lx=4, Ly=4, alpha=2, J1=1.0, J2=0.5, device=DEVICE)
    problem.set_seed(SEED)
    print(
        f"Problem: 4x4 J1-J2 PBC, J2/J1=0.5, D_params={problem.dim}",
        flush=True,
    )

    optimizer = TCBMOptimizer(problem, cfg)
    install_optimizer_patches(optimizer)

    cb = make_callback(stage, problem=problem)
    t0 = time.time()
    optimizer.optimize(callback=cb, callback_every=1)
    elapsed = time.time() - t0

    summary = compute_stage_summary(
        _state['stageA_per_step'] if stage == 'A'
        else _state['stageB_per_step']
    )
    summary['elapsed_seconds'] = elapsed
    summary['total_svd_warnings'] = _capture['svd_warning_count']

    print(f"\n[stage {stage} DONE in {elapsed / 60:.1f} min]")
    print(f"  first_theta_nan_step    = {summary['first_theta_nan_step']}")
    print(f"  first_grad_nan_step     = {summary['first_grad_nan_step']}")
    print(f"  first_theta_frozen_step = {summary['first_theta_frozen_step']}")
    print(f"  first_grad_dead_step    = {summary['first_grad_dead_step']}")
    print(f"  first_svd_fail_step     = {summary['first_svd_fail_step']}")
    print(f"  total SVD warnings      = {summary['total_svd_warnings']}")
    print(f"  psi_call_history (len)  = {len(summary['psi_call_history'])}")
    if summary['psi_call_history']:
        print(f"    first 5: {summary['psi_call_history'][:5]}")

    if stage == 'A':
        _state['stageA_summary'] = summary
        _state['stageA_completed'] = True
    else:
        _state['stageB_summary'] = summary
        _state['stageB_completed'] = True

    dump_state(f'stage{stage}_final')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    _state['start_time'] = time.time()
    print(f"narrow_band_diagnostic starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output: {OUT_A}, {OUT_B}")

    print("\nStage A: Langevin-only baseline starting", flush=True)
    run_stage('A')

    print("\nSTAGE_A_COMPLETE — pausing 5s before stage B", flush=True)
    time.sleep(5)

    print("\nStage B: TCBM full starting", flush=True)
    run_stage('B')

    elapsed = time.time() - _state['start_time']
    print(f"\n{'=' * 70}")
    print(f"narrow_band_diagnostic COMPLETE in {elapsed / 60:.1f} min")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
