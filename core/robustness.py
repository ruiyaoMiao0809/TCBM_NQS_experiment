"""
Robustness utilities for production experiment scripts.

Three-layer protection against atexit defect (Issue 6 in known_issues_day2.md):
- Layer A: periodic checkpoint (every N steps, sync write)
- Layer B: callback abort + immediate dump + sys.exit on anomaly
- Layer C: SIGTERM handler dumps CPU state before exit

Helper update_capture_with_optimizer_state(capture, optimizer, step, extra) 提供
standard pattern 让 experiment scripts 把 theta + best_x + extra tensors 加进
periodic checkpoint (Day 6 Phase 2.5 Path δ: 修 Day 5 0-.pt-files defect).

Usage in experiment scripts:
    from core.robustness import (
        install_robustness_handlers,
        make_periodic_checkpoint_callback,
        make_anomaly_detection_callback,
    )

    install_robustness_handlers(state_capture, output_dir)
    callbacks = [
        make_periodic_checkpoint_callback(state_capture, every_n=500),
        make_anomaly_detection_callback(state_capture, abort_conditions),
    ]
"""

import atexit
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch


# Module-level state capture dict, populated by experiment script
_global_state: Dict[str, Any] = {
    'capture_dict': None,
    'output_dir': None,
    'run_name': None,
}


def install_robustness_handlers(
    state_capture: Dict[str, Any],
    output_dir: str,
    run_name: str,
) -> None:
    """
    Install SIGTERM/SIGINT handler + atexit fallback.

    state_capture should be a mutable dict that the experiment script writes to
    (theta, positions, step, cost, etc). Handlers will dump whatever is in it.
    """
    _global_state['capture_dict'] = state_capture
    _global_state['output_dir'] = output_dir
    _global_state['run_name'] = run_name

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Layer C: signal handlers
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Atexit fallback (last resort, not relied upon)
    atexit.register(_atexit_dump)


def _signal_handler(signum, frame):
    """SIGTERM/SIGINT: dump CPU state, then exit."""
    print(f"\n*** Signal {signum} received, dumping state and exiting ***", flush=True)
    _dump_state(suffix='signal_handler')
    sys.exit(0)


def _atexit_dump():
    """Last-resort atexit dump. Not relied upon (Issue 6 defect)."""
    _dump_state(suffix='atexit')


def _dump_state(suffix: str) -> None:
    """Dump state_capture to disk. Called from signal handler or atexit."""
    capture = _global_state.get('capture_dict')
    output_dir = _global_state.get('output_dir')
    run_name = _global_state.get('run_name')

    if capture is None or output_dir is None:
        print(f"[{suffix}] no state to dump (handlers not installed?)", flush=True)
        return

    # Separate tensor and non-tensor parts
    tensors = {}
    metadata = {}
    for k, v in capture.items():
        if isinstance(v, torch.Tensor):
            tensors[k] = v.detach().cpu()
        else:
            try:
                json.dumps(v)
                metadata[k] = v
            except (TypeError, ValueError):
                metadata[k] = str(v)

    # Tensors → .pt
    if tensors:
        pt_path = Path(output_dir) / f"{run_name}_{suffix}_state.pt"
        torch.save(tensors, pt_path)
        print(f"[{suffix}] saved tensors to {pt_path}", flush=True)

    # Metadata → .json
    if metadata:
        metadata['_dump_suffix'] = suffix
        metadata['_dump_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        json_path = Path(output_dir) / f"{run_name}_{suffix}_metadata.json"
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"[{suffix}] saved metadata to {json_path}", flush=True)


def update_capture_with_optimizer_state(
    capture_dict: Dict[str, Any],
    optimizer,
    step: int,
    extra_tensors: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    """
    Update capture_dict with current optimizer tensors (theta + best_x + grad if available).

    Called by experiment script's callback to ensure periodic checkpoints include tensor state.

    Args:
        capture_dict: The mutable state dict passed to install_robustness_handlers
        optimizer: TCBMOptimizer instance exposing _latest_positions (M, D) and
            _best_positions (D,) — set by optimize() right before callback fire
            (Phase 2.5 Path κ1).
        step: Current step (for logging)
        extra_tensors: Optional dict of additional tensors to capture
            (e.g. {'grad_norm_history': tensor, 'log_psi_diff_max': tensor})
    """
    if hasattr(optimizer, '_latest_positions') and optimizer._latest_positions is not None:
        capture_dict['theta_replicas'] = optimizer._latest_positions.detach().cpu()

    if hasattr(optimizer, '_best_positions') and optimizer._best_positions is not None:
        capture_dict['best_x'] = optimizer._best_positions.detach().cpu()

    if extra_tensors is not None:
        for k, v in extra_tensors.items():
            if isinstance(v, torch.Tensor):
                capture_dict[f'extra_{k}'] = v.detach().cpu()
            else:
                capture_dict[f'extra_{k}'] = v

    capture_dict['last_step_captured'] = step


def make_periodic_checkpoint_callback(
    state_capture: Dict[str, Any],
    every_n: int = 500,
) -> Callable:
    """
    Layer A: every N steps, force-write checkpoint.

    Returns a callback compatible with TCBMOptimizer.optimize() callback signature.
    """
    def callback(step: int, info: Dict[str, Any]) -> None:
        if step > 0 and step % every_n == 0:
            _dump_state(suffix=f'checkpoint_step{step}')
            print(f"[checkpoint] step {step} saved", flush=True)

    return callback


def make_anomaly_detection_callback(
    state_capture: Dict[str, Any],
    abort_conditions: Optional[Dict[str, Callable]] = None,
) -> Callable:
    """
    Layer B: callback checks abort conditions, immediate dump + sys.exit on trigger.

    abort_conditions: dict of {condition_name: callable(info_dict) -> bool}
    Default conditions: grad_nan, swap_acc_extreme, cost_diverge.
    """
    if abort_conditions is None:
        abort_conditions = _default_abort_conditions()

    swap_acc_history = []  # for sustained-extreme detection
    best_cost = [float('inf')]  # mutable closure for cost_diverge

    def callback(step: int, info: Dict[str, Any]) -> None:
        # Update best_cost for cost_diverge check
        cur_cost = info.get('cost_min')
        if cur_cost is not None and cur_cost < best_cost[0]:
            best_cost[0] = cur_cost

        # swap_acc history (sustained extreme)
        cur_swap = info.get('swap_acc')
        if cur_swap is not None:
            swap_acc_history.append(cur_swap)
            if len(swap_acc_history) > 5:
                swap_acc_history.pop(0)

        # Run conditions
        for name, check in abort_conditions.items():
            try:
                triggered = check(info, swap_acc_history=swap_acc_history, best_cost=best_cost[0])
            except Exception as e:
                print(f"[anomaly_check:{name}] exception: {e}, skipping", flush=True)
                continue

            if triggered:
                print(f"\n*** ABORT: condition '{name}' triggered at step {step} ***", flush=True)
                state_capture['abort_step'] = step
                state_capture['abort_condition'] = name
                state_capture['abort_info'] = {k: v for k, v in info.items()
                                                if not isinstance(v, torch.Tensor)}
                _dump_state(suffix=f'abort_{name}_step{step}')
                print(f"*** Dump complete, exiting ***", flush=True)
                sys.exit(0)

    return callback


def _default_abort_conditions() -> Dict[str, Callable]:
    """Default abort conditions for production runs."""

    def grad_nan(info, **kwargs):
        return info.get('is_grad_nan', False) is True

    def swap_acc_extreme(info, swap_acc_history=None, **kwargs):
        if swap_acc_history is None or len(swap_acc_history) < 3:
            return False
        recent = swap_acc_history[-3:]
        return all(s < 0.1 for s in recent) or all(s > 0.9 for s in recent)

    def cost_diverge(info, best_cost=None, **kwargs):
        cur_cost = info.get('cost_min')
        if cur_cost is None or best_cost == float('inf'):
            return False
        if best_cost == 0:
            return abs(cur_cost) > 1000
        return abs(cur_cost - best_cost) > 100 * abs(best_cost)

    return {
        'grad_nan': grad_nan,
        'swap_acc_extreme': swap_acc_extreme,
        'cost_diverge': cost_diverge,
    }
