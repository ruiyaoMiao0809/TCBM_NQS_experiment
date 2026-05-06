"""
Tests for core.robustness atexit defect workaround (Issue 6).
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch


def test_periodic_checkpoint_callback_writes_disk(tmp_path):
    """Layer A: every N steps callback should write checkpoint."""
    from core.robustness import (
        install_robustness_handlers,
        make_periodic_checkpoint_callback,
    )

    state = {'theta': torch.zeros(10), 'step': 0}
    install_robustness_handlers(state, output_dir=str(tmp_path), run_name='test_a')

    cb = make_periodic_checkpoint_callback(state, every_n=10)
    for step in range(0, 25):
        info = {'cost_min': -1.0 - step * 0.01}
        state['step'] = step
        cb(step, info)

    # 步 10 和 步 20 应触发 checkpoint
    files = list(tmp_path.glob('test_a_checkpoint_step*_state.pt'))
    assert len(files) == 2, f"expected 2 checkpoints, got {len(files)}: {files}"


def test_anomaly_callback_grad_nan_triggers_exit(tmp_path):
    """Layer B: grad_nan condition should trigger immediate dump + sys.exit(0)."""
    from core.robustness import (
        install_robustness_handlers,
        make_anomaly_detection_callback,
    )

    state = {'theta': torch.zeros(5)}
    install_robustness_handlers(state, output_dir=str(tmp_path), run_name='test_b')

    cb = make_anomaly_detection_callback(state)

    info_clean = {'is_grad_nan': False, 'cost_min': -1.0}
    cb(5, info_clean)  # should not exit

    # Trigger NaN
    info_nan = {'is_grad_nan': True, 'cost_min': -1.0}
    try:
        cb(10, info_nan)
        assert False, "expected sys.exit(0)"
    except SystemExit as e:
        assert e.code == 0

    files = list(tmp_path.glob('test_b_abort_grad_nan_step*_state.pt'))
    assert len(files) == 1, f"expected 1 abort dump, got {len(files)}: {files}"


def test_explicit_save_on_sigterm(tmp_path):
    """Layer C: subprocess receives SIGTERM, dump should appear within 2s."""

    # Write a tiny test script that registers handlers and sleeps
    test_script = tmp_path / 'sigterm_test.py'
    test_script.write_text(f"""
import sys
sys.path.insert(0, {repr(str(Path(__file__).parent.parent))})
import time
import torch
from core.robustness import install_robustness_handlers

state = {{'theta': torch.zeros(5), 'step': 42, 'note': 'sigterm_test'}}
install_robustness_handlers(state, output_dir={repr(str(tmp_path))}, run_name='sigterm_subprocess')

print('HANDLERS_READY', flush=True)  # readiness sentinel
time.sleep(60)
""")

    proc = subprocess.Popen(
        [sys.executable, str(test_script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,  # line-buffered
        text=True,
    )

    # Poll stdout for HANDLERS_READY sentinel (max 30s timeout)
    ready = False
    deadline = time.time() + 30.0
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:  # EOF
            time.sleep(0.05)
            continue
        if 'HANDLERS_READY' in line:
            ready = True
            break

    if not ready:
        proc.kill()
        stderr = proc.stderr.read() if proc.stderr else ''
        assert False, f"subprocess never printed HANDLERS_READY in 30s. stderr: {stderr}"

    # Grace period: wait for subprocess to fully exit print() and enter time.sleep().
    # Without this, SIGTERM can arrive mid-print causing "reentrant call inside
    # BufferedWriter" — the signal handler's own print clashes with the in-flight
    # print, signal handler aborts, only atexit fallback fires (signal_handler dump
    # never produced).
    time.sleep(0.5)
    proc.send_signal(signal.SIGTERM)

    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        assert False, "subprocess did not exit within 10s of SIGTERM"

    # Verify dump file exists
    state_files = list(tmp_path.glob('sigterm_subprocess_signal_handler_*'))
    assert len(state_files) >= 1, \
        f"expected dump file from signal handler, got: {list(tmp_path.iterdir())}"
