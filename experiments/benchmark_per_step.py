"""
experiments/benchmark_per_step.py
==================================
Day 3 micro-benchmark. Measure real per-step wall-clock on a short 100-step run
+ final eval, then project full 3000-step production wall before launching.

Day 2 lesson: a 30-min wall estimate (based on Day 1 single evaluate timing)
was 6× too low (actual was 6h22min). 100 steps × current per-step cost is
much more reliable than extrapolating from a single evaluate.

Caveat: subspace_warmup=150 > 100, so SVD overhead is NOT measured here.
SVD adds maybe 5-10% per-step in the production run; account for that
when interpreting the projection.

Run:
    python -u experiments/benchmark_per_step.py 2>&1 \
      | tee logs/benchmark_$(date +%Y%m%d_%H%M).log

Decision rule (printed at end):
    < 4h projected  -> launch production immediately (run_gradient_baseline_v2.py)
    4-8h projected  -> ask Nick for go/no-go
    > 8h projected  -> escalate, consider batched evaluate refactor first
"""
import sys
import time
from pathlib import Path

import torch

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.j1j2_problem import J1J2Problem
from core.tcbm_optimizer_NQS import TCBMOptimizer, TCBMConfig

N_BENCH_STEPS = 100
N_PROD_STEPS = 3000


def main():
    torch.manual_seed(42)
    problem = J1J2Problem(Lx=4, Ly=4, alpha=2, J1=1.0, J2=0.5, device='cuda')
    print(f"Benchmarking {N_BENCH_STEPS} steps with M=12 replicas on cuda...")
    print(f"D_params = {problem.dim}")

    cfg = TCBMConfig(
        M=12, n_steps=N_BENCH_STEPS, k=20,
        T_min=0.005, T_max=2.0, T_min_floor=0.002,
        lambda_min=0.05, lambda_max=2.0, tau_lambda=500,
        subspace_warmup=150,        # 100 < 150 → SVD never triggers in bench
        subspace_update_freq=80,
        psi_star=1.5, min_warmup=120,
        grad_buffer_size=400, grad_buffer_use=200,
        subspace_source='gradient',
        n_vmc_samples=2000,
        n_vmc_samples_final=2,      # match production v2
        noise_temperature_alpha=1.0,
        seed=42,
    )

    print(f"Note: subspace_warmup=150 > {N_BENCH_STEPS}, SVD overhead NOT measured")
    print(f"Note: n_vmc_samples_final={cfg.n_vmc_samples_final} matches production v2")
    print(f"{'='*70}\n")

    t_total_start = time.time()

    # Track when main loop exits (last callback fires at step N-1 if divisible by callback_every)
    main_loop_end_time = [None]

    def cb(step, info):
        # Last main-loop callback marks ~end of main loop (final eval comes after).
        if step == N_BENCH_STEPS - 1:
            main_loop_end_time[0] = time.time()
        if step % 20 == 0:
            print(
                f"[step {step:3d}] cost_min={info.get('cost_min', 0):.4f} "
                f"sigma_E_max={info.get('sigma_E_max', 0):.4f}",
                flush=True,
            )

    optimizer = TCBMOptimizer(problem, cfg)
    t_main_start = time.time()
    result = optimizer.optimize(callback=cb, callback_every=10)
    t_total = time.time() - t_total_start

    if main_loop_end_time[0] is not None:
        t_main = main_loop_end_time[0] - t_main_start
        t_final = t_total - t_main
    else:
        # Fallback: rough 70/30 split if callback at last step missed.
        t_main = t_total * 0.7
        t_final = t_total * 0.3
        print(f"WARNING: callback at step {N_BENCH_STEPS - 1} missed; using rough 70/30 split")

    per_step = t_main / N_BENCH_STEPS
    projected_main_min = per_step * N_PROD_STEPS / 60.0
    projected_final_min = t_final / 60.0
    projected_total_min = projected_main_min + projected_final_min
    projected_total_h = projected_total_min / 60.0

    print(f"\n{'='*70}")
    print(f"Benchmark results")
    print(f"{'='*70}")
    print(f"{N_BENCH_STEPS} steps + final eval (n_final={cfg.n_vmc_samples_final}) wall: {t_total:.1f}s")
    print(f"  Main loop only:  {t_main:.1f}s ({per_step*1000:.0f} ms/step)")
    print(f"  Final eval:      {t_final:.1f}s")
    print()
    print(f"Projected for {N_PROD_STEPS} steps + final:")
    print(f"  Main loop:  {projected_main_min:.1f} min")
    print(f"  Final eval: {projected_final_min:.1f} min (constant w.r.t. n_steps)")
    print(f"  Total:      {projected_total_min:.1f} min = {projected_total_h:.2f} h")
    print()
    print(f"Decision:")
    if projected_total_h < 4:
        print(f"  [GO]      < 4h, launch production immediately")
    elif projected_total_h < 8:
        print(f"  [ASK]     {projected_total_h:.1f}h, ask Nick for go/no-go")
    else:
        print(f"  [ESCALATE] {projected_total_h:.1f}h, consider batched evaluate refactor first")
    print()
    print(f"Compare to Day 2 actual: 6.4h (4.3h main + ~2h final eval at n_final=4)")
    print(f"v2 with n_final=2 should subtract ~1h from final eval.")

    # Sanity print on result diagnostics
    print(f"\nSanity: best_cost_raw={result['best_cost_raw']:.4f}, "
          f"best_cost_debiased={result['best_cost_debiased']:.4f}")


if __name__ == '__main__':
    main()
