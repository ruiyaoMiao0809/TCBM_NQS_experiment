# Day 2 Discovered Issues

## VMC evaluate cost severely underestimated

**Discovered**: Day 2 baseline run, 09:06-12:21+ (>3h vs 30 min expected, 6.5× overrun)

**Symptoms**:
- Python alive, GPU util 87-99% throughout, mem stable 3415 MB
- No NaN/inf, no OOM, no crash
- Pure wall-clock cost issue
- Indirect health signals (process state R, wchan=0, no lock-holder pattern, 91% CPU
  utilization on main thread, sustained GPU util) all consistent with healthy
  long-running compute, not hang/deadlock

**Root cause hypothesis**:
- `J1J2Problem.evaluate()` is per-replica sequential, not batched over M=12 replicas
- Per-step cost ≈ 3-5s × 3000 steps = ~3-4 hours wall clock
- Originally estimated 0.6s/step based on Day 1 single-evaluate timing

**Implication**:
- Day 3 retry with n_vmc_samples=4000 → ~6h, not viable in single sitting
- Week 3 robustness sweep with 15 seeds × 4 methods → ~250-300h wall clock if not optimized
- **Must be fixed before Week 3 sweep, ideally Day 8-9**

**Fix path** (not Day 2 scope):
- Batch `_evaluate_vmc` over the replica dimension M=12
- Estimated speedup: 5-8× (limited by GPU mem, not full M=12 since each replica's 2000 samples × VMC chain state)
- Alternative: vectorize over the inner Monte Carlo step in `J1J2Problem._sample_chain`

**Validated against Bukov 2021**:
- Bukov uses NMC=2^15=32768 (16× larger), but runs on multi-day timeframes
- Their per-step is also sequential (jVMC TDVP loop)
- Our 4×4 (16 spins) vs their 6×6 (36 spins) — Hilbert dim ratio 1024×, but VMC sample cost scales with n_spins not Hilbert dim
- Conclusion: our cost is "expected" for unoptimized 4×4 VMC, not anomalously slow

## record_every vs swap_interval LCM stale issue

**Discovered**: Day 2 Step 3 pre-verify
**Status**: caveat noted in script docstring + JSON output
**Fix path**: Day 19-20 SI write-up, recompute swap statistics from raw per-call data

## ptrace-based stack inspection unavailable on this server

**Discovered**: Day 2 mid-run debugging attempt 2026-04-28 12:21
**Symptoms**: ptrace_scope=1 restricts non-parent attach; py-spy/gdb both blocked without
sudo. Only /proc/PID/{wchan,stat,task/} and process-level health signals available.

**Workaround applied**: Used /proc/PID/task/*/wchan to confirm no deadlock pattern.
Infer healthy compute from: main thread state=R + sustained CPU util + sustained GPU
util + stable GPU mem. Sufficient for Day 2 alive-check, insufficient for fine-grained
profiling.

**Fix path** (Week 2+):
- Either ask server admin to set ptrace_scope=0 (system-wide, may have security review)
- Or `sudo setcap cap_sys_ptrace=eip $(which py-spy)` (one-time)
- Or use Python's built-in `faulthandler.dump_traceback_later()` proactively in
  long-running scripts to dump every N seconds without external attach
