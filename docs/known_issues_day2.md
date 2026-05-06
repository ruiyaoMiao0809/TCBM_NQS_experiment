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

## TCBM vs Adam final-eval target asymmetry (Day 6 SI must disclose)

**Discovered**: Day 3 Adam baseline writing
**Status**: Accepted as algorithmic difference, not bug

**TCBM (run_gradient_baseline_v2.py)**:
- Final eval target = best_x (the θ from the replica with lowest single-shot cost)
- This selection bias is intrinsic to PT: best replica's θ is what algorithm "picks"
- Mitigated by N evaluations averaged at the final stage (NQS-1e debiasing)

**Adam (run_adam_baseline.py)**:
- Final eval target = final θ (from last step of optimization)
- Adam assumes monotonic convergence; selection bias would inflate apparent performance
- Final eval = 4-shot mean (no NQS-1e equivalent)

**Why not unified?**:
PT (M=12) needs replica selection by construction; Adam (M=1) doesn't have replicas
to select among. Forcing both to "final θ" hides PT's actual algorithmic output.
Forcing both to "best across noisy replicas" introduces selection bias on Adam.

**Day 6 paper SI must disclose this asymmetry** in the methods section, alongside
final-eval statistic (both = mean of 4 evaluations).

**Reference for handling**: Bukov 2021 reports final E from "lowest E across multiple
seeds at fixed config" (Section 7.1 plateau ≈ -0.5019), implying selection-by-cost
across runs is field standard for PT-like methods.

## NaN gradient at log_2cosh_complex inner=0 (hidden_act z = i·(π/2 + kπ))

**Discovered**: Day 4 (2026-04-30), via narrow_band_diagnostic + probe_grad_nan trail.

**Symptoms**: 
production v2 cost frozen at -1.6900 across step 100-400 (byte-identical), grad NaN first triggered at Stage A step 87 (subspace forced off, M=12 replicas Langevin-only). NaN replica index 5, ||θ||=6.2146 at trigger. forward intermediate trace 锁定 first non-finite quantity = `log(inner)` 在 core/j1j2_problem.py:228, shape [2000, 32], exactly 1 element = -inf, 0 NaN at forward time.

**Root cause hypothesis**: 
RBM forward 计算 `log(2·cosh(z))` with stable form `log_magnitude = |u| + 0.5·log(inner)` where `inner = (1+exp(-2|u|))² - 4·sin²(v)·exp(-2|u|)`, z = u + iv. 当 u≈0 且 v≈π/2 + kπ: `a = exp(-2|u|) ≈ 1`, `(1+a)² ≈ 4`, `4·sin²(v)·a ≈ 4` → `inner = 4 - 4 = 0` (float32 rounding 给精确 0). `log(0) = -inf` → forward 该 sample |ψ|²=0 被 VMC 自然排除 cost 仍 finite, 但 backward 链式法则的 `1/inner = 1/0 = +inf` → `inf × finite = inf` → `inf - inf = NaN`, 污染 34/1120 grad entries. NaN gradient 进 Langevin step → forces NaN → Metropolis acceptance 全拒 → theta 冻在 step 87 last-good value → cost 报告 frozen pattern.

**Author predicted but misjudged POC scope safety**: 
core/j1j2_problem.py:185-203 注释里作者明确预言此 bug ("inner is strictly positive except at the cosh zeros z = i·(π/2 + k·π), where inner = 0 and log diverges to -∞") 并主动选择不加 epsilon clamp ("clamps inject zero gradients"). 作者 judge POC scope (init_scale=0.01, 3000 steps) 安全, 错误来源是没把 sample dimension 放进概率计算: M=12 × 2000 samples × 32 hidden = 768K elements/step, 单点共振概率 ~1/64000 per element-step, 累计 ~50 step 必撞一次.

**Implication**: 
所有 NQS production runs 都会在 step ~50-100 触发 NaN. Day 4 Stage A subspace-off run + production v2 (含 subspace) 都触发, 与 TCBM 机制无关. 不 fix 整个 NMI submission NQS 部分死.

**Fix path (Day 5 Phase 2)**: 
Custom autograd Function for `_log_2cosh_complex`. Forward 保持原样 (log(inner)=-inf 是正确 analytic behavior, |ψ|²=0 让 sample 自动从 VMC 排除). Backward 显式处理 `inner→0` 极限, 返回 0 gradient 而不是 1/inner. 数学 + physics 一致: cosh=0 → ψ=0 → sample 对训练贡献为 0 → gradient 贡献也应为 0. 不用 epsilon clamp (避免作者反对的 "silently corrupt training" 失真).

**Verification**: 
用 `results/state_at_first_nan.pt` load 原本会爆的 state, fix 后 `total.backward()` 必须给 finite gradient (pytest test_log_2cosh_no_nan_at_singularity).

**Evidence**: 
- results/state_at_first_nan.pt (114 KB)
- results/probe_at_first_nan.json (4 KB)
- experiments/probe_grad_nan.py
- experiments/narrow_band_diagnostic.py


## Atexit dump systematic defect under SIGTERM/SIGINT during deep CUDA call

**Discovered**: Day 4 (2026-04-30), 100% trigger rate (production v2 + narrow_band 两次独立 run 都失败).

**Symptoms**: 
Python `atexit.register(...)` 注册的 partial-state dump handler 在进程被 SIGTERM 或 SIGINT 终止时未触发. Day 4 production v2 (kill via SIGTERM after 2.5h stall) + narrow_band Stage A (Ctrl-C/SIGINT after step 102) 两次都是 atexit 完全没 fire, 0 partial JSON 落盘. log 末尾无 `[atexit] dumping...` 字样.

**Root cause hypothesis**: 
Python signal handler register 在 Python interpreter 层. CUDA kernel launch 后控制权交 GPU driver, Python signal handler 要等下一个 Python bytecode tick 才执行. 如果 SIGTERM 直接 terminate 进程 (Linux default behavior 即如此, atexit 不在 SIGTERM 路径上, 只在 sys.exit / 自然 return 路径), 或 SIGINT 在 deep CUDA call 期间被 short-circuit, atexit chain 不跑.

**Implication**: 
Day 17-19 sweep risk: 60 runs (15 seeds × 4 methods), 统计上 ≥3-5 runs 需 mid-run interrupt (OOM/cluster maintenance/误操作), 每个 lost partial 重跑 5-6h, 累计风险 15-30h. 不 fix 整个 sweep robustness 数据完整性受威胁.

**Validated workaround (Day 4)**: 
"显式 save-on-failure + sys.exit(0)" pattern. 在 callback 检测到 abort condition (e.g. `is_grad_nan=True`) 时立即:
1. `torch.save({state}, path)` 显式同步写盘
2. `json.dump(probe_report, ...)` 显式同步写盘
3. `sys.exit(0)` 走 Python 自然 exit path

实测 (probe_on_nan mode) 完全绕过 defect, state.pt + probe.json 都落盘 + 进程干净退出.

**Fix path (Day 5 Phase 1)**: 
将 workaround pattern 正式 backport 到所有 production scripts:
- experiments/run_gradient_baseline_v2.py
- experiments/run_adam_baseline.py
- experiments/run_sr_baseline.py (skeleton, 实现时同步加保护)
- 未来所有 sweep scripts

具体 pattern:
- 每 N step (N=500) 显式 `torch.save` checkpoint (而非 atexit-only)
- callback 检测异常 (NaN, swap_acc 越界, cost diverge >2σ) 时立即 dump + sys.exit
- atexit 保留作为 last-resort safety net (不依赖)
- SIGTERM handler 改成 显式 dump + sys.exit (而非默认 termination)

**Verification**: 
pytest test_explicit_save_on_sigterm: 启动 short run subprocess, 中途投 SIGTERM, 验证 expected dump file 在 SIGTERM 后 ≤2s 内落盘.

**Evidence**: 
- logs/STALL_evidence_production_v2_seed42_20260505_1431.log
- logs/STALL_evidence_tmux_capture_20260505_1431.log
- experiments/narrow_band_diagnostic.py (probe_on_nan mode validates workaround)
