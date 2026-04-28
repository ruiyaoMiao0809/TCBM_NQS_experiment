# Daily Log

## Day 1 (2026-04-27)

### 计划
- [x] Phase 0: 清理 conda channel 配置（删除失效 nvidia/pytorch 镜像 + defaults）
- [x] Phase 1: 创建 tcbm_nqs env (minimal conda + pip PyTorch CUDA 12.1)
- [x] Phase 2: 项目目录补全 + week1_setup.sh 6/6 smoke tests + git init + initial commit
- [x] 子任务 A: stable log(2·cosh) + physical factor docstring
- [x] 子任务 B: pytest suite (10 tests) + fix latent gradient autograd bug
- [x] 子任务 C: ED reference + tmp_check_evaluate (sanity 全过)
- [x] Protocol/Prediction/ed_reference 三处文档修正（Nick 文献验证收敛后）
- [x] daily_log + final commit

### 实际完成

**Phase 0–2 (env + setup)**:
- conda user-level condarc 仅保留清华 main + conda-forge；defaults 由系统级 condarc 兜底
- pytorch 2.5.1+cu121, scipy 1.17.1, numpy 2.4.3, matplotlib 3.10.9 全部装上
- A10 23.6 GB GPU + CUDA 12.1 + cuDNN 9.1 sanity ✓
- tmux session `tcbm_nqs` 建立，env 已激活，作为后续 long-running 任务的标准容器
- commit `fbc2c91`: "Day 0: Initial skeleton + Protocol v2.1 + env configured"（21 文件，6229 行）

**子任务 A (stable form refactor)**:
- 替换 `_log_cosh_complex` → `_log_2cosh_complex`（C2 模/相位分解 + Option β API）
- 公式: `log|2·cosh(z)| = |u| + (1/2)·log[(1+e^(-2|u|))² - 4·sin²(v)·e^(-2|u|)]`
        `arg(cosh(z))   = atan2(tanh(u)·sin(v), cos(v))`
- 旧实现在 |Re(z)| > 89 (float32) `cosh/sinh` 溢出 + `clamp(min=1e-30)` 注入 zero gradient
- 新形式：a=exp(-2|u|) ∈ [0,1] 永不溢出，autograd 处处 differentiable（除 cosh 零点）
- 同步在 `_local_energy_batch` docstring 添加 spin-1/2 physical factor 推导（J·1/4 对角 + J·1/2 非对角）
- 7 组 (u,v) 边界测试 + autograd backward + log_psi_rbm 集成 sanity 全过
- commit `2c94301`: "Day 1.A: stable log(2·cosh) + physical factor docstring" (+88/-35)

**子任务 B (pytest + gradient fix)**:
- 新建 `tests/test_j1j2_problem.py`，10 个 test（test_9 parametrize 展开 7 → pytest 跑 16 项）
- 发现并修复 latent gradient autograd bug（Day 0 骨架的 `evaluate()` in-place 赋值 +
  `_evaluate_vmc/_evaluate_exact` `.detach()` 切断反向传播）
- 修复方案：`torch.stack` 替代 in-place buffer + 去 3 处 detach + 添加 partial-gradient
  contract docstring（partial gradient 缺 score-function term，Bukov 2021 / NetKet
  Week 2 替换为 log-derivative estimator）
- N=4 (2×2 PBC) ED matrix cross-check 通过：abs error 1.572e-07（float32 精度下限）
- commit `bbe7290`: "Day 1.B: pytest suite (10 tests) + fix latent gradient autograd bug" (+318/-11)

**子任务 C (ED reference + runtime checks)**:
- ED Lanczos 在 4×4 PBC 三个 J2 点上跑通：
  - J2=0.5: E_0/N = -0.5286 ⚠ 与 Protocol v2.1 §1.1 引用的 -0.497 不符
  - J2=0.0 (sanity): E_0/N = -0.7018 ✓ 匹配 well-known benchmark -0.7017
  - J2 scan (J2 ∈ [0,1] 步长 0.1): 经典 V 形 + frustrated 区域 gap 塌陷至 0.086 (J2=0.7)
- tmp_check_evaluate.py 三个 sanity 全 PASS：
  - (a) energy 在 [-32, 32] 物理范围 ✓
  - (b) σ_E ∝ 1/√N: ratio(500/2000)=1.999, ratio(2000/8000)=1.978，与 √4=2 吻合到 1% ✓
  - (c) 同一 θ 5 次 evaluate 的 std=0.0098（≈ σ_E 量级），证明 VMC RNG 正确 reseeding ✓

### 关键数字

| 量 | 值 | 来源 |
|---|---|---|
| D（4×4 RBM α=2 实参数维度） | 1120 | `2·(N+M+NM) = 2·(16+32+512)` |
| Hilbert dim（S_z=0 sector） | 12870 | `C(16, 8)` |
| **E_0 (J2=0.5, our ED)** | **-8.4579** | `results/ed_reference_j1j2_4x4_J2=0.50.json` (verified, lit-attribution corrected) |
| **E_0/N (J2=0.5, our ED)** | **-0.5286** | 同上；Protocol/Prediction v2/ed_reference docstring 已同步修正（原 "-0.497" 为 LLM thermodynamic-limit attribution 错误） |
| gap (J2=0.5) | 0.3189 | 同上 |
| E_0/N (J2=0.0 sanity) | **-0.7018** | matches Sandvik benchmark -0.7017 ✓ |
| E_0/N (J2=0.6 V-shape peak) | -0.5259 | most frustrated region |
| gap (J2=0.7 critical) | 0.0858 | classic frustrated minimum |
| pytest 16 tests | 全 pass in 6:35 | `tests/test_j1j2_problem.py` |
| autograd 开销 | +57% (4:12 → 6:35) | 修复 detach 后预期范围内 |
| ‖grad‖_∞ at random init | 0.666 | `test_gradient_shape` |
| ‖grad‖_2 per row (B=2) | [0.95, 1.51] | 同上，O(1) 健康 |
| σ_E (n=500) at random init | 0.008129 | tmp_check (b) |
| σ_E (n=2000) | 0.004066 | tmp_check (b) |
| σ_E (n=8000) | 0.002056 | tmp_check (b) |
| σ_E sqrt-scaling 500/2000 | **1.999** | 解析 = 2.0，匹配到 0.05% |
| σ_E sqrt-scaling 2000/8000 | **1.978** | 同上 |
| 5 calls stochasticity std | 0.00979 | tmp_check (c)，> 1e-4 阈值 ✓ |
| Random init mean E (4×4 J2=0.5) | 11.99 | tmp_check (d) |
| L_basin = init - E_0 | **20.45** | 比 Prediction v2 §1.6 估计 ≈ 8 大 2.5× |

### 发现/问题

1. **(子任务 A) `_log_cosh_complex` 重构为 stable form C2 + Option β API**
   - 旧实现暴露 zero-gradient 污染（clamp）+ overflow（|u|>89 float32），
     autograd 通过它会破坏 NQS 训练 correctness
   - 新形式数值稳定 + autograd 处处可微 + 跟 NetKet/jVMC convention 一致
   - **影响**: 此前 Day 4-5 baseline 跑起来，weights 涨到 |W·σ|≈5-10 时会出现训练发散；现在 robust

2. **(子任务 B) Latent gradient autograd bug 修复**
   - **重要发现**: Day 0 骨架交付的 `J1J2Problem.gradient()` 实际**完全不能用**
     ——`evaluate()` 用 `torch.zeros + cost[b] = e_val` in-place 切断 grad，
     `_evaluate_vmc/_evaluate_exact` 又显式 `.detach()` 双重断链
   - 如果 Day 1 不修，Day 4 P0-1.2 启动 TCBM-gradient baseline 立刻崩溃
   - 修复 + 添加 partial-gradient docstring 说明 Week 1 用 fixed-sample 路径，
     Week 2 P0-2.1 之后切换到 log-derivative estimator（Bukov 2021 / NetKet）

3. **(子任务 C) ED reference 与 Protocol v2.1 原引用值不符 — 已验证 + 修正完成**
   - 实测：4×4 PBC J2/J1=0.5 → **E_0/N = -0.5286**（first-principles ED）
   - 旧协议引用：-0.497（错误 attributed 到 Schulz 1996, Sandvik 2007）
   - 偏差 6.4% > 测量噪声
   - **Sanity 1**: J2=0 → E_0/N = -0.7018，与 well-known benchmark -0.7017 匹配到 4 位小数 ✓
   - **Sanity 2**: J2 scan 给经典 V 形 + frustrated region gap 塌陷（J2=0.7 gap=0.086）✓
   - **Sanity 3**: N=4 (2×2 PBC) ED matrix cross-check 通过到 1.6e-7（float32 精度下限）✓
   - **Nick 文献结论**: Schulz-Ziman-Poilblanc 1996 自己 explicitly warns "the 16 site
     cluster shows anomalous finite size effects" in $J_2/J_1 \in [0.3, 0.7]$。
     Darradi et al. (arXiv:0806.3825) Fig 2 N=32 ED 也低于 CCM thermodynamic limit。
     N=16 finite-size effect 比 N=32 更强，所以 E/N 应当更负——我们的 -0.5286 完全合理。
     "$-0.497$" 是 LLM-attribution 错误，原数字属于 thermodynamic-limit extrapolation
     而非 4×4 PBC finite-size 真值。
   - **修正动作（已完成）**: Protocol v2.1 §1.1 reference 段、Prediction v2 §1.2 表 E_0 行 +
     R-abort-1 / P0-1.2 rebase note、ed_reference.py 顶 docstring 全部更新到 -8.4579
     / -0.5286，并加 explanatory note 解释为何从 -0.497 改为 -0.5286（避免未来 git
     blame 困惑）

4. **(子任务 C bonus) L_basin ≈ 20.4，不是预测的 ≈ 8**
   - Random init E ≈ 12.0 的解析推导（见报告）：uniform ψ 下 E_var = sum_all_H_entries/dim
     = (32·J1 + 32·J2)·2^(N-2)/2^N = 32·(1+0.5)/4 = **12.0**（实测 11.99 ✓）
   - Prediction v2 §1.6 写"L_basin ≈ 8"基于直觉 `<H>_uniform = trace/dim = 0`，
     但 VMC 实际算的不是 trace 而是 sum-of-all-entries
   - **影响**: Day 4 baseline 的 burn-in 阶段会比预测的长（要 close 20+ 而非 8 的能量差）；
     不影响 10% 误差判据，但 step_size/n_steps 应该按这个 scale 检查

   **[Day 1 末尾追加 — Nick's spot-check, Day 2 进入前]**:
   L_basin ≈ 20 的实测数字可信（Random init E ≈ 12 是直接 evaluate 出来的）。但
   "sum-of-all-H-entries/dim = 12" 这个理论推导跳了几步：H 是 traceless（对角项
   ∑_σ H_σσ = 0，因为 σ_i σ_j 在四种 (±,±) 上正负抵消），非对角项的精确 sum 涉及
   binomial coefficients (∑_bond J · 2 · C(N-2, N/2-1))，不是简单的 1/4 因子。
   实测数字 12 是从 evaluate(theta) under uniform-ish init 直接得到的，可信用作
   Day 4-5 baseline 估计。但 Week 3 SI 写作时**这个推导段不能直接抄**，需要重做
   解析或改为纯 empirical 表述（"observed L_basin ≈ 20 from N runs at random init"）。

### 红线状态

- **R-abort-1** (Week 1 end |E_TCBM - E_0|/|E_0| > 15%): pending Day 4-5 baseline
- **R-abort-2** (Week 2 end QGT > 90s/step): pending Week 2
- **R-abort-3** (Week 3 σ_TCBM/σ_Adam > 0.8): pending Week 3 robustness sweep
- **R-abort-4** (Day 7 ψ < 1.1, T4a RED): pending Day 7 T4a gate

### 明日 (Day 2)

- Day 1 大幅超出原计划（包含 Day 3 的 ED reference + Day 4 准备工作的 partial gradient
  fix），可以直接进 Day 4 P0-1.2 的 TCBM-gradient baseline 第一次 end-to-end run
- 关键 cfg 模板（Protocol §1.2）:
  ```python
  cfg = TCBMConfig(M=12, n_steps=3000, k=20,
                   T_min=0.005, T_max=2.0, T_min_floor=0.002,
                   lambda_min=0.05, lambda_max=2.0, tau_lambda=500,
                   subspace_warmup=150, subspace_update_freq=80,
                   psi_star=1.5, min_warmup=120,
                   grad_buffer_size=400, grad_buffer_use=200,
                   subspace_source='gradient',
                   n_vmc_samples=2000, n_vmc_samples_final=4,
                   noise_temperature_alpha=1.0)
  ```
- Day 4 关注点:
  - swap_acceptance ∈ [0.15, 0.35]
  - best_cost_debiased < best_cost_raw（NQS-1e 去偏 sanity）
  - 收敛目标: |E_TCBM - (-8.4579)| / 8.4579 < 10%（rebase 后的判据）
  - L_basin ≈ 20 意味着 burn-in 较长，关注前 ~500 步是否有明显下降

### Day 1 反思

- LLM-hallucinated reference value 在协议级文档里造成 6.4% 量级的判据偏差，被三层
  sanity（J2=0 benchmark + V-shape + N=4 cross-check）独立 catch 住。教训：所有
  quantitative reference values 必须在 Day 1 first-principles 验证，不能信赖 LLM
  prior knowledge。后续每个 P0 判据数值都应该有 verification artifact。
- Day 0 骨架的 `gradient()` 在静态测试下看似 ok，但在 backward 时直接崩。教训：
  unit test 必须实际跑反向传播，光检查 forward shape 不够。子任务 B 的 pytest
  集成是 Day 4 baseline 不会崩的保险栓。

---

## Day 2 (2026-04-28)

### 计划
- [x] Step 0a: σ_E ratio re-verification (raw stdout, no rounding)
- [x] Step 0b: 加 L_basin theoretical-derivation caveat 到 Day 1 log
- [x] Step 1: Reference code orientation (vmc_jax + netket_reference)
- [x] Step 2: Bukov 2021 anchor 提取 + 修正 Prediction v2 三处 hallucination
- [x] Step 3: 写 `experiments/run_gradient_baseline.py` (production)
- [x] Step 4: nvidia-smi sanity (GPU 0 占用 → 切到 GPU 3)
- [⏰] Step 5: 启动 P0-1.2 baseline run — **6h22min timeout, SIGTERM, no verdict**
- [—] Step 6: verdict + Day 2 收尾 (无法判 PASS/MARGINAL/R-ABORT，inconclusive)
- [x] (forward work) `experiments/run_gradient_baseline_smaller.py` 写了不跑
- [x] (forward work) `docs/known_issues_day2.md` 记录 3 issues
- [x] Day 2 收尾: GPU 清理 verify + commit + daily_log

### 实际完成

**Step 0a (σ_E ratio re-verification)**:
- tmux 内重跑 `tmp_sigma_check.py`（`torch.manual_seed(0)`, J2=0.5 4×4, 3 个 sample sizes）
- raw stdout（精度 8 d.p.）：
  - n=500: σ_E = 0.00955384
  - n=2000: σ_E = 0.00474325
  - n=8000: σ_E = 0.00236895
- ratio 500/2000 = **2.014196**, ratio 2000/8000 = **2.002260**（理论 = 2.000）
- Day 1 ratios 1.999/1.978 也是真实数据（不同 RNG state，统计涨落 0.7-2% 内）
- commit `10d4981`: "Day 2: add L_basin theoretical-derivation caveat to Day 1 log"

**Step 0b (L_basin caveat)**:
- 加在 Day 1 §section 4 "L_basin bonus" 末尾
- 指出 "sum-of-all-H-entries/dim = 12" 推导跳步（H traceless 但非对角和涉及 binomial coeff），
  实测 12 可信但 Week 3 SI 不能直接抄推导

**Step 1 (Reference code orientation)**:
- vmc_jax 索引建立：`jVMC/util/{tdvp,minsr}.py` (SR/NaturalGradient)，`jVMC/operator/{base,branch_free}.py`，
  无专用 J1J2 example（broader keyword grep `frustrated|j2|J_2|nnn` 也零命中），Bukov 2021 reproducer 不在此 mirror
- netket_reference 索引：**`Examples/HeisenbergJ1J2/heisenbergJ1J2.py` is 2D square** with
  `nk.graph.Square(L, max_neighbor_order=2)`, `nk.operator.Heisenberg(... J=[1.0, 0.5])`, `total_sz=0`
  → 直接 fork target for Day 8 SR baseline (只需 L=10 → L=4)

**Step 2 (Bukov 2021 anchor + 3 hallucination fixes)**:
- arxiv:2011.11214 PDF fetch 成功，`docs/bukov_2021_anchor.md` 写入 verbatim quotes
- 修正 Prediction v2 三处 LLM hallucination（commit `07e005e`，amend 后 hash）：
  1. §2.2: "Bukov ΔE_hump/N ≈ 0.05-0.10" → 实际 paper 不做 barrier interpolation 分析
  2. §2.3: "λ_min/λ_max ≈ 0.2-0.3" → Fig 11 不显式给 ratio；4×4 ~10^-2，6×6 ~10^2
  3. §5.4/§5.6: "σ(E)/|E| ≈ 5-10%" → Fig 12 spread ~10^-3 to 10^-2 → ~0.5-2% on N=6×6
- §2.4 T2 ratio 主体改 conservative/optimistic 双边界 (T2 = 20 reference / 160 sensitivity)
- §6.2 / §8.1 / end-of-doc Stage V deliverable 同步 sync (single source of truth principle)
- 2 文件改动: 172 inserts / 35 deletes

**Step 3 (production script)**:
- `experiments/run_gradient_baseline.py` 写完, 165 行
- TCBMConfig 字段全部 verify（`subspace_source`, `n_vmc_samples`, `n_vmc_samples_final`, etc.）
- result dict key verify (`T_w` 不是 `T_w_step`, `trajectory.swap_acceptance` 是 per-record list 而非 mean)
- 加 swap_acceptance early/mid/late 分段诊断 + LCM(10,15)=30 stale-value caveat
- 加 NQS-1e debiasing sanity check (`debiased < raw`)
- save_data 改 full trajectory dump (~300 records × 3 fields ≈ 7 KB)

**Step 4 (GPU sanity)**:
- GPU 0 free 仅 5.6 GB（别人 17.4 GB qwen9b training 占用）
- GPU 3 free 22.7 GB, util 17% → switched to `device='cuda:3'`
- Verify J1J2Problem + TCBMOptimizer 内 device 全部 thread `self.device`，无 hardcode

**Step 5 (baseline run, FAILED)**:
- 09:06 launch, expected 30 min, **6h22min timeout @ 15:28 SIGTERM**
- stdout log = 0 bytes (block-buffered through tee + Python -u not applied)
- JSON output = 不存在 (atexit handler 没注册 SIGTERM trigger)
- Heartbeat (90s) 完整 4h coverage（+13 min initial gap due to path bug at launch）
- 抢救出 5 个 phase 的 wall-clock 结构（见关键数字段）

### 关键数字

| 量 | 值 | 来源 |
|---|---|---|
| σ_E ratio 500/2000 (Day 2 redo) | **2.014196** | Step 0a raw stdout |
| σ_E ratio 2000/8000 (Day 2 redo) | **2.002260** | Step 0a raw stdout |
| Bukov 2021 anchor σ(E)/|E| (6×6) | 0.5%-2% (Fig 12) | not 5-10% as in old draft |
| Bukov 2021 anchor Hessian λ ratio | 4×4: ~10^-2 / 6×6: ~10^2 | Fig 11，无显式 ratio |
| Bukov 2021 NMC | 2^15 = 32768 samples/iter | Section 6.1 |
| Production script 行数 | 165 | run_gradient_baseline.py |
| GPU 3 baseline mem (Phase 1) | 3415 MB stable 4h38min | heartbeat |
| GPU 3 baseline mem (Phase 2-5) | **22064 MB** stable 1h45min | NQS-1e final eval |
| GPU 3 util (Phase 1) | 86-99% sustained | heartbeat |
| GPU 3 util (Phase 2-5) | 8-100% volatile (3 次 dip-ramp) | heartbeat |
| Baseline wall-clock | **6h22min** vs 估算 30 min | **6.5× underestimate** |
| stdout output | **0 bytes** | block-buffered, no flush |
| JSON output | **不存在** | atexit 不在 SIGTERM 触发 |
| Days 2 commits | 4 个 (10d4981, 07e005e, b8908c7, +daily_log) | + 1 amend |

### 发现/问题

1. **VMC evaluate cost 严重低估 (~6× wall-clock)**
   - Day 1 single-evaluate timing 推算 0.6s/step，实测 ~7s/step
   - Root cause hypothesis: `J1J2Problem.evaluate()` per-replica sequential，未 batched over M=12
   - Implication: Day 3 retry n_vmc_samples=4000 → 12h, 不可行；Week 3 sweep 15 seeds × 4 methods → 250-300h
   - **Fix path** (Day 8-9 scope): batch `_evaluate_vmc` over M dimension, 估计 5-8× speedup
   - Validated: Bukov 2021 用 NMC=2^15 (16× larger) on multi-day timeframes，我们 4×4 慢是"expected"

2. **Python SIGTERM 默认不触发 atexit, stdout block-buffered through tee**
   - Production script 没注册 `signal.signal(SIGTERM, ...)` 也没 `python -u`
   - 6h22min 计算结果 → 0 bytes 数据。**100% 数据丢失**
   - Day 3 fix: signal handler + `sys.stdout.reconfigure(line_buffering=True)` + atexit partial JSON dump

3. **Bukov 2021 三处 fabricated 数字成功定位 + 修正**
   - 在 Prediction v2 全文 propagate 7 处（主推导 2 处 + deliverable block 2 处 + summary table 1 处 + end-of-doc 2 处）
   - 单 source of truth (§2.4) 改后下游 4 处全部 sync，避免 incremental edit drift
   - 文件: `docs/bukov_2021_anchor.md` (110 行 verbatim quotes 锚定)

4. **GPU 3 ptrace_scope=1 + py-spy 装上但拿不到 stack frame**
   - 疑似 hang 时无法 inspection stack（需 sudo or setcap）
   - Workaround: `/proc/PID/task/*/wchan` 查 deadlock pattern (tested ✓)
   - Day 3+ scope: 提前 register `faulthandler.dump_traceback_later(N=900)` 自动每 15min dump

5. **NetKet J1J2 example is 2D square (good news for Day 8)**
   - `Examples/HeisenbergJ1J2/heisenbergJ1J2.py` uses `nk.graph.Square(L, max_neighbor_order=2)`
   - 默认 L=10 (10×10), 改 L=4 即可，其他 SR/preconditioner/solver API 直接借用

### 红线状态

- **R-abort-1** (Week 1 end |E_TCBM - E_0|/|E_0| > 15%): **inconclusive — no verdict data**
  - Day 3 重跑 production v2 (with robustness fixes) 才能判定
- **R-abort-2** (Week 2 end QGT > 90s/step): pending Week 2
- **R-abort-3** (Week 3 σ_TCBM/σ_Adam > 0.8): pending Week 3
- **R-abort-4** (Day 7 ψ < 1.1, T4a RED): pending Day 7

### 明日 (Day 3)

**Plan: production retry with full robustness (option D in 裁决 matrix)**

- 09:00-10:00 **Step A**: 改 `core/tcbm_optimizer_NQS.py` 加 `callback: Optional[Callable[[int, dict], None]]` 参数到 `optimize()`，每 100 步触发；
  pytest 验证 callback 真的被调用 + 不破 Day 1 16 tests
- 10:00-11:00 **Step B**: 写 `experiments/run_gradient_baseline_v2.py`：
  - `sys.stdout.reconfigure(line_buffering=True)`
  - `signal.signal(SIGTERM, _sigterm_handler)`
  - `atexit.register(_atexit_dump)` partial JSON
  - cfg: `n_vmc_samples_final = 4 → 2`（final eval ~30% wall reduction，σ_E ↑ 30%，对 verdict 无影响）
- 11:00-11:15 **Step C**: 写 `experiments/benchmark_per_step.py` (n_steps=100)，测真实 per-step wall:
  - projected total < 4h → 进 Step D
  - 4-8h → 评估 trade-off，等 Nick 裁决
  - > 8h → 提前做 batched evaluate refactor
- 11:15+ **Step D**: production launch with `python -u`，预期 3-5h based on benchmark
- **Step E** monitoring: 每 30 min check log file 有无新 progress lines (script will print every 100 steps)
- **Step F** verdict: PASS / MARGINAL / R-ABORT 按原阈值（10% / 15%）

**GPU 选择 caveat**: GPU 3 现已被 dglg 另一个 baselines/train_compare.py 占；Day 3 启动前重新 nvidia-smi 选最 free 的 GPU。

### Day 2 反思

1. **Wall-clock 估算系统性低估 6×**——Day 1 单次 evaluate timing 不能直接外推到 3000 步 + M=12 replicas + autograd 重复构图。下次估算前必须先做 micro-benchmark (100 步实测)，不能信任单次 evaluate 时间外推。

2. **Python stdout buffering + SIGTERM atexit 是单点故障**——production-grade 长任务必须三件套：`python -u` (unbuffered) + `signal.signal(SIGTERM, ...)` (graceful) + `atexit.register(dump)` (partial JSON)。Day 3 v2 全部加上。

3. **Hard checkpoint 应主动 ping，不只等 Monitor trigger**——Day 2 的 14:30 / 14:40 / 15:00 三个 cap 我都是 passively 等。Nick 反馈 calibrated：被动等 Monitor 是合理 LLM default，但用户 spec 写"下次行动 14:30"那种就是 hard checkpoint，必须 active polling。

4. **Bukov hallucination 发现链值得 paper 写作时引用**——LLM (Claude) prior knowledge 给的 Bukov 数字三处都错（5-10% σ, 0.05-0.10 hump, 0.2-0.3 λ ratio），但通过 web fetch 原 PDF 全部纠正。这暴露 LLM-only 文献综述的 systematic risk，下次不能再依赖 prior knowledge，必须 fetch 原 source。

5. **Forward work during long wait 是 productive**——Day 2 6h22min 等待期写了 `run_gradient_baseline_smaller.py` fallback + `known_issues_day2.md`，没浪费完全。下次 long wait 也按这个模式。

