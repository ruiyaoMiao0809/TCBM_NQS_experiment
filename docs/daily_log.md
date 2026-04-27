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
