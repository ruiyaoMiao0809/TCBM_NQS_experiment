# TCBM-NQS Proof-of-Concept 实验协议 v2.1

**版本**: v2.1（集成 T4a 精化）
**日期**: 2026-04-24
**负责人**: Nick (PhD, HKUST-GZ, Xiong Lab)
**工期**: 3 周（21 天）

**基于**:
- v2.0: `TCBM_NQS_POC_Protocol_v2.md`（2D J1-J2 4×4 任务，GREEN 判定）
- T4a 精化: `T4a_early_abort_supplement.md`（2026-04-24 user 补充）
- Prediction: `NQS_J1J2_Prediction_v1.md`
- 方法论: `TCBM_Task_Applicability_Methodology_v1.md` v1.1（含 T4a/T4b 分解）

**v2.1 相对 v2.0 的变化**:
- **新增 R-abort-4**: T4a early-abort 红线（Day 7 的 15 min GPU diagnostic）
- **新增 P0-1.4**: Day 7 上午 T4a early diagnostic 任务
- **日程调整**: Week 1 Day 7 重构，SR baseline 移至 Day 8
- **成功判据**: Week 1 增加 T4a PASS 判据
- **风险矩阵**: 新增 T4a RED 风险（5% 概率，影响 Plan B 触发）
- **Plan B 降级**: 新增 T4a RED 情况下的 SI wording

---

## 0. 协议总体结构

### 0.1 POC 范围声明

本 POC 追求三件事:

1. **可行性验证**: TCBM 可在 NQS 框架中运行，在 4×4 J1-J2 at $J_2/J_1=0.5$ 收敛到 DMRG reference 能量（误差 < 10%）
2. **机制区分证据**: 通过 gradient / qgt / hybrid 三种 subspace source 对比，证明 TCBM clamping 提供 natural-gradient metric 不包含的 directional information
3. **Robustness 证据**: 多 seed 下 TCBM 相对 Adam baseline 的 final-energy 方差显著更小（目标 $\sigma_\text{TCBM}/\sigma_\text{Adam} \leq 0.5$）

**不追求**: 打败 Chen-Heyl 2024 MinSR 的 SOTA 能量。

### 0.2 工作日历（v2.1 revised）

| 周 | 日期 | 主题 | 主要交付 |
|---|---|---|---|
| Week 1 | Day 1-7 | 基础设施 + gradient baseline + **T4a 诊断** | J1J2Problem + TCBM-gradient < 10% 误差 + **T4a PASS** |
| Week 2 | Day 8-14 | QGT + hybrid 模式 | 三种 subspace source 对比数据 |
| Week 3 | Day 15-21 | Robustness study + paper SI | 15 seed 对比 + SI 草稿 |

### 0.3 优先级定义

**P0**: 必须完成。未完成则 POC 失败
**P1**: 强烈推荐。完成后 SI 质量显著提升
**P2**: 加分项

### 0.4 全局红线（Abort Conditions）

- **R-abort-1**: Week 1 结束时 TCBM-gradient 相对误差 > 15%
- **R-abort-2**: Week 2 结束时 QGT 计算单步耗时 > 90 秒
- **R-abort-3**: Week 3 seed variance 显示 $\sigma_\text{TCBM}/\sigma_\text{Adam} > 0.8$
- **R-abort-4 (NEW, T4a gate)**: Week 1 Day 7 ψ(t) diagnostic 显示累计最大 ψ < 1.1 直到 step 1000
  - 含义: Gradient buffer 无低秩结构，TCBM 的 clamping 机制无 substrate
  - 对应方法论 v1.1 的 T4a RED 判定
  - **这是最早的 abort gate**，比 R-abort-1/2/3 都早

红线触发时按 Plan B 重写 SI（见 §4.2）。

### 0.5 T4a/T4b 分解（新）

原 T4 条件精化为:
- **T4a**（structure existence）: $\psi(t) = \sigma_k/\sigma_{k+1} > 1.5$ 在 warmup 结束前达到
- **T4b**（structure correctness）: $|\cos(U_{\text{SVD}}, v_{\text{escape}})| > 0.3$

T4a 是 T4b 的前提。方法论 §9.3 ResNet-56-NS 案例证明 T4a FAIL 可以**直接在 warmup 阶段检测到**，无需完整实验。本 POC 在 Day 7 用 15 分钟 GPU 做 T4a 诊断。

---

## 1. Week 1: 基础设施 + Gradient Baseline + T4a 诊断

### 1.1 目标系统

**2D spin-1/2 J1-J2 Heisenberg model on 4×4 square lattice, PBC**

$$
\hat H = J_1 \sum_{\langle i,j \rangle} \hat{\vec S}_i \cdot \hat{\vec S}_j + J_2 \sum_{\langle\langle i,j \rangle\rangle} \hat{\vec S}_i \cdot \hat{\vec S}_j
$$

**参数**: Lattice 4×4 PBC, $J_1 = 1, J_2 = 0.5$（maximally frustrated point）, Hilbert dim $= 2^{16} = 65536$

**Reference ground state energy**:
- Source: Schulz-Ziman-Poilblanc 1996 + Sandvik 2007
- Value: $E_0/N \approx -0.497$（4×4 PBC）
- Day 3 自行验证

### 1.2 P0 任务清单

#### P0-1.1: 构建 J1J2Problem 类（Day 1-3）

见 `src/j1j2_problem.py`。

接口契约:
```python
class J1J2Problem:
    dim: int                  # D = 1120
    device: str
    def random_feasible(self, batch_size) -> (B, D)
    def evaluate(self, x, n_samples=None) -> (cost, penalty, cost_std)
    def gradient(self, positions) -> (B, D)
    def qgt(self, x, k_qgt=None) -> (D, D)    # Day 9 delivery
```

**成功判据**:
- 7 个 unit tests 全通过（见骨架的 test_* 函数）
- uniform-random $\theta$ 下 energy 在 $[-2N, 2N]$ 物理范围
- $\sigma_E$ 随 $\sqrt{n_\text{samples}}$ 下降（采样正确性）

#### P0-1.2: 对接 TCBM-gradient 模式（Day 4-5）

```python
cfg = TCBMConfig(
    M=12, n_steps=3000, k=20,
    T_min=0.005, T_max=2.0, T_min_floor=0.002,
    lambda_min=0.05, lambda_max=2.0, tau_lambda=500,
    subspace_warmup=150, subspace_update_freq=80,
    psi_star=1.5, min_warmup=120,
    grad_buffer_size=400, grad_buffer_use=200,
    subspace_source='gradient',
    n_vmc_samples=2000, n_vmc_samples_final=4,
    noise_temperature_alpha=1.0,
)
```

**检查项**:
- 3000 步跑完不崩
- `best_cost_debiased < best_cost_raw`（NQS-1e 去偏工作）
- `sigma_E` trajectory 下降
- `swap_acceptance` 在 [0.15, 0.35]

#### P0-1.3: ED 对照（Day 5-6）

scipy sparse Lanczos 在 $S_z=0$ sector 计算 $E_0$。

**成功判据**（P0）:
$$
\frac{|E_\text{TCBM} - E_0^\text{ED}|}{|E_0^\text{ED}|} < 0.10
$$

#### P0-1.4: T4a Early Diagnostic（Day 7 上午，NEW）

**目标**: 验证 gradient buffer 的 SVD 谱有明显 gap，即 TCBM 的 clamping 机制有"可学的 structure"。

**实现** (见 `experiments/run_t4a_diagnostic.py`):

```python
from core.tcbm_optimizer_NQS import TCBMOptimizer, TCBMConfig
from core.j1j2_problem import J1J2Problem

problem = J1J2Problem(Lx=4, Ly=4, J1=1.0, J2=0.5, alpha=2, device='cuda')

cfg_diag = TCBMConfig(
    M=12, n_steps=1000,              # 只跑 1000 步
    k=20,
    subspace_warmup=150,
    subspace_update_freq=50,         # 频繁 SVD 观察 ψ 演化
    psi_star=float('inf'),           # 不 trigger T_w, ψ 自由演化
    min_warmup=10**9,
    subspace_source='gradient',
    n_vmc_samples=1500,
    record_every=10,
)
opt = TCBMOptimizer(problem, cfg_diag)
result = opt.optimize()

psi_history = result['trajectory']['psi']
psi_cum_max = max(psi_history) if psi_history else 0.0
```

**判定**:
| ψ at step 1000 | 判定 | 行动 |
|---|---|---|
| ≥ 1.5 | **GREEN** | 继续 Week 2 计划 |
| 1.1-1.5 | **YELLOW** | 下午调参 retry |
| < 1.1 | **RED** | 触发 R-abort-4, 切 Plan B |

**YELLOW retry 调参优先级**:
1. `grad_buffer_size: 400 → 600`, `subspace_warmup: 150 → 300`
2. `k: 20 → 12`（更接近 $\dim \mathcal{H}_{A_1} \approx 9$）
3. 换 GCNN ansatz（最后手段，Week 2 工程量大）

**预期概率分布**（Prediction v1 §2.5）:
- GREEN: 70%
- YELLOW → GREEN after retry: 20%
- YELLOW → RED: 5%
- RED: 5%

**GPU cost**: 1 次 diagnostic 约 15 分钟（1000 步 × 1500 samples on A10）

### 1.3 P1 任务清单

#### P1-1.1: Adam baseline（Day 6）

见 `baselines/adam_baseline.py`. 同一 `J1J2Problem`，标准 Adam optimizer, lr=0.001, 3000 steps。

#### P1-1.2: SR baseline（Day 8, 原 Day 7）

**Note**: v2.1 将 SR baseline 推迟到 Day 8，以腾出 Day 7 给 T4a 诊断。如果 T4a GREEN 早上就判定，Day 7 下午可以开始 SR。

课本版 Stochastic Reconfiguration. 对 D=1120 scale, $(D,D)$ full QGT + 直接 solve 可行。

### 1.4 Week 1 日程（v2.1 revised）

| Day | 上午 | 下午 | 交付 |
|---|---|---|---|
| 1 | J1J2Problem: lattice + RBM + log_psi | Metropolis sampler | Partial class |
| 2 | VMC local energy computation | Gradient via autograd | Class complete |
| 3 | Unit tests + ED reference script | Run ED, get $E_0$ | All tests pass + $E_0$ |
| 4 | TCBM-gradient baseline run 1 | Analyze trajectory | First run data |
| 5 | TCBM-gradient + tuning | Error < 10% | P0-1.2 PASS |
| 6 | Adam baseline (P1-1.1) | Cross-check vs TCBM | Baselines comparable |
| **7** | **P0-1.4: T4a diagnostic** | **Judge: GREEN/YELLOW/RED** | **T4a gate** |

**Day 7 详细**:
- **09:00-09:30**: 启动 T4a diagnostic run（15 min GPU）
- **09:30-11:00**: 分析 ψ(t) 曲线，判定 GREEN/YELLOW/RED
- **11:00-12:00**:
  - GREEN → 准备 SR baseline 代码（为 Day 8 做好）
  - YELLOW → 调参 retry（`grad_buffer_size` 先调大）
  - RED → 触发 R-abort-4, 组织 Plan B
- **13:00-17:00**:
  - GREEN → 继续 Day 6 遗留 + Day 8 预备
  - YELLOW → 做 3 次 retry（不同参数组合），综合判定
  - RED → Plan B 文档准备

### 1.5 Week 1 成功判据（v2.1 revised）

| 判据 | 级别 | 通过标志 |
|---|---|---|
| J1J2Problem 实现完成 | P0 | 7 unit tests pass |
| ED reference 验证 | P0 | $E_0$ 与 Schulz 1996 一致 |
| TCBM-gradient 收敛 < 10% | P0 | error < 0.10 |
| **T4a PASS (NEW)** | **P0** | **ψ ≥ 1.5 by step 1000** |
| Adam baseline 跑通 | P1 | 收敛到任意能量 |
| SR baseline 跑通 | P1 | QGT 数值稳定 |

### 1.6 Week 1 失败分岔

**分岔 A (原)**: TCBM-gradient 误差 > 15% → 诊断 3 步，最后触发 R-abort-1

**分岔 B (原)**: $\sigma_E$ > 0.5 → 增加 thermalization

**分岔 C (原)**: OOM/数值崩溃 → 降 M 到 8 debug

**分岔 D (NEW, T4a)**: ψ 在 Day 7 始终 < 1.1
- 立即行动: 切 Plan B, 不要进入 Week 2
- 详细见 §4.2 Plan B 的 T4a RED wording

---

## 2. Week 2: QGT + Hybrid 模式

（未修改，保持 v2.0 内容）

### 2.1 核心目标

验证 NQS-3 假说: TCBM clamping subspace 与 natural-gradient metric (QGT) 不完全重叠，在 J1-J2 上提供 SR 缺失的 directional information。

### 2.2 P0 任务清单

#### P0-2.1: 实现 problem.qgt() 接口（Day 8-9）

QGT 定义:
$$
S_{ij}(\theta) = \text{Re}\left[\langle O_i^* O_j \rangle_{|\psi|^2} - \langle O_i^* \rangle \langle O_j \rangle \right]
$$

实现 via `torch.func.vmap` + `torch.func.grad`。见 `src/j1j2_problem.py` 的 `qgt()` 方法（Day 9 完成）。

**成功判据**:
- 对称: $\|S - S^T\|_F / \|S\|_F < 10^{-4}$
- 半正定: $\lambda_\min > -10^{-4}$
- 低秩结构: 前 30 个特征值显著

#### P0-2.2: subspace_source='qgt' 模式（Day 10）

```python
cfg.subspace_source = 'qgt'
cfg.k_qgt = 40
```

观察 `qgt_rank_effective`, best_cost。

#### P0-2.3: subspace_source='hybrid' 模式（Day 11-12）

```python
cfg.subspace_source = 'hybrid'
```

**核心实验**. 观察 `n_gradient_directions_kept`。预期 5-15 范围（2D J1-J2 non-stoquastic 的 gradient 方向不被 QGT 完全覆盖）。

#### P0-2.4: 三模式对比主图（Day 13-14）

5 seeds 的 mean±std 能量收敛曲线。

**成功判据**:
- 三条曲线均收敛到 <10% 误差
- `n_gradient_directions_kept` ≥ 5
- 如果 hybrid 模式 final error < qgt 模式 → 强 mechanism claim

### 2.3 P1 任务

- P1-2.1: QGT 特征谱图（Day 14）
- P1-2.2: Subspace overlap 矩阵
- P1-2.3: T4b alignment 实测（可用 T4a 诊断中记录的 `U_SVD` + 后验 `v_escape`）

### 2.4 Week 2 成功判据

| 判据 | 级别 | 通过标志 |
|---|---|---|
| QGT 对称半正定 | P0 | 测试通过 |
| qgt 模式收敛 <10% | P0 | error < 0.10 |
| hybrid 模式收敛 <10% | P0 | error < 0.10 + n_kept ≥ 5 |
| 三模式对比主图 | P0 | 5-seed mean±std |
| QGT 特征谱 | P1 | 低秩结构可视化 |
| Overlap 表 | P1 | 三对数字 |
| T4b 实测 | P1 | \|cos\| ∈ [0.3, 0.7] |

### 2.5 Week 2 失败分岔

- 分岔 A: QGT 超时 > 90 秒 → R-abort-2, 降 ansatz
- 分岔 B: $n_\text{kept} = 0$ → 降 k_qgt 或升 system size
- 分岔 C: qgt 模式发散 → 加 Tikhonov

---

## 3. Week 3: Robustness Study + SI 草稿

### 3.1 核心目标

证明 TCBM 相对 Adam/SR baseline 的 seed-variance 优势。

### 3.2 P0 任务清单

#### P0-3.1: 实验设计（Day 15）

- 4 methods: Adam / SR / TCBM-gradient / TCBM-hybrid
- 15 seeds
- 同 budget: 3000 steps × 2000 samples
- Total GPU: ~50-70 hrs on A10

#### P0-3.2: 运行（Day 16-18）

`experiments/run_robustness_sweep.py`. 分 3 天，夜间 run。

#### P0-3.3: Metric 计算（Day 19）

Primary claim:
$$
\frac{\sigma(\text{TCBM-gradient})}{\sigma(\text{Adam})} \leq 0.5
$$

Levene's test for variance difference, $p < 0.05$.

#### P0-3.4: Robustness 主图（Day 20）

Box plot, 4 methods × 15 seeds。

### 3.3 P1 任务

- P1-3.1: SI 草稿（Day 20-21）
- P1-3.2: 代码和数据开放

### 3.4 P2 任务

- P2-3.1: $J_2/J_1$ scan（0.3, 0.5, 0.7）
- P2-3.2: 6×6 demo（匹配 Bukov 2021）

### 3.5 Week 3 成功判据

| 判据 | 级别 | 通过标志 |
|---|---|---|
| 15-seed sweep 完成 | P0 | CSV 齐全 |
| TCBM σ 优势 2× | P0 | σ 比 ≤ 0.5, p < 0.05 |
| Box plot | P0 | 可视化 |
| SI 草稿 | P1 | 3-4 页 |
| 代码开放 | P1 | GitHub/Zenodo |

### 3.6 失败分岔

- A: σ 无优势 → R-abort-3, Plan B
- B: 超时 → 降 seeds / steps / samples
- C: hybrid 发散 → 依赖 fallback 逻辑

---

## 4. 风险管理和 Plan B

### 4.1 整体风险矩阵（v2.1）

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| Week 1 Hamiltonian bug | 中 | 高 | 单元测试 + ED cross-check |
| **T4a RED (NEW)** | **5%** | **高** | **Plan B**（早诊断 = 早省时） |
| Week 2 QGT 超时 | 中 | 中 | vmap + 降 ansatz |
| Week 3 robustness 无优势 | 低 | 高 | 3 层诊断 + Plan B |
| 3 周不够 | 中 | 中 | P1/P2 可删 |

**新增风险 5: T4a RED**
- **概率**: 5%（based on Prediction v1 §2.5 + T4a 精化推理）
- **影响**: Day 7 abort, POC 降级到 Plan B
- **缓解**: 无直接缓解（这是 fundamental limitation）
- **收益**: Day 7 abort 比 Week 2 abort 节省 6-7 天（净节省）

**总体风险 vs v2.0**:
- v2.0 的 3 条红线都在实验进行了相当进度后才能触发
- v2.1 的 R-abort-4 在 Day 7（投入仅 7 天）就能触发，**显著降低最坏情况**

### 4.2 Plan B 协议

**触发条件**:
- R-abort-1, 2, 3, **4** 任一触发
- 或 Week 2 三模式无显著差异

**Plan B 内容（根据触发原因）**:

**(a) R-abort-4 触发（T4a RED）**:

SI 标题: *"Theoretical Extension to Neural Quantum States with Diagnostic Preliminaries"*

SI 内容:
- 理论: TCBM NQS 扩展的五项改造（NQS-1a-e）
- 诊断: 报告 T4a 测量结果（ψ < 1.1）作为 informative negative finding
- Discussion wording:
  > "In preliminary warmup diagnostics on Complex-RBM ansatz, the gradient buffer SVD did not exhibit the spectral gap structure (ψ = σ_k/σ_{k+1} < 1.1) required for TCBM's earned subspace clamping to have meaningful impact on this system. This suggests that for this specific ansatz choice, the symmetry-induced subspace structure is below the TCBM-detection threshold. Future work with symmetry-hardcoded ansätze (e.g., GCNN per Roth et al. PRB 2023, or the minimum-SR approach of Chen & Heyl Nat. Phys. 2024) may resolve this diagnostic limitation. The T4a early-abort criterion itself represents a methodological contribution, enabling TCBM applicability assessment in 15 minutes of GPU time rather than full experimental sweeps."

**(b) R-abort-1, 2, 3 触发（原 v2.0 Plan B）**:

SI 标题: *"Theoretical Extension to Neural Quantum States with Preliminary Numerical Validation"*

内容: 单次 demo + 理论讨论. 不做 robustness claim.

**(c) 主文 claim 保持不变**: DC-OPF + MDMF 主 result 独立于本 POC.

---

## 5. 具体文件清单（v2.1 revised）

```
tcbm_nqs/                                # 项目根（见下文 §7 的完整设计）
├── core/
│   ├── tcbm_optimizer_NQS.py           # 已交付
│   ├── j1j2_problem.py                 # Day 1-2 (P0-1.1)
│   └── __init__.py
├── baselines/
│   ├── adam_baseline.py                # Day 6 (P1-1.1)
│   └── sr_baseline.py                  # Day 8 (P1-1.2)
├── utils/
│   ├── ed_reference.py                 # Day 3 (P0-1.3)
│   ├── seed_utils.py
│   └── logging_utils.py
├── experiments/
│   ├── run_t4a_diagnostic.py           # Day 7 AM (P0-1.4, NEW)
│   ├── run_gradient_baseline.py        # Day 4-5 (P0-1.2)
│   ├── run_subspace_compare.py         # Day 10-13 (P0-2.2~4)
│   ├── run_robustness_sweep.py         # Day 16-18 (P0-3.2)
│   └── run_j2j1_scan.py                # Day 20 (P2-3.1)
├── analysis/
│   ├── plot_psi_evolution.py           # T4a 结果可视化
│   ├── plot_convergence.py
│   ├── plot_robustness_box.py
│   ├── compute_subspace_overlap.py
│   └── compute_T4b_alignment.py
├── results/
│   ├── week1_t4a_diagnostic.json
│   ├── week1_gradient_baseline.csv
│   ├── week2_three_modes.csv
│   ├── week3_robustness_sweep.csv
│   └── figures/
├── docs/
│   ├── TCBM_NQS_POC_Protocol_v2_1.md   # 本文档
│   ├── NQS_J1J2_Prediction_v1.md
│   ├── T4a_early_abort_supplement.md
│   └── daily_log.md                    # 每日进展日志
├── tests/
│   ├── test_j1j2_problem.py
│   └── test_tcbm_optimizer_nqs.py
├── README.md
└── environment.yml                     # conda env spec
```

---

## 6. 开工 checklist (Day 0)

开始 Day 1 之前确认:

1. **Server 环境**:
   - [ ] `/home/dglg/miao_workplace/tcbm_nqs/` 已建立
   - [ ] conda env 含: torch >= 2.1, scipy >= 1.10, numpy, pandas, matplotlib, pytest
   - [ ] GPU 访问权限（A10 或更好）

2. **代码到位**:
   - [ ] `core/tcbm_optimizer_NQS.py` 拷贝进 `core/`
   - [ ] `core/j1j2_problem.py` 拷贝进 `core/`
   - [ ] `__init__.py` 文件创建

3. **文档到位**:
   - [ ] `docs/TCBM_NQS_POC_Protocol_v2_1.md`（本文档）
   - [ ] `docs/NQS_J1J2_Prediction_v1.md`
   - [ ] `docs/T4a_early_abort_supplement.md`
   - [ ] `docs/daily_log.md` 创建（空文件）

4. **GPU 预算**:
   - [ ] Week 1: ~20 GPU hrs（含 Day 7 T4a diag 15 min）
   - [ ] Week 2: ~30 GPU hrs
   - [ ] Week 3: ~60 GPU hrs（robustness sweep 主力）
   - [ ] Total: ~110 GPU hrs，A10 上约 5-7 天实际 wall clock

5. **Git**:
   - [ ] `git init` 在 `tcbm_nqs/`
   - [ ] 第一个 commit: "Initial skeleton, Day 0 scaffolding"

---

## 7. 每日 checklist 模板

`docs/daily_log.md` 用此模板:

```markdown
## Day X (YYYY-MM-DD)

### 计划
- [ ] 任务 A (预计 Nh, 对应 P0-x.y)
- [ ] 任务 B (预计 Nh)

### 实际完成
- [x/✗] ...

### 关键数字
- best_cost = ...
- best_cost_debiased = ...
- sigma_E = ...
- swap_acceptance = ...
- psi_cumulative_max = ...   ← Day 7 后必记
- n_kept (hybrid) = ...       ← Week 2 后必记

### 发现/问题
- ...

### 红线状态
- R-abort-1: ... (active/NA/triggered)
- R-abort-2: ...
- R-abort-3: ...
- R-abort-4 (T4a): ...         ← Day 7 前 pending

### 明日
- ...
```

---

## 附录 A: 与 v2.0 的 diff

| 维度 | v2.0 | v2.1 |
|---|---|---|
| 红线数 | 3 | **4**（新增 R-abort-4） |
| Week 1 P0 任务数 | 3 | **4**（新增 P0-1.4 T4a diag） |
| Day 7 内容 | SR baseline | **T4a diagnostic** (上午) + SR 预备 (下午, 或 YELLOW retry) |
| Week 1 成功判据 | 3 个 | **4**（新增 T4a PASS） |
| Plan B wording | 单一 | **3 种触发原因的不同 wording** |
| 风险矩阵 | 4 risks | **5 risks** (新增 T4a RED) |
| T4 概念 | undifferentiated | **T4a (structure existence) + T4b (correctness)** |
| 早期 abort 能力 | 只有 Week 1 end (R-abort-1) | **Day 7 即可 abort (R-abort-4)** |

---

## 附录 B: Day 7 T4a 决策树

```
P0-1.4 Diagnostic @ step 1000
            │
            ▼
     ψ_max at step 1000?
            │
  ┌─────────┼──────────────┐
  │         │              │
< 1.1     1.1-1.5        ≥ 1.5
  │         │              │
  ▼         ▼              ▼
T4a RED  T4a YELLOW     T4a GREEN
  │         │              │
  ▼         ▼              ▼
R-abort-4  下午 retry:   继续 Week 2
(Plan B)   1. buf 400→600
           2. warmup 150→300
           3. k 20→12
            │
            ├── retry 后 ψ ≥ 1.5? → GREEN, 继续
            └── 三次 retry 后仍 < 1.5 → R-abort-4
```

---

## 附录 C: GPU 时间详细估算（v2.1）

| 阶段 | 任务 | 单次时间 | 次数 | 小计 |
|---|---|---|---|---|
| Week 1 | J1J2Problem unit tests | 2 min | 1 | 2 min |
| | TCBM-gradient baseline | 30 min | 3 (tuning) | 1.5 hr |
| | Adam baseline | 25 min | 2 | 50 min |
| | **T4a diagnostic** | **15 min** | **1-3 (retry)** | **15-45 min** |
| | Week 1 total | — | — | **~4 hr GPU** |
| Week 2 | QGT + gradient mode | 40 min | 5 seeds | 3.3 hr |
| | qgt mode | 40 min | 5 seeds | 3.3 hr |
| | hybrid mode | 55 min | 5 seeds | 4.6 hr |
| | SR baseline | 35 min | 3 | 1.8 hr |
| | Week 2 total | — | — | **~13 hr GPU** |
| Week 3 | Adam × 15 seeds | 30 min | 15 | 7.5 hr |
| | SR × 15 seeds | 35 min | 15 | 8.8 hr |
| | TCBM-gradient × 15 | 40 min | 15 | 10 hr |
| | TCBM-hybrid × 15 | 55 min | 15 | 13.8 hr |
| | J2/J1 scan (P2) | 35 min | 15 | 8.8 hr |
| | Week 3 total | — | — | **~49 hr GPU** |
| **总计** | | | | **~66 hr GPU** |

A10 on dedicated node: 约 3-5 个 wall-clock 天即可全部完成（含 overnight）。

---

**文档结束**

> v2.1 核心升级：通过 Day 7 的 15-分钟 T4a diagnostic gate，把"发现 TCBM 不适用"的成本从 2 周降到 7 天，与早 abort 的节省比率约 1:14 (GPU time base) 或 1:2 (wall-clock base)。这是方法论 v1.1 的 T4a/T4b 分解在 POC 层面的直接应用。
