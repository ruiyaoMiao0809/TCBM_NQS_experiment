# NQS J1-J2 (4×4) 任务的 TCBM 适用性数学证明 v3

**版本**: v3.0 (Adam → SR reframe complete)
**日期**: 2026-05-05 (Day 4 evening)
**分析师**: Nick (HKUST-GZ, Xiong Lab)
**方法论基础**: `TCBM_Task_Applicability_Methodology_v2.md`
**前置文档**:
- v2 推导: `NQS_J1J2_Prediction_v2.md` (1098 lines, kept as historical artifact)
- v3 outline: `NQS_J1J2_Prediction_v3_outline.md` (Step 4 update with Day 3 v2 verified data)
- Day 3 verdict: `daily_log_day3.md` + `results/quick_adam_diagnostic_v2.json`
- Anchor: `bukov_2021_anchor.md` (Bukov 2021 SciPost Phys 10.147 verbatim quotes + Nick visual readings)

**v3 关键差异 vs v2**:
- Adam → SR reframe (per Day 3 v2 verdict):
  - Primary baseline: SR (Stochastic Reconfiguration, NQS field standard)
  - Secondary baseline: Adam (5 seeds, ablation only — vanilla gradient descent failure mode)
- Success criteria: 4-tier replacement (was single rel_err < 10%):
  - Tier 1 (feasibility), Tier 2 (NQS comparable), Tier 3 (robustness, dual sub-criteria),
    Tier 4 (mechanism, n_kept)
- Tier 3 single tier with two sub-criteria, BOTH must hold for PASS:
  - (a) σ_TCBM/σ_Adam ≤ 0.5 (vs unstable baseline, safety net)
  - (b) σ_TCBM ≤ 2× σ_SR (vs best baseline, narrative ceiling)
- New hypothesis H2: TCBM-hybrid n_kept ≥ 5/20 SR-orthogonal directions (mechanism claim C)
- New §9 (Adam → SR reframe rationale), §10 (Plan A/B/C hedges), §11 (disclosures)

**Day 3 verified data anchor**:
- σ_Adam_rel = 9.495% (5-seed std, RNG-fixed)
- Mean rel_err 74.45%, range [58.30%, 84.40%]
- Multi-basin signature confirmed: 5/5 seeds positive drift, mean +1.705
- Source: `results/quick_adam_diagnostic_v2.json` summary block

**Section IDs preserved from v2 for diff clarity** (§1.1, §1.3, §2, §4.1-4.6, §7.1-7.3,
§8.3 are near-verbatim from v2). New sections: §9, §10, §11.

**目的**: 提供 ex ante 严格证明 NQS J1-J2 4×4 落在 TCBM 适用边界内, 同时建立 SR 为 primary
baseline 的判据架构 + paper SI 必需的 disclosures.

---

## 0. 证明结构总览

按方法论 v2 §2 的五阶段流程, v3 reframe 不改变 stage 结构, 只调整 baseline + success criteria:

| Stage | 名称 | 证明结论 |
|---|---|---|
| I | 问题形式化 | 连续优化, $D=1120$, 无约束 |
| II | 能量势垒结构 | **T2 GREEN** (Bukov 2021 直接证据 + Hessian 证据) |
| III-A | 代数结构识别 | **B 类** (对称群 \|G\|=128, $\dim \mathcal{H}_{A_1}\approx 9$) |
| III-B | ψ warmup probe | **T4a 预测 GREEN 70%** (Day 7 实测) |
| IV | 同构映射 | YELLOW-GREEN (RBM identity + Bukov anchor) |
| V | T1-T5 量化 | T1 N/A, **T2 G**, T3 G, **T4a G predicted**, T4b G predicted, T5 GREEN |

**总判定**: **STRONG GREEN with Day 7 verification**

按 v2 §8.3 决策表 "T2 G + T4a G + T3 G + T4b G" 行 → STRONG GREEN, 建议全力做.

**v3 新增 baseline + success criteria layer** (§1.2 + §5.7 + §6.7-6.8):
4-tier success criteria (Tier 1-4), single Tier 3 with dual sub-criteria,
SR primary + Adam secondary baseline structure, 5 R-abort triggers
(R-abort-1, -2, -3 sub-criterion (a), -3 sub-criterion (b), -4, -5).

---

## 1. Stage I: 问题形式化

按方法论 v2 §3 流程.

### 1.1 标准化优化形式

**目标**: 找到 spin-1/2 J1-J2 Heisenberg model 在 4×4 PBC 上的基态

**Hamiltonian**:
$$
\hat{H} = J_1 \sum_{\langle i,j\rangle} \hat{\vec{S}}_i \cdot \hat{\vec{S}}_j + J_2 \sum_{\langle\langle i,j\rangle\rangle} \hat{\vec{S}}_i \cdot \hat{\vec{S}}_j
$$

**变分原理给出的优化问题**:

$$
\min_{\theta \in \mathbb{R}^D} E(\theta) = \frac{\langle\psi(\theta) | \hat{H} | \psi(\theta)\rangle}{\langle\psi(\theta) | \psi(\theta)\rangle}
$$

**约束**: 无 ($m=0, p=0$). 归一化由 Rayleigh 商分母自动处理.

**第一过滤器 (v2 §3.1)**: $\theta \in \mathbb{R}^D$ 是连续变量 → **PASS**

### 1.2 P0-1.2 Success criteria (4-tier, dual-baseline Tier 3)

v2 used a single rel_err < 10% criterion. v3 reframes to 4 layered tiers, paper
strength dependent on which tiers PASS:

| Tier | Metric | Threshold (PASS) | R-abort | Claim Level |
|------|--------|------------------|---------|-------------|
| 1 (feasibility) | rel_err_debiased vs ED truth | < 10% | > 15% | TCBM works (claim A) |
| 2 (NQS 可比性) | rel_err_TCBM / rel_err_SR | ≤ 2.0 | > 5.0 | competitive accuracy |
| 3 (核心 robustness) | (a) σ_TCBM/σ_Adam ≤ 0.5 **AND** (b) σ_TCBM ≤ 2× σ_SR | both hold | see §6.8 | claim B (primary) |
| 4 (核心 mechanism) | n_kept (TCBM ⊥ SR) | ≥ 5/20 | < 3/20 | claim C (mechanism) |

**Tier 3 structural note (single tier, dual sub-criteria)**:

Tier 3 is a single tier (NOT split into Tier 3a/3b sub-tiers), with two
sub-criteria; **BOTH must hold for Tier 3 PASS**:

- (a) σ_TCBM/σ_Adam ≤ 0.5 (vs unstable baseline, safety net):
  Ratio form. Day 3 v2 anchored σ_Adam = 9.5% → σ_TCBM ≤ 4.7% achievable bar.
- (b) σ_TCBM ≤ 2× σ_SR (vs best baseline, narrative ceiling):
  Multiplicative form. Avoids division-by-zero edge case if σ_SR → 0
  (vs σ_TCBM/σ_SR ≤ 2.0 form which is mathematically equivalent for σ_SR > 0).

Combined narrative reading: "TCBM 显著好于 unstable baseline AND 不输给 best
baseline by more than 2×".

**Paper outcome dependency**:

| Tier 1 | Tier 2 | Tier 3 (a)+(b) | Tier 4 | Paper outcome |
|--------|--------|----------------|--------|---------------|
| ✓ | ✓ | ✓ | ✓ | NMI strong paper |
| ✓ | ✗ | ✓ | ✓ | Publishable, narrower (TCBM σ smaller but absolute weaker) |
| ✓ | ✓ | ✓ | ✗ | Narrower (σ smaller, mechanism unclear) |
| ✓ | - | ✗ | - | Re-evaluation (likely 6×6 pivot) |
| ✗ | - | - | - | R-abort-1 trigger |

**关键数值** (按 v2 §3.2 必须记录的量, preserved from v2 with ED truth correction):

| 量 | 符号 | 数值 | 来源 |
|---|---|---|---|
| 参数维度 | $D$ | **1120** | $D = 2(N+M+NM)$, $N=16, M=2N=32$ |
| 等式约束数 | $m$ | 0 | NQS 无显式约束 |
| 不等式约束数 | $p$ | 0 | NQS 无显式约束 |
| 目标函数典型值 | $f(\theta_0)$ | $\approx 12$ | random init 时 (Day 1 实测 11.99, 来自 J1·1/4 + J2·1/4 = 12.0 解析) |
| 基态能量 | $E_0$ | $-8.4579$ | First-principles ED via `utils/ed_reference.py` (4×4 PBC, $J_2/J_1=0.5$, $S_z=0$ sector dim=12870, Day 1 verification) |
| 基态能量 per site | $E_0/N$ | $-0.5286$ | Same source |
| Loss basin 量级 | $L_{\text{basin}}$ | $\approx 20.45$ | Day 1 实测 ($f(\theta_0) - E_0 = 11.99 - (-8.4579)$) |
| 梯度典型值 | $\|\nabla f(\theta_0)\|$ | $\approx 5$ | Bukov 2021 类似 setup |

R-abort-1 / Tier 1 thresholds rebased to ED truth $E_0 = -8.4579$:
- R-abort-1 (Week 1 end): $|E_\text{TCBM} - E_0^\text{ED}| / |E_0^\text{ED}| > 15\%$
- Tier 1 (P0-1.2) success: $|E_\text{TCBM} - E_0^\text{ED}| / |E_0^\text{ED}| < 10\%$

### 1.3 量级分析

按方法论 v2 §10.1 快速筛查:

**$D \in [10^2, 10^5]$ 检查**:
$$D = 1120 \in [10^2, 10^5] \checkmark$$

按 v2 附录 A "TCBM 适用谱":
- $D < 10^4$: NN variant 下处于"TCBM 强势"区间
- T5 量级 ($D=1120$) 远低于 NN variant 失效阈值 ($D > 10^5$)

### 1.4 约束处理

无约束问题, 按 v2 §3.3 不需要 penalty 或 projection. 直接优化 $E(\theta)$ 即可.

### 1.5 Stage I 交付物

**结论**:
- 连续优化问题 ✓
- $D = 1120$ 在 NN variant 适用范围内 ✓
- 无约束, 纯目标最小化 ✓
- 23 pytest tests 全过 (16 Day 1 + 4 Day 3 callback + 3 Day 3 RNG seeding)
- Day 3 RNG bug fix (commit `18716b3`): J1J2Problem.set_seed() method added,
  cross-seed reproducibility verified (see §11.4 disclosure)
- 进入 Stage II

---

## 2. Stage II: 能量势垒结构分析

按方法论 v2 §4 三层检查 + WKB 量化. v3 不修改本 stage 内容; SR 也 lives on the
same energy landscape as Adam/TCBM, so the barrier topography analysis remains
load-bearing for all baselines.

### 2.1 第一层: 多个 basin 的存在性证明

**主张**: 在 4×4 J1-J2 at $J_2/J_1 = 0.5$ 上, 至少存在两个独立 basin.

**Basin A (sign rule trap basin)**:

定义为满足 Marshall-Peierls sign rule 的能量极小:
$$
\mathcal{B}_A = \{\theta : \text{sign}(\psi(\sigma; \theta)) = (-1)^{N_{\uparrow,A}(\sigma)} \text{ for all } \sigma\}
$$

其中 $A$ 是 bipartite sublattice 之一, $N_{\uparrow,A}(\sigma)$ 是 $A$ sublattice 上 ↑ 自旋数.

**Basin B (true ground state basin)**:

定义为接近真实基态 $\psi_0$ 的局部极小:
$$
\mathcal{B}_B = \{\theta : \|\psi(\theta) - \psi_0\| \leq \epsilon\}
$$

**两个 basin 独立性的物理证据**:

引理 1 (Szabó & Castelnovo 2020 PRR 2, 033075, 本项目 cards #101):

> 在 frustrated 2D Heisenberg with $J_2/J_1 > J_c \approx 0.4$ 区间, Marshall sign rule **不**对应真实基态的 sign 结构, 但 NQS 优化器**仍会收敛到** Marshall-rule satisfying state (该状态是局部极小但非全局极小).

证明等价物: Szabó 2020 的 Fig 3 (J2/J1=0.55) 显式展示了 NQS 训练曲线收敛到比 DMRG 真值高约 $\Delta E/N \approx 0.005$ 的状态, 且训练曲线**长时间停留**在该值 (非渐近降低), 证明这是局部极小而非缓慢收敛过程.

**$E(\mathcal{B}_A) \approx E(\mathcal{B}_B)$ 验证**:

按 v2 §4.1 第一层, 需要 $f(x_A) \approx f(x_B)$:
- $E_A/N \approx -0.495$ (Szabó 2020 Marshall-rule trap 报告值)
- $E_B/N \approx -0.503$ (DMRG ground state benchmark)

绝对差 $|E_A - E_B|/N \approx 0.008$, 远小于 Loss basin 量级 $L_{\text{basin}}/N = 0.5$.

**满足"两个 basin 能量近似相等"的判据**: $|E_A - E_B|/L_{\text{basin}} \approx 0.016 \ll 1$ ✓

**Day 3 v2 实测 evidence (NEW v3)**: 5 seeds Adam 收敛到 final cost spread
$[-3.53, -1.32]$, 2-cluster structure (seed 42 outlier in better basin
$-3.53$, other 4 seeds clustered in $[-2.46, -1.32]$ worse basin).
gap_ratio = 0.48 (borderline due to 5-seed sample size, not noise floor).
Source: `results/quick_adam_diagnostic_v2.json` summary.

**结论第一层**: 两个独立 basin 存在 ✓ (理论 + Day 3 实测).

### 2.2 第二层: Basin 之间真实 barrier

按 v2 §4.1 第二层, 沿 $\theta_A \to \theta_B$ 的直线插值 $\gamma(t) = (1-t)\theta_A + t\theta_B$, $t \in [0,1]$, 测 $E(\gamma(t))$ 是否有 hump.

**Note on barrier hump estimate (corrected Day 2)**:

Earlier protocol drafts cited "Bukov 2021 reports $\Delta E_{\text{hump}}/N \approx 0.05-0.10$" — this was a Claude (LLM) hallucination during prediction draft. Bukov 2021 does **not** perform direct barrier-interpolation analysis; their landscape characterization uses Hessian spectrum + seed-trajectory divergence (Fig 11 + Fig 12) rather than hump magnitude. See `bukov_2021_anchor.md` for verbatim quotes.

The basin-to-basin barrier in 4×4 PBC $J_2/J_1=0.5$ is best evidenced by:

- Sign-rule trap basin existence (Szabó 2020 PRR 2, 033075, multiple convergent
  metastable states observed)
- Hessian negative eigenvalues at saddles (Bukov 2021 Fig 11)
- Spin-glass-like landscape (Bukov 2021 Section 6.1 quote: "located in deep valleys,
  separated by high and difficult to overcome energy barriers")

The specific $\Delta E/N$ magnitude is empirical and should be verified by direct interpolation between optimizer-found saddle points during Day 4-5 baseline runs. For T2 ratio computation, we use $\Delta E \in [0.1, 0.8]$ as a plausible range; even the conservative $\Delta E = 0.1$ gives T2 = 20 (still GREEN per v2 §7.2).

**Day 3 v2 anchored**: Adam mean drift +1.705 (5/5 seeds positive,
final_cost > best_during_cost) — consistent with saddle crossing into
worse basin under noisy optimization, indirect evidence of barrier
existence and saddle-point structure.

**结论第二层**: barrier hump 存在 (qualitative evidence + Day 3 indirect evidence),
$\Delta E$ 量级待 Day 4-5 saddle interpolation 实测; WKB ratio T2 ≥ 20 robust ✓.

### 2.3 第三层: Energetic vs Entropic

按 v2 §4.1 第三层, 需要在 barrier 顶部计算 Hessian 负特征值.

**理论论证** (Hessian negative eigenvalue 估计):

在 sign-rule transition 的 saddle 附近, Hessian $H$ 包含两类负特征值:
1. **Single-flip mode**: 翻转一个 spin 的 sign 引起的能量变化方向
2. **Cluster-flip mode**: 协同翻转一个 cluster 的 sign 引起的能量变化方向

**Note on Hessian eigenvalue ratio (corrected Day 2)**:

Bukov 2021 Fig 11 (App G) shows (Nick visual readings, see `bukov_2021_anchor.md`):

- **N=4×4** (iter 499, $E_{gs}=-8.457917$): most eigenvalues $10^0$ to $10^5$
  positive; few negative $\sim 10^{-2}$ magnitude (inset shows up to $\approx -0.04$);
  $|\lambda_{\min}|/|\lambda_{\max}| \approx 4 \times 10^{-7}$ (effectively flat
  negative directions). Caption: "flat directions on the variational manifold".
- **N=6×6** (iter 675, $E_{gs}=-18.073818$): most eigenvalues $10^0$ to $10^9$
  positive; few **large** negative eigenvalues $\sim 10^2$ magnitude;
  $|\lambda_{\min}|/|\lambda_{\max}| \approx 10^{-7}$. Caption: "highly curved
  sparse directions on the manifold, hard for optimizer".

**Key insight**: $|\lambda_{\min}|/|\lambda_{\max}|$ ratio is similar at both
sizes ($\sim 10^{-7}$), but the ABSOLUTE magnitude of $\lambda_{\min}$ differs
by 4 orders of magnitude ($10^{-2}$ vs $10^2$). 4×4 lacks sharp narrow valleys;
6×6 has them.

**Implication for TCBM mechanism on 4×4**: Our system has flat-negative-curvature
landscape. TCBM's clamping-via-SVD mechanism should focus on **diversity injection**
(gradient subspace orthogonal to QGT subspace, paper claim C) rather than
narrow-valley detection. This motivates the n_kept ≥ 5/20 hypothesis (H2, §3.2).

**结论第三层**: barrier 是 energetic (qualitative; 4×4 negatives small $\sim 10^{-2}$
but real). Mechanism on 4×4 = diversity injection, not narrow-valley detection. ✓

### 2.4 WKB 条件量化 (T2)

按 v2 §7.2:
$$\text{T2 ratio} = \frac{\Delta E}{T_{\min}}$$

输入数据:
- $\Delta E$: barrier height between Marshall-rule trap basin and ground state basin.
  Specific magnitude is **not directly measured by Bukov 2021** (their landscape
  characterization uses Hessian + seed scatter, not direct interpolation).

  We use a range based on physical reasoning:
  - **Conservative** (used as reference): $\Delta E \approx 0.1$ (lower bound from
    sign-rule trap depth observed in Szabó 2020)
  - **Optimistic** (sensitivity check): $\Delta E \approx 0.8$ (if Bukov-style
    metastable energy spread $\sim 0.05 \cdot N$ holds for 4×4)
- $T_{\min} = 0.005$ (Protocol v2.1 配置)

**Reference T2 (used for GREEN judgment)**:
$$\text{T2 ratio} = \frac{0.1}{0.005} = 20$$

**Sensitivity analysis (optimistic case)**:
$$\text{T2 ratio} = \frac{0.8}{0.005} = 160$$

按 v2 §7.2 GREEN 判据 "$> 10$":
- Reference: $20 > 10 \Rightarrow$ **T2 GREEN** ✓
- Optimistic: $160 \gg 10 \Rightarrow$ **T2 GREEN strong**

**Robustness**: Both bounds give GREEN. Even if true $\Delta E$ falls below 0.05,
T2 = 10 just hits the boundary. Judgment robust to factor-of-2 uncertainty in $\Delta E$.

### 2.5 Stage II Gate

按 v2 §4.3:
- **GREEN**: barrier 存在 + energetic + WKB 满足 ✓✓✓

**Stage II 总判定: GREEN**

进入 Stage III.

### 2.6 Stage II 交付物

```
Barrier analysis summary (v3 verified Day 3 evidence integrated):
  Two basins identified:
    - Basin A: Marshall sign rule trap (E_A/N ≈ -0.495, theory: Szabó 2020)
    - Basin B: True ground state (E_B/N ≈ -0.503, ED reference)
    - Day 3 v2 实测 5-seed Adam: 2-cluster structure (gap_ratio 0.48 borderline)

  Barrier hump (along straight interpolation):
    - Empirical estimate ΔE ∈ [0.1, 0.8] (待 Day 4-5 saddle interpolation 实测)
    - Day 3 v2 indirect evidence: mean drift +1.705 (5/5 seeds positive,
      consistent with saddle crossing)

  Barrier nature: energetic (qualitative on 4×4)
    - 4×4 Hessian λ ratio: |λ_min|/|λ_max| ≈ 4×10⁻⁷, |λ_min| ~ 10⁻² (flat negatives)
    - 6×6 contrast: |λ_min| ~ 10² (sharp narrow valleys, our 4×4 lacks)
    - Mechanism implication: TCBM advantage on 4×4 = diversity injection,
      not narrow-valley detection → n_kept hypothesis (H2)

  WKB ratio: T2 = ΔE/T_min ≥ 20 (conservative ΔE=0.1) → GREEN
```

---

## 3. Stage III: Hypotheses + Algebraic Structure (REWRITE)

v2 §3 was algebraic structure analysis (now relocated to §3.4); v3 §3 leads with
the three hypotheses driving the experiment + their measurement procedures.

### 3.1 H1, H2, H3 (Hypotheses, v3 reframe)

| Hypothesis | Statement | Measurement | Claim |
|------------|-----------|-------------|-------|
| H1 (renamed) | TCBM σ_seed ≤ 0.5 × SR σ_seed (15 seeds) | Week 3 robustness sweep | Primary claim B (robustness, Tier 3 sub-criterion (b)) |
| H2 (NEW) | TCBM-hybrid n_kept ≥ 5/20 SR-orthogonal directions | Day 7-8 subspace overlap analysis (see §3.2) | Mechanism claim C (Tier 4) |
| H3 (reframed) | TCBM clamping subspace ≠ QGT principal subspace; TCBM and SR explore complementary directions on the variational manifold | Day 8+ subspace overlap heatmap | Supporting evidence for H2 |

**Removed from v2**: implicit assumption "Adam is the optimizer to beat" — it
isn't, on this problem class (Day 3 v2 Adam ceiling 74.45% rel_err vs Bukov SR
~10⁻³). Adam demoted to ablation (5 seeds, SI demonstrates vanilla gradient
descent failure mode).

**H2 quantification**:
- TCBM hybrid mode tracks `subspace_TCBM` (SVD top-k of clamped gradient buffer)
- SR optimizer (with `record_qgt_eigvals=True`) gives `subspace_SR` (top-k QGT eigvecs)
- `n_kept = number of TCBM directions with cosine similarity < calibrated_threshold to all SR directions`
- Claim C passes if `n_kept ≥ 5` (≥25% of TCBM subspace is SR-orthogonal)

### 3.2 H2 measurement procedure (Day 7-8, 4-step)

**Step 1 (Day 7): Random null calibration**

Goal: establish baseline n_kept distribution under random subspace pairs.

- Generate 50 pairs of random k=20 subspaces in D=1120 ambient space
  (each subspace = SVD top-20 of a random Gaussian D×D matrix)
- For each pair, compute n_kept using cosine threshold (to be calibrated below)
- Record random_null_n_kept distribution: p5, p50, p95

**Step 2 (Day 7): Calibrate cosine threshold**

Goal: pick threshold where random null is dominated by chance.

- Vary cosine threshold from 0.1 to 0.5 (steps of 0.05)
- Find threshold where random_null_p95 ≈ 5 (i.e., expected n_kept under null ≤ 5)
- Use this calibrated threshold for actual TCBM vs SR measurement

**Step 3 (Day 8): TCBM vs SR n_kept measurement**

Goal: real measurement of subspace orthogonality.

- subspace_TCBM = SVD top-20 of TCBM hybrid clamped buffer (latest update at
  step T_w + warmup, when buffer is "mature")
- subspace_SR = top-20 QGT eigenvectors (from SR run at corresponding training step,
  with record_qgt_eigvals=True)
- Compute cosine similarity matrix M_ij = |⟨u_i^TCBM, u_j^SR⟩|
- n_kept = number of TCBM directions where max_j M_ij < calibrated_threshold

**Step 4 (Day 8): Statistical test**

Goal: distinguish signal from chance.

- n_kept > random_null_p95: significant orthogonality (claim C strong)
- n_kept ∈ [random_null_p5, random_null_p95]: indistinguishable from null (claim C weak)
- n_kept < random_null_p5: anti-correlated (TCBM stays close to SR, claim C fails)

**R-abort-5 trigger**: n_kept < 3 → mechanism claim demoted to "TCBM matches SR
robustness only" (no orthogonal-subspace narrative).

### 3.3 (For H2 substrate) Algebraic structure analysis (v2 §3.1-3.5 relocated)

按方法论 v2 §5.2 三类代数结构 (A > B > C). Load-bearing for §4 ψ probe theory.

**(A) 物理 operator 特征子空间** — NOT applicable on NQS:
- Hamiltonian $\hat{H}$: 作用于 Hilbert space, 不是参数空间. NOT applicable.
- Hessian of $E(\theta)$: data-dependent (取决于 $\theta$ 位置), 不是物理给定. NOT applicable.
- Quantum Geometric Tensor (QGT) $S_{ij}$: 是参数空间 metric, 但**不是 barrier escape
  direction 的 operator**. NOT applicable for $v_{\text{escape}}$ (但作为 NQS-3 hybrid
  模式输入有用 — H2 测量基础).

**(B) 对称群投影分解** — APPLICABLE:

对 4×4 square lattice with PBC J1-J2 Heisenberg, 对称群:
$$G = (T_{4} \times T_{4}) \times C_{4v} \times SU(2)$$

- $T_4 \times T_4$: 16 lattice translations
- $C_{4v}$: 8 elements (4 rotations × 2 reflections)
- $SU(2)$: continuous; $S_z=0$ sector 内离散投影 $\mathbb{Z}_2$

$|G_{\text{eff}}| = 16 \times 8 = 128$ (在 $S_z=0$ + spin-flip even sector).

**Ground state irrep** (Schulz-Ziman-Poilblanc 1996; Capriotti-Sorella 2000):
$\rho_0$ = "$\Gamma$-point $A_1$, even spin parity"

**理论子空间维度**:
$$\dim \mathcal{H}_{\rho_0} = \frac{D}{|G_{\text{eff}}|} = \frac{1120}{128} = 8.75 \approx 9$$

**TCBM $k = 20$ overcomplete ratio**: $20/9 \approx 2.3\times$ → 健康 overcomplete
(给学习留 margin). Escape direction $v_{\text{escape}}^{\text{predicted}} \in
\mathcal{H}_{\rho_0} \subset \mathbb{R}^{1120}$, 是 Marshall sign rule basin → true
ground state basin saddle 切线方向, $A_1$-irrep 内的 amplitude pattern flipping mode.

**(C) Emergent 低维性** — UNCERTAIN:
- 对 J1-J2 NQS 文献无专门 SGD trajectory PCA 分析
- Bukov 2021 "rugged landscape" + Westerhout 2020 "shattered" 现象让 C 类可靠性受质疑
- 唯一可靠区分方法: ψ probe (Stage III-B)
- 但我们有 B 类 backup, 即使 C 类不利, B 类 $A_1$-irrep 投影机制仍工作

**Stage III-A judgment**: GREEN (B 类), 与 v2 §9 案例对比:

| 案例 | III-A 类 | III-A 结论 |
|---|---|---|
| DC-OPF | A 类 (PTDF) | STRONG GREEN |
| SCM | 不清晰 | RED |
| ResNet-56-NS | C 类 (uncertain) | 需 III-B 验证 |
| **NQS J1-J2 (本工作)** | **B 类 (\|G\|=128)** | **GREEN** |

NQS J1-J2 III-A 强度: 介于 DC-OPF 和 ResNet 之间. 比 ResNet 强 (B 类 > C 类), 但
弱于 DC-OPF (B 类 < A 类). 这与方法论 v2 §1.2 三层物理框架一致.

### 3.4 Stage III-A 总判定 + III 总 Gate

| 维度 | 结果 | 判定 |
|---|---|---|
| III-A 代数 (B 类) | $|G|=128$, $\rho_0 = A_1$ at $\Gamma$, dim 9, overcomplete 2.3× | GREEN |
| III-B ψ probe (predicted) | $\psi \in [1.5, 2.0]$ at warmup end | GREEN provisional |

**Stage III judgment**: **GREEN provisional, pending Day 7 ψ verification**

按 v2 §5.4 规则: "III-A 找到 A 类或 B 类 AND III-B ψ > 1.5 → 继续 Stage IV". 我们目前
是 "B 类 AND ψ predicted > 1.5". Day 7 ψ 实测确认后, 升级为 **GREEN confirmed**.

---

## 4. Stage III-B: ψ Warmup Probe (关键早期诊断)

按方法论 v2 §5.3 协议设计 + 数学预测. v3 不修改本 stage 内容; ψ probe 是 TCBM
mechanism 的 substrate identification, 与 SR baseline 对比 orthogonal.

### 4.1 ψ 演化的物理机制 (理论预期)

**ψ 定义**:
$$\psi(t) = \frac{\sigma_k(t)}{\sigma_{k+1}(t)}$$

其中 $\sigma_k$ 是 gradient buffer SVD 的第 $k$ 个奇异值.

**与 III-A 的关联**: 如果 III-A 的 B 类预测正确 (gradient 优先 sample $\rho_0$
irrep 方向), 则 gradient buffer 的奇异值谱有:
- $\sigma_1, \ldots, \sigma_9$: "signal", 对应 $A_1$-irrep 9 维子空间
- $\sigma_{10}, \ldots, \sigma_{20}$: 过渡区
- $\sigma_{21}, \ldots$: noise floor

**TCBM $k=20$ 处的 ψ**: $k=20$ 落在过渡区末端, 预期:
- $\sigma_{20}$ 仍有部分 signal (对应 $A_1$-irrep secondary modes)
- $\sigma_{21}$ 接近 noise floor

预期 ψ 范围:
- 乐观 (B 类强 alignment): $\psi \approx 2-3$
- 中性: $\psi \approx 1.5-2$
- 悲观 (B 类信号被 noise 稀释): $\psi \approx 1.1-1.3$

### 4.2 量化 ψ 预测的数学论证

**论证 1: gradient 的 symmetry inheritance**

Hamiltonian $\hat{H}$ 对 group $G$ 不变, 所以参数空间梯度满足:
$$\nabla_\theta E(g \cdot \theta) = g \cdot \nabla_\theta E(\theta), \quad \forall g \in G$$

在 $G$-irrep 分解 $\mathbb{R}^D = \bigoplus_\rho \mathcal{H}_\rho$ 下, gradient 的分量
在每个 irrep 内独立演化.

**论证 2: gradient 的 ground state irrep concentration**

对接近 ground state 的 $\theta$ ($\psi(\theta) \approx \psi_0$), 梯度项
$\nabla_\theta \langle\psi|\hat{H}|\psi\rangle$ 中含 $|\psi_0\rangle$ 因子. 由于
$|\psi_0\rangle$ 属于 $\rho_0 = A_1$ at $\Gamma$, gradient 的 $A_1$ 分量被显著放大:
$$\frac{\|\nabla_\theta E\|_{A_1}}{\|\nabla_\theta E\|_{\text{other}}} \gg 1$$

**论证 3: ψ 估计**

设 gradient 在 $A_1$ 子空间内的"信号"幅度为 $s$, 其他 irrep 上的"噪声"幅度为
$n$, $s \gg n$. Gradient buffer covariance $\Sigma$ 在 $A_1$ 9 维子空间内特征值
约 $s^2$, 其他方向约 $n^2$. 前 9 个奇异值 $\sigma_1, \ldots, \sigma_9 \sim s$,
后续 $\sigma_{10}, \ldots \sim n$.

考虑 finite-sample SVD spectrum:
$$\sigma_k / \sigma_{k+1} \approx 1 + \frac{(s^2 - n^2)}{n^2 (k+1)} \cdot \mathbb{1}[k \leq \dim(A_1)]$$

对 $k = 20 > \dim(A_1) = 9$, 这个 ratio $\to 1 + \text{small correction}$ → **预期
ψ 在 1.1-2 之间**.

### 4.3 与方法论 v2 已知案例对照

按 v2 §9 案例表中的 ψ 实测:

| 案例 | III-A 类 | $\dim(\text{signal subspace})$ | TCBM $k$ | overcomplete | 实测 ψ |
|---|---|---|---|---|---|
| DC-OPF | A (PTDF) | ~20 | 20 | 1.0× | **3-10** |
| ResNet-56-NS | C-shattered | (不存在 signal subspace) | 20 | $\infty$ | **1.00 ± 0.01** |
| 1D TFIM (preview) | B 弱 | ~12 | 16 | 1.3× | **1.2-1.6** (预测) |
| **NQS J1-J2 4×4 (本工作)** | **B 中** | **9** | **20** | **2.3×** | **1.5-2** (**预测**) |

**插值估计**: NQS J1-J2 的 ψ 介于 DC-OPF (3-10) 和 ResNet (1.0) 之间, 但靠近 1D
TFIM (1.2-1.6) 一侧 (同为 B 类).

### 4.4 ψ probe 协议 (Day 7 执行)

按方法论 v2 §5.3 协议:

```python
配置:
  M = 12 replicas
  n_steps = 1000
  k = 20
  subspace_warmup = 150
  subspace_update_freq = 50
  psi_star = ∞               # NEVER trigger T_w
  min_warmup = ∞
  subspace_source = 'gradient'
  n_vmc_samples = 1500
```

**判定**:

| ψ at step 1000 | T4a 判定 | Action |
|---|---|---|
| ψ ≥ 1.5 | **GREEN** | 继续 Week 2 |
| ψ ∈ [1.1, 1.5) | **YELLOW** | retry: grad_buffer 600, warmup 300 |
| ψ < 1.1 | **RED** | 触发 R-abort-4, Plan B |

**预期分布** (基于 §4.2 论证): GREEN 70%, YELLOW 25%, RED 5%.

### 4.5 Stage III-B 交付物

```
ψ warmup probe predictions (Day 7 verification target):

  Theoretical basis:
    - B-type symmetry group |G| = 128
    - Ground state irrep ρ_0 = Γ-point A_1
    - Theoretical signal subspace dim = 9
    - TCBM k = 20 (overcomplete 2.3×)

  Predicted ψ range: [1.5, 2.0]
  Distribution: GREEN 70%, YELLOW 25%, RED 5%
  Day 7 verification: required to confirm prediction.
```

### 4.6 Stage III 总 Gate (NN variant config)

按方法论 v2 §5.4 + v3 baseline reframe:

| 维度 | 结果 | 判定 |
|---|---|---|
| III-A 代数 | B 类 ($|G|=128$) | GREEN |
| III-B ψ probe | predicted 1.5-2 | GREEN (provisional) |

**Stage III judgment**: **GREEN provisional, pending Day 7 ψ verification**.

v2 baseline references ("Adam baseline") 在 v3 内 reflow 为 "SR primary baseline +
Adam ablation" (see §6.7 + §6.8 for full table).

---

## 5. Stage IV: 同构映射构造

按方法论 v2 §6 流程. v3 reframe 使本 stage 的 Bukov anchor 仍是 load-bearing
(SR also uses Bukov 2021 as reference benchmark).

### 5.1 候选目标物理系统 (v2 §6.3)

| 选项 | 适用性 | 评估 |
|---|---|---|
| Josephson junction array | ✗ | 需要网络拓扑 + 传递矩阵, NQS 无对应 |
| 双阱 QHO / 量子隧穿 | △ | 适合 barrier-crossing, 但太抽象 |
| **RBM free energy landscape** | **✓** | **NQS 本质就是 RBM 变分** |

### 5.2 RBM Free Energy 同构 (identity 映射)

NQS 的变分能量 $E(\theta)$:
$$E(\theta) = \sum_\sigma \frac{|\psi(\sigma; \theta)|^2}{Z(\theta)} \cdot E_{\text{loc}}(\sigma; \theta) = \mathbb{E}_{\sigma \sim p_\theta}[E_{\text{loc}}(\sigma; \theta)]$$

这是 RBM free energy 的精确形式. **Identity 映射**:

| NQS 量 | RBM 量 | 对应 |
|---|---|---|
| 参数 $\theta$ | RBM weights $\{a_i, b_j, W_{ij}\}$ | 直接相同 |
| 波函数 $\psi(\sigma; \theta)$ | RBM amplitude | 直接相同 |
| Gibbs distribution $p_\theta$ | RBM marginal | 直接相同 |
| Variational energy | RBM free energy | 直接相同 |

### 5.3 Identity 映射的局限

按 v2 §6.4 病症 check:

DC-OPF ↔ Josephson 同构提供了非平凡 predictions ($I_c \leftrightarrow b_{ij}$,
washboard tilt ↔ generation dispatch, tunneling time ↔ TCBM crossing time).

NQS = RBM 的 identity 映射: 映射两边是同一个对象, **不**提供"哪些参数方向重要"的
额外信息, **不**给出 escape direction $v_{\text{escape}}$ 的显式公式.

**Stage IV 信息量**: 中等. 比 1D TFIM 更强 (Bukov 2021 提供 quantitative anchor),
但比 DC-OPF 弱 (mapping 本身没 enrichment).

### 5.4 Bukov 2021 作为 Quantitative Anchor (D3.A: embed numbers + reference)

Direct number embedding (per D3.A); verbatim quotes + Nick visual readings
in `bukov_2021_anchor.md`:

| Bukov 报告 | 值 | 来源 (anchor file) |
|---|---|---|
| Final $\|E-E_{GS}\|/N$ spread, 6×6, 4 seeds | $[10^{-3}, 10^{-2}]$ → σ ≈ 0.34% | Fig 12 (Nick 视读) |
| Plateau $E/N$, 6×6 | $\approx -0.5019$ | Section 7.1 |
| Final $\|E-E_{GS}\|/N$, 4×4 partial learning ($\log\|\psi\|$) | $\approx 3 \times 10^{-4}$ | Fig 8 left blue (Nick 视读) |
| Final $\|E-E_{GS}\|/N$, 4×4 partial learning ($\phi$) | $\approx 5 \times 10^{-5}$ | Fig 8 left red (Nick 视读) |
| Hessian $\|\lambda_{\min}\|/\|\lambda_{\max}\|$, 4×4 | $\approx 4 \times 10^{-7}$ ($\|\lambda_{\min}\| \sim 10^{-2}$) | Fig 11 inset (Nick 视读) |
| Hessian $\|\lambda_{\min}\|/\|\lambda_{\max}\|$, 6×6 | $\approx 10^{-7}$ ($\|\lambda_{\min}\| \sim 10^2$) | Fig 11 inset (Nick 视读) |

**关键 v3 implication**: σ_seed ≈ 0.34% on 6×6 SR — 这是 v3 sub-criterion (b)
$\sigma_{TCBM} \leq 2 \times \sigma_{SR}$ 的 best-baseline anchor. 若 4×4 SR
σ extrapolates to ≈ 0.5-1.5%, 则 σ_TCBM ≤ 1-3% needed (Tier 3 PASS bar (b)).

### 5.5 Stage IV Gate

按 v2 §6.5 + v3 reframe (Bukov anchor 强化):
- 严格 identity mapping (NQS = RBM 变分)
- Bukov 2021 提供具体数值锚点 (σ_seed 6×6 ≈ 0.34%, 4×4 partial learning ~10⁻⁴)
- Mapping 本身信息量中等

**Stage IV judgment: YELLOW-GREEN** (v2 wording preserved). 与 v2 §1.2 三层物理
框架兼容: 靠 Stage II + III 通过严格判定, Stage IV 是辅助.

### 5.6 Stage IV 交付物 (EXPANDED for v3)

```
Isomorphism mapping summary (v3 expanded with SR deliverables):

  Primary mapping:
    NQS variational energy ⟷ RBM free energy
    Type: identity mapping (rigorous but information-poor)

  Quantitative anchor (Bukov 2021 SciPost Phys 10.147):
    - Final |E-EGS|/N spread ~10⁻³ to 10⁻² across 4 seeds, 6×6 (Fig 12)
      → σ_seed/|E| ≈ 0.34% (Bukov SR best baseline)
    - Plateau E/N ≈ -0.5019, 6×6 (Section 7.1)
    - Hessian: 4×4 |λ_min| ~ 10⁻² (flat); 6×6 |λ_min| ~ 10² (sharp)
    - 4×4 partial learning achieves 3×10⁻⁴ (log|ψ|), 5×10⁻⁵ (φ)
    Anchor file: docs/bukov_2021_anchor.md (verbatim quotes)

  v3 Stage IV deliverables (expanded for SR primary baseline):
    Day 5: core/sr_optimizer.py implementation
      (4 helpers: compute_log_derivatives, compute_qgt, regularize_qgt,
       solve_sr_update; SROptimizer.optimize main loop)
    Day 6: experiments/run_sr_baseline.py single-seed 4×4 J2=0.5 (~30-90 min)
    Day 7+: random null calibration + cosine threshold for n_kept (Step 1-2 of §3.2)
    Day 8+: TCBM vs SR n_kept measurement (Step 3-4 of §3.2)
    Day 17-19: experiments/run_sr_robustness_sweep.py 15-seed
    Day 8+: analyze_subspace_overlap.py — n_kept measurement infrastructure

  Information level: medium-high
    - Stronger than 1D TFIM (no Bukov anchor)
    - Stronger than v2 (now have Day 3 Adam verified data + SR target σ ≈ 0.34%)
    - Weaker than DC-OPF (Josephson rich mapping)

  Verdict: YELLOW-GREEN (unchanged from v2)
```

### 5.7 Success thresholds table (5-row, σ_SR-based + n_kept)

v3 success threshold table replaces v2 §5.4's 3-row table. Single Tier 3 row
shows BOTH sub-criteria; Tier 3 PASS = both hold.

| Metric | Threshold (PASS) | R-abort | Source |
|--------|------------------|---------|--------|
| rel_err_debiased (vs ED) | < 10% | > 15% | TCBM final-eval mean |
| rel_err_TCBM / rel_err_SR | ≤ 2.0 | > 5.0 | TCBM vs SR final-eval |
| **Tier 3 (BOTH must hold)**: (a) σ_TCBM/σ_Adam ≤ 0.5 AND (b) σ_TCBM ≤ 2× σ_SR | both PASS | (a) σ_TCBM/σ_Adam > 0.8 OR (b) σ_TCBM > 4× σ_SR (TODO Day 6 calibrate) | Week 3 robustness sweep, 15 seeds |
| n_kept (SR-orthogonal directions) | ≥ 5 / 20 | < 3 / 20 | Day 8+ subspace overlap analysis |
| swap_acc avg | ∈ [0.2, 0.6] | < 0.1 or > 0.9 | TCBM PT diagnostic |

**Tier 3 sub-criterion (b) R-abort threshold note**: Tentative 4× σ_SR (i.e., 2×
over the PASS bar of 2× σ_SR). Final calibration pending Day 6 SR baseline σ_SR
data; if σ_SR ≈ 0.5-1.5%, R-abort fires when σ_TCBM > 2-6%. See §6.8 R-abort
table for the deferred TODO marker.

---

## 6. Stage V: T1-T5 量化验证 + R-aborts

按方法论 v2 §7 + §8 决策表. v3 不修改 T1-T5 derivation; only §6.7 (judgment matrix
reflow Adam → SR) and §6.8 (R-abort table expansion) are rewritten.

### 6.1 T1: 梯度 vs 噪声 (v2 降级, 仅 v5 optimizer 适用)

NN variant 的 adaptive step size 自动化解 T1. 我们用 NN variant, **T1 不是必须验证的条件**.

**仅供参考的 v5-style 计算**:
$$\text{T1 ratio} = \frac{1}{\|\nabla f\|}\sqrt{\frac{2 T_{\min} D}{\eta}}$$

代入 $\|\nabla f\| = 5, T_{\min}=0.005, D=1120, \eta=0.0005$:
$$\text{T1 ratio} = \frac{1}{5}\sqrt{22400} \approx 30$$

按 v2 §7.1 判据: 10-100 range → YELLOW (v5 standard); NN variant: **N/A**.

**T1 judgment (NN variant)**: **不适用**.

### 6.2 T2: WKB Condition (核心条件)

T2 ratio computation: see Stage II §2.4 for full derivation.

**Summary**:
- Reference (conservative $\Delta E = 0.1$): T2 = 20 → **GREEN**
- Sensitivity (optimistic $\Delta E = 0.8$): T2 = 160 → **STRONG GREEN**

**T2 judgment**: **GREEN** ✓

### 6.3 T3: Cross-temperature Metropolis

按 v2 §7.3:
$$p_{\text{swap}} = \min(1, \exp(-\Delta f \cdot (1/T_{\text{hot}} - 1/T_{\text{cold}})))$$

**温度阶梯**: $T_{\min} = 0.005, T_{\max} = 2.0$, $M = 12$ replicas, geometric ratio
$r = (400)^{1/11} \approx 1.78$.

$1/T_{i} - 1/T_{i+1} = (1 - 1/r)/T_i = 0.438/0.005 \approx 88$ (cold end).

**典型 ΔE between adjacent replicas** (统计力学 $\Delta E \sim T \sqrt{C_V}$,
$C_V \sim N$): $\Delta E \approx 0.005 \times \sqrt{16} = 0.020$.

**Swap acceptance**: $p_{\text{swap}} = \exp(-0.020 \times 88) = \exp(-1.76) \approx 0.17 > 0.1$.

**T3 judgment**: **GREEN** ✓ (0.17 > 0.1)

### 6.4 T4a: Subspace Existence (核心条件, v2 新增)

按 v2 §7.4a: $\psi(t) = \sigma_k/\sigma_{k+1}$, $k = 20$.

判据: ψ > 1.5 stable → GREEN; ψ ∈ [1.1, 1.5] → YELLOW; ψ ≤ 1.1 → RED.

**预测** (Stage III-B §4.2): $\psi \in [1.5, 2.0]$, 70% GREEN.

**T4a judgment (predicted)**: **GREEN** ✓ (Day 7 verification needed).

### 6.5 T4b: Subspace Correctness

按 v2 §7.4b: T4b score = $\max_i |\cos(v_{\text{escape}}^{\text{predicted}}, U_{\text{SVD}}[:,i])|$.

判据: > 0.3 → GREEN; 0.1-0.3 → YELLOW; < 0.1 → RED.

**Predicted**: $|\cos| \in [0.3, 0.7]$, 70% GREEN. 来自 Stage III-A: $A_1$-irrep
内的 amplitude-flipping mode. Random alignment baseline $1/\sqrt{D} \approx 0.030$;
B 类 alignment 期望 $\sqrt{9/20} \approx 0.67$ (full $A_1$ capture);
实际 alignment 0.3-0.7 (部分 capture).

**T4b judgment (predicted)**: **GREEN** ✓ (Stage 2 verification needed).

T4b conditional on T4a PASS. 如果 T4a RED, T4b 无意义.

### 6.6 T5: Hot Replica Containment (v2 降级)

按 v2 §7.5 (降级为 configuration concern):
$$\text{T5 ratio} = \frac{T_{\max}\sqrt{\eta D}}{L_{\text{basin}}}$$

代入: $T_{\max}=2.0, \eta=0.0005, D=1120, L_{\text{basin}}=8$ (v2 estimate; Day 1
实测 $L_{\text{basin}} \approx 20.45$ even tighter):

$$\text{T5 ratio} = \frac{2.0 \times \sqrt{0.56}}{8} = \frac{1.5}{8} \approx 0.187$$

按 v2 §7.5 判据: < 0.3 → GREEN.

**T5 judgment**: **GREEN** (0.187 < 0.3); even with Day 1 实测 $L_{\text{basin}}=20.45$,
T5 = 0.073 (still GREEN, more comfortable margin).

### 6.7 Stage V 综合判定矩阵 (Adam → SR reflow)

v3 reflow: replace v2's "Adam baseline" columns with SR primary + Adam secondary
(ablation). Single Tier 3 dual sub-criteria reflected.

| 条件 | Value | Judgment | Hard? |
|---|---|---|---|
| T1 | N/A (NN variant) | - | NO |
| **T2** | 20-160 (range, $\Delta E \in [0.1, 0.8]$) | **GREEN** ✓ | **YES** |
| T3 | 0.17 | GREEN ✓ | NO |
| **T4a** | predicted 1.5-2 | **GREEN** ✓ | **YES** |
| T4b | predicted 0.3-0.7 | GREEN ✓ | NO |
| T5 | 0.187 (or 0.073 w/ Day 1 L_basin) | GREEN | NO |

按 v2 §8.1 Hard Necessary Conditions: T2 GREEN ✓ + T4a GREEN ✓.

**两个 hard 条件都通过** → TCBM 至少有理论基础工作 ✓.

按 v2 §8.3 决策表: "T2 G + T4a G + T3 G + T4b G" → **STRONG GREEN, 全力做** ✓.

**v3 baseline reflow (4-tier success criteria layer)**:

| Tier | Metric | Primary baseline | Secondary baseline |
|------|--------|------------------|-------------------|
| 1 (feasibility) | rel_err < 10% | n/a (TCBM only) | n/a |
| 2 (NQS comparable) | rel_err_TCBM/rel_err_SR ≤ 2 | **SR** | n/a |
| 3 (a) (safety net) | σ_TCBM/σ_Adam ≤ 0.5 | n/a | **Adam** (5 seeds ablation) |
| 3 (b) (ceiling) | σ_TCBM ≤ 2× σ_SR | **SR** | n/a |
| 4 (mechanism) | n_kept ≥ 5/20 | **SR** (QGT comparison) | n/a |

### 6.8 R-abort triggers table (REWRITE for v3)

Full v3 R-abort table:

| Trigger | Condition | When measurable | Status (Day 4) |
|---------|-----------|-----------------|----------------|
| R-abort-1 | rel_err_TCBM > 15% vs ED truth | Week 1 end (Day 7) | inconclusive (等 Day 4 production v2) |
| R-abort-2 | swap_acc < 0.1 OR > 0.9 | Day 4 production v2 trajectory | inconclusive |
| R-abort-3 sub-criterion (a) | σ_TCBM/σ_Adam > 0.8 (TCBM not meaningfully better than unstable Adam) | Week 3 sweep (Day 12-13) | pending |
| R-abort-3 sub-criterion (b) | σ_TCBM > [TODO: calibrate Day 6 against SR baseline σ_SR; tentative 4× σ_SR pending data — corresponds to 2× over Tier 3 PASS bar of σ_TCBM ≤ 2× σ_SR] | Week 3 sweep (Day 12-13), pending Day 6 SR data | pending |
| R-abort-4 | T4a ψ < 1.1 by step 1000 (gradient buffer no low-rank structure) | Day 7 (15 min GPU) | pending Day 7 |
| R-abort-5 (NEW v3) | n_kept < 3 / 20 (TCBM clamping fully contained in SR's QGT subspace; mechanism claim C demoted) | Day 8 measurement | pending Day 8 |

**R-abort-5 narrative**: If TCBM's clamping subspace is fully contained within
SR's curvature subspace, TCBM is just SR with extra steps, not a complementary
technique. Mechanism claim (Tier 4) demotes to "TCBM matches SR robustness"
only (no orthogonal subspace narrative).

**R-abort-3 sub-criterion (b) threshold TODO**: Day 6 SR baseline produces single-seed
σ_SR estimate. Once σ_SR known, recalibrate sub-criterion (b) R-abort:
- If σ_SR ≈ 0.5%, R-abort at σ_TCBM > 2.0% (4× σ_SR baseline)
- If σ_SR ≈ 1.5%, R-abort at σ_TCBM > 6.0% (4× σ_SR baseline)
- Sub-criterion (b) PASS bar (σ_TCBM ≤ 2× σ_SR) stays fixed at 2×; only R-abort
  multiplier (4× tentative) is calibrated against actual σ_SR magnitude.

**Stage V 交付物**:

```
T conditions table (v3, NN variant):

  v2 hard necessary:
    T2 (WKB ratio):      20-160 (ΔE ∈ [0.1, 0.8] range) → GREEN ✓
    T4a (ψ existence):   predicted [1.5, 2.0] → GREEN ✓ (Day 7 verify)

  v2 soft conditions:
    T3 (swap acceptance): 0.17 → GREEN
    T4b (alignment):     predicted [0.3, 0.7] → GREEN

  v2 advisory (NN variant):
    T1: 30 (would be YELLOW under v5, N/A under NN variant)
    T5: 0.187 (GREEN, well within threshold)

  v3 R-aborts (5 triggers):
    R-abort-1 (Tier 1 fail): pending Day 4 evening verdict
    R-abort-2 (swap acc): pending Day 4 evening verdict
    R-abort-3(a) (vs Adam): pending Day 12-13 sweep
    R-abort-3(b) (vs SR): pending Day 12-13 sweep, threshold TODO Day 6
    R-abort-4 (T4a ψ): pending Day 7 diagnostic
    R-abort-5 (n_kept): pending Day 8 measurement

  Optimizer variant: NN variant (TCBMOptimizer NQS)

  Overall: STRONG GREEN per v2 §8.3 decision table (Tier 1-4 success
  criteria layer adds independent gates per §1.2).
```

---

## 7. 完整决策推导 (按 v2 §8)

### 7.1 应用 v2 §8.3 最终判定规则 (PRESERVE v2, baseline reflow only)

输入:
- T2: GREEN ✓
- T4a: GREEN (predicted, 70% confidence)
- T3: GREEN
- T4b: GREEN (predicted)

查 v2 §8.3 决策表:

| T2 | T4a | T3 | T4b | 总判定 | 建议 |
|---|---|---|---|---|---|
| **G** | **G** | **G** | **G** | **STRONG GREEN** | **全力做, 写论文主场景** |

**v3 baseline reflow**: "Adam baseline" references in v2 → "SR primary baseline +
Adam secondary (ablation)" throughout. The STRONG GREEN judgment is unaffected by
the baseline change (it's a Stage V T-condition outcome, not a baseline-comparison).

### 7.2 与 v2 §9 已知案例的最终对比

| 案例 | T2 | T4a | T3 | T4b | 总判定 | 实验结果 |
|---|---|---|---|---|---|---|
| DC-OPF | $10^6$ G | 3-10 G | 0.3 G | 0.85 G | STRONG G | p<0.001, d=1.16 |
| SCM | RED | RED | - | RED | RED | failed |
| ResNet-56-NS | G | **RED 1.0** | - | N/A | RED | failed |
| 1D TFIM | 2-20 Y | 1.2-1.6 Y | 0.25 G | 0.2-0.4 Y | YELLOW | not recommended |
| **NQS J1-J2 4×4** | **20-160 G** | **1.5-2 G (pred)** | **0.17 G** | **0.3-0.7 G (pred)** | **STRONG G** | **待 Day 7 验证 (T4a) + Day 4 production v2 (rel_err)** |

**位置定位**:
- 强于 1D TFIM 在每一个维度
- 弱于 DC-OPF 在 T4a/T4b (B 类 < A 类, 符合理论)
- 远强于 ResNet-NS (不存在 T4a RED 风险, 因为有 B 类代数结构 backup)
- 与 SCM 完全不同 (barrier 真实 vs 不存在)

### 7.3 Symptom 分类 (按 v2 §11)

| Symptom | ψ signature | 适用? |
|---|---|---|
| A: Premature commitment | N/A (TCBM 设计避免) | 否 |
| B: No commitment | N/A (TCBM 设计避免) | 否 |
| C: Subspace never stabilizes | ψ ~ 1.0 缓慢 | 否 (B 类预测 ψ 1.5-2) |
| C': Pipeline pollution | ψ ∈ [0.01, 10³] 跳跃 | 否 (NN variant 预防) |
| D: Structure absence | ψ ≡ 1.0 平坦 | 否 (B 类提供 substrate) |

**预测结果**: 不属于任何 failure symptom, 是 healthy "B 类带 backup" 案例.

如果 Day 7 实测意外发现 ψ ≡ 1.0 (D 类) 或 ψ 跳跃 (C' 类), 方法论 v2 在 NQS B 类
场景的预测被 falsify, 需要 v3 修订.

---

## 8. 核心证明结论与 ex ante 预测

### 8.1 总判定: **STRONG GREEN with Day 7 verification**

按 v2 §1.2 三层物理框架严格通过:

| Level | 条件 | NQS J1-J2 4×4 | 证据 |
|---|---|---|---|
| **Level 1**: 能量地形 | 真实 energetic barrier | **PASS** | Sign-rule trap (Szabó 2020) + Bukov 2021 Fig 11 spin-glass landscape; Day 3 Adam multi-basin verdict (5/5 seeds drift +1.7) |
| **Level 2**: 代数结构 | 梯度有稳定低秩结构 | **PASS** (predicted) | B 类 \|G\|=128, $A_1$-irrep dim 9, TCBM k=20 overcomplete 2.3× |
| **Level 3**: 物理尺度 | 维度/温度/basin 匹配 | **PASS** | T2 ≥ 20, T3=0.17, T5=0.19 全 GREEN |

**v3 supplementary verdict**: Day 3 Adam multi-basin signal (σ_Adam = 9.495%,
mean drift +1.705 across 5 seeds) supports 4×4 retention (no 6×6 pivot needed).
Adam ceiling 74.45% rel_err (vs Bukov 4×4 SR ~10⁻³) confirms SR is the meaningful
primary baseline.

### 8.2 Ex Ante 可证伪预测 (v3 anchored to Day 3 verified data)

按 v2 §1.1 哲学 ("直觉不可靠, 需要可证伪的预测"):

**预测 1 (T4a 主预测)**:
- ψ at warmup end 在 [1.5, 2.0] 区间 (Day 7 验证)
- Falsifiable: 如果 ψ < 1.1 持续, 方法论预测失败 → v3 修订

**预测 2 (T4b)**:
- $\max_i |\cos(v_{\text{escape}}^{A_1}, U_{\text{SVD}}[:,i])|$ 在 [0.3, 0.7]
  (Week 2 验证)
- Falsifiable: 如果 score < 0.1, B 类预测失败

**预测 3 (Robustness, sub-criterion (a) NEW v3)**:
- σ_TCBM ≤ 4.7% (corresponds to σ_TCBM/σ_Adam ≤ 0.5 with σ_Adam = 9.5% from
  Day 3 v2 anchor)
- Falsifiable: 如果 σ_TCBM > 7.6%, sub-criterion (a) fails

**预测 4 (Robustness, sub-criterion (b) NEW v3)**:
- σ_TCBM ≤ 2 × σ_SR (pending Day 6 SR baseline σ_SR; if σ_SR ≈ 0.5-1.5%,
  needs σ_TCBM ≤ 1-3%)
- Falsifiable: 如果 σ_TCBM > 4 × σ_SR, sub-criterion (b) fails

**预测 5 (Energy convergence, Tier 1)**:
- $|E_{\text{TCBM}} - E_0|/|E_0| < 0.10$ at training end (Day 4 production v2)
- Falsifiable: 如果误差 > 15%, R-abort-1 触发

**预测 6 (Mechanism, NEW v3)**:
- n_kept ≥ 5 / 20 SR-orthogonal directions (Day 8 measurement)
- Falsifiable: n_kept < 3, R-abort-5 triggers, mechanism claim demoted

### 8.3 与方法论 v2 §1.1 哲学呼应 (PRESERVE v2)

v2 §1.1 原文: "肉眼判断一个任务'看起来适合 TCBM'是不可靠的"

我们没有靠 intuition. 走完了 5 个 stages, 用了:
- Bukov 2021 (#103) 直接 quantitative anchor (verbatim quotes in `bukov_2021_anchor.md`)
- Szabó 2020 (#101) sign rule trap 机制
- Schulz 1996 / Sandvik 2007 ED reference values
- 对称群理论 ($T_4 \times T_4 \times C_{4v}$) 的严格分解
- 方法论 v2 已知案例 (DC-OPF, SCM, ResNet-NS) 的 calibration
- **v3 NEW**: Day 3 quick_adam_diagnostic_v2 5-seed verified data
  (σ_Adam = 9.5%, multi-basin verdict)

**STRONG GREEN 判定不是"我觉得能做", 而是"按 v2 严格走流程得到的判定"**.

如果 6 个 ex ante 预测中有 4 个以上失败, honest admission of methodology v2 在
B 类场景的局限性, v3 升级到 v4.

### 8.4 Day 7 决策依据 (明确 actionable)

**Go**: 如果 ψ ≥ 1.5 → 按 Protocol v2.1 继续 Week 2

**Adjust**: 如果 ψ ∈ [1.1, 1.5) → retry with grad_buffer 600, warmup 300; 若仍
YELLOW, retry with k=12

**No-Go**: 如果 ψ < 1.1 (T4a RED) → 触发 R-abort-4, 按 Protocol v2.1 §4.2(a) 切换 Plan B

**v3 Day 7 16:00 supplementary trigger**: If SR rel_err > 5% on 4×4 single seed
(Day 6 result), trigger Plan B (NetKet fork) — see §10.2 for details.

---

## 9. NEW: Adam → SR reframe rationale

This section documents why the v2 → v3 reframe was triggered, and why SR is the
correct primary baseline for this problem class.

### 9.1 Day 3 v2 verified data (anchor for reframe)

Source: `results/quick_adam_diagnostic_v2.json` (commit `d8e37b4`),
analysis: `experiments/analyze_diagnostic_post.py`.

**Configuration**: 5 seeds [42, 7, 13, 21, 99], Adam M=1, lr=0.001, betas=(0.9, 0.999),
n_steps=2000, n_vmc_samples=2000, n_final=4 (mean), 4×4 J1-J2 PBC J2/J1=0.5,
RNG-fixed (J1J2Problem.set_seed() per seed, post commit `18716b3`).

**Per-seed results**:

| Seed | best_during | final | drift | rel_err |
|------|-------------|-------|-------|---------|
| 42 | -4.35 | -3.53 | +0.82 | 58.30% |
| 7 | -2.91 | -1.32 | +1.59 | 84.40% |
| 13 | -4.05 | -2.08 | +1.98 | 75.47% |
| 21 | -3.89 | -2.46 | +1.43 | 70.86% |
| 99 | -4.13 | -1.42 | +2.71 | 83.22% |

**Statistical summary**:
- σ_Adam_abs = 0.8031 (cross-seed std of final cost)
- σ_Adam_rel = 9.495% (relative to |E_0|=8.4579)
- Mean rel_err = 74.45%, range [58.30%, 84.40%]
- Sorted final costs: [-3.53, -2.46, -2.08, -1.42, -1.32]
- Gap ratio = 0.48 (max gap 1.06 between seed 42 and rest cluster)
- Mean drift = +1.705 (5/5 seeds positive)

### 9.2 Why Adam ceiling matters

Bukov 2021 4×4 partial learning achieves $|E - E_{GS}|/N \approx 3 \times 10^{-4}$
(log|ψ| optimization, see §5.4 anchor table). Full learning expected $\approx 10^{-3}$.

**Adam vs SR gap**: Adam's 58-84% rel_err is **3-4 orders of magnitude worse**
than what SR achieves on the same problem. This is not a small calibration issue;
it's a fundamental capability gap. Vanilla Adam without natural-gradient
preconditioning cannot find the J1-J2 ground state at this lattice size.

**Implication for "TCBM finds ground state" claim**: If TCBM is compared only
to Adam, the comparison is uninformative — anyone who knows the literature will
ask "vs SR?" The reframe makes SR the primary comparison, with Adam retained
as ablation evidence.

### 9.3 Why SR is the NQS field standard

Stochastic Reconfiguration (Sorella 1998 PRB 57, R10001) is the canonical
natural-gradient method for variational quantum Monte Carlo. Field references:

- **Bukov 2021** (SciPost Phys 10.147): Uses SR + Runge-Kutta adaptive learning
  rate (Section 7, App A) for J1-J2 NQS optimization
- **NetKet** (open-source NQS framework): SR is the default optimizer for
  Heisenberg J1-J2 example (Examples/HeisenbergJ1J2/heisenbergJ1J2.py)
- **jVMC** (jax-based VMC): SR + min-SR variants in util/{tdvp,minsr}.py
- **Carleo-Troyer 2017** (Science 355, 602): Original NQS paper used SR

For any "TCBM vs NQS-standard" comparison, SR is the correct primary baseline.
Adam is a generic optimizer, not an NQS-tailored one.

### 9.4 Why Adam is retained as ablation

Adam provides a useful **secondary** baseline for SI demonstration:

- **Vanilla gradient descent failure mode**: Adam achieves only 26-42% accuracy
  (final mean -2.16 → -8.46 ratio, 26% best to 16% worst). SI plot of Adam
  trajectory vs TCBM/SR shows the value of natural-gradient preconditioning AND
  parallel tempering in NQS.
- **Cost**: 5 seeds × ~30 min = 2.5h (60% reduction from v2's 15-seed plan)
- **Implementation**: existing `experiments/run_adam_baseline.py` + Day 3
  `results/quick_adam_diagnostic_v2.json` data reused

This dual-baseline structure (SR primary + Adam secondary) is what makes v3
Tier 3 sub-criterion (a) (vs Adam, safety net) and sub-criterion (b) (vs SR,
narrative ceiling) both meaningful.

---

## 10. NEW: Reframe risks + Plan A/B/C hedges

This section documents the explicit risks of the Adam → SR reframe + planned
hedges for each failure mode.

### 10.1 Reframe risks

**Risk 1: SR may be too good (Tier 2 fail)**

Empirical scenario: SR achieves rel_err 0.1%, TCBM achieves 0.5% (5× worse) →
Tier 2 fail. But TCBM may still pass Tier 3 sub-criterion (a) (σ_TCBM/σ_Adam < 0.5).

**Verified Day 3 baseline (5 seeds, RNG-fixed)**:
- σ_Adam_rel = 9.495% (cross-seed std)
- Mean rel_err = 74.45%, range [58.30%, 84.40%]
- Multi-basin signature confirmed: 5/5 seeds show positive drift, mean +1.705
- 2-cluster structure: seed 42 in better basin (-3.53), other 4 seeds clustered
  in [-2.46, -1.32] worse basin

This provides quantitative anchor for Tier 3 dual-baseline:
- Sub-criterion (a) σ_TCBM/σ_Adam ≤ 0.5: with σ_Adam = 9.5%, needs σ_TCBM ≤ 4.7%.
  Even modest TCBM PT advantage (σ_TCBM ~1-3%) easily clears this safety-net bound.
- Sub-criterion (b) σ_TCBM ≤ 2× σ_SR: pending Day 6 SR baseline σ measurement.

**Narrative defense (if Tier 2 fails)**:

- **Empirical**: rel_err_TCBM 增加 X% but σ_TCBM 降低 50% — accuracy/robustness
  trade-off
- **Theoretical**: parallel tempering 引入的 inter-replica diversity 必然以一定
  convergence accuracy 换取 variance reduction (well-known in optimization theory)
- **Practical**: production NQS simulations (quantum chemistry, materials design)
  value reliability over single-shot accuracy (uncertainty quantification needs
  low σ over multiple runs, not lowest single E)
- **Field precedent**: simulated annealing vs gradient descent in classical
  optimization established this trade-off; TCBM brings it to NQS

**Required citation**: Day 6 SI prep includes literature search for one quantum
chem / materials NQS paper that emphasizes "reliability over accuracy" framing.

**Risk 2: n_kept may be 0**

If TCBM clamping subspace is fully contained in SR's QGT principal subspace,
mechanism story collapses. Day 8 measurement is binary signal — if n_kept = 0,
immediately consider switching to a different problem class (e.g., longer-range
J1-J2-J3) where the two subspaces could plausibly differ.

R-abort-5 triggers at n_kept < 3 (mechanism claim demoted to "TCBM matches SR
robustness only", no orthogonal-subspace narrative for paper claim C).

**Risk 3: SR implementation cost**

Implementing SR from scratch in PyTorch (`core/sr_optimizer.py`, 197-line skeleton
already in place) carries non-zero risk:
- QGT computation: $O(N \cdot D^2)$ memory ≈ 10 MB for D=1120 (safe)
- pinv solve: $O(D^3)$ time (D=1120 → manageable, but profiling needed Day 5)
- Numerical regularization (scale-aware ε·diag(S)): well-understood from Bukov 2021 App A

If SR rel_err > 5% on 4×4 single seed by Day 7 (likely impl bug), Plan B triggers.

### 10.2 Plan A/B/C trigger rules + execution

**Plan A (default, currently active)**:
- Day 5: implement `core/sr_optimizer.py` (4 helpers + SROptimizer.optimize)
- Day 5: pytest 5 tests including 2×2 trivial case ED comparison
- Day 6: `experiments/run_sr_baseline.py` single-seed 4×4 J2=0.5
- Day 7: Random null calibration for n_kept threshold (Step 1-2 of §3.2)
- Day 8: TCBM vs SR n_kept measurement (Step 3-4 of §3.2)
- Day 12-13: 15-seed SR sweep + 15-seed TCBM sweep + 5-seed Adam ablation

**Plan B (Day 7 16:00 trigger if SR rel_err > 5% on 4×4 single seed)**:

Switch to NetKet fork for SR baseline.
- Day 8: install NetKet, fork `Examples/HeisenbergJ1J2/heisenbergJ1J2.py`
- Day 9: adapt L=4, J2=0.5, run baseline in jax
- Day 10: extract data, compare with PyTorch TCBM (SI explains hybrid setup)
- Day 11-12: subspace overlap with NetKet SR data
- Cost: +2-3 days, schedule slips to Day 22-24
- Acceptable: paper still clean, just longer
- Reference: `/home/dglg/miao_workplace/netket_reference/`

**Plan C (Day 10 16:00 trigger if NetKet fork also fails)**:

Fallback to Adam-only paper.
- Reframe Adam ceiling as "negative result + mechanism contribution"
- Paper narrower: "TCBM PT mechanism studied, comparison with NQS-standard SR
  deferred to follow-up"
- Don't switch to this until Day 10 Plan B is provably stuck
- Schedule unchanged from Day 21 NMI cap, but paper scope narrower

**Decision rule summary**:
- Plan A → B at Day 7 16:00 if SR rel_err > 5% on 4×4 single seed
- Plan B → C at Day 10 16:00 if NetKet not running cleanly

**Schedule budget**:
- Original: Day 21 NMI submission cap
- Current Day 4 slip: 1 day (production v2 not launched Day 3)
- Buffer remaining: ~1 day
- Plan B reserve: +2-3 days (NetKet fork) → Day 22-24
- Plan C reserve: 不延期但 scope 缩窄

---

## 11. NEW: Disclosures and asymmetries (paper SI required)

This section documents 5 algorithmic/operational asymmetries between TCBM and the
Adam/SR baselines, all of which the paper SI methods section must disclose to
avoid reviewer concerns.

### 11.1 Final-eval target asymmetry (TCBM best_x vs Adam/SR final_θ)

**TCBM** (`run_gradient_baseline_v2.py`):
- Final eval target = `best_x` (the θ from the PT replica with lowest single-shot cost)
- Selection bias intrinsic to PT: best replica's θ is what the algorithm "picks"
- Mitigated by N=2 evaluations averaged at termination (NQS-1e debiasing, see §11.2)

**Adam** (`run_adam_baseline.py`):
- Final eval target = `final θ` (last optimizer step)
- Adam assumes monotonic convergence; selection bias would inflate apparent performance
- Final eval = 4-shot mean (no NQS-1e equivalent, see §11.3)

**SR** (`run_sr_baseline.py`, Day 6+ skeleton):
- Final eval target = `final θ` (matches Adam, no PT replicas to select among)
- Final eval = 4-shot mean

**Why not unified**: PT (M=12) needs replica selection by construction; Adam/SR
(M=1) don't have replicas to select among. Forcing both to "final θ" hides PT's
actual algorithmic output. Forcing both to "best across noisy replicas" introduces
selection bias on Adam/SR.

**Reference for handling**: Bukov 2021 reports final E from "lowest E across
multiple seeds at fixed config" (Section 7.1 plateau ≈ -0.5019), implying
selection-by-cost across runs is field standard for PT-like methods.

**Source**: `docs/known_issues_day2.md` final-eval asymmetry section
(commit `f2d8223`).

### 11.2 NQS-1e debiasing (TCBM only)

VMC noise → best-observed energy is biased low (selection effect). TCBM
re-evaluates `best_x` with `n_vmc_samples_final` samples to de-bias the reported
best_cost.

- TCBM: `best_cost_debiased = mean over n_vmc_samples_final shots`
  (see `core/tcbm_optimizer_NQS.py:1139` NQS-1e module docstring)
- Adam / SR: no equivalent (M=1, no replica selection to debias)

This is a TCBM-specific algorithmic step, not a generic optimization technique.
Paper SI must explain that `best_cost_debiased` for TCBM is constructed
differently from Adam/SR's `final_cost_mean`.

### 11.3 n_vmc_samples_final asymmetry (TCBM=2, Adam/SR=4)

**Verified Day 4 cfg grep** (sources cited):

| Script | Final-eval shot count | Source line |
|--------|----------------------|-------------|
| TCBM v2 (`run_gradient_baseline_v2.py`, Day 4 production) | **2** | line 170 (`n_vmc_samples_final=2, # ← Day 3 change (was 4 in v1)`) |
| TCBM v1 (`run_gradient_baseline.py`, Day 2 historical) | 4 | line 59 (original cfg, Day 2 6h22min run) |
| Adam baseline (`run_adam_baseline.py`) | 4 | line 55 (`N_FINAL_AVERAGES = 4`) |
| Adam diagnostic v2 (`quick_adam_diagnostic_v2.py`) | 4 | line 47 (`N_FINAL_SHOTS = 4`) |
| SR baseline (`run_sr_baseline.py`, Day 6+ skeleton) | 4 | line 58 (`N_FINAL_AVERAGES = 4`, matches Adam) |

**Asymmetry**: TCBM v2 final eval uses 2 shots; Adam/SR use 4 shots.

**Why TCBM=2**: Day 3 reduction from 4 to 2 saves ~30% wall-clock on the final
eval phase (Day 2 baseline final eval round-1 alone took ~57 min at 4 shots).
The 2-shot mean has higher variance per replica vs Adam/SR's 4-shot mean, but
TCBM's M=12 PT structure provides more total exploration before final eval.

**SI disclosure**: TCBM debiased estimate has σ_E ≈ 1.4× higher than Adam/SR's
4-shot equivalent (since σ scales as 1/√n_final). Disclosed for transparency;
does not bias mean estimate.

### 11.4 RNG bug timeline + fix (Day 3 evening)

**Discovery**: Day 3 16:44-17:30 UTC, via Adam diagnostic v1 byte-identical results.
Seed 42 final cost = -2.4822, seed 7 final cost = -2.4822 (identical to 4 decimal
places). Final shots `[-2.5053, -2.5002, -2.4331, -2.4904]` byte-identical for
both seeds.

**Root cause** (`core/j1j2_problem.py:319-320` pre-fix):
```python
self._gen = torch.Generator(device=device)
self._gen.manual_seed(0)   # users override via optimizer's seed
```

The comment "users override via optimizer's seed" was a lie — no optimizer reset
the internal generator. Used in 4 places: `random_feasible` init theta, VMC chain
init, Metropolis flip site, Metropolis accept log_u.

**Impact scope**: All Day 1-3 production runs effectively seed=0. Day 17-19
robustness sweep (15 seeds × 4 methods × 4×4 = 60 runs) would have produced
**identical results** → σ = 0 → entire sweep vacuous → paper claim B
(robustness)击穿 by reviewers.

**Fix** (commit `18716b3`):
```python
def set_seed(self, seed: int):
    """Reset internal RNG to given seed."""
    self._gen.manual_seed(seed)
    return self
```

Production scripts updated to call `problem.set_seed(SEED)` after instantiation
+ before `random_feasible(1)`. 3 new pytest tests in `tests/test_rng_seeding.py`:
- `test_problem_set_seed_changes_init` (different seeds → different theta)
- `test_problem_set_seed_reproducible` (same seed twice → identical theta)
- `test_evaluate_seeded_reproducible` (same seed + theta → same evaluate result)

All 23 tests pass post-fix (16 Day 1 + 4 Day 3 callback + 3 Day 3 RNG seeding).

### 11.5 Seed sequencing (pre-fix vs post-fix consistency)

Detailed timeline for paper SI methodology:

- **Day 1** (2026-04-27): σ_E sqrt-scaling test (`tmp_check_evaluate.py`)
  effectively seed=0 due to RNG bug. Test result valid since it tests internal
  Metropolis chain consistency, not cross-seed reproducibility.
- **Day 2** (2026-04-28): baseline run launched 09:06 UTC, SIGTERM at 15:28 UTC
  (6h22min wall, 0 bytes data salvaged due to engineering defects). Effectively
  seed=0 — but data lost regardless, so not a contamination issue.
- **Day 3 morning** (2026-04-29 06:00-13:00 UTC): Phase A forward work
  (callback hook + production v2 + micro-benchmark scripts). No production runs.
- **Day 3 afternoon** (2026-04-29 13:00-17:30 UTC): Adam baseline skeleton +
  Bukov anchor + Adam diagnostic v1 launched. v1 ran seed 42 + seed 7 with
  byte-identical results — RNG bug discovery.
- **Day 3 17:30-18:30 UTC**: Claude Code grep located bug at
  `core/j1j2_problem.py:319-320`. Fix + 3 pytest cases (commit `18716b3`).
- **Day 3 evening** (2026-04-29 19:00-21:30 UTC): Adam diagnostic v2 launched
  with `J1J2Problem.set_seed()` properly called per seed. 5 seeds [42, 7, 13,
  21, 99] sequential on GPU 3, ~2.5h wall. **First valid cross-seed data**.
- **Day 4+ (this work)**: All production runs use `J1J2Problem.set_seed(seed)`
  before `random_feasible(1)`. Verified by 3 pytest cases.

**Implication for paper SI**: All sweep results post-Day 3 18:00 UTC have valid
cross-seed reproducibility. Pre-fix data (`results/quick_adam_diagnostic.json`)
preserved in repo as RNG bug evidence (3 seeds byte-identical), `.gitignore` not
modified — `results/*.json` still ignored for future sweep artifacts.

**Sweep seed plan** (Day 12-13):
- 15 seeds × {TCBM-gradient, TCBM-hybrid, SR, Adam} = 60 runs total
- Same 15 seeds applied to all 4 methods (no inter-method seed asymmetry)
- 4× A10 GPUs × 4 runs parallel ≈ 2 days wall (12-13 runs per GPU at ~5-6h each)

---

## 附录 A: 完整 T 条件计算速查 (preserved from v2)

```
NQS J1-J2 4×4 at J2/J1=0.5:
  System: 16 sites, PBC, S_z=0 sector
  Hilbert dim: 65536
  Ansatz: Complex-RBM, α=2, M=32 hidden
  Optimizer: NN variant (TCBMOptimizer NQS)

Configuration:
  D = 1120
  T_min = 0.005, T_max = 2.0
  η (step_size) = 0.0005
  M_replicas = 12
  k = 20
  L_basin ≈ 8 (v2 estimate) / 20.45 (Day 1 实测)

T1 (NN variant: N/A): T1 ratio ≈ 30 under v5; N/A under NN variant.
T2 (WKB ratio): 20-160 (range, ΔE ∈ [0.1, 0.8]) → GREEN.
T3 (swap acceptance): 0.17 → GREEN.
T4a (ψ probe, predicted): 1.5-2.0, 70% GREEN, Day 7 verify.
T4b (alignment, predicted): 0.3-0.7, 70% GREEN, Stage 2 verify.
T5 (NN variant: GREEN): 0.187 (or 0.073 w/ Day 1 L_basin=20.45) → GREEN.

Summary:
  Hard conditions (T2, T4a): both GREEN ✓
  Soft conditions (T3, T4b): all GREEN ✓
  Advisory (T1, T5): N/A or GREEN ✓

Overall: STRONG GREEN per v2 §8.3
Action: Day 4 production v2 (Tier 1 verdict),
        Day 5+ SR implementation (Plan A, primary baseline),
        Day 7 ψ verification gate (T4a).
```

## 附录 B: 关键文献引用 (按 v2 §9 风格)

- **Bukov, Schmitt, Dupont 2021**, SciPost Phys. 10, 147
  - 提供 quantitative barrier evidence + Hessian 数据 + 4×4 partial learning
    achievable 10⁻⁴ + 6×6 σ_seed/|E| ≈ 0.34%
  - Stage II §2.2-2.3 + Stage IV §5.4 主 anchor
  - cards #103, verbatim quotes in `bukov_2021_anchor.md`
- **Sorella 1998**, PRB 57, R10001
  - Original Stochastic Reconfiguration formulation
  - §9.3 (SR field standard) + Day 5 implementation reference
- **Szabó & Castelnovo 2020**, PRR 2, 033075
  - Marshall sign rule trap 机制证据
  - Stage II §2.1 basin 构造依据
  - cards #101
- **Schulz, Ziman, Poilblanc 1996**, J. Phys. I 6, 675
  - 4×4 J1-J2 ED ground state energy reference (with anomalous finite size note)
- **Sandvik 2007**, PRL 98, 227202 — J1-J2 DMRG benchmark
- **Capriotti & Sorella 2000**, PRL 84, 3173 — Symmetry classification
- **Roth, Szabó, MacDonald 2023**, PRB 108, 054410
  - GCNN 证明对称群 hardcoding 有效, Stage III-A §3.3 B 类机制证据
  - cards #104
- **Bravyi & Terhal 2009**, SIAM J. Comp. 39, 1462
  - Stoquastic Hamiltonian 定义 (J1-J2 是 non-stoquastic, Stage II §2.1 基础)
- **Carleo & Troyer 2017**, Science 355, 602
  - Original NQS paper, used SR (§9.3 SR field standard)

## 附录 C: NEW v3 deliverable / experimental file inventory

**Production code (committed, post-RNG fix)**:
- `core/j1j2_problem.py` — added `set_seed()` method (commit `18716b3`)
- `core/tcbm_optimizer_NQS.py` — added `callback` parameter to `optimize()` (commit `7580cc6`)
- `experiments/run_gradient_baseline_v2.py` — production with three-pack robustness
  (commit `fddbc6f`)
- `experiments/run_adam_baseline.py` — Adam M=1 baseline (commit `d86bff9`)
- `experiments/quick_adam_diagnostic_v2.py` — RNG-fixed 5-seed Adam diagnostic
  (commit `d8e37b4`)
- `experiments/analyze_diagnostic_post.py` — 5-signal post-analysis (commit `d8e37b4`)

**Skeletons (NOT yet committed, Day 5+ implementation)**:
- `core/sr_optimizer.py` — 234-line skeleton with NotImplementedError, F1 forward work
- `experiments/run_sr_baseline.py` — 282-line skeleton, schema parity with Adam baseline

**Documentation (committed)**:
- `docs/bukov_2021_anchor.md` — verbatim quotes + Fig 11/12/8 readings
- `docs/known_issues_day2.md` — VMC cost, swap stale, ptrace, final-eval asymmetry
- `docs/NQS_J1J2_Prediction_v2.md` — original prediction (pre-reframe, v3 supersedes)
- `docs/NQS_J1J2_Prediction_v3_outline.md` — Day 3 reframe outline + Day 4 update
  (Step 4 commit `dc50b43`)
- `docs/daily_log_day3.md` — Day 3 detailed log (Tier 3 normalized commit `307597e`)

**Tests (committed)**: 23 passing total
- `tests/test_j1j2_problem.py` — 16 Day 1 tests
- `tests/test_callback.py` — 4 Day 3 callback tests
- `tests/test_rng_seeding.py` — 3 Day 3 RNG cross-seed reproducibility tests

**Results (force-added JSONs, evidence)**:
- `results/quick_adam_diagnostic.json` — v1 partial (3 seeds, RNG bug evidence,
  byte-identical reproducibility failure)
- `results/quick_adam_diagnostic_v2.json` — v2 full (5 seeds, ~580 KB with
  trajectories, primary anchor for v3)

---

**文档结束** (v3, ~1280 lines target)

> 本证明严格按方法论 v2 五阶段流程完成 + v3 baseline reframe (Adam → SR primary).
> 所有 6 个 ex ante 预测都有 falsifiability + Day-numbered verification gates.
> Day 7 T4a probe + Day 4 production v2 Tier 1 verdict + Day 6 SR baseline + Day 8
> n_kept measurement = the four critical experimental gates.
> 即使预测失败, 方法论自身的 falsifiability 价值 + RNG bug discovery (engineering
> lesson 1: cross-seed reproducibility test mandatory at Day 1) + Adam ceiling
> documentation (negative result for SI) 都有 standalone 价值.

> NQS J1-J2 4×4 = 方法论 v2 的第一个 ex ante prediction 验证案例 + v3 reframe
> 的 first-application Adam → SR. 这是 v2 升级到 v4 的关键 evidence-generating
> experiment (v3 = baseline + tier reframe; v4 trigger = Day 7-8 measurement
> outcomes if predictions falsified).
