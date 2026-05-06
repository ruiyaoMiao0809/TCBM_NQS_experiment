═══════════════════════════════════════════════════════════════
Week 1: Diagnostic + SR baseline implementation
═══════════════════════════════════════════════════════════════

Day 3 (DONE, 2026-04-29):
  ✓ Adam diagnostic v1 (5 seeds) — RNG bug 发现 (byte-identical results)
  ✓ Codebase-wide RNG fix (J1J2Problem.set_seed() + 3 pytest cases)
  ✓ Adam diagnostic v2 (RNG-fixed, 5 seeds) — STRONG multi-basin verdict
    - σ_Adam_rel = 9.495%, mean_drift = +1.705 (5/5 positive)
    - mean rel_err 74.45%, min rel_err 58.30% (Adam ceiling 真实数字)
  ✓ v3 outline reframe (Adam → SR primary baseline, ~250 lines)
  ✓ Forward work: callback hook + production v2 script + micro-benchmark
  ✓ Bukov Fig 11/12/8 anchor with Nick visual readings
  ✓ 9 commits, 23 tests passing

Day 4 (今天):
  Morning (Step 1-4):
    - Step 1: 仓库 + GPU state 核对 (DONE)
    - Step 2: Day 3 daily_log entry
    - Step 3: push pending commits to GitHub
    - Step 4: v3 outline Adam ceiling 数字 update (65% → 74.45% v2 verified)
  Afternoon (Step 5-7):
    - Step 5: full v3 doc 升级 (250 lines outline → ~800-1000 lines)
      • Tier 3 拆成 Tier 3a/3b dual-baseline (σ_TCBM/σ_Adam + σ_TCBM/σ_SR)
      • §11 final-eval asymmetry disclosure (TCBM best_x vs Adam/SR final_θ)
      • Bukov 数字直接从 docs/bukov_2021_anchor.md 引用
    - Step 6: launch production v2 TCBM-gradient (seed=42, GPU 3, ~5-6h wall)
    - Step 7: SR impl design + tests/test_sr_optimizer.py skeleton (parallel)
  Evening (Step 8-9):
    - Step 8: production v2 verdict
      • Tier 1 (rel_err < 10%): PASS / MARGINAL / R-ABORT-1
      • Tier 3a/3b deferred 到 Day 6 (SR baseline 数据 not yet available)
      • R-abort-2: swap_acc ∈ [0.1, 0.9]
    - Step 9: Day 4 daily_log + Day 5 path decision

Day 5:
  - SR core implementation (core/sr_optimizer.py)
    五个函数实现顺序: compute_log_derivatives → compute_qgt → 
    regularize_qgt → solve_sr_update → SROptimizer.optimize
  - pytest 5 tests (2×2 trivial case ED 对照, QGT Hermitian/PSD, etc.)
  - SR cfg initial tune (lr, ε scale-aware regularization)
  - 如 Day 4 production v2 是 MARGINAL: 先 retune (n_vmc 2000→4000, 
    warmup 150→300, ψ_star 1.5→1.3) 再开 SR

Day 6:
  - SR full run on 4×4 J2=0.5 single seed (~5-6h wall)
  - Verify SR 达到 Bukov 4×4 reference (~10⁻³ rel_err full learning)
  - σ_SR 暂时还没有 (single seed)，但 rel_err_SR 可以读
  - Tier 2 (rel_err_TCBM/rel_err_SR ≤ 2.0) 第一次可执行

Day 7:
  - SR final tuning
  - n_kept measurement infrastructure:
    • Step 1: random null calibration (50 pairs random k=20 subspaces in D=1120)
    • Step 2: cosine threshold 校准 (random_null_p95 ≈ 5)
    • Step 3: TCBM 与 SR QGT subspace 对比
    • Step 4: 统计检验 against null distribution
  ⚠ Plan B trigger Day 7 16:00 UTC: 如 SR rel_err > 5% → fork NetKet

═══════════════════════════════════════════════════════════════
Week 2: Production runs (TCBM + SR side-by-side)
═══════════════════════════════════════════════════════════════

Day 8-9:
  - TCBM-hybrid baseline (subspace_source='hybrid', single seed)
  - 验证 n_kept ≥ 5/20 (Claim C 第一个 evidence)
  - 与 Day 4 production v2 (gradient mode) 对比 — 看 hybrid 是否带来增益
  - Plan B 备选 path: Day 8 install NetKet, Day 9 fork 
    Examples/HeisenbergJ1J2/heisenbergJ1J2.py 到 L=4 J2=0.5

Day 10-11:
  - TCBM-hybrid 3-5 seeds preliminary sweep (σ 初读)
  - SR 3-5 seeds preliminary sweep (σ_SR 初读)
  - 中期 σ ratio 第一次可计算
  ⚠ Plan C trigger Day 10 16:00 UTC: 如 NetKet fork 也失败
                                       → Adam-only fallback paper

Day 12-13:
  - 主 sweep launch:
    • 15 seeds × TCBM-gradient
    • 15 seeds × TCBM-hybrid  
    • 15 seeds × SR
    • 5 seeds × Adam (Day 3 已有部分数据可复用)
  - 4× A10 GPUs × 4 runs 并行, 总 50 runs ≈ 2 天 wall
  - 每 run 5-6h, 即每 GPU 跑 12-13 runs

Day 14:
  - 中期 verdict (5-tier 全套):
    Tier 1: rel_err < 10%
    Tier 2: rel_err_TCBM/rel_err_SR ≤ 2.0
    Tier 3a: σ_TCBM/σ_Adam ≤ 0.5
    Tier 3b: σ_TCBM/σ_SR ≤ 2.0
    Tier 4: n_kept ≥ 5/20
  - 6×6 决策:
    4×4 结果强 → 不做 6×6, defer 到 NMI revision response
    4×4 结果弱 → 评估 vast.ai H100 ($50-150 USD) or 找 Hui Xiong 教授借 cluster

═══════════════════════════════════════════════════════════════
Week 3: Robustness + writing
═══════════════════════════════════════════════════════════════

Day 15-16:
  - sweep 完整数据收集与统计分析
    • σ ratios (3 个 ratio: TCBM/Adam, TCBM/SR, hybrid/gradient)
    • n_kept distribution (15 个 hybrid runs)
    • cluster gap analysis (验证 multi-basin claim 一致性)
  - subspace overlap analyzer 输出 (TCBM ⊥ SR 量化, Claim C 主图素材)

Day 17-18:
  - SI 写作:
    • Methods 4 sections (TCBM cfg / SR cfg / Adam cfg / measurement procedures)
    • Supplementary figures 8-10 张
    • Disclosure items 一节 (final-eval asymmetry, RNG bug timeline, 
      seed asymmetry, FE budget asymmetry from main TCBM project)
  - Bukov anchor 直接引用 docs/bukov_2021_anchor.md

Day 19-20:
  - Main figures 3-4 张 (NMI main text):
    • Fig 1: TCBM mechanism schematic + ψ trigger phase transition
    • Fig 2: 5-tier comparison bar chart (TCBM-grad / TCBM-hybrid / SR / Adam)
    • Fig 3: n_kept evidence (subspace overlap heatmap)
    • Fig 4 (optional): Bukov 4×4 vs our results overlay
  - Discussion section 起草

Day 21:
  - Buffer day
  - Submit to NMI 或 release as preprint
  - 如 Plan B 触发: Day 22-24 缓冲到位

═══════════════════════════════════════════════════════════════
关键日程节点 (Trigger conditions)
═══════════════════════════════════════════════════════════════

  Day 4 evening: production v2 Tier 1 verdict
                  PASS → Plan A 维持
                  MARGINAL → Day 5 retune 后续
                  R-ABORT-1 → 停下来重新评估 cfg / problem class

  Day 7 16:00 UTC: Plan B trigger
                    如 SR rel_err > 5% on 4×4 single seed
                    → fork NetKet HeisenbergJ1J2, 进度推 Day 22-24

  Day 10 16:00 UTC: Plan C trigger
                     如 NetKet fork 也失败 (Plan B 也死)
                     → Adam-only fallback paper, scope 缩窄

  Day 14: 6×6 决策点
           4×4 强 → defer 6×6 到 NMI revision response
           4×4 弱 → 评估 vast.ai H100 / HKUST-GZ cluster

  Day 21: NMI submission cap (或 Day 22-24 if Plan B 触发)

═══════════════════════════════════════════════════════════════
Buffer status
═══════════════════════════════════════════════════════════════

  原始 budget: 21 天
  Day 3 slip: 1 天 (production v2 未在 Day 3 launch)
  剩余 buffer: ~1 天
  Plan B 储备: +2-3 天 (NetKet fork) → Day 22-24
  Plan C 储备: 不延期但 scope 缩窄 (Adam-only fallback)

═══════════════════════════════════════════════════════════════
Open decisions (pending)
═══════════════════════════════════════════════════════════════

  1. n_kept threshold (5/20): Day 7 random null calibration 后才能定
  2. Tier 2 阈值 (rel_err_TCBM/rel_err_SR ≤ 2.0): Day 6 SR 数据来后再 review
  3. Adam ablation 5 vs 10 seeds: 暂定 5, Day 14 看够不够再决定
  4. Tier 3 dual-baseline: ✓ DECIDED (Tier 3a + Tier 3b, Day 4 早上拍板)
  5. 6×6 future work: defer 到 NMI revision response (除非 Day 14 verdict 弱)