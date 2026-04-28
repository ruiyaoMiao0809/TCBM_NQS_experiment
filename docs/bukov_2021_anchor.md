# Bukov 2021 Quantitative Anchor

**Source**: Bukov, Schmitt, Dupont 2021, SciPost Phys. 10, 147
**arXiv**: 2011.11214v3 (June 17, 2021)
**Fetched from**: https://arxiv.org/pdf/2011.11214 (Day 2, 2026-04-27)
**Purpose**: Calibration target for TCBM-NQS POC Week 3 robustness experiments

## Key quantitative findings

### System sizes used
- N = 4×4 (16 spins): "converges easily to the ground state" (Fig 2 caption)
- N = 6×6 (36 spins): main difficulty regime
- Both with periodic boundary conditions, J1=J2=0.5 maximally frustrated point

### Best variational energy achieved

For N=6×6:
- Plateau value E/N ≈ −0.5019 (Section 7.1)
- "energy density difference with the exact one is approximately 2·10^-3" (Section 7.3)
- This holds across multiple architecture / optimizer combinations

For N=4×4:
- "Minimal energy differences achieved with N = 4 × 4 are of the order of 10^−7" (Section 5.2 footnote 6)
- Note: this is full-basis simulation, not VMC; in VMC achieved ≤ 10^-3 (Fig 8 left)

### Seed-to-seed variability (most important for robustness comparison)

Section 6.1, Fig 12:
- 4 seeds, identical hyperparameters, N=6×6
- "different runs of the algorithm follow different trajectories"
- "they eventually get stuck in landscape saddles corresponding to different physical states"
- **Final energies spread approximately 10^-3 to 10^-2 in |E-EGS|/N** (read from Fig 12)
- **σ(E)/|E| 估计 0.5%-2%** ← 待 Nick 直接看图 12 确认精确值

### Training configuration (for replicating their setup if needed)

- NMC = 2^15 = 32768 samples per iteration (main configuration, Section 6.1, Fig 12)
- NMC ablation: 2^11, 2^13, 2^15 → "increasing the sample size does not necessarily lead to a lower energy" (Fig 9)
- Iterations: ~2000 for N=6×6, ~600 for N=4×4 to reach plateau
- Optimizer: SR + Runge-Kutta adaptive learning rate (Section 7, App A)

### Hessian spectrum analysis (mechanism evidence)

Section 5.2, Fig 11:
- N=4×4 at iteration 499 (E_GS = -8.457917):
  - Most eigenvalues positive (10^0 to 10^5 range)
  - Few small negative eigenvalues (10^-2 magnitude)
  - "flat directions on the variational manifold"
- N=6×6 at iteration 675 (E_GS = -18.073818):
  - Most eigenvalues positive
  - **Few large negative eigenvalues (10^2 magnitude)**
  - "highly curved sparse directions on the manifold, which are hard to find by the optimizer"
- Specific λ_min/λ_max ratio: NOT EXPLICITLY GIVEN; must be read from Fig 11

Note: Earlier protocol drafts (NQS_J1J2_Prediction_v2 §2.3) cited "|λ_min|/|λ_max| ≈ 0.2-0.3"
attributed to Bukov 2021 — this was a Claude (LLM) hallucination. Paper's Fig 11 does
not explicitly give this ratio; need direct figure inspection. From inset data:
- N=4×4: ~10^-2 / ~10^5 ≈ 10^-7 (mostly flat)
- N=6×6: ~10^2 / ~10^5 ≈ 10^-3 (some sparse curvature)

### Rugged landscape characterization

Section 6.1:
> "different runs end up in different saddles even when all hyperparameters are held fixed"
> "located in deep valleys, separated by high and difficult to overcome energy barriers"
> "This topography is reminiscent of spin glasses"

Section 7.1:
> "increasing the number of parameters in the ansatz is not guaranteed to lead to an
> improved variational energy at J2/J1 = 0.5"

## Implications for TCBM-NQS POC

### For Day 4-5 P0-1.2 (TCBM-gradient baseline)
- Target: |E - E_0| / |E_0| < 10% (P0-1.2 success)
- Bukov achieves ~0.4% on N=6×6 (Section 7.3); on N=4×4 even better
- **A 10% threshold is generous compared to Bukov's literature standard**
- This is appropriate for POC (mechanism validation, not SOTA accuracy)

### For Week 3 robustness (most important comparison)

Bukov anchor: σ(E) on identical config, different seeds, on N=6×6:
- **Spread roughly 10^-3 to 10^-2 in |E-EGS|/N → σ ≈ 0.5-2%** (待 Nick 看图 12 直接读取)
- They didn't compute aggregate σ; Fig 12 shows 4 individual curves

For TCBM POC main claim "σ(TCBM)/σ(Adam) ≤ 0.5":
- Adam on 4×4 likely gives σ smaller than 6×6 case (smaller system = less rugged)
- Reference target for 4×4 Adam: σ/|E| in range [0.5%, 5%] (我估计；不确定下限)
- TCBM target: σ/|E| ≤ 0.25-2.5%

### For mechanism interpretation
- Bukov diagnose "rugged landscape" + "saddle proliferation" via Fig 12 (seed scatter) + Fig 11 (Hessian)
- TCBM hypothesis: clamping subspace catches the few negative-curvature directions
- Direct connection: Fig 11's "few large negative eigenvalues" = TCBM's escape directions

## Items needing direct figure inspection

These require Nick or Claude with vision capability to directly read Bukov's figures:

1. Fig 12: extract precise final energies for the 4 different seeds → compute σ
2. Fig 11 N=6×6 panel: extract specific value of largest negative eigenvalue and largest positive eigenvalue → compute ratio
3. Fig 8 left panel (N=4×4, partial learning): extract final |E-EGS|/N value

These are not blockers for Day 2 baseline run; they are calibration-level details for SI write-up.

## Sources verification

- Paper accessed: 2026-04-27 via https://arxiv.org/pdf/2011.11214
- All quotes above are verbatim from paper (not LLM-paraphrased)
- All numbers above either explicit in paper text/tables or marked as "待图直接读取"
