# TCBM-NQS Proof-of-Concept

Extension of TCBM (Tunneling-Clamping Boltzmann Machine) optimizer to Neural
Quantum State (NQS) variational ground-state search on frustrated spin systems.

**Target task**: 2D J1-J2 Heisenberg model on 4×4 square lattice at $J_2/J_1=0.5$
(maximally frustrated point, non-stoquastic, following Bukov et al. SciPost
Phys. 10, 147 (2021)).

**Ansatz**: Complex-RBM, hidden density $\alpha=2$, $D=1120$ real parameters.

## Goals

1. **Feasibility**: TCBM converges to within 10% of ED ground state energy on
   4×4 J1-J2 at $J_2/J_1=0.5$.
2. **Mechanism**: Three subspace sources (`gradient` / `qgt` / `hybrid`) show
   distinguishable behavior, demonstrating that TCBM clamping provides
   directional information orthogonal to the natural-gradient metric (QGT).
3. **Robustness**: $\sigma(\text{TCBM}) / \sigma(\text{Adam}) \leq 0.5$ across
   15 random seeds, with $p < 0.05$ from Levene's test.

## Quick Start

```bash
cd /home/dglg/miao_workplace/tcbm_nqs

# 1. Create conda env
conda env create -f environment.yml
conda activate tcbm_nqs

# 2. Run Day 0 setup verification
bash scripts/week1_setup.sh

# 3. (Day 3) Compute ED reference
python utils/ed_reference.py --J2 0.5

# 4. (Day 7 AM) Run T4a diagnostic gate
bash scripts/day7_t4a_gate.sh
```

## Project Structure

```
tcbm_nqs/
├── core/                   TCBM optimizer + J1-J2 problem class
├── baselines/              Adam, SR baselines
├── utils/                  ED reference, seed utils, logging
├── experiments/            Day-numbered experimental scripts
├── analysis/               Plotting & post-processing
├── results/                CSV/JSON outputs + figures
├── docs/                   Protocols, predictions, methodology
├── tests/                  Unit tests (pytest)
├── scripts/                One-click runners
└── old/                    Archived / superseded files
```

## Documents (in `docs/`)

- **`TCBM_NQS_POC_Protocol_v2_1.md`** — current protocol (**start here**)
- **`NQS_J1J2_Prediction_v1.md`** — pre-experimental prediction (GREEN judgment)
- **`T4a_early_abort_supplement.md`** — T4a/T4b methodology refinement
- **`TCBM_Task_Applicability_Methodology_v1.md`** — general applicability framework
- **`daily_log.md`** — progress log (updated daily)

## Timeline

| Week | Days | Main tasks |
|------|------|-----------|
| 1 | 1-7 | Infrastructure + gradient baseline + **Day 7 T4a gate** |
| 2 | 8-14 | QGT implementation + hybrid mode comparison |
| 3 | 15-21 | Robustness sweep + SI draft |

## Red Lines (abort conditions)

- **R-abort-1**: Week 1 end, gradient error > 15%
- **R-abort-2**: Week 2 end, QGT > 90s per step
- **R-abort-3**: Week 3 end, $\sigma(\text{TCBM})/\sigma(\text{Adam}) > 0.8$
- **R-abort-4**: Day 7, $\psi(t) < 1.1$ (T4a RED)

## Dependencies

- Python 3.11
- PyTorch ≥ 2.1 (with CUDA 12.1 ideally)
- scipy ≥ 1.10, numpy ≥ 1.24
- pandas, matplotlib, seaborn

See `environment.yml` for full spec.

## Citation (for papers that use this codebase)



## Contact

