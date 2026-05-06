# TCBM-NQS POC — Project Context for Claude Code

## Working with this project

Always activate the project conda env first:

```bash
conda activate tcbm_nqs
```

All commands (pytest, python, experiment launches) assume this env is active.
`tcbm_nqs` has all required deps (torch 2.7.1+cu118, pytest 9.0.3, etc).
The default login env (`tcbm`) does NOT have pytest installed, so direct
pytest calls in `tcbm` will fail.

When suggesting shell commands that involve pytest / python / experiment
launches, prefix with `conda activate tcbm_nqs && ...` if the user's session
may not have the env active.

## Project Overview

See `README.md` for full description. Brief: 3-week POC sprint extending TCBM
(Tunneling-Clamping Boltzmann Machine) optimizer to NQS variational ground-state
search on 2D J1-J2 Heisenberg 4×4 PBC at J2/J1=0.5. ED truth E_0 = -8.4579.

## Key Documents (in `docs/`)

- `TCBM_NQS_POC_Protocol_v2_1.md` — current protocol (start here)
- `NQS_J1J2_Prediction_v3.md` — current prediction doc (Adam → SR reframe)
- `NQS_experiment_plan.md` — daily experiment plan (5-tier dual-baseline)
- `daily_log.md` + `daily_log_dayN.md` — progress logs
- `known_issues_day2.md` — discovered issues + fix paths

## Convention

- Never modify `core/` files mid-experiment without explicit user approval.
- Long-running runs (>30 min) require `python -u` + SIGTERM handler + atexit
  JSON dump + 100-step micro-benchmark first.
- Production launches (5-6h wall) must run inside `tmux` to survive SSH disconnect.
