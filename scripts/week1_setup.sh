#!/bin/bash
# week1_setup.sh — Day 0 setup verification for TCBM-NQS POC
#
# Creates directory skeleton, verifies Python imports, runs quick sanity check.
#
# Usage:
#   cd /home/dglg/miao_workplace/tcbm_nqs
#   bash scripts/week1_setup.sh
# or:
#   chmod +x scripts/week1_setup.sh
#   ./scripts/week1_setup.sh

set -e   # exit on any error

PROJECT_ROOT="/home/dglg/miao_workplace/tcbm_nqs"
cd "$PROJECT_ROOT"

echo "======================================================================"
echo "TCBM-NQS Week 1 Setup — $(date)"
echo "======================================================================"
echo ""

# ──────────────────────────────────────────────────────────────────────────
# Step 1: verify directory structure
# ──────────────────────────────────────────────────────────────────────────
echo "[1/6] Verifying directory structure..."
for dir in core baselines utils experiments analysis results results/figures \
           docs tests scripts old; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "       Created: $dir/"
    fi
done

# Create __init__.py in Python package dirs
for pkg in core baselines utils experiments analysis tests; do
    if [ ! -f "$pkg/__init__.py" ]; then
        touch "$pkg/__init__.py"
    fi
done
echo "       ✓ All directories and package __init__.py present"

# ──────────────────────────────────────────────────────────────────────────
# Step 2: verify core files (critical — abort if missing)
# ──────────────────────────────────────────────────────────────────────────
echo ""
echo "[2/6] Verifying core files..."
MISSING=0
for f in core/tcbm_optimizer_NQS.py \
         core/j1j2_problem.py \
         core/device.py \
         experiments/run_t4a_diagnostic.py \
         utils/ed_reference.py \
         docs/TCBM_NQS_POC_Protocol_v2_1.md \
         docs/NQS_J1J2_Prediction_v1.md; do
    if [ ! -f "$f" ]; then
        echo "       ❌ MISSING: $f"
        MISSING=1
    fi
done
if [ $MISSING -eq 1 ]; then
    echo ""
    echo "       Required files missing. Please copy from /mnt/user-data/outputs/"
    echo "       See docs/PROJECT_SETUP_GUIDE.md §2 for the full list."
    exit 1
fi
echo "       ✓ All core files present"

# ──────────────────────────────────────────────────────────────────────────
# Step 3: verify Python environment
# ──────────────────────────────────────────────────────────────────────────
echo ""
echo "[3/6] Verifying Python environment..."
python -c "
import sys
print(f'       Python: {sys.version.split()[0]}')
import torch
print(f'       torch:  {torch.__version__} (CUDA: {torch.cuda.is_available()})')
import scipy
print(f'       scipy:  {scipy.__version__}')
import numpy
print(f'       numpy:  {numpy.__version__}')
try:
    import matplotlib
    print(f'       matplotlib: {matplotlib.__version__}')
except ImportError:
    print('       ⚠️ matplotlib not installed (needed for T4a plots)')
"

# ──────────────────────────────────────────────────────────────────────────
# Step 4: GPU check
# ──────────────────────────────────────────────────────────────────────────
echo ""
echo "[4/6] GPU check..."
python -c "
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'       ✓ GPU: {props.name}')
    print(f'       ✓ Memory: {props.total_memory / 1e9:.1f} GB')
    print(f'       ✓ CUDA: {torch.version.cuda}')
else:
    print('       ⚠️ WARNING: No GPU detected. Will run on CPU (much slower).')
"

# ──────────────────────────────────────────────────────────────────────────
# Step 5: Smoke test — import J1J2Problem
# ──────────────────────────────────────────────────────────────────────────
echo ""
echo "[5/6] Smoke test: import J1J2Problem and initialize..."
python -c "
import sys
sys.path.insert(0, '.')
from core.j1j2_problem import J1J2Problem
# Use CPU for quick test
prob = J1J2Problem(Lx=4, Ly=4, alpha=2, device='cpu')
print(f'       ✓ J1J2Problem ok: D={prob.dim}')
print(f'       ✓ box_constraint: {prob.box_constraint} (should be False for NQS)')
theta_batch = prob.random_feasible(2)
print(f'       ✓ random_feasible ok: shape={theta_batch.shape}, '
      f'range=[{theta_batch.min():.3f}, {theta_batch.max():.3f}]')
"

# ──────────────────────────────────────────────────────────────────────────
# Step 6: Smoke test — TCBM optimizer imports & mock run
# ──────────────────────────────────────────────────────────────────────────
echo ""
echo "[6/6] Smoke test: TCBM optimizer import..."
python -c "
import sys
sys.path.insert(0, '.')
from core.tcbm_optimizer_NQS import TCBMConfig, TCBMOptimizer
cfg = TCBMConfig(M=2, n_steps=10, k=4, subspace_warmup=5,
                 psi_star=2.0, min_warmup=3, record_every=2,
                 subspace_source='gradient', n_vmc_samples=None)
print(f'       ✓ TCBMConfig built: box_constraint default = {cfg.box_constraint}')
"

echo ""
echo "======================================================================"
echo "✅ Week 1 setup complete. Ready for Day 1 implementation."
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Review docs/TCBM_NQS_POC_Protocol_v2_1.md"
echo "  2. Day 1 task P0-1.1: Complete J1J2Problem implementation"
echo "     (see skeleton's TODO comments in core/j1j2_problem.py)"
echo "  3. Day 3 task P0-1.3: Run ED reference:"
echo "     python utils/ed_reference.py --J2 0.5"
echo "  4. Log progress in docs/daily_log.md"
echo ""
echo "Red lines to watch:"
echo "  R-abort-1: Week 1 end gradient error > 15%"
echo "  R-abort-2: Week 2 end QGT > 90s/step"
echo "  R-abort-3: Week 3 σ(TCBM)/σ(Adam) > 0.8"
echo "  R-abort-4: Day 7 ψ < 1.1 (T4a RED)"
echo ""
