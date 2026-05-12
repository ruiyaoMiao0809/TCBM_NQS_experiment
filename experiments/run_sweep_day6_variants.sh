#!/bin/bash
# Day 6 Phase 2.5 4-variant sweep launcher.
#
# Variants (PT temperature ladder cfg, all else unchanged from production v2):
#   A: T_min=0.005, T_max=1.0  ratio=200 (lower hot end only)
#   B: T_min=0.05,  T_max=2.0  ratio=40  (raise cold end only)
#   C: T_min=0.05,  T_max=1.0  ratio=20  (double, tight ladder)
#   D: T_min=0.1,   T_max=1.0  ratio=10  (aggressive narrow ladder)
#
# seed=42, n_steps=3000, ~15h wall each, parallel on GPU 0/1/2/3.

set -e
cd /home/dglg/miao_workplace/tcbm_nqs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tcbm_nqs

LAUNCH_TIME=$(date -u +%Y%m%d_%H%M)
echo "Sweep launch time (UTC): $LAUNCH_TIME"
echo ""

launch_variant() {
    local variant=$1
    local gpu=$2
    local tmin=$3
    local tmax=$4

    local session="sweep_${variant}_day6"
    local run_name="sweep_v_${variant}_day6_seed42"
    local logfile="logs/sweep_${variant}_day6_${LAUNCH_TIME}.log"

    tmux kill-session -t "$session" 2>/dev/null || true
    tmux new -s "$session" -d
    tmux send-keys -t "$session" \
        'source ~/miniconda3/etc/profile.d/conda.sh && conda activate tcbm_nqs && cd /home/dglg/miao_workplace/tcbm_nqs' Enter
    sleep 1
    tmux send-keys -t "$session" \
        "TCBM_DEVICE=cuda:${gpu} TCBM_N_STEPS=3000 TCBM_RUN_NAME=${run_name} TCBM_T_MIN=${tmin} TCBM_T_MAX=${tmax} python -u experiments/run_gradient_baseline_v2.py 2>&1 | tee ${logfile}" Enter

    echo "Variant ${variant} launched: GPU ${gpu}, T_min=${tmin}, T_max=${tmax}, tmux=${session}, log=${logfile}"
}

launch_variant a 0 0.005 1.0
launch_variant b 1 0.05  2.0
launch_variant c 2 0.05  1.0
launch_variant d 3 0.1   1.0

echo ""
echo "All 4 variants launched. Monitor:"
echo "  tmux ls | grep sweep_"
echo "  ls -la results/sweep_v_*_day6_seed42*"
echo "  tail -f logs/sweep_*_day6_${LAUNCH_TIME}.log"
