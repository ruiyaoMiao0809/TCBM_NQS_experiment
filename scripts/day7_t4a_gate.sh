#!/bin/bash
# day7_t4a_gate.sh — Day 7 T4a Early-Abort Diagnostic Gate (Protocol v2.1)
#
# Runs T4a diagnostic with automatic retry progression:
#   retry_level=0 (default) → if YELLOW, try 1 → if YELLOW, try 2 → if all fail, R-abort-4.
#
# Exit codes:
#   0 = GREEN at some retry level, continue Week 2
#   1 = All retry levels YELLOW, trigger R-abort-4
#   2 = RED at retry_level=0, trigger R-abort-4 immediately (fundamental limit)
#
# Usage:
#   cd /home/dglg/miao_workplace/tcbm_nqs
#   bash scripts/day7_t4a_gate.sh
#   # or for single retry level:
#   python experiments/run_t4a_diagnostic.py --retry_level 0

set -e

PROJECT_ROOT="/home/dglg/miao_workplace/tcbm_nqs"
cd "$PROJECT_ROOT"

echo "======================================================================"
echo "Day 7 T4a Early-Abort Diagnostic Gate (Protocol v2.1 §P0-1.4)"
echo "Start time: $(date)"
echo "======================================================================"
echo ""
echo "This gate determines whether TCBM's clamping mechanism has substrate"
echo "on 2D J1-J2 4x4. It runs up to 3 retry configurations (~45 min total)."
echo ""

# ──────────────────────────────────────────────────────────────────────────
# Helper: run one retry level
# ──────────────────────────────────────────────────────────────────────────

run_one() {
    local retry=$1
    local name=$2
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "  Attempt $((retry + 1))/3: retry_level=$retry ($name)"
    echo "  Expected time: ~15 minutes on A10 GPU"
    echo "────────────────────────────────────────────────────────────────"

    # Capture exit code without aborting the shell
    set +e
    python experiments/run_t4a_diagnostic.py --retry_level $retry
    local ec=$?
    set -e

    return $ec
}

# ──────────────────────────────────────────────────────────────────────────
# Main gate logic
# ──────────────────────────────────────────────────────────────────────────

# Try retry_level=0 (default config)
run_one 0 "default"
result=$?

case $result in
    0)
        echo ""
        echo "======================================================================"
        echo "✅ T4a GREEN at retry_level=0."
        echo "    Action: Continue to Week 2 as planned."
        echo "    Next: SR baseline (P1-1.2), then start Week 2 Day 8."
        echo "======================================================================"
        exit 0
        ;;
    2)
        echo ""
        echo "======================================================================"
        echo "❌ T4a RED at retry_level=0 (ψ < 1.1)."
        echo "    Action: TRIGGER R-ABORT-4. Skip further retries."
        echo ""
        echo "    Retry levels 1 and 2 adjust buffer size and k, but cannot"
        echo "    recover from fundamental absence of low-rank structure."
        echo "    (See Protocol v2.1 §4.2 for Plan B wording.)"
        echo "======================================================================"
        exit 2
        ;;
    1)
        echo ""
        echo "⚠️ T4a YELLOW at retry_level=0 (ψ in [1.1, 1.5))."
        echo "   Trying retry_level=1 (enlarged buffer)..."
        ;;
    *)
        echo ""
        echo "❓ Unknown exit code $result from diagnostic script."
        echo "   Manual investigation required."
        exit 3
        ;;
esac

# Try retry_level=1 (enlarged buffer)
run_one 1 "enlarged_buffer"
result=$?

case $result in
    0)
        echo ""
        echo "======================================================================"
        echo "✅ T4a GREEN at retry_level=1."
        echo "    Action: Use enlarged buffer config for Week 2/3 runs."
        echo "    Add to Protocol: grad_buffer_size=600, subspace_warmup=300"
        echo "======================================================================"
        exit 0
        ;;
    1)
        echo ""
        echo "⚠️ T4a YELLOW at retry_level=1."
        echo "   Trying retry_level=2 (reduced k)..."
        ;;
    2|*)
        echo ""
        echo "❌ T4a worsened or failed at retry_level=1."
        echo "   TRIGGER R-ABORT-4."
        exit 1
        ;;
esac

# Try retry_level=2 (reduced k)
run_one 2 "reduced_k"
result=$?

case $result in
    0)
        echo ""
        echo "======================================================================"
        echo "✅ T4a GREEN at retry_level=2 (reduced k)."
        echo "    Action: Use reduced k config for Week 2/3 runs."
        echo "    Add to Protocol: k=12, grad_buffer_size=600, subspace_warmup=300"
        echo "======================================================================"
        exit 0
        ;;
    *)
        echo ""
        echo "======================================================================"
        echo "❌ All 3 retry levels failed to achieve ψ ≥ 1.5."
        echo "    TRIGGER R-ABORT-4."
        echo ""
        echo "    Summary of attempts:"
        ls -la results/week1_t4a_diagnostic_retry*.json 2>/dev/null || true
        echo ""
        echo "    Action: Switch to Plan B. Prepare downgraded SI per Protocol §4.2(a)."
        echo "    Rationale: Complex-RBM ansatz may lack sufficient symmetry-induced"
        echo "    low-rank structure for TCBM clamping. Future work should consider"
        echo "    GCNN (Roth et al. PRB 2023) which hardcodes symmetries."
        echo "======================================================================"
        exit 1
        ;;
esac
