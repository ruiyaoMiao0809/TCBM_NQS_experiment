# TCBM-NQS POC 项目启动指南

**服务器路径**: `/home/dglg/miao_workplace/tcbm_nqs/`
**工期**: 2026-04-25 (Day 1) 至 2026-05-15 (Day 21)
**Driver**: Claude Code
**监督**: Nick

---

## 1. 项目文件架构（Day 0 建立）

```
/home/dglg/miao_workplace/tcbm_nqs/
├── README.md                              # 项目总览
├── environment.yml                        # conda env spec
├── .gitignore
├── pyproject.toml                         # 可选：pip-installable package
│
├── core/                                  # 核心算法代码
│   ├── __init__.py
│   ├── tcbm_optimizer_NQS.py              # TCBM-NQS optimizer (Day 0 已交付, updated)
│   ├── j1j2_problem.py                    # Problem 类骨架 (Day 0, updated)
│   └── device.py                          # GPU device utility (Day 1 写)
│
├── baselines/                             # Baseline optimizers
│   ├── __init__.py
│   ├── adam_baseline.py                   # Day 6 (P1-1.1)
│   └── sr_baseline.py                     # Day 8 (P1-1.2)
│
├── utils/                                 # 工具模块
│   ├── __init__.py
│   ├── ed_reference.py                    # Day 3 (P0-1.3): scipy sparse ED
│   ├── seed_utils.py                      # Day 1: 全局 seed 管理
│   └── logging_utils.py                   # Day 1: 日志工具
│
├── experiments/                           # 实验脚本（每个对应 Protocol 的 P0/P1）
│   ├── run_t4a_diagnostic.py              # Day 7 AM (P0-1.4)  ← Day 0 已交付
│   ├── run_gradient_baseline.py           # Day 4-5 (P0-1.2)
│   ├── run_subspace_compare.py            # Day 10-14 (P0-2.2~4)
│   ├── run_robustness_sweep.py            # Day 16-18 (P0-3.2)
│   └── run_j2j1_scan.py                   # Day 20 (P2-3.1)
│
├── analysis/                              # 后处理/绘图
│   ├── plot_psi_evolution.py              # T4a 诊断可视化 (integrated in run_t4a_diagnostic)
│   ├── plot_convergence.py                # 三模式对比
│   ├── plot_robustness_box.py             # 最终主图
│   ├── compute_subspace_overlap.py        # subspace angle 分析
│   └── compute_T4b_alignment.py           # T4b cos similarity
│
├── results/                               # 实验结果（CSV/JSON/图）
│   ├── week1_t4a_diagnostic_retry0.json   # Day 7 输出
│   ├── week1_gradient_baseline.csv
│   ├── week2_three_modes.csv
│   ├── week3_robustness_sweep.csv
│   └── figures/
│       ├── t4a_psi_evolution_retry0.png   # Day 7
│       ├── fig_s1_three_modes_convergence.pdf
│       ├── fig_s2_robustness_box.pdf
│       └── fig_s3_qgt_spectrum.pdf
│
├── docs/                                  # 文档（Day 0 填充）
│   ├── TCBM_NQS_POC_Protocol_v2_1.md      # 当前协议（v2.1）
│   ├── NQS_J1J2_Prediction_v1.md          # 预测分析
│   ├── T4a_early_abort_supplement.md      # T4a 精化
│   ├── TCBM_Task_Applicability_Methodology_v1.md  # 方法论
│   ├── tcbm_literature_cards_99-110_NNQuantumState.md  # 文献卡片
│   └── daily_log.md                       # 每日进展日志
│
├── tests/                                 # 单元测试
│   ├── test_j1j2_problem.py               # Day 3 (P0-1.1 单元测试)
│   └── test_tcbm_optimizer_nqs.py         # Day 4 可选
│
├── scripts/                               # 便利脚本（每周一键启动）
│   ├── week1_setup.sh                     # 建目录 + sanity checks
│   ├── day7_t4a_gate.sh                   # Day 7 一键 diagnostic
│   └── week3_robustness_sweep.sh          # 夜间 sweep 启动
│
└── old/                                   # 归档：废弃文件
    ├── 1D_TFIM_prediction_v1.md           # 已废弃（stoquastic，task 不适合）
    └── TCBM_NQS_POC_Protocol_v2_0.md      # 已被 v2.1 替代
```

---

## 2. Day 0 文件部署清单

### 2.1 需要从本次 session 的 outputs 拷贝的文件

以下文件从 `/mnt/user-data/outputs/` 拷贝到服务器对应位置:

| Source (outputs/) | Destination (tcbm_nqs/) |
|---|---|
| `tcbm_optimizer_NQS.py` (updated) | `core/tcbm_optimizer_NQS.py` |
| `j1j2_problem.py` (updated) | `core/j1j2_problem.py` |
| `run_t4a_diagnostic.py` | `experiments/run_t4a_diagnostic.py` |
| `TCBM_NQS_POC_Protocol_v2_1.md` | `docs/TCBM_NQS_POC_Protocol_v2_1.md` |
| `NQS_J1J2_Prediction_v1.md` (updated) | `docs/NQS_J1J2_Prediction_v1.md` |
| `T4a_early_abort_supplement.md` | `docs/T4a_early_abort_supplement.md` |
| `NQS_Ising_Prediction_v1.md` | `old/NQS_Ising_Prediction_v1.md` (归档) |
| `TCBM_NQS_POC_Protocol.md` (v1) | `old/TCBM_NQS_POC_Protocol_v1.md` (归档) |
| `TCBM_NQS_POC_Protocol_v2.md` (v2.0) | `old/TCBM_NQS_POC_Protocol_v2_0.md` (归档) |
| `TCBM_literature_cards_99-110_NNQuantumState.md` | `docs/tcbm_literature_cards_99-110_NNQuantumState.md` |

### 2.2 需要自己准备的文件

这些文件不是 session 输出，需要从你本地或其他项目取:

| 文件 | 来源 |
|---|---|
| `TCBM_Task_Applicability_Methodology_v1.md` | 你上传到 session 的文件，可从本地 copy |
| `core/device.py` | 从 `tcbm_cyberphysical_v2/core/device.py` 简单 adapt |

### 2.3 需要当场新建的文件

| 文件 | 内容 | 来源 |
|---|---|---|
| `README.md` | 项目总览（见 §3） | 下文 |
| `environment.yml` | Conda env spec | 下文 §4 |
| `.gitignore` | Python + IDE 标准 | 下文 §5 |
| `core/__init__.py` | Empty | - |
| `baselines/__init__.py` | Empty | - |
| `utils/__init__.py` | Empty | - |
| `experiments/__init__.py` | Empty | - |
| `docs/daily_log.md` | 每日日志模板 | 下文 §6 |
| `scripts/week1_setup.sh` | 建目录 + sanity | 下文 §7 |

---

## 3. README.md 模板

创建 `/home/dglg/miao_workplace/tcbm_nqs/README.md`:

```markdown
# TCBM-NQS Proof-of-Concept

2D J1-J2 Heisenberg 4×4 frustrated spin system, TCBM with 3 subspace sources.

**Task**: 2D J1-J2 at J2/J1=0.5, Lx=Ly=4, PBC
**Ansatz**: Complex-RBM, α=2, D=1120
**Goals**:
1. TCBM converges to within 10% of ED ground state
2. Three subspace sources (gradient/qgt/hybrid) show distinguishable behavior
3. σ(TCBM)/σ(Adam) ≤ 0.5 on 15-seed robustness study

**Status**: Day 0 (2026-04-24)
**Protocol**: `docs/TCBM_NQS_POC_Protocol_v2_1.md`

## Quick Start

```bash
cd /home/dglg/miao_workplace/tcbm_nqs
conda env create -f environment.yml
conda activate tcbm_nqs
pytest tests/test_j1j2_problem.py -v
```

## Week 1 Timeline

Day 1-2: Build J1J2Problem
Day 3: ED reference, unit tests
Day 4-5: TCBM-gradient baseline
Day 6: Adam baseline
**Day 7: T4a diagnostic gate** (go/no-go)
Day 8: SR baseline (Week 2 preparation)

## Red Lines (abort conditions)

- R-abort-1: Week 1 end gradient error > 15%
- R-abort-2: Week 2 end QGT > 90s per step
- R-abort-3: Week 3 σ(TCBM)/σ(Adam) > 0.8
- **R-abort-4: Day 7 ψ < 1.1 (T4a RED)**

## Contact

Nick, HKUST-GZ Xiong Lab.
```

---

## 4. environment.yml

```yaml
name: tcbm_nqs
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.11
  - pytorch>=2.1
  - pytorch-cuda=12.1
  - numpy>=1.24
  - scipy>=1.10
  - pandas>=2.0
  - matplotlib>=3.7
  - seaborn
  - pytest>=7.0
  - tqdm
  - pyyaml
  - pip
  - pip:
    - netket>=3.12  # optional, only if GCNN extension
```

**创建命令**:
```bash
conda env create -f environment.yml
conda activate tcbm_nqs
```

---

## 5. .gitignore

```
# Byte-compiled
__pycache__/
*.pyc
*.pyo

# IDE
.idea/
.vscode/
*.swp

# Results (large files, not versioned)
results/*.csv
results/*.json
results/figures/*.pdf
results/figures/*.png

# Environment
.venv/
*.egg-info/

# Logs
*.log

# Checkpoints
*.pt
*.ckpt

# Jupyter
.ipynb_checkpoints/
```

---

## 6. docs/daily_log.md 模板

```markdown
# Daily Log

## Day 1 (2026-04-25)

### 计划
- [ ] Set up project structure (P0 prerequisite)
- [ ] Copy & verify Day 0 skeleton files (core/, experiments/)
- [ ] Start J1J2Problem implementation (P0-1.1, lattice + RBM)

### 实际完成
- [ ] ...

### 关键数字
- Hilbert dim: 65536
- RBM D: 1120
- ED E_0/N target: ~-0.497

### 发现/问题
- ...

### 红线状态
- R-abort-1: pending (Week 1 end)
- R-abort-2: pending (Week 2 end)
- R-abort-3: pending (Week 3 end)
- R-abort-4 (T4a): pending (Day 7 gate)

### 明日
- ...

---

## Day 2 (2026-04-26)

[template...]

---
```

---

## 7. scripts/week1_setup.sh

```bash
#!/bin/bash
# week1_setup.sh — Day 0 setup verification
#
# Creates directory skeleton, verifies Python imports, runs quick sanity check.

set -e  # exit on error

PROJECT_ROOT="/home/dglg/miao_workplace/tcbm_nqs"
cd $PROJECT_ROOT

echo "======================================================================"
echo "TCBM-NQS Week 1 Setup"
echo "======================================================================"
echo ""

# Step 1: verify directory structure
echo "[1/5] Verifying directory structure..."
for dir in core baselines utils experiments analysis results results/figures \
           docs tests scripts old; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "   Created: $dir/"
    fi
done
echo "  ✓ All directories present"

# Step 2: verify core files
echo ""
echo "[2/5] Verifying core files..."
for f in core/tcbm_optimizer_NQS.py core/j1j2_problem.py \
         experiments/run_t4a_diagnostic.py; do
    if [ ! -f "$f" ]; then
        echo "  ❌ MISSING: $f"
        echo "     Please copy from /mnt/user-data/outputs/"
        exit 1
    fi
done
echo "  ✓ Core files present"

# Step 3: verify Python environment
echo ""
echo "[3/5] Verifying Python environment..."
python -c "import torch; print(f'  torch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
python -c "import scipy; print(f'  scipy: {scipy.__version__}')"
python -c "import numpy; print(f'  numpy: {numpy.__version__}')"

# Step 4: GPU check
echo ""
echo "[4/5] GPU check..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'  ✓ GPU: {torch.cuda.get_device_name(0)}')
    print(f'  ✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('  ⚠️ WARNING: No GPU detected, will run on CPU (much slower)')
"

# Step 5: Smoke test
echo ""
echo "[5/5] Smoke test: import J1J2Problem and initialize..."
python -c "
import sys
sys.path.insert(0, '.')
from core.j1j2_problem import J1J2Problem
prob = J1J2Problem(Lx=4, Ly=4, alpha=2, device='cpu')
print(f'  ✓ J1J2Problem ok: D={prob.dim}')
theta_batch = prob.random_feasible(2)
print(f'  ✓ random_feasible ok: shape={theta_batch.shape}, '
      f'range=[{theta_batch.min():.3f}, {theta_batch.max():.3f}]')
"

echo ""
echo "======================================================================"
echo "✅ Week 1 setup complete. Ready for Day 1 implementation."
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Review docs/TCBM_NQS_POC_Protocol_v2_1.md"
echo "  2. Start Day 1 task P0-1.1: J1J2Problem full implementation"
echo "  3. Log progress in docs/daily_log.md"
```

---

## 8. scripts/day7_t4a_gate.sh

```bash
#!/bin/bash
# day7_t4a_gate.sh — Day 7 T4a diagnostic one-click runner (Protocol v2.1)
#
# Runs retry_level=0 first. If YELLOW, runs 1. If still YELLOW, runs 2.
# If 2 also YELLOW or any RED, signals abort.

set -e

PROJECT_ROOT="/home/dglg/miao_workplace/tcbm_nqs"
cd $PROJECT_ROOT

echo "======================================================================"
echo "Day 7 T4a Early-Abort Diagnostic Gate (Protocol v2.1 §P0-1.4)"
echo "Start time: $(date)"
echo "======================================================================"
echo ""

run_one() {
    local retry=$1
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "  Running retry_level=$retry..."
    echo "────────────────────────────────────────────────────────────────"

    python experiments/run_t4a_diagnostic.py --retry_level $retry
    local exit_code=$?
    return $exit_code
}

# Try level 0 (default)
set +e
run_one 0
result=$?
set -e

if [ $result -eq 0 ]; then
    echo ""
    echo "✅ T4a GREEN at retry_level=0. Continue Week 2 plan."
    exit 0
elif [ $result -eq 1 ]; then
    echo ""
    echo "⚠️ T4a YELLOW at retry_level=0. Trying retry_level=1..."
    set +e
    run_one 1
    result=$?
    set -e
    if [ $result -eq 0 ]; then
        echo "✅ T4a GREEN at retry_level=1."
        exit 0
    elif [ $result -eq 1 ]; then
        echo "⚠️ Still YELLOW. Trying retry_level=2..."
        set +e
        run_one 2
        result=$?
        set -e
        if [ $result -eq 0 ]; then
            echo "✅ T4a GREEN at retry_level=2."
            exit 0
        fi
    fi
fi

echo ""
echo "❌ T4a FAIL after all retries. TRIGGER R-ABORT-4."
echo "   Switch to Plan B: downgrade POC to theoretical extension + diagnostic."
exit 1
```

---

## 9. 明天（Day 1）的工作流

### 9.1 开工前（09:00，预计 30 min）

```bash
cd /home/dglg/miao_workplace/tcbm_nqs

# 1. 拷贝 Day 0 交付文件（见 §2.1 清单）
# 从本地 laptop 或 outputs/ 拷贝

# 2. 建 conda env
conda env create -f environment.yml
conda activate tcbm_nqs

# 3. 建 git repo（若未建）
git init
git add -A
git commit -m "Day 0: Initial skeleton + Protocol v2.1"

# 4. 运行 Day 0 setup script
chmod +x scripts/week1_setup.sh
./scripts/week1_setup.sh
```

**预期**: setup.sh 最后打印 "✅ Week 1 setup complete"

### 9.2 Day 1 主要工作（09:30 - 17:00）

**任务**: P0-1.1 (J1J2Problem 类完整实现)

**具体步骤**:

1. **09:30-12:00**: 补全 `j1j2_problem.py` 的实现细节
   - 检查 `_log_cosh_complex` 的数值稳定性（在大 |u| 下）
   - 补充缺失的边界 case（empty VMC samples, inf cosh）
   - 添加 log-derivative gradient estimator (可选)
2. **13:00-15:00**: 写 `tests/test_j1j2_problem.py`
   - 加入 j1j2_problem.py 末尾 `if __name__` 块里那 7 个 test 作 pytest 版本
   - 加 ED cross-check: `test_evaluate_exact_matches_ed_small()` (N=8 system)
3. **15:00-17:00**: 跑 tests, fix bugs, commit

### 9.3 Day 1 结束自检

```bash
pytest tests/test_j1j2_problem.py -v
```

所有 test 通过，Day 1 P0 任务完成。

### 9.4 每天结束流程

```bash
# 1. 更新 daily_log
vim docs/daily_log.md

# 2. Commit
git add -A
git commit -m "Day N: [brief description]"

# 3. 如果关键里程碑，push (如果有 remote)
# git push
```

---

## 10. Claude Code 启动指令（给你明天早上用）

**对话起点**:

```
项目: TCBM-NQS POC
路径: /home/dglg/miao_workplace/tcbm_nqs
今日: Day 1 (2026-04-25)
主任务: P0-1.1 (J1J2Problem 完整实现)

先读 docs/TCBM_NQS_POC_Protocol_v2_1.md 了解协议。
然后读 core/j1j2_problem.py 的骨架。
按 §1.2 P0-1.1 的成功判据完成 Day 1 任务。

注意事项：
1. Complex-RBM 的 log_cosh 在 |u| > 30 时需要 stable 形式（骨架 _log_cosh_complex 的 TODO）
2. random_feasible 的 init_scale=0.01 关键，不要改
3. 测试 σ_E ∝ 1/√N_samples 是 P0 判据
4. 任何偏离协议的决定写进 docs/daily_log.md

下班前：pytest tests/test_j1j2_problem.py -v 全绿，git commit。
```

---

## 11. 关键提醒

1. **Day 7 T4a gate 是最重要的 gate**。把 `scripts/day7_t4a_gate.sh` 放在 desktop 或 home screen 方便那天一键跑

2. **Week 3 nighttime runs**: 要在 Day 15 下班前启动，因为 sweep 需要 3 天 + 分析 2 天 = 5 天

3. **Plan B wording** 已经写好（Protocol v2.1 §4.2）。如果 Day 7 RED，**不要自责**，直接切 Plan B——这就是 T4a gate 的价值

4. **GPU 独占性**: 如果服务器共享，确认 Day 7 和 Week 3 sweep 期间 GPU 有排他预约

5. **备份**: 每晚结束 `rsync` 一次 `results/` 到另一个存储位置（POC 的数据非常珍贵）

---

**本文档结束**

> 按这个工作流，从 Day 0 到 Day 21 的每一步都有 clear action item。明天 09:30 拷完文件 → setup script 通过 → Claude Code 按上面指令开工，到 Day 7 上午 T4a gate 检查，决定是否继续 Week 2。三周后 Day 21 交付 SI 草稿。

**祝实验顺利！**
