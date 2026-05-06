# Day 4 Daily Log

**Date**: 2026-04-30  
**Phase**: Week 1, Day 4 (Production v2 launch — diverted to NaN diagnosis)  
**Codebase**: `/home/dglg/miao_workplace/tcbm_nqs/`  
**GPU**: 4× NVIDIA A10 24GB (HKUST-GZ lab server)  
**Commits**: TBD (Day 4 closing batch pending)  
**Tests**: 23 passing (Day 1 + Day 3 baseline，无 regression；Day 4 未新增 test)

---

## 1. 计划 vs 实际

**Day 4 原计划** (Day 3 daily_log handoff):
```
Morning (Step 1-4): 仓库 + GPU 状态核对、Day 3 daily_log、push commits、
v3 outline Adam ceiling 数字 update
Afternoon (Step 5-7): full v3 doc 升级 (250→800-1000 行)、launch production v2
TCBM-gradient (5-6h wall)、SR impl design + tests skeleton (parallel)
Evening (Step 8-9): production v2 verdict (Tier 1 + R-abort-2)、Day 4 daily_log +
Day 5 path decision
```

**Day 4 实际 path**:
```
Morning (06:00-11:30 UTC):

Step 1-4 全部按计划完成
顺手 update docs/NQS_experiment_plan.md 反映 v3 reframe (含 5-tier dual-baseline + Plan A/B/C trigger)

Late morning (11:45 UTC):

launch production v2 TCBM-gradient on GPU 3 (seed=42)

Afternoon (14:24 UTC, +2h39min):

发现 stall: cost_min/cost_mean/sigma_E_max 跨 step 100→400 byte-identical
kill 进程 (PID 727231, SIGTERM 即生效)
备份 stall log + tmux scrollback dump 作 evidence

Afternoon-Evening (15:00-17:00 UTC, ~3h diagnostic):

Diagnostic 第一击: narrow_band_diagnostic.py Stage A (subspace 强制关闭的 Langevin baseline)
→ step 87 grad_nan first occurrence; theta_nan 始终 False (Metropolis 拒绝 NaN move)
→ root cause 锁定在 NQS layer，与 TCBM subspace / SVD / clamp 完全无关
Diagnostic 第二击: probe_grad_nan.py synthetic ladder (||θ|| ∈ [1.15, 30])
→ 11/11 全 clean，norm-only 假设否决
→ bug 不是 "params 变大就爆"，是 trajectory-specific direction structure
Diagnostic 第三击: probe_on_nan inline mode (modified narrow_band)
→ step 87 触发 NaN 瞬间 inline probe 抓取所有 forward intermediate
→ 显式 save-on-failure + sys.exit 绕过 atexit defect (production v2 + narrow_band 两次确诊)
→ 锁定 first non-finite intermediate: log_2cosh.log(inner) 在 core/j1j2_problem.py:228

Late evening (17:30 UTC+):

root cause 确认 + Day 4 收尾 + Day 5 plan adjustment
```

**偏离原计划的原因**: production v2 stall 触发深度诊断 (~3h)。Step 5 (full v3 doc) / Step 7 (SR impl prep) / Step 8 (Tier 1 verdict) 全部推迟。Schedule slip 累计 = 2 天 (Day 3 1天 + Day 4 1天)，buffer 用尽。Plan B 线 (Day 7 16:00 UTC) 仍可达。

---

## 2. 关键数字

### 2.1 Production v2 stall trace

| Step | cost_min | cost_mean | sigma_E_max | swap_acc | psi_max | Time |
|------|----------|-----------|-------------|----------|---------|------|
| 0 | 10.9770 | 11.8056 | 0.0468 | 1.000 | 0.000 | 11:45 |
| 100 | -1.6900 | 3.6852 | 0.0769 | 1.000 | 0.000 | ~12:00 |
| 200 | -1.6900 | 3.6852 | 0.0769 | 0.727 | 0.000 | ~12:35 |
| 300 | -1.6900 | 3.6852 | 0.0769 | 0.545 | 0.000 | ~13:10 |
| 400 | -1.6900 | 3.6852 | 0.0769 | 0.455 | 0.000 | 14:02 |
| (kill) | — | — | — | — | — | 14:24 |

cost_min/cost_mean/sigma_E_max 跨 300 step **byte-identical**。VMC 采样有 RNG，sigma_E_max 不变 = chain 不在跑或 chain 在 frozen theta 上重复采样。swap_acc 单调下降 (0.727→0.455) 但 cost 不动，是 PT swap 在反复但 theta 已锁死的特征。

### 2.2 Stage A subspace-off baseline

| Step | ||θ|| | grad_finite | cost_min |
|------|-------|-------------|----------|
| 0 | (initial) | ✓ | 10.98 |
| 50 | 15.37 | ✓ (||grad||=16.13) | (下降中) |
| 87 | (NaN trigger) | ✗ grad_nan=True | -1.6900 |
| 100 | 19.39 | ✗ grad_nan=True | -1.6900 (frozen) |
| 102 | (kill) | ✗ | -1.6900 (frozen) |

Stage A subspace 强制关闭 (subspace_warmup=999, freq=999) 仍在 step 87 触发 grad NaN，证明 root cause 在 NQS layer 不在 TCBM subspace 机制。

### 2.3 Synthetic ladder (norm-only 假设否决)

| ||θ|| | verdict | ||grad|| |
|-------|---------|----------|
| 1.15 | CLEAN | 0.718 |
| 5.0 | CLEAN | 0.873 |
| 10.0 | CLEAN | 0.069 |
| 15.0 | CLEAN | 0.071 |
| 17.0 | CLEAN | 0.265 |
| 18.0 | CLEAN | 0.332 |
| 19.0 | CLEAN | 0.588 |
| 20.0 | CLEAN | 0.139 |
| 22.0 | CLEAN | 0.165 |
| 25.0 | CLEAN | 0.109 |
| 30.0 | CLEAN | 0.035 |

11/11 全 clean，包括 ||θ||=30 远超真实 step 87 的 ~6.21。scaled-random direction 不能复现 trajectory 创造的 anisotropic structure，bug 是 trajectory-specific 不是 norm-specific。Synthetic 阶 ||grad|| 在 0.03-0.87 量级，真实 Stage A step 50 的 ||grad||=16.13 是 synthetic 的 20-460×。

### 2.4 probe_on_nan smoking gun (root cause)

| 项 | 值 |
|----|-----|
| Step trigger | 87 |
| NaN replica index | 5 (out of M=12) |
| ||θ|| at replica 5 | **6.2146** (反直觉地小) |
| cost (forward) | -1.3828, all finite ✓ |
| grad (backward) | NaN, **34/1120 entries** |
| First non-finite forward intermediate | `log_2cosh.log(inner)` |
| Failure precision | shape [2000, 32] → exactly **1 element = -inf**, 0 NaN |
| Source location | `core/j1j2_problem.py:228` |
| state.pt size | 114 KB (lean) |
| probe.json size | 4 KB (lean, vs 178 MB old version) |

64,000 个 hidden_act elements 中 exactly 1 个 element 触发 inner = 0 → log(inner) = -inf → forward 的 sample 被 VMC 排除 (cost 仍 finite) → 但 backward 的 1/inner = 1/0 = inf → 链式法则 inf × finite = inf → inf - inf = NaN，污染 34 个 grad entries。

---

## 3. 发现 / 问题

### 3.1 NaN bug 因果链 (root cause)

**位置**: `core/j1j2_problem.py:228`，`_log_2cosh_complex` 函数。

**机制**:
1. RBM forward 计算 `log(2·cosh(z))`，z = u + iv 是复数 hidden activation
2. 代码用数学等价的稳定 form: `log_magnitude = |u| + 0.5·log(inner)`，其中 `inner = (1+exp(-2|u|))² - 4·sin²(v)·exp(-2|u|)`
3. 当 u≈0 且 v≈π/2 + kπ: `a = exp(-2|u|) ≈ 1`, `(1+a)² ≈ 4`, `4·sin²(v)·a ≈ 4·1·1 = 4` → `inner = 4 - 4 = 0` (float32 rounding 给精确 0)
4. `log(inner) = -inf` → forward 该 sample 的 |ψ|² = 0 被 VMC 自然排除，cost 仍 finite
5. backward 的链式法则要算 `d(log inner)/d(theta) = (1/inner) · d(inner)/d(theta)` → `1/0 = +inf` → `inf × finite = inf` → 后续 `inf - inf = NaN` 或 `0 × inf = NaN`
6. NaN gradient 进 Langevin step → forces NaN → Metropolis 全拒 → theta 一直停在 last-good value → cost 报告 frozen

**作者预言但误判**: `core/j1j2_problem.py:185-203` 注释里作者明确说:
- "inner is strictly positive except at the cosh zeros z = i·(π/2 + k·π), where inner = 0 and log diverges to -∞ (correct analytic behavior)"
- "No epsilon clamp is added — clamps inject zero gradients..."
- "For NQS training in the POC scope (init_scale=0.01, 3000 steps), hidden activation imag parts stay close to 0 and never approach π."

作者预测 POC scope 下 imag 部分 stay close to 0 不会触发。**实测 step 87 (远早于 3000) 已触发**——POC scope 的 M=12 replica × 2000 VMC samples × 32 hidden units = 768K hidden_act elements/step 把单点共振概率放大到 ~50 step 必撞一次。

### 3.2 Atexit defect 系统性确诊

**两次独立 run 100% 触发**:
- production v2 (run 1, kill 后): atexit dump 未触发，无 partial JSON
- narrow_band_diagnostic Stage A (run 2, Ctrl-C 后): atexit dump 未触发，无 partial JSON

**根因怀疑**: SIGINT/SIGTERM 在 deep CUDA call 期间投递可能 bypass Python signal handler 的 atexit chain。Python signal handler 注册在 Python interpreter 层，但 CUDA kernel launch 后控制权交给 GPU driver，Python signal handler 要等下一个 Python bytecode tick。如果 SIGTERM 直接 terminate 进程或 SIGINT 被某种方式 short-circuit，atexit 不会跑。

**Day 17-19 sweep 风险评估**: 60 runs (15 seeds × 4 methods)，按当前 atexit defect，任何 mid-run interrupt 的 partial trajectory 全失。统计上 60 runs 里 ≥3-5 个会需要 interrupt (OOM、cluster maintenance、误操作)，潜在重跑成本 15-30 hour。**Day 5 必须正式 fix**。

**临时 workaround 已 validated**: probe_on_nan mode 用 "在 callback 检测异常时立即显式 torch.save + sys.exit(0)" 的 pattern 完全绕过 atexit defect。validate run 跑通 (state.pt + probe.json 都落盘 + 进程干净退出)。Day 5 正式 fix 前所有新 experiment 用这个 pattern 暂代。

### 3.3 production v2 cost frozen 机制重新解读

之前假设: grad NaN → theta 被污染成 NaN → cost frozen 是 NaN 被 cache。

**实际机制 (诊断后修正)**: grad NaN → forces NaN → Metropolis acceptance 全拒 → theta 冻在 last-good step (~87) → VMC 在 frozen theta 上反复采样 → cost 报告 step 87 附近的真实 cost (-1.6900) → byte-identical 是 frozen theta 上 deterministic 部分的 cost (sigma_E_max 也字节一致是 VMC chain RNG 在 frozen theta 上的某种 cache 效应，非 chain 不跑)。

**operational implication**: cost = -1.6900 不是 garbage cache 值，是 RBM 在该 trajectory step 87 附近的真实 cost (rel_err = 80%)。这是 NaN bug fix 之后 production v2 应能突破的下限锚点。

### 3.4 Synthetic vs trajectory: 数值稳定性的 anisotropy

synthetic ||θ||=30 vs 真实 step 87 ||θ||=6.21，数值稳定性 inverted——high-norm random direction 比 low-norm trajectory direction 更稳定。意味着 gradient descent 创造的方向结构 (W 的 top 奇异向量、a-W 列对齐、bond-pair 不对称) 是 hidden_act imag 部分接近 π/2 的真正驱动。这种 trajectory-specific structure 不能用 random scaling 复现，未来类似诊断要避免依赖 norm-only sampling。

---

## 4. 红线状态

| Trigger | 状态 | 备注 |
|---------|------|------|
| R-abort-1 (rel_err > 15% vs ED) | inconclusive | production v2 stall，未拿到合法 verdict |
| R-abort-2 (swap_acc < 0.1 or > 0.9) | inconclusive | swap_acc 0.727→0.455 在合法范围，但 theta frozen 让 swap_acc 数据无意义 |
| R-abort-3a (σ_TCBM/σ_Adam > 0.8) | pending | 等 Day 12-13 sweep |
| R-abort-3b (σ_TCBM/σ_SR > 0.8) | pending | 等 Day 12-13 sweep, SR data Day 6+ |
| R-abort-5 (n_kept < 3) | pending | 等 Day 8 hybrid 测量 |

Day 4 evening 没拿到任何合法 verdict。Tier 1 verdict 推迟到 Day 5 evening (NaN fix + production v2 重跑后)。

---

## 5. 反思 (6 lessons)

### 5.1 数值奇点的 prevalence 比 init_scale=0.01 注释预测的高 ~3 个数量级

源码作者基于 "init_scale 小 + 训练步数有限" 预测 hidden_act imag 不会接近 π/2。预测错的根因是 **没把 sample dimension 放进概率计算**: M=12 × 2000 samples × 32 hidden = 768K elements/step，单点共振 even at 1e-5 概率 also 期望 ~50 step 必撞。教训: **数值奇点风险评估必须乘 sample dimension，不只看参数初始化 scale**。任何带数学 singularity 的稳定 form (log + clamp、divide + epsilon) 都要按 sample dimension 算累计概率。

### 5.2 诊断假设要尽早伪造，避免沉没成本

我最初的假设是 "step 100 SVD fail → NaN gradient → frozen"，被 step 87 grad_nan 数据 (早于 SVD warning) 否决。第二个假设 "high ||θ|| → cosh overflow"，被 synthetic ladder 11/11 clean 否决。**两次假设都在 30 min 内否决并 pivot，没有沉没成本陷阱**。教训: **每个诊断假设都要写出 falsification criterion，先伪造再 elaborate**。

### 5.3 inline probe 比 staged save+probe 信息密度高 ~3×

最初设计是 "Stage 1 rerun 拿 step 87 theta → save state → Stage 2 probe load state"，3-step pipeline，每个 interface 有信息丢失风险 (cfg drift、device mismatch、tensor storage 共享 bug)。改成 inline probe-on-nan (NaN 触发瞬间当场 probe + save + exit) 是 2-step pipeline，少一个 interface，且省 ~30 min 二次 launch 时间。教训: **诊断 pipeline 的 step 数和 interface 数正相关于 bug propagation 风险，能 inline 就 inline**。

### 5.4 Atexit defect 触发 100% 后 still 用 atexit 是认知惯性

production v2 atexit 没 fire 时，narrow_band 设计还在依赖 atexit 三件套。在 narrow_band 也 fail 后才认识到 atexit defect 是系统性。教训: **任何 "保护机制" 在第一次 fail 时就要假设 systematic defect，不要等第二次 confirmation**。第一次 atexit fail 时就该立刻在 narrow_band 设计里替换成显式 save-on-failure + sys.exit。

### 5.5 NaN 触发位置 ||θ||=6.21 出乎预期，反直觉是好事

预测 NaN region 是 ||θ|| ∈ [15, 19]，实际 NaN 在 replica 5 ||θ||=6.21 触发——比预期低 3×。反直觉的根因是 **NaN 是单点共振，不是连续分布**: 12 replicas 各自有自己的 trajectory，replica 5 比其他 replica 更早撞到 hidden_act imag ≈ π/2 的共振点，与 ||θ|| 没强关系。教训: **多 replica 系统的诊断必须 per-replica 分析，aggregate metrics (||θ|| 总览、cost_mean) 会掩盖 single-replica catastrophe**。Day 5 起 narrow_band 默认 per-replica recording。

### 5.6 源码注释里的 "predicted but judged safe" warnings 必须当作红旗

`core/j1j2_problem.py:185-203` 里作者明确预言了 inner=0 的 cosh zero 边界 + 主动选择不加 clamp。这种 "我知道有 risk 但 judge it as acceptable" 的注释在 production-grade NQS 里**就是隐藏 bug 的黄金线索**。教训: **codebase 接手时 grep 所有 "POC scope"、"should not occur"、"unlikely"、"in practice" 类注释做 risk register**，不接受作者的 risk judgment 作为真理。Day 5 起对 j1j2_problem.py 完整 scan 这类注释，建一个 numerical_risks_register.md。

---

## 6. Day 5 plan adjustment

Day 5 工作内容**完全重写**。原计划 SR core implementation + pytest 推迟到 Day 6+。

**Day 5 上午 (~3-4h, 必须做)**:
1. **Atexit defect 正式 fix** (~1h):
   - `experiments/run_gradient_baseline_v2.py` 加显式 save-on-failure handler
   - 所有现有 experiment scripts review，确保有同款保护
   - 写 1 pytest 验证 SIGTERM 后 partial state 落盘
2. **NaN bug fix** (~2-3h):
   - 方案选择: custom autograd Function for `_log_2cosh_complex`
   - forward 保持原样 (log(inner) = -inf 在 inner=0 是正确 analytic behavior, |ψ|²=0 让 sample 自动从 VMC 排除)
   - backward 显式处理 inner→0 极限: 返回 0 gradient 而不是 1/eps
   - 数学和 physics 一致性: cosh=0 → ψ=0 → sample 对训练贡献为 0 → gradient 贡献也应为 0
   - verify: 用 `results/state_at_first_nan.pt` load 原本 NaN 的 state，fix 后 backward 给 finite gradient

**Day 5 下午 (~5-6h, 必须 launch)**:
3. **Production v2 重跑** (5-6h wall):
   - NaN fix verified 后立刻 launch on GPU 3
   - probe-on-nan mode 仍激活 (atexit fix + 显式 save-on-failure 双保险)
   - Day 5 evening 拿 Tier 1 verdict (Day 6 早上前 latest)

**推迟到 Day 6+**:
- full v3 doc (Step 5)
- SR optimizer implementation (原 Day 5 主体)
- SR pytest

**Schedule**: 当前 累计 slip 2 天 / buffer 0 天 / Plan B 线 Day 7 16:00 UTC。如 Day 5 NaN fix 顺利 + production v2 跑通到 step 3000，schedule 维持 ~Day 23 submission (1天 over original)。如 Day 5 NaN fix 卡住或 production v2 再 stall，Day 6 上午激活 Plan B。

---

## 7. Day 4 commit handoff

**待 commit (Day 4 closing batch)**:
- `experiments/narrow_band_diagnostic.py` (含 probe_on_nan mode)
- `experiments/probe_grad_nan.py` (synthetic ladder + probe_one_theta)
- `results/state_at_first_nan.pt` (force-add, 114 KB)
- `results/probe_at_first_nan.json` (force-add, 4 KB)
- `logs/STALL_evidence_production_v2_seed42_20260505_1431.log` (force-add)
- `logs/STALL_evidence_tmux_capture_20260505_1431.log` (force-add)
- `docs/known_issues_day2.md` 加新条目 (NaN bug + atexit defect)
- `docs/NQS_experiment_plan.md` (Day 4 morning update)
- `docs/daily_log_day3.md` (Day 4 morning Step 2)
- `docs/daily_log_day4.md` (本文)
- `docs/NQS_J1J2_Prediction_v3_outline.md` (Day 4 morning Step 4 update)

**不 commit**:
- `results/narrow_band_diagnostic_stageA_partial.json` (reconstructed sparse, evidence 价值低)
- `logs/narrow_band_*` (gitignored, ephemeral)
- `logs/probe_*` (gitignored, ephemeral)

**push status**: Day 3 末尾 2 commits ahead 已被 Day 4 morning Step 3 push。Day 4 closing batch commit 数 ~10，全部 push 后到 Day 5 早上 origin/main 同步。

---

**End of Day 4 daily_log.**
