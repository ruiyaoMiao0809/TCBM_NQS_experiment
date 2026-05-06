"""
core/sr_optimizer.py
=====================
Stochastic Reconfiguration (SR) optimizer for NQS variational optimization.

References:
  - Sorella 1998 (PRB 57, R10001) -- original SR formulation
  - Bukov, Schmitt, Dupont 2021 (SciPost Phys 10.147) -- 4x4/6x6 J1-J2 NQS benchmark
  - NetKet examples/Heisenberg1d, examples/J1J2

Update rule:
    Δθ = -lr · S^(-1) · g

where:
    g_i  = E[E_local · O_i^*] - E[E_local] · E[O_i^*]   (energy gradient)
    S_ij = E[O_i^* · O_j]     - E[O_i^*] · E[O_j]       (Quantum Geometric Tensor)
    O_i(s) = ∂_i log ψ(s; θ)                            (log-derivative wrt param i)

Regularization (S is ill-conditioned):
    'identity':    S_reg = S + ε · I
    'scale-aware': S_reg = S + ε · diag(S)              (Bukov 2021 default)

Solver:
    'pinv': Δθ = -lr · pinv(S_reg) · g                  (D ≤ 2000, our D≈1120 fits)
    'cg':   Δθ = -lr · CG_solve(S_reg, g)               (D > 2000, future scale-up)

Status: skeleton (Day 5 implementation pending).
Day 5 will implement the four helpers + SROptimizer.{__init__, optimize}, then add
pytest test against NetKet on a 2x2 trivial case for ground-truth correctness.
"""
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, List

import torch


@dataclass
class SRConfig:
    """SR optimizer configuration. Defaults follow Bukov 2021 Section 5/Appendix A."""
    lr: float = 0.05                      # Bukov uses adaptive lr in [0.01, 0.1]
    epsilon: float = 1e-3                 # regularization strength
    epsilon_mode: str = 'scale-aware'     # 'identity' | 'scale-aware'
    solver: str = 'pinv'                  # 'pinv' | 'cg'
    cg_tol: float = 1e-6                  # CG convergence tolerance (only if solver='cg')
    cg_max_iter: int = 200                # CG iteration cap

    n_steps: int = 2000                   # outer loop iterations
    n_vmc_samples: int = 2000             # VMC samples per gradient estimate
    n_vmc_samples_final: int = 4          # multi-shot for final-eval debiasing
    seed: int = 42

    # Diagnostic recording (Day 5 will populate)
    record_qgt_eigvals: bool = False      # save eigvals every N steps for SI plot
    record_qgt_every: int = 100
    record_trajectory: bool = True


class SROptimizer:
    """
    Stochastic Reconfiguration optimizer for single-replica NQS.

    No PT, no replica swap -- this is the "natural gradient" baseline that will
    be the PRIMARY comparison target for TCBM in NQS_J1J2_Prediction_v3.

    Schema parity with TCBMOptimizer.optimize() so the same downstream analysis
    (run_*_baseline.py, robustness aggregation) works without code changes.

    Day 5 TODO:
        1. __init__: store theta as leaf tensor, set seeds, alloc trajectory buffers
        2. optimize(): main loop (sample → log-deriv → QGT → solve → step → record)
        3. final-eval: n_final-shot mean on converged θ
        4. callback hook (signature matches TCBMOptimizer for parity)
    """

    def __init__(self, problem, cfg: SRConfig):
        self.problem = problem
        self.cfg = cfg
        # Day 5 TODO:
        #   torch.manual_seed(cfg.seed)
        #   self.theta = problem.random_feasible(1).requires_grad_(True)
        #   self._init_trajectory_buffers()
        raise NotImplementedError("SROptimizer.__init__ -- Day 5")

    def optimize(
        self,
        callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
        callback_every: int = 100,
    ) -> Dict[str, Any]:
        """
        Main SR optimization loop.

        Returns dict (schema parity with TCBMOptimizer.optimize()):
            best_cost_raw            : float, min single-shot during training (noisy)
            best_cost_debiased       : float, mean of n_final-shot eval on final θ
            final_sigma_E_mean       : float
            final_sigma_E_max        : float
            final_costs_per_shot     : list[float]
            final_sigmas_per_shot    : list[float]
            trajectory               : {'cost': [...], 'sigma_E': [...]}
            final_theta              : torch.Tensor (D,)

            # SR-specific diagnostics (none of these in TCBM result):
            qgt_condition_number_history : list[float]   log10(λ_max/λ_min) per step
            qgt_eigvals_final            : torch.Tensor  full spectrum at final step
            log_derivative_norm_history  : list[float]   ‖O‖ over time (sanity check)
            sr_solve_residual_history    : list[float]   ‖S Δθ - g‖ / ‖g‖

        Day 5 TODO:
            for step in range(cfg.n_steps):
                samples = self.problem.sample_chain(theta, cfg.n_vmc_samples)
                E_loc, sigma_E = self.problem.local_energy(theta, samples)
                O = compute_log_derivatives(self.problem, theta, samples)
                S = compute_qgt(O)
                g = (O.conj() * (E_loc - E_loc.mean()).unsqueeze(1)).mean(0)
                S_reg = regularize_qgt(S, cfg.epsilon, cfg.epsilon_mode)
                dtheta = solve_sr_update(S_reg, g, method=cfg.solver, ...)
                theta = theta - cfg.lr * dtheta
                # record + callback
        """
        raise NotImplementedError("SROptimizer.optimize -- Day 5")


# === Helper functions (Day 5 will implement) ============================

def compute_log_derivatives(problem, theta: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
    """
    Compute O_i(s_n) = ∂_i log ψ(s_n; θ) for each sample s_n and each param i.

    Args:
        problem : J1J2Problem instance with log_psi(theta, samples) method
        theta   : (D,) leaf tensor
        samples : (N, n_spins) bit configurations

    Returns:
        log_derivs : (N, D) tensor (real if RBM real, complex if RBM complex)

    Day 5 implementation note:
        Two valid paths:
        (a) torch.autograd.grad with create_graph=False, looped over N
            -- simple, slow, O(N) backward passes
        (b) torch.func.jacrev on log_psi(theta, samples)
            -- batched, faster on GPU, may OOM at N×D large

        Bukov uses (b) via JAX vmap. Start with (a) for correctness, switch
        to (b) if profiling shows it's the bottleneck.
    """
    raise NotImplementedError("compute_log_derivatives -- Day 5")


def compute_qgt(log_derivatives: torch.Tensor) -> torch.Tensor:
    """
    Compute Quantum Geometric Tensor:
        S_ij = E[O_i^* O_j] - E[O_i^*] E[O_j]

    Args:
        log_derivatives : (N, D) tensor of O_i(s_n)

    Returns:
        S : (D, D) Hermitian PSD matrix (or symmetric PSD if real RBM)

    Day 5 implementation note:
        S = (O.conj().T @ O) / N - (O.conj().mean(0).unsqueeze(1) @ O.mean(0).unsqueeze(0))
        Watch O(D²) memory: D=1120 → S is 1120² × 8 bytes ≈ 10 MB, fine.
        For D=10000 → 800 MB, then need CG solver.
    """
    raise NotImplementedError("compute_qgt -- Day 5")


def regularize_qgt(qgt: torch.Tensor, epsilon: float, mode: str = 'scale-aware') -> torch.Tensor:
    """
    Make QGT invertible by adding diagonal regularization.

    'identity':    S_reg = S + ε · I
    'scale-aware': S_reg = S + ε · diag(S)         (Bukov 2021 Appendix A)

    The scale-aware variant adapts to per-parameter curvature -- safer when
    different params have very different effective scales (RBM hidden vs visible
    weights).

    Day 5 implementation note:
        if mode == 'identity':
            return qgt + epsilon * torch.eye(qgt.shape[0], device=qgt.device, dtype=qgt.dtype)
        elif mode == 'scale-aware':
            d = torch.diag(qgt).real
            return qgt + epsilon * torch.diag(d)
    """
    raise NotImplementedError("regularize_qgt -- Day 5")


def solve_sr_update(
    qgt_reg: torch.Tensor,
    gradient: torch.Tensor,
    method: str = 'pinv',
    cg_tol: float = 1e-6,
    cg_max_iter: int = 200,
) -> torch.Tensor:
    """
    Solve S_reg · Δθ = g for Δθ.

    method='pinv' : Δθ = pinv(S_reg) @ g
        Direct, robust, OK for D ≤ a few thousand.
        torch.linalg.pinv has SVD-based pseudoinverse with rcond cutoff.

    method='cg'   : Conjugate gradient solver
        Iterative, lower per-iter memory, good for D > 2000.
        Use torch's iterative solvers or roll our own (S is HPD after reg).

    Returns:
        dtheta   : (D,) update direction (NOT yet scaled by lr)
        residual : float, ‖S_reg Δθ - g‖ / ‖g‖   (for diagnostics)

    Day 5 implementation note:
        Day 5 ships with 'pinv' only. Day 8+ adds CG if D scales up
        (e.g. α=4 RBM with D=4000+ for Week 2 ablation).
    """
    raise NotImplementedError("solve_sr_update -- Day 5")


# === Callback info schema (matches TCBMOptimizer for cross-method parity) ===

# def callback(step: int, info: dict) -> None:
#     info contains (SR-flavored):
#         'cost_min'           : E_local mean this step (single-replica, M=1 always)
#         'cost_mean'          : same as cost_min for SR
#         'sigma_E_max'        : sample std of E_local
#         'sigma_E_mean'       : same as sigma_E_max for SR
#         'qgt_cond_number'    : log10(λ_max/λ_min(S_reg))
#         'sr_solve_residual'  : ‖S Δθ - g‖ / ‖g‖
#         'log_deriv_norm'     : ‖O‖ Frobenius norm
#         # PT-flavored fields are None (kept for schema parity):
#         'swap_acc_recent'    : None
#         'subspace_active'    : None
#         'psi_max'            : None
#         'T_w_step'           : None
