"""
core/tcbm_optimizer_NQS.py
==========================
TCBM NQS — Tunneling-Clamping Boltzmann Machine Optimizer (Neural-network Quantum State variant)
Forked from tcbm_optimizer_NN.py — adapted for NQS ground-state search.

Scope (Path B, proof-of-concept)
═══════════════════════════════════════════════════════════════════════════════
Target task: variational minimization of E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩ / ⟨ψ(θ)|ψ(θ)⟩
for neural-network-parametrized wave functions ψ(σ; θ).

Key difference from deterministic optimization:
  1. evaluate(x) returns stochastic cost with VMC estimation noise σ_E.
  2. The parameter manifold has a non-Euclidean metric (quantum geometric
     tensor, QGT). This POC does NOT implement natural-gradient dynamics;
     it only exposes a hook so that future versions can plug in SR/MinSR.

Therefore this version implements ONLY two modifications relative to NN v5:
  • NQS-1  (stochastic loss handling)
  • NQS-3  (configurable subspace source: gradient / qgt / hybrid)

All other mechanisms — tunneling, clamping, ψ(t) maturity, T_w, ablation
framework, v5 diagnostics — are preserved unchanged.

═══════════════════════════════════════════════════════════════════════════════
NQS-1 — Stochastic loss handling
═══════════════════════════════════════════════════════════════════════════════

  NQS-1a  — Extended problem.evaluate() contract:
             (cost, penalty, cost_std) where cost_std: Tensor(M,) or None.
             Deterministic problems return cost_std=None and the optimizer
             falls back to exact NN-v5 behaviour.

  NQS-1b  — Effective-temperature Metropolis:
             T_eff = T_physical + noise_temperature_alpha · σ_E
             With noise_temperature_alpha default 1.0, this matches the
             standard SGLD-style correction for stochastic Metropolis.
             Prevents noise-driven false rejections in the cold replica.

  NQS-1c  — Noise-aware EMA smoothing:
             ema_decay for grad_norm_ema and energy history increased from
             0.95 → 0.98 when σ_E > 0; provides longer averaging window to
             suppress VMC sampling noise in gradient-norm tracking.

  NQS-1d  — Stochastic swap acceptance:
             Replica exchange uses (E_i - E_j) / √(σ_i² + σ_j² + T²) as the
             Bayesian-corrected energy-difference test statistic, preventing
             noise-dominated swap decisions when σ_E is comparable to ΔE.

  NQS-1e  — Re-evaluation of best_cost at termination:
             VMC noise means best-observed energy is biased low (selection
             effect). At termination we re-evaluate best_x with larger
             n_samples (default 4×) to de-bias the reported best_cost.

═══════════════════════════════════════════════════════════════════════════════
NQS-3 — Configurable subspace source
═══════════════════════════════════════════════════════════════════════════════

  NQS-3a  — New config field: subspace_source ∈ {'gradient', 'qgt', 'hybrid'}
             'gradient': current NN-v5 behaviour (SVD on gradient buffer)
             'qgt':      eigendecomposition of quantum geometric tensor
             'hybrid':   orthogonal decomposition of gradient SVD against
                          top-k QGT eigenvectors; TCBM subspace = gradient
                          directions NOT captured by QGT. This is the key
                          experimental knob for distinguishing TCBM's
                          clamping from SR's natural-gradient metric.

  NQS-3b  — problem.qgt(x) contract:
             Returns either:
               • (M, D, D) full QGT,
               • (M, k_qgt, V_qgt) low-rank factorization where QGT ≈ V·Vᵀ,
               • None (qgt/hybrid modes unavailable; optimizer raises).
             For D > ~5000, low-rank form is required for tractability.

  NQS-3c  — Hybrid orthogonalization:
             Given gradient SVD U_g ∈ R^(D×k) and QGT eigenbasis V_q ∈ R^(D×k_qgt),
             TCBM subspace := QR((I - V_q V_qᵀ) U_g)[:, :k]
             = residual gradient directions after projecting out QGT span.
             This is the subspace on which natural-gradient methods are
             BLIND, and where TCBM's earned directional constraint provides
             non-redundant information.

Interface contract (extended from NN v5)
═══════════════════════════════════════════════════════════════════════════════
Required:
  • problem.dim              : int
  • problem.evaluate(x, n_samples=None) -> (cost, penalty, cost_std)
      - cost, penalty: (M,) tensors as before
      - cost_std: (M,) tensor or None
  • problem.random_feasible(batch_size) -> (B, D) tensor

Optional:
  • problem.gradient(positions) -> (M, D)
  • problem.clamp_dims           : int < D (partial clamping)
  • problem.qgt(positions, k_qgt) -> (M, D, k_qgt) low-rank or None
  • problem.dissipation_a, dissipation_b : float (v5-4)

All NN-v5 fields and methods are retained. Setting subspace_source='gradient'
and n_vmc_samples=None exactly reproduces NN v5 behaviour.
"""

import time
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List, Union, Callable

import numpy as np
import torch
import torch.nn.functional as F

from core.device import get_device


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TCBMConfig:
    """
    Hyperparameters for TCBM NQS. Defaults tuned for D~5000 NQS on
    1D transverse-field Ising (proof-of-concept scale).

    NQS additions (all with defaults that preserve NN-v5 behaviour when
    subspace_source='gradient' and n_vmc_samples=None):

        subspace_source : str    ∈ {'gradient', 'qgt', 'hybrid'} (NQS-3a)
        k_qgt           : int    QGT eigenbasis dim for hybrid (NQS-3c)
        n_vmc_samples   : Optional[int]  MC samples per evaluate call
        n_vmc_samples_final : int  MC samples for termination re-evaluation (NQS-1e)
        noise_temperature_alpha : float  effective-T correction coeff (NQS-1b)
        noise_ema_decay : float  EMA decay when σ_E > 0 (NQS-1c)
    """
    # ── Replica population ────────────────────────────────────────────────────
    M: int = 16                 # NQS: slightly fewer replicas; each evaluate is costly
    T_min: float = 0.005        # NQS: E/N ~ O(1), per-site fluctuations ~ 1e-3
    T_max: float = 2.0          # NQS: upper end driven by barrier scale in NN landscape

    # ── Clamping (λ) schedule ─────────────────────────────────────────────────
    # NQS: NN weights typically have ||θ||~O(√D) but are often normalized.
    # lambda_max scaled conservatively; final value tuned in POC experiments.
    lambda_min: float = 0.05
    lambda_max: float = 2.0
    tau_lambda: float = 500.0

    # ── Entanglement (Γ) schedule ─────────────────────────────────────────────
    Gamma_0: float = 0.5        # NQS: weaker entanglement for stochastic loss stability
    tau_Gamma: float = 800.0

    # ── Subspace ─────────────────────────────────────────────────────────────
    k: int = 20                 # NQS: subspace dim, ~1% of D=5000 ansatz
    subspace_warmup: int = 150  # NQS: longer warmup to average out VMC noise in grads
    subspace_update_freq: int = 80  # NQS: less frequent SVD due to per-call cost

    # ── NQS-3: Subspace source (NEW) ──────────────────────────────────────────
    subspace_source: str = 'gradient'    # 'gradient' | 'qgt' | 'hybrid'
    k_qgt: int = 40             # QGT eigenbasis dim for hybrid mode (> k to leave room)

    # ── v3-B: Belief maturity (unchanged) ─────────────────────────────────────
    psi_star: float = 1.5
    min_warmup: int = 100       # NQS: aligned with subspace_warmup

    # ── v5-2: Temperature floor ───────────────────────────────────────────────
    T_min_floor: float = 0.002  # NQS: matches T_min scale

    # ── Gradient buffer for SVD ───────────────────────────────────────────────
    grad_buffer_size: int = 400  # NQS: larger buffer, gradient signal is noisier
    grad_buffer_use: int = 200

    # ── Dynamics ─────────────────────────────────────────────────────────────
    n_steps: int = 3000
    step_size: float = 0.0005   # NQS: smaller step for stochastic landscape
    grad_clip: float = 100.0
    grad_norm_floor: float = 0.05

    # ── Temperature annealing ─────────────────────────────────────────────────
    T_anneal_gamma: float = 0.5  # NQS: slower annealing, stochastic loss needs exploration
    T_anneal_tau: float = 1500.0

    # ── Replica exchange ──────────────────────────────────────────────────────
    swap_interval: int = 15
    reseed_interval: int = 400
    reseed_threshold: int = 6    # NQS: higher threshold (noise-driven fake stagnation)
    n_hot_reseed: int = 2

    # ── NQS-1: Stochastic loss handling (NEW) ─────────────────────────────────
    n_vmc_samples: Optional[int] = None    # forwarded to problem.evaluate
    n_vmc_samples_final: int = 4           # termination re-eval: 4× the usual count
    noise_temperature_alpha: float = 1.0   # NQS-1b: T_eff = T + α·σ_E
    noise_ema_decay: float = 0.98          # NQS-1c: EMA decay when σ_E > 0

    # ── Misc ─────────────────────────────────────────────────────────────────
    seed: int = 42
    record_trajectory: bool = True
    record_every: int = 10

    # ── NQS BUGFIX: Box-constraint on parameter space ────────────────────────
    # DC-OPF has box constraints θ ∈ [0, 1] (normalized generator dispatch),
    # but NQS parameter space is unconstrained real numbers with typical scale
    # |θ| ~ 0.01-2 and Gaussian initialization (symmetric around 0).
    # Setting box_constraint=False disables the clamp(0, 1) in Langevin step.
    #
    # If problem provides `problem.box_constraint`, it overrides this config.
    # Default True for backward compatibility with DC-OPF pipelines.
    box_constraint: bool = False        # NQS default: unconstrained
    box_low:  float = 0.0
    box_high: float = 1.0

    def __post_init__(self):
        self._max_warmup: int = self.subspace_warmup
        assert self.subspace_source in {'gradient', 'qgt', 'hybrid'}, (
            f"subspace_source must be one of 'gradient', 'qgt', 'hybrid'; "
            f"got {self.subspace_source!r}")

    @property
    def max_warmup(self) -> int:
        return self._max_warmup

    @max_warmup.setter
    def max_warmup(self, v: int):
        self._max_warmup = v
        self.subspace_warmup = v


# ─────────────────────────────────────────────────────────────────────────────
# Main optimizer
# ─────────────────────────────────────────────────────────────────────────────

class TCBMOptimizer:
    """
    TCBM NQS — Tunneling-Clamping Boltzmann Machine for variational
    neural-network quantum states.

    Parameters
    ----------
    problem : any object exposing
              .dim, .evaluate(x, n_samples=None), .random_feasible(B),
              optional .gradient(), .qgt(), .clamp_dims
    config  : TCBMConfig (optional)
    """

    def __init__(self, problem, config: Optional[TCBMConfig] = None):
        self.problem = problem
        self.config  = config or TCBMConfig()
        self.device  = getattr(problem, 'device', get_device())
        self.D       = problem.dim
        self.dtype   = torch.float32

        self._rng_cpu = np.random.default_rng(self.config.seed)
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(self.config.seed)

        # NQS-3: validate QGT availability if requested
        if self.config.subspace_source in {'qgt', 'hybrid'}:
            if not hasattr(self.problem, 'qgt'):
                raise RuntimeError(
                    f"subspace_source={self.config.subspace_source!r} requires "
                    f"problem.qgt() method; got problem type {type(self.problem).__name__}")

    # ─────────────────────────────────────────────────────────────────────────
    # NQS-1a: Unified stochastic-evaluate wrapper
    # ─────────────────────────────────────────────────────────────────────────

    def _eval_total(
        self,
        positions: torch.Tensor,
        n_samples: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        NQS-1a: evaluate positions with extended contract.

        Returns
        -------
        energies : (M,)
        std      : (M,)   zeros if problem is deterministic
        """
        ns = n_samples if n_samples is not None else self.config.n_vmc_samples
        # Try extended contract first; fall back to deterministic 2-tuple.
        try:
            ret = self.problem.evaluate(positions, n_samples=ns)
        except TypeError:
            ret = self.problem.evaluate(positions)

        if isinstance(ret, tuple) and len(ret) == 3:
            cost, pen, std = ret
            energies = (cost + pen).detach()
            if std is None:
                std = torch.zeros_like(energies)
            else:
                std = std.detach()
        else:
            cost, pen = ret
            energies = (cost + pen).detach()
            std = torch.zeros_like(energies)
        return energies, std

    # ─────────────────────────────────────────────────────────────────────────
    # Initialisation helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _init_temperatures(self) -> torch.Tensor:
        """Geometric temperature ladder. shape (M,)."""
        M = self.config.M
        ratio = (self.config.T_max / self.config.T_min) ** (1.0 / max(M - 1, 1))
        temps = [self.config.T_min * ratio ** i for i in range(M)]
        return torch.tensor(temps, dtype=self.dtype, device=self.device)

    def _init_replicas(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initialize M replicas via problem.random_feasible().
        Returns (positions, energies, stds) all on device.
        """
        M = self.config.M
        positions = self.problem.random_feasible(batch_size=M).to(
            self.device, dtype=self.dtype)
        energies, stds = self._eval_total(positions)
        return positions, energies, stds

    def _init_subspace(
        self,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bootstrap subspace from initial replica positions via SVD.
        For all subspace_source modes the initial U is built from positions
        (QGT unavailable before first gradient pass).
        """
        clamp_dims = getattr(self.problem, 'clamp_dims', self.D)
        pos_for_svd = positions[:, :clamp_dims] if clamp_dims < self.D else positions
        eff_dim = pos_for_svd.shape[1]
        k = min(self.config.k, eff_dim, max(positions.shape[0] - 1, 0))

        if k == 0:
            k = min(self.config.k, eff_dim)
            U_new = torch.randn(eff_dim, k, dtype=self.dtype, device=self.device)
            U_new, _ = torch.linalg.qr(U_new)
            sv = torch.ones(k, dtype=self.dtype, device=self.device)
            return U_new, sv

        centered = pos_for_svd - pos_for_svd.mean(dim=0, keepdim=True)
        try:
            _, S, Vh = torch.linalg.svd(centered, full_matrices=False)
            U_new = Vh[:k].T.contiguous()
            sv = S[:k]
        except Exception:
            U_new = torch.eye(eff_dim, k, dtype=self.dtype, device=self.device)
            sv = torch.ones(k, dtype=self.dtype, device=self.device)
        return U_new, sv

    # ─────────────────────────────────────────────────────────────────────────
    # NQS-3: Subspace update with configurable source
    # ─────────────────────────────────────────────────────────────────────────

    def _update_subspace(
        self,
        grad_buffer : torch.Tensor,   # (N_buf, D)
        old_U       : torch.Tensor,   # (D, k)
        positions   : Optional[torch.Tensor] = None,   # (M, D), for QGT
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float, Dict]:
        """
        Update subspace with configurable source (NQS-3).

        Returns
        -------
        U_new           : (D, k)
        singular_values : (k,)
        psi             : float   maturity index σ_k / σ_{k+1}
        delta_k_star    : float   spectral gap σ_k − σ_{k+1}
        diag            : dict    {
            'mode': str,
            'n_gradient_directions_kept': int (hybrid only),
            'qgt_rank_effective': int (qgt/hybrid only),
            ...
        }
        """
        mode = self.config.subspace_source
        diag = {'mode': mode}

        # Gradient SVD is always computed (used by all three modes).
        U_grad, sv_grad, psi, delta_k_star = self._gradient_svd(grad_buffer, old_U)

        if mode == 'gradient':
            return U_grad, sv_grad, psi, delta_k_star, diag

        # For qgt/hybrid modes we need the QGT.
        if positions is None:
            diag['fallback_reason'] = 'positions unavailable; fell back to gradient'
            return U_grad, sv_grad, psi, delta_k_star, diag

        V_qgt, eigvals_qgt = self._qgt_eigenbasis(positions)
        if V_qgt is None:
            diag['fallback_reason'] = 'problem.qgt returned None; fell back to gradient'
            return U_grad, sv_grad, psi, delta_k_star, diag

        diag['qgt_rank_effective'] = int((eigvals_qgt > 1e-6 * eigvals_qgt.max()).sum())

        if mode == 'qgt':
            # NQS-3: subspace = top-k QGT eigenvectors directly.
            k = min(self.config.k, V_qgt.shape[1])
            U_new = V_qgt[:, :k].contiguous()
            # For psi/delta_k we reuse the gradient-SVD quantities since
            # they still govern the maturity trigger (a QGT-native maturity
            # measure is future work).
            return U_new, eigvals_qgt[:k], psi, delta_k_star, diag

        # mode == 'hybrid': U = QR((I - V_q V_qᵀ) U_grad)[:, :k]
        # Project out QGT span from gradient directions, take residual.
        proj_onto_qgt = V_qgt @ (V_qgt.T @ U_grad)     # (D, k)
        U_residual = U_grad - proj_onto_qgt             # (D, k)
        residual_norms = U_residual.norm(dim=0)
        n_kept = int((residual_norms > 1e-4).sum())
        diag['n_gradient_directions_kept'] = n_kept

        if n_kept == 0:
            # All gradient directions already in QGT span; hybrid degenerates.
            diag['fallback_reason'] = 'gradient subspace fully captured by QGT'
            return U_grad, sv_grad, psi, delta_k_star, diag

        try:
            U_new, _ = torch.linalg.qr(U_residual)
            k = min(self.config.k, U_new.shape[1], n_kept)
            U_new = U_new[:, :k].contiguous()
        except Exception:
            U_new = U_grad

        return U_new, sv_grad[:U_new.shape[1]], psi, delta_k_star, diag

    def _gradient_svd(
        self,
        grad_buffer: torch.Tensor,
        old_U      : torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        Core gradient-buffer SVD (former _update_subspace from NN-v5).
        Returns (U_new, sv, psi, delta_k_star).
        """
        k = min(self.config.k, self.D, grad_buffer.shape[0] - 1)
        if grad_buffer.shape[0] < k + 2:
            sv_dummy = torch.ones(k, dtype=self.dtype, device=self.device)
            return old_U, sv_dummy, 0.0, 0.0

        G = grad_buffer[-self.config.grad_buffer_use:]
        G_c = G - G.mean(dim=0, keepdim=True)
        row_norms = G_c.norm(dim=1, keepdim=True).clamp(min=1e-6)
        G_c = G_c / row_norms

        try:
            _, S, Vh = torch.linalg.svd(G_c, full_matrices=False)
            U_raw = Vh[:k].T.contiguous()
            sv    = S[:k]

            sigma_k  = float(S[k - 1]) if len(S) >= k else 1.0
            sigma_k1 = float(S[k])     if len(S) > k  else float(S[-1]) * 0.1
            psi      = sigma_k / (sigma_k1 + 1e-10)
            delta_k_star = sigma_k - sigma_k1
        except Exception:
            sv_dummy = torch.ones(k, dtype=self.dtype, device=self.device)
            return old_U, sv_dummy, 0.0, 0.0

        if old_U is not None and old_U.shape == U_raw.shape:
            signs = torch.sign(torch.sum(U_raw * old_U, dim=0, keepdim=True))
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            U_raw = U_raw * signs
            U_blend = 0.7 * U_raw + 0.3 * old_U
            U_new, _ = torch.linalg.qr(U_blend)
            U_new = U_new[:, :k].contiguous()
        else:
            U_new = U_raw

        return U_new, sv, psi, delta_k_star

    def _qgt_eigenbasis(
        self,
        positions: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        NQS-3: Get top-k_qgt QGT eigenvectors at mean position.

        Uses problem.qgt() which should return:
          - (D, k_qgt) low-rank factor V where QGT ≈ V·Vᵀ, OR
          - (D, D) full QGT, OR
          - None

        Returns (V, eigvals) or (None, None).
        """
        cfg = self.config
        # Take QGT at the cold replica position (most likely basin).
        # Alternative: mean over replicas. For POC we use cold replica.
        x_for_qgt = positions[0:1]     # (1, D)

        try:
            qgt_out = self.problem.qgt(x_for_qgt, k_qgt=cfg.k_qgt)
        except TypeError:
            qgt_out = self.problem.qgt(x_for_qgt)

        if qgt_out is None:
            return None, None

        # Squeeze batch dim; accept either (D, k_qgt) or (D, D) or (1, D, ...).
        Q = qgt_out.detach()
        if Q.ndim == 3:
            Q = Q[0]

        if Q.shape[0] == Q.shape[1] and Q.shape[0] == self.D:
            # Full QGT: symmetric eigendecomposition.
            try:
                eigvals, eigvecs = torch.linalg.eigh(Q)
                # eigh returns ascending; reverse to descending.
                eigvals = eigvals.flip(0)
                eigvecs = eigvecs.flip(1)
                k_take = min(cfg.k_qgt, eigvecs.shape[1])
                return eigvecs[:, :k_take].contiguous(), eigvals[:k_take]
            except Exception:
                return None, None
        else:
            # Low-rank factor V with shape (D, k_qgt).
            # QGT ≈ V Vᵀ; eigenvectors of QGT = left singular vectors of V.
            try:
                U_v, S_v, _ = torch.linalg.svd(Q, full_matrices=False)
                k_take = min(cfg.k_qgt, U_v.shape[1])
                # Eigenvalues of V·Vᵀ are squared singular values of V.
                return U_v[:, :k_take].contiguous(), (S_v[:k_take] ** 2)
            except Exception:
                return None, None

    # ─────────────────────────────────────────────────────────────────────────
    # Low-rank projection (unchanged)
    # ─────────────────────────────────────────────────────────────────────────

    def _project_to_subspace(
        self, x: torch.Tensor, U: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        coeff  = x @ U
        x_proj = coeff @ U.T
        x_perp = x - x_proj
        return x_proj, x_perp

    # ─────────────────────────────────────────────────────────────────────────
    # Batch gradient (unchanged)
    # ─────────────────────────────────────────────────────────────────────────

    def _batch_gradient(self, positions: torch.Tensor) -> torch.Tensor:
        if hasattr(self.problem, 'gradient'):
            return self.problem.gradient(positions)
        x = positions.detach().requires_grad_(True)
        ret = self.problem.evaluate(x)
        if isinstance(ret, tuple) and len(ret) == 3:
            cost, pen, _ = ret
        else:
            cost, pen = ret
        total = (cost + pen).sum()
        total.backward()
        return x.grad.detach().clone()

    # ─────────────────────────────────────────────────────────────────────────
    # Force computation (unchanged from NN-v5)
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_forces_batch(
        self,
        positions    : torch.Tensor,
        energies     : torch.Tensor,
        grads        : torch.Tensor,
        step         : int,
        lam_eff      : float,
        gamma_eff    : float,
        U            : Optional[torch.Tensor],
        grad_norm_ema: float,
        temps        : torch.Tensor,
        stds         : Optional[torch.Tensor] = None,    # NQS-1c
    ) -> Tuple[torch.Tensor, float]:
        """
        Langevin force assembly. NQS-1c: if stds is non-zero, EMA uses the
        noise-aware decay (config.noise_ema_decay, default 0.98) instead of
        the deterministic default 0.95.
        """
        cfg = self.config
        M   = positions.shape[0]

        grad_norms    = torch.norm(grads, dim=1, keepdim=True).clamp(min=1e-10)
        clip_scale    = (cfg.grad_clip / grad_norms).clamp(max=1.0)
        grads_clipped = grads * clip_scale

        clipped_norms = torch.norm(grads_clipped, dim=1, keepdim=True)
        mean_gn       = float(clipped_norms.mean())

        # NQS-1c: longer EMA window when stochastic
        is_stochastic = (stds is not None and float(stds.max()) > 0.0)
        ema_decay = cfg.noise_ema_decay if is_stochastic else 0.95
        grad_norm_ema = ema_decay * grad_norm_ema + (1.0 - ema_decay) * mean_gn

        force_scale = max(grad_norm_ema, cfg.grad_norm_floor)

        F_grad = -grads_clipped

        if U is not None and lam_eff > 0.0:
            clamp_dims = getattr(self.problem, 'clamp_dims', self.D)
            if clamp_dims < self.D:
                pos_thermal = positions[:, :clamp_dims]
                _, x_perp_thermal = self._project_to_subspace(pos_thermal, U)
                F_clamp = torch.zeros_like(positions)
                F_clamp[:, :clamp_dims] = -lam_eff * x_perp_thermal
            else:
                _, x_perp = self._project_to_subspace(positions, U)
                F_clamp = -lam_eff * x_perp
        else:
            F_clamp = torch.zeros_like(positions)

        if M > 1 and gamma_eff > 0.0:
            beta_ent = 1.0 / (float(temps[0]) + 1e-10)
            log_w    = -beta_ent * energies
            log_w    = log_w - log_w.max()
            w        = torch.softmax(log_w, dim=0)

            W = w.unsqueeze(0).expand(M, M).clone()
            W.fill_diagonal_(0.0)
            W_sum = W.sum(dim=1, keepdim=True).clamp(min=1e-30)
            W     = W / W_sum
            x_targets = W @ positions

            direction = x_targets - positions
            dir_norms = torch.norm(direction, dim=1, keepdim=True).clamp(min=1e-10)
            dir_unit  = direction / dir_norms
            mag_clamp = dir_norms.clamp(max=0.5)

            F_entangle = gamma_eff * force_scale * dir_unit * mag_clamp
        else:
            F_entangle = torch.zeros_like(positions)

        F_total = F_grad + F_clamp + F_entangle

        f_norms = torch.norm(F_total, dim=1, keepdim=True).clamp(min=1e-10)
        f_clip = cfg.grad_clip * (20.0 if lam_eff > 0.0 else 3.0)
        F_total = F_total * (f_clip / f_norms).clamp(max=1.0)

        return F_total, grad_norm_ema

    # ─────────────────────────────────────────────────────────────────────────
    # NQS-1b: Effective-temperature Langevin + Metropolis
    # ─────────────────────────────────────────────────────────────────────────

    def _langevin_step_batch(
        self,
        positions    : torch.Tensor,
        energies     : torch.Tensor,
        stds         : torch.Tensor,            # NQS-1b
        F_total      : torch.Tensor,
        temps        : torch.Tensor,
        grad_norm_ema: float,
        lam_eff      : float = 0.0,
        U            : Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batched MALA step with effective-temperature Metropolis (NQS-1b).

        T_eff = T_physical + α · σ_E
        where α = config.noise_temperature_alpha. For deterministic problems
        (σ_E = 0) this exactly reproduces NN-v5 behaviour.

        Returns (positions, energies, stds) — stds updated with new values
        where the move was accepted.
        """
        cfg = self.config
        M   = positions.shape[0]

        base_dt = cfg.step_size / (1.0 + 0.1 * grad_norm_ema)
        if lam_eff > 0.0:
            clamp_scale = 1.0 / (1.0 + 0.1 * lam_eff)
            effective_dt = base_dt * clamp_scale
        else:
            effective_dt = base_dt

        # NQS-1b: effective temperature for noise scaling AND Metropolis.
        # Noise scaling: use T_eff so that Langevin diffusion respects the
        # total (physical + estimation) stochasticity.
        T_eff_pre = temps + cfg.noise_temperature_alpha * stds
        noise_scale = torch.sqrt(2.0 * T_eff_pre.unsqueeze(1) * effective_dt)
        noise = torch.randn(
            M, self.D, dtype=self.dtype, device=self.device,
            generator=self._gen) * noise_scale

        # NQS BUGFIX: Box-constraint is optional, disabled by default for NQS.
        # DC-OPF needed clamp(0,1) for normalized generator dispatch, but NQS
        # parameter space is unconstrained real numbers.
        x_new = positions + F_total * effective_dt + noise
        use_box = getattr(self.problem, 'box_constraint', cfg.box_constraint)
        if use_box:
            box_low  = getattr(self.problem, 'box_low',  cfg.box_low)
            box_high = getattr(self.problem, 'box_high', cfg.box_high)
            x_new = x_new.clamp(box_low, box_high)
        new_energies, new_stds = self._eval_total(x_new)

        if lam_eff > 0.0 and U is not None:
            clamp_dims = getattr(self.problem, 'clamp_dims', self.D)
            pos_for_proj = positions[:, :clamp_dims]
            x_new_for_proj = x_new[:, :clamp_dims]
            _, x_perp_curr = self._project_to_subspace(pos_for_proj, U)
            _, x_perp_new  = self._project_to_subspace(x_new_for_proj, U)
            clamp_potential_curr = 0.5 * lam_eff * (x_perp_curr ** 2).sum(dim=1)
            clamp_potential_new  = 0.5 * lam_eff * (x_perp_new ** 2).sum(dim=1)
            V_eff_curr = energies + clamp_potential_curr
            V_eff_new  = new_energies + clamp_potential_new
        else:
            V_eff_curr = energies
            V_eff_new  = new_energies

        # NQS-1b: Metropolis uses T_eff_post = max(T_pre, T_post) for symmetry
        # between current and new stochastic estimates.
        T_eff_post = temps + cfg.noise_temperature_alpha * torch.maximum(stds, new_stds)

        delta_V   = V_eff_new - V_eff_curr
        log_alpha = (-delta_V / T_eff_post.clamp(min=1e-10)).clamp(max=0.0)
        log_u     = torch.rand(M, dtype=self.dtype, device=self.device,
                               generator=self._gen).log()
        accept    = (log_u < log_alpha)

        positions = torch.where(accept.unsqueeze(1), x_new,        positions)
        energies  = torch.where(accept,              new_energies, energies)
        stds      = torch.where(accept,              new_stds,     stds)
        return positions, energies, stds

    # ─────────────────────────────────────────────────────────────────────────
    # Replica reseed (unchanged)
    # ─────────────────────────────────────────────────────────────────────────

    def _check_reseed(
        self,
        positions           : torch.Tensor,
        energies            : torch.Tensor,
        stds                : torch.Tensor,
        stagnation_counters : List[int],
        energy_prev         : torch.Tensor,
        step                : int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        cfg = self.config
        M   = cfg.M
        n   = cfg.n_hot_reseed

        energy_change = (energies - energy_prev).abs()

        for i in range(M - 1, M - n - 1, -1):
            # NQS: stagnation threshold scaled by std to avoid noise-driven false positives
            stagnation_thresh = 1e-4 * float(energies[i].abs() + 1.0) + 2.0 * float(stds[i])
            if float(energy_change[i]) < stagnation_thresh:
                stagnation_counters[i] += 1
            else:
                stagnation_counters[i] = 0

            if stagnation_counters[i] >= cfg.reseed_threshold:
                x_new = self.problem.random_feasible(batch_size=1).to(
                    self.device, dtype=self.dtype)[0]
                positions[i] = x_new
                e_new, s_new = self._eval_total(x_new.unsqueeze(0))
                energies[i]  = e_new.squeeze()
                stds[i]      = s_new.squeeze()
                stagnation_counters[i] = 0

        return positions, energies, stds, stagnation_counters

    # ─────────────────────────────────────────────────────────────────────────
    # NQS-1d: Noise-corrected replica exchange
    # ─────────────────────────────────────────────────────────────────────────

    def _replica_exchange_v2(
        self,
        positions : torch.Tensor,
        energies  : torch.Tensor,
        stds      : torch.Tensor,
        temps     : torch.Tensor,
        step      : Optional[int]   = None,
        U         : Optional[torch.Tensor] = None,
        clamp_dims: Optional[int]   = None,
        swap_events : Optional[list] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Parallel tempering swap with NQS-1d noise correction.

        The standard PT swap criterion is
            Δ = (1/T_i - 1/T_j)(E_j - E_i)
            accept if Δ ≤ 0 or exp(-Δ) > u

        With stochastic energies, we compute the swap statistic using
        temperatures inflated by noise contribution:
            T_i_eff = T_i + α · σ_i
        This prevents noise-dominated swap decisions when σ_E ~ ΔE / (ΔT/T²).

        v5.1 swap-alignment diagnostics retained.
        """
        cfg   = self.config
        M     = self.config.M
        alpha = cfg.noise_temperature_alpha
        n_acc = 0

        pairs = list(range(M - 1))
        self._rng_cpu.shuffle(pairs)

        for i in pairs:
            T_i = temps[i] + alpha * stds[i]
            T_j = temps[i + 1] + alpha * stds[i + 1]
            delta = (
                (1.0 / T_i.clamp(min=1e-30) - 1.0 / T_j.clamp(min=1e-30))
                * (energies[i + 1] - energies[i])
            )
            log_u = torch.rand(
                1, dtype=self.dtype, device=self.device,
                generator=self._gen).log()
            if float(delta) <= 0.0 or float(log_u) < float(-delta):
                # Swap-alignment event recording (unchanged from v5.1)
                if swap_events is not None and U is not None and clamp_dims is not None:
                    if float(temps[i]) < float(temps[i+1]):
                        cold_in_pair, hot_in_pair = i, i + 1
                    else:
                        cold_in_pair, hot_in_pair = i + 1, i

                    x_cold_before = positions[cold_in_pair].clone().detach()
                    x_cold_after  = positions[hot_in_pair].clone().detach()

                    dx_full  = x_cold_after - x_cold_before
                    dx_clamp = dx_full[:clamp_dims]
                    norm_dx  = float(dx_clamp.norm())
                    cos_align = None
                    if norm_dx > 1e-8:
                        proj = U.T @ dx_clamp
                        cos_align = float(proj.norm() / norm_dx)

                    swap_events.append({
                        'step':            step,
                        'cold_idx':        cold_in_pair,
                        'hot_idx':         hot_in_pair,
                        'x_cold_before':   x_cold_before.cpu().numpy().copy(),
                        'x_cold_after':    x_cold_after.cpu().numpy().copy(),
                        'x_cold_at_swap':  x_cold_after.cpu().numpy().copy(),
                        'U_at_swap':       U.cpu().numpy().copy(),
                        'delta_energy':    float(energies[hot_in_pair] - energies[cold_in_pair]),
                        'sigma_cold':      float(stds[cold_in_pair]),
                        'sigma_hot':       float(stds[hot_in_pair]),
                        'cos_alignment':   cos_align,
                        'clamping_active': True,
                    })

                # Swap positions, energies, and stds
                tmp_pos        = positions[i].clone()
                positions[i]   = positions[i+1]
                positions[i+1] = tmp_pos
                tmp_e          = energies[i].clone()
                energies[i]    = energies[i+1]
                energies[i+1]  = tmp_e
                tmp_s          = stds[i].clone()
                stds[i]        = stds[i+1]
                stds[i+1]      = tmp_s
                n_acc += 1

        return positions, energies, stds, n_acc / max(len(pairs), 1)

    # ─────────────────────────────────────────────────────────────────────────
    # Barrier crossing detection (unchanged)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_barrier_crossing(
        obj_hist       : List[float],
        window         : int   = 20,
        spike_threshold: float = 0.02,
    ) -> List[Dict]:
        events = []
        n = len(obj_hist)
        drop_threshold = 0.001

        if n < window * 2:
            return events

        for t in range(window, n - window):
            pre_window  = obj_hist[t - window:t]
            post_window = obj_hist[t:t + window]
            pre_mean  = float(np.mean(pre_window))
            post_mean = float(np.mean(post_window))
            if pre_mean > 0:
                rel_drop = (pre_mean - post_mean) / abs(pre_mean)
            else:
                rel_drop = 0.0
            if rel_drop > drop_threshold:
                events.append({
                    'step': t,
                    'cost_before': pre_mean,
                    'cost_after':  post_mean,
                    'relative_drop': rel_drop,
                })
        return events

    # ─────────────────────────────────────────────────────────────────────────
    # C5 condition check (unchanged)
    # ─────────────────────────────────────────────────────────────────────────

    def _check_c5_condition(
        self,
        lam_eff        : float,
        grad_norm_ema  : float,
        tau_lambda_eff : float,
        step           : int,
    ) -> Optional[str]:
        if lam_eff < 1e-6:
            return None
        kappa_eff = max(grad_norm_ema, self.config.grad_norm_floor)
        tau_relax = 1.0 / kappa_eff
        if tau_lambda_eff < 5.0 * tau_relax:
            return (f"[Step {step}] C5 warning: τ_λ={tau_lambda_eff:.1f} < "
                    f"5·τ_relax={5*tau_relax:.1f}; H-theorem monotonicity "
                    f"approximation may break (Theorem 4.1 v5.0).")
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Main optimize loop
    # ─────────────────────────────────────────────────────────────────────────

    def optimize(
        self,
        callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
        callback_every: int = 100,
    ) -> Dict:
        """
        Main TCBM NQS optimization loop.

        Returns dict with all NN-v5 keys plus NQS-specific diagnostics:
          • best_cost_debiased     (NQS-1e: re-evaluated with 4× samples)
          • best_cost_raw          (original best observation)
          • subspace_source_diag   (NQS-3 mode diagnostics)
          • final_sigma_E          (σ_E at termination)

        Optional progress hook:
          callback(step, info_dict) is called every callback_every steps
          (and at step 0). info_dict contains read-only snapshot floats:
            cost_min, cost_mean, sigma_E_max, sigma_E_mean,
            swap_acc_recent, subspace_active, psi_max, T_w_step.
          callback failures are caught and warned, never abort optimize().
        """
        cfg  = self.config
        D    = self.D
        M    = cfg.M

        # ── Initialize ─────────────────────────────────────────────────────
        temps_base = self._init_temperatures()
        temps      = temps_base.clone()
        positions, energies, stds = self._init_replicas()
        U, sv = self._init_subspace(positions)
        grad_buffer = torch.zeros(cfg.grad_buffer_size, D,
                                   dtype=self.dtype, device=self.device)
        grad_buffer_idx = 0
        grad_buffer_filled = 0

        clamp_dims_problem = getattr(self.problem, 'clamp_dims', D)

        # State
        step_T_w : Optional[int] = None
        psi_history : List[Tuple[int, float]] = []
        sigma_history : List[Tuple[int, float, float]] = []
        belief_mature = False
        psi_at_T_w : Optional[float] = None
        grad_norm_ema = 0.1
        stagnation_counters = [0] * M
        energy_prev = energies.clone()

        # Best tracking (NQS-1e uses selection-bias correction at termination)
        best_idx_cold = torch.argmin(energies).item()
        best_x   = positions[best_idx_cold].clone().detach()
        best_cost_raw = float(energies[best_idx_cold])
        best_sigma = float(stds[best_idx_cold])

        # Trajectory
        traj = {
            'cost':            [], 'obj':             [], 'violation':       [],
            'subspace_dev':    [], 'lyapunov':        [], 'replica_energies': [],
            'free_energy':     [], 'swap_acceptance': [], 'lambda':          [],
            'gamma':           [], 'temperature':     [], 'time':            [],
            'psi':             [], 'psi_steps':       [],
            'sigma_k':         [], 'sigma_k1':        [],
            'delta_k':         [],
            'sigma_E':         [],       # NQS: mean σ_E across replicas
        }

        # Diagnostics
        swap_events : List[Dict] = []
        subspace_diag_history : List[Dict] = []
        t0 = time.time()
        swap_acc_last = 0.0

        # C5 warning dedup
        c5_warnings_issued = 0

        # ── Main loop ──────────────────────────────────────────────────────
        for step in range(cfg.n_steps):

            # ── Temperature annealing + v5-2 floor ────────────────────────
            anneal = math.exp(-cfg.T_anneal_gamma * step / cfg.T_anneal_tau)
            temps = (temps_base * anneal).clamp(min=cfg.T_min_floor)

            # ── λ and Γ schedules ─────────────────────────────────────────
            # Γ: exponential decay from Gamma_0
            gamma_eff = cfg.Gamma_0 * math.exp(-step / cfg.tau_Gamma)

            # λ: 0 before T_w, then smooth ramp from lambda_min toward lambda_max
            # driven by clock (step - step_T_w) with time constant tau_lambda.
            if step_T_w is None:
                lam_eff = 0.0
            else:
                t_since = step - step_T_w
                lam_eff = cfg.lambda_min + (cfg.lambda_max - cfg.lambda_min) * (
                    1.0 - math.exp(-t_since / cfg.tau_lambda)
                )

            # ── Snapshot cold replica for barrier-crossing analysis ───────
            cold_idx = 0
            x_cold_snap = positions[cold_idx].clone().detach()

            # ── Batch gradient ────────────────────────────────────────────
            grads = self._batch_gradient(positions)

            # Feed gradient buffer (cold + some replica gradients)
            for idx in range(min(4, M)):
                grad_buffer[grad_buffer_idx] = grads[idx].detach()
                grad_buffer_idx = (grad_buffer_idx + 1) % cfg.grad_buffer_size
                grad_buffer_filled = min(grad_buffer_filled + 1, cfg.grad_buffer_size)

            # ── NQS-3: Update subspace ────────────────────────────────────
            if (step >= cfg.subspace_warmup
                    and step % cfg.subspace_update_freq == 0
                    and grad_buffer_filled > cfg.k + 2):
                U_new, sv_new, psi, delta_k_star, sdiag = self._update_subspace(
                    grad_buffer[:grad_buffer_filled], U, positions=positions)
                U, sv = U_new, sv_new
                psi_history.append((step, psi))
                traj['delta_k'].append(delta_k_star)

                subspace_diag_history.append({'step': step, **sdiag})

                # v3-B: Check maturity trigger
                if (not belief_mature and step >= cfg.min_warmup
                        and psi >= cfg.psi_star):
                    belief_mature = True
                    step_T_w = step
                    psi_at_T_w = psi
            else:
                psi, delta_k_star = 0.0, 0.0

            # ── C5 runtime check ──────────────────────────────────────────
            tau_lambda_eff = cfg.tau_lambda
            c5_msg = self._check_c5_condition(
                lam_eff, grad_norm_ema, tau_lambda_eff, step)
            if c5_msg is not None and c5_warnings_issued < 3:
                print(c5_msg)
                c5_warnings_issued += 1

            # ── Forces + Langevin step + Metropolis (NQS-1b) ───────────────
            F_total, grad_norm_ema = self._compute_forces_batch(
                positions, energies, grads, step, lam_eff, gamma_eff,
                U, grad_norm_ema, temps, stds=stds)

            positions, energies, stds = self._langevin_step_batch(
                positions, energies, stds, F_total, temps, grad_norm_ema,
                lam_eff=lam_eff, U=U)

            # ── Reseed check ──────────────────────────────────────────────
            if step > 0 and step % cfg.reseed_interval == 0:
                positions, energies, stds, stagnation_counters = self._check_reseed(
                    positions, energies, stds, stagnation_counters, energy_prev, step)

            # ── Replica exchange (NQS-1d) ─────────────────────────────────
            if step % cfg.swap_interval == 0:
                positions, energies, stds, swap_acc = self._replica_exchange_v2(
                    positions, energies, stds, temps, step=step,
                    U=U, clamp_dims=clamp_dims_problem, swap_events=swap_events)
                swap_acc_last = swap_acc

            # ── Update best ───────────────────────────────────────────────
            # Conservative: take argmin of energies as current best candidate,
            # but delay de-biasing to termination (NQS-1e).
            idx_cur = int(torch.argmin(energies).item())
            if float(energies[idx_cur]) < best_cost_raw:
                best_x = positions[idx_cur].clone().detach()
                best_cost_raw = float(energies[idx_cur])
                best_sigma = float(stds[idx_cur])

            # ── Trajectory recording ──────────────────────────────────────
            if cfg.record_trajectory and step % cfg.record_every == 0:
                _, x_perp = self._project_to_subspace(positions, U)
                sub_dev = float(torch.norm(x_perp, dim=1).mean())
                lyap = float(energies.min()) + 0.5 * lam_eff * sub_dev ** 2

                traj['cost'].append(float(energies.min()))
                traj['obj'].append(float(energies.min()))
                traj['violation'].append(0.0)  # placeholder; NQS has no hard constraint
                traj['subspace_dev'].append(sub_dev)
                traj['lyapunov'].append(lyap)
                traj['replica_energies'].append(energies.detach().cpu().numpy().copy())
                traj['free_energy'].append(float(energies.mean()))
                traj['swap_acceptance'].append(swap_acc_last)
                traj['lambda'].append(lam_eff)
                traj['gamma'].append(gamma_eff)
                traj['temperature'].append(float(temps[0]))
                traj['time'].append(time.time() - t0)
                traj['psi_steps'].append(step)
                traj['psi'].append(psi)
                traj['sigma_k'].append(float(sv[-1]) if len(sv) > 0 else 0.0)
                traj['sigma_k1'].append(0.0)
                traj['sigma_E'].append(float(stds.mean()))

                sigma_history.append((step, float(stds.mean()), float(stds.max())))

            # ── Optional progress callback (NQS-Day3) ─────────────────────
            # Read-only snapshot. Callback failures must not abort the run.
            # Phase 2.5 Path κ1: expose latest replica/best state for callback-side
            # capture (probe instrumentation). Surgical attribute assignment; no
            # side effect on the optimization loop itself.
            self._latest_positions = positions
            self._best_positions = best_x
            if callback is not None and step % callback_every == 0:
                try:
                    psi_max_so_far = max((p for _, p in psi_history), default=0.0)
                    callback(step, {
                        'cost_min':         float(energies.min().item()),
                        'cost_mean':        float(energies.mean().item()),
                        'sigma_E_max':      float(stds.max().item()),
                        'sigma_E_mean':     float(stds.mean().item()),
                        'swap_acc_recent':  float(swap_acc_last),
                        'subspace_active':  bool(belief_mature),
                        'psi_max':          float(psi_max_so_far),
                        'T_w_step':         step_T_w,
                    })
                except Exception as _cb_err:
                    import warnings
                    warnings.warn(
                        f"callback at step {step} raised "
                        f"{type(_cb_err).__name__}: {_cb_err}"
                    )

            energy_prev = energies.clone()

        # ── NQS-1e: De-biased best re-evaluation ───────────────────────────
        n_final = cfg.n_vmc_samples_final
        if n_final and n_final > 1:
            # Re-evaluate best_x with more samples to correct selection bias.
            # We run multiple independent evaluations and take their mean.
            best_batch = best_x.unsqueeze(0).expand(n_final, -1).contiguous()
            final_energies, final_stds = self._eval_total(best_batch)
            best_cost_debiased = float(final_energies.mean())
            best_sigma_debiased = float(final_energies.std() / math.sqrt(n_final))
        else:
            best_cost_debiased = best_cost_raw
            best_sigma_debiased = best_sigma

        # ── Assemble result dict ───────────────────────────────────────────
        barrier_events = self._detect_barrier_crossing(traj['cost'])

        # Swap alignment summary
        swap_aligns = [e['cos_alignment'] for e in swap_events
                       if e.get('cos_alignment') is not None]
        swap_alignment_mean = float(np.mean(swap_aligns)) if swap_aligns else None
        swap_alignment_std  = float(np.std(swap_aligns))  if swap_aligns else None

        # v5-4: R_infinity_sq
        if hasattr(self.problem, 'dissipation_a') and hasattr(self.problem, 'dissipation_b'):
            a_diss = float(self.problem.dissipation_a)
            b_diss = float(self.problem.dissipation_b)
            R_infinity_sq = (b_diss + D * float(cfg.T_max)) / a_diss if a_diss > 0 else None
        else:
            R_infinity_sq = None

        # v5-5: n_eff_design from replica correlations
        rho_avg, n_eff_design = self._estimate_n_eff(traj['replica_energies'])

        result = {
            # v2/v3/v5 interface (all preserved)
            'best_x':           best_x.detach().cpu().numpy(),
            'best_cost':        best_cost_debiased,   # NQS-1e: debiased
            'best_obj':         best_cost_debiased,
            'best_violation':   0.0,
            'n_eval':           cfg.n_steps * cfg.M,
            'total_time':       time.time() - t0,
            'final_U':          U.detach().cpu().numpy(),
            'singular_values':  sv.detach().cpu().numpy(),
            'barrier_events':   barrier_events,
            'trajectory':       traj,
            'T_w':              step_T_w,
            'belief_mature':    belief_mature,
            'psi_at_T_w':       psi_at_T_w,
            'delta_k_star':     traj['delta_k'][-1] if traj['delta_k'] else 0.0,
            'R_infinity_sq':    R_infinity_sq,
            'rho_avg':          rho_avg,
            'n_eff_design':     n_eff_design,
            'swap_events':            swap_events,
            'swap_alignment_mean':    swap_alignment_mean,
            'swap_alignment_std':     swap_alignment_std,
            'n_accepted_swaps':       len(swap_events),
            'swap_recovery_alignment_mean': None,   # POC: not computed
            'swap_recovery_alignment_std':  None,
            'n_recovery_measured':          0,
            # NQS-specific additions
            'best_cost_raw':          best_cost_raw,
            'best_cost_debiased':     best_cost_debiased,
            'best_sigma_E':           best_sigma_debiased,
            'final_sigma_E_mean':     float(stds.mean()),
            'final_sigma_E_max':      float(stds.max()),
            'subspace_source':        cfg.subspace_source,
            'subspace_source_diag':   subspace_diag_history,
            'sigma_history':          sigma_history,
            # T4a diagnostic: clean (step, psi) sequence from SVD updates.
            # Unlike trajectory['psi'] which is sampled at record_every and
            # contains zeros when step is not an SVD update step, this list
            # contains ONLY actual SVD-update psi values.
            'psi_history':            psi_history,
        }
        return result

    def _estimate_n_eff(
        self,
        replica_energies_hist: List[np.ndarray],
    ) -> Tuple[Optional[float], Optional[float]]:
        """v5-5: Corrected design-effect sample count estimator."""
        if len(replica_energies_hist) < 20:
            return None, None
        tail = replica_energies_hist[-20:]
        arr = np.stack(tail, axis=0)    # (T_tail, M)
        if arr.shape[1] < 2:
            return 0.0, float(arr.shape[0])
        C = np.corrcoef(arr.T)
        M_rep = C.shape[0]
        off_diag = C[~np.eye(M_rep, dtype=bool)]
        rho_avg = float(np.mean(off_diag))
        T_eff = arr.shape[0]
        n_eff_design = T_eff * M_rep / (1.0 + (M_rep - 1) * rho_avg)
        return rho_avg, n_eff_design

    # ─────────────────────────────────────────────────────────────────────────
    # Warm-start entry point
    # ─────────────────────────────────────────────────────────────────────────

    def optimize_from(self, x0: torch.Tensor, n_steps: int) -> Dict:
        """Run optimization starting from a specific x0. Retained for API compatibility."""
        orig_n_steps = self.config.n_steps
        self.config.n_steps = n_steps
        # Inject x0 as replica 0's initial position by patching _init_replicas
        orig_init = self._init_replicas
        def patched_init():
            positions, energies, stds = orig_init()
            positions[0] = x0.to(self.device, dtype=self.dtype)
            e0, s0 = self._eval_total(positions[0:1])
            energies[0] = e0.squeeze()
            stds[0] = s0.squeeze()
            return positions, energies, stds
        self._init_replicas = patched_init
        try:
            result = self.optimize()
        finally:
            self._init_replicas = orig_init
            self.config.n_steps = orig_n_steps
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Ablation wrapper (unchanged structure from NN-v5)
# ─────────────────────────────────────────────────────────────────────────────

class TCBMAblation(TCBMOptimizer):
    """
    Ablation modes for NQS. Inherits all modes from NN-v5 plus two new:
      • '-qgt-orthogonal'    : force subspace_source='qgt' (TCBM subspace = QGT span)
      • '-hybrid-residual'   : force subspace_source='hybrid' (TCBM subspace ⊥ QGT)

    These two new ablations are NQS-3's key experimental knobs: they let
    you measure whether the TCBM clamping performance gain persists when
    restricted to the QGT span (it should vanish, if SR already covers
    those directions) vs. restricted to the QGT-orthogonal residual
    (where TCBM provides non-redundant directional information).
    """

    MODES = {
        'full', '-tunnel', '-clamp', '-entangle',
        '-subspace', '-anneal', 'premature_clamp',
        '-qgt-subspace', '-hybrid-subspace',    # NQS-3 ablations
    }

    def __init__(
        self,
        problem,
        config       : Optional[TCBMConfig] = None,
        ablation_mode: str = 'full',
    ):
        assert ablation_mode in self.MODES, (
            f"Unknown ablation mode '{ablation_mode}'. Valid: {self.MODES}")
        super().__init__(problem, config)
        self.ablation_mode = ablation_mode
        self._apply_ablation()

    def _apply_ablation(self):
        cfg  = self.config
        mode = self.ablation_mode

        if mode == '-tunnel':
            cfg.n_steps        *= cfg.M
            cfg.M               = 1
            cfg.Gamma_0         = 0.0
            cfg.swap_interval   = 10 ** 9
            cfg.reseed_interval = 10 ** 9

        elif mode == '-clamp':
            cfg.lambda_min = 0.0
            cfg.lambda_max = 0.0
            cfg.psi_star   = float('inf')
            cfg.max_warmup = 10 ** 9

        elif mode == '-entangle':
            cfg.Gamma_0 = 0.0

        elif mode == '-subspace':
            cfg.subspace_update_freq = 10 ** 9
            cfg.psi_star             = float('inf')

        elif mode == '-anneal':
            lam_mid            = (cfg.lambda_min * cfg.lambda_max) ** 0.5
            cfg.lambda_min     = lam_mid
            cfg.lambda_max     = lam_mid
            cfg.tau_lambda     = 1.0
            cfg.T_anneal_gamma = 0.0
            cfg.Gamma_0        = cfg.Gamma_0 * 0.5

        elif mode == 'premature_clamp':
            cfg.psi_star   = 0.0
            cfg.min_warmup = 0

        elif mode == '-qgt-subspace':
            # NQS-3 ablation: force QGT-only subspace
            cfg.subspace_source = 'qgt'

        elif mode == '-hybrid-subspace':
            # NQS-3 ablation: force hybrid (QGT-orthogonal residual)
            cfg.subspace_source = 'hybrid'


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Smoke test with a synthetic deterministic problem to verify that the
    NQS optimizer reduces to NN-v5 behaviour when cost_std=None.

    A separate smoke test with a stochastic TFIM problem is in
    tests/test_tcbm_nqs_stochastic.py (to be written in the POC experiment plan).
    """
    import sys
    sys.path.insert(0, "../..")

    # Mock problem: deterministic quadratic with noise toggle
    class MockProblem:
        def __init__(self, D=100, stochastic=False, device='cpu'):
            self.dim = D
            self.device = device
            self.stochastic = stochastic
            torch.manual_seed(0)
            self._A = torch.randn(D, D, device=device) * 0.1
            self._A = self._A @ self._A.T + 0.01 * torch.eye(D, device=device)
            self._b = torch.randn(D, device=device) * 0.5

        def random_feasible(self, batch_size):
            return torch.rand(batch_size, self.dim, device=self.device)

        def evaluate(self, x, n_samples=None):
            cost = 0.5 * (x @ self._A * x).sum(dim=1) + x @ self._b
            pen  = torch.zeros_like(cost)
            if self.stochastic:
                # Simulate VMC noise: σ ~ 0.01 / √n_samples
                ns = n_samples if n_samples is not None else 1000
                noise_scale = 0.01 / math.sqrt(ns)
                noise = torch.randn_like(cost) * noise_scale
                cost = cost + noise
                std = torch.full_like(cost, noise_scale)
                return cost, pen, std
            return cost, pen, None

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}")

    # ── Test 1: deterministic mode (should match NN-v5 behaviour) ─────────
    print("\n[1] Deterministic mock problem (std=None) ...")
    prob_det = MockProblem(D=100, stochastic=False, device=dev)
    cfg = TCBMConfig(M=8, n_steps=200, k=6,
                     subspace_warmup=30, subspace_update_freq=10,
                     psi_star=2.0, min_warmup=20,
                     T_min_floor=0.005, record_every=10,
                     subspace_source='gradient',
                     n_vmc_samples=None)
    opt = TCBMOptimizer(prob_det, cfg)
    res = opt.optimize()
    print(f"  best_cost={res['best_cost']:.4f}  n_eval={res['n_eval']}")
    print(f"  T_w={res['T_w']}  belief_mature={res['belief_mature']}")
    print(f"  final_sigma_E_mean={res['final_sigma_E_mean']:.6f}  (should be ~0)")
    print(f"  subspace_source={res['subspace_source']}  "
          f"#diag_events={len(res['subspace_source_diag'])}")

    # ── Test 2: stochastic mode ───────────────────────────────────────────
    print("\n[2] Stochastic mock problem (VMC-like noise) ...")
    prob_stoch = MockProblem(D=100, stochastic=True, device=dev)
    cfg2 = TCBMConfig(M=8, n_steps=200, k=6,
                      subspace_warmup=30, subspace_update_freq=10,
                      psi_star=2.0, min_warmup=20,
                      T_min_floor=0.005, record_every=10,
                      subspace_source='gradient',
                      n_vmc_samples=500,
                      noise_temperature_alpha=1.0)
    opt2 = TCBMOptimizer(prob_stoch, cfg2)
    res2 = opt2.optimize()
    print(f"  best_cost_raw     ={res2['best_cost_raw']:.4f}")
    print(f"  best_cost_debiased={res2['best_cost_debiased']:.4f}  (NQS-1e)")
    print(f"  best_sigma_E      ={res2['best_sigma_E']:.6f}")
    print(f"  final_sigma_E_mean={res2['final_sigma_E_mean']:.6f}  (should be >0)")

    # ── Test 3: Interface compatibility ───────────────────────────────────
    print("\n[3] Interface compatibility check ...")
    required_keys = ['best_x', 'best_cost', 'best_obj', 'best_violation',
                     'n_eval', 'total_time', 'final_U', 'singular_values',
                     'barrier_events', 'trajectory',
                     'T_w', 'belief_mature', 'psi_at_T_w',
                     'delta_k_star', 'R_infinity_sq', 'rho_avg', 'n_eff_design',
                     'best_cost_raw', 'best_cost_debiased', 'best_sigma_E',
                     'final_sigma_E_mean', 'subspace_source',
                     'subspace_source_diag']
    for k in required_keys:
        assert k in res2, f"Missing key: {k}"
    required_traj_keys = ['cost', 'obj', 'violation', 'subspace_dev',
                          'lyapunov', 'replica_energies', 'free_energy',
                          'swap_acceptance', 'lambda', 'gamma', 'temperature',
                          'time', 'psi', 'psi_steps', 'sigma_k', 'sigma_k1',
                          'delta_k', 'sigma_E']
    for k in required_traj_keys:
        assert k in res2['trajectory'], f"Missing traj key: {k}"
    print("  All required keys present. NQS interface compatible with NN-v5.")

    # ── Test 4: Ablation (gradient source only; qgt/hybrid need problem.qgt) ──
    print("\n[4] Ablation smoke test (gradient-source modes only) ...")
    for mode in ['full', '-clamp', '-tunnel', '-entangle']:
        cfg_ab = TCBMConfig(M=4, n_steps=40, k=4, subspace_warmup=10,
                            subspace_update_freq=5, psi_star=2.0, min_warmup=5,
                            T_min_floor=0.005, record_every=5,
                            subspace_source='gradient', n_vmc_samples=None)
        ab = TCBMAblation(prob_det, cfg_ab, ablation_mode=mode)
        r = ab.optimize()
        print(f"  mode={mode:22s}  best_cost={r['best_cost']:.4f}  "
              f"T_w={r['T_w']}  source={r['subspace_source']}")

    print("\n[SMOKE TEST PASSED]")
