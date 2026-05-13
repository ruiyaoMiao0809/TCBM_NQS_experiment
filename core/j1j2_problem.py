"""
core/j1j2_problem.py
====================
J1J2Problem — 2D spin-1/2 frustrated Heisenberg model on square lattice.

Target task for TCBM-NQS POC (Path γ, per TCBM_NQS_POC_Protocol_v2.md).

Hamiltonian:
    H = J1 · Σ_{<i,j>}   S_i · S_j    (nearest-neighbor)
      + J2 · Σ_{<<i,j>>} S_i · S_j    (next-nearest-neighbor diagonal)

This is the prototypical non-stoquastic Hamiltonian where NN-based variational
Monte Carlo is known to struggle with rugged landscapes and sign-rule traps
[Bukov, Schmitt & Dupont, SciPost Phys. 10, 147 (2021)].

POC configuration (as per Protocol v2):
  - Lattice: Lx × Ly with PBC, default 4×4
  - Couplings: J1=1.0, J2=0.5 (maximally frustrated point)
  - Ansatz: Complex Restricted Boltzmann Machine (Carleo-Troyer 2017)
  - Hidden density: α = M/N, default 2 → M=2N hidden units
  - Total parameters (real): D = 2·(N + M + N·M) = 1120 for 4×4 α=2

═══════════════════════════════════════════════════════════════════════════════
Interface contract (matches tcbm_optimizer_NQS.py expectations)
═══════════════════════════════════════════════════════════════════════════════

Required (all implemented in this skeleton):
  • self.dim : int                            # D = 1120 for 4×4 α=2
  • self.device : str
  • self.random_feasible(batch_size) -> (B, D) Tensor
  • self.evaluate(x, n_samples=None) -> (cost, penalty, cost_std)
       cost:    (B,) energy estimates
       penalty: (B,) zeros (NQS has no hard constraint)
       cost_std: (B,) VMC statistical error, or None if deterministic mode
  • self.gradient(positions) -> (B, D)         # via autograd

Optional (implemented as Week 2 delivery stub, complete by Day 9):
  • self.qgt(x, k_qgt=None) -> (D, D) Tensor  # quantum geometric tensor
  • self.clamp_dims : None                     # no partial clamping needed
  • self.dissipation_a, dissipation_b : None   # no R_inf bound needed

═══════════════════════════════════════════════════════════════════════════════
Week 1 delivery scope (this skeleton = Day 1-2 target):
  - Lattice geometry + neighbor lists (PBC) ................  [implemented]
  - Complex-RBM ansatz: log_psi, psi_amplitude ............  [implemented]
  - Metropolis-Hastings VMC sampler .........................  [implemented]
  - Local energy computation ................................  [implemented]
  - evaluate() with n_samples stochastic loss ...............  [implemented]
  - gradient() via autograd .................................  [implemented]
  - qgt() stub (raise NotImplementedError) ..................  [Day 9 delivery]
  - Unit tests for each method ..............................  [Day 3]

═══════════════════════════════════════════════════════════════════════════════
"""

import math
from typing import Optional, Tuple, List

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# RBM parameter layout utilities
# ─────────────────────────────────────────────────────────────────────────────

def rbm_param_layout(N: int, M: int) -> dict:
    """
    Complex RBM parameter vector layout (real representation).

    The RBM has complex parameters:
        visible bias a ∈ C^N      → 2N real params
        hidden bias  b ∈ C^M      → 2M real params
        weights      W ∈ C^(N×M)  → 2NM real params
    Total D = 2(N + M + NM).

    Real-valued flat vector θ stores them in this order:
        [Re(a), Im(a), Re(b), Im(b), Re(W.flatten()), Im(W.flatten())]

    Returns dict with slice indices and shapes for unpacking.
    """
    offsets = {}
    ptr = 0
    offsets['a_real'] = (slice(ptr, ptr + N), (N,));   ptr += N
    offsets['a_imag'] = (slice(ptr, ptr + N), (N,));   ptr += N
    offsets['b_real'] = (slice(ptr, ptr + M), (M,));   ptr += M
    offsets['b_imag'] = (slice(ptr, ptr + M), (M,));   ptr += M
    offsets['W_real'] = (slice(ptr, ptr + N*M), (N, M)); ptr += N*M
    offsets['W_imag'] = (slice(ptr, ptr + N*M), (N, M)); ptr += N*M
    offsets['total_dim'] = ptr
    return offsets


def unpack_rbm(theta: torch.Tensor, N: int, M: int):
    """
    Unpack flat real parameter vector into complex RBM parameters.

    theta : (D,) real tensor
    Returns a, b, W as complex tensors.
    """
    layout = rbm_param_layout(N, M)
    a = torch.complex(theta[layout['a_real'][0]], theta[layout['a_imag'][0]])   # (N,)
    b = torch.complex(theta[layout['b_real'][0]], theta[layout['b_imag'][0]])   # (M,)
    W_real = theta[layout['W_real'][0]].reshape(N, M)
    W_imag = theta[layout['W_imag'][0]].reshape(N, M)
    W = torch.complex(W_real, W_imag)                                            # (N, M)
    return a, b, W


# ─────────────────────────────────────────────────────────────────────────────
# Complex-RBM log amplitude (stable implementation)
# ─────────────────────────────────────────────────────────────────────────────

def log_psi_rbm(theta: torch.Tensor, sigma: torch.Tensor, N: int, M: int) -> torch.Tensor:
    """
    Compute log ψ(σ; θ) for Complex-RBM ansatz.

    ψ(σ; θ) = exp(Σ_i a_i σ_i) · Π_j 2 cosh(b_j + Σ_i W_ij σ_i)
    log ψ = Σ_i a_i σ_i  +  Σ_j log(2 cosh(b_j + Σ_i W_ij σ_i))

    The hidden contribution Σ_j log(2·cosh(θ_j)) is computed by
    `_log_2cosh_complex`, which absorbs the factor of 2 directly into the
    function output (Carleo-Troyer 2017 / NetKet convention). The caller
    therefore does not add a separate M·log(2) term.

    Parameters
    ----------
    theta : (D,) real tensor
    sigma : (..., N) tensor of ±1 spins
    N, M  : system dims

    Returns
    -------
    log_psi : (...) complex tensor
    """
    a, b, W = unpack_rbm(theta, N, M)   # (N,), (M,), (N, M)

    sigma_c = sigma.to(a.dtype)          # cast to complex
    # visible contribution: σ · a, shape (...,)
    v_contrib = sigma_c @ a              # (...,)

    # hidden activations: θ_j = b_j + Σ_i W_ij σ_i, shape (..., M)
    hidden_act = sigma_c @ W + b.unsqueeze(0)

    # Σ_j log(2·cosh(θ_j)); factor of 2 already inside _log_2cosh_complex
    log_2cosh_vals = _log_2cosh_complex(hidden_act)     # (..., M)
    h_contrib = log_2cosh_vals.sum(dim=-1)              # (...,)

    return v_contrib + h_contrib


def _log_2cosh_complex_native(z: torch.Tensor) -> torch.Tensor:
    """
    Stable log(2·cosh(z)) for complex z = u + iv. RAW form — call sites should use
    `_log_2cosh_complex` (the safe public wrapper) instead, which routes through
    `_SafeLog2CoshComplex.apply()` to handle the inner=0 singularity in backward.

    NaN at inner=0 (z = i·(π/2 + k·π)) is correct analytic behavior here in
    forward (cosh(z)=0 → ψ=0), but PyTorch native autograd produces 1/inner = inf
    in backward, which propagates to NaN gradients. See Issue 5 in
    docs/known_issues_day2.md and Day 4 diagnostic trail.

    Returns the natural quantity for RBM hidden activations (Carleo-Troyer 2017
    convention; see also NetKet log_cosh implementation), where each hidden
    unit contributes log(2·cosh(θ_j)) and the factor of 2 is absorbed into
    the function rather than tracked separately by the caller.

    Mathematical derivation
    -----------------------
    Magnitude: from |cosh(u+iv)|² = cosh²(u)·cos²(v) + sinh²(u)·sin²(v),
    using cosh² = 1 + sinh² gives
        |cosh(u+iv)|² = cosh²(u) - sin²(v).
    Factor cosh²(u) = e^(2|u|)·(1 + e^(-2|u|))²/4 to expose the dominant
    exponential:
        |cosh(u+iv)|² = (e^(2|u|)/4) · [(1 + e^(-2|u|))² - 4·sin²(v)·e^(-2|u|)]
    Taking log and adding log(2) for the factor of 2 in the function output,
    the -log(2) from the |cosh| derivation cancels:
        log|2·cosh(z)| = |u| + (1/2)·log[(1 + e^(-2|u|))² - 4·sin²(v)·e^(-2|u|)]

    Phase: arg(2·cosh(z)) = arg(cosh(z)) since 2 > 0. Standard form:
        arg(cosh(u+iv)) = atan2(sinh(u)·sin(v), cosh(u)·cos(v))
    Divide both arguments by cosh(u) > 0 (which cancels in atan2):
        arg(cosh(z)) = atan2(tanh(u)·sin(v), cos(v))
    tanh has bounded output [-1, 1] for any u ∈ ℝ, so this form has no
    overflow and a stable autograd backward.

    Numerical stability
    -------------------
    a = exp(-2·|u|) ∈ [0, 1] never overflows. For |u| ≳ 51 (float32) or
    ≳ 354 (float64), a underflows to 0 cleanly: inner → 1, log(inner) → 0,
    and the result reduces to its asymptote |u| + i·sign(u)·v.
    inner is strictly positive except at the cosh zeros z = i·(π/2 + k·π),
    where inner = 0 and log diverges to -∞ (correct analytic behavior;
    cosh is zero there). No epsilon clamp is added — clamps inject zero
    gradients into the backward pass, which would silently corrupt training.

    Phase branch note
    -----------------
    The phase output uses atan2 with branch cut at v = ±π, which is the
    standard principal value convention. When v ≈ π·k (integer k), float32
    rounding of math.pi causes sin(v) to evaluate as a tiny number with
    sign that depends on how the input was constructed. atan2 then returns
    ±π. This is mathematically correct (±π are equivalent mod 2π) but
    produces a piecewise-discontinuous gradient near these branch
    boundaries.
    For NQS training in the POC scope (init_scale=0.01, 3000 steps),
    hidden activation imag parts stay close to 0 and never approach π.
    If extending to longer training where this could become an issue,
    consider unwrapping the phase before atan2, or using log(complex_value)
    directly with PyTorch's unwrapped complex log.

    Autograd notes
    --------------
    abs(u) has zero subgradient at u=0 in PyTorch. Through the chain rule
    on the full formula:
        d log_magnitude / du
            = sign(u) + (1/(2·inner)) · d_inner/du
            = sign(u)·(1 - a²)/inner          (after algebraic simplification)
    At u=0: a=1 ⇒ 1-a²=0 ⇒ derivative = 0, matching the analytic value
    (log|cosh(iv)| is locally constant in u to first order). So abs(u)
    is safe to use directly; no need for sqrt(u²+ε) regularization.

    Parameters
    ----------
    z : (...,) complex tensor

    Returns
    -------
    log_2cosh : (...,) complex tensor of same shape and dtype as z
    """
    u, v = z.real, z.imag
    abs_u = torch.abs(u)
    a = torch.exp(-2.0 * abs_u)                                     # ∈ [0, 1]
    inner = (1.0 + a) ** 2 - 4.0 * torch.sin(v) ** 2 * a            # > 0 except at cosh zeros
    log_magnitude = abs_u + 0.5 * torch.log(inner)
    phase = torch.atan2(torch.tanh(u) * torch.sin(v), torch.cos(v))
    return torch.complex(log_magnitude, phase)


class _SafeLog2CoshComplex(torch.autograd.Function):
    """
    Numerically safe wrapper for log(2·cosh(z)) on complex z.

    Forward path: identical to _log_2cosh_complex_native (stable form).
    Backward path: catches NaN/inf in gradient at the singularity
                   z = i·(π/2 + k·π) and replaces them with 0.

    Mathematical justification:
    - cosh(z) = 0 at z = i·(π/2 + k·π) → wavefunction ψ = 0 there
    - VMC weights samples by |ψ|² = 0 → such sample contributes 0 to loss
    - Therefore gradient contribution should also be 0, not NaN

    Resolves Issue 5 in docs/known_issues_day2.md.

    Discovered Day 4 (2026-04-30) via narrow_band_diagnostic + probe_grad_nan.
    Fixed Day 5 Phase 2.
    """

    @staticmethod
    def forward(ctx, z):
        ctx.save_for_backward(z)
        with torch.no_grad():
            result = _log_2cosh_complex_native(z)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        z, = ctx.saved_tensors

        with torch.enable_grad():
            z_clone = z.detach().clone().requires_grad_(True)
            out = _log_2cosh_complex_native(z_clone)
            grad_z, = torch.autograd.grad(
                out, z_clone, grad_outputs=grad_output,
                create_graph=False, retain_graph=False,
                allow_unused=False,
            )

        # NaN/inf detection at the singularity inner=0:
        # PyTorch backward gives 1/inner * d(inner)/dz which is inf or NaN there.
        # Replace with 0 (mathematically correct: |ψ|²=0 → gradient contribution=0).
        if torch.is_complex(grad_z):
            grad_real = grad_z.real
            grad_imag = grad_z.imag
            mask_real = torch.isnan(grad_real) | torch.isinf(grad_real)
            mask_imag = torch.isnan(grad_imag) | torch.isinf(grad_imag)
            grad_real = torch.where(mask_real, torch.zeros_like(grad_real), grad_real)
            grad_imag = torch.where(mask_imag, torch.zeros_like(grad_imag), grad_imag)
            grad_z = torch.complex(grad_real, grad_imag)
        else:
            mask = torch.isnan(grad_z) | torch.isinf(grad_z)
            grad_z = torch.where(mask, torch.zeros_like(grad_z), grad_z)

        return grad_z


def _log_2cosh_complex(z: torch.Tensor) -> torch.Tensor:
    """Public API: routes through _SafeLog2CoshComplex.apply for safe backward."""
    return _SafeLog2CoshComplex.apply(z)


# ─────────────────────────────────────────────────────────────────────────────
# Lattice geometry (square lattice PBC)
# ─────────────────────────────────────────────────────────────────────────────

def build_square_lattice_bonds(Lx: int, Ly: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Build NN and NNN bond lists for square lattice with PBC.

    Site labeling: site (x, y) -> index y*Lx + x, for x in [0, Lx), y in [0, Ly).

    Returns
    -------
    nn_bonds  : list of (i, j) tuples, nearest-neighbor pairs (horizontal + vertical)
    nnn_bonds : list of (i, j) tuples, next-nearest-neighbor pairs (diagonal)
    """
    def idx(x, y):
        return (y % Ly) * Lx + (x % Lx)

    nn_bonds = []
    nnn_bonds = []

    for y in range(Ly):
        for x in range(Lx):
            i = idx(x, y)
            # NN: right, up (avoiding double-count via direction choice)
            nn_bonds.append((i, idx(x + 1, y)))   # horizontal
            nn_bonds.append((i, idx(x, y + 1)))   # vertical
            # NNN: upper-right, upper-left diagonals
            nnn_bonds.append((i, idx(x + 1, y + 1)))
            nnn_bonds.append((i, idx(x - 1, y + 1)))

    return nn_bonds, nnn_bonds


# ─────────────────────────────────────────────────────────────────────────────
# Main J1J2Problem class
# ─────────────────────────────────────────────────────────────────────────────

class J1J2Problem:
    """
    2D spin-1/2 J1-J2 Heisenberg model on square lattice (PBC).

    Example
    -------
    >>> problem = J1J2Problem(Lx=4, Ly=4, J1=1.0, J2=0.5, alpha=2, device='cuda')
    >>> theta_batch = problem.random_feasible(batch_size=12)         # (12, D)
    >>> cost, pen, std = problem.evaluate(theta_batch, n_samples=2000)
    >>> grad = problem.gradient(theta_batch)                          # (12, D)
    """

    def __init__(
        self,
        Lx: int = 4,
        Ly: int = 4,
        J1: float = 1.0,
        J2: float = 0.5,
        alpha: int = 2,
        device: str = 'cuda',
        thermalization_steps: int = 500,
        init_scale: float = 0.01,
    ):
        # Lattice
        self.Lx = Lx
        self.Ly = Ly
        self.N = Lx * Ly
        self.J1 = J1
        self.J2 = J2
        self.device = device

        # RBM dimensions
        self.M = alpha * self.N
        self.alpha = alpha
        layout = rbm_param_layout(self.N, self.M)
        self.dim = layout['total_dim']   # D = 2(N + M + NM)
        self._layout = layout

        # Lattice bonds (precomputed)
        nn_bonds, nnn_bonds = build_square_lattice_bonds(Lx, Ly)
        self._nn_bonds  = torch.tensor(nn_bonds,  dtype=torch.long, device=device)   # (n_nn, 2)
        self._nnn_bonds = torch.tensor(nnn_bonds, dtype=torch.long, device=device)   # (n_nnn, 2)

        # VMC parameters
        self.thermalization = thermalization_steps
        self.init_scale = init_scale

        # RNG
        self._gen = torch.Generator(device=device)
        self._gen.manual_seed(0)   # users override via optimizer's seed

        # Dtype
        self.real_dtype = torch.float32
        self.complex_dtype = torch.complex64

        # ── Optimizer interface hints ─────────────────────────────────────
        # NQS parameter space is unconstrained real numbers. The optimizer
        # queries these attrs via getattr; we declare them explicitly for
        # clarity. See tcbm_optimizer_NQS.py §Configuration §Box-constraint.
        self.box_constraint = False    # do NOT clamp θ to [0, 1] in Langevin
        self.clamp_dims = self.dim     # clamping applies to all D params

    def set_seed(self, seed: int) -> "J1J2Problem":
        """Reset internal RNG to ``seed`` for reproducible sampling.

        Call after construction (and before random_feasible/evaluate) when
        running multi-seed sweeps; module-level torch.manual_seed does NOT
        affect ``self._gen``. Returns self for fluent chaining.
        """
        self._gen.manual_seed(seed)
        return self

    # ─────────────────────────────────────────────────────────────────────
    # Interface required by TCBMOptimizer
    # ─────────────────────────────────────────────────────────────────────

    def random_feasible(self, batch_size: int) -> torch.Tensor:
        """
        Initial θ values (small random, centered around 0).

        Small initialization is critical for RBM: large |W| causes cosh overflow
        and gives poor initial energy. init_scale=0.01 is the common choice for
        NQS papers (Carleo 2017, Bukov 2021).
        """
        return (torch.randn(batch_size, self.dim, dtype=self.real_dtype,
                            device=self.device, generator=self._gen) * self.init_scale)

    def evaluate(
        self,
        x: torch.Tensor,
        n_samples: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Evaluate variational energy E(θ) for a batch of parameters.

        Parameters
        ----------
        x : (B, D) real tensor
        n_samples : int or None
            If None: fallback to a fixed default (2000)
            If positive: use this many VMC samples
            If 0 or negative: use exact evaluation (only if N ≤ 12)

        Returns
        -------
        cost : (B,) energies — autograd-tracked through θ for gradient()
        penalty : (B,) zeros
        cost_std : (B,) statistical errors (detached; not used for backward)

        Autograd contract
        -----------------
        evaluate() returns autograd-tracked tensors so that gradient() can
        call .backward() on (cost + penalty).sum(). The gradient computed
        this way is a "partial gradient" — it differentiates only the
        fixed-sample path, missing the score-function term
        ∂_θ log P(σ) · E_loc that comes from re-sampling under |ψ|².
        Bukov 2021 / NetKet use the full log-derivative estimator for
        unbiased gradients; we adopt that in Week 2 (see TODO at gradient()).
        For Week 1 baselines, partial gradient is sufficient per Protocol v2.1.

        Implementation note: per-batch outputs are collected into a list and
        torch.stack'd at the end, rather than written via in-place index
        assignment to a torch.zeros buffer. In-place assignment to a
        non-grad-tracking buffer would silently break the autograd graph.
        """
        B = x.shape[0]
        ns = n_samples if n_samples is not None else 2000

        costs = []
        stds = []

        for b in range(B):
            theta_b = x[b]
            if self.N <= 12 and ns <= 0:
                # Exact (full Hilbert space enumeration) — use for small tests only
                e_val = self._evaluate_exact(theta_b)
                e_std = torch.zeros((), dtype=self.real_dtype, device=self.device)
            else:
                # Stochastic VMC estimate
                e_val, e_std = self._evaluate_vmc(theta_b, n_samples=ns)
            costs.append(e_val)
            stds.append(e_std)

        cost = torch.stack(costs)                       # autograd-tracked
        cost_std = torch.stack(stds).detach()           # statistical info; backward never called on it
        penalty = torch.zeros_like(cost)
        return cost, penalty, cost_std

    def gradient(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Batched gradient of variational energy via autograd.

        Note: for stochastic evaluate, this is a noisy estimate.
        For production NQS, the proper gradient uses the log-derivative estimator:
            ∇_θ E = 2 Re[ ⟨ E_loc(σ) · (∂_θ log ψ*(σ)) ⟩ - ⟨E_loc⟩ · ⟨∂_θ log ψ*⟩ ]
        That estimator has lower variance. For POC scope we use autograd on the
        stochastic evaluate, which is simpler and sufficient for demonstration.

        TODO (Week 2 optimization): implement the log-derivative gradient estimator
        when VMC gradient noise becomes the bottleneck.
        """
        x = positions.detach().requires_grad_(True)
        cost, pen, _ = self.evaluate(x)
        total = (cost + pen).sum()
        total.backward()
        return x.grad.detach().clone()

    def qgt(
        self,
        x: torch.Tensor,
        k_qgt: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Quantum Geometric Tensor (QGT) at x[0] (cold replica).

        S_ij = Re[ ⟨ O_i* O_j ⟩ - ⟨O_i*⟩ ⟨O_j⟩ ]
        where O_i(σ) = ∂_i log ψ(σ, θ).

        Implementation (POC, D=1120 scale):
          - Draw VMC samples from |ψ|^2
          - Per-sample log-derivative via torch.func.vmap
          - Build full (D, D) QGT matrix

        For larger D, return low-rank factor V with QGT ≈ V·V.T. Not needed for POC.

        Returns
        -------
        S : (D, D) real tensor, symmetric, PSD

        NOTE: This is the Day 9 delivery. Stub for now.
        """
        raise NotImplementedError(
            "qgt() is Week 2 Day 9 delivery. For Week 1, use subspace_source='gradient'."
        )

    # ─────────────────────────────────────────────────────────────────────
    # Internal: VMC sampling
    # ─────────────────────────────────────────────────────────────────────

    def _sample_configs(
        self,
        theta: torch.Tensor,
        n_samples: int,
        n_chains: int = 4,
    ) -> torch.Tensor:
        """
        Draw samples from |ψ(σ; θ)|^2 via Metropolis-Hastings.

        Uses single spin-flip proposal. Runs n_chains parallel Markov chains,
        each contributing n_samples/n_chains samples after thermalization.

        Parameters
        ----------
        theta : (D,) real tensor
        n_samples : int
        n_chains : int, default 4

        Returns
        -------
        samples : (n_samples, N) tensor of ±1 spins
        """
        per_chain = n_samples // n_chains
        total_steps = self.thermalization + per_chain

        # Initialize chains uniformly random
        sigma = (2 * torch.randint(
            0, 2, (n_chains, self.N), dtype=torch.long,
            device=self.device, generator=self._gen) - 1).to(self.real_dtype)

        log_psi_current = log_psi_rbm(theta, sigma, self.N, self.M)
        log_prob_current = 2.0 * log_psi_current.real

        collected = []
        for step in range(total_steps):
            # Propose single-spin flips
            flip_sites = torch.randint(
                0, self.N, (n_chains,), device=self.device, generator=self._gen)
            sigma_proposed = sigma.clone()
            # Flip sign at proposed site for each chain
            rows = torch.arange(n_chains, device=self.device)
            sigma_proposed[rows, flip_sites] = -sigma[rows, flip_sites]

            log_psi_proposed = log_psi_rbm(theta, sigma_proposed, self.N, self.M)
            log_prob_proposed = 2.0 * log_psi_proposed.real

            log_u = torch.rand(
                n_chains, device=self.device, generator=self._gen).log()
            accept = log_u < (log_prob_proposed - log_prob_current)

            sigma = torch.where(accept.unsqueeze(-1), sigma_proposed, sigma)
            log_prob_current = torch.where(
                accept, log_prob_proposed, log_prob_current)

            if step >= self.thermalization:
                collected.append(sigma.clone())

        samples = torch.cat(collected, dim=0)    # (n_chains * per_chain, N)
        return samples[:n_samples]

    def _evaluate_vmc(
        self,
        theta: torch.Tensor,
        n_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate E(θ) = ⟨E_loc(σ)⟩ via VMC.

        Returns (E_mean, E_std_of_mean).
        """
        samples = self._sample_configs(theta, n_samples)
        E_loc, sample_safe = self._local_energy_batch(theta, samples)    # (n_samples,)

        # Phase 2.5 Branch 1' safeguard: drop unsafe samples (ratio blowup, |Re log ratio| >= 30)
        # before averaging. If ALL samples unsafe (extreme pathology), return NaN to
        # signal optimizer.
        n_safe = int(sample_safe.sum().item())
        if n_safe == 0:
            nan_t = torch.tensor(float('nan'), dtype=self.real_dtype, device=self.device)
            return nan_t, nan_t

        E_loc_safe = E_loc[sample_safe]
        E_mean = E_loc_safe.mean()
        E_std_of_mean = E_loc_safe.std() / math.sqrt(n_safe)
        # No .detach(): keep autograd graph through E_mean for partial gradient
        # via gradient(). cost_std is detached at evaluate() call site (it never
        # enters the backward path).
        return E_mean.real, E_std_of_mean.real

    def _evaluate_exact(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Exact evaluation via full Hilbert-space enumeration. Use only for N ≤ 12.
        """
        # Enumerate all 2^N spin configurations
        all_sigma = []
        for i in range(2 ** self.N):
            bits = [(i >> k) & 1 for k in range(self.N)]
            sigma_i = torch.tensor(
                [2*b - 1 for b in bits], dtype=self.real_dtype, device=self.device)
            all_sigma.append(sigma_i)
        all_sigma = torch.stack(all_sigma, dim=0)    # (2^N, N)

        log_psi = log_psi_rbm(theta, all_sigma, self.N, self.M)     # (2^N,) complex
        prob_unnorm = torch.exp(2.0 * log_psi.real)
        Z = prob_unnorm.sum()
        probs = prob_unnorm / Z

        E_loc, sample_safe = self._local_energy_batch(theta, all_sigma)  # (2^N,)
        # Phase 2.5 Branch 1' safeguard: zero unsafe configs from the weighted
        # sum. In exact mode, unsafe should be rare in healthy regimes; bias is
        # negligible. NOTE: addresses Variant B/C blowup, not A/D stall.
        safe_complex = sample_safe.to(E_loc.dtype)
        E = (probs.to(E_loc.dtype) * E_loc * safe_complex).sum().real
        # No .detach(): keep autograd graph through θ → log_psi → probs and E_loc
        # so gradient() can backpropagate. See evaluate() docstring.
        return E

    def reset_unsafe_count(self) -> None:
        """Reset Phase 2.5 unsafe sample tracking. Call at start of each optimize() run."""
        self._unsafe_sample_count = 0
        self._unsafe_total_evaluated = 0

    # ─────────────────────────────────────────────────────────────────────
    # Internal: Local energy computation (heart of the Hamiltonian)
    # ─────────────────────────────────────────────────────────────────────

    def _local_energy_batch(
        self,
        theta: torch.Tensor,
        sigma: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute E_loc(σ) for a batch of spin configurations.

        For Heisenberg-type Hamiltonian H = Σ_{<ij>} J_ij S_i · S_j:
            S_i · S_j = S_i^z S_j^z + (1/2)(S_i^+ S_j^- + S_i^- S_j^+)
                      = (1/4) σ_i^z σ_j^z + (1/2)(σ_i^+ σ_j^- + σ_i^- σ_j^+)

        Phase 2.5 Branch 1' safeguard (Day 7)
        --------------------------------------
        Returns a per-sample ``sample_safe`` mask alongside E_loc. A sample is
        marked unsafe if |Re(log ψ(σ')/ψ(σ))| ≥ UNSAFE_THRESHOLD = 15 on ANY
        contributing bond (i.e., bond where the physics mask σ_i ≠ σ_j fires).
        e^30 ≈ 1e13 — float32 ratio in this region typically signals 1/ψ
        underflow numerical artifact (verified Day 7 forensic on Variants B/C).
        Upstream callers (_evaluate_vmc, _evaluate_exact) filter unsafe samples
        from the mean. NOTE: addresses Variant B/C catastrophic blowup pathology,
        NOT Variant A/D stall pathology (separate root cause).

        Tracking: ``self._unsafe_sample_count`` and ``self._unsafe_total_evaluated``
        accumulate across calls; reset via ``self.reset_unsafe_count()``.

        Physical factor derivation (cross-check for Day 4-5 baseline)
        --------------------------------------------------------------
        Convention: spin-1/2 with S^z eigenvalues ±1/2, σ^z eigenvalues ±1.

        Diagonal term:
            S_i^z · S_j^z = (½ σ_i^z) · (½ σ_j^z) = (1/4) · σ_i^z σ_j^z
            Coefficient per bond: J · (1/4) · σ_i σ_j        →  see J*0.25 below

        Off-diagonal term:
            S_i^+ = |↑⟩⟨↓|_i, matrix element = 1   (NOT 1/2; this is the
                                                    raising operator on
                                                    |↓⟩, which gives |↑⟩
                                                    with coefficient 1).
            S_j^- = |↓⟩⟨↑|_j, matrix element = 1
            ⟨↑↓| S_i^+ S_j^- |↓↑⟩ = 1
            ⟨↓↑| S_i^- S_j^+ |↑↓⟩ = 1   (h.c.)
            For a given σ with σ_i ≠ σ_j, exactly ONE of (S^+ S^-, S^- S^+)
            connects σ → σ' = flip(σ, i, j); the other gives 0.
            So ⟨σ'| (S_i^+ S_j^- + S_i^- S_j^+) |σ⟩ = 1 (whenever σ_i ≠ σ_j).
            With the (1/2) prefactor in H:
            Coefficient per bond per such σ: J · (1/2) · ψ(σ')/ψ(σ)  → see J*0.5 below

        E_loc(σ) = Σ_{σ'} <σ|H|σ'> ψ(σ')/ψ(σ)

        The only σ' that contribute are σ itself (diagonal, from S^z S^z) and
        σ with a pair (i,j) of opposite-sign spins FLIPPED (off-diagonal, from
        S^+ S^- terms).

        Parameters
        ----------
        theta : (D,) real tensor
        sigma : (B, N) tensor of ±1

        Returns
        -------
        E_loc : (B,) complex tensor
        sample_safe : (B,) bool tensor — True for samples passing the Phase 2.5
            ratio safeguard on all contributing bonds. Callers should filter
            E_loc[sample_safe] before computing means.
        """
        UNSAFE_THRESHOLD = 15.0  # Phase 2.5 Option X: tightened from 30 (forensic showed healthy A/D max|x|≈12, ~3x margin to 15)

        B = sigma.shape[0]
        E_loc = torch.zeros(B, dtype=self.complex_dtype, device=self.device)
        sample_safe = torch.ones(B, dtype=torch.bool, device=self.device)

        log_psi_cur = log_psi_rbm(theta, sigma, self.N, self.M)   # (B,) complex

        # Process NN bonds (coupling J1)
        for bond_type, bonds, J in [
            ('NN', self._nn_bonds, self.J1),
            ('NNN', self._nnn_bonds, self.J2),
        ]:
            if len(bonds) == 0:
                continue
            for pair_idx in range(bonds.shape[0]):
                i = int(bonds[pair_idx, 0])
                j = int(bonds[pair_idx, 1])
                si = sigma[:, i]
                sj = sigma[:, j]

                # Diagonal part: J · (1/4) σ_i^z σ_j^z   (σ^z eigenvalue = ±1)
                # Factor 1/4 from spin-1/2: S^z = (1/2) σ^z
                E_loc = E_loc + (J * 0.25 * si * sj).to(self.complex_dtype)

                # Off-diagonal: J · (1/2)(S_i^+ S_j^- + S_i^- S_j^+)
                # Non-zero only when σ_i ≠ σ_j (i.e., si*sj = -1)
                # Flipping both spins in this case gives σ' with ψ(σ') amplitude
                # Contribution: J · (1/2) · ψ(σ')/ψ(σ) for those σ
                mask = (si * sj < 0)        # (B,) boolean
                if mask.any():
                    sigma_flipped = sigma.clone()
                    sigma_flipped[:, i] = -si
                    sigma_flipped[:, j] = -sj
                    log_psi_flipped = log_psi_rbm(
                        theta, sigma_flipped, self.N, self.M)
                    log_psi_diff = log_psi_flipped - log_psi_cur    # (B,) complex

                    # Phase 2.5 safeguard: a sample is unsafe if its ratio on
                    # this contributing bond has |Re(log ratio)| >= 30 (e^30 ≈ 1e13).
                    # Only checked where physics mask fires (mask=True), since
                    # bonds where mask=False don't contribute to E_loc anyway.
                    bond_unsafe = (log_psi_diff.real.abs() >= UNSAFE_THRESHOLD) & mask
                    sample_safe = sample_safe & ~bond_unsafe

                    ratio = torch.exp(log_psi_diff)    # (B,) complex
                    # Factor 1/2 from (1/2)(S^+ S^- + S^- S^+): for spin-1/2,
                    # S^+|↓⟩ = |↑⟩, S^-|↑⟩ = |↓⟩, so S_i^+ S_j^- |↓↑⟩ = |↑↓⟩
                    # coefficient is 1 (not 1/2). Combined with 1/2 in front:
                    # matrix element of (S_i^+ S_j^- + h.c.) between |↑↓⟩ and |↓↑⟩ is 1.
                    # So contribution is J · (1/2) · ratio · (mask).
                    off_diag_contrib = J * 0.5 * ratio
                    E_loc = E_loc + off_diag_contrib * mask.to(self.complex_dtype)

        # Tracking: accumulate unsafe count across calls
        n_unsafe_this_call = int((~sample_safe).sum().item())
        if not hasattr(self, '_unsafe_sample_count'):
            self._unsafe_sample_count = 0
            self._unsafe_total_evaluated = 0
        self._unsafe_sample_count += n_unsafe_this_call
        self._unsafe_total_evaluated += B

        return E_loc, sample_safe


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests (Week 1 Day 3 delivery)
# ─────────────────────────────────────────────────────────────────────────────

def test_dimensions():
    """Verify parameter count matches formula D = 2(N + M + NM)."""
    problem = J1J2Problem(Lx=4, Ly=4, alpha=2, device='cpu')
    assert problem.N == 16
    assert problem.M == 32
    expected_D = 2 * (16 + 32 + 16 * 32)
    assert problem.dim == expected_D, f"Expected D={expected_D}, got {problem.dim}"
    print(f"  ✓ dim check: D = {problem.dim}")


def test_bonds():
    """Verify bond counts on 4×4 PBC."""
    problem = J1J2Problem(Lx=4, Ly=4, alpha=2, device='cpu')
    # On L×L PBC: 2·L² NN bonds (horizontal + vertical), 2·L² NNN bonds (2 diagonals)
    expected_nn = 2 * 16
    expected_nnn = 2 * 16
    assert problem._nn_bonds.shape[0] == expected_nn
    assert problem._nnn_bonds.shape[0] == expected_nnn
    print(f"  ✓ bonds: {expected_nn} NN, {expected_nnn} NNN")


def test_log_psi_stability():
    """Verify log_psi returns finite values for random θ."""
    problem = J1J2Problem(Lx=4, Ly=4, alpha=2, device='cpu')
    theta = problem.random_feasible(1)[0]
    sigma = torch.randint(0, 2, (8, problem.N), dtype=torch.long) * 2 - 1
    sigma = sigma.to(torch.float32)
    log_psi = log_psi_rbm(theta, sigma, problem.N, problem.M)
    assert torch.isfinite(log_psi.real).all()
    assert torch.isfinite(log_psi.imag).all()
    print(f"  ✓ log_psi stability: max |log_psi| = {log_psi.abs().max():.3f}")


def test_evaluate_stochastic():
    """Verify evaluate returns (cost, penalty, std) with std > 0."""
    problem = J1J2Problem(Lx=4, Ly=4, alpha=2, device='cpu')
    theta_batch = problem.random_feasible(3)
    cost, pen, std = problem.evaluate(theta_batch, n_samples=500)
    assert cost.shape == (3,)
    assert pen.shape == (3,) and pen.abs().max() == 0
    assert std.shape == (3,) and (std > 0).all()
    print(f"  ✓ evaluate: cost = {cost.tolist()}, std = {std.tolist()}")


def test_sigma_std_sqrt_scaling():
    """Verify σ_E scales as 1/√N_samples."""
    problem = J1J2Problem(Lx=4, Ly=4, alpha=2, device='cpu')
    theta = problem.random_feasible(1)
    _, _, std_small = problem.evaluate(theta, n_samples=200)
    _, _, std_large = problem.evaluate(theta, n_samples=800)   # 4× samples
    ratio = (std_small / std_large).item()
    # Should be ~ √4 = 2, with Monte Carlo noise tolerance ±50%
    assert 1.3 < ratio < 3.0, f"std scaling wrong: ratio = {ratio}"
    print(f"  ✓ σ scaling with 1/√N: ratio = {ratio:.2f} (expected ~2)")


def test_energy_physical_range():
    """Verify E for random θ is in [-2N, 2N] physical range."""
    problem = J1J2Problem(Lx=4, Ly=4, alpha=2, device='cpu')
    theta_batch = problem.random_feasible(5)
    cost, _, _ = problem.evaluate(theta_batch, n_samples=500)
    N = problem.N
    assert (-2 * N < cost).all() and (cost < 2 * N).all(), \
        f"Energy out of range [{-2*N}, {2*N}]: {cost.tolist()}"
    print(f"  ✓ energies in [{-2*N}, {2*N}]: {cost.tolist()}")


def test_gradient_shape():
    """Verify gradient returns (B, D)."""
    problem = J1J2Problem(Lx=4, Ly=4, alpha=2, device='cpu')
    theta_batch = problem.random_feasible(2)
    grad = problem.gradient(theta_batch)
    assert grad.shape == (2, problem.dim)
    assert torch.isfinite(grad).all()
    print(f"  ✓ gradient: shape {grad.shape}, ||grad|| = {grad.norm(dim=1).tolist()}")


if __name__ == "__main__":
    print("J1J2Problem skeleton — unit tests")
    print("=" * 60)

    print("\n[1] Dimensions")
    test_dimensions()

    print("\n[2] Bond structure")
    test_bonds()

    print("\n[3] log_psi stability")
    test_log_psi_stability()

    print("\n[4] evaluate() stochastic contract")
    test_evaluate_stochastic()

    print("\n[5] σ_E ∝ 1/√N_samples scaling")
    test_sigma_std_sqrt_scaling()

    print("\n[6] Energy physical range")
    test_energy_physical_range()

    print("\n[7] gradient() shape")
    test_gradient_shape()

    print("\n" + "=" * 60)
    print("[ALL SKELETON TESTS PASSED]")
    print("\nTODOs for Week 1 Day 3-7:")
    print("  - [ ] Build sparse H via scipy, ED reference E_0 (see Protocol P0-1.3)")
    print("  - [ ] TCBM-gradient baseline reaches < 10% error (P0-1.2)")
    print("  - [ ] Adam baseline (P1-1.1)")
    print("  - [ ] SR baseline (P1-1.2)")
    print("\nTODO for Week 2 Day 8-9:")
    print("  - [ ] Implement qgt() method using torch.func.vmap + grad")
    print("  - [ ] Verify symmetric + PSD + low-rank structure")
