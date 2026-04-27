"""
utils/ed_reference.py
======================
Exact diagonalization reference for 2D J1-J2 Heisenberg model.

Purpose: Provides the ground-truth E_0 for TCBM-NQS convergence validation
(Day 3 P0-1.3 in Protocol v2.1).

Approach: Lanczos on the S_z=0 sector of H. For 4×4 this gives a sparse
12870×12870 matrix; computing the 3 lowest eigenvalues takes a few minutes
on CPU.

Physics:
    H = J1 Σ_{<i,j>}   S_i · S_j    (NN)
      + J2 Σ_{<<i,j>>} S_i · S_j    (NNN diagonal)

    S_i · S_j = S_i^z S_j^z + (1/2)(S_i^+ S_j^- + S_i^- S_j^+)

Convention:
    - Spin-1/2: S^z eigenvalue ±1/2, σ^z eigenvalue ±1
    - Bitstring representation: bit k = 1 means spin ↑ at site k,
      bit k = 0 means spin ↓ at site k.
    - Site indexing: site (x, y) at index y*Lx + x, for x in [0, Lx), y in [0, Ly).

Reference values (4×4 PBC, J1=1):
    J2=0.0:  E_0/N ≈ -0.701  (Néel limit)
    J2=0.5:  E_0/N ≈ -0.497  (maximally frustrated)
    J2=1.0:  E_0/N ≈ -0.500  (near-stripe)

Usage:
    python utils/ed_reference.py                           # default: 4x4, J2=0.5
    python utils/ed_reference.py --Lx 4 --Ly 4 --J2 0.5
    python utils/ed_reference.py --J2_scan                 # sweeps J2 ∈ [0.0, 1.0]

Outputs (when run as script):
    results/ed_reference_j1j2_4x4_J2={value}.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


# ─────────────────────────────────────────────────────────────────────────────
# Bit utilities
# ─────────────────────────────────────────────────────────────────────────────

def popcount(x: int) -> int:
    """Count bits set to 1 in x."""
    count = 0
    while x:
        count += x & 1
        x >>= 1
    return count


def get_bit(state: int, site: int) -> int:
    """Return bit at position `site`. 1 = ↑, 0 = ↓."""
    return (state >> site) & 1


def flip_bits(state: int, i: int, j: int) -> int:
    """Flip bits at positions i and j simultaneously."""
    return state ^ ((1 << i) | (1 << j))


# ─────────────────────────────────────────────────────────────────────────────
# Lattice geometry (matches core/j1j2_problem.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_square_lattice_bonds(Lx: int, Ly: int) -> Tuple[List[Tuple[int, int]],
                                                          List[Tuple[int, int]]]:
    """
    NN and NNN bonds for square lattice PBC. Matches j1j2_problem.py convention.

    Returns
    -------
    nn_bonds  : list of (i, j)
    nnn_bonds : list of (i, j)
    """
    def idx(x, y):
        return (y % Ly) * Lx + (x % Lx)

    nn_bonds = []
    nnn_bonds = []
    for y in range(Ly):
        for x in range(Lx):
            i = idx(x, y)
            nn_bonds.append((i, idx(x + 1, y)))
            nn_bonds.append((i, idx(x, y + 1)))
            nnn_bonds.append((i, idx(x + 1, y + 1)))
            nnn_bonds.append((i, idx(x - 1, y + 1)))
    return nn_bonds, nnn_bonds


# ─────────────────────────────────────────────────────────────────────────────
# S_z=0 basis construction
# ─────────────────────────────────────────────────────────────────────────────

def build_sz0_basis(N: int) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Enumerate all states with exactly N/2 up-spins (S_z = 0 sector).

    Parameters
    ----------
    N : int, number of sites (must be even)

    Returns
    -------
    basis : (dim,) ndarray of int64, states in increasing order
    state_to_idx : dict, maps state bitstring → basis index
    """
    assert N % 2 == 0, "S_z=0 sector requires even N"
    target_popcount = N // 2

    basis_list = []
    for s in range(2 ** N):
        if popcount(s) == target_popcount:
            basis_list.append(s)

    basis = np.array(basis_list, dtype=np.int64)
    state_to_idx = {int(s): idx for idx, s in enumerate(basis)}
    return basis, state_to_idx


# ─────────────────────────────────────────────────────────────────────────────
# Hamiltonian construction (sparse CSR)
# ─────────────────────────────────────────────────────────────────────────────

def build_j1j2_hamiltonian_sz0(
    Lx: int, Ly: int,
    J1: float = 1.0, J2: float = 0.5,
    verbose: bool = True,
) -> Tuple[sp.csr_matrix, np.ndarray, Dict[int, int]]:
    """
    Construct Heisenberg J1-J2 Hamiltonian matrix in S_z=0 sector.

    Matrix element ⟨σ| H | σ'⟩ where σ, σ' are basis states:
        Diagonal: H_{σσ} = Σ_{ij} J_{ij} · (1/4) · σ_i^z σ_j^z
        Off-diagonal: H_{σσ'} = Σ_{ij} J_{ij} · (1/2) · [⟨σ| (S_i^+ S_j^- + h.c.) |σ'⟩]
            = J_{ij} · (1/2) if σ' = flip(σ, i, j) AND σ_i ≠ σ_j, else 0

    Parameters
    ----------
    Lx, Ly : lattice dimensions
    J1, J2 : coupling strengths
    verbose : print progress

    Returns
    -------
    H : scipy.sparse.csr_matrix, shape (dim, dim)
    basis : (dim,) ndarray, basis states
    state_to_idx : dict
    """
    N = Lx * Ly
    if verbose:
        print(f"  Building H for {Lx}x{Ly} ({N} sites), J1={J1}, J2={J2}")

    # S_z=0 basis
    basis, state_to_idx = build_sz0_basis(N)
    dim = len(basis)
    if verbose:
        print(f"  S_z=0 sector dim: {dim}")

    # Lattice bonds
    nn_bonds, nnn_bonds = build_square_lattice_bonds(Lx, Ly)
    bond_specs = [('NN', nn_bonds, J1), ('NNN', nnn_bonds, J2)]

    # Build sparse matrix in COO format, then convert
    row_idx = []
    col_idx = []
    data    = []

    t0 = time.time()
    for state_idx, state in enumerate(basis):
        state = int(state)
        diag_val = 0.0

        for bond_type, bonds, J in bond_specs:
            for (i, j) in bonds:
                si = 2 * get_bit(state, i) - 1   # ±1
                sj = 2 * get_bit(state, j) - 1

                # Diagonal: J · (1/4) σ_i^z σ_j^z
                diag_val += J * 0.25 * si * sj

                # Off-diagonal: J · (1/2) if σ_i ≠ σ_j
                if si * sj < 0:
                    state_flipped = flip_bits(state, i, j)
                    # Note: flipping simultaneously an ↑↓ pair keeps S_z unchanged
                    flipped_idx = state_to_idx[state_flipped]
                    row_idx.append(state_idx)
                    col_idx.append(flipped_idx)
                    data.append(J * 0.5)

        # Add diagonal element
        row_idx.append(state_idx)
        col_idx.append(state_idx)
        data.append(diag_val)

        if verbose and (state_idx + 1) % max(dim // 10, 1) == 0:
            elapsed = time.time() - t0
            print(f"    {state_idx + 1:6d}/{dim}  ({100*(state_idx+1)/dim:.0f}%, {elapsed:.1f}s)")

    # Build CSR
    H = sp.coo_matrix((data, (row_idx, col_idx)), shape=(dim, dim)).tocsr()

    # Symmetrize (numerical safety): H should already be symmetric by
    # construction (off-diagonal contributions from bond (i,j) and its reversal
    # cancel properly), but we enforce explicitly.
    H = 0.5 * (H + H.T)

    elapsed = time.time() - t0
    if verbose:
        print(f"  H built in {elapsed:.2f}s, nnz = {H.nnz}")
    return H, basis, state_to_idx


# ─────────────────────────────────────────────────────────────────────────────
# Lanczos ground state
# ─────────────────────────────────────────────────────────────────────────────

def compute_ground_state(
    Lx: int, Ly: int,
    J1: float = 1.0, J2: float = 0.5,
    k: int = 3,
    tol: float = 1e-10,
    verbose: bool = True,
) -> Dict:
    """
    Compute lowest k eigenvalues and ground-state eigenvector via Lanczos.

    Returns
    -------
    result : dict with keys
        E_eigvals : (k,) smallest algebraic eigenvalues
        E_0       : float, ground state energy
        E_per_site: float, E_0 / N
        gap       : float, E_1 - E_0 (first excitation gap)
        psi_0     : (dim,) ground state vector
        N         : int, number of sites
        dim       : int, Hilbert space dim
    """
    N = Lx * Ly
    if verbose:
        print(f"\n{'='*60}")
        print(f"ED computation for {Lx}x{Ly} J1-J2 at J2/J1={J2}")
        print(f"{'='*60}")

    H, basis, state_to_idx = build_j1j2_hamiltonian_sz0(
        Lx, Ly, J1=J1, J2=J2, verbose=verbose)

    if verbose:
        print(f"\n  Running Lanczos (k={k}, tol={tol})...")
    t0 = time.time()

    eigvals, eigvecs = eigsh(H, k=k, which='SA', tol=tol, maxiter=10000)

    elapsed = time.time() - t0

    # Sort eigenvalues in ascending order
    sort_idx = np.argsort(eigvals)
    eigvals = eigvals[sort_idx]
    eigvecs = eigvecs[:, sort_idx]

    E_0 = float(eigvals[0])
    gap = float(eigvals[1] - eigvals[0]) if k >= 2 else None
    psi_0 = eigvecs[:, 0]

    if verbose:
        print(f"  Lanczos converged in {elapsed:.2f}s")
        print(f"\n  Ground state energy E_0      = {E_0:.6f}")
        print(f"  E_0 per site                   = {E_0/N:.6f}")
        if gap is not None:
            print(f"  First excitation gap E_1 - E_0 = {gap:.6f}")
        print(f"  All eigvals (first {k}):")
        for i, e in enumerate(eigvals):
            print(f"    E_{i} = {e:.6f}  (per site: {e/N:.6f})")

    return {
        'E_eigvals': eigvals,
        'E_0': E_0,
        'E_per_site': E_0 / N,
        'gap': gap,
        'psi_0': psi_0,
        'N': N,
        'dim': len(basis),
        'Lx': Lx, 'Ly': Ly,
        'J1': J1, 'J2': J2,
        'lanczos_time_sec': elapsed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='ED reference for J1-J2 Heisenberg')
    parser.add_argument('--Lx', type=int, default=4)
    parser.add_argument('--Ly', type=int, default=4)
    parser.add_argument('--J1', type=float, default=1.0)
    parser.add_argument('--J2', type=float, default=0.5)
    parser.add_argument('--k', type=int, default=3, help='Number of eigenvalues to compute')
    parser.add_argument('--J2_scan', action='store_true',
                        help='Scan J2 ∈ [0.0, 1.0] with step 0.1 instead of single point')
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.J2_scan:
        J2_values = np.linspace(0.0, 1.0, 11)
        results_list = []
        for J2 in J2_values:
            result = compute_ground_state(
                Lx=args.Lx, Ly=args.Ly,
                J1=args.J1, J2=float(J2), k=args.k,
                verbose=True)
            results_list.append({
                'J2': float(J2),
                'E_0': result['E_0'],
                'E_per_site': result['E_per_site'],
                'gap': result['gap'],
            })
            print(f"\n  J2={J2:.1f}: E_0/N = {result['E_per_site']:.6f}, gap = {result['gap']:.6f}")

        # Save scan
        out_path = output_dir / f'ed_reference_j1j2_{args.Lx}x{args.Ly}_scan.json'
        with open(out_path, 'w') as f:
            json.dump({
                'Lx': args.Lx, 'Ly': args.Ly, 'J1': args.J1,
                'scan': results_list,
            }, f, indent=2)
        print(f"\n  Scan saved: {out_path}")

        # Print summary
        print(f"\n{'='*60}")
        print(f"  J2 scan summary ({args.Lx}x{args.Ly} PBC):")
        print(f"{'='*60}")
        print(f"    J2     E_0/N       gap")
        for r in results_list:
            print(f"    {r['J2']:.1f}    {r['E_per_site']:.6f}    {r['gap']:.6f}")
        print()

    else:
        # Single point
        result = compute_ground_state(
            Lx=args.Lx, Ly=args.Ly,
            J1=args.J1, J2=args.J2, k=args.k,
            verbose=True)

        # Save JSON (strip ndarray fields)
        save_data = {
            'Lx': result['Lx'], 'Ly': result['Ly'],
            'J1': result['J1'], 'J2': result['J2'],
            'N': result['N'], 'dim': result['dim'],
            'E_eigvals': result['E_eigvals'].tolist(),
            'E_0': result['E_0'],
            'E_per_site': result['E_per_site'],
            'gap': result['gap'],
            'lanczos_time_sec': result['lanczos_time_sec'],
            # Note: psi_0 omitted (large array, save separately if needed)
        }
        out_path = output_dir / f'ed_reference_j1j2_{args.Lx}x{args.Ly}_J2={args.J2:.2f}.json'
        with open(out_path, 'w') as f:
            json.dump(save_data, f, indent=2)

        # Save psi_0 separately as .npy
        psi_path = output_dir / f'ed_reference_j1j2_{args.Lx}x{args.Ly}_J2={args.J2:.2f}_psi0.npy'
        np.save(psi_path, result['psi_0'])

        print(f"\n  Saved: {out_path}")
        print(f"  Saved: {psi_path}")

    print("\n  ✅ ED reference computation complete.")


if __name__ == '__main__':
    main()
