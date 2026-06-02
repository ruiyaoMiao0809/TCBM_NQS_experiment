"""Plan C — 4x4 J1-J2 system builders (single source of truth).

ENERGY CONVENTION (F-level decision, ratified by Nick+AI on Day 8):
    NetKet's nk.operator.Heisenberg uses the sigma.sigma (Pauli) convention,
    H = sum_edges J (sigma_i . sigma_j),  sigma = Pauli matrices (eigvals +-1).
    Phase A's ED / truth E0 = -8.4579 uses the spin-1/2 convention,
    H = sum_edges J (S_i . S_j),  S = sigma/2,  so  S.S = (1/4) sigma.sigma.

    => To make NetKet emit energies natively in the S.S scale (so Phase A's
       config + methodology thresholds — T1 rel_err, T2, gradient/step scale —
       carry over unchanged), we pass J/4 to NetKet. The /4 lives ONLY inside
       build_j1j2_hamiltonian(); every caller uses the physical J1=1.0, J2=0.5.

    Verified: lanczos_ed(build_j1j2_hamiltonian(...)) = -8.457923 (delta 2.3e-5).
    Guarded by plan_c/tests/test_energy_convention.py.

Edge-color convention for nk.operator.Heisenberg (J indexed 0-based by color):
    color 0 -> NN  -> J1
    color 1 -> NNN (diagonal) -> J2
"""
import netket as nk

E0_TARGET = -8.4579   # Phase A ED reference (4x4 PBC, J2/J1=0.5), S.S convention


def build_j1j2_graph(L: int = 4):
    """Build the LxL square-lattice graph (PBC) with NN(color0)+NNN(color1) edges.

    site(x, y) = (x % L) * L + (y % L).  Each bond added once:
      NN  : right (x+1,y), up (x,y+1)            -> 2*L^2 = 32 bonds for L=4
      NNN : "\\" (x+1,y+1), "/" (x+1,y-1)         -> 2*L^2 = 32 bonds for L=4
    """
    def site(x, y):
        return (x % L) * L + (y % L)

    edges = []
    for x in range(L):
        for y in range(L):
            s = site(x, y)
            edges.append((s, site(x + 1, y), 0))      # NN right
            edges.append((s, site(x, y + 1), 0))      # NN up
            edges.append((s, site(x + 1, y + 1), 1))  # NNN "\"
            edges.append((s, site(x + 1, y - 1), 1))  # NNN "/"
    return nk.graph.Graph(edges=edges)


def build_j1j2_hilbert(graph, total_sz: float = 0.0):
    """Spin-1/2 Hilbert space on the graph nodes, fixed total Sz (default 0)."""
    return nk.hilbert.Spin(s=0.5, N=graph.n_nodes, total_sz=total_sz)


def build_j1j2_hamiltonian(hilbert, graph, J1: float = 1.0, J2: float = 0.5):
    """J1-J2 Heisenberg Hamiltonian in the S.S (spin-1/2) convention.

    NetKet Heisenberg is sigma.sigma; we pass J/4 so the returned operator's
    spectrum is in the S.S scale (E0 = -8.4579). The /4 appears ONLY here.
    Callers pass physical couplings: build_j1j2_hamiltonian(hi, g, J1=1.0, J2=0.5).

    With a sequence J, NetKet's sign_rule defaults to False (no Marshall basis
    change); eigenvalues are unaffected by sign_rule regardless.
    """
    return nk.operator.Heisenberg(hilbert, graph, J=[J1 / 4.0, J2 / 4.0])
