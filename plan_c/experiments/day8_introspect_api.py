"""Day 8 — NetKet API introspection (Plan C, experiment line).

Verifies the *actual installed* NetKet 3.21.0 API signatures before writing
day8_smoke.py, per Phase-A Lesson 3 (don't copy second-hand snippets).
CPU-only (4x4 is tiny; avoid jax/torch CUDA contention).

Run:  python -u plan_c/experiments/day8_introspect_api.py
"""
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import inspect
import jax
import netket as nk

print("=== versions ===")
print("netket", nk.__version__)
print("jax", jax.__version__)
print("jax default backend:", jax.default_backend())


def sig(obj, name):
    try:
        print(f"\n--- {name} ---")
        print(name, inspect.signature(obj))
    except (TypeError, ValueError) as e:
        print(f"{name}: <no signature: {e}>")


# (1) graph
sig(nk.graph.Graph, "nk.graph.Graph")
# (2) hilbert
sig(nk.hilbert.Spin, "nk.hilbert.Spin")
# (3) Heisenberg operator — does it accept J as list by edge color?
print("\n--- nk.operator.Heisenberg ---")
print("has Heisenberg:", hasattr(nk.operator, "Heisenberg"))
if hasattr(nk.operator, "Heisenberg"):
    sig(nk.operator.Heisenberg, "nk.operator.Heisenberg")
    print(inspect.getdoc(nk.operator.Heisenberg))
# (4) lanczos_ed
sig(nk.exact.lanczos_ed, "nk.exact.lanczos_ed")
# (5) RBM — param_dtype vs dtype
sig(nk.models.RBM, "nk.models.RBM")
# (6) MetropolisExchange
sig(nk.sampler.MetropolisExchange, "nk.sampler.MetropolisExchange")
# (7) MCState
sig(nk.vqs.MCState, "nk.vqs.MCState")
# (8) SR / driver options
print("\n--- SR / drivers ---")
print("nk.optimizer.SR:", hasattr(nk.optimizer, "SR"))
print("nk.optimizer.Sgd:", hasattr(nk.optimizer, "Sgd"))
print("nk.VMC:", hasattr(nk, "VMC"))
print("nk.driver.VMC:", hasattr(nk.driver, "VMC"))
print("nk.driver.VMC_SR:", hasattr(nk.driver, "VMC_SR"))
if hasattr(nk.optimizer, "SR"):
    sig(nk.optimizer.SR, "nk.optimizer.SR")
if hasattr(nk, "VMC"):
    sig(nk.VMC, "nk.VMC")
if hasattr(nk.driver, "VMC_SR"):
    sig(nk.driver.VMC_SR, "nk.driver.VMC_SR")
# (9) QGT access on MCState
print("\n--- QGT access ---")
print("MCState.quantum_geometric_tensor:",
      hasattr(nk.vqs.MCState, "quantum_geometric_tensor"))
if hasattr(nk.vqs.MCState, "quantum_geometric_tensor"):
    sig(nk.vqs.MCState.quantum_geometric_tensor,
        "MCState.quantum_geometric_tensor")

print("\n=== introspection done ===")
