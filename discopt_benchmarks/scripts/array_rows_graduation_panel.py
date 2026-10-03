"""Graduation panel for ``DISCOPT_IN_TREE_ARRAY_ROWS`` (#1568; CLAUDE.md §5).

The flag changes nothing on a scalar-layout model (no FBBT view is built), so the
``.nl`` corpus is unaffected by construction; the class it can change is models
with array variables and array-valued rows. Panel: #1513's dispersion family
(array form) and the ``discopt.ml`` embeddings (ReLU big-M, sigmoid full space,
sigmoid reduced space) over a width x depth x seed grid.

Per instance, flag OFF then ON, each on a FRESHLY built model (a solve writes
implied bounds back onto its model), ``deterministic=True``. Checks:

* cert-clean -- when both arms certify, objectives agree; no published bound
  crosses the best incumbent either arm found; every published incumbent
  re-verifies with ``verify_point`` on the model; no certificate is lost ON;
* net-positive -- certificates gained, nodes on instances both arms certify,
  final bound on instances neither certifies, total wall.

Prints an executed-check count and exits non-zero if nothing was compared (§6).

    python -u discopt_benchmarks/scripts/array_rows_graduation_panel.py [--time-limit 30]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

FLAG = "DISCOPT_IN_TREE_ARRAY_ROWS"


def dispersion(n: int):
    import discopt.modeling as dm

    m = dm.Model(f"disp{n}")
    x = m.continuous("x", shape=(n,), lb=0.0, ub=1.0)
    y = m.continuous("y", shape=(n,), lb=0.0, ub=1.0)
    t = m.continuous("t", lb=0.0, ub=2.0)
    for i in range(n):
        for j in range(i + 1, n):
            m.subject_to((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 >= t)
    m.maximize(t)
    return m


def network(act: str, method: str, width: int, depth: int, seed: int):
    import discopt.modeling as dm
    from discopt.ml import add_predictor
    from discopt.ml.network import DenseLayer, NetworkDefinition

    rng = np.random.default_rng(seed)
    sizes = [2] + [width] * depth + [1]
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
        w = rng.normal(0, 1 / np.sqrt(a), size=(a, b))
        bias = rng.normal(0, 0.1, size=b)
        layers.append(DenseLayer(w, bias, act if i < depth else "linear"))
    m = dm.Model(f"{act}_{method}_{width}x{depth}_s{seed}")
    x = m.continuous("x", shape=(2,), lb=-1.0, ub=1.0)
    net = NetworkDefinition(layers, input_bounds=(np.full(2, -1.0), np.full(2, 1.0)))
    y, _ = add_predictor(m, x, net, method=method)
    m.minimize(y[0])
    return m


def instances():
    for n in (3, 4, 5, 6):
        yield f"disp{n}", (lambda n=n: dispersion(n))
    forms = (("relu", "relu_bigm"), ("sigmoid", "full_space"), ("sigmoid", "reduced_space"))
    for act, method in forms:
        for width in (10, 25):
            for depth in (1, 2):
                for seed in (0, 1):
                    yield (
                        f"{method}_{width}x{depth}_s{seed}",
                        lambda a=act, me=method, w=width, d=depth, s=seed: network(a, me, w, d, s),
                    )


def solve(build, flag: str, tl: float):
    from discopt.validation.feasibility import verify_point

    os.environ[FLAG] = flag
    m = build()
    t0 = time.perf_counter()
    r = m.solve(time_limit=tl, deterministic=True)
    wall = time.perf_counter() - t0
    bad = None
    if r.x is not None and r.status in ("optimal", "feasible"):
        flat = np.concatenate(
            [np.ravel(np.asarray(r.x[v.name], dtype=np.float64)) for v in m._variables]
        )
        bad = not verify_point(m, flat).ok
    from discopt.modeling.core import ObjectiveSense

    maximize = m._objective.sense == ObjectiveSense.MAXIMIZE
    return r, wall, bad, maximize


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=30.0)
    args = ap.parse_args()
    import discopt

    print(f"discopt from {discopt.__file__}; load {os.getloadavg()}", flush=True)
    checks = compared = 0
    viol, gained, lost = [], [], []
    nodes_both = {"0": 0, "1": 0}
    wall = {"0": 0.0, "1": 0.0}
    tighter = looser = 0
    for name, build in instances():
        res = {f: solve(build, f, args.time_limit) for f in ("0", "1")}
        compared += 1
        r0, r1 = res["0"][0], res["1"][0]
        maximize = res["0"][3]
        for f in ("0", "1"):
            wall[f] += res[f][1]
            if res[f][2] is not None:
                checks += 1
                if res[f][2]:
                    viol.append(f"{name} flag={f}: published incumbent fails verify_point")
        incs = [r.objective for r in (r0, r1) if r.objective is not None and r.x is not None]
        best = (max(incs) if maximize else min(incs)) if incs else None
        for f, r in (("0", r0), ("1", r1)):
            if r.bound is not None and best is not None and np.isfinite(r.bound):
                checks += 1
                crossed = (
                    r.bound < best - 1e-6 * (1 + abs(best))
                    if maximize
                    else (r.bound > best + 1e-6 * (1 + abs(best)))
                )
                if crossed:
                    viol.append(f"{name} flag={f}: bound {r.bound} crosses incumbent {best}")
        c0, c1 = bool(r0.gap_certified), bool(r1.gap_certified)
        if c0 and c1:
            checks += 1
            if abs(r0.objective - r1.objective) > 1e-5 * (1 + abs(r0.objective)):
                viol.append(f"{name}: certified objectives differ {r0.objective} vs {r1.objective}")
            nodes_both["0"] += r0.node_count
            nodes_both["1"] += r1.node_count
        if c0 and not c1:
            lost.append(name)
        if c1 and not c0:
            gained.append(name)
        if not c0 and not c1 and r0.bound is not None and r1.bound is not None:
            better = r1.bound < r0.bound if maximize else r1.bound > r0.bound
            worse = r1.bound > r0.bound if maximize else r1.bound < r0.bound
            tighter += bool(better and abs(r1.bound - r0.bound) > 1e-9)
            looser += bool(worse and abs(r1.bound - r0.bound) > 1e-9)
        st = r1.solver_stats or {}
        print(
            f"{name:28s} OFF {r0.status:9s} {'C' if c0 else '-'} n={r0.node_count:<6d} "
            f"b={r0.bound!s:>22.12s} | "
            f"ON {r1.status:9s} {'C' if c1 else '-'} n={r1.node_count:<6d} "
            f"b={r1.bound!s:>22.12s} expanded={int(st.get('reduce/array_rows_expanded', 0))} "
            f"declined={int(st.get('reduce/array_rows_declined', 0))} "
            f"wall {res['0'][1]:.1f}/{res['1'][1]:.1f} load={os.getloadavg()[0]:.2f}",
            flush=True,
        )
    print(f"\nCOMPARED {compared}; executed checks {checks}")
    print(f"violations {len(viol)}")
    for v in viol:
        print("  VIOLATION", v)
    print(f"certificates gained ON {len(gained)} {gained}")
    print(f"certificates lost ON   {len(lost)} {lost}")
    print(f"nodes on both-certified: OFF {nodes_both['0']} ON {nodes_both['1']}")
    print(f"uncertified both: ON bound tighter {tighter}, looser {looser}")
    print(f"total wall: OFF {wall['0']:.0f}s ON {wall['1']:.0f}s")
    return 0 if compared and checks else 1


if __name__ == "__main__":
    sys.exit(main())
