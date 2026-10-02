"""Neural-network embedding benchmark: discopt.ml vs OMLT (Pyomo), separating the
FORMULATION from the SOLVER.

Every instance is one numpy-generated feedforward network ``y = NN(x)`` embedded
twice with byte-identical weights -- once by ``discopt.ml.add_predictor`` and once
by OMLT -- and minimised over the input box ``x in [-1, 1]^d``. Four arms:

    F-discopt + S-discopt   discopt.ml model, Model.solve
    F-omlt    + S-ref       OMLT model, HiGHS (relu, MILP) or SCIP via .nl (sigmoid)
    F-omlt    + S-discopt   OMLT model, SolverFactory("discopt")
    F-discopt + S-ref       discopt.ml model -> to_nl -> SCIP (pyscipopt)

so a difference between rows 1 and 3 is the formulation, between rows 1 and 4 the
solver. SCIP has no tanh handler, so the smooth activation is sigmoid
(1/(1+exp(-z)), which both tools emit with exp).

Correctness first: certified objectives must agree across arms (within 1e-4 rel /
1e-6 abs); a disagreement is printed as DISAGREE and counted. The script asserts
its own instrument first -- both embeddings reproduce a numpy forward pass at a
pinned input (CLAUDE.md §6) -- and refuses to time anything above a load gate
(CLAUDE.md §9).

    python -u discopt_benchmarks/scripts/omlt_scaling_panel.py --time-limit 30 \
        --widths 10,25,50,100 --depths 1,2,3 --seeds 2 --out results.tsv
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

import numpy as np

ARMS = ("discopt", "omlt+ref", "omlt+discopt", "discopt+scip")


# ── networks ──────────────────────────────────────────────────────────────────


def make_net(d, width, depth, act, seed):
    rng = np.random.default_rng(seed)
    sizes = [d] + [width] * depth + [1]
    return [
        (
            rng.normal(0, 1 / np.sqrt(a), size=(a, b)),
            rng.normal(0, 0.1, size=b),
            act if i < depth else "linear",
        )
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:], strict=True))
    ]


def forward(layers, x):
    for w, b, act in layers:
        x = x @ w + b
        if act == "relu":
            x = np.maximum(x, 0)
        elif act == "sigmoid":
            x = 1.0 / (1.0 + np.exp(-x))
    return x


# ── builders ──────────────────────────────────────────────────────────────────


def build_omlt(layers, d, form):
    import omlt
    import pyomo.environ as pyo
    from omlt.neuralnet import (
        FullSpaceSmoothNNFormulation,
        NetworkDefinition,
        ReducedSpaceSmoothNNFormulation,
        ReluBigMFormulation,
    )
    from omlt.neuralnet.layer import DenseLayer, InputLayer

    net = NetworkDefinition(scaled_input_bounds=dict.fromkeys(range(d), (-1.0, 1.0)))
    prev = InputLayer([d])
    net.add_layer(prev)
    for w, b, a in layers:
        layer = DenseLayer(list(prev.output_size), [w.shape[1]], activation=a, weights=w, biases=b)
        net.add_layer(layer)
        net.add_edge(prev, layer)
        prev = layer
    m = pyo.ConcreteModel()
    m.nn = omlt.OmltBlock()
    cls = {
        "bigm": ReluBigMFormulation,
        "full": FullSpaceSmoothNNFormulation,
        "reduced": ReducedSpaceSmoothNNFormulation,
    }[form]
    m.nn.build_formulation(cls(net))
    m.obj = pyo.Objective(expr=m.nn.outputs[0])
    return m


def build_discopt(layers, d, form):
    import discopt.modeling as dm
    from discopt.ml import add_predictor
    from discopt.ml.network import DenseLayer, NetworkDefinition

    m = dm.Model("nn")
    x = m.continuous("x", shape=(d,), lb=-1.0, ub=1.0)
    net = NetworkDefinition(
        [DenseLayer(w, b, a) for w, b, a in layers],
        input_bounds=(np.full(d, -1.0), np.full(d, 1.0)),
    )
    method = {"bigm": "relu_bigm", "full": "full_space", "reduced": "reduced_space"}[form]
    y, _ = add_predictor(m, x, net, method=method)
    m.minimize(y[0])
    return m, x


# ── solvers ───────────────────────────────────────────────────────────────────


_SCIP_CHILD = """
import json, sys
from pyscipopt import Model
s = Model(); s.hideOutput(); s.setParam("limits/time", float(sys.argv[2]))
s.readProblem(sys.argv[1]); s.optimize()
print(json.dumps([s.getStatus(), s.getObjVal() if s.getNSols() else None, s.getDualbound()]))
"""


def _scip_nl(path, tl):
    """SCIP on an ``.nl`` file, in a child process: SCIP's NL reader segfaults on
    some inputs (discopt.ml's sigmoid full-space export, measured), and a native
    crash must cost one row, not the whole panel."""
    import json
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-c", _SCIP_CHILD, path, str(tl)],
        capture_output=True,
        text=True,
        timeout=tl + 120,
    )
    if proc.returncode != 0:
        return f"crashed:rc={proc.returncode}", None, None, False
    status, obj, bound = json.loads(proc.stdout.strip().splitlines()[-1])
    return status, obj, bound, status == "optimal"


def solve_arm(arm, layers, d, form, tl):
    """(status, objective, bound, certified, size) -- never raises; an exception is
    returned as the status so it is reported, not hidden (CLAUDE.md §7)."""
    import pyomo.environ as pyo

    try:
        if arm in ("discopt", "discopt+scip"):
            m, _ = build_discopt(layers, d, form)
            size = (m.num_variables, m.num_constraints, m.num_integer)
            if arm == "discopt":
                r = m.solve(time_limit=tl)
                return r.status, r.objective, r.bound, bool(r.gap_certified), size
            with tempfile.TemporaryDirectory() as td:
                f = os.path.join(td, "m.nl")
                with open(f, "w") as fh:
                    fh.write(m.to_nl())
                return (*_scip_nl(f, tl), size)
        om = build_omlt(layers, d, form)
        size = (
            sum(1 for _ in om.component_data_objects(pyo.Var, active=True)),
            sum(1 for _ in om.component_data_objects(pyo.Constraint, active=True)),
            sum(1 for v in om.component_data_objects(pyo.Var, active=True) if v.is_binary()),
        )
        if arm == "omlt+discopt":
            import discopt.pyomo  # noqa: F401 -- registers SolverFactory("discopt")

            res = pyo.SolverFactory("discopt").solve(om, timelimit=tl)
            tc = str(res.solver.termination_condition)
            lb = getattr(res.problem, "lower_bound", None)
            return tc, pyo.value(om.obj), lb, tc == "optimal", size
        if form == "bigm":
            res = pyo.SolverFactory("highs").solve(om, timelimit=tl, load_solutions=False)
            tc = str(res.solver.termination_condition)
            ub, lb = res.problem.upper_bound, res.problem.lower_bound
            obj = None if ub is None or not np.isfinite(ub) else float(ub)
            return tc, obj, lb, tc == "optimal", size
        with tempfile.TemporaryDirectory() as td:
            f = os.path.join(td, "m.nl")
            om.write(f, format="nl")
            return (*_scip_nl(f, tl), size)
    except Exception as exc:  # reported as an outcome
        return f"raised:{type(exc).__name__}", None, None, False, None


# ── driver ────────────────────────────────────────────────────────────────────


def self_test():
    """Both embeddings reproduce numpy at a pinned input, or nothing is measured."""
    import discopt.modeling as dm  # noqa: F401
    import pyomo.environ as pyo

    checks = 0
    for act, form in (("relu", "bigm"), ("sigmoid", "full"), ("sigmoid", "reduced")):
        layers = make_net(2, 8, 2, act, seed=0)
        xp = np.array([0.3, -0.7])
        truth = float(forward(layers, xp)[0])
        m, x = build_discopt(layers, 2, form)
        m.subject_to(x[0] == float(xp[0]))
        m.subject_to(x[1] == float(xp[1]))
        r = m.solve(time_limit=30)
        assert r.objective is not None and abs(r.objective - truth) < 1e-5, (
            form,
            r.objective,
            truth,
        )
        om = build_omlt(layers, 2, form)
        for i in range(2):
            om.nn.inputs[i].fix(float(xp[i]))
        with tempfile.TemporaryDirectory() as td:
            f = os.path.join(td, "m.nl")
            om.write(f, format="nl")
            _, o_val, _, _ = _scip_nl(f, 30)
        assert o_val is not None and abs(o_val - truth) < 1e-5, (form, o_val, truth)
        checks += 1
        _ = pyo
    print(f"self-test: {checks} embeddings x 2 tools reproduce the numpy forward pass", flush=True)
    assert checks == 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--widths", default="10,25,50,100")
    ap.add_argument("--depths", default="1,2,3")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--d", type=int, default=2)
    ap.add_argument("--max-load", type=float, default=1.5)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--forms",
        default="relu:bigm,sigmoid:full,sigmoid:reduced",
        help="comma-separated act:form pairs to run (chunk a long sweep)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="append to --out, skipping (instance, arm) rows it already holds",
    )
    args = ap.parse_args()

    load = os.getloadavg()[0]
    print(f"load {os.getloadavg()}", flush=True)
    if load > args.max_load:
        print(f"refusing to time: 1-min load {load:.2f} > {args.max_load} (CLAUDE.md §9)")
        return 2
    self_test()

    rows, disagree, compared = [], 0, 0

    done: set[tuple] = set()
    if args.resume and args.out and os.path.exists(args.out):
        with open(args.out) as fh:
            for row in fh.read().splitlines()[1:]:
                f = row.split("\t")
                if len(f) >= 6:
                    done.add(tuple(f[:6]))
        print(f"resume: {len(done)} rows already in {args.out}", flush=True)

    def emit(line: str, *, first: bool = False) -> None:
        print(line, flush=True)
        if args.out:  # appended line by line so a long run leaves a usable partial file
            with open(args.out, "w" if first else "a") as fh:
                print(line, file=fh)

    hdr = (
        "act\tform\twidth\tdepth\tseed\tarm\tstatus\tcert\tobjective\tbound\twall\tvars\tcons\tbins"
        "\tload"
    )
    if not (args.resume and done):
        emit(hdr, first=True)
    pairs = [tuple(p.split(":")) for p in args.forms.split(",")]
    for act, form in pairs:
        for depth in map(int, args.depths.split(",")):
            for width in map(int, args.widths.split(",")):
                for seed in range(args.seeds):
                    layers = make_net(args.d, width, depth, act, seed)
                    certs = {}
                    for arm in ARMS:  # interleaved: all arms of one instance back to back
                        key = (act, form, str(width), str(depth), str(seed), arm)
                        if key in done:
                            continue
                        load = os.getloadavg()[0]  # per row: separates quiet from loaded timings
                        t0 = time.perf_counter()
                        st, obj, bnd, cert, size = solve_arm(
                            arm, layers, args.d, form, args.time_limit
                        )
                        wall = time.perf_counter() - t0
                        if cert and obj is not None:
                            certs[arm] = obj
                        nv, nc, nb = size if size else ("", "", "")
                        line = (
                            f"{act}\t{form}\t{width}\t{depth}\t{seed}\t{arm}\t{st}\t{int(cert)}\t"
                            f"{'' if obj is None else f'{obj:.8g}'}\t"
                            f"{'' if bnd is None else f'{float(bnd):.8g}'}\t{wall:.2f}\t"
                            f"{nv}\t{nc}\t{nb}\t{load:.2f}"
                        )
                        emit(line)
                        rows.append(line)
                    if len(certs) >= 2:
                        vals = list(certs.values())
                        compared += 1
                        ref = vals[0]
                        if any(abs(v - ref) > max(1e-6, 1e-4 * max(1.0, abs(ref))) for v in vals):
                            disagree += 1
                            print(
                                f"DISAGREE {act}/{form} w={width} d={depth} s={seed}: {certs}",
                                flush=True,
                            )
    print(
        f"\nINSTANCES {len(rows) // len(ARMS)}; certified-objective cross-checks {compared}, "
        f"disagreements {disagree}",
        flush=True,
    )
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
