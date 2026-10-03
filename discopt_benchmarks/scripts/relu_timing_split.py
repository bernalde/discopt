"""ReLU big-M: repeated, interleaved timing of discopt vs OMLT+HiGHS, with a split
of where the wall time goes (model build / framework overhead / inside HiGHS).

Both arms reach HiGHS through ``highspy``: discopt's LP/MILP route
(``solvers/lp_milp_highs.py``) and Pyomo's ``SolverFactory("highs")``. Every
``highspy.Highs.run`` call is timed and summed, so

    overhead = solve wall - HiGHS time        (translation, presolve, certificate)

is measured, not inferred. Each (instance, arm, rep) runs in a FRESH process, and
the arm order alternates per rep (CLAUDE.md §9: interleaved, with a spread).

    python -u relu_timing_split.py --reps 5 --out timing.tsv
    python -u relu_timing_split.py --one <width>:<depth>:<seed> --arm discopt --tl 30

Prints the executed-solve count and exits non-zero when it is zero (§6).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

INSTANCES = [
    # certified: compare wall
    (100, 1, 0),
    (100, 1, 1),
    (10, 3, 0),
    (10, 3, 1),
    (25, 2, 0),
    (25, 2, 1),
    # time-limited: compare how much of the budget reaches HiGHS, and the bound
    (100, 2, 0),
    (50, 3, 0),
]


def _timed_highs():
    import highspy

    acc = {"t": 0.0, "n": 0}
    orig = highspy.Highs.run

    def run(self, *a, **k):
        t0 = time.perf_counter()
        try:
            return orig(self, *a, **k)
        finally:
            acc["t"] += time.perf_counter() - t0
            acc["n"] += 1

    highspy.Highs.run = run
    return acc


def _solve(arm, layers, build_discopt, build_omlt, tl):
    if arm == "discopt":
        m, _ = build_discopt(layers, 2, "bigm")
        m.solve(time_limit=tl)
    else:
        import pyomo.environ as pyo

        om = build_omlt(layers, 2, "bigm")
        pyo.SolverFactory("highs").solve(om, timelimit=tl, load_solutions=False)


def one(spec: str, arm: str, tl: float) -> dict:
    sys.path.insert(0, HERE)
    import discopt
    from omlt_scaling_panel import build_discopt, build_omlt, make_net

    w, d, s = (int(v) for v in spec.split(":"))
    acc = _timed_highs()
    # Untimed warm-up on a tiny net in this same arm: imports and first-call setup
    # (Pyomo/OMLT import ~4.7 s, discopt's lazy first-solve setup ~1.4 s, measured)
    # are one-off per process, so the timed solve below is steady state.
    _solve(arm, make_net(2, 4, 1, "relu", 99), build_discopt, build_omlt, 10.0)
    acc["t"], acc["n"] = 0.0, 0
    layers = make_net(2, w, d, "relu", s)
    t0 = time.perf_counter()
    if arm == "discopt":
        m, _ = build_discopt(layers, 2, "bigm")
        t1 = time.perf_counter()
        r = m.solve(time_limit=tl)
        t2 = time.perf_counter()
        status, cert, obj, bound = r.status, bool(r.gap_certified), r.objective, r.bound
    else:
        import pyomo.environ as pyo

        om = build_omlt(layers, 2, "bigm")
        t1 = time.perf_counter()
        res = pyo.SolverFactory("highs").solve(om, timelimit=tl, load_solutions=False)
        t2 = time.perf_counter()
        status = str(res.solver.termination_condition)
        cert = status == "optimal"
        ub, lb = res.problem.upper_bound, res.problem.lower_bound
        obj = None if ub is None or abs(ub) == float("inf") else float(ub)
        bound = None if lb is None else float(lb)
    if acc["n"] == 0:
        raise RuntimeError(
            f"{arm} {spec}: highspy.Highs.run was never called -- probe measured nothing"
        )
    return {
        "spec": spec,
        "arm": arm,
        "build": t1 - t0,
        "solve": t2 - t1,
        "highs": acc["t"],
        "highs_calls": acc["n"],
        "status": status,
        "cert": cert,
        "obj": obj,
        "bound": bound,
        "discopt": discopt.__file__,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--one")
    ap.add_argument("--arm")
    ap.add_argument("--tl", type=float, default=30.0)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="timing.tsv")
    a = ap.parse_args()
    if a.one:
        print(json.dumps(one(a.one, a.arm, a.tl)))
        return 0

    print(f"load {os.getloadavg()}", flush=True)
    rows, n = [], 0
    cols = [
        "spec",
        "arm",
        "rep",
        "build",
        "solve",
        "highs",
        "highs_calls",
        "status",
        "cert",
        "obj",
        "bound",
        "load",
    ]
    with open(a.out, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for w, d, s in INSTANCES:
            spec = f"{w}:{d}:{s}"
            reps = a.reps if d == 1 or (w, d) in ((10, 3), (25, 2)) else max(2, a.reps // 2)
            for rep in range(reps):
                arms = ("discopt", "omlt+highs") if rep % 2 == 0 else ("omlt+highs", "discopt")
                for arm in arms:
                    p = subprocess.run(
                        [
                            sys.executable,
                            "-u",
                            __file__,
                            "--one",
                            spec,
                            "--arm",
                            arm,
                            "--tl",
                            str(a.tl),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=a.tl * 4 + 120,
                    )
                    if p.returncode != 0:
                        raise RuntimeError(f"{spec} {arm}: child failed\n{p.stderr[-2000:]}")
                    r = json.loads(p.stdout.strip().splitlines()[-1])
                    r["rep"], r["load"] = rep, os.getloadavg()[0]
                    rows.append(r)
                    n += 1
                    fh.write("\t".join(str(r[c]) for c in cols) + "\n")
                    fh.flush()
                    print(
                        f"{spec:9s} {arm:11s} rep{rep} build {r['build']:.3f} "
                        f"solve {r['solve']:.3f} "
                        f"highs {r['highs']:.3f} ({r['highs_calls']} runs) {r['status']} "
                        f"bound={r['bound']} load={r['load']:.2f}",
                        flush=True,
                    )
    print("\nspec      arm         n  total med (sd)      build med  overhead med  highs med")
    for w, d, s in INSTANCES:
        spec = f"{w}:{d}:{s}"
        for arm in ("discopt", "omlt+highs"):
            sel = [r for r in rows if r["spec"] == spec and r["arm"] == arm]
            tot = [r["build"] + r["solve"] for r in sel]
            ovh = [r["solve"] - r["highs"] for r in sel]
            sd = st.stdev(tot) if len(tot) > 1 else float("nan")
            print(
                f"{spec:9s} {arm:11s} {len(sel)}  {st.median(tot):7.3f} ({sd:.3f})  "
                f"{st.median(r['build'] for r in sel):9.3f}  {st.median(ovh):12.3f}  "
                f"{st.median(r['highs'] for r in sel):9.3f}"
            )
    print(f"\nexecuted solves {n}")
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
