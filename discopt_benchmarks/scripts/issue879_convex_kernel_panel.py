#!/usr/bin/env python
"""Issue #879 panel: the convex kernel over every in-repo instance it routes.

This is the measurement the #879 fix rests on, and the check whose absence let a
false certificate ship: a routed instance's CERTIFIED objective is compared
against a known optimum, not only against exactness/convexity of its rows.

Per run it reports status / incumbent / dual bound / node count, and verifies the
certificate invariant in the instance's own sense (for a minimization the dual
bound is a LOWER bound, so `bound > optimum` is the unsound side).

Usage::

    python discopt_benchmarks/scripts/issue879_convex_kernel_panel.py
    python discopt_benchmarks/scripts/issue879_convex_kernel_panel.py --ablate

``--ablate`` additionally re-runs `clay0303hfsg` with the dominated-column bound
off, which is one of the three guards the instance needs to certify (the other two
— the per-node relaxation size cap and the cold retry of a broken warm solve — are
unconditional and are ablated by editing `convex_kernel.rs`).

Exits non-zero if any certificate is unsound, if an instance that certified before
stops certifying, or if no run executed at all (a panel that measures nothing must
never read as a pass).
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("DISCOPT_CONVEX_KERNEL", "1")

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "python" / "tests"))

import discopt.modeling as dm  # noqa: E402
from _optima import known_optimum  # noqa: E402
from discopt.solvers._convex_kernel import build_convex_spec, solve_convex_tree  # noqa: E402

_NL = _ROOT / "python" / "tests" / "data" / "minlplib_nl"
_NL2 = _ROOT / "python" / "tests" / "data" / "minlplib"

# name -> (path, optimum or None, sense_max, must_certify)
PANEL = {
    "syn05m": (_NL2 / "syn05m.nl", 837.7324009, True, True),
    "syn05hfsg": (_NL / "syn05hfsg.nl", 837.7324009, True, True),
    # Routed but not certifiable within the budget; kept so a regression that makes
    # it report `optimal` on an unexplored tree is caught.
    "cvxnonsep_psig40r": (_NL / "cvxnonsep_psig40r.nl", None, True, False),
    # The #879 instance: declined before, certifies now.
    "clay0303hfsg": (_NL / "clay0303hfsg.nl", known_optimum("clay0303hfsg"), False, True),
}


def run(name, path, opt, sense_max, must_certify, time_limit_s, **cfg):
    """One panel run. Returns (ok, executed_checks)."""
    spec = build_convex_spec(dm.from_nl(str(path)))
    if spec is None:
        print(f"{name:20s} DECLINED")
        return not must_certify, 0
    r = solve_convex_tree(spec, time_limit_s=time_limit_s, **cfg)
    inc, bound, status = r["incumbent"], r["bound"], r["status"]
    checks = 0
    problems = []
    if opt is not None:
        tol = 1e-4 * max(1.0, abs(opt))
        checks += 1
        # Sound side: dual bound is an UPPER bound for a max, a LOWER bound for a min.
        if sense_max and bound < opt - tol:
            problems.append(f"UNSOUND dual bound {bound} < optimum {opt}")
        if not sense_max and bound > opt + tol:
            problems.append(f"UNSOUND dual bound {bound} > optimum {opt}")
        if status == "optimal":
            checks += 1
            if inc is None or abs(inc - opt) > tol:
                problems.append(f"certified objective {inc} != optimum {opt}")
    if must_certify and status != "optimal":
        problems.append(f"expected `optimal`, got `{status}`")
    print(
        f"{name:20s} {status:10s} inc={inc} bound={bound} nodes={r['node_count']}"
        + ("  <<< " + "; ".join(problems) if problems else "")
    )
    return not problems, checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--ablate", action="store_true")
    args = ap.parse_args()

    ok, executed, runs = True, 0, 0
    print("=== convex kernel, shipped configuration ===")
    for name, (path, opt, smax, must) in PANEL.items():
        good, n = run(name, path, opt, smax, must, args.time_limit)
        ok &= good
        executed += n
        runs += 1

    if args.ablate:
        print("\n=== ablation: DISCOPT_CVX_DOMINATED_COLS off ===")
        path, opt, smax, _must = PANEL["clay0303hfsg"]
        # `must_certify=False`: the point of the ablation is that it does NOT
        # certify, but its bound must still be sound.
        good, n = run("clay0303hfsg", path, opt, smax, False, args.time_limit, dominated_cols=False)
        ok &= good
        executed += n
        runs += 1

    print(f"\nruns={runs} certificate_checks={executed}")
    if runs == 0 or executed == 0:
        print("PANEL MEASURED NOTHING", file=sys.stderr)
        return 2
    print("PANEL OK" if ok else "PANEL FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
