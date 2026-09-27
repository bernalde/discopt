#!/usr/bin/env python3
"""Differential panel for ``DISCOPT_FBBT_EVEN_POW_HOLE`` (CLAUDE.md §5).

The flag makes FBBT's backward step through an even power ``u^n >= r`` keep the
hole ``|u| < r^(1/n)`` when the base box reaches only one side of it, instead of
relaxing to the hull. On a minimum distance constraint ``Σ (y_i - z_i)^2 >= δ^2``
this is the per-coordinate rule of Hojny & Liberti (ORL 2027, Prop. 1), which
SCIP 11 ships as ``prop_distance``.

The flag is read ONCE per process (Rust ``OnceLock``), so each arm runs in its own
subprocess; setting ``os.environ`` in-process would silently measure one arm twice.

Two populations:

* ``corpus`` -- every ``python/tests/data/minlplib_nl/*.nl`` (the in-repo corpus),
  oracle ``docs/dev/data/cert-optima.json``.
* ``disp`` -- point dispersion in the unit square, ``max t`` s.t.
  ``(x_i - x_j)^2 + (y_i - y_j)^2 >= t`` for all pairs: the pure min-distance
  class. Known optima of the squared distance: n=2: 2, n=3: 8 - 4√3, n=4: 1,
  n=5: 1/2, n=6: 13/36 (Schaer; Graham & Lubachevsky).

Per instance and arm it records the solve (status, objective, bound,
gap_certified, nodes, wall) and the root ``fbbt_box`` -- the latter is the
*firing* measurement: an instance where the ON root box equals the OFF root box
is one the rule did not touch at the root.

Gate 1 (cert-clean): bound never past the incumbent or the oracle, no
certification regression, no objective drift, root ON box ⊆ OFF box, no
false-infeasible. Gate 2 (net-positive) is reported, not decided here.

Usage::

    python -u discopt_benchmarks/scripts/even_pow_hole_panel.py \
        [--set corpus,disp] [--time-limit 30] [--out reports/even_pow_hole_panel.json]

Against a MINLPLib snapshot (the graduation test: instances with minimum distance
constraints, e.g. the Hojny & Liberti set)::

    python -u discopt_benchmarks/scripts/even_pow_hole_panel.py --set corpus \
        --nl-dir <snapshot>/minlplib/nl --solu <snapshot>/minlplib.solu \
        --instances <comma-separated names> --time-limit 60

Exit codes: 0 = cert-clean with executed checks, 1 = violation or zero checks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
_NL_DIR = _REPO / "python" / "tests" / "data" / "minlplib_nl"
_OPTIMA = _REPO / "docs" / "dev" / "data" / "cert-optima.json"
_FLAG = "DISCOPT_FBBT_EVEN_POW_HOLE"
_TOL_REL = 1e-4
_TOL_ABS = 1e-6

DISP_OPT = {2: 2.0, 3: 8.0 - 4.0 * math.sqrt(3.0), 4: 1.0, 5: 0.5, 6: 13.0 / 36.0}


def _dispersion(n: int, arrays: bool = False):
    """``disp<n>`` builds the points from scalars, ``adisp<n>`` from shape-(n,)
    arrays. Both forms are kept: before #1513 the array form never ran in-tree
    FBBT (disp4: 0 in-tree calls, 2639 nodes vs 255 / 311 as scalars), so the two
    must agree now that it does."""
    import discopt.modeling as dm

    m = dm.Model(f"{'a' if arrays else ''}disp{n}")
    if arrays:
        x = m.continuous("x", shape=(n,), lb=0.0, ub=1.0)
        y = m.continuous("y", shape=(n,), lb=0.0, ub=1.0)
    else:
        x = [m.continuous(f"x{i}", lb=0.0, ub=1.0) for i in range(n)]
        y = [m.continuous(f"y{i}", lb=0.0, ub=1.0) for i in range(n)]
    t = m.continuous("t", lb=0.0, ub=2.0)
    for i in range(n):
        for j in range(i + 1, n):
            m.subject_to((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 >= t)
    m.maximize(t)
    return m


def _load(name: str):
    import discopt.modeling as dm

    # Exact match: the corpus has ``dispatch``, which a prefix test would take.
    hit = re.fullmatch(r"(a?)disp(\d+)", name)
    if hit:
        return _dispersion(int(hit.group(2)), arrays=bool(hit.group(1)))
    return dm.from_nl(str(_NL_DIR / f"{name}.nl"))  # rebound by --nl-dir


def _worker(name: str, time_limit: float) -> None:
    """Runs in a subprocess with the flag already in its environment."""
    import discopt
    from discopt.modeling.core import ObjectiveSense
    from discopt.tightening import fbbt_box

    # CLAUDE.md §8: prove which code is loaded.
    so = Path(discopt.__file__).parent / "_rust.abi3.so"
    assert _FLAG.encode() in so.read_bytes(), f"{so} does not carry {_FLAG}"

    model = _load(name)
    box = fbbt_box(model)
    model = _load(name)
    t0 = time.perf_counter()
    res = model.solve(time_limit=time_limit, deterministic=True)
    wall = time.perf_counter() - t0
    minimize = model._objective is None or model._objective.sense == ObjectiveSense.MINIMIZE
    print(
        "RESULT "
        + json.dumps(
            {
                "status": str(res.status),
                "objective": res.objective,
                "bound": res.bound,
                "gap_certified": bool(res.gap_certified),
                "nodes": int(res.node_count or 0),
                "wall": wall,
                "minimize": minimize,
                "root_infeasible": bool(box.infeasible),
                "root_lb": [float(v) for v in box.lb],
                "root_ub": [float(v) for v in box.ub],
            }
        ),
        flush=True,
    )


def _run_arm(name: str, flag: str, time_limit: float) -> dict:
    env = dict(os.environ, **{_FLAG: flag})
    proc = subprocess.run(
        [
            sys.executable,
            "-u",
            __file__,
            "--worker",
            name,
            "--time-limit",
            str(time_limit),
            "--nl-dir",
            str(_NL_DIR),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=time_limit * 4 + 300,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    # CLAUDE.md §7: never swallow -- a broken arm crashes the panel.
    raise RuntimeError(f"{name}[{_FLAG}={flag}] produced no result:\n{proc.stderr[-4000:]}")


def main() -> int:
    global _NL_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default=None)
    ap.add_argument("--set", default="corpus,disp")
    ap.add_argument("--instances", default="")
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--out", default=str(_REPO / "reports" / "even_pow_hole_panel.json"))
    ap.add_argument(
        "--nl-dir",
        default=str(_NL_DIR),
        help="directory of .nl files for the corpus set (e.g. a MINLPLib snapshot's minlplib/nl)",
    )
    ap.add_argument(
        "--solu",
        default="",
        help="MINLPLib .solu file; its =opt= values extend the oracle (cert-optima.json wins)",
    )
    args = ap.parse_args()
    _NL_DIR = Path(args.nl_dir)
    if args.worker:
        _worker(args.worker, args.time_limit)
        return 0

    optima: dict[str, float] = {}
    if args.solu:
        for line in Path(args.solu).read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "=opt=":
                optima[parts[1]] = float(parts[2])
    if _OPTIMA.exists():
        optima.update(json.loads(_OPTIMA.read_text()))
    optima.update({f"{p}disp{n}": v for n, v in DISP_OPT.items() for p in ("", "a")})
    sets = set(args.set.split(","))
    if args.instances:
        names = [s.strip() for s in args.instances.split(",") if s.strip()]
    else:
        names = []
        if "disp" in sets:
            names += [f"disp{n}" for n in sorted(DISP_OPT)]
            names += [f"adisp{n}" for n in sorted(DISP_OPT)]
        if "corpus" in sets:
            names += sorted(p.stem for p in _NL_DIR.glob("*.nl"))

    rows, violations, checks = [], [], 0
    for i, name in enumerate(names):
        order = ("0", "1") if i % 2 == 0 else ("1", "0")  # interleave (§9)
        arms = {flag: _run_arm(name, flag, args.time_limit) for flag in order}
        off, on = arms["0"], arms["1"]

        # Firing + root soundness: ON root box must be inside the OFF root box.
        fired = 0
        if on["root_infeasible"] and not off["root_infeasible"]:
            fired += 1
        elif not on["root_infeasible"] and not off["root_infeasible"]:
            for lo0, hi0, lo1, hi1 in zip(
                off["root_lb"], off["root_ub"], on["root_lb"], on["root_ub"], strict=True
            ):
                checks += 1
                if lo1 < lo0 - 1e-9 or hi1 > hi0 + 1e-9:
                    violations.append(f"{name}: ON root box looser than OFF")
                if lo1 > lo0 + 1e-9 or hi1 < hi0 - 1e-9:
                    fired += 1

        ref = optima.get(name)
        for tag, a in (("off", off), ("on", on)):
            if a["objective"] is not None and a["bound"] is not None:
                checks += 1
                slack = _TOL_REL * max(1.0, abs(a["objective"])) + _TOL_ABS
                bad = (
                    a["bound"] > a["objective"] + slack
                    if a["minimize"]
                    else a["bound"] < a["objective"] - slack
                )
                if bad:
                    violations.append(f"{name}[{tag}]: bound {a['bound']} vs obj {a['objective']}")
            if ref is not None and a["bound"] is not None:
                checks += 1
                slack = _TOL_REL * max(1.0, abs(ref)) + _TOL_ABS
                bad = a["bound"] > ref + slack if a["minimize"] else a["bound"] < ref - slack
                if bad:
                    violations.append(f"{name}[{tag}]: bound {a['bound']} past optimum {ref}")
            if ref is not None and "infeasible" in a["status"].lower():
                violations.append(f"{name}[{tag}]: declared infeasible, oracle {ref}")
        checks += 1
        if off["gap_certified"] and not on["gap_certified"]:
            violations.append(f"{name}: certification lost with the flag ON")
        both = off["objective"] is not None and on["objective"] is not None
        if both and off["gap_certified"] and on["gap_certified"]:
            checks += 1
            drift = abs(on["objective"] - off["objective"])
            if drift > _TOL_REL * max(1.0, abs(off["objective"])) + _TOL_ABS:
                violations.append(
                    f"{name}: objective drift {off['objective']} -> {on['objective']}"
                )

        row = {"instance": name, "root_bounds_tightened": fired, "off": off, "on": on}
        for a in (off, on):
            a.pop("root_lb")
            a.pop("root_ub")
        rows.append(row)
        print(
            f"{name:22s} fired={fired:3d} | OFF cert={off['gap_certified']!s:5s} "
            f"nodes={off['nodes']:6d} wall={off['wall']:7.2f} obj={off['objective']} "
            f"| ON cert={on['gap_certified']!s:5s} nodes={on['nodes']:6d} "
            f"wall={on['wall']:7.2f} obj={on['objective']}",
            flush=True,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "flag": _FLAG,
                "time_limit": args.time_limit,
                "executed_checks": checks,
                "violations": violations,
                "rows": rows,
            },
            indent=2,
        )
    )
    fired_names = [r["instance"] for r in rows if r["root_bounds_tightened"]]
    print(f"\ninstances       : {len(rows)}")
    print(f"executed checks : {checks}")
    print(f"root firing     : {len(fired_names)} {fired_names}")
    print(f"violations      : {len(violations)}")
    for v in violations:
        print("  !", v)
    print(f"report          : {out}")
    if checks == 0:
        print("VERDICT: FAIL - zero executed checks (the panel measured nothing).")
        return 1
    if violations:
        print("VERDICT: FAIL - not cert-clean.")
        return 1
    print("VERDICT: cert-clean (net-positive is judged from the rows, not here).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
