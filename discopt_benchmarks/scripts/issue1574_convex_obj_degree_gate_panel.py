#!/usr/bin/env python3
"""#1574 graduation panel -- ``DISCOPT_CONVEX_OBJ_DEGREE_GATE``.

The flag changes how :func:`discopt.solver._objective_is_convex_quadratic` decides
that the objective is a convex quadratic (objective-only polynomial degree bound,
definiteness tested on the Hessian's support). When the gate says yes, every B&B
node bound becomes ``max(LP bound, convex supporting-hyperplane box bound)`` --
bound-changing, so CLAUDE.md §5 applies in full. Per instance, the two arms are
interleaved (order alternated by index, measurement rule 9) and the panel checks:

Gate 1, *cert-clean*:
  (1a) every incumbent is independently feasibility-verified against the pristine
       model -- constraints at ``tol=1e-5``, the declared box, integrality;
  (1b) the certificate invariant: a MINIMIZE bound never above its own incumbent;
  (1c) oracle bracket: no dual bound past the reference optimum
       (``docs/dev/data/cert-optima.json`` for the .nl corpus; for the generated
       DAE / weighted-least-squares models, which have no oracle, the *other*
       arm's verified incumbent stands in -- a bound above any feasible
       objective is false);
  (1d) no certification regression (``gap_certified`` True OFF -> False ON);
  (1e) no objective drift on instances both arms certify;
  (1f) on every instance where the ON gate fires, a fixed-box differential test
       of the bound itself on random sub-boxes of the root box: the convex bound
       must not exceed the box optimum of the objective (computed by L-BFGS-B,
       exact for a convex objective), and must not exceed the objective at any of
       200 sampled points of the box (feasible-point sampling: every feasible
       point of a node lies in its box, so this is a superset test).
       These run on the exact ``(model, evaluator)`` the gate admitted inside the
       solve, which is frequently a REFORMULATED model. The first run of this
       panel checked the user's model instead and reported 8 "violations" on
       nvs06, whose original objective has ``/(x0*x1)**4`` and was never admitted;
       the solver gated the fractional-lifted model ``0.1*x0**2 + 0.1*aux0 +
       0.1*aux2 + 1.2``. That was an instrument defect, retracted.

Gate 2, *net-positive*: over the instances whose gate verdict the flag changes,
certified count, node count (only on instances BOTH arms certify -- a
time-limited node count is throughput, not search), total wall and final
bound, ON vs OFF. Wall alone does not count as a win.

Exit codes: 0 = PASS both gates, 2 = cert-clean but not net-positive,
1 = any violation or zero executed checks.

Usage::

    python -u discopt_benchmarks/scripts/issue1574_convex_obj_degree_gate_panel.py \
        [--time-limit 30] [--out-dir reports] [--instances a,b]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent.parent
_NL_DIR = _REPO / "python" / "tests" / "data" / "minlplib_nl"
_OPTIMA = _REPO / "docs" / "dev" / "data" / "cert-optima.json"
_FLAG = "DISCOPT_CONVEX_OBJ_DEGREE_GATE"

_TOL_REL = 1e-4
_TOL_ABS = 1e-6


# --------------------------------------------------------------------------- models
def _dae_control(nfe: int, ncp: int, target: float, power: int):
    """``integral(u**2) + 10*(x(T)-target)**2`` s.t. ``dx/dt = -x**power + u``."""
    import discopt.modeling as dm
    from discopt.dae import ContinuousSet, DAEBuilder

    m = dm.Model(f"dae_n{nfe}c{ncp}p{power}")
    cs = ContinuousSet("t", bounds=(0, 1), nfe=nfe, ncp=ncp)
    dae = DAEBuilder(m, cs)
    dae.add_state("x", initial=1.0, bounds=(-5, 5))
    dae.add_control("u", bounds=(-2, 2))
    dae.set_ode(lambda t, s, a, c: {"x": -(s["x"] ** power) + c["u"]})
    dae.discretize()
    xv = dae.get_state("x")
    m.minimize(dae.integral(lambda t, s, a, c: c["u"] ** 2) + 10 * (xv[-1, -1] - target) ** 2)
    return m


def _weighted_ls(n: int, seed: int):
    """Weighted least squares with a nonconvex (bilinear) model: ``pred_i = a*b*t_i +
    a*t_i``, ``min sum w_i (pred_i - y_i)**2`` over array variables."""
    import discopt.modeling as dm

    rng = np.random.default_rng(seed)
    t = np.linspace(0.1, 1.0, n)
    y = 1.5 * 0.7 * t + 1.5 * t + 0.05 * rng.standard_normal(n)
    w = rng.uniform(0.5, 3.0, n)
    m = dm.Model(f"wls_n{n}s{seed}")
    a = m.continuous("a", lb=-3, ub=3)
    b = m.continuous("b", lb=-3, ub=3)
    pred = m.continuous("pred", shape=(n,), lb=-20, ub=20)
    for i in range(n):
        m.subject_to(pred[i] == a * b * float(t[i]) + a * float(t[i]))
    m.minimize(sum(float(w[i]) * (pred[i] - float(y[i])) ** 2 for i in range(n)))
    return m


def _array_2y():
    """#1574's hand-built array model with a scaled square ``2*y**2``."""
    import discopt.modeling as dm

    m = dm.Model("arr2y")
    x = m.continuous("x", shape=(3,), lb=-2, ub=2)
    y = m.continuous("y", lb=-3, ub=3)
    m.subject_to(x[0] * x[1] >= 0.5)
    m.subject_to(x[2] * y >= -1)
    m.minimize(sum((x[i] - 0.3) ** 2 for i in range(3)) + 2 * y**2)
    return m


def _issue1569():
    """#1569's repro: array variable, bare squares."""
    import discopt.modeling as dm

    m = dm.Model("issue1569")
    x = m.continuous("x", shape=(3,), lb=-2, ub=2)
    m.subject_to(x[0] * x[1] >= 0.5)
    m.minimize(sum((x[i] - 0.3) ** 2 for i in range(3)))
    return m


GENERATED = {
    "dae_n3c2p3": lambda: _dae_control(3, 2, 0.2, 3),
    "dae_n4c2p3": lambda: _dae_control(4, 2, 0.2, 3),
    "dae_n3c3p3": lambda: _dae_control(3, 3, -0.3, 3),
    "dae_n3c2p2": lambda: _dae_control(3, 2, 0.5, 2),
    "wls_n6s0": lambda: _weighted_ls(6, 0),
    "wls_n10s1": lambda: _weighted_ls(10, 1),
    "wls_n15s2": lambda: _weighted_ls(15, 2),
    "arr2y": _array_2y,
    "issue1569": _issue1569,
}


def _build(name: str):
    if name in GENERATED:
        return GENERATED[name]()
    import discopt.modeling as dm

    return dm.from_nl(str(_NL_DIR / f"{name}.nl"))


# ---------------------------------------------------------------------- one solve
def _solve_once(name: str, flag: str, time_limit: float):
    """One solve with the flag set; records the gate verdict(s). Never swallows an
    exception (measurement rule 7)."""
    from discopt import solver as solver_mod

    verdicts: list[bool] = []
    admitted: list = []
    orig = solver_mod._objective_is_convex_quadratic

    def _wrap(*a, **k):
        r = orig(*a, **k)
        verdicts.append(bool(r))
        if r:
            # The model the gate admitted is the one the solver bounds, which is
            # often a REFORMULATED model (fractional terms lifted to aux vars,
            # nvs06), not the user's. The box checks must test exactly that.
            admitted.append((a[0], a[1]))
        return r

    prev = os.environ.get(_FLAG)
    os.environ[_FLAG] = flag
    solver_mod._objective_is_convex_quadratic = _wrap
    try:
        model = _build(name)
        t0 = time.perf_counter()
        res = model.solve(time_limit=time_limit, deterministic=True)
        wall = time.perf_counter() - t0
    finally:
        solver_mod._objective_is_convex_quadratic = orig
        if prev is None:
            os.environ.pop(_FLAG, None)
        else:
            os.environ[_FLAG] = prev
    return model, res, wall, verdicts, admitted


def _evaluator(model):
    from discopt._tape_nlp_evaluator import build_evaluator

    def _jax():
        from discopt._relax.nlp_evaluator import cached_evaluator

        return cached_evaluator(model)

    return build_evaluator(model, _jax)


def _flat_x(model, res) -> np.ndarray:
    return np.concatenate(
        [np.asarray(res.x[v.name], dtype=float).ravel() for v in model._variables]
    )


def _verified(model, res) -> bool | None:
    """Independent feasibility check of the incumbent against the pristine model."""
    if not res.x:
        return None
    from discopt._relax.primal_heuristics import _check_constraint_feasibility
    from discopt.modeling.core import VarType
    from discopt.solver import _flat_var_box

    x = _flat_x(model, res)
    lb, ub = _flat_var_box(model)
    if np.any(x < lb - 1e-6 * (1 + np.abs(lb))) or np.any(x > ub + 1e-6 * (1 + np.abs(ub))):
        return False
    off = 0
    for v in model._variables:
        if v.var_type in (VarType.BINARY, VarType.INTEGER):
            xi = x[off : off + v.size]
            if np.any(np.abs(xi - np.round(xi)) > 1e-5):
                return False
        off += v.size
    if not model._constraints:
        return True
    return bool(_check_constraint_feasibility(_evaluator(model), x, tol=1e-5))


def _is_minimize(model) -> bool:
    from discopt.modeling.core import ObjectiveSense

    return model._objective is None or model._objective.sense == ObjectiveSense.MINIMIZE


# ------------------------------------------------------- fixed-box bound checks (1f)
def _box_checks(name: str, model, ev, n_boxes: int = 20, n_pts: int = 200, seed: int = 0):
    """Differential bound test + feasible-point sampling of the convex bound on
    random sub-boxes of the root box. Returns (checks, violations, n_tightened)."""
    from discopt.solver import _convex_objective_lower_bound, _flat_var_box
    from scipy.optimize import minimize

    lb, ub = _flat_var_box(model)
    # Open sides: a node with an infinite side gets -inf from the bound, so only
    # finite boxes exercise it; close them around 0 for the sample boxes.
    lb = np.where(np.isfinite(lb) & (lb > -1e6), lb, -10.0)
    ub = np.where(np.isfinite(ub) & (ub < 1e6), ub, 10.0)
    ub = np.maximum(ub, lb)
    rng = np.random.default_rng(seed)
    checks, viol, tightened = 0, [], 0
    for k in range(n_boxes):
        if k == 0:
            blo, bhi = lb.copy(), ub.copy()
        else:
            p, q = rng.uniform(lb, ub), rng.uniform(lb, ub)
            blo, bhi = np.minimum(p, q), np.maximum(p, q)
        cvx = _convex_objective_lower_bound(ev, blo, bhi)
        if not np.isfinite(cvx):
            continue

        def f(x):
            return float(ev.evaluate_objective(np.asarray(x, dtype=float)))

        def g(x):
            return np.asarray(ev.evaluate_gradient(np.asarray(x, dtype=float)), dtype=float)

        best = np.inf
        for x0 in (0.5 * (blo + bhi), rng.uniform(blo, bhi)):
            r = minimize(f, x0, jac=g, method="L-BFGS-B", bounds=list(zip(blo, bhi, strict=True)))
            best = min(best, float(r.fun))
        checks += 1
        slack = 1e-6 * (1.0 + abs(best))
        if cvx > best + slack:
            viol.append(f"{name} box{k}: convex bound {cvx} > box optimum {best}")
        pts = rng.uniform(blo, bhi, size=(n_pts, lb.size))
        for x in pts:
            fx = f(x)
            checks += 1
            if cvx > fx + 1e-6 * (1.0 + abs(fx)):
                viol.append(f"{name} box{k}: convex bound {cvx} > f(sample) {fx}")
                break
        if cvx > -np.inf:
            tightened += 1
    return checks, viol, tightened


# ---------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--out-dir", default=str(_REPO / "reports"))
    ap.add_argument("--instances", default="")
    args = ap.parse_args()

    # CLAUDE.md section 8: prove which code is loaded. Without the flag field the
    # two arms run identical code and the panel compares a solver with itself.
    import discopt
    from discopt.solver_tuning import SolverTuning

    print(f"discopt loaded from {discopt.__file__}", flush=True)
    if not hasattr(SolverTuning(), "convex_obj_degree_gate"):
        raise SystemExit("loaded discopt has no convex_obj_degree_gate; wrong tree")

    optima = json.loads(_OPTIMA.read_text()) if _OPTIMA.exists() else {}
    names = (
        [n.strip() for n in args.instances.split(",") if n.strip()]
        if args.instances
        else sorted(p.stem for p in _NL_DIR.glob("*.nl")) + list(GENERATED)
    )

    rows: list[dict] = []
    checks = 0
    violations: list[str] = []
    box_checks_total = 0

    for i, name in enumerate(names):
        order = ("0", "1") if i % 2 == 0 else ("1", "0")
        arms: dict[str, dict] = {}
        for flag in order:
            model, res, wall, verdicts, admitted = _solve_once(name, flag, args.time_limit)
            if flag == "1":
                admitted_on = admitted
            arms[flag] = {
                "status": res.status,
                "objective": res.objective,
                "bound": res.bound,
                "gap_certified": bool(res.gap_certified),
                "nodes": int(res.node_count or 0),
                "wall": wall,
                "verified": _verified(model, res),
                "minimize": _is_minimize(model),
                "gate": verdicts,
            }
        off, on = arms["0"], arms["1"]
        fired_on = any(on["gate"])
        flipped = fired_on and not any(off["gate"])
        row = {"instance": name, "off": off, "on": on, "flipped": flipped}

        ref = optima.get(name)
        for tag, a, other in (("off", off, on), ("on", on, off)):
            # (1a) independent feasibility of the incumbent.
            if a["verified"] is not None:
                checks += 1
                if a["verified"] is False:
                    violations.append(f"{name}[{tag}]: incumbent failed feasibility verification")
            # (1b) bound vs own incumbent.
            if a["objective"] is not None and a["bound"] is not None:
                checks += 1
                slack = _TOL_REL * max(1.0, abs(a["objective"])) + _TOL_ABS
                if a["minimize"] and a["bound"] > a["objective"] + slack:
                    violations.append(
                        f"{name}[{tag}]: bound {a['bound']} above incumbent {a['objective']}"
                    )
                if not a["minimize"] and a["bound"] < a["objective"] - slack:
                    violations.append(
                        f"{name}[{tag}]: bound {a['bound']} below incumbent {a['objective']}"
                    )
            # (1c) oracle bracket (cross-arm verified incumbent when no oracle).
            cap = ref
            if cap is None and other["verified"] and other["objective"] is not None:
                cap = other["objective"]
            if cap is not None and a["bound"] is not None:
                checks += 1
                slack = _TOL_REL * max(1.0, abs(cap)) + _TOL_ABS
                if a["minimize"] and a["bound"] > cap + slack:
                    violations.append(f"{name}[{tag}]: bound {a['bound']} above reference {cap}")
                if not a["minimize"] and a["bound"] < cap - slack:
                    violations.append(f"{name}[{tag}]: bound {a['bound']} below reference {cap}")
        # (1d) certification regression, (1e) objective drift.
        checks += 1
        if off["gap_certified"] and not on["gap_certified"]:
            violations.append(f"{name}: certification lost with the flag ON")
        if off["gap_certified"] and on["gap_certified"]:
            checks += 1
            drift = abs(on["objective"] - off["objective"])
            if drift > _TOL_REL * max(1.0, abs(off["objective"])) + _TOL_ABS:
                violations.append(
                    f"{name}: objective drift {off['objective']} -> {on['objective']}"
                )
        # (1f) fixed-box differential + sampling where the ON gate fires.
        if fired_on:
            assert admitted_on, f"{name}: gate fired but no admitted model captured"
            bc, bv, bt = 0, [], 0
            for gm, gev in admitted_on:
                c_, v_, t_ = _box_checks(name, gm, gev)
                bc, bv, bt = bc + c_, bv + v_, bt + t_
            checks += bc
            box_checks_total += bc
            violations.extend(bv)
            row["box_checks"] = bc
            row["box_finite_bounds"] = bt
        rows.append(row)

        print(
            f"{name:20s} gate OFF={any(off['gate'])!s:5s} ON={fired_on!s:5s} | "
            f"OFF {str(off['status']):10s} cert={off['gap_certified']!s:5s} "
            f"nodes={off['nodes']:6d} wall={off['wall']:6.2f} bound={off['bound']} | "
            f"ON {str(on['status']):10s} cert={on['gap_certified']!s:5s} "
            f"nodes={on['nodes']:6d} wall={on['wall']:6.2f} bound={on['bound']}",
            flush=True,
        )

    flipped = [r for r in rows if r["flipped"]]

    def _tot(rs, arm, key):
        return sum(r[arm][key] for r in rs)

    def _bdelta(r):
        a, b = r["off"]["bound"], r["on"]["bound"]
        if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
            return 0.0
        tol = _TOL_REL * max(1.0, abs(a)) + _TOL_ABS
        d = (b - a) if r["on"]["minimize"] else (a - b)
        return d if abs(d) > tol else 0.0

    uncert = [r for r in flipped if not (r["off"]["gap_certified"] and r["on"]["gap_certified"])]
    bound_gain = [r["instance"] for r in uncert if _bdelta(r) > 0]
    bound_loss = [r["instance"] for r in uncert if _bdelta(r) < 0]
    # Node counts are search evidence only where BOTH arms finished: on an
    # instance that hits the time limit in both arms the node count measures
    # throughput under load, not search. The 2026-10-02 run of this panel read
    # dae_n4c2p3 (time-limited both arms, OFF 3491 / ON 3421 nodes; the run before
    # it had OFF 3315 / ON 3439 -- the sign flips) as "fewer nodes" and printed a
    # PASS on that alone. Retracted; only finished instances count.
    finished = [r for r in flipped if r["off"]["gap_certified"] and r["on"]["gap_certified"]]
    cert_off = sum(r["off"]["gap_certified"] for r in flipped)
    cert_on = sum(r["on"]["gap_certified"] for r in flipped)
    summary = {
        "instances": len(rows),
        "flipped": [r["instance"] for r in flipped],
        "executed_checks": checks,
        "box_checks": box_checks_total,
        "violations": violations,
        "flipped_cert_off": cert_off,
        "flipped_cert_on": cert_on,
        "flipped_nodes_off": _tot(flipped, "off", "nodes"),
        "flipped_nodes_on": _tot(flipped, "on", "nodes"),
        "finished_nodes_off": _tot(finished, "off", "nodes"),
        "finished_nodes_on": _tot(finished, "on", "nodes"),
        "flipped_wall_off": _tot(flipped, "off", "wall"),
        "flipped_wall_on": _tot(flipped, "on", "wall"),
        "flipped_bound_gain": bound_gain,
        "flipped_bound_loss": bound_loss,
        "all_wall_off": _tot(rows, "off", "wall"),
        "all_wall_on": _tot(rows, "on", "wall"),
        "all_cert_off": sum(r["off"]["gap_certified"] for r in rows),
        "all_cert_on": sum(r["on"]["gap_certified"] for r in rows),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "issue1574_convex_obj_degree_gate_panel.json"
    out.write_text(json.dumps({"time_limit": args.time_limit, **summary, "rows": rows}, indent=2))

    print()
    for k, v in summary.items():
        if k != "violations":
            print(f"{k:22s}: {v}")
    print(f"violations            : {len(violations)}")
    for v in violations:
        print("  !", v)
    print(f"report                : {out}")

    if checks == 0 or not flipped:
        print("\nVERDICT: FAIL - the panel measured nothing (no checks / no flipped instance).")
        return 1
    if violations:
        print("\nVERDICT: FAIL - gate 1 (cert-clean) violated.")
        return 1
    # Wall alone is not evidence (sub-second, load-sensitive): helpful means more
    # certificates, fewer nodes, or a tighter final bound where neither arm
    # certifies -- and no instance certifies with fewer than with the flag OFF.
    better = (
        cert_on > cert_off
        or summary["finished_nodes_on"] < summary["finished_nodes_off"]
        or len(bound_gain) > 0
    )
    worse = cert_on < cert_off or len(bound_loss) > 0
    if better and not worse:
        print("\nVERDICT: PASS candidate - cert-clean; see the net-positive columns above.")
        return 0
    print("\nVERDICT: cert-clean but NOT net-positive.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
