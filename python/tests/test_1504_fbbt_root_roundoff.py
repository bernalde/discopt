"""#1504: FBBT round-off through a root must never cut a feasible integer.

Measured on ``6596922`` before the fix, with ``b**3 - 2*x0**3 == 5``, ``x0 in [-2, 0]``
and ``b`` binary:

1. backward through ``x0**3 in [-2.5, -2]`` gave ``x0 >= -(2.5)**(1/3)`` rounded to
   nearest, ``-1.3572088082974532`` -- which lies ABOVE the true root: in exact
   rational arithmetic ``5 + 2*x0_lo**3 = +1.368e-15 > 0``;
2. the next forward sweep therefore gave ``b**3 >= 1.776e-15``;
3. the cube-root inverse turned that into ``b >= 1.2110908904786693e-05``, above the
   1e-6 integrality snap tolerance, so the Rust FBBT fixed ``b = 1`` and the solver
   certified 1 (true optimum 0). With ``b`` continuous the certified optimum was
   exactly that ``1.21109e-05``.

The fix (``crates/discopt-core/src/presolve/directed.rs``) makes every FBBT endpoint
an outward-rounded enclosure -- directed ``+ - * /`` via error-free transforms,
directed integer powers, and roots VERIFIED against a directed power -- so a residual
that is ``<= 0`` in exact arithmetic can never reach a root as a positive phantom.

These tests pin the class: the two issue repros, the continuous-``b`` variant, a
squares variant, and a seeded sweep over ``a*y**p + c*x0**q == R`` against a
brute-force oracle (enumerate the integer, solve the 1-D root for ``x0`` exactly).
"""

from __future__ import annotations

import math
import random

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._rust import model_to_repr

TOL = 1e-6


def _binary_cube_model(extra: bool, continuous_b: bool = False) -> dm.Model:
    m = dm.Model("r1504")
    x0 = m.continuous("x0", lb=-2, ub=0)
    x1 = m.continuous("x1", lb=0, ub=10)
    b = m.continuous("b", lb=0, ub=1) if continuous_b else m.binary("b")
    m.minimize(b)
    m.subject_to(b**3 - 2 * x0**3 == 5)
    if extra:
        m.subject_to(x1 * x0 * x0 <= 100)  # max lhs = 40: never tight
    return m


# ── FBBT level: the propagator itself ─────────────────────────────────────────


@pytest.mark.parametrize("extra", [False, True])
def test_rust_fbbt_keeps_b_zero(extra):
    lb, ub = model_to_repr(_binary_cube_model(extra)).fbbt(max_iter=20, tol=1e-9)
    # Variable order: x0, x1, b.
    assert lb[2] <= 0.0, f"FBBT lifted binary b to lb={lb[2]!r} (feasible b = 0 cut)"
    root = -(2.5 ** (1.0 / 3.0))
    # The true root may be one ulp below the rounded one; the box must contain it.
    assert lb[0] <= np.nextafter(root, -np.inf) and root <= ub[0], (lb[0], ub[0])


def test_rust_fbbt_continuous_b_lower_bound_is_not_positive():
    lb, _ub = model_to_repr(_binary_cube_model(True, continuous_b=True)).fbbt(max_iter=20, tol=1e-9)
    assert lb[2] <= 0.0, f"continuous b lb = {lb[2]!r}; the true infimum is 0"


# ── Solve level: the issue's repros ───────────────────────────────────────────


@pytest.mark.parametrize("extra", [False, True])
def test_repro1_binary_cube(extra):
    r = _binary_cube_model(extra).solve(time_limit=30)
    assert r.status == "optimal"
    assert abs(r.objective) <= TOL, r.objective
    assert r.bound <= TOL, r.bound


def test_repro1_continuous_b_bound():
    r = _binary_cube_model(True, continuous_b=True).solve(time_limit=30)
    assert r.status == "optimal"
    assert r.objective <= TOL, r.objective
    assert r.bound <= TOL, r.bound


def test_repro2_false_infeasible():
    m = dm.Model("r1504b")
    x0 = m.continuous("x0", lb=-2, ub=2)
    y = m.integer("y", lb=0, ub=3)
    m.minimize(y)
    m.subject_to(0.5 * y**3 + x0**3 == 5)
    m.subject_to(x0**2 * y <= 0.5)
    r = m.solve(time_limit=30)
    # y = 0, x0 = 5**(1/3) is feasible (row 2 = 0).
    assert r.status == "optimal", r.status
    assert abs(r.objective) <= TOL
    assert r.bound <= TOL


@pytest.mark.parametrize(
    "a, c, rhs, x_lo",
    [
        # y = 0: x0**2 = 0.5, x0 = +-sqrt(0.5).
        (0.005, -2.0, -1.0, -2.0),
        # y = 0: x0 = sqrt(3). Certified 1 before the fix (found by the sweep below).
        (1.0, 2.0, 6.0, 0.0),
    ],
)
def test_squares_variant(a, c, rhs, x_lo):
    m = dm.Model("r1504c")
    x0 = m.continuous("x0", lb=x_lo, ub=2)
    x1 = m.continuous("x1", lb=0, ub=10)
    y = m.integer("y", lb=0, ub=3)
    m.minimize(y)
    m.subject_to(a * y**3 + c * x0**2 == rhs)
    m.subject_to(x1 * x0 * x0 <= 100)
    r = m.solve(time_limit=30)
    assert r.status == "optimal", r.status
    assert abs(r.objective) <= TOL
    assert r.bound <= TOL


# ── Seeded sweep against a brute-force oracle ────────────────────────────────


def _real_roots(t: float, q: int) -> list[float]:
    """Real solutions of ``x**q == t``."""
    if q % 2 == 1:
        return [math.copysign(abs(t) ** (1.0 / q), t)]
    if t < 0:
        return []
    r = t ** (1.0 / q)
    return [r, -r] if r > 0 else [0.0]


def _oracle(spec):
    """Optimal y by enumeration, or None if infeasible; 'ambiguous' near a boundary."""
    feas = []
    for yv in range(spec["y_lo"], spec["y_hi"] + 1):
        t = (spec["R"] - spec["a"] * yv ** spec["p"]) / spec["c"]
        ok = False
        for x in _real_roots(t, spec["q"]):
            margin = min(x - spec["x_lo"], spec["x_hi"] - x)
            if abs(margin) < 1e-5:
                return "ambiguous"
            if margin < 0:
                continue
            if spec["row"] == "coupled":
                slack = spec["K"] - x * x * yv
                if abs(slack) < 1e-5:
                    return "ambiguous"
                if slack < 0:
                    continue
            ok = True
        if ok:
            feas.append(yv)
    if not feas:
        return None
    return min(feas) if spec["sense"] == "min" else max(feas)


def _build(spec) -> dm.Model:
    m = dm.Model("sweep1504")
    x0 = m.continuous("x0", lb=spec["x_lo"], ub=spec["x_hi"])
    x1 = m.continuous("x1", lb=0, ub=10)
    y = m.integer("y", lb=spec["y_lo"], ub=spec["y_hi"])
    if spec["sense"] == "min":
        m.minimize(y)
    else:
        m.maximize(y)
    m.subject_to(spec["a"] * y ** spec["p"] + spec["c"] * x0 ** spec["q"] == spec["R"])
    if spec["row"] == "coupled":
        m.subject_to(x0**2 * y <= spec["K"])
    else:
        m.subject_to(x1 * x0 * x0 <= 100)  # never tight
    return m


def _sweep_specs():
    specs = []
    for R in range(2, 10):
        for p in (2, 3):
            for q in (2, 3):
                rng = random.Random(1504 * 1000 + R * 100 + p * 10 + q)
                spec = {
                    "R": float(R),
                    "p": p,
                    "q": q,
                    "a": rng.choice([0.5, 0.25, 0.005, 1.0]),
                    # c > 0 so y = 0 needs x0**q = R/c with R/c within the box.
                    "c": rng.choice([1.0, 2.0]),
                    "x_lo": rng.choice([-2.0, 0.0]) if q == 2 else -2.0,
                    "x_hi": 2.0,
                    "y_lo": 0,
                    "y_hi": 3,
                    "sense": rng.choice(["min", "min", "max"]),
                    "row": rng.choice(["coupled", "loose"]),
                    "K": 0.5,
                }
                specs.append(spec)
    return specs


def test_seeded_sweep_matches_brute_force():
    checked = 0
    failures = []
    for spec in _sweep_specs():
        want = _oracle(spec)
        if want == "ambiguous":
            continue
        r = _build(spec).solve(time_limit=30)
        tag = {k: spec[k] for k in ("R", "p", "q", "a", "c", "x_lo", "sense", "row")}
        if want is None:
            if r.status == "optimal":
                failures.append((tag, "oracle infeasible", r.status, r.objective, r.bound))
        else:
            ok = r.status == "optimal" and abs(r.objective - want) <= TOL
            if spec["sense"] == "min":
                ok = ok and r.bound <= want + TOL
            else:
                ok = ok and r.bound >= want - TOL
            if not ok:
                failures.append((tag, want, r.status, r.objective, r.bound))
        checked += 1
    # Anti-vacuity (CLAUDE.md section 6): a sweep that skipped everything passes.
    assert checked >= 20, f"only {checked} sweep instances were checked"
    assert not failures, "\n".join(map(str, failures))
