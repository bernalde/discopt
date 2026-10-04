"""#1617 C-12b: the GDP hull of a *convex* nonlinear disjunct was a nonconvex row.

The hull writes a nonlinear disjunct row ``g(x) <= 0`` as Furman-Sawaya-Grossmann's
eps-perspective ``yhat * g(v / yhat) <= 0`` with ``yhat = (1 - eps) * y + eps``.
That row is convex whenever ``g`` is (a perspective of a convex function composed
with an affine map), but the convexity classifier called the product
``E * yhat`` UNKNOWN because ``v / yhat`` is not itself convex. The row then went
to McCormick, the model lost its convex-MINLP route, and on the process-synthesis
witness below the hull root bound was **-1898** against an exact continuous hull
relaxation of **-503** and an optimum of **-482.18** -- the hull, whose whole
point is a tighter relaxation than big-M, reported a looser root than big-M's -571.

The fix is the perspective-composite rule in ``convexity/rules.py``, gated by
``DISCOPT_PERSPECTIVE_COMPOSITE`` (CLAUDE.md section 5). These tests pin its
classification, its root bound, a differential bound panel on fixed boxes, and the
validity of every tangent (OA) cut it licenses at sampled feasible points.
"""

from __future__ import annotations

import discopt.modeling as dm
import discopt.transformations as dt
import numpy as np
import pytest
from discopt._relax.convexity.rules import (
    classify_expr,
    classify_model,
    perspective_composite_enabled,
)

FLAG = "DISCOPT_PERSPECTIVE_COMPOSITE"

# Process synthesis (Turkay & Grossmann style): pick one of three reactors, whose
# yield is concave in feed (P <= a*log(1 + F/c), a convex row), and one of two
# separators; minimise negative profit.
_R = [(40.0, 10.0, 60.0, 1.5), (55.0, 20.0, 110.0, 2.0), (70.0, 40.0, 180.0, 2.6)]
_SP = [(0.9, 50.0, 1.0), (0.97, 90.0, 1.4)]
SYNTH_OPT = -482.1779  # big-M and hull (flag OFF) both certify this
SYNTH_HULL_RELAXATION = -502.99  # exact continuous hull relaxation (convex NLP)


def build_synthesis_nl(f_lb=0.0, f_ub=100.0, p_ub=200.0):
    m = dm.Model("synthesis_nl")
    F = m.continuous("F", lb=f_lb, ub=f_ub)
    P = m.continuous("P", lb=0, ub=p_ub)
    S = m.continuous("S", lb=0, ub=200)
    CR = m.continuous("CR", lb=0, ub=600)
    CS = m.continuous("CS", lb=0, ub=600)
    m.either_or(
        [[P <= a * dm.log(1 + F / c), CR >= f + v * F] for (a, c, f, v) in _R], name="reactor"
    )
    m.either_or([[S <= b * P, CS >= f + v * P] for (b, f, v) in _SP], name="separator")
    m.minimize(-(12 * S - 2 * F - CR - CS))
    return m


def _perspective_rows(h):
    return [
        i
        for i, c in enumerate(h._constraints)
        if c.name.startswith("_hull_reactor_d") and c.name.endswith("_c0")
    ]


# ── classification ──────────────────────────────────────────────────────────


def test_hull_perspective_rows_are_proved_convex(monkeypatch):
    h = dt.create_using("gdp.hull", build_synthesis_nl())
    rows = _perspective_rows(h)
    assert len(rows) == 3

    monkeypatch.setenv(FLAG, "0")
    assert not perspective_composite_enabled()
    convex_off, mask_off = classify_model(h)
    assert not convex_off
    assert not any(mask_off[i] for i in rows)

    monkeypatch.setenv(FLAG, "1")
    assert perspective_composite_enabled()
    convex_on, mask_on = classify_model(h)
    assert convex_on
    assert all(mask_on[i] for i in rows)
    # Nothing that was convex stops being convex.
    assert all(on or not off for on, off in zip(mask_on, mask_off))


def _unit_model():
    m = dm.Model("u")
    x = m.continuous("x", lb=-5, ub=5)
    xp = m.continuous("xp", lb=0, ub=5)
    w = m.continuous("w", lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    s = m.continuous("s", lb=-1, ub=1)
    L = (1 - 1e-8) * y + 1e-8
    return m, x, xp, w, y, s, L


@pytest.mark.parametrize(
    "build, expected",
    [
        (lambda x, xp, w, y, s, L: -dm.log(1 + xp / L) * L, "convex"),
        (lambda x, xp, w, y, s, L: dm.log(1 + xp / L) * L, "concave"),
        # already proved without the flag by the quad-over-affine perspective rule
        (lambda x, xp, w, y, s, L: ((x / L) ** 2 + 3 * (xp / L) - 2) * L, "convex-known"),
        (lambda x, xp, w, y, s, L: (2 * (x / L) + 1) * L, "affine"),
        # a variable outside the ratio: not a perspective
        (lambda x, xp, w, y, s, L: -dm.log(1 + xp / L + w) * L, None),
        # z**3 on a mixed-sign z is neither convex nor concave
        (lambda x, xp, w, y, s, L: ((x / L) ** 3) * L, None),
        # divisor differs from the multiplier
        (lambda x, xp, w, y, s, L: -dm.log(1 + xp / (y + 1)) * L, None),
        # multiplier not provably positive (s in [-1, 1])
        (lambda x, xp, w, y, s, L: -dm.log(1 + xp / (s + 2)) * s, None),
    ],
)
def test_perspective_composite_rule(monkeypatch, build, expected):
    from discopt._relax.convexity.lattice import Curvature

    m, x, xp, w, y, s, L = _unit_model()
    e = build(x, xp, w, y, s, L)
    want = {
        "convex": Curvature.CONVEX,
        "convex-known": Curvature.CONVEX,
        "concave": Curvature.CONCAVE,
        "affine": Curvature.AFFINE,
        None: Curvature.UNKNOWN,
    }[expected]
    monkeypatch.setenv(FLAG, "1")
    assert classify_expr(e, m) == want
    monkeypatch.setenv(FLAG, "0")
    if expected in ("convex", "concave"):
        assert classify_expr(e, m) == Curvature.UNKNOWN


# ── soundness: every tangent cut the rule licenses is valid ─────────────────


def _lift(h, idx, rng):
    """A random feasible point of the original GDP, lifted into the hull space."""
    names = [v.name for v in h._variables]
    x = np.zeros(len(names))
    k = int(rng.integers(3))
    a, c, f, v = _R[k]
    F = rng.uniform(0, 100)
    P = rng.uniform(0, min(200.0, a * np.log1p(F / c)))
    CR = rng.uniform(min(600.0, f + v * F), 600.0)
    j = int(rng.integers(2))
    b, fs, vs = _SP[j]
    S = rng.uniform(0, min(200.0, b * P))
    CS = rng.uniform(min(600.0, fs + vs * P), 600.0)
    for nm, val in dict(F=F, P=P, S=S, CR=CR, CS=CS).items():
        x[idx[nm]] = val
    x[idx[f"_gdp_aux_hull_reactor_{k}_{k}"]] = 1.0
    x[idx[f"_gdp_aux_hull_separator_{j}_{3 + j}"]] = 1.0
    for nm, val in dict(P=P, F=F, CR=CR).items():
        x[idx[f"_hull_reactor_v_{nm}_{k}"]] = val
    for nm, val in dict(S=S, P=P, CS=CS).items():
        x[idx[f"_hull_separator_v_{nm}_{j}"]] = val
    return x


def test_tangent_cuts_never_cut_a_feasible_point(monkeypatch):
    from discopt._relax.nlp_evaluator import NLPEvaluator

    monkeypatch.setenv(FLAG, "1")
    h = dt.create_using("gdp.hull", build_synthesis_nl())
    _, mask = classify_model(h)
    rows = _perspective_rows(h)
    assert all(mask[i] for i in rows)
    names = [v.name for v in h._variables]
    idx = {n: i for i, n in enumerate(names)}
    lb = np.array([float(v.lb) for v in h._variables])
    ub = np.array([float(v.ub) for v in h._variables])
    ev = NLPEvaluator(h)
    rng = np.random.default_rng(1617)

    feas = [_lift(h, idx, rng) for _ in range(200)]
    g_feas = np.array([ev.evaluate_constraints(p) for p in feas])
    # The lifted points really are feasible for the perspective rows.
    assert np.all(g_feas[:, rows] <= 1e-6)

    checked = 0
    for _ in range(150):
        # Linearisation points anywhere in the box, binaries fractional too
        # (as an LP/NLP relaxation would visit), plus near-zero selectors.
        x0 = rng.uniform(lb, ub)
        if rng.random() < 0.3:
            for k in range(3):
                x0[idx[f"_gdp_aux_hull_reactor_{k}_{k}"]] = 10.0 ** rng.uniform(-6, 0)
        g0 = ev.evaluate_constraints(x0)
        J = ev.evaluate_jacobian(x0)
        for i in rows:
            assert np.all(np.isfinite(J[i]))
            tang = g0[i] + (np.array(feas) - x0) @ J[i]
            scale = 1.0 + np.abs(g0[i]) + np.abs(np.array(feas) - x0) @ np.abs(J[i])
            assert np.all(tang <= 1e-6 + 1e-9 * scale), (i, tang.max())
            checked += len(feas)
    # Midpoint convexity of each row over the whole box.
    for _ in range(300):
        a, b = rng.uniform(lb, ub), rng.uniform(lb, ub)
        ga, gb, gm = (ev.evaluate_constraints(p) for p in (a, b, 0.5 * (a + b)))
        for i in rows:
            tol = 1e-7 * (1.0 + abs(ga[i]) + abs(gb[i]))
            assert gm[i] <= 0.5 * (ga[i] + gb[i]) + tol
            checked += 1
    assert checked > 0
    print(f"executed tangent/midpoint comparisons: {checked}")


# ── bounds ──────────────────────────────────────────────────────────────────


def _solve(monkeypatch, flag, method="hull", **box):
    monkeypatch.setenv(FLAG, flag)
    return build_synthesis_nl(**box).solve(gdp_method=method, time_limit=600)


def test_hull_root_bound_is_the_perspective_relaxation(monkeypatch):
    r = _solve(monkeypatch, "1")
    assert r.status == "optimal"
    assert r.objective == pytest.approx(SYNTH_OPT, abs=1e-3)
    assert r.root_bound is not None
    # At least as tight as the exact continuous hull relaxation, never past the optimum.
    assert r.root_bound >= SYNTH_HULL_RELAXATION - 1e-3
    assert r.root_bound <= r.objective + 1e-6 * (1 + abs(r.objective))
    assert r.bound <= r.objective + 1e-6 * (1 + abs(r.objective))


@pytest.mark.slow
@pytest.mark.parametrize(
    "box",
    [dict(), dict(f_ub=30.0), dict(f_lb=40.0, f_ub=80.0), dict(p_ub=60.0)],
)
def test_differential_root_bound_on_fixed_boxes(monkeypatch, box):
    oracle = _solve(monkeypatch, "0", method="big-m", **box)
    assert oracle.status == "optimal"
    off = _solve(monkeypatch, "0", **box)
    on = _solve(monkeypatch, "1", **box)
    assert (off.status, on.status) == ("optimal", "optimal"), (off.wall_time, on.wall_time)
    tol = 1e-6 * (1 + abs(oracle.objective))
    assert on.objective == pytest.approx(oracle.objective, abs=1e-3)
    assert on.root_bound >= off.root_bound - tol  # never looser
    assert on.root_bound <= oracle.objective + tol  # never past the box optimum
    assert on.bound <= oracle.objective + tol


def test_corpus_hull_instance_syn05hfsg_rows_are_proved_convex(monkeypatch):
    """The real-corpus member of the class: MINLPLib's own hull/perspective synthesis.

    ``syn05hfsg`` is a convex MINLP written in exactly the eps-perspective form;
    without the rule three of its rows (and so the model) are UNKNOWN.
    """
    from pathlib import Path

    from discopt.modeling.core import from_nl

    path = Path(__file__).parent / "data" / "minlplib_nl" / "syn05hfsg.nl"
    m = from_nl(str(path))
    monkeypatch.setenv(FLAG, "0")
    convex_off, mask_off = classify_model(m)
    monkeypatch.setenv(FLAG, "1")
    convex_on, mask_on = classify_model(m)
    assert (convex_off, convex_on) == (False, True)
    assert sum(on and not off for on, off in zip(mask_on, mask_off)) == 3
    assert all(on or not off for on, off in zip(mask_on, mask_off))
