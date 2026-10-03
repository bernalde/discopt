"""#1537 C: exact recentring of large-offset variables (``DISCOPT_RECENTRE``).

The pass rebuilds a model whose boxes sit far from the origin in coordinates
``x = z + c`` with every scalar affine subtree constant-folded, solves that, and
maps the result back (``discopt/modeling/_recentre.py``). Default ON since the
2026-10-03 graduation panel (CLAUDE.md §5); these tests pin its mechanics and its
fixes with the flag ON, that it is on by default, and that ``=0`` changes nothing.
"""

from __future__ import annotations

import os

import discopt.modeling as dm
import numpy as np
import pytest
from _invariance import translate
from discopt.modeling._recentre import (
    RecentreUnsupported,
    plan_shifts,
    recentre,
)

CORPUS = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl")


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv("DISCOPT_RECENTRE", "1")


def _moved(r):
    return (r.solver_stats or {}).get("recentre/variables_moved")


# ── mechanics ─────────────────────────────────────────────────────────────────


def test_selectivity_and_shift_values():
    m = dm.Model("sel")
    a = m.continuous("a", lb=1e6, ub=1e6 + 3)  # moved: |lb| >= 100 * width
    b = m.continuous("b", lb=0, ub=5)  # well scaled: untouched
    c = m.continuous("c", lb=1e3, ub=1e3 + 50)  # 1e3 < 100 * 50: untouched
    i = m.integer("i", lb=-2_000_000, ub=-1_999_996)  # moved, integer shift
    m.minimize(a + b + c + i)
    plan = plan_shifts(m, 100.0)
    assert set(plan) == {id(a), id(i)}
    assert float(plan[id(a)]) == 1e6 and float(plan[id(i)]) == -2e6
    assert recentre(dm.Model("empty")) is None


def test_constant_folding_removes_the_cancelling_constant():
    """``(y - c)`` written by the user becomes exactly ``z``: the unfolded
    ``(z + c) - c`` certified a false infeasible on st_e36 (#1542)."""
    m = dm.Model("fold")
    y = m.continuous("y", lb=1e6, ub=1e6 + 3)
    i = m.integer("i", lb=-2e6, ub=-2e6 + 4)
    m.subject_to((y - 1e6) + 2 * (i + 2e6) <= 5)
    m.minimize((y - 1e6 - 0.3) ** 2 + (i + 2e6) ** 3)
    rc = recentre(m)
    z = {v.name: v for v in rc.model._variables}
    obj = rc.model._objective.expression
    assert obj.right.left is z["i"]  # (i + 2e6) ** 3  ->  i ** 3, constant gone
    assert repr(rc.model._objective.expression.left.left).startswith("Σ")
    for v in rc.model._variables:
        assert float(v.lb) == 0.0
    # exactness at sampled points: f_new(x - c) == f(x)
    from discopt._tape_nlp_evaluator import cached_tape_evaluator

    ev, ev_rc = cached_tape_evaluator(m), cached_tape_evaluator(rc.model)
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = np.array([1e6 + rng.uniform(0, 3), -2e6 + rng.integers(0, 5)])
        zpt = x - np.array([1e6, -2e6])
        assert ev_rc.evaluate_objective(zpt) == pytest.approx(ev.evaluate_objective(x), abs=1e-6)
        assert np.allclose(ev_rc.evaluate_constraints(zpt), ev.evaluate_constraints(x), atol=1e-6)


@pytest.mark.parametrize("lb", [1e6, 3e3])
def test_shared_square_keeps_its_verdict(lb):
    """The rewrite preserves the DAG's sharing. ``x * x`` is ONE node on both
    sides, and the square rule reads that identity (``left is right``). Folding
    each use of ``x`` into a fresh ``z + c`` sum made the two sides distinct
    objects, so ``min x*x + w`` was convex as written and not convex once
    recentred: the solve left the convex route on a model it had proven."""
    from discopt._relax.convexity import classify_model

    m = dm.Model("share")
    x = m.continuous("x", lb=lb, ub=lb + 10)
    w = m.continuous("w", lb=0, ub=1)
    m.minimize(x * x + w)
    m.subject_to(x * x - w <= (lb + 5) ** 2)
    rc = recentre(m)
    assert rc is not None and rc.model is not m
    sq = rc.model._objective.expression.left
    assert sq.left is sq.right
    assert classify_model(m) == (True, [True])
    assert classify_model(rc.model) == classify_model(m)


def test_parameters_are_not_folded_and_stay_live(flag_on):
    m = dm.Model("par")
    y = m.continuous("y", lb=1e6, ub=1e6 + 10)
    p = m.parameter("p", value=3.0)
    m.minimize((y - 1e6 - p) ** 2)
    r1 = m.solve(time_limit=20)
    p.value = np.asarray(7.0)
    r2 = m.solve(time_limit=20)
    assert _moved(r1) == 1.0 and _moved(r2) == 1.0
    assert float(np.asarray(r1.x["y"])) == pytest.approx(1e6 + 3, abs=1e-5)
    assert float(np.asarray(r2.x["y"])) == pytest.approx(1e6 + 7, abs=1e-5)


def test_unsupported_structure_refuses_and_solve_falls_back(flag_on, caplog):
    m = dm.Model("gdp")
    x = m.continuous("x", lb=1e6, ub=1e6 + 10)
    m.either_or([[x <= 1e6 + 1], [x >= 1e6 + 9]])
    m.minimize((x - 1e6 - 5) ** 2)
    with pytest.raises(RecentreUnsupported):
        recentre(m)
    with caplog.at_level("WARNING", logger="discopt"):
        r = m.solve(time_limit=20)  # unrecentred, still solved
    assert _moved(r) is None and r.status in ("optimal", "feasible")
    # The skip is visible: a WARNING and the reason in solver_stats (N2).
    assert "recentring was skipped" in caplog.text
    assert isinstance(r.solver_stats.get("recentre/skipped"), str)


def test_mapped_point_is_reverified_against_the_original(flag_on, monkeypatch):
    """A mapping defect cannot publish a point the user's model rejects: corrupt
    ``to_outer`` and the mapped point is withheld exactly as main's #772
    false-primal guard withholds one -- whatever the inner status was."""
    import discopt.modeling._recentre as rmod

    real = rmod.Recentring.to_outer

    def off_by_one(self, values):
        return {k: np.asarray(v) + 1.0 for k, v in real(self, values).items()}

    monkeypatch.setattr(rmod.Recentring, "to_outer", off_by_one)
    m = dm.Model("guard")
    y = m.continuous("y", lb=1e6, ub=1e6 + 3)
    m.subject_to(y <= 1e6 + 2.5)
    m.minimize(-(y - 1e6))  # optimum at the row: y = 1e6 + 2.5
    r = m.solve(time_limit=20)
    assert r.status == "error" and r.x is None and r.objective is None
    assert r.incumbent_verification_failed and not r.gap_certified
    assert r.solver_stats.get("recentre/mapped_point_refused") == 1.0


def test_initial_solution_and_warm_start_are_mapped(flag_on):
    m = dm.Model("init")
    y = m.continuous("y", lb=1e6, ub=1e6 + 3)
    m.minimize((y - 1e6 - 1.7) ** 2)
    r = m.solve(time_limit=20, initial_solution={y: 1e6 + 1.0})
    assert r.gap_certified and float(np.asarray(r.x["y"])) == pytest.approx(1e6 + 1.7, abs=1e-5)
    r2 = m.solve(time_limit=20, warm_start=r)
    assert r2.gap_certified and float(np.asarray(r2.x["y"])) == pytest.approx(1e6 + 1.7, abs=1e-5)


def _shifted_qp():
    m = dm.Model("off")
    y = m.continuous("y", lb=1e6, ub=1e6 + 3)
    i = m.integer("i", lb=-2e6, ub=-2e6 + 5)
    m.subject_to((y - 1e6) + (i + 2e6) >= 3.5)
    m.minimize((y - 1e6 - 1.2) ** 2 + (i + 2e6 - 2.6) ** 2)
    return m


def test_flag_is_on_by_default(monkeypatch):
    """Graduated (#1537): with ``DISCOPT_RECENTRE`` unset the pass runs and moves
    the far-offset variables, exactly as with ``=1``."""
    monkeypatch.delenv("DISCOPT_RECENTRE", raising=False)
    r = _shifted_qp().solve(time_limit=20, deterministic=True)
    assert _moved(r) == 2.0
    monkeypatch.setenv("DISCOPT_RECENTRE", "1")
    r1 = _shifted_qp().solve(time_limit=20, deterministic=True)
    assert (r.status, r.objective, r.bound, r.node_count, r.gap_certified) == (
        r1.status,
        r1.objective,
        r1.bound,
        r1.node_count,
        r1.gap_certified,
    )


def test_flag_off_is_untouched(monkeypatch):
    """``DISCOPT_RECENTRE=0`` (the opt-out) never enters the pass: a ``recentre``
    that raises if called changes nothing, and nothing is reported as moved."""
    import discopt.modeling._recentre as rmod

    def unreachable(*a, **k):
        raise AssertionError("recentre() was called with the flag OFF")

    monkeypatch.setattr(rmod, "recentre", unreachable)
    monkeypatch.setenv("DISCOPT_RECENTRE", "0")
    r = _shifted_qp().solve(time_limit=20, deterministic=True)
    assert _moved(r) is None and "recentre/skipped" not in r.solver_stats


# ── the fixes (flag ON). Each was a false certificate or a crash on the main of
# the time; several pass with the flag OFF on today's main too (other #1537
# workstreams fixed them), so each asserts the pass actually ran (``_moved``) --
# they pin that recentring keeps them right, not that only recentring does. ──


@pytest.mark.parametrize("c", [1e5, 1e6])
def test_1543_one_variable_integer_qp(flag_on, c):
    m = dm.Model("sq")
    y = m.integer("y", lb=c, ub=3 + c)
    u = y - c
    m.minimize(6 * u * u - 12 * u)
    r = m.solve(time_limit=20)
    assert _moved(r) == 1.0
    assert r.gap_certified and r.objective == pytest.approx(-6.0, abs=1e-6)
    assert float(np.asarray(r.x["y"])) == pytest.approx(c + 1.0, abs=1e-6)


def test_1542_shifted_cubic(flag_on):
    rng = np.random.default_rng(1)
    A = rng.integers(-4, 5, size=(3, 5)).astype(float)
    b = rng.integers(2, 10, size=3).astype(float)
    cc = rng.integers(-5, 6, size=5).astype(float)
    s = np.array([798491.0, -1314226.0, -1100101.0, -1228561.0, 555147.0])
    m = dm.Model("cube")
    ys = [m.integer(f"i{k}", lb=s[k], ub=4 + s[k]) for k in range(3)] + [
        m.continuous(f"c{k}", lb=s[k], ub=5 + s[k]) for k in range(3, 5)
    ]
    x = [ys[j] - s[j] for j in range(5)]
    for r_ in range(3):
        m.subject_to(sum(A[r_, j] * x[j] for j in range(5)) <= b[r_])
    m.minimize(sum(cc[j] * x[j] for j in range(5)) + 0.2 * x[3] ** 3)
    r = m.solve(time_limit=20)
    assert _moved(r) == 5.0
    assert r.gap_certified and r.objective == pytest.approx(-36.0, abs=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["st_miqp2", "st_miqp3", "st_miqp4", "alan"])
def test_corpus_shift_false_certificates_recover(flag_on, name):
    """The false-certificate / guard-refusal shift failures of the invariance
    corpus panel (#1543) come back certified at the base optimum. st_miqp2 is the
    graduation panel's ON-arm false certificate (262 vs 2): its x1 box
    [-1.3e6, 9.9987e9] slipped past a fixed "open" cutoff and stayed unfolded."""
    m = dm.from_nl(os.path.join(CORPUS, f"{name}.nl"))
    base = m.solve(time_limit=20)
    assert base.gap_certified
    r = translate(m, 1e6, seed=2).solve(time_limit=20)
    assert _moved(r)
    assert r.gap_certified and r.status == "optimal"
    assert r.objective == pytest.approx(base.objective, abs=1e-6, rel=1e-4)


@pytest.mark.slow
def test_corpus_nvs09_shift_no_longer_crashes(flag_on):
    """#1544: the shifted nvs09 raised RecursionError. Recentred it solves without
    a crash and without a false certificate; it does not re-certify within 20 s
    (the entry experiment's folded arm lost it too), so that is not asserted."""
    m = dm.from_nl(os.path.join(CORPUS, "nvs09.nl"))
    base = m.solve(time_limit=20)
    r = translate(m, 1e6, seed=2).solve(time_limit=20)
    assert _moved(r)
    if r.gap_certified:
        assert r.objective == pytest.approx(base.objective, abs=1e-6, rel=1e-4)


def test_one_sided_offsets_are_moved():
    m = dm.Model("one")
    a = m.continuous("a", lb=1e6, ub=1e15)  # effectively x >= 1e6
    b = m.integer("b", lb=-3e6, ub=1e15)  # ub effectively open (a bare lb gets the 1e6 default box)
    d = m.continuous("d", ub=-5e5)  # lb open: anchored at ub
    e = m.continuous("e", lb=50.0)  # |lb| < ratio: untouched
    m.minimize(a + b - d + e)
    plan = plan_shifts(m, 100.0)
    assert set(plan) == {id(a), id(b), id(d)}
    assert float(plan[id(d)]) == -5e5


# ── builder-resident state (fast API): translated exactly, or refused ─────────
# Before the fix every one of these was dropped from the recentred model: the
# rows vanished (a looser problem) and a builder objective became the zero
# placeholder, certifying 0 as optimal. Each optimum is at a vertex the dropped
# state decides.

C6 = 1e6
_FAST_OPT = 3e6 + 20  # min x0 + 2 x1, x0 + x1 >= 2e6 + 15, x in [1e6, 1e6 + 10]^2


def _fast_box(m):
    return m.continuous("x", shape=(2,), lb=[C6, C6], ub=[C6 + 10, C6 + 10])


def _assert_fast_optimum(r, opt):
    assert _moved(r) == 1.0
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(opt, abs=1e-6)
    assert r.bound <= opt + 1e-6
    x = np.asarray(r.x["x"], dtype=float)
    assert x[0] + x[1] >= 2e6 + 15 - 1e-6


def test_builder_add_linear_constraints_is_translated(flag_on):
    m = dm.Model("fast_rows")
    x = _fast_box(m)
    m.add_linear_constraints(np.array([[1.0, 1.0]]), x, ">=", np.array([2e6 + 15]))
    m.minimize(x[0] + 2 * x[1])
    _assert_fast_optimum(m.solve(time_limit=20), _FAST_OPT)


def test_builder_add_linear_objective_is_translated(flag_on):
    m = dm.Model("fast_lin_obj")
    x = _fast_box(m)
    m.subject_to(x[0] + x[1] >= 2e6 + 15)
    m.add_linear_objective(np.array([1.0, 2.0]), x, 0.5, "minimize")
    _assert_fast_optimum(m.solve(time_limit=20), _FAST_OPT + 0.5)


def test_builder_add_quadratic_objective_is_translated(flag_on):
    # 0.5 x'Qx + q'x + k with an off-diagonal Q: in z-space it gains the linear
    # term S c and the constant q'c + 0.5 c'Sc, all exact here.
    m = dm.Model("fast_quad_obj")
    x = _fast_box(m)
    m.add_linear_constraints(np.array([[1.0, 1.0]]), x, ">=", np.array([2e6 + 15]))
    Q = np.array([[2.0, 1.0], [1.0, 2.0]])
    q = np.array([-4e6 - 20.0, -2e6 - 10.0])  # shifts the minimiser into the box
    m.add_quadratic_objective(Q, q, x, 7.0, "minimize")

    def f(v):
        return 0.5 * v @ Q @ v + q @ v + 7.0

    grid = [np.array([C6 + a, C6 + b]) for a in range(11) for b in range(11) if a + b >= 15]
    # convex: the continuous optimum is no worse than the best lattice point
    best = min(f(v) for v in grid)
    r = m.solve(time_limit=20)
    assert _moved(r) == 1.0 and r.status == "optimal" and r.gap_certified
    tol = 1e-9 * abs(best)  # |f| ~ 4e12 here
    assert r.objective <= best + tol
    assert r.objective == pytest.approx(f(np.asarray(r.x["x"], dtype=float)), abs=tol)
    assert r.bound <= r.objective + tol


def test_builder_model_constraint_fast_family_is_translated(flag_on):
    m = dm.Model("fam")
    x = m.continuous("x", shape=(3,), lb=C6, ub=C6 + 10)
    idx = m.set("I", [0, 1, 2])
    m.constraint(idx, lambda i: x[i] >= C6 + 2.0 + i, name="lo", fast=True)
    m.minimize(dm.sum((x - C6) ** 2))
    r = m.solve(time_limit=20)
    assert _moved(r) == 1.0 and r.status == "optimal" and r.gap_certified  # one (3,) variable
    assert r.objective == pytest.approx(4 + 9 + 16, abs=1e-6)


def test_builder_row_with_inexact_translation_refuses(flag_on, caplog):
    """``b - A c`` that is not a double cannot be translated exactly: refuse
    (and solve unrecentred) rather than round a row."""
    m = dm.Model("fast_inexact")
    x = m.continuous("x", shape=(2,), lb=[C6 + 0.1, C6], ub=[C6 + 10, C6 + 10])
    A = np.array([[0.7, 1.0]])  # 1.3e6 + 7 - 0.7 * 1e6 - 1e6 is not a double
    m.add_linear_constraints(A, x, ">=", np.array([1.3e6 + 7]))
    m.minimize(x[0] + x[1])
    with pytest.raises(RecentreUnsupported, match="not exactly representable"):
        recentre(m)
    with caplog.at_level("WARNING", logger="discopt"):
        r = m.solve(time_limit=20)
    assert _moved(r) is None and "recentre/skipped" in r.solver_stats
    assert "recentring was skipped" in caplog.text


# ── N1, N3, N6 ────────────────────────────────────────────────────────────────


def test_deep_sum_does_not_recurse_out(flag_on):
    m = dm.Model("deep")
    x = m.continuous("x", lb=C6, ub=C6 + 5)
    y = m.continuous("y", shape=(2000,), lb=0, ub=1)
    expr = x - C6
    for k in range(2000):
        expr = expr + y[k]
    m.subject_to(expr <= 50)
    m.minimize((x - C6 - 2) ** 2 + y[0])
    rc = recentre(m)  # raised RecursionError before the fix
    assert rc is not None and len(rc.shifts) == 1


def test_shift_bounds_rounds_outward():
    from fractions import Fraction

    import discopt.modeling._recentre as rmod

    lb = np.array([140.25, 0.1, -np.inf, 3.0])
    ub = np.array([1e17, 7.3, -5e5 + 0.1, np.inf])
    c = np.array([140.0, 3.0, -5e5, 1.0])
    lo, hi = rmod.shift_bounds(lb, ub, c)
    checked = 0
    for k in range(len(lb)):
        if np.isfinite(lb[k]):
            assert Fraction(float(lo[k])) <= Fraction(float(lb[k])) - Fraction(float(c[k]))
            checked += 1
        else:
            assert lo[k] == -np.inf
        if np.isfinite(ub[k]):
            assert Fraction(float(hi[k])) >= Fraction(float(ub[k])) - Fraction(float(c[k]))
            checked += 1
        else:
            assert hi[k] == np.inf
    assert checked == 6
    # the motivating case: 1e17 - 140 rounds to nearest INSIDE the box
    assert Fraction(1e17 - 140.0) < Fraction(1e17) - 140
    assert hi[0] > 1e17 - 140.0


def test_inner_only_columns_are_not_returned(flag_on):
    m = _shifted_qp()
    r = m.solve(time_limit=20)
    assert _moved(r) == 2.0
    assert set(r.x) == {v.name for v in m._variables}


def _affine_product_model(kind: str, c: float) -> dm.Model:
    m = dm.Model(kind)
    if kind == "int_sq":
        y = m.integer("y", lb=c, ub=c + 20)
        m.minimize((6 * y) * y - 12 * (c + 7.3) * y)
    elif kind == "cont_sq_loop":
        y = m.continuous("y", lb=c, ub=c + 20)
        f = (6 * y) * y - 12 * (c + 7.3) * y
        for i in range(200):
            f = f + 1e-3 * m.continuous(f"z{i}", lb=0, ub=1)
        m.minimize(f)
    elif kind == "bilin_cross":
        x = m.continuous("x", lb=c, ub=c + 10)
        y = m.continuous("y", lb=c, ub=c + 10)
        m.minimize((2 * x) * x + (3 * y) * y - x * y - 4 * c * x - 5 * c * y)
    else:  # miqp_con
        x = m.continuous("x", lb=c, ub=c + 10)
        k = m.integer("k", lb=0, ub=5)
        m.subject_to((x * 1.0) * (x * 1.0) <= (c + 7) ** 2)
        m.minimize((2 * x) * x - 4 * (c + 9) * x + k)
    return m


@pytest.mark.parametrize("c", [3000.5, 1e5 + 0.5])
@pytest.mark.parametrize("kind", ["int_sq", "cont_sq_loop", "bilin_cross", "miqp_con"])
def test_recentred_affine_products_keep_their_certificate(monkeypatch, kind, c):
    """Recentring turns ``(a*y)*y`` into ``(a*z + a*s)*(z + s)``, a product of two
    affine expressions, which the quadratic extractor does not expand (#1537,
    review round 2). Measured on this class: no certificate is lost, because the
    solve still certifies by another route. This pins that, and cross-checks each
    arm's bound against the other arm's objective."""
    out = {}
    for arm in ("0", "1"):
        monkeypatch.setenv("DISCOPT_RECENTRE", arm)
        out[arm] = _affine_product_model(kind, c).solve(time_limit=60)
    r0, r1 = out["0"], out["1"]
    assert _moved(r1) and _moved(r1) > 0
    assert r0.gap_certified and r1.gap_certified
    tol = 1e-4 * max(1.0, abs(r0.objective))
    assert abs(r1.objective - r0.objective) <= tol
    assert r1.bound <= r0.objective + 1e-9 * abs(r0.objective)
    assert r0.bound <= r1.objective + 1e-9 * abs(r1.objective)
