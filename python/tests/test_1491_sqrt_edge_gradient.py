"""Regression tests for #1491: a flat-zero ``sqrt'`` at 0 produced invalid OA cuts.

POUNCE's ``sqrt`` derivative rule returns ``0`` wherever its argument is ``<= 0``
(and ``x ** 0.5`` is canonicalised onto the same rule), while the true one-sided
slope at 0 is ``+inf``. The tape evaluator passed that ``0`` on, the objective OA
cut at ``x = 0`` became ``eta >= 0.3 x`` for ``max sqrt(x) - 0.3 x``, and the convex
MINLP route certified ``x = 0`` (objective ``0``) against a true ``0.832`` at ``x = 3``.

Two halves, both tested here:

1. the evaluator reports the honest non-finite slope at the edge -- through both
   the arena and the Python-DAG lowering -- and stays bit-identical everywhere the
   argument is positive;
2. every linearization generator refuses a non-finite value/gradient (no cut, and
   no certificate computed from it), independently of how the gradient arose.
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import cutting_planes as cp
from discopt._tape_nlp_evaluator import cached_tape_evaluator

pounce = pytest.importorskip("pounce")

_ARENA_MODES = ["1", "0"]  # DISCOPT_ARENA_TAPE: arena lowering, Python-DAG lowering


def _fresh_evaluator(m):
    ev = cached_tape_evaluator(m)
    assert ev is not None, "model must be tape-representable for this test"
    return ev


# --------------------------------------------------------------------------- #
# 1. The evaluator reports the edge slope honestly.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("arena", _ARENA_MODES)
@pytest.mark.parametrize(
    "build, x0, expected",
    [
        # d/dx sqrt(x) at 0 is +inf; maximize negates the internal objective.
        (lambda x: dm.sqrt(x), 0.0, -math.inf),
        (lambda x: x**0.5, 0.0, -math.inf),
        # d/dx sqrt(1 - x^2) at x = 1 is -inf -> +inf after the maximize negation.
        (lambda x: dm.sqrt(1 - x**2), 1.0, math.inf),
        # sqrt(x) - 0.3 x: inf - 0.3 is still inf.
        (lambda x: dm.sqrt(x) - 0.3 * x, 0.0, -math.inf),
    ],
)
def test_gradient_at_sqrt_edge_is_infinite(monkeypatch, arena, build, x0, expected):
    monkeypatch.setenv("DISCOPT_ARENA_TAPE", arena)
    m = dm.Model("edge")
    x = m.continuous("x", lb=-1.0, ub=10.0)
    m.maximize(build(x))
    g = _fresh_evaluator(m).evaluate_gradient(np.array([x0]))
    assert g.shape == (1,)
    assert g[0] == expected, f"gradient at the sqrt edge must be {expected}, got {g[0]}"


@pytest.mark.parametrize("arena", _ARENA_MODES)
def test_jacobian_at_sqrt_edge_is_infinite(monkeypatch, arena):
    """Constraint rows get the same honest slope (``t <= sqrt(x)`` at ``x = 0``)."""
    monkeypatch.setenv("DISCOPT_ARENA_TAPE", arena)
    m = dm.Model("edge_con")
    x = m.continuous("x", lb=0.0, ub=10.0)
    t = m.continuous("t", lb=0.0, ub=10.0)
    m.subject_to(t - dm.sqrt(x) <= 0)
    m.subject_to(t - x**0.5 <= 0)
    m.maximize(t)
    jac = _fresh_evaluator(m).evaluate_jacobian(np.array([0.0, 0.0]))
    assert jac[0, 0] == -math.inf and jac[1, 0] == -math.inf
    assert jac[0, 1] == 1.0 and jac[1, 1] == 1.0


def test_negative_argument_has_no_derivative():
    """``sqrt`` of a negative argument has no derivative: NaN, never a finite 0."""
    m = dm.Model("neg")
    x = m.continuous("x", lb=-5.0, ub=5.0)
    m.minimize(dm.sqrt(x))
    g = _fresh_evaluator(m).evaluate_gradient(np.array([-1.0]))
    assert not np.isfinite(g[0])


@pytest.mark.parametrize("arena", _ARENA_MODES)
def test_positive_argument_is_bit_identical_to_plain_sqrt(monkeypatch, arena):
    """Where the argument is positive the edge handling is invisible: value,
    gradient and Hessian equal a plain ``NlExpr.sqrt`` tape exactly."""
    monkeypatch.setenv("DISCOPT_ARENA_TAPE", arena)
    m = dm.Model("pos")
    x = m.continuous("x", lb=0.0, ub=3.0)
    y = m.continuous("y", lb=0.0, ub=3.0)
    m.minimize(dm.sqrt(x * y + 1) * dm.log(y + 3) + dm.sqrt(x) ** 2 + x**0.5)
    ev = _fresh_evaluator(m)

    E = pounce.NlExpr
    X, Y = E.var(0), E.var(1)
    one, three, half = E.const_(1.0), E.const_(3.0), E.const_(0.5)
    ref_expr = E.sqrt(X * Y + one) * E.log(Y + three) + E.sqrt(X) ** E.const_(2.0) + X**half
    ref = pounce.build_nl_problem(2, ref_expr, constraints=None)

    rng = np.random.default_rng(1491)
    compared = 0
    for _ in range(200):
        v = rng.uniform(1e-6, 3.0, size=2)
        assert ev.evaluate_objective(v) == ref.objective(v)
        assert np.array_equal(ev.evaluate_gradient(v), np.asarray(ref.gradient(v)))
        assert np.array_equal(
            ev.evaluate_hessian_values(v, 1.0, np.zeros(0)),
            np.asarray(ref.hessian(v, lam=np.zeros(0), obj_factor=1.0)),
        )
        compared += 1
    assert compared == 200


# --------------------------------------------------------------------------- #
# 2. Linearization generators refuse a non-finite value/gradient.
# --------------------------------------------------------------------------- #


class _StubEvaluator:
    """Two variables, two constraints; row 0 has an infinite slope at x."""

    n_variables = 2
    n_constraints = 2

    def __init__(self, grad, jac, cons, obj=0.0):
        self._g = np.asarray(grad, dtype=float)
        self._j = np.asarray(jac, dtype=float)
        self._c = np.asarray(cons, dtype=float)
        self._f = float(obj)

    def evaluate_objective(self, x):
        return self._f

    def evaluate_gradient(self, x):
        return self._g.copy()

    def evaluate_constraints(self, x):
        return self._c.copy()

    def evaluate_jacobian(self, x):
        return self._j.copy()


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_objective_oa_cut_refused_on_non_finite_gradient(bad):
    ev = _StubEvaluator([bad, 1.0], np.eye(2), [1.0, 1.0])
    assert cp.generate_objective_oa_cut(ev, np.zeros(2), 3, z_index=2) is None
    finite = _StubEvaluator([2.0, 1.0], np.eye(2), [1.0, 1.0])
    assert cp.generate_objective_oa_cut(finite, np.zeros(2), 3, z_index=2) is not None


def test_objective_oa_cut_refused_on_non_finite_value():
    ev = _StubEvaluator([1.0, 1.0], np.eye(2), [1.0, 1.0], obj=math.nan)
    assert cp.generate_objective_oa_cut(ev, np.zeros(2), 3, z_index=2) is None


@pytest.mark.parametrize("bad", [math.inf, math.nan])
def test_generate_oa_cut_raises_on_non_finite(bad):
    with pytest.raises(cp.NonFiniteLinearizationError):
        cp.generate_oa_cut(np.array([bad, 1.0]), 0.0, np.zeros(2))
    with pytest.raises(cp.NonFiniteLinearizationError):
        cp.generate_oa_cut(np.array([1.0, 1.0]), bad, np.zeros(2))


def test_constraint_oa_report_skips_and_keeps_row_attribution():
    jac = np.array([[-math.inf, 1.0], [2.0, 3.0]])
    ev = _StubEvaluator([0.0, 0.0], jac, [1.0, 1.0])
    rep = cp.generate_oa_cuts_from_evaluator_report(ev, np.zeros(2))
    assert rep.rows == (1,)
    assert len(rep.cuts) == 1
    assert np.array_equal(rep.cuts[0].coeffs, jac[1])
    assert [(s.constraint_index, s.reason) for s in rep.skipped] == [
        (0, cp.NON_FINITE_LINEARIZATION)
    ]
    assert len(cp.generate_oa_cuts_from_evaluator(ev, np.zeros(2))) == 1


def test_separation_report_skips_and_keeps_row_attribution():
    jac = np.array([[math.nan, 1.0], [2.0, 3.0]])
    ev = _StubEvaluator([0.0, 0.0], jac, [1.0, 1.0])  # both rows violated (g > 0)
    rep = cp.separate_oa_cuts_report(ev, np.zeros(2))
    assert rep.rows == (1,)
    assert np.array_equal(rep.cuts[0].coeffs, jac[1])
    assert rep.skipped[0].reason == cp.NON_FINITE_LINEARIZATION
    assert len(cp.separate_oa_cuts(ev, np.zeros(2))) == 1


def test_gbd_box_min_linear_refuses_non_finite():
    from discopt.decomposition.benders.gbd import _box_min_linear

    lb, ub = np.zeros(2), np.full(2, 10.0)
    _, finite = _box_min_linear(np.array([-math.inf, 1.0]), np.zeros(2), range(2), lb, ub)
    assert finite is False
    _, finite = _box_min_linear(np.array([math.nan, 1.0]), np.zeros(2), range(2), lb, ub)
    assert finite is False


def test_convex_certificate_gate_refuses_non_finite_gradient():
    """The #849 KKT-residual gate must not read an infinite gradient as stationary
    (``inf / inf`` scaling made the residual 0)."""
    from discopt.solver import _convex_nlp_certificate_gap

    class _Ev:
        n_variables = 1
        n_constraints = 0

        def evaluate_gradient(self, x):
            return np.array([-math.inf])

        def evaluate_objective(self, x):
            return 0.0

    out = _convex_nlp_certificate_gap(
        _Ev(),
        np.array([0.0]),
        None,
        np.array([0.0]),
        np.array([10.0]),
        np.zeros(0),
        np.zeros(0),
        0.0,
    )
    assert out is None


# --------------------------------------------------------------------------- #
# 3. End to end: the issue's repros no longer certify a wrong optimum.
# --------------------------------------------------------------------------- #


def _assert_correct_max(r, true_obj, tol=1e-4):
    assert r.objective is not None
    assert r.objective == pytest.approx(true_obj, abs=tol), (r.status, r.objective, r.x)
    if r.bound is not None and np.isfinite(r.bound):
        # Maximize: a valid dual bound is never below the true optimum.
        assert r.bound >= true_obj - tol, (r.bound, true_obj)


def _sqrt_minus_linear():
    m = dm.Model("s")
    x = m.integer("x", lb=0, ub=10)
    m.maximize(dm.sqrt(x) - 0.3 * x)
    return m, math.sqrt(3.0) - 0.9


def _pow_half_minus_linear():
    m = dm.Model("p")
    x = m.integer("x", lb=0, ub=10)
    m.maximize(x**0.5 - 0.3 * x)
    return m, math.sqrt(3.0) - 0.9


def _two_sqrt_coupled():
    m = dm.Model("a")
    x = m.integer("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.subject_to(x + y <= 6)
    m.maximize(dm.sqrt(x) + dm.sqrt(y) - 0.1 * x)
    # x integer: best is x=2, y=4 -> sqrt2 + 2 - 0.2.
    return m, math.sqrt(2.0) + 2.0 - 0.2


def _sqrt_in_constraint():
    m = dm.Model("c")
    x = m.integer("x", lb=0, ub=10)
    t = m.continuous("t", lb=0, ub=10)
    m.subject_to(t <= dm.sqrt(x))
    m.maximize(t - 0.3 * x)
    return m, math.sqrt(3.0) - 0.9


def _sqrt_with_binary():
    m = dm.Model("b")
    x = m.integer("x", lb=0, ub=10)
    b = m.binary("b")
    m.subject_to(x <= 10 * b)
    m.maximize(dm.sqrt(x) - 0.3 * x - 0.1 * b)
    return m, math.sqrt(3.0) - 0.9 - 0.1


@pytest.mark.parametrize(
    "factory",
    [
        _sqrt_minus_linear,
        _pow_half_minus_linear,
        _two_sqrt_coupled,
        _sqrt_in_constraint,
        _sqrt_with_binary,
    ],
)
def test_issue_repro_default_route(factory):
    m, true_obj = factory()
    r = m.solve(time_limit=60)
    _assert_correct_max(r, true_obj)


@pytest.mark.parametrize("factory", [_sqrt_minus_linear, _sqrt_in_constraint])
@pytest.mark.parametrize("solver", ["mip-nlp", "bb"])
def test_issue_repro_explicit_routes(factory, solver):
    m, true_obj = factory()
    r = m.solve(time_limit=60, solver=solver)
    _assert_correct_max(r, true_obj)


def _optimum_at_the_edge():
    """The optimum sits exactly where the slope is infinite (x = 0)."""
    m = dm.Model("edge_opt")
    x = m.integer("x", lb=0, ub=10)
    m.maximize(dm.sqrt(x) - 2.0 * x)
    return m, 0.0


@pytest.mark.parametrize("solver", [None, "mip-nlp", "bb"])
def test_optimum_at_the_edge_is_not_misreported(solver):
    m, true_obj = _optimum_at_the_edge()
    r = m.solve(time_limit=60, solver=solver)
    _assert_correct_max(r, true_obj)


def test_interior_support_points_stay_in_box_and_move_inward():
    lb = np.array([0.0, 0.0, -np.inf, 2.0])
    ub = np.array([10.0, 0.0, np.inf, np.inf])
    x = np.array([0.0, 0.0, 5.0, 2.0])
    pts = list(cp.interior_support_points(x, lb, ub))
    assert len(pts) == len(cp._INTERIOR_PULL_FRACTIONS)
    for p in pts:
        assert np.all(p >= lb) and np.all(p <= ub)
        assert p[0] > 0.0  # pulled toward the centre of [0, 10]
        assert p[1] == 0.0  # fixed coordinate never moves
        assert p[2] == 5.0  # free coordinate has no interior to move to
        assert p[3] > 2.0  # half-bounded: one unit inward
    assert pts == sorted(pts, key=lambda p: p[0])
