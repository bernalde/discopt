"""Regression tests for the convex-quadratic objective node-bound gate.

* #1569 -- ``_objective_is_convex_quadratic`` built its flat box by repeating each
  variable's WHOLE ``lb``/``ub`` array ``v.size`` times, so every model with a
  ``shape=(n,)`` variable raised inside the Hessian call, a broad ``except`` filed
  it as a user-code failure, and the bound never engaged on an array model.
* #1574 -- ``DISCOPT_CONVEX_OBJ_DEGREE_GATE``: decide "at most quadratic" from the
  objective alone (``polynomial_degree_bound``) and test definiteness on the
  Hessian's support, so weighted squares (``c*x**2``) and DAE ``integral(u**2)``
  objectives get the bound.

Both are bound-changing in effect, so besides the gate verdicts these tests run
the §5 differential bound test (the convex bound never exceeds the box optimum)
and feasible-point sampling (never exceeds the objective at any point of the box)
on the newly admitted models.
"""

from __future__ import annotations

import logging

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import solver
from discopt._tape_nlp_evaluator import build_evaluator
from scipy.optimize import minimize

_FLAG = "DISCOPT_CONVEX_OBJ_DEGREE_GATE"
_WARN = "convex-quadratic objective test failed"


def _evaluator(model):
    def _jax():
        from discopt._relax.nlp_evaluator import cached_evaluator

        return cached_evaluator(model)

    return build_evaluator(model, _jax)


def _gate(model) -> bool:
    n = sum(v.size for v in model._variables)
    return solver._objective_is_convex_quadratic(model, _evaluator(model), n)


def _gate_warnings(caplog):
    return [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == "discopt.solver" and _WARN in r.getMessage()
    ]


# ------------------------------------------------------------------ model builders
def _issue1569(lb=-2.0, ub=2.0):
    m = dm.Model("issue1569")
    x = m.continuous("x", shape=(3,), lb=lb, ub=ub)
    m.subject_to(x[0] * x[1] >= 0.5)
    m.minimize(sum((x[i] - 0.3) ** 2 for i in range(3)))
    return m


def _issue1569_scalar():
    m = dm.Model("issue1569_scalar")
    xs = [m.continuous(f"x{i}", lb=-2, ub=2) for i in range(3)]
    m.subject_to(xs[0] * xs[1] >= 0.5)
    m.minimize(sum((xs[i] - 0.3) ** 2 for i in range(3)))
    return m


def _array_2y():
    m = dm.Model("arr2y")
    x = m.continuous("x", shape=(3,), lb=-2, ub=2)
    y = m.continuous("y", lb=-3, ub=3)
    m.subject_to(x[0] * x[1] >= 0.5)
    m.subject_to(x[2] * y >= -1)
    m.minimize(sum((x[i] - 0.3) ** 2 for i in range(3)) + 2 * y**2)
    return m


def _dae():
    from discopt.dae import ContinuousSet, DAEBuilder

    m = dm.Model("dae")
    cs = ContinuousSet("t", bounds=(0, 1), nfe=3, ncp=2)
    dae = DAEBuilder(m, cs)
    dae.add_state("x", initial=1.0, bounds=(-5, 5))
    dae.add_control("u", bounds=(-2, 2))
    dae.set_ode(lambda t, s, a, c: {"x": -(s["x"] ** 3) + c["u"]})
    dae.discretize()
    xv = dae.get_state("x")
    m.minimize(dae.integral(lambda t, s, a, c: c["u"] ** 2) + 10 * (xv[-1, -1] - 0.2) ** 2)
    return m


def _two_var(obj_fn, cubic_con: bool = False):
    m = dm.Model("two")
    x = m.continuous("x", lb=-2, ub=3)
    y = m.continuous("y", lb=-1, ub=2)
    z = m.continuous("z", lb=-1, ub=1)
    m.subject_to(x * y >= -1.5)
    if cubic_con:
        m.subject_to(z**3 + x <= 2)
    m.minimize(obj_fn(x, y, z))
    return m


# ------------------------------------------------------- fixed-box bound checks
def _assert_bound_sound_on_boxes(model, n_boxes=15, n_pts=150, seed=0) -> int:
    """Differential bound test + feasible-point sampling on random sub-boxes.
    Returns the number of executed comparisons (must be > 0)."""
    ev = _evaluator(model)
    lb, ub = solver._flat_var_box(model)
    rng = np.random.default_rng(seed)
    checks = 0
    for k in range(n_boxes):
        if k == 0:
            blo, bhi = lb.copy(), ub.copy()
        else:
            p, q = rng.uniform(lb, ub), rng.uniform(lb, ub)
            blo, bhi = np.minimum(p, q), np.maximum(p, q)
        cvx = solver._convex_objective_lower_bound(ev, blo, bhi)
        assert np.isfinite(cvx)

        def f(x):
            return float(ev.evaluate_objective(np.asarray(x, dtype=float)))

        def g(x):
            return np.asarray(ev.evaluate_gradient(np.asarray(x, dtype=float)), dtype=float)

        res = minimize(
            f, 0.5 * (blo + bhi), jac=g, method="L-BFGS-B", bounds=list(zip(blo, bhi, strict=True))
        )
        # the bound must not exceed the true box optimum (exact for convex f) ...
        assert cvx <= float(res.fun) + 1e-6 * (1.0 + abs(float(res.fun)))
        checks += 1
        # ... nor the objective at any point of the box (feasible-point sampling:
        # every feasible point of a node lies in its box).
        for x in rng.uniform(blo, bhi, size=(n_pts, lb.size)):
            fx = f(x)
            assert cvx <= fx + 1e-6 * (1.0 + abs(fx))
            checks += 1
    return checks


# ============================================================================ #1569
@pytest.mark.parametrize(
    "lb,ub", [(-2.0, 2.0), ([-2.0, -1.0, -3.0], [2.0, 3.0, 1.0])], ids=["uniform", "ragged"]
)
def test_1569_array_variable_gate_engages_without_warning(lb, ub, caplog):
    caplog.set_level(logging.DEBUG, logger="discopt.solver")
    solver._FALLBACK_WARNINGS.seen = set()
    # Fails before #1569: the box was 9 long (uniform) or ragged, the Hessian call
    # raised, and the gate returned False with a misleading warning.
    assert _gate(_issue1569(lb, ub)) is True
    assert _gate_warnings(caplog) == []


def test_1569_issue_repro_solve(caplog):
    caplog.set_level(logging.DEBUG, logger="discopt.solver")
    seen: list[bool] = []
    orig = solver._objective_is_convex_quadratic

    def _wrap(*a, **k):
        r = orig(*a, **k)
        seen.append(r)
        return r

    solver._objective_is_convex_quadratic = _wrap
    try:
        res = _issue1569().solve(time_limit=20)
    finally:
        solver._objective_is_convex_quadratic = orig
    assert seen == [True]
    assert _gate_warnings(caplog) == []
    assert res.status == "optimal"
    assert res.objective == pytest.approx(0.33147187, abs=1e-6)


def test_1569_scalar_and_array_forms_agree():
    """The scalar form always engaged; the array form now does too, and both
    solve identically (bound-neutral on the scalar form)."""
    ra = _issue1569().solve(time_limit=20)
    rs = _issue1569_scalar().solve(time_limit=20)
    assert ra.status == rs.status == "optimal"
    assert ra.node_count == rs.node_count
    assert ra.objective == pytest.approx(rs.objective, abs=1e-8)


def test_1569_box_layout_mismatch_raises():
    """A flat-box/n_vars mismatch is a solver defect: it raises, not a warning."""
    m = _issue1569()
    with pytest.raises(RuntimeError, match="flat variable box"):
        solver._objective_is_convex_quadratic(m, _evaluator(m), 9)


def test_1569_non_numerical_evaluator_error_propagates():
    """Only the evaluator's numerical/domain failures are absorbed; a TypeError
    from the evaluator is a bug and must surface (§3, §7)."""
    m = _issue1569()

    class _Ev:
        def evaluate_hessian(self, _x):
            raise TypeError("bug")

    with pytest.raises(TypeError, match="bug"):
        solver._objective_is_convex_quadratic(m, _Ev(), 3)


def test_1569_differential_bound_on_array_model():
    assert _assert_bound_sound_on_boxes(_issue1569()) > 0
    assert _assert_bound_sound_on_boxes(_issue1569([-2.0, -1.0, -3.0], [2.0, 3.0, 1.0])) > 0


# ============================================================================ #1574
def test_1574_flag_off_keeps_legacy_verdicts(monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    assert _gate(_array_2y()) is False  # scaled square -> general_nl
    assert _gate(_dae()) is False
    assert _gate(_two_var(lambda x, y, z: x**2 + y**2, cubic_con=True)) is False


def test_1574_flag_on_admits_weighted_squares_and_dae(monkeypatch):
    monkeypatch.setenv(_FLAG, "1")
    assert _gate(_array_2y()) is True
    # positive definite on the objective's support (controls + final state) only
    assert _gate(_dae()) is True
    # the constraint cubic no longer rejects a convex quadratic objective
    assert _gate(_two_var(lambda x, y, z: 3 * x**2 + 0.5 * y**2, cubic_con=True)) is True


@pytest.mark.parametrize(
    "obj",
    [
        lambda x, y, z: x**3 + y**2,  # cubic objective
        lambda x, y, z: dm.exp(x) + y**2,  # transcendental
        lambda x, y, z: x * y,  # indefinite quadratic
        lambda x, y, z: 2 * x**2 - y**2,  # indefinite, weighted
        lambda x, y, z: (x - y) ** 2,  # PSD but singular on its support
        lambda x, y, z: 2 * x + y,  # linear
        lambda x, y, z: x**2 / y,  # variable denominator
    ],
    ids=["cubic", "exp", "bilinear", "indefinite", "singular", "linear", "ratio"],
)
def test_1574_flag_on_still_abstains(monkeypatch, obj):
    monkeypatch.setenv(_FLAG, "1")
    assert _gate(_two_var(obj)) is False


def test_1574_maximize_concave_quadratic(monkeypatch):
    """Maximizing a concave weighted quadratic is minimizing a convex one."""
    monkeypatch.setenv(_FLAG, "1")
    m = _two_var(lambda x, y, z: x**2)
    m.maximize(-2 * m._variables[0] ** 2 - 0.5 * m._variables[1] ** 2)
    assert _gate(m) is True
    m.maximize(2 * m._variables[0] ** 2 + 0.5 * m._variables[1] ** 2)
    assert _gate(m) is False


@pytest.mark.parametrize("builder", [_array_2y, _dae], ids=["arr2y", "dae"])
def test_1574_differential_bound_on_admitted_models(monkeypatch, builder):
    monkeypatch.setenv(_FLAG, "1")
    m = builder()
    assert _gate(m) is True
    assert _assert_bound_sound_on_boxes(m) > 0


def test_1574_solve_parity_flag_on_vs_off(monkeypatch):
    """Bound-changing but objective-preserving: both arms certify the same optimum."""
    out = {}
    for flag in ("0", "1"):
        monkeypatch.setenv(_FLAG, flag)
        r = _array_2y().solve(time_limit=20)
        assert r.status == "optimal"
        out[flag] = r.objective
    assert out["1"] == pytest.approx(out["0"], abs=1e-6)
