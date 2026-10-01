"""``Model.solve(solver="pounce")``: one POUNCE interior-point solve, nothing else (#1533).

The route sends an LP to POUNCE's convex LP IPM (``lp-ipm``), a PSD QP to its
``qp-ipm``, and every other continuous model to the filter line-search NLP IPM
once. Pinned here:

* which engine each class reaches, and that nothing else does (no HiGHS, no
  convexity classification, no second NLP solve);
* certificate honesty: LP/convex QP report ``optimal``; the NLP arm reports
  ``local_optimal`` with no bound, and an engine's infeasible/unbounded verdict
  is only certified once verified;
* ``pounce_options`` reaches POUNCE, and an option the convex engine lacks is
  refused rather than dropped;
* the loud refusals (integers, GDP constraints, callbacks).
"""

from __future__ import annotations

import discopt.modeling as dm
import discopt.solver as solver_mod
import numpy as np
import pounce.qp
import pytest
from discopt.solvers import convex_ipm_pounce

pytestmark = [pytest.mark.requires_pounce]


def _lp():
    m = dm.Model("lp")
    x = m.continuous("x", shape=2, lb=0, ub=10)
    m.minimize(-x[0] - 2 * x[1])
    m.subject_to(x[0] + x[1] <= 4)
    m.subject_to(x[0] - x[1] == 1)
    return m


def _convex_qp():
    m = dm.Model("qp")
    x = m.continuous("x", shape=2, lb=-5, ub=5)
    m.minimize(x[0] ** 2 + x[1] ** 2 - 3 * x[0] - 4 * x[1])
    m.subject_to(x[0] + x[1] <= 1)
    return m


def _double_well():
    """Nonconvex NLP with a local minimum near x=+1 and the global one near x=-1."""
    m = dm.Model("well")
    x = m.continuous("x", lb=-3, ub=3)
    y = m.continuous("y", lb=-3, ub=3)
    m.minimize((x**2 - 1) ** 2 + y**2 + 0.3 * x)
    m.subject_to(x + y >= -2)
    return m, x, y


@pytest.fixture
def spy_convex(monkeypatch):
    calls: list[dict] = []
    real = pounce.qp.solve_qp

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(pounce.qp, "solve_qp", spy)
    return calls


@pytest.fixture
def spy_nlp(monkeypatch):
    from discopt.solvers import nlp_pounce

    calls: list[dict] = []
    real = nlp_pounce.solve_nlp

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(nlp_pounce, "solve_nlp", spy)
    return calls


@pytest.fixture
def no_other_route(monkeypatch):
    """Anything outside the route that a default solve would call raises."""

    def boom(*_a, **_k):
        raise AssertionError("solver='pounce' reached a non-POUNCE route")

    for name in ("_solve_lp_highs", "_solve_lp", "_solve_qp", "_classify_model_convexity"):
        monkeypatch.setattr(solver_mod, name, boom)


# ── routing ──────────────────────────────────────────────────────────────────


@pytest.mark.smoke
def test_lp_reaches_lp_ipm_and_matches_default(spy_convex, spy_nlp, no_other_route):
    res = _lp().solve(solver="pounce")
    assert res.algorithm_route == "pounce:lp-ipm"
    assert len(spy_convex) == 1 and spy_convex[0]["P"] is None  # P=None: the LP engine
    assert spy_convex[0]["method"] == "ipm"
    assert spy_nlp == []
    assert res.status == "optimal" and res.gap_certified
    assert res.objective == pytest.approx(-5.5, abs=1e-7)
    assert res.solver_stats["pounce/iterations"] >= 1
    # Duals are in discopt's convention, identical to the default route's.
    assert res.constraint_duals["c0"] == pytest.approx(1.5, abs=1e-6)
    assert res.constraint_duals["c1"] == pytest.approx(-0.5, abs=1e-6)


def test_convex_qp_reaches_qp_ipm(spy_convex, spy_nlp, no_other_route):
    res = _convex_qp().solve(solver="pounce")
    assert res.algorithm_route == "pounce:qp-ipm"
    assert len(spy_convex) == 1 and spy_convex[0]["P"] is not None
    assert spy_nlp == []
    assert res.status == "optimal"
    # min (x0-1.5)^2 + (x1-2)^2 - 6.25 on x0 + x1 <= 1: x = (0.25, 0.75).
    np.testing.assert_allclose(res.x["x"], [0.25, 0.75], atol=1e-6)
    assert res.objective == pytest.approx(-3.125, abs=1e-6)


def test_indefinite_qp_is_solved_locally_by_the_nlp_ipm(spy_nlp, no_other_route):
    m = dm.Model("ncqp")
    x = m.continuous("x", shape=2, lb=-5, ub=5)
    m.minimize(x[0] ** 2 - x[1] ** 2)
    m.subject_to(x[0] + x[1] <= 1)
    res = m.solve(solver="pounce")
    assert res.algorithm_route == "pounce:nlp"
    assert len(spy_nlp) == 1
    assert res.status == "local_optimal"
    assert res.bound is None and not res.gap_certified


@pytest.mark.smoke
def test_nonconvex_nlp_is_one_local_ipm_call(spy_nlp, no_other_route):
    m, x, y = _double_well()
    res = m.solve(solver="pounce", initial_solution={x: 2.0, y: 0.0})
    assert len(spy_nlp) == 1
    assert res.algorithm_route == "pounce:nlp"
    # The local minimum near x=+1, not the global one near x=-1 (obj ~ -0.305):
    # the route did not search, and it does not claim otherwise.
    assert res.status == "local_optimal"
    assert res.x["x"] == pytest.approx(0.96, abs=1e-2)
    assert res.objective == pytest.approx(0.2941, abs=1e-3)
    assert res.bound is None and res.gap is None and not res.gap_certified


def test_nlp_route_does_not_tighten_the_declared_box(monkeypatch, spy_nlp):
    def boom(*_a, **_k):
        raise AssertionError("bound tightening ran on solver='pounce'")

    monkeypatch.setattr(solver_mod, "_apply_nonlinear_tightening_with_status", boom)
    m, _x, _y = _double_well()
    assert m.solve(solver="pounce").status == "local_optimal"


# ── options ──────────────────────────────────────────────────────────────────


def test_pounce_options_reach_the_nlp_engine(spy_nlp):
    m, _x, _y = _double_well()
    opts = {"print_level": 0, "mu_strategy": "adaptive", "tol": 1e-9}
    m.solve(solver="pounce", pounce_options=opts)
    sent = spy_nlp[0]["options"]
    assert sent["mu_strategy"] == "adaptive" and sent["tol"] == 1e-9
    assert sent["print_level"] == 0


def test_pounce_options_reach_the_convex_engine(spy_convex, capsys):
    _lp().solve(solver="pounce", pounce_options={"tol": 1e-10, "max_iter": 50, "print_level": 1})
    assert spy_convex[0]["tol"] == 1e-10 and spy_convex[0]["max_iter"] == 50
    assert spy_convex[0]["collect_iterates"] is True
    assert "lp-ipm" in capsys.readouterr().out  # print_level>0 prints the trace


def test_nlp_engine_option_on_an_lp_is_refused():
    with pytest.raises(ValueError, match="mu_strategy"):
        _lp().solve(solver="pounce", pounce_options={"mu_strategy": "adaptive"})


def test_pounce_options_is_an_alias_on_the_default_route_too(spy_nlp):
    m, _x, _y = _double_well()
    m.solve(pounce_options={"tol": 1e-9, "print_level": 0}, time_limit=20)
    assert spy_nlp and all(c["options"]["tol"] == 1e-9 for c in spy_nlp)


def test_conflicting_aliases_are_refused():
    with pytest.raises(ValueError, match="pounce_options and ipopt_options"):
        _lp().solve(solver="pounce", pounce_options={"tol": 1e-8}, ipopt_options={"tol": 1e-6})


# ── refusals ─────────────────────────────────────────────────────────────────


@pytest.mark.smoke
def test_integer_variables_are_refused():
    m = dm.Model("milp")
    x = m.integer("x", lb=0, ub=3)
    m.minimize(x)
    with pytest.raises(ValueError, match="integer or binary"):
        m.solve(solver="pounce")


def test_disjunctive_constraints_are_refused():
    m = dm.Model("gdp")
    x = m.continuous("x", lb=0, ub=10)
    m.minimize(x)
    m.either_or([[x <= 2], [x >= 8]])
    with pytest.raises(ValueError, match="logical/disjunctive"):
        m.solve(solver="pounce")


def test_feasibility_callbacks_are_refused():
    with pytest.raises(ValueError, match="pounce"):
        _lp().solve(solver="pounce", lazy_constraints=lambda *a: [])


# ── certificates ─────────────────────────────────────────────────────────────


def test_infeasible_lp_is_certified_after_verification():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=1)
    m.minimize(x)
    m.subject_to(x >= 2)
    assert m.solve(solver="pounce").status == "infeasible"


def test_unverified_infeasibility_is_not_certified(monkeypatch):
    """POUNCE's own verdict alone never becomes a certificate."""
    monkeypatch.setattr(convex_ipm_pounce, "_simplex_feasibility_verdict", lambda *a: "undecided")
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=1)
    m.minimize(x)
    m.subject_to(x >= 2)
    res = m.solve(solver="pounce")
    assert res.status == "error" and res.error


def test_unbounded_lp_with_infinite_bounds():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=np.inf)
    y = m.continuous("y", lb=0, ub=np.inf)
    m.minimize(-x)
    m.subject_to(x - y <= 1)
    assert m.solve(solver="pounce").status == "unbounded"


def test_unbounded_over_the_default_box_is_not_certified():
    """The 9.999e19 default box is finite as declared; 'unbounded' would be false."""
    m = dm.Model()
    x = m.continuous("x", lb=0)
    m.minimize(-x)
    m.subject_to(x >= 1)
    res = m.solve(solver="pounce")
    assert res.status == "error"
    assert "9.999e19" in res.error
