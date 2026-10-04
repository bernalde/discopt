"""Regression tests for the items of #1618 left open after PR #1641.

Each test reproduces the issue's own instance (or the smallest model showing
the same class of defect) and asserts the fixed behaviour.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# B-21a / B-19b: constraint duals for vector / family / builder rows
# ---------------------------------------------------------------------------


def _lp_reference_duals():
    # min x0 + x1  s.t.  x0 + 2 x1 >= 4,  3 x0 + x1 >= 5:  y = (0.4, 0.2).
    return np.array([[1.0, 2.0], [3.0, 1.0]]), np.array([4.0, 5.0]), np.array([0.4, 0.2])


def test_b21_vector_constraint_duals_reported():
    A, b, y = _lp_reference_duals()
    m = dm.Model("v")
    x = m.continuous("x", shape=(2,), lb=0, ub=10)
    m.minimize(x[0] + x[1])
    m.subject_to(A @ x >= b, name="Ax")
    r = m.solve()
    assert r.status == "optimal"
    assert r.constraint_duals is not None
    np.testing.assert_allclose(r.constraint_duals["Ax"], y, atol=1e-7)
    assert r.constraint_duals["Ax"].shape == (2,)


def test_b21_family_constraint_duals_reported():
    m = dm.Model("f")
    idx = m.set("I", [0, 1])
    x = m.continuous("x", shape=(2,), lb=0, ub=10)
    m.minimize(x[0] + x[1])
    m.subject_to(x[0] + 2 * x[1] >= 4, name="r1")
    m.constraint(idx, lambda i: x[i] >= 0.5, name="fam")
    r = m.solve()
    assert r.status == "optimal"
    assert r.constraint_duals is not None
    # Reference: x = (0.5, 1.75); r1 binds with 0.5, fam_0 with 0.5, fam_1 slack.
    assert float(r.constraint_duals["r1"]) == pytest.approx(0.5, abs=1e-7)
    assert float(r.constraint_duals["fam_0"]) == pytest.approx(0.5, abs=1e-7)
    assert float(r.constraint_duals["fam_1"]) == pytest.approx(0.0, abs=1e-7)


def test_b21_bulk_linear_constraint_duals_reported():
    A, b, y = _lp_reference_duals()
    m = dm.Model("bulk")
    z = m.continuous("z", shape=(2,), lb=0, ub=10)
    m.add_linear_constraints(A, z, ">=", b, name="Ab")
    m.add_linear_objective(np.ones(2), z)
    r = m.solve()
    assert r.status == "optimal"
    assert r.constraint_duals is not None
    assert float(r.constraint_duals["Ab_0"]) == pytest.approx(y[0], abs=1e-7)
    assert float(r.constraint_duals["Ab_1"]) == pytest.approx(y[1], abs=1e-7)
    assert r.bound_duals_lower is not None


def test_b21_vector_equality_dual_sign():
    # min x0 + 2 x1  s.t. [x0 + x1, x0 - x1] == [2, 0]  ->  x = (1, 1).
    # Stationarity c + J^T lam = 0 with J = [[1, 1], [1, -1]] gives lam = (-1.5, 0.5).
    m = dm.Model("veq")
    x = m.continuous("x", shape=(2,), lb=-10, ub=10)
    m.minimize(x[0] + 2 * x[1])
    M = np.array([[1.0, 1.0], [1.0, -1.0]])
    m.subject_to(M @ x == np.array([2.0, 0.0]), name="E")
    r = m.solve()
    assert r.status == "optimal"
    assert r.constraint_duals is not None
    lam = r.constraint_duals["E"]
    # Whatever the sign convention, it must agree with the scalar route's.
    ms = dm.Model("seq")
    xs = ms.continuous("x", shape=(2,), lb=-10, ub=10)
    ms.minimize(xs[0] + 2 * xs[1])
    ms.subject_to(xs[0] + xs[1] == 2, name="e0")
    ms.subject_to(xs[0] - xs[1] == 0, name="e1")
    rs = ms.solve()
    assert rs.constraint_duals is not None
    ref = np.array([float(rs.constraint_duals["e0"]), float(rs.constraint_duals["e1"])])
    np.testing.assert_allclose(np.abs(ref), [1.5, 0.5], atol=1e-7)
    np.testing.assert_allclose(lam, ref, atol=1e-7)


# ---------------------------------------------------------------------------
# B-18: last iterate of a failed continuous solve is reported
# ---------------------------------------------------------------------------


def test_b18_last_iterate_on_iteration_limit():
    m = dm.Model("n")
    x = m.continuous("x", shape=(2,), lb=-3, ub=3)
    m.minimize((1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2)
    m.subject_to(x[0] ** 2 + x[1] ** 2 == 2)
    m.subject_to(dm.exp(x[0]) - x[1] <= 1.5)
    r = m.solve(
        solver="pounce",
        pounce_options={"max_iter": 2},
        initial_solution={x: np.array([2.5, -2.0])},
    )
    assert r.status == "iteration_limit"
    assert r.x is None  # no certified/feasible point is claimed
    assert r.last_iterate is not None and "x" in r.last_iterate
    assert r.last_iterate["x"].shape == (2,)
    assert r.last_iterate_violation is not None and r.last_iterate_violation > 1e-3


# ---------------------------------------------------------------------------
# C-17: presolve-infeasible provenance and element-level IIS
# ---------------------------------------------------------------------------


def test_c17_presolve_infeasible_termination_and_route():
    from discopt.status import TERMINATION_PRESOLVE, TERMINATION_REASONS

    assert TERMINATION_PRESOLVE in TERMINATION_REASONS
    m = dm.Model("inf")
    x = m.integer("x", shape=(3,), lb=0, ub=5)
    y = m.binary("y")
    m.subject_to(x[0] + x[1] >= 8 + y)
    m.subject_to(x[0] <= 3)
    m.subject_to(x[1] <= 4)
    m.minimize(x[0] + x[1] + x[2])
    r = m.solve()
    assert r.status == "infeasible"
    assert r.termination == TERMINATION_PRESOLVE
    assert r.algorithm_route is not None and r.algorithm_route.startswith("presolve")


def test_c17_iis_names_bound_elements():
    m = dm.Model("iis")
    z = m.continuous("z", shape=(3,), lb=0, ub=1)
    m.subject_to(z[0] + z[1] >= 3, name="need3")
    m.minimize(z[2])
    iis = m.compute_iis()
    assert [c.name for c in iis.constraints] == ["need3"]
    elems = sorted((v.name, side, idx) for v, side, idx in iis.bound_elements)
    assert elems == [("z", "upper", (0,)), ("z", "upper", (1,))]
    assert len(iis) == 3
    text = iis.summary()
    assert "z[0] <= 1" in text and "z[1] <= 1" in text and "z[2]" not in text


# ---------------------------------------------------------------------------
# D-34: NBI on a discrete model warns instead of silently thinning the front
# ---------------------------------------------------------------------------


def test_d34_nbi_warns_on_discrete_model():
    from discopt import mo

    Q = np.array([15.0, 10.0, 5.0, 20.0])
    C = np.array([3.2, 3.4, 1.9, 7.0])
    E = np.array([26.0, 2.5, 4.0, 0.5])
    m = dm.Model("heat")
    n = m.integer("n", shape=(4,), lb=0, ub=[4, 3, 4, 2])
    cost = dm.sum(lambda j: C[j] * n[j], over=range(4))
    co2 = dm.sum(lambda j: E[j] * n[j], over=range(4))
    m.subject_to(dm.sum(lambda j: Q[j] * n[j], over=range(4)) >= 55.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mo.normal_boundary_intersection(m, [cost, co2], n_points=7)
    msgs = [str(x.message) for x in w]
    assert any("integer/binary variables" in s for s in msgs), msgs


# ---------------------------------------------------------------------------
# B-10: weakly active bound flagged as a degenerate active set
# ---------------------------------------------------------------------------


def test_b10_weakly_active_bound_is_flagged():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    a = np.array([0.04, 0.03, 0.05])
    pmin = np.array([50.0, 30, 20])
    pmax = np.array([300.0, 200, 150])
    d0 = (32 - 22) / a[0] + (32 - 27) / a[1] + pmin[2]
    m = dm.Model("d")
    b2 = m.parameter("b2", 27.0)
    b3 = m.parameter("b3", 31.0)
    D = m.parameter("D", d0)
    P = m.continuous("P", shape=(3,), lb=pmin, ub=pmax)
    m.subject_to(P[0] + P[1] + P[2] == D)
    m.minimize(sum(0.5 * a[i] * P[i] ** 2 for i in range(3)) + 22 * P[0] + b2 * P[1] + b3 * P[2])
    phi = dm.argmin_layer(m, [b2, b3, D], options={"tol": 1e-10})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        jax.jacobian(phi)(jnp.array([27.0, 31.0, d0]))
    assert any("degenerate active set" in str(x.message) for x in w)
    info = phi.last_solve_info()
    assert info["degenerate"] is True
    assert 2 in info["weakly_active_bounds"]


# ---------------------------------------------------------------------------
# X-29a / X-29b: stochastic structure and CVaR auxiliaries
# ---------------------------------------------------------------------------


def _plant(name):
    c1 = np.array([2.0, 3.0])
    r = np.array([5.0, 6.0])
    m = dm.Model(name)
    x = m.continuous("x", shape=(2,), lb=0, ub=200)

    def rb(model, data, s):
        y = model.continuous(f"y_{s}", shape=(2,), lb=0, ub=200)
        for g in range(2):
            model.subject_to(y[g] <= x[g])
            model.subject_to(y[g] <= float(data["d"][g]))
        return dm.sum(lambda g: float(-r[g]) * y[g], over=range(2))

    return m, x, rb, dm.sum(lambda i: float(c1[i]) * x[i], over=range(2))


def _scenarios():
    from discopt.stochastic import ScenarioSet

    D = np.array([[60.0, 40.0], [100.0, 70.0], [140.0, 90.0]])
    return ScenarioSet.from_list([(p, {"d": d}) for p, d in zip([0.3, 0.4, 0.3], D)])


def test_x29a_lshaped_structure_reports_scenario_blocks():
    from discopt.stochastic import solve_lshaped

    m, x, rb, fc = _plant("ls")
    ls = solve_lshaped(
        m, first_stage_vars=[x], scenarios=_scenarios(), recourse_builder=rb, first_stage_cost=fc
    )
    assert ls.result.status == "optimal"
    assert len(ls.structure.blocks) == 3
    assert ls.structure.is_separable


def test_x29b_cvar_auxiliaries_do_not_trigger_large_bound_warning():
    from discopt.stochastic import CVaR, build_extensive_form

    m, x, rb, fc = _plant("cvar")
    build_extensive_form(
        m, scenarios=_scenarios(), recourse_builder=rb, first_stage_cost=fc, risk=CVaR(0.9)
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = m.solve()
    assert res.status == "optimal"
    assert not [str(x.message) for x in w if "large" in str(x.message).lower()]
