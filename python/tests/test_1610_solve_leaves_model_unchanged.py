"""``solve()`` must leave the caller's model observably unchanged (#1610).

The solve path uses the caller's ``Model`` as its working copy: root FBBT writes
the box it tightened into ``v.lb``/``v.ub``, ``DISCOPT_COEF_TIGHTEN`` replaces
big-M rows in ``model._constraints``, and the structure-cut presolve appends
auxiliary columns and rows. Before the fix every one of those survived the solve.
The C2 repro is the wrong-answer witness: FBBT derived ``x <= 8`` from
``x <= p`` at ``p = 8`` and left it as the *declared* bound, so a re-solve at
``p = 10`` returned ``x = 8`` labelled ``optimal``.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import _solve_owns_model
from discopt.solver import solve_model


def _parameter_model():
    m = dm.Model("r")
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", 8.0)
    m.maximize(x / (4 + x))
    m.subject_to(x <= p)
    return m, x, p


@pytest.mark.smoke
def test_parameter_resolve_is_not_served_a_stale_box():
    m, x, p = _parameter_model()
    r = m.solve()
    assert r.status == "optimal"
    assert r.value(x) == pytest.approx(8.0, abs=1e-6)
    # The declared bound, not the FBBT-tightened one.
    assert float(x.ub) == 20.0 and float(x.lb) == 0.0
    p.value = 10.0
    r = m.solve()
    assert r.status == "optimal"
    assert r.value(x) == pytest.approx(10.0, abs=1e-6)


def test_direct_solve_model_call_restores_bounds():
    m, x, _ = _parameter_model()
    solve_model(m)
    assert float(x.ub) == 20.0


def _bottling(T=10):
    d = [12, 18, 0, 25, 30, 8, 0, 22, 35, 15.0]
    C = [40, 40, 40, 20, 40, 40, 40, 40, 40, 40.0]
    m = dm.Model("bottling")
    x = [m.continuous(f"x{t}", lb=0, ub=1000) for t in range(1, T + 1)]
    s = [m.continuous(f"s{t}", lb=0, ub=1000) for t in range(T + 1)]
    y = [m.binary(f"y{t}") for t in range(1, T + 1)]
    m.minimize(dm.sum(lambda t: 5 * y[t] + 0.2 * s[t + 1], over=range(T)))
    m.subject_to(s[0] == 10)
    for t in range(T):
        m.subject_to(s[t] + x[t] - s[t + 1] == d[t])
        m.subject_to(x[t] <= C[t])
        m.subject_to(x[t] <= 1000 * y[t], name=f"co{t + 1}")
    return m


def test_coef_tighten_does_not_rewrite_the_users_rows(monkeypatch):
    monkeypatch.setenv("DISCOPT_COEF_TIGHTEN", "1")
    m = _bottling()
    lp_before = m.to_lp()
    cons_before = list(m._constraints)
    bounds_before = [(np.array(v.lb), np.array(v.ub)) for v in m._variables]
    r1 = m.solve()
    assert r1.status == "optimal"
    # The pass did fire (the root bound is the tightened one, 22.635 vs 1.775
    # untightened), so the unchanged model below is not a vacuous pass.
    assert r1.root_bound is not None and r1.root_bound > 20.0
    assert m.to_lp() == lp_before
    assert all(a is b for a, b in zip(m._constraints, cons_before))
    assert len(m._constraints) == len(cons_before)
    for v, (lb, ub) in zip(m._variables, bounds_before):
        assert np.array_equal(v.lb, lb) and np.array_equal(v.ub, ub)
    # And the second solve is the same solve as the first.
    r2 = m.solve()
    assert r2.status == r1.status
    assert r2.objective == pytest.approx(r1.objective, abs=1e-9)
    assert r2.root_bound == pytest.approx(r1.root_bound, abs=1e-9)


def test_owner_restores_every_kind_of_write_even_on_error():
    m = dm.Model("w")
    x = m.continuous("x", lb=0, ub=5)
    m.subject_to(x <= 4)
    m.minimize(x)
    c = m._constraints[0]
    cons, obj, names = list(m._constraints), m._objective, set(m._names)
    with pytest.raises(RuntimeError, match="boom"):
        with _solve_owns_model(m) as owner:
            assert owner
            x.ub = 1.0
            m._constraints[0] = dm.Model("o").continuous("z") <= 1  # replaced row
            m.continuous("aux", lb=0, ub=1)  # appended column
            m.subject_to(x >= 0)  # appended row
            m.maximize(x)  # replaced objective
            raise RuntimeError("boom")
    assert float(x.ub) == 5.0
    assert len(m._constraints) == 1 and m._constraints[0] is c and cons[0] is c
    assert m._objective is obj
    assert [v.name for v in m._variables] == ["x"]
    assert m._names == names
    m.continuous("aux")  # the name is free again


def test_nested_entry_on_an_owned_model_is_part_of_the_outer_solve():
    m = dm.Model("n")
    x = m.continuous("x", lb=0, ub=5)
    with _solve_owns_model(m) as outer:
        assert outer
        with _solve_owns_model(m) as inner:
            assert not inner
            x.ub = 2.0
        # The inner entry must not undo the outer solve's working state.
        assert float(x.ub) == 2.0
    assert float(x.ub) == 5.0


def test_sensitivity_reference_stays_live_after_restore():
    # The result's problem fingerprint is stamped inside the solve; after the
    # caller's box is restored it must describe the caller's problem, or
    # sensitivity() would discard a live reference as stale.
    from discopt._evaluator_cache import solution_state_fingerprint

    m, x, _ = _parameter_model()
    r = m.solve()
    fp = getattr(r, "_problem_fingerprint", None)
    assert fp is not None
    assert fp == solution_state_fingerprint(m)
