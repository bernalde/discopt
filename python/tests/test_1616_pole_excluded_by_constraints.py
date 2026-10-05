"""#1616 A-04: a denominator the box lets vanish but the linear rows do not.

``n_i / sum(n)`` with ``n >= 0`` has a pole on the variable box, but the element
balances keep ``sum(n)`` in ``[3, 5]``. The no-bound diagnostic used to say "a pole
inside the variable box", which mis-states why the relaxation cannot bound the
objective, and its advice did not lead anywhere. It now reports the LP-implied
range and the reformulation that, measured below, lets the solve certify.
"""

from __future__ import annotations

import logging

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.poles import describe_poles, objective_poles

G0 = np.array([19.492, -192.590, -200.275, 0.0, -395.886]) / (8.314e-3 * 1000)
A = np.array([[1, 0, 1, 0, 1], [4, 2, 0, 2, 0], [0, 1, 1, 0, 2]])
B = A @ np.array([1.0, 2, 0, 0, 0])


def _reformer(aux_lb=None):
    m = dm.Model("reformer")
    n = [m.continuous(f"n{i}", lb=0, ub=10) for i in range(5)]
    if aux_lb is None:
        tot = sum(n)
    else:
        tot = m.continuous("N", lb=aux_lb, ub=50)
        m.subject_to(tot == sum(n))
    m.minimize(sum(n[i] * (G0[i] + dm.log(n[i] / tot)) for i in range(5)))
    for k in range(3):
        m.subject_to(sum(int(A[k, i]) * n[i] for i in range(5)) == B[k])
    return m


def _genuine_pole():
    # x + y == 1 with y in [0, 10] leaves x in [-9, 1]: the pole is reachable.
    m = dm.Model("genuine")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(1.0 / x + y)
    m.subject_to(x + y == 1)
    return m


def test_implied_range_is_reported_for_an_affine_denominator():
    poles = objective_poles(_reformer())
    assert poles, "the box pole must still be detected"
    for p in poles:
        assert p.lo <= 0.0 <= p.hi
        assert p.implied_lo == pytest.approx(3.0, abs=1e-7)
        assert p.implied_hi == pytest.approx(5.0, abs=1e-7)
        assert p.excluded_by_constraints
    text = describe_poles(poles)
    assert "linear constraints keep it in [3, 5]" in text
    # One clause per distinct denominator, not one per term.
    assert text.count("whose range over the variable box") == 1


def test_reachable_pole_is_not_called_excluded():
    poles = objective_poles(_genuine_pole())
    assert len(poles) == 1
    p = poles[0]
    assert p.implied_lo == pytest.approx(-5.0) and p.implied_hi == pytest.approx(1.0)
    assert not p.excluded_by_constraints
    assert "linear constraints keep" not in describe_poles(poles)


def test_nonlinear_denominator_gets_no_implied_range():
    m = dm.Model("nl_den")
    x = m.continuous("x", lb=-2, ub=2)
    m.minimize(1.0 / (x * x - 1.0))
    (p,) = objective_poles(m)
    assert p.implied_lo is None and p.implied_hi is None
    assert not p.excluded_by_constraints


def _pole_warnings(caplog):
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "discopt.solver" and "No valid dual bound" in r.getMessage()
    ]


def test_solve_warning_names_the_implied_range_not_a_box_pole(caplog):
    with caplog.at_level(logging.WARNING, logger="discopt.solver"):
        r = _reformer().solve(time_limit=30)
    assert r.bound is None  # unchanged: the diagnostic changes no bound
    (msg,) = _pole_warnings(caplog)
    assert "linear constraints keep it away from 0" in msg
    assert "pole inside the variable box" not in msg
    assert "[3, 5]" in msg


def test_genuine_pole_keeps_the_pole_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="discopt.solver"):
        r = _genuine_pole().solve(time_limit=30)
    assert r.bound is None  # x -> 0- is feasible, so no bound exists
    (msg,) = _pole_warnings(caplog)
    assert "pole inside the variable box" in msg


def test_the_advised_reformulation_certifies():
    """The advice is measured, not hoped: N with the implied bound certifies."""
    plain = _reformer().solve(time_limit=30)
    lifted = _reformer(aux_lb=3.0).solve(time_limit=30)
    assert lifted.status == "optimal" and lifted.gap_certified
    assert lifted.bound <= lifted.objective + 1e-6
    assert lifted.objective == pytest.approx(plain.objective, rel=1e-6)
