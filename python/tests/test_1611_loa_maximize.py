"""Regression tests for #1611 C-11: GDPopt-LOA under ``maximize``.

The evaluator negates a MAXIMIZE objective, but LOA put the *raw* linear
objective in the master and returned its internal (negated) incumbent and bound.
On a linear objective the master then optimised the wrong direction (synthesis:
``feasible 244.0`` with bound ``-900`` vs the true 406.316); on a nonlinear one
the bound came back with its sign flipped (+2 for an incumbent of -2).

Also covers the stall that followed: when the master re-proposes an integer
configuration whose NLP was already solved, LOA spun to the time limit.
"""

from __future__ import annotations

import time

import discopt.modeling as dm
import pytest

_R = [(0.60, 120.0, 2.0), (0.80, 260.0, 2.5), (0.88, 380.0, 2.4)]
_SP = [(0.95, 150.0, 1.2), (0.90, 60.0, 1.6)]
_SYNTH_OPT = 7720.0 / 19.0  # 406.3157894..., big-m certified


def _toy(maximize: bool):
    m = dm.Model("toy")
    x = m.continuous("x", lb=0, ub=5)
    y = m.continuous("y", lb=0, ub=5)
    m.either_or([[x <= 1, y <= 1], [x >= 3, y >= 3]], name="d")
    f = (x - 2) ** 2 + (y - 2) ** 2
    if maximize:
        m.maximize(-f)
    else:
        m.minimize(f)
    return m


def _synthesis(maximize: bool):
    m = dm.Model("synthesis")
    F = m.continuous("F", lb=0, ub=100)
    P = m.continuous("P", lb=0, ub=100)
    S = m.continuous("S", lb=0, ub=60)
    CR = m.continuous("CR", lb=0, ub=600)
    CS = m.continuous("CS", lb=0, ub=300)
    YR, YS = m.boolean("YR", shape=3), m.boolean("YS", shape=2)
    yr, ys = YR.variable, YS.variable
    m.either_or(
        [[P == a * F, CR >= f + v * F, yr[k] == 1] for k, (a, f, v) in enumerate(_R)],
        name="reactor",
    )
    m.either_or(
        [[S == a * P, CS >= f + v * P, ys[k] == 1] for k, (a, f, v) in enumerate(_SP)],
        name="separator",
    )
    m.exactly(1, [yr[k] for k in range(3)])
    m.exactly(1, [ys[k] for k in range(2)])
    m.logical(YR[1].implies(~YS[1]))
    profit = 30 * S - 9 * F - CR - CS
    if maximize:
        m.maximize(profit)
    else:
        m.minimize(-profit)
    return m


@pytest.mark.parametrize("maximize", [False, True])
def test_loa_nonlinear_objective_bound_on_correct_side(maximize):
    r = _toy(maximize).solve(gdp_method="loa", time_limit=30)
    sign = -1.0 if maximize else 1.0
    assert r.status == "optimal"
    assert r.objective == pytest.approx(sign * 2.0, abs=1e-5)
    assert r.bound is not None
    assert r.bound == pytest.approx(sign * 2.0, abs=1e-5)


@pytest.mark.parametrize("maximize", [False, True])
def test_loa_linear_objective_maximize_reaches_optimum(maximize):
    t0 = time.perf_counter()
    r = _synthesis(maximize).solve(gdp_method="loa", time_limit=30)
    wall = time.perf_counter() - t0
    sign = 1.0 if maximize else -1.0
    assert r.objective is not None
    assert sign * r.objective == pytest.approx(_SYNTH_OPT, rel=1e-6)
    assert r.bound is not None
    # The bound is on the user's side of the optimum: an upper bound for a max.
    assert sign * r.bound == pytest.approx(_SYNTH_OPT, rel=1e-6)
    assert sign * r.bound >= _SYNTH_OPT - 1e-6 * _SYNTH_OPT
    # A re-proposed, already-solved configuration stops the loop instead of
    # re-solving it until the time limit (pre-fix: the full 30 s).
    assert wall < 20.0
