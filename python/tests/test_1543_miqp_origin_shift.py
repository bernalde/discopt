"""MIQP-BB under a translated box certifies the right answer (#1543).

``x = y - c`` with ``|c| >= 1e5`` made the convex MIQP B&B certify wrong answers:
``min 6u^2 - 12u`` over integer ``u = y - c in {0..3}`` certified ``0`` (true -6),
and ``st_miqp3`` certified 15/0 against a true -6. The node QPs were solved in the
model's absolute coordinates, on the expanded ``6y^2 - 1200012y + 6.00012e10``, and
POUNCE returned the root at ``y = 100001.478`` and declared the fixed leaf
``[100001, 100001]`` -- which holds the optimum -- ``primal_infeasible``.

The fix solves every node QP in coordinates whose origin lies in the root box
(``solver._miqp_origin_shift``). These tests fail on the pre-fix tree (measured:
the repro at 1e5/1e6 certifies 0, ``st_miqp3`` at +-1e5/1e6 certifies 15/0/15).
"""

from __future__ import annotations

import discopt.modeling as dm
import discopt.solver as solver_mod
import numpy as np
import pytest
from discopt._relax.problem_classifier import dense_Q, extract_qp_data

TOL = 1e-6


def _certified_at(r, opt):
    tol = TOL * max(1.0, abs(opt))
    assert r.status == "optimal", (r.status, r.objective, r.bound)
    assert r.gap_certified
    assert abs(r.objective - opt) <= tol, (r.objective, opt)
    # The certificate invariant: the dual bound never crosses the optimum.
    assert r.bound <= opt + tol, (r.bound, opt)


def _sq(c, maximize=False):
    m = dm.Model("sq")
    y = m.integer("y", lb=c, ub=3 + c)
    u = y - c
    if maximize:
        m.maximize(-(6 * u * u - 12 * u))
    else:
        m.minimize(6 * u * u - 12 * u)
    return m


def _st_miqp3(c):
    m = dm.Model("st_miqp3")
    y0 = m.integer("y0", lb=c, ub=3 + c)
    y1 = m.integer("y1", lb=c, ub=1e15 + c)
    x0, x1 = y0 - c, y1 - c
    m.subject_to(-4 * x0 + x1 <= 0)
    m.minimize(6 * x0 * x0 - 3 * x1)
    return m


def _st_miqp4(c):
    m = dm.Model("st_miqp4")
    ys = [m.continuous(f"y{i}", lb=c, ub=1e15 + c) for i in range(3)]
    ys += [m.integer(f"y{i}", lb=c, ub=1 + c) for i in range(3, 6)]
    x = [y - c for y in ys]
    m.subject_to(-((x[0] + x[1]) - x[2]) <= 0)
    m.subject_to(x[0] - 5 * x[3] <= 0)
    m.subject_to(x[1] - 10 * x[4] <= 0)
    m.subject_to(x[2] - 30 * x[5] <= 0)
    m.minimize(
        5 * x[0] * x[0]
        + 2 * x[0]
        + 5 * x[1] * x[1]
        + 3 * x[1]
        + 10 * x[2] * x[2]
        - 500 * x[2]
        + 10 * x[3]
        - 4 * x[4]
        + 5 * x[5]
    )
    return m


@pytest.fixture
def shift_spy(monkeypatch):
    """Count the shifts actually applied, so a green run proves the path fired.

    Pins ``DISCOPT_RECENTRE=0``: recentring (default ON since #1537) moves these
    offset boxes to the origin before the MIQP route, so ``_miqp_origin_shift``
    would see an unshifted model and this file would test nothing. The recentred
    arm is covered by test_1543_real_instance_shift.py and test_1537_recentre.py."""
    monkeypatch.setenv("DISCOPT_RECENTRE", "0")
    calls = {"applied": 0}
    real = solver_mod._miqp_origin_shift

    def spy(*a, **k):
        out = real(*a, **k)
        if out is not None:
            calls["applied"] += 1
        return out

    monkeypatch.setattr(solver_mod, "_miqp_origin_shift", spy)
    return calls


@pytest.mark.parametrize("c", [1e4, 1e5, 1e6, -1e6, 1e9])
def test_issue_repro_certifies_minus_six(c, shift_spy):
    _certified_at(_sq(c).solve(time_limit=30), -6.0)
    assert shift_spy["applied"] == 1


@pytest.mark.parametrize("c", [1e5, -1e6])
def test_maximize_sense(c, shift_spy):
    r = _sq(c, maximize=True).solve(time_limit=30)
    assert r.status == "optimal" and r.gap_certified
    assert abs(r.objective - 6.0) <= TOL * 6
    # Maximisation: the bound is an upper bound and must not fall below 6.
    assert r.bound >= 6.0 - TOL * 6
    assert shift_spy["applied"] == 1


def test_fractional_offset_bound_is_valid():
    """``c = 123456.7``: the extracted constant ``6c^2 + 12c`` is itself rounded by
    ~1e-5, which put the bound above the optimum before the shifted constant was
    taken from the model's own expression."""
    m = dm.Model("sqf")
    c = 123456.7
    y = m.integer("y", lb=c, ub=3 + c)  # y in {123457, 123458, 123459}
    u = y - c  # u in {0.3, 1.3, 2.3}
    m.minimize(6 * u * u - 12 * u)
    opt = min(6 * v * v - 12 * v for v in (0.3, 1.3, 2.3))
    _certified_at(m.solve(time_limit=30), opt)


@pytest.mark.parametrize("c", [1e5, 1e6, -1e6])
def test_st_miqp3_shifted(c, shift_spy):
    _certified_at(_st_miqp3(c).solve(time_limit=60), -6.0)
    assert shift_spy["applied"] == 1


@pytest.mark.parametrize("c", [1e5, 1e6])
def test_st_miqp4_shifted(c, shift_spy):
    _certified_at(_st_miqp4(c).solve(time_limit=60), -4574.0)
    assert shift_spy["applied"] == 1


def test_no_shift_when_every_box_contains_zero():
    """Boxes containing 0 leave the data untouched: the path is bit-identical."""
    qp = extract_qp_data(_st_miqp3(0.0))
    assert solver_mod._miqp_origin_shift(qp, np.array([0.0, 0.0]), np.array([3.0, 1e15]), 2) is None


def test_shifted_qp_is_the_same_function():
    """``f_q(x - s) == f(x)`` on random points, with an integral ``s`` in the box."""
    c = 1e5
    m = _st_miqp4(c)
    qp = extract_qp_data(m)
    n = 6
    lb = np.full(n, c)
    ub = np.array([1e15 + c] * 3 + [1 + c] * 3)
    s, qs = solver_mod._miqp_origin_shift(qp, lb, ub, n, model=m)
    assert np.array_equal(s, np.full(n, c))
    Q = dense_Q(qp.Q)[:n, :n]
    rng = np.random.default_rng(1543)
    checks = 0
    for _ in range(20):
        d = rng.uniform(0.0, 3.0, n)
        x = s + d
        u = x - c  # the model's own affine argument, formed exactly here
        f_true = (
            5 * u[0] ** 2 + 2 * u[0] + 5 * u[1] ** 2 + 3 * u[1] + 10 * u[2] ** 2 - 500 * u[2]
        ) + (10 * u[3] - 4 * u[4] + 5 * u[5])
        f_q = float(qs.c[:n] @ d + 0.5 * d @ Q @ d + qs.obj_const)
        assert abs(f_q - f_true) <= 1e-9 * (1 + abs(f_true))
        checks += 1
    assert checks == 20


def test_overflowing_shift_declines_instead_of_raising():
    """A shift whose exact products overflow returns ``None`` (the unshifted data
    are then used, as before #1543)."""
    qp = extract_qp_data(_sq(0.0))
    for s in (1e300, -1e300):
        assert solver_mod._miqp_origin_shift(qp, np.array([s]), np.array([s]), 1) is None


def test_exact_sum_mixed_infinities_is_nan_not_valueerror():
    """``math.fsum`` raises on +inf and -inf together; the helper must hand the
    non-finite result to the caller's overflow check instead."""
    assert np.isnan(solver_mod._exact_sum([np.inf], [-np.inf], [1.0]))
    assert solver_mod._exact_sum([1e16], [1.0], [-1e16]) == 1.0
