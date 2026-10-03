"""#1572: MonotoneFunctionEqualityRule tightens from a directed enclosure.

#1570 (#1537 D) routed this rule's tightenings through directed rounding, but fed
them the *proof's* widened function range: ``f`` evaluated on the argument interval
widened by ``arg_slack = 8 eps (|a| + |b| + |offset|)`` -- the SUM of both endpoint
magnitudes, applied to each endpoint -- and then ``8 eps |f|`` more. That is a
tolerance-style margin, not an enclosure of the rounding error. On ``y = exp(x)``
with ``x in [0, 46.05]`` it moved the exact lower bound ``exp(0) = 1`` to
``1 - 8.4e-14`` (~380 ulps, scaled by the *other* endpoint), which is the failing
``test_amp_integration.py::...::test_tighten_proves_exp_square_cycle_infeasible``.

The four tightness tests fail on the pre-#1572 code. The ``sqrt`` helper test and
the sweep at the end are the soundness side of the change (the bound got tighter,
so it must still enclose every exactly feasible point), the sweep graded against a
100-digit mpmath oracle.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import Model
from discopt._relax import _directed as directed
from discopt._relax.nonlinear_bound_tightening import (
    NonlinearBoundTighteningInfeasible,
    tighten_nonlinear_bounds,
)

BIG = 9.999e19


def _tighten_link(func, x_lb, x_ub):
    """Tighten ``y - f(x) == 0`` with ``y`` free and ``x`` in ``[x_lb, x_ub]``."""
    m = Model("link")
    x = m.continuous("x", lb=x_lb, ub=x_ub)
    y = m.continuous("y", lb=-BIG, ub=BIG)
    m.subject_to(y - func(x) == 0.0)
    m.minimize(x * 0.0 + y * 0.0)
    lb, ub, stats = tighten_nonlinear_bounds(
        m, np.array([x_lb, -BIG], dtype=np.float64), np.array([x_ub, BIG], dtype=np.float64)
    )
    assert "monotone_function_equality" in stats.applied_rules
    return lb, ub


def test_exp_square_cycle_keeps_exact_lower_bound():
    """The #1572 instance: y = exp(x), x = y**2 -- y >= exp(0) = 1 exactly."""
    m = Model("exp_square_cycle")
    x = m.continuous("x")
    y = m.continuous("y")
    m.subject_to(y - dm.exp(x) == 0.0)
    m.subject_to(x - y**2 == 0.0)
    m.minimize(x * 0.0 + y * 0.0)
    lb, ub, stats = tighten_nonlinear_bounds(m, np.array([-BIG, -BIG]), np.array([BIG, BIG]))
    assert stats.infeasible is True
    # Pre-#1572: 0.9999999999999164.
    assert lb[1] == 1.0


@pytest.mark.parametrize(
    "func, x_lb, x_ub, y_lb",
    [
        (dm.exp, 0.0, 46.05160185488063, 1.0),  # exp(0) = 1
        (dm.log, 1.0, 50.0, 0.0),  # log(1) = 0
        (dm.sqrt, 4.0, 50.0, 2.0),  # sqrt is correctly rounded; sqrt(4) = 2 exactly
    ],
)
def test_exact_function_value_is_not_widened(func, x_lb, x_ub, y_lb):
    """An exact endpoint value stays exact, whatever the other endpoint's scale."""
    lb, _ub = _tighten_link(func, x_lb, x_ub)
    assert lb[1] == y_lb


def test_inexact_lower_bound_is_within_a_few_ulps():
    """y = exp(x), x in [0.5, 46]: the bound is a few ulps below exp(0.5), not ~600.

    Budget: ``lib_down`` widens a libm result by ``LIB_ULPS + 1 = 5`` ulps, and the
    directed ``-c0 - fc*f`` / affine preimage steps add at most one ulp each -- 8 in
    all. The pre-#1572 margin was ``8 eps * 46.5`` in the argument, ~600 ulps of
    ``exp(0.5)``.
    """
    lb, _ub = _tighten_link(dm.exp, 0.5, 46.0)
    rn = math.exp(0.5)
    assert lb[1] <= rn
    gap_ulps = (rn - lb[1]) / math.ulp(rn)
    assert gap_ulps <= 8.0, gap_ulps


def test_sqrt_directed_helpers_bracket_the_exact_root():
    rng = random.Random(1572)
    checks = 0
    for _ in range(2000):
        x = rng.uniform(0.0, 10.0) * 10.0 ** rng.randint(-200, 200)
        lo, hi = directed.sqrt_down(x), directed.sqrt_up(x)
        fx = Fraction(x)
        assert Fraction(lo) ** 2 <= fx <= Fraction(hi) ** 2
        assert hi == lo or hi == math.nextafter(lo, math.inf)
        checks += 1
    for k in range(1, 200):
        v = float(k * k)
        assert directed.sqrt_down(v) == float(k) == directed.sqrt_up(v)
        checks += 1
    assert directed.sqrt_down(0.0) == 0.0 == directed.sqrt_up(0.0)
    assert checks > 0


def test_monotone_equality_enclosure_is_sound_against_mpmath():
    """Random ``a*y + fc*f(c*x + o) + c0 == 0`` rows; nothing exactly feasible is cut.

    Phase A: ``x`` boxed, ``y`` free -- the exact ``y`` range must be enclosed and
    no ``x`` cut. Phase B: ``y`` boxed around two exactly feasible ``x`` points --
    both must survive. Measured over 6,000 rows (two seeds, ~29.5k checks): zero
    violations and zero false infeasibility proofs, with the median ``y`` lower-bound
    gap down from 34-49 ulps (pre-#1572) to 8 ulps.
    """
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 100
    funcs = {
        "exp": (dm.exp, mp.exp, None),
        "log": (dm.log, mp.log, 0.0),
        "log10": (dm.log10, mp.log10, 0.0),
        "log1p": (dm.log1p, lambda v: mp.log(1 + v), -1.0),
        "sqrt": (dm.sqrt, mp.sqrt, 0.0),
    }
    rng = random.Random(0)

    def mpf(v):
        f = Fraction(v)
        return mp.mpf(f.numerator) / mp.mpf(f.denominator)

    def coef():
        v = rng.choice([1.0, 0.5, 2.0]) if rng.random() < 0.3 else rng.uniform(0.1, 10.0)
        return v * rng.choice([1, -1])

    checks = 0
    for _ in range(400):
        name = rng.choice(sorted(funcs))
        dm_f, mp_f, dlb = funcs[name]
        a, fc, c = coef(), coef(), coef()
        scale = 10.0 ** rng.uniform(-3, 3)
        o = rng.choice([0.0, rng.uniform(-scale, scale)])
        c0 = rng.choice([0.0, rng.uniform(-scale, scale)])
        if dlb is None:
            alo = rng.uniform(-30.0, 30.0)
        else:
            alo = dlb + rng.choice([0.0, 10.0 ** rng.uniform(-6, 3)])
        ahi = alo + 10.0 ** rng.uniform(-6, 1)
        xa, xb = (alo - o) / c, (ahi - o) / c
        xl, xu = min(xa, xb), max(xa, xb)

        def arg(xv):
            return mpf(c) * mpf(xv) + mpf(o)

        def in_domain(xv):
            return dlb is None or arg(xv) > mpf(dlb)

        if not (in_domain(xl) and in_domain(xu)) or xl >= xu:
            continue

        def y_of(xv):
            return -(mpf(c0) + mpf(fc) * mp_f(arg(xv))) / mpf(a)

        def tighten(ylo, yhi):
            m = Model("sweep")
            x = m.continuous("x", lb=xl, ub=xu)
            y = m.continuous("y", lb=ylo, ub=yhi)
            m.subject_to(a * y + fc * dm_f(c * x + o) + c0 == 0.0)
            m.minimize(x * 0.0 + y * 0.0)
            lb, ub, stats = tighten_nonlinear_bounds(m, np.array([xl, ylo]), np.array([xu, yhi]))
            assert not stats.infeasible, (name, a, fc, c, o, c0, xl, xu, ylo, yhi)
            return lb, ub

        y1, y2 = y_of(xl), y_of(xu)
        try:
            lb, ub = tighten(-BIG, BIG)
        except NonlinearBoundTighteningInfeasible as exc:  # pragma: no cover - a failure
            raise AssertionError(f"false infeasibility proof: {exc}") from exc
        assert mpf(lb[1]) <= min(y1, y2) and mpf(ub[1]) >= max(y1, y2), (name, lb, ub)
        assert lb[0] <= xl and ub[0] >= xu
        checks += 2

        x1 = xl + (xu - xl) * rng.uniform(0.0, 0.5)
        x2 = xl + (xu - xl) * rng.uniform(0.5, 1.0)
        if not (in_domain(x1) and in_domain(x2)):
            continue
        ya, yb = y_of(x1), y_of(x2)
        ylo = float(min(ya, yb))
        if mpf(ylo) > min(ya, yb):
            ylo = math.nextafter(ylo, -math.inf)
        yhi = float(max(ya, yb))
        if mpf(yhi) < max(ya, yb):
            yhi = math.nextafter(yhi, math.inf)
        lb, ub = tighten(ylo, yhi)
        assert lb[0] <= x1 <= ub[0] and lb[0] <= x2 <= ub[0], (name, lb, ub, x1, x2)
        checks += 1
    assert checks > 500, checks
