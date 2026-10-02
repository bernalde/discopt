"""#1537 D: Python-side bound tightening must round its endpoints outward.

Every witness here was found by the #1537 D entry experiment and fails on the code
before the fix:

* ``_tighten_affine_upper_bound`` summed a row's other-term minimum in
  round-to-nearest and divided in round-to-nearest. On cancelling large-magnitude
  rows (Fraction oracle, 20,000 random rows) it cut 33,571 exactly-feasible
  extremes (worst by 22.2 units) and raised 1,080 false infeasibility proofs.
* ``_tighten_affine_argument_interval`` computed ``(arg_lb - offset) / coeff`` in
  round-to-nearest and then ``ceil(lb - 1e-9)``. One rounding ulp of a 4e8-magnitude
  quotient (6e-8) is far above the 1e-9 integrality slack, so the only feasible
  integer of ``R2 <= exp(a*x + b) <= R1`` was cut: 361 of 4,000 generated two-sided
  cases, and every one of six solved end to end returned a certified
  ``infeasible``.

The data below were generated so that the named integer ``K`` satisfies the float
model exactly (``R1``/``R2`` are the floats just above/below ``exp(a*K + b)``
evaluated in 200-digit mpmath), and a brute-force scan of ``[K-50, K+50]`` confirmed
``K`` is the only feasible integer.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction
from types import SimpleNamespace

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import Model
from discopt._relax import _directed as directed
from discopt._relax.nonlinear_bound_tightening import (
    _tighten_affine_upper_bound,
    tighten_nonlinear_bounds,
)
from discopt.modeling.core import VarType

# (K, a, b, R1, R2): exp(a*K + b) lies in [R2, R1] exactly; K is the only feasible
# integer in [K-50, K+50]. All six returned a certified ``infeasible`` before #1537 D.
EXP_WITNESSES = [
    (
        416125107,
        6.2896846837200755,
        -2617295718.427276,
        0.0016319191567007768,
        0.0016319191567007766,
    ),
    (505944117, 9.267730587439564, -4688953775.961391, 0.0006719074177630091, 0.000671907417763009),
    (
        15870570426,
        1.255606029412651,
        -19927183923.89504,
        0.001123469301797278,
        0.0011234693017972777,
    ),
    (
        425686142,
        1.3348616851705493,
        -568232126.9955553,
        0.002172915305023316,
        0.0021729153050233157,
    ),
    (
        2061093589,
        4.997815172307458,
        -10300964817.943913,
        0.0006795497732232149,
        0.0006795497732232148,
    ),
    (
        228931874,
        5.322365402019699,
        -1218459092.7407029,
        0.0007899271604339974,
        0.0007899271604339973,
    ),
]


def _two_sided_exp_model(K, a, b, R1, R2):
    m = Model("issue_1537_d_exp")
    x = m.integer("x", lb=K - 50, ub=K + 50)
    m.subject_to(dm.exp(a * x + b) <= R1)
    m.subject_to(dm.exp(a * x + b) >= R2)
    m.minimize(x - float(K + 5))
    return m


def test_directed_helpers_enclose_the_exact_result():
    """Each ``*_down``/``*_up`` brackets the exact rational result (Fraction oracle)."""
    rng = random.Random(15375)
    ops = {
        "add": (directed.add_down, directed.add_up, lambda p, q: p + q),
        "sub": (directed.sub_down, directed.sub_up, lambda p, q: p - q),
        "mul": (directed.mul_down, directed.mul_up, lambda p, q: p * q),
        "div": (directed.div_down, directed.div_up, lambda p, q: p / q),
    }
    checked = 0
    for _ in range(4000):
        a = rng.choice([-1, 1]) * 10 ** rng.uniform(-12, 12)
        b = rng.choice([-1, 1]) * 10 ** rng.uniform(-12, 12)
        for down, up, exact in ops.values():
            ref = exact(Fraction(a), Fraction(b))
            lo, hi = down(a, b), up(a, b)
            assert Fraction(lo) <= ref <= Fraction(hi)
            # True directed rounding: the bracket is one float wide, or a point
            # when the result is exact.
            assert hi == lo if Fraction(lo) == ref else hi == math.nextafter(lo, math.inf)
            checked += 1
    assert checked == 16000


def test_exact_results_are_not_widened():
    """Exact operations stay exact, so well-scaled bounds do not move (cf. #1415)."""
    assert directed.sub_down(0.0, 0.0) == 0.0 == directed.sub_up(0.0, 0.0)
    assert directed.mul_down(2.0, 0.5) == 1.0 == directed.mul_up(2.0, 0.5)
    assert directed.div_down(6.0, 3.0) == 2.0 == directed.div_up(6.0, 3.0)
    assert directed.add_down(1e16, 2.0) == 1e16 + 2.0 == directed.add_up(1e16, 2.0)
    assert directed.affine_preimage(2.0, 1.0, 5.0, lower=True) == 2.0
    assert directed.affine_preimage(-2.0, 1.0, 5.0, lower=False) == -2.0
    # Inexact: 1/3 is bracketed by adjacent floats.
    assert directed.div_up(1.0, 3.0) == math.nextafter(directed.div_down(1.0, 3.0), math.inf)


def test_affine_preimage_encloses_every_sign_combination():
    """``affine_preimage`` bounds ``(target - offset) / coeff`` on the requested side."""
    rng = random.Random(15376)
    checked = 0
    for _ in range(4000):
        coeff = rng.choice([-1, 1]) * 10 ** rng.uniform(-3, 3)
        offset = rng.choice([-1, 1]) * 10 ** rng.uniform(0, 12)
        target = rng.choice([-1, 1]) * 10 ** rng.uniform(0, 12)
        ref = (Fraction(target) - Fraction(offset)) / Fraction(coeff)
        assert Fraction(directed.affine_preimage(coeff, offset, target, lower=True)) <= ref
        assert Fraction(directed.affine_preimage(coeff, offset, target, lower=False)) >= ref
        checked += 2
    assert checked == 8000


def test_lib_widening_keeps_non_finite_values():
    assert directed.lib_up(math.inf) == math.inf
    assert directed.lib_down(-math.inf) == -math.inf
    assert directed.lib_down(1.0) < 1.0 < directed.lib_up(1.0)


def test_affine_upper_bound_never_proves_a_feasible_row_infeasible():
    """A cancelling row whose min-activity point is feasible; it raised before."""
    coeffs = [
        -0.0010780775286668607,
        -10.230624477227593,
        0.0018556414061995596,
        0.16590897458405204,
        -3.23911098347633,
    ]
    lo = [
        141931644.15074503,
        -464795841.44885886,
        60387125.78047192,
        -347864124.8248886,
        142482904.38224912,
    ]
    hi = [
        141931646.5307412,
        -464727652.5488638,
        60387321.64228982,
        -347842407.3187702,
        142483063.47561106,
    ]
    rhs = 4235180904.835712
    # The exact min-activity point satisfies the row: the row is feasible.
    point = [lv if c > 0 else hv for c, lv, hv in zip(coeffs, lo, hi)]
    assert sum(Fraction(c) * Fraction(v) for c, v in zip(coeffs, point)) <= Fraction(rhs)

    tl = np.array(lo)
    tu = np.array(hi)
    meta = SimpleNamespace(flat_var_types=[VarType.CONTINUOUS] * len(coeffs))
    _tighten_affine_upper_bound(tl, tu, meta, np.array(coeffs), 0.0, rhs)
    for j, v in enumerate(point):
        assert tl[j] <= v <= tu[j], (j, tl[j], v, tu[j])


def test_affine_upper_bound_never_cuts_an_exact_extreme():
    """Fraction audit: the exact extreme of every column survives the tightening."""
    rng = random.Random(15374)
    checked = 0
    for _ in range(2000):
        n = rng.randint(2, 6)
        mag = 10 ** rng.uniform(0, 12)
        coeffs = [rng.choice([-1, 1]) * 10 ** rng.uniform(-3, 3) for _ in range(n)]
        lo = [rng.uniform(-mag, mag) for _ in range(n)]
        hi = [lv + 10 ** rng.uniform(-2, 6) for lv in lo]
        mins = [Fraction(c) * Fraction(lv if c > 0 else hv) for c, lv, hv in zip(coeffs, lo, hi)]
        tot = sum(mins)
        rhs = float(tot)
        while Fraction(rhs) < tot:
            rhs = math.nextafter(rhs, math.inf)
        tl = np.array(lo)
        tu = np.array(hi)
        meta = SimpleNamespace(flat_var_types=[VarType.CONTINUOUS] * n)
        _tighten_affine_upper_bound(tl, tu, meta, np.array(coeffs), 0.0, rhs)
        for k, c in enumerate(coeffs):
            cap = (Fraction(rhs) - (tot - mins[k])) / Fraction(c)
            if c > 0 and cap <= Fraction(hi[k]):
                assert Fraction(float(tu[k])) >= cap
                checked += 1
            if c < 0 and cap >= Fraction(lo[k]):
                assert Fraction(float(tl[k])) <= cap
                checked += 1
    assert checked > 1000, checked


@pytest.mark.parametrize("K,a,b,R1,R2", EXP_WITNESSES)
def test_exp_argument_tightening_keeps_the_feasible_integer(K, a, b, R1, R2):
    """Component level: the tightened box still contains the only feasible integer."""
    # Feasible within the solver's tolerance in plain floats too.
    v = math.exp(a * K + b)
    assert max(v - R1, R2 - v) < 1e-8
    m = _two_sided_exp_model(K, a, b, R1, R2)
    tl, tu, stats = tighten_nonlinear_bounds(m, np.array([K - 50.0]), np.array([K + 50.0]))
    assert not stats.infeasible
    assert tl[0] <= K <= tu[0], (tl[0], K, tu[0])


@pytest.mark.parametrize("K,a,b,R1,R2", EXP_WITNESSES[:2])
def test_exp_two_sided_solve_is_not_falsely_infeasible(K, a, b, R1, R2):
    """End to end: these solves returned a certified ``infeasible`` before #1537 D."""
    r = _two_sided_exp_model(K, a, b, R1, R2).solve(time_limit=60)
    assert r.status == "optimal", r.status
    assert r.objective == pytest.approx(-5.0, abs=1e-6)
