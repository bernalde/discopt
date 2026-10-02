"""Issue #1544: a product of shifted affine factors must not be multiplied out.

Under the exact change of variables ``x = y - c`` a product ``prod_k x_k`` becomes
``prod_k (y_k - c_k)``. The factorable reformulation used to distribute it into
``2**n`` monomials whose coefficients reach ``c**n`` while the value stays small,
so float64 rounding swamped it: nvs09 shifted by c ~ 1e3 had a distributed
objective of -1123.3 at a point whose true value is 10.87, and the solve ended
uncertified. The product is now lifted (``w_k == y_k - c_k``, typed ``INTEGER``
when the factor is integer-valued) whenever the expansion's rounding amplification
exceeds ``_DISTRIBUTE_CANCELLATION_LIMIT``.
"""

from __future__ import annotations

from pathlib import Path

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.factorable_reform import (
    _DISTRIBUTE_CANCELLATION_LIMIT,
    _distribution_cancellation,
    _is_integer_valued_affine,
    factorable_reformulate,
)
from discopt.modeling.core import VarType

_NL_DIR = Path(__file__).parent / "data" / "minlplib_nl"


def _shifted_product_model(shift):
    m = dm.Model("shiftprod")
    ys = [m.integer(f"y{k}", lb=1 + s, ub=6 + s) for k, s in enumerate(shift)]
    u = [y - s for y, s in zip(ys, shift)]
    m.minimize(0.3 * u[0] + 0.2 * u[1] + 0.25 * u[2] - (u[0] * u[1] * u[2]) ** 0.5)
    return m, ys, u


def test_cancellation_ratio_measures_the_shift():
    m, _, u = _shifted_product_model([0.0, 0.0, 0.0])
    assert _distribution_cancellation(u, m) == 1.0  # no shift, nothing cancels
    m, _, u = _shifted_product_model([1450.0, -644.0, 812.0])
    ratio = _distribution_cancellation(u, m)
    # per factor: (max|y| + |c|) / max|y - c|, with y in [1 + c, 6 + c]
    expected = np.prod(
        [(max(abs(1 + c), abs(6 + c)) + abs(c)) / 6.0 for c in (1450.0, -644.0, 812.0)]
    )
    assert ratio == pytest.approx(expected, rel=1e-12)
    assert ratio > _DISTRIBUTE_CANCELLATION_LIMIT


def test_unbounded_factor_is_not_evidence_of_cancellation():
    m = dm.Model("unbounded")
    x = m.continuous("x", lb=-1e30, ub=1e30)
    z = m.continuous("z", lb=0.0, ub=1.0)
    assert _distribution_cancellation([x + 1.0, z - 2.0], m) < _DISTRIBUTE_CANCELLATION_LIMIT


def test_integer_valued_affine():
    m = dm.Model("iv")
    y = m.integer("y", lb=0, ub=5)
    b = m.binary("b")
    x = m.continuous("x", lb=0, ub=5)
    assert _is_integer_valued_affine(y - 1450.0)
    assert _is_integer_valued_affine(2.0 * y - 3.0 * b + 7.0)
    assert not _is_integer_valued_affine(y - 0.5)
    assert not _is_integer_valued_affine(0.5 * y)
    assert not _is_integer_valued_affine(y + x)


def test_reformulation_lifts_shifted_factors_as_integers():
    shift = [1450.0, -644.0, 812.0]
    m, _, _ = _shifted_product_model(shift)
    r = factorable_reformulate(m)
    lifted = [v for v in r._variables if v.name.startswith("_fr_aux")]
    factor_auxes = [v for v in lifted if float(v.lb) == 1.0 and float(v.ub) == 6.0]
    assert len(factor_auxes) == 3
    assert all(v.var_type == VarType.INTEGER for v in factor_auxes)


@pytest.mark.parametrize("shift", [[1450.0, -644.0, 812.0], [1e6, -1e6, 2e6]], ids=["c1e3", "c1e6"])
def test_shifted_product_certifies_the_unshifted_optimum(shift):
    base = _shifted_product_model([0.0, 0.0, 0.0])[0].solve(time_limit=60)
    assert base.status == "optimal" and base.gap_certified
    r = _shifted_product_model(shift)[0].solve(time_limit=60)
    assert r.status == "optimal"
    assert r.gap_certified
    assert r.objective == pytest.approx(base.objective, rel=1e-6, abs=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("c, seed", [(1e3, 1), (1e6, 1)])
def test_nvs09_translated_certifies(c, seed):
    """The issue's instance under the harness's ``translate`` (seeded shifts ~c)."""
    m = dm.from_nl(str(_NL_DIR / "nvs09.nl"))
    base = m.solve(time_limit=60)
    assert base.status == "optimal" and base.gap_certified
    rng = np.random.default_rng(seed)
    shifts = np.round(rng.choice([-1.0, 1.0], size=10) * c * rng.uniform(0.5, 1.5, size=10))
    t = dm.Model("nvs09_shift")
    ys = [
        t.integer(f"y{i}", lb=float(v.lb) + s, ub=float(v.ub) + s)
        for i, (v, s) in enumerate(zip(m._variables, shifts))
    ]
    xs = [y - s for y, s in zip(ys, shifts)]
    obj = 0.0
    for x in xs:
        obj = obj + dm.log(x - 2.0) ** 2 + dm.log(10.0 - x) ** 2
    prod = xs[0]
    for x in xs[1:]:
        prod = prod * x
    t.minimize(obj - prod**0.2)
    r = t.solve(time_limit=60)
    assert r.status == "optimal"
    assert r.gap_certified
    assert r.objective == pytest.approx(base.objective, rel=1e-6, abs=1e-6)
