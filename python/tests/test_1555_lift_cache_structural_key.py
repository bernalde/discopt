"""#1555: the factorable lift's aux cache must key on EXACT structure.

``_Lifter.expression`` deduplicated aux variables by ``repr(expr)``. ``repr`` is a
display string: ``SumOverExpression`` prints as ``"Σ[n terms]"`` and ``Constant``
rounds to ``.6g``, so distinct expressions shared one aux. ``dm.sum([x, 1])`` and
``dm.sum([y, 1])`` -> one aux defined as ``x + 1`` -> ``(y+1)**1.7`` silently became
``(x+1)**1.7`` and a false optimum was certified. The key may not regress to a
bare ``id()`` either: that was the ex7_2_3 false optimum (recycled addresses).
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.factorable_reform import _Lifter
from discopt.modeling.core import Constant, SumOverExpression


def _two_vars():
    m = dm.Model("t")
    return m, m.continuous("x", lb=0, ub=4), m.continuous("y", lb=0, ub=4)


@pytest.mark.parametrize("sense", ["max", "min"])
def test_dm_sum_inside_fractional_powers_certifies_the_true_optimum(sense):
    """The issue's repro: certified 1.8095 (true 4.0) before the fix."""
    m, x, y = _two_vars()
    m.subject_to(dm.sum([x, 1.0]) ** 1.2 * dm.sum([y, 1.0]) ** 1.7 <= 20)
    if sense == "max":
        m.maximize(x - 10 * y)
    else:
        m.minimize(-x + 10 * y)
    r = m.solve(time_limit=20)
    assert r.gap_certified and r.status == "optimal"
    assert abs(r.objective) == pytest.approx(4.0, abs=1e-5)
    assert float(np.asarray(r.x["x"])) == pytest.approx(4.0, abs=1e-5)


def test_maximize_x_bound_is_not_the_aliased_models():
    """With ``maximize(x)`` the aliased lift published bound 1.8095 < 4.0."""
    m, x, y = _two_vars()
    m.subject_to(dm.sum([x, 1.0]) ** 1.2 * dm.sum([y, 1.0]) ** 1.7 <= 20)
    m.maximize(x)
    r = m.solve(time_limit=20)
    if r.bound is not None:
        assert r.bound >= 4.0 - 1e-6  # an upper bound must not cut off x = 4
    assert r.gap_certified and r.objective == pytest.approx(4.0, abs=1e-5)


def test_distinct_sums_never_share_an_aux_identical_ones_still_do():
    m, x, y = _two_vars()
    lifter = _Lifter(m)
    ax = lifter.expression(dm.sum([x, 1.0]))
    assert lifter.expression(dm.sum([y, 1.0])) is not ax
    assert lifter.expression(SumOverExpression([x, Constant(2.0)])) is not ax
    # deduplication of a genuinely identical (but separately built) node is kept
    assert lifter.expression(dm.sum([x, 1.0])) is ax
    assert lifter.expression(SumOverExpression([x, Constant(1.0)])) is ax


def test_key_is_lossless_where_repr_was_not():
    from discopt._relax.factorable_reform import _structural_key

    m, x, _ = _two_vars()
    pins: list = []
    # ``repr`` rounds constants to .6g (#1497): these printed identically
    a = _structural_key(x + 1.0000001, pins)
    b = _structural_key(x + 1.0000004, pins)
    assert a != b
    # two-term sums of different variables printed identically as "Σ[2 terms]"
    assert repr(dm.sum([x, 1.0])) == repr(dm.sum([x, 2.0]))
    assert _structural_key(dm.sum([x, 1.0]), pins) != _structural_key(dm.sum([x, 2.0]), pins)
    # and structurally equal expressions agree
    assert _structural_key(x + 1.0, pins) == _structural_key(x + 1.0, pins)
    assert not pins  # every node above has an exact rule: nothing fell back to id()
