"""#1510 item 2: no warm LP entry may run from a basis that parks a nonbasic column
on an INFINITE bound.

``PreparedDual::prepare`` documented that it rejects an unusable warm basis, but
it accepted a nonbasic column sitting at an infinite bound (``AT_LOWER`` with
``lb = -inf`` while ``ub`` is finite, or ``AT_UPPER`` with ``ub = +inf``). Such a
column has no vertex value: the engines read it at 0 (or the 1e20 sentinel),
a point it can leave both ways, but price it one-sided. The dual simplex then
stopped ``optimal`` at a non-optimal vertex: on ``min y s.t. y >= w - x/2`` with
``w`` open below it returned -0.5 against the true -2.07 (#1492, where
``_dual_start_slack_basis`` built exactly this start). Every entry now refuses
it and solves cold, so the answer is the LP optimum whoever built the basis.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt._rust import solve_lp_warm_py

_INF = 1e20
AT_LOWER, BASIC, AT_UPPER = 0, 1, 2


def _lp(w_ub):
    # columns [x, y, w, s]:  -x/2 - y + w + s = 0,  s >= 0  <=>  y >= w - x/2
    c = np.array([0.0, 1.0, 0.0, 0.0])
    a = np.array([[-0.5, -1.0, 1.0, 1.0]])
    b = np.array([0.0])
    lb = np.array([0.0, -2.07, -_INF, 0.0])
    ub = np.array([1.0, 10.0, w_ub, _INF])
    return c, a, b, lb, ub


@pytest.mark.parametrize("w_ub", [0.0, 3.0])
@pytest.mark.parametrize(
    "label,col_status,basic",
    [
        # w nonbasic AT_LOWER on its -inf side, slack basic
        ("at_minus_inf", [AT_LOWER, AT_LOWER, AT_LOWER, BASIC], [3]),
        # slack nonbasic AT_UPPER on its +inf side, w basic
        ("at_plus_inf", [AT_LOWER, AT_LOWER, BASIC, AT_UPPER], [2]),
    ],
)
def test_warm_start_at_an_infinite_bound_returns_the_true_optimum(w_ub, label, col_status, basic):
    c, a, b, lb, ub = _lp(w_ub)
    cold = solve_lp_warm_py(c, a, b, lb, ub)
    assert cold[0] == "optimal" and cold[2] == pytest.approx(-2.07, abs=1e-9)
    warm = solve_lp_warm_py(
        c,
        a,
        b,
        lb,
        ub,
        start_col_status=np.array(col_status, dtype=np.int8),
        start_basic_vars=np.array(basic, dtype=np.int64),
    )
    assert warm[0] == "optimal", (label, warm[0])
    assert warm[2] == pytest.approx(-2.07, abs=1e-9), (label, w_ub, warm[2])
