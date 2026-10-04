"""Regression tests for #1611 X-30a / X-30b: the budget (polyhedral) robust
counterpart in ``discopt.ro``.

X-30a: a row with no uncertain data was given a dual penalty ``-b^T lam``; on an
equality row that penalty is a free slack, so ``sum(w) == 1`` admitted ``w = 0``
and the "robust" optimum (0.0) fell below the nominal one.

X-30b: the duals were capped at ``sum|b| + 100``. A dual equals a coefficient of
the protected row -- a decision-variable expression in the model's units -- so
the same model rescaled by 1000 was declared infeasible.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.ro import RobustCounterpart, budget_uncertainty_set

pytestmark = pytest.mark.filterwarnings("ignore:Variables with very large")

_S = np.array([5.0, 10.0, 20.0])
_DELTA = np.array([1.0, 2.0, 4.0])


def _blend(sense: str, scale: float = 1.0):
    m = dm.Model("blend")
    w = m.continuous("w", shape=(3,), lb=0, ub=scale)
    s = m.parameter("s", value=_S)
    m.minimize(3 * w[0] + 2 * w[1] + 1 * w[2])
    tot = w[0] + w[1] + w[2]
    m.subject_to(tot == scale if sense == "==" else tot >= scale, name="demand")
    m.subject_to(s @ w - 15.0 * tot <= 0, name="sulfur")
    RobustCounterpart(m, budget_uncertainty_set(s, delta=_DELTA, gamma=2.0)).formulate()
    return m, w


@pytest.mark.parametrize("sense", ["==", ">="])
def test_parameter_free_row_gets_no_counterpart(sense):
    m, w = _blend(sense)
    r = m.solve()
    assert r.status == "optimal"
    # The robust optimum: worst case puts +2 on w1 and +4 on w2's assay.
    assert r.objective == pytest.approx(1.75, abs=1e-5)
    wv = np.asarray(r.value(w))
    assert wv.sum() == pytest.approx(1.0, abs=1e-6)


def test_robust_optimum_never_below_nominal_on_equality_row():
    m, _ = _blend("==")
    nominal = dm.Model("nominal")
    w = nominal.continuous("w", shape=(3,), lb=0, ub=1)
    nominal.minimize(3 * w[0] + 2 * w[1] + 1 * w[2])
    nominal.subject_to(w[0] + w[1] + w[2] == 1.0)
    nominal.subject_to(_S @ w - 15.0 * (w[0] + w[1] + w[2]) <= 0)
    rn = nominal.solve()
    rr = m.solve()
    assert rr.objective >= rn.objective - 1e-6


@pytest.mark.parametrize("scale", [1.0, 1000.0])
def test_counterpart_is_scale_invariant(scale):
    m, _ = _blend(">=", scale=scale)
    r = m.solve()
    assert r.status == "optimal"
    assert r.objective == pytest.approx(1.75 * scale, rel=1e-5)


def test_uncertain_equality_row_must_hold_for_every_realization():
    """An equality whose data is uncertain is robust only if it holds at every
    xi: here ``s @ w == 10`` cannot hold for all assays unless w = 0, which
    violates the demand row, so the counterpart is infeasible -- the one-sided
    worst-case maximum used to accept it."""
    m = dm.Model("eq_unc")
    w = m.continuous("w", shape=(3,), lb=0, ub=1)
    s = m.parameter("s", value=_S)
    m.minimize(w[0] + w[1] + w[2])
    m.subject_to(w[0] + w[1] + w[2] >= 1.0)
    m.subject_to(s @ w == 10.0, name="assay")
    RobustCounterpart(m, budget_uncertainty_set(s, delta=_DELTA, gamma=2.0)).formulate()
    r = m.solve()
    assert r.status == "infeasible"
