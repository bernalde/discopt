"""Regression tests for #1501.

``solver="mip-nlp"`` on an integer-free model runs ONE local NLP. Before #1501 it
certified that solve whenever the model was classified convex:

* a point from an ``ITERATION_LIMIT`` exit (``min |x-2|`` stopped at ``x=2.94``)
  was returned as ``optimal`` with ``bound = objective``;
* ANY failed solve (``Error_In_Step_Computation`` on a feasible model,
  ``Diverging_Iterates`` on an unbounded LP) was returned as ``infeasible`` with
  ``gap_certified=True``.

The contract tested here is honesty, not strength: whatever the route returns, it
must not certify a wrong value and must not certify infeasibility of a feasible
model.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import pytest

pytestmark = pytest.mark.smoke


def _abs1():
    m = dm.Model("abs")
    x = m.continuous("x", lb=-3, ub=3)
    m.minimize(dm.abs(x - 2))
    return m, 0.0


def _abs2():
    m = dm.Model("abs2")
    x = m.continuous("x", lb=-3, ub=3)
    y = m.continuous("y", lb=-3, ub=3)
    m.subject_to(x + y >= 1)
    m.minimize(dm.abs(x - 2) + dm.abs(y + 1))
    return m, 0.0


def _max1():
    m = dm.Model("mx")
    x = m.continuous("x", lb=-3, ub=3)
    m.minimize(dm.maximum(x - 2, 2 - x))
    return m, 0.0


def _abs3():
    m = dm.Model("abs3")
    x = m.continuous("x", lb=-3, ub=3)
    m.minimize(dm.abs(x - 2) + 0.3 * dm.abs(x))
    return m, 0.6


def _unbounded_min():
    m = dm.Model("u")
    x = m.continuous("x")
    m.minimize(x)
    return m, None


def _unbounded_max():
    m = dm.Model("u2")
    x = m.continuous("x", lb=0)
    m.maximize(x)
    return m, None


def _solve(m, method, profile):
    opts = {"mip_nlp_profile": profile} if profile == "shot" else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m.solve(time_limit=15, solver="mip-nlp", mip_nlp_method=method, mip_nlp_options=opts)


_METHODS = [
    ("oa", "default"),
    ("ecp", "default"),
    ("goa", "default"),
    ("lp_nlp_bb", "default"),
    ("oa", "shot"),
]


@pytest.mark.parametrize("method, profile", _METHODS)
@pytest.mark.parametrize("build", [_abs1, _abs2, _max1, _abs3], ids=lambda f: f.__name__)
def test_nonsmooth_convex_not_falsely_certified(build, method, profile):
    m, true_opt = build()
    r = _solve(m, method, profile)
    info = (r.status, r.objective, r.bound, r.gap_certified)
    # The models are feasible: an infeasibility certificate is always false.
    assert r.status != "infeasible", info
    if r.status == "optimal" or r.gap_certified:
        assert r.objective is not None, info
        assert r.objective == pytest.approx(true_opt, abs=1e-4), info
    if r.bound is not None:
        assert r.bound <= true_opt + 1e-4, info
    if r.objective is not None:
        # Any reported incumbent value is attained, so it cannot beat the optimum.
        assert r.objective >= true_opt - 1e-4, info


@pytest.mark.parametrize("method, profile", _METHODS)
@pytest.mark.parametrize("build", [_unbounded_min, _unbounded_max], ids=lambda f: f.__name__)
def test_unbounded_lp_not_certified_infeasible(build, method, profile):
    m, _ = build()
    r = _solve(m, method, profile)
    info = (r.status, r.objective, r.bound, r.gap_certified)
    # x = 0 is feasible, so "infeasible" is a false certificate.
    assert r.status != "infeasible", info
