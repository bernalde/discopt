"""Regression tests for issue #1500.

``lazy_constraints`` / ``incumbent_callback`` define which points are acceptable.
They were silently ignored (0 calls) on ``solver="amp"`` and ``"mip-nlp"``
(certified answers violating them), on ``"direct"`` / ``"surrogate"`` (uncertified
but violating), and on the default route for models with no integer variable
(the LP / convex QP / convex NLP entry routes certified the callback-rejected
point). Every route must either enforce the callbacks or refuse loudly with a
``ValueError``, as ``nlp_bb=True`` already did.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.callbacks import CutResult
from discopt.modeling.core import SolveResult

# Every solver= selector other than the spatial B&B's (None / "bb"); the table
# test below pins the dispatch's own allow-list to exactly those two.
_NON_ENFORCING = ["amp", "mip-nlp", "direct", "surrogate", "gp", "gp-minlp", "sgo", "gurobi"]


class _Counter:
    def __init__(self):
        self.n = 0


def _int_model():
    m = dm.Model("cb")
    x = m.integer("x", lb=0, ub=5)
    y = m.integer("y", lb=0, ub=5)
    m.minimize(-x - y - 0.1 * x)
    return m, x, y


def _lazy_sum_le_4(x, y, counter):
    def lazy(ctx, mdl):
        counter.n += 1
        xr = ctx.x_relaxation
        if xr[0] + xr[1] > 4 + 1e-6:
            return [CutResult([(x, 1.0), (y, 1.0)], "<=", 4.0)]
        return []

    return lazy


def _reject_all(counter):
    def cb(ctx, mdl, sol):
        counter.n += 1
        return False

    return cb


def test_non_enforcing_table_is_complete():
    from discopt.solver import _CALLBACK_ENFORCING_SOLVERS

    assert _CALLBACK_ENFORCING_SOLVERS == frozenset({None, "bb"})


@pytest.mark.parametrize("solver", _NON_ENFORCING)
@pytest.mark.parametrize("which", ["lazy", "incumbent"])
def test_non_bb_solver_families_refuse_callbacks(solver, which):
    m, x, y = _int_model()
    c = _Counter()
    kw = (
        {"lazy_constraints": _lazy_sum_le_4(x, y, c)}
        if which == "lazy"
        else {"incumbent_callback": _reject_all(c)}
    )
    with pytest.raises(ValueError, match="cannot be used with solver="):
        m.solve(time_limit=10, solver=solver, **kw)
    assert c.n == 0


@pytest.mark.parametrize(
    "kw",
    [{"decomposition": "benders"}, {"decomposition": "lagrangian"}, {"gdp_method": "loa"}],
)
def test_decomposition_routes_refuse_callbacks(kw):
    m, x, y = _int_model()
    c = _Counter()
    with pytest.raises(ValueError, match="cannot be used with"):
        m.solve(time_limit=10, lazy_constraints=_lazy_sum_le_4(x, y, c), **kw)
    assert c.n == 0


def test_default_integer_route_still_enforces():
    m, x, y = _int_model()
    c = _Counter()
    r = m.solve(time_limit=20, lazy_constraints=_lazy_sum_le_4(x, y, c))
    assert c.n > 0
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-4.4, abs=1e-6)
    assert float(np.asarray(r.x["x"])) + float(np.asarray(r.x["y"])) <= 4 + 1e-6


def _cont_model(kind):
    m = dm.Model(kind)
    x = m.continuous("x", lb=0, ub=5)
    y = m.continuous("y", lb=0, ub=5)
    if kind == "lp":
        m.minimize(-x - y)
        opt = -4.0
    elif kind == "convex_qp":
        m.minimize((x - 5) ** 2 + (y - 5) ** 2)
        opt = 18.0
    elif kind == "convex_nlp":
        m.minimize(dm.exp(5 - x) + dm.exp(5 - y))
        opt = 2 * float(np.exp(3.0))
    else:
        raise AssertionError(kind)
    return m, x, y, opt


@pytest.mark.parametrize("kind", ["lp", "convex_qp", "convex_nlp"])
def test_continuous_entry_routes_enforce_lazy_constraints(kind):
    m, x, y, opt = _cont_model(kind)
    c = _Counter()
    r = m.solve(time_limit=30, lazy_constraints=_lazy_sum_le_4(x, y, c))
    assert c.n > 0, "the callback was never called"
    assert r.status == "optimal"
    assert r.objective == pytest.approx(opt, rel=1e-5, abs=1e-5)
    xs = float(np.asarray(r.x["x"])) + float(np.asarray(r.x["y"]))
    assert xs <= 4 + 1e-5, (kind, r.x)


@pytest.mark.parametrize("kind", ["lp", "convex_qp"])
def test_continuous_entry_routes_enforce_incumbent_rejection(kind):
    m, _x, _y, _opt = _cont_model(kind)
    c = _Counter()
    r = m.solve(time_limit=30, incumbent_callback=_reject_all(c))
    assert c.n > 0, "the callback was never called"
    assert r.status != "optimal"
    assert not r.gap_certified
    assert not r.x


def test_backstop_refuses_an_unscreened_point():
    """The outermost wrapper refuses a returned point that no callback-screening
    engine produced, so a route added later cannot skip the callbacks silently."""
    from discopt.solver import _mark_callbacks_enforced, _refusing_on_callback_failure

    def fake_route(model, lazy_constraints=None, incumbent_callback=None, mark=False):
        if mark:
            _mark_callbacks_enforced()
        return SolveResult(status="optimal", objective=0.0, x={"x": np.array(1.0)})

    wrapped = _refusing_on_callback_failure(fake_route)
    lazy = lambda ctx, mdl: []  # noqa: E731
    with pytest.raises(ValueError, match="lazy_constraints cannot be used"):
        wrapped(None, lazy_constraints=lazy)
    # Enforced, or no callback at all, passes through.
    assert wrapped(None, lazy_constraints=lazy, mark=True).status == "optimal"
    assert wrapped(None).status == "optimal"
    # A result with no point cannot violate a callback.
    empty = _refusing_on_callback_failure(
        lambda model, lazy_constraints=None: SolveResult(status="infeasible")
    )
    assert empty(None, lazy_constraints=lazy).status == "infeasible"
