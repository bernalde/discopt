"""Regression tests for #1492: a certified bound above the optimum at an
infinite-slope domain edge.

``min asin(x) - x/2`` on ``[-1, 0]`` has its optimum at the edge ``x = -1``, where
``asin'`` is infinite. The local NLP stalls at ``x ~ -1 + 1e-6`` and spatial B&B
bisects toward the edge until the node ``[-1, -1 + 9.5e-7]`` falls below the Rust
brancher's ``SPATIAL_MIN_WIDTH`` (1e-6 relative). The tree then fathoms that node
as a TIGHT box and drops its bound (-1.0707968, valid) from the global bound, which
collapsed onto the stalled incumbent -1.0694229: certified ``optimal`` 1.4e-3 above
the true -1.0707963. A relative width of 1e-6 makes the relaxation gap negligible
only for a Lipschitz objective; at an infinite-slope edge the function still moves
by ``sqrt(2e-6)`` over it. ``acos`` at -1 and ``acosh`` at 1 failed the same way.

Fix, on the Python side of the tree: a node the tree may fathom as tight (a
conservative mirror of the Rust test) tries its relaxation point AND its
gradient-descent box vertex as incumbents, and if its bound is still outside the
gap it is kept as a floor of the reported bound (decertifying, like a non-rigorous
fathom, until a later incumbent closes the floor-inclusive gap).

Testing the class also exposed a false LP optimum: ``_dual_start_slack_basis``
placed a zero-cost column with ``lb = -inf`` nonbasic at its (infinite) lower
bound, and the engine returned ``optimal`` at a non-optimal vertex that
``trusted_vertex`` then certified (``min y s.t. y >= asin(x - 1) - x/2``: certified
-0.5 against -pi/2).
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import numpy as np
import pytest


def _check_min(r, opt):
    """Sound (bound never above the optimum) and, when certified, correct."""
    tol = 1e-6 * (1.0 + abs(opt))
    assert r.bound is None or r.bound <= opt + tol, (r.status, r.objective, r.bound, opt)
    if r.gap_certified:
        assert r.objective == pytest.approx(opt, rel=1e-4, abs=1e-6), (r.objective, opt)


_EDGE_CASES = [
    # (id, f, lb, ub, argmin)  -- each optimum sits at an infinite-slope edge
    ("asin_at_-1", lambda x: dm.asin(x) - 0.5 * x, -1.0, 0.0, -1.0),
    ("acos_at_-1", lambda x: -dm.acos(x) - 0.5 * x, -1.0, 0.0, -1.0),
    ("acosh_at_1", lambda x: dm.acosh(x) - 0.5 * x, 1.0, 2.0, 1.0),
    # mirrored edges (were already right; kept so the class stays covered)
    ("asin_at_+1", lambda x: -dm.asin(x) + 0.5 * x, 0.0, 1.0, 1.0),
    ("acos_at_+1", lambda x: dm.acos(x) + 0.5 * x, 0.0, 1.0, 1.0),
]
_REF = {
    "asin_at_-1": math.asin(-1.0) + 0.5,
    "acos_at_-1": -math.acos(-1.0) + 0.5,
    "acosh_at_1": -0.5,
    "asin_at_+1": -math.asin(1.0) + 0.5,
    "acos_at_+1": 0.5,
}


@pytest.mark.parametrize("sense", ["min", "max"])
@pytest.mark.parametrize("case", _EDGE_CASES, ids=[c[0] for c in _EDGE_CASES])
def test_infinite_slope_edge_optimum_is_certified_correctly(case, sense):
    name, f, lb, ub, argmin = case
    opt = _REF[name]
    m = dm.Model(name)
    x = m.continuous("x", lb=lb, ub=ub)
    if sense == "min":
        m.minimize(f(x))
    else:
        m.maximize(-f(x))
    r = m.solve(time_limit=30)
    if sense == "max":  # compare in the minimization sense
        r_obj = None if r.objective is None else -r.objective
        r_bnd = None if r.bound is None else -r.bound
    else:
        r_obj, r_bnd = r.objective, r.bound
    tol = 1e-6 * (1.0 + abs(opt))
    assert r_bnd is None or r_bnd <= opt + tol, (sense, r.status, r_obj, r_bnd, opt)
    # These are 1-D problems: the fixed path certifies the edge optimum.
    assert r.gap_certified, (sense, r.status, r_obj, r_bnd)
    assert r_obj == pytest.approx(opt, rel=1e-4, abs=1e-6)
    assert float(np.asarray(r.x["x"])) == pytest.approx(argmin, abs=1e-5)


def test_edge_optimum_inside_a_constraint_is_sound():
    """The edge reached through a constraint (epigraph and a shifted argument)."""
    m = dm.Model("asin_shift_con")
    x = m.continuous("x", lb=0, ub=1)
    y = m.continuous("y", lb=-10, ub=10)
    m.subject_to(y >= dm.asin(x - 1) - 0.5 * x)
    m.minimize(y)
    r = m.solve(time_limit=30)
    _check_min(r, -math.pi / 2)


def test_dual_start_basis_never_rests_on_an_infinite_bound():
    from discopt.solvers.milp_simplex import _INF, _dual_start_slack_basis

    c = np.array([0.0, 1.0, 0.0, 0.0, -1.0])
    lb = np.array([0.0, -2.0, -_INF, -_INF, -1.0])
    ub = np.array([1.0, 10.0, 0.0, _INF, 3.0])
    cs, bv = _dual_start_slack_basis(c, lb, ub, 2)
    # zero-cost with only ub finite -> at upper; zero-cost free -> engine's free
    # encoding (0); zero-cost with finite lb -> lower; signed costs by sign.
    assert list(cs[:5]) == [0, 0, 2, 0, 2]
    assert list(cs[5:]) == [1, 1] and list(bv) == [5, 6]


@pytest.mark.parametrize("w_ub", [0.0, 5e-324, 3.0])
def test_deadline_lp_with_open_below_zero_cost_column(w_ub):
    """``min y s.t. y >= w - x/2`` with ``w <= w_ub`` open below: optimum is y's
    own lower bound. The deadline path used to return ``optimal`` at -0.5."""
    from discopt._relax.milp_relaxation import MilpRelaxationModel

    def _lp():
        return MilpRelaxationModel(
            np.array([0.0, 1.0, 0.0]),
            np.array([[-0.5, -1.0, 1.0]]),
            np.array([0.0]),
            [(0.0, 1.0), (-2.07, 10.0), (-np.inf, w_ub)],
            0.0,
            None,
        )

    r = _lp().solve(backend="simplex", time_limit=5.0)
    ref = _lp().solve(backend="highs")
    assert ref.status == "optimal" and ref.objective == pytest.approx(-2.07)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-2.07), r.objective


def test_may_be_tight_fathomed_mirrors_the_brancher():
    from discopt.solver import _may_be_tight_fathomed

    glb, gub = np.array([-1.0, 0.0]), np.array([0.0, 10.0])
    no_int = np.zeros(2, dtype=bool)
    none = np.zeros(0, dtype=np.int64)
    sol = np.array([-1.0, 5.0])
    # both continuous columns below 1e-6 relative width -> may be fathomed
    assert _may_be_tight_fathomed(
        np.array([-1.0, 5.0]), np.array([-1.0 + 9.5e-7, 5.0 + 5e-6]), sol, False,
        glb, gub, no_int, none,
    )  # fmt: skip
    # one column still wide -> branched, not fathomed
    assert not _may_be_tight_fathomed(
        np.array([-1.0, 5.0]), np.array([-1.0 + 9.5e-7, 6.0]), sol, False,
        glb, gub, no_int, none,
    )  # fmt: skip
    # column 1 integer: a fractional value means the tree integer-branches
    int1 = np.array([False, True])
    assert not _may_be_tight_fathomed(
        np.array([-1.0, 0.0]), np.array([-1.0 + 9.5e-7, 10.0]), np.array([-1.0, 4.5]), False,
        glb, gub, int1, none,
    )  # fmt: skip
    # integral, and not a spatial-integer column -> may be fathomed
    assert _may_be_tight_fathomed(
        np.array([-1.0, 0.0]), np.array([-1.0 + 9.5e-7, 10.0]), np.array([-1.0, 4.0]), False,
        glb, gub, int1, none,
    )  # fmt: skip
    # ... unless it is a spatial-integer column of width >= 1 (partitioned)
    assert not _may_be_tight_fathomed(
        np.array([-1.0, 0.0]), np.array([-1.0 + 9.5e-7, 10.0]), np.array([-1.0, 4.0]), False,
        glb, gub, int1, np.array([1], dtype=np.int64),
    )  # fmt: skip
