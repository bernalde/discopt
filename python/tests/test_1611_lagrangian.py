"""Regression tests for #1611 D-24 / D-25 (Lagrangian decomposition).

D-24: the level bundle grew its multiplier box on every iteration whose QP
projection touched the box, before any serious step proved the box binding.
The stability center kept being reset, the cutting-plane model never matured,
and on a 4x12 generalized assignment problem the bundle reported bound 0.0
after 50 iterations while Kelley reached ~331.6.

D-25: ``Model.solve(decomposition="lagrangian")`` returned the Lagrangian
result (no incumbent) to ``Model.solve``, whose #844 no-incumbent fallback then
re-solved the model with LP-per-node B&B and reported ``optimal`` -- ignoring
``max_nodes`` and hiding that the requested method never produced the answer.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.decomposition.lagrangian import solve_lagrangian

_M, _N = 4, 12
_rng = np.random.default_rng(0)
_W = _rng.integers(5, 26, (_M, _N))
_C = _rng.integers(10, 51, (_M, _N))
_B = np.floor(0.8 * _W.sum(1) / _M).astype(int)
# Integer optimum of this instance (B&B, certified); z_LD can be no larger.
_Z_IP = 342.0


def _gap():
    m = dm.Model("gap")
    x = [m.binary(f"x{i}", shape=(_N,)) for i in range(_M)]
    m.minimize(
        dm.sum(lambda i: dm.sum(lambda j: _C[i, j] * x[i][j], over=range(_N)), over=range(_M))
    )
    for i in range(_M):
        m.subject_to(dm.sum(lambda j: _W[i, j] * x[i][j], over=range(_N)) <= _B[i])
    for j in range(_N):
        m.subject_to(dm.sum(lambda i: x[i][j], over=range(_M)) == 1, name=f"a{j}")
        m.mark_coupling(f"a{j}")
    return m


def test_level_bundle_bound_matches_kelley():
    kelley = solve_lagrangian(_gap(), method="kelley", max_iterations=100).bound
    bundle = solve_lagrangian(_gap(), method="bundle", max_iterations=50).bound
    assert kelley is not None and bundle is not None
    # Pre-fix: bundle == 0.0 while kelley ~= 331.6.
    assert bundle >= kelley - 1.0
    # A Lagrangian dual bound never exceeds the integer optimum.
    assert bundle <= _Z_IP + 1e-6


def test_requested_lagrangian_is_not_replaced_by_bnb_fallback():
    # max_nodes caps the Kelley iterations. 100, not 60: the masters moved to HiGHS
    # (#1614 D-27), which breaks ties between optimal subproblem vertices differently
    # from the in-house simplex, so Kelley's trajectory changes. Measured: both
    # engines converge to 331.6; HiGHS reaches it between 60 and 70 iterations
    # (330.40 at 60), the simplex by 60.
    r = _gap().solve(decomposition="lagrangian", lagrangian_method="kelley", max_nodes=100)
    # Pre-fix: status "optimal", bound_source "bnb_tree", node_count 484 > max_nodes.
    assert r.bound_source != "bnb_tree"
    assert r.status != "optimal" or r.gap_certified
    assert r.bound is not None
    assert r.bound <= _Z_IP + 1e-6
    assert r.bound == pytest.approx(331.6, abs=1.0)
