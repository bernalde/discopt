"""Regression tests for #1510: the Rust tree floors its dual bound at every node it
fathoms as a TIGHT box.

In nonconvex mode ``process_evaluated`` fathoms a node when no dimension is
branchable any more (every continuous column narrower than ``SPATIAL_MIN_WIDTH``,
1e-6 of its root width). Before #1510 it dropped that node's bound from the
global bound. A width of 1e-6 bounds the relaxation gap only for a Lipschitz
objective: at an infinite-slope edge (``asin``/``acos`` at -1, ``acosh`` at 1) the
objective still moves by ``O(sqrt(1e-6))`` over the box, and ``min asin(x) - x/2``
was certified 1.4e-3 above its optimum (#1492).

#1492 shipped a Python-side floor keyed on a mirror of the Rust tight-box test
(``_may_be_tight_fathomed``); soundness therefore rested on that mirror staying
exact. The tests below disable the mirror entirely and require the certificate
to stay sound -- which holds only when the tree itself keeps the floor.
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import discopt.solver as solver_mod
import numpy as np
import pytest
from discopt._rust import PyTreeManager

_EDGES = [
    ("asin_at_-1", lambda x: dm.asin(x) - 0.5 * x, -1.0, 0.0, math.asin(-1.0) + 0.5),
    ("acos_at_-1", lambda x: -dm.acos(x) - 0.5 * x, -1.0, 0.0, -math.acos(-1.0) + 0.5),
    ("acosh_at_1", lambda x: dm.acosh(x) - 0.5 * x, 1.0, 2.0, -0.5),
]


@pytest.mark.parametrize("case", _EDGES, ids=[c[0] for c in _EDGES])
def test_certificate_sound_without_the_python_tight_box_mirror(case, monkeypatch):
    """With the Python predictor off (no edge-vertex incumbent), the stalled
    local incumbent sits ~1e-3 above the edge optimum. Before #1510 the tree
    dropped the tight edge box's bound and certified that incumbent ``optimal``
    (bound above the true optimum on all three cases)."""
    name, f, lb, ub, opt = case
    monkeypatch.setattr(solver_mod, "_may_be_tight_fathomed", lambda *a, **k: False)
    m = dm.Model(name)
    x = m.continuous("x", lb=lb, ub=ub)
    m.minimize(f(x))
    r = m.solve(time_limit=30)
    tol = 1e-6 * (1.0 + abs(opt))
    assert r.bound is not None
    assert r.bound <= opt + tol, (r.status, r.objective, r.bound, opt)
    if r.gap_certified:
        assert r.objective == pytest.approx(opt, rel=1e-4, abs=1e-6)


def test_no_incumbent_tight_fathom_is_not_an_infeasibility_proof():
    """A tree that drains over a tight-box floor with no incumbent has not
    proved the model empty (``_tree_exhausted_with_proof`` is False)."""
    t = PyTreeManager(1, [0.0], [1.0], [], [], "best_first")
    t.set_nonconvex(True)
    t.initialize()
    _blb, _bub, ids = t.export_batch(1)[:3]
    t.set_node_bounds(int(ids[0]), np.array([0.0]), np.array([1e-7]))
    t.import_results(
        np.asarray(ids, dtype=np.int64),
        np.array([-2.0]),
        np.array([[5e-8]]),
        np.array([True]),
        np.array([False]),
        np.array([False]),
    )
    t.process_evaluated()
    st = t.stats()
    assert st["open_nodes"] == 0 and t.is_finished()
    assert st["unresolved_floor"] == -2.0
    assert not solver_mod._tree_exhausted_with_proof(t)

    tf = t.take_tight_fathoms()
    assert list(tf) == [int(ids[0])]
    lo, hi = t.node_box(int(ids[0]))
    assert float(lo[0]) == 0.0 and float(hi[0]) == 1e-7
    t.raise_tight_fathom_floor(int(ids[0]), -1.5)
    assert t.stats()["unresolved_floor"] == -1.5
    with pytest.raises(ValueError):
        t.raise_tight_fathom_floor(int(ids[0]) + 7, 0.0)
