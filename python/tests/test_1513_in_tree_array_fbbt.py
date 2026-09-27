"""#1513: in-tree FBBT on per-scalar node boxes (models with array variables).

Before #1513 both Python node loops handed the Rust in-tree presolve kernel one
interval per variable BLOCK and ``continue``d past any node whose box length
differed -- which is every node of every model with a ``shape=(n,)`` variable,
since node boxes are per scalar. No array model ever got in-tree FBBT, cutoff
FBBT or branch-and-reduce, and nothing said so. The kernel now takes the
per-scalar box directly.
"""

from __future__ import annotations

import logging

import discopt.modeling as dm
import discopt.solver as S
import numpy as np
import pytest
from discopt._rust import model_to_repr
from discopt.tightening import probe_box


def _x0_plus_x1_le_5_array():
    m = dm.Model("arr")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=10.0)
    m.subject_to(x[0] + x[1] <= 5.0)
    m.minimize(x[0] + x[1])
    return m


def test_kernel_takes_per_scalar_box_on_array_model():
    """x[0] >= 3 at the node tightens x[1] <= 2 -- element by element."""
    repr_ = model_to_repr(_x0_plus_x1_le_5_array())
    assert repr_.n_var_blocks == 1 and repr_.n_vars == 2
    d = repr_.in_tree_presolve(
        np.array([3.0, 0.0]), np.array([10.0, 10.0]), node_depth=0, depth_stride=1, tol=1e-9
    )
    assert d["ran"] and not d["infeasible"]
    assert len(d["lb"]) == 2
    assert d["lb"][0] == 3.0
    assert d["ub"][1] == pytest.approx(2.0, abs=1e-6)
    assert int(d["bounds_tightened"]) == 2


def test_kernel_refuses_wrong_length_loudly():
    """A per-BLOCK box on an array model is an error, not a silent no-op."""
    repr_ = model_to_repr(_x0_plus_x1_le_5_array())
    with pytest.raises(ValueError, match="scalar variables"):
        repr_.in_tree_presolve(np.array([0.0]), np.array([10.0]), depth_stride=1)


def test_probe_box_array_binary_is_not_falsely_infeasible():
    """z in {0,1}^2 with z[0] + z[1] == 1 is feasible.

    The pre-#1513 ``probe_box`` handed the kernel one hull interval per block,
    so probing fixed the WHOLE binary block to 0 (sum 0) and to 1 (sum 2), found
    both infeasible and reported the model infeasible.
    """
    m = dm.Model("zb")
    z = m.binary("z", shape=(2,))
    m.subject_to(z[0] + z[1] == 1)
    m.minimize(z[0])
    bt = probe_box(m)
    assert not bt.infeasible
    assert bt.lb.shape == (2,) and bt.ub.shape == (2,)
    assert np.all(bt.lb <= 0.0) and np.all(bt.ub >= 1.0)


def test_refusal_reasons():
    """Box/repr mismatches are named; the flag gates only array models."""
    arr = model_to_repr(_x0_plus_x1_le_5_array())
    assert "node box has 3" in S._in_tree_presolve_refusal(arr, 3)
    m = dm.Model("sc")
    a = m.continuous("a", lb=0, ub=1)
    b = m.continuous("b", lb=0, ub=1)
    m.subject_to(a + b <= 1)
    m.minimize(a)
    sc = model_to_repr(m)
    assert S._in_tree_presolve_refusal(sc, 2) is None


def _dispersion(n: int, array: bool) -> dm.Model:
    m = dm.Model(f"disp{n}")
    if array:
        x = m.continuous("x", shape=(n,), lb=0.0, ub=1.0)
        y = m.continuous("y", shape=(n,), lb=0.0, ub=1.0)
        X = [x[i] for i in range(n)]
        Y = [y[i] for i in range(n)]
    else:
        X = [m.continuous(f"x{i}", lb=0.0, ub=1.0) for i in range(n)]
        Y = [m.continuous(f"y{i}", lb=0.0, ub=1.0) for i in range(n)]
    t = m.continuous("t", lb=0.0, ub=2.0)
    for i in range(n):
        for j in range(i + 1, n):
            m.subject_to((X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2 >= t)
    m.maximize(t)
    return m


_CAP = dict(time_limit=600, deterministic=True, max_nodes=60)


def test_array_model_runs_in_tree_fbbt_and_matches_scalar_form(monkeypatch):
    """The array form fires the kernel and walks the SAME tree as the scalar form.

    Both forms denote the same scalar model; with per-scalar in-tree FBBT the
    kernel sees the same box and derives the same bounds, so a deterministic,
    node-capped solve must agree on node count, status and objective. Runs
    with the flag UNSET: the per-scalar kernel is the default (graduated).
    """
    monkeypatch.delenv("DISCOPT_IN_TREE_ARRAY_FBBT", raising=False)
    rs = _dispersion(3, array=False).solve(**_CAP)
    calls_scalar = S._in_tree_presolve_global_calls()
    ra = _dispersion(3, array=True).solve(**_CAP)
    calls_array = S._in_tree_presolve_global_calls()
    assert calls_array > 0, "the array model never ran in-tree FBBT (#1513)"
    assert S._in_tree_presolve_skipped() == {}
    assert calls_array == calls_scalar
    assert ra.node_count == rs.node_count
    assert ra.status == rs.status
    assert ra.objective == pytest.approx(rs.objective, abs=1e-9)


def test_flag_off_skip_is_counted_and_warned(monkeypatch, caplog):
    """=0 keeps the legacy skip -- but counted and at WARNING, never silent."""
    monkeypatch.setenv("DISCOPT_IN_TREE_ARRAY_FBBT", "0")
    with caplog.at_level(logging.WARNING, logger="discopt.solver"):
        _dispersion(3, array=True).solve(**_CAP)
    assert S._in_tree_presolve_global_calls() == 0
    skipped = S._in_tree_presolve_skipped()
    assert sum(skipped.values()) > 0
    assert any("DISCOPT_IN_TREE_ARRAY_FBBT=0" in k for k in skipped)
    assert any("#1513" in r.getMessage() for r in caplog.records)
