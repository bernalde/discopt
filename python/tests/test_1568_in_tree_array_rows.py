"""#1568: in-tree FBBT through array-valued rows (``DISCOPT_IN_TREE_ARRAY_ROWS``).

The per-scalar FBBT view (#1513) points every array-valued reference at a
never-tightened hull proxy, so a row such as ``zh - (sum(W.T * x, axis=1) + b) == 0``
or ``z - sigmoid(zh) == 0`` -- every ``discopt.ml`` layer -- tightens nothing
per element, and branching on the inputs never reaches the activations. With the
flag on (the default since the graduation panel; ``=0`` opts out), the view
expands each array row into one scalar row per element.
"""

from __future__ import annotations

import re

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._rust import model_to_repr

FLAG = "DISCOPT_IN_TREE_ARRAY_ROWS"


def _net(width: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0, 1 / np.sqrt(2), size=(2, width))
    b1 = rng.normal(0, 0.1, size=width)
    w2 = rng.normal(0, 1 / np.sqrt(width), size=(width, 1))
    b2 = rng.normal(0, 0.1, size=1)
    return w1, b1, w2, b2


def _sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))


def _layer_model(vector_rows: bool):
    """One sigmoid layer, array variables, rows written as arrays or per element."""
    w1, b1, w2, b2 = _net()
    n = len(b1)
    zl = b1 + np.minimum(-w1, w1).sum(0)
    zu = b1 + np.maximum(-w1, w1).sum(0)
    m = dm.Model("layer")
    x = m.continuous("x", shape=(2,), lb=-1, ub=1)
    zh = m.continuous("zh", shape=(n,), lb=zl, ub=zu)
    z = m.continuous("z", shape=(n,), lb=_sigmoid(zl), ub=_sigmoid(zu))
    if vector_rows:
        m.subject_to(zh - (dm.sum(w1.T * x, axis=1) + b1) == 0)
        m.subject_to(z - dm.sigmoid(zh) == 0)
    else:
        for j in range(n):
            m.subject_to(zh[j] == sum(float(w1[i, j]) * x[i] for i in range(2)) + float(b1[j]))
            m.subject_to(z[j] == dm.sigmoid(zh[j]))
    m.minimize(sum(float(w2[j, 0]) * z[j] for j in range(n)) + float(b2[0]))
    return m


def _box(m):
    lb = np.concatenate([np.ravel(np.broadcast_to(v.lb, v.shape)) for v in m._variables])
    ub = np.concatenate([np.ravel(np.broadcast_to(v.ub, v.shape)) for v in m._variables])
    return lb.astype(float), ub.astype(float)


def _kernel(m, lb, ub):
    repr_ = model_to_repr(m, getattr(m, "_builder", None))
    return repr_.in_tree_presolve(lb.copy(), ub.copy(), 0, 1, 16, 1e-9, None, False, 0)


def _halved_box(m):
    lb, ub = _box(m)
    lb[0], ub[0] = 0.0, 1.0  # x0 in [0, 1]
    lb[1], ub[1] = -1.0, 0.0  # x1 in [-1, 0]
    return lb, ub


def test_flag_off_proxy_view_tightens_nothing_through_array_rows(monkeypatch):
    """The control: the pre-#1568 behaviour (the ``=0`` opt-out), pinned so the
    fix stays measurable and the opt-out keeps restoring the legacy view."""
    monkeypatch.setenv(FLAG, "0")
    m = _layer_model(vector_rows=True)
    d = _kernel(m, *_halved_box(m))
    assert d["ran"] and not d["infeasible"]
    assert d["array_rows"] is None
    assert d["bounds_tightened"] == 0


def test_flag_on_expands_rows_and_matches_the_scalar_row_model(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    vec = _layer_model(vector_rows=True)
    lb, ub = _halved_box(vec)
    d = _kernel(vec, lb, ub)
    assert d["ran"] and not d["infeasible"]
    assert d["array_rows"] == "expanded"
    assert d["bounds_tightened"] > 0
    n = 10
    zs = slice(2 + n, 2 + 2 * n)
    width = (np.asarray(d["ub"]) - np.asarray(d["lb"]))[zs] / (ub - lb)[zs]
    assert width.mean() < 0.6, width  # the activations now narrow with the inputs

    # The expanded view IS the per-element model: same box out, to round-off.
    sca = _layer_model(vector_rows=False)
    ds = _kernel(sca, lb, ub)
    np.testing.assert_allclose(d["lb"], ds["lb"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(d["ub"], ds["ub"], rtol=0, atol=1e-9)


def test_flag_on_never_cuts_a_feasible_point(monkeypatch):
    """Forward-pass points of random inputs inside random node boxes stay inside."""
    monkeypatch.setenv(FLAG, "1")
    w1, b1, _, _ = _net()
    m = _layer_model(vector_rows=True)
    root_lb, root_ub = _box(m)
    rng = np.random.default_rng(1568)
    checked = 0
    for _ in range(60):
        x = rng.uniform(-1, 1, size=2)
        zh = x @ w1 + b1
        pt = np.concatenate([x, zh, _sigmoid(zh)])
        lb = root_lb + (pt - root_lb) * rng.uniform(0, 1, size=pt.size)
        ub = pt + (root_ub - pt) * rng.uniform(0, 1, size=pt.size)
        d = _kernel(m, lb, ub)
        assert d["array_rows"] == "expanded"
        assert not d["infeasible"], "a box holding a feasible point was fathomed"
        assert np.all(np.asarray(d["lb"]) <= pt + 1e-7)
        assert np.all(pt <= np.asarray(d["ub"]) + 1e-7)
        checked += pt.size
    assert checked == 60 * 22


def test_refused_row_stays_on_the_proxy_view_and_says_so(monkeypatch):
    """A row ``expand`` refuses (``dm.maximum`` has no scalar opcode) keeps its
    pre-#1568 proxy-view form -- per row, not for the whole view -- is counted
    as ``array_rows_on_hull``, and the outcome names it instead of failing
    silently. With no other array row to expand, the box is the OFF box."""

    def build():
        m = dm.Model("decline")
        x = m.continuous("x", shape=(3,), lb=-2, ub=2)
        y = m.continuous("y", shape=(3,), lb=-10, ub=10)
        m.subject_to(y - dm.maximum(x, 0.5) == 0)
        m.minimize(dm.sum(y))
        return m

    m = build()
    lb, ub = _box(m)
    lb[:3], ub[:3] = 0.0, 1.0
    monkeypatch.setenv(FLAG, "0")
    off = _kernel(build(), lb, ub)
    monkeypatch.setenv(FLAG, "1")
    on = _kernel(build(), lb, ub)
    assert off["array_rows"] is None
    assert off["array_rows_added"] == 0 and off["array_rows_on_hull"] == 0
    assert on["array_rows"].startswith("partial: constraint 0:"), on["array_rows"]
    assert on["array_rows_added"] == 0 and on["array_rows_on_hull"] == 1
    for k in ("lb", "ub"):
        np.testing.assert_array_equal(on[k], off[k])
    assert on["bounds_tightened"] == off["bounds_tightened"]


def test_refused_row_does_not_block_its_neighbours(monkeypatch):
    """Per-row mode: the sigmoid layer's rows expand (and tighten exactly as the
    per-element model does) even with an unexpandable ``maximum`` row beside them."""
    monkeypatch.setenv(FLAG, "1")

    def build(vector_rows):
        m = _layer_model(vector_rows=vector_rows)
        w = m.continuous("w", shape=(3,), lb=-10, ub=10)
        u = m.continuous("u", shape=(3,), lb=-2, ub=2)
        m.subject_to(w - dm.maximum(u, 0.5) == 0)
        return m

    vec = build(True)
    lb, ub = _halved_box(vec)
    d = _kernel(vec, lb, ub)
    assert d["array_rows"].startswith("partial: constraint 2:"), d["array_rows"]
    assert d["array_rows_added"] == 20 and d["array_rows_on_hull"] == 1
    monkeypatch.delenv(FLAG, raising=False)
    sca = _kernel(build(False), lb, ub)
    n = 2 + 10 + 10
    np.testing.assert_allclose(d["lb"][:n], sca["lb"][:n], rtol=0, atol=1e-9)
    np.testing.assert_allclose(d["ub"][:n], sca["ub"][:n], rtol=0, atol=1e-9)


def test_kwarg_overrides_the_environment(monkeypatch):
    m = _layer_model(vector_rows=True)
    lb, ub = _halved_box(m)
    r = model_to_repr(m, getattr(m, "_builder", None))
    monkeypatch.setenv(FLAG, "1")
    off = r.in_tree_presolve(lb.copy(), ub.copy(), 0, 1, expand_array_rows=False)
    assert off["array_rows"] is None and off["array_rows_added"] == 0
    monkeypatch.delenv(FLAG, raising=False)
    on = r.in_tree_presolve(lb.copy(), ub.copy(), 0, 1, expand_array_rows=True)
    assert on["array_rows"] == "expanded" and on["array_rows_added"] == 20
    assert on["array_rows_on_hull"] == 0


def test_default_is_on_and_zero_opts_out(monkeypatch):
    """Graduated (#1568): with the variable unset the kernel expands; ``=0`` and
    its spellings restore the proxy view exactly."""
    m = _layer_model(vector_rows=True)
    lb, ub = _halved_box(m)
    monkeypatch.delenv(FLAG, raising=False)
    on = _kernel(m, lb, ub)
    assert on["array_rows"] == "expanded" and on["array_rows_added"] == 20
    assert on["bounds_tightened"] > 0
    for v in ("0", "false", "off", "no", " OFF "):
        monkeypatch.setenv(FLAG, v)
        off = _kernel(_layer_model(vector_rows=True), lb, ub)
        assert off["array_rows"] is None and off["array_rows_added"] == 0, v
        assert off["bounds_tightened"] == 0, v


def test_unknown_outcome_is_refused_loudly():
    import discopt.solver as S

    with pytest.raises(ValueError):
        S._note_array_rows({"ran": True, "array_rows": "bogus: x", "array_rows_added": 0})


@pytest.mark.slow
def test_discopt_ml_sigmoid_layer_certifies_by_default(monkeypatch):
    """The issue's case end to end: the ``discopt.ml`` full-space 2-10-1 sigmoid
    network. With ``=0`` it does not certify in 30 s (bound stalls near the
    interval bound -0.313); by default it certifies the optimum -0.220785, the
    value the per-element hand-built model certifies in 3 nodes."""
    from discopt.ml import add_predictor
    from discopt.ml.network import DenseLayer, NetworkDefinition

    monkeypatch.delenv(FLAG, raising=False)
    w1, b1, w2, b2 = _net()
    m = dm.Model("nn")
    x = m.continuous("x", shape=(2,), lb=-1.0, ub=1.0)
    net = NetworkDefinition(
        [DenseLayer(w1, b1, "sigmoid"), DenseLayer(w2, b2, "linear")],
        input_bounds=(np.full(2, -1.0), np.full(2, 1.0)),
    )
    y, _ = add_predictor(m, x, net, method="full_space")
    m.minimize(y[0])
    r = m.solve(time_limit=30)
    assert r.gap_certified and r.status == "optimal"
    assert r.objective == pytest.approx(-0.220785, abs=1e-5)
    assert (r.solver_stats or {}).get("reduce/array_rows_expanded", 0) > 0
    assert (r.solver_stats or {}).get("reduce/array_row_calls", 0) > 0


def test_on_box_is_never_looser_than_off_and_keeps_every_feasible_point():
    """§5 differential bound test, on random node boxes of the sigmoid layer.

    Per box: the ON (expanded) box is contained in the OFF (proxy-view) box, and
    both contain the forward-pass point the box was drawn around -- a feasible
    point, so no valid point is cut and no bound passes the box optimum. Uses the
    ``expand_array_rows`` kwarg so the comparison does not depend on the default.
    """
    w1, b1, _, _ = _net()
    m = _layer_model(vector_rows=True)
    r = model_to_repr(m, getattr(m, "_builder", None))
    root_lb, root_ub = _box(m)
    rng = np.random.default_rng(15681)
    checks = strictly_tighter = 0
    for _ in range(80):
        x = rng.uniform(-1, 1, size=2)
        zh = x @ w1 + b1
        pt = np.concatenate([x, zh, _sigmoid(zh)])
        lb = root_lb + (pt - root_lb) * rng.uniform(0, 1, size=pt.size)
        ub = pt + (root_ub - pt) * rng.uniform(0, 1, size=pt.size)
        off = r.in_tree_presolve(lb.copy(), ub.copy(), 0, 1, expand_array_rows=False)
        on = r.in_tree_presolve(lb.copy(), ub.copy(), 0, 1, expand_array_rows=True)
        assert on["array_rows"] == "expanded"
        assert not off["infeasible"] and not on["infeasible"]
        on_lb, on_ub = np.asarray(on["lb"]), np.asarray(on["ub"])
        off_lb, off_ub = np.asarray(off["lb"]), np.asarray(off["ub"])
        assert np.all(on_lb >= off_lb - 1e-9) and np.all(on_ub <= off_ub + 1e-9)
        assert np.all(on_lb <= pt + 1e-7) and np.all(pt <= on_ub + 1e-7)
        strictly_tighter += int(np.any(on_lb > off_lb + 1e-9) or np.any(on_ub < off_ub - 1e-9))
        checks += 4 * pt.size
    assert checks == 80 * 4 * 22
    assert strictly_tighter > 0  # the comparison is not vacuous


def _op_cases():
    """Array-row shapes beyond the sigmoid layer, each as ``(name, build, sample)``.

    ``build()`` returns a model whose rows are ``out == expr`` over array
    variables; ``sample(rng)`` returns a FEASIBLE point as ``{name: array}``
    (inputs drawn in their bounds, outputs computed forward). Outputs get wide
    bounds so the drawn point is always inside the root box.
    """
    rng0 = np.random.default_rng(7)
    a = rng0.normal(size=(3, 4))
    c = rng0.normal(size=2)
    w = rng0.normal(size=(3, 2))
    big = 50.0

    def matmul():
        m = dm.Model("matmul")
        x = m.continuous("x", shape=(4,), lb=-1, ub=1)
        y = m.continuous("y", shape=(3,), lb=-big, ub=big)
        m.subject_to(y - a @ x == 0)
        m.minimize(dm.sum(y))
        return m

    def matmul_sample(rng):
        x = rng.uniform(-1, 1, 4)
        return {"x": x, "y": a @ x}

    def matvec_var():
        m = dm.Model("matvec_var")
        xm = m.continuous("X", shape=(3, 2), lb=-1, ub=1)
        y = m.continuous("y", shape=(3,), lb=-big, ub=big)
        m.subject_to(y - xm @ c == 0)
        m.minimize(dm.sum(y))
        return m

    def matvec_var_sample(rng):
        xm = rng.uniform(-1, 1, (3, 2))
        return {"X": xm, "y": xm @ c}

    def slicing():
        m = dm.Model("slicing")
        x = m.continuous("x", shape=(5,), lb=-1, ub=2)
        y = m.continuous("y", shape=(3,), lb=-big, ub=big)
        m.subject_to(y - x[1:4] * x[0:3] == 0)
        m.minimize(dm.sum(y))
        return m

    def slicing_sample(rng):
        x = rng.uniform(-1, 2, 5)
        return {"x": x, "y": x[1:4] * x[0:3]}

    def broadcast():
        m = dm.Model("broadcast")
        u = m.continuous("u", shape=(3,), lb=-1, ub=1)
        xm = m.continuous("X", shape=(3, 2), lb=0.5, ub=2)
        ym = m.continuous("Y", shape=(3, 2), lb=-big, ub=big)
        m.subject_to(ym - u[:, None] * xm == 0)
        m.minimize(dm.sum(ym))
        return m

    def broadcast_sample(rng):
        u = rng.uniform(-1, 1, 3)
        xm = rng.uniform(0.5, 2, (3, 2))
        return {"u": u, "X": xm, "Y": u[:, None] * xm}

    def axis_sum():
        m = dm.Model("axis_sum")
        xm = m.continuous("X", shape=(3, 2), lb=-1, ub=1)
        y = m.continuous("y", shape=(3,), lb=-big, ub=big)
        m.subject_to(y - dm.sum(xm * w, axis=1) == 0)
        m.minimize(dm.sum(y))
        return m

    def axis_sum_sample(rng):
        xm = rng.uniform(-1, 1, (3, 2))
        return {"X": xm, "y": (xm * w).sum(axis=1)}

    def partial_view():
        # An expandable exp row next to a refused ``maximum`` row: per-row mode.
        m = dm.Model("partial_view")
        x = m.continuous("x", shape=(3,), lb=-1, ub=1)
        y = m.continuous("y", shape=(3,), lb=-big, ub=big)
        u = m.continuous("u", shape=(3,), lb=-2, ub=2)
        v = m.continuous("v", shape=(3,), lb=-big, ub=big)
        m.subject_to(y - dm.exp(x) * x == 0)
        m.subject_to(v - dm.maximum(u, 0.5) == 0)
        m.minimize(dm.sum(y) + dm.sum(v))
        return m

    def partial_view_sample(rng):
        x = rng.uniform(-1, 1, 3)
        u = rng.uniform(-2, 2, 3)
        return {"x": x, "y": np.exp(x) * x, "u": u, "v": np.maximum(u, 0.5)}

    return [
        ("matmul", matmul, matmul_sample),
        ("matvec_var", matvec_var, matvec_var_sample),
        ("slicing", slicing, slicing_sample),
        ("broadcast", broadcast, broadcast_sample),
        ("axis_sum", axis_sum, axis_sum_sample),
        ("partial_view", partial_view, partial_view_sample),
    ]


@pytest.mark.parametrize("case", _op_cases(), ids=lambda c: c[0])
def test_differential_bound_across_array_ops(case):
    """§5 differential test beyond the sigmoid layer: matmul (constant and
    variable matrix), slicing, broadcasting, axis sums, and a hybrid partial view.
    Per random node box around a feasible point: ON box within OFF box, and the
    point inside both (no valid point cut)."""
    name, build, sample = case
    m = build()
    r = model_to_repr(m, getattr(m, "_builder", None))
    root_lb, root_ub = _box(m)
    rng = np.random.default_rng(sum(map(ord, name)))
    checks = strictly_tighter = 0
    n_boxes = 60
    for _ in range(n_boxes):
        pts = sample(rng)
        pt = np.concatenate([np.ravel(np.asarray(pts[v.name], dtype=float)) for v in m._variables])
        assert np.all(root_lb <= pt) and np.all(pt <= root_ub), name
        lb = root_lb + (pt - root_lb) * rng.uniform(0, 1, size=pt.size)
        ub = pt + (root_ub - pt) * rng.uniform(0, 1, size=pt.size)
        off = r.in_tree_presolve(lb.copy(), ub.copy(), 0, 1, expand_array_rows=False)
        on = r.in_tree_presolve(lb.copy(), ub.copy(), 0, 1, expand_array_rows=True)
        if name == "partial_view":
            assert str(on["array_rows"]).startswith("partial:"), on["array_rows"]
        else:
            assert on["array_rows"] == "expanded", (name, on["array_rows"])
        assert not off["infeasible"] and not on["infeasible"], name
        on_lb, on_ub = np.asarray(on["lb"]), np.asarray(on["ub"])
        off_lb, off_ub = np.asarray(off["lb"]), np.asarray(off["ub"])
        assert np.all(on_lb >= off_lb - 1e-9) and np.all(on_ub <= off_ub + 1e-9), name
        assert np.all(on_lb <= pt + 1e-7) and np.all(pt <= on_ub + 1e-7), name
        assert np.all(off_lb <= pt + 1e-7) and np.all(pt <= off_ub + 1e-7), name
        strictly_tighter += int(np.any(on_lb > off_lb + 1e-9) or np.any(on_ub < off_ub - 1e-9))
        checks += 6 * pt.size
    assert checks == n_boxes * 6 * len(root_lb)
    assert strictly_tighter > 0, name  # the expansion actually changed something


def test_unexpandable_row_default_solve_emits_no_warning(monkeypatch, caplog):
    """Default-ON: a model with a row ``expand`` refuses (``maximum``) must not
    WARN on an ordinary solve -- it is a missing reduction, not a fault -- and the
    note it does log names no internal expression-node id."""
    import logging

    import discopt.solver as S

    monkeypatch.delenv(FLAG, raising=False)
    S._IN_TREE_ARRAY_ROWS_DECLINED_SEEN.clear()
    # Nonconvex objective (sin(3x)*x) so spatial B&B must branch: the kernel runs
    # at many nodes, not only at a root that a faster machine might close first.
    m = dm.Model("max_row")
    x = m.continuous("x", shape=(3,), lb=-3, ub=3)
    y = m.continuous("y", shape=(3,), lb=-10, ub=10)
    m.subject_to(y - dm.maximum(x, 0.5) == 0)
    m.minimize(dm.sum(y) + dm.sum(dm.sin(3 * x) * x))
    # Set the level ON the emitting logger: caplog.at_level() without ``logger=``
    # only lowers the root, so a level another test left on ``discopt`` (e.g.
    # INFO) silently filters the DEBUG note -- the xdist failure on PR #1592.
    with caplog.at_level(logging.DEBUG, logger="discopt.solver"):
        res = m.solve(time_limit=30, max_nodes=40)
    assert res.objective is not None
    assert res.node_count > 1, res.node_count
    stats = res.solver_stats or {}
    # The kernel saw the refused row (separates "never ran" from "not logged").
    assert stats.get("reduce/array_rows_partial", 0) > 0, stats
    ours = [rec for rec in caplog.records if "DISCOPT_IN_TREE_ARRAY_ROWS" in rec.getMessage()]
    assert ours, "the partial-expansion note was never logged: the refused row never ran"
    for rec in ours:
        assert rec.levelno < logging.WARNING, rec.getMessage()
        assert not re.search(r"node \d+", rec.getMessage()), rec.getMessage()
    warned = [
        rec.getMessage()
        for rec in caplog.records
        if rec.levelno >= logging.WARNING and "array" in rec.getMessage().lower()
    ]
    assert not warned, warned
