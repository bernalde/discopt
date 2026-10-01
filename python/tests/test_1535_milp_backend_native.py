"""#1535: ``milp_backend=`` selects the engine for a pure LP / MILP.

``"native"`` forces discopt's own branch and bound so the search can be watched
(teaching, inspection); ``"highs"`` is the #1229 HiGHS route; ``None`` keeps the env
var / default. These tests pin the routing (which driver ran, with which options),
the options the native MILP tree honours (``strategy``, ``node_callback``,
``milp_cuts``), the route reported on the result, and every combination that would
otherwise quietly run a different engine than the one named.
"""

from __future__ import annotations

import discopt.modeling as dm
import discopt.solver as S
import numpy as np
import pytest

pytest.importorskip("highspy")


def _knapsack(n: int = 20, seed: int = 0, maximize: bool = True) -> dm.Model:
    rng = np.random.default_rng(seed)
    w = rng.integers(5, 30, n)
    v = rng.integers(5, 40, n)
    m = dm.Model("knapsack")
    x = m.binary("x", shape=(n,))
    value = sum(int(v[i]) * x[i] for i in range(n))
    if maximize:
        m.maximize(value)
    else:
        m.minimize(-value)
    m.subject_to(sum(int(w[i]) * x[i] for i in range(n)) <= int(w.sum() // 2))
    return m


def _lp() -> dm.Model:
    m = dm.Model("lp")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(-x - 2 * y)
    m.subject_to(x + y <= 8)
    m.subject_to(x - y >= -4)
    return m


def _spy(monkeypatch, name: str) -> list:
    """Record every call of ``discopt.solver.<name>`` and pass it through."""
    calls: list = []
    real = getattr(S, name)

    def wrapper(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(S, name, wrapper)
    return calls


@pytest.mark.smoke
def test_native_knapsack_branches_and_matches_highs():
    """The issue's acceptance: a 20-variable knapsack on the native tree explores
    nodes and certifies the HiGHS optimum."""
    ref = _knapsack().solve()
    res = _knapsack().solve(milp_backend="native")
    assert ref.status == res.status == "optimal"
    assert res.node_count > 0
    assert res.objective == pytest.approx(ref.objective, abs=1e-6)
    assert res.gap_certified
    assert res.algorithm_route.startswith("native-milp")


def test_native_milp_runs_the_python_tree_with_the_callers_options(monkeypatch):
    highs = _spy(monkeypatch, "_solve_milp_highs")
    engine = _spy(monkeypatch, "_solve_milp_simplex")
    tree = _spy(monkeypatch, "_solve_milp_bb")
    _knapsack().solve(milp_backend="native", strategy="depth_first", milp_cuts=False)
    assert highs == [] and engine == []
    assert len(tree) == 1
    args, kwargs = tree[0]
    assert args[4] == "depth_first"  # ``strategy`` is _solve_milp_bb's 5th positional
    assert kwargs["root_cuts"] is False


def test_default_route_is_unchanged(monkeypatch):
    monkeypatch.delenv("DISCOPT_LP_MILP_BACKEND", raising=False)
    tree = _spy(monkeypatch, "_solve_milp_bb")
    res = _knapsack().solve()
    assert tree == []
    assert res.node_count == 0 or res.algorithm_route.startswith("highs-milp")
    assert (res.solver_stats or {}).get("route/lp_milp_backend") == 1.0


def test_keyword_overrides_env_var_both_ways(monkeypatch):
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    res = _knapsack().solve(milp_backend="highs")
    assert (res.solver_stats or {}).get("route/lp_milp_backend") == 1.0
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "highs")
    res = _knapsack().solve(milp_backend="native")
    assert res.algorithm_route.startswith("native-milp")


def test_milp_cuts_false_skips_the_root_cut_loop():
    on = _knapsack().solve(milp_backend="native")
    off = _knapsack().solve(milp_backend="native", milp_cuts=False)
    cut_keys = lambda r: {k for k in (r.solver_stats or {}) if k.startswith("cuts/")}  # noqa: E731
    # The probe must have something to switch off, or the OFF arm proves nothing.
    assert cut_keys(on), "root cut loop added no cut on the knapsack; test instance is stale"
    assert cut_keys(off) == set()
    assert "root cuts off" in off.algorithm_route
    assert off.objective == pytest.approx(on.objective, abs=1e-6)
    assert off.status == "optimal" and off.gap_certified


@pytest.mark.parametrize("maximize", [True, False])
def test_node_callback_fires_with_values_in_the_users_sense(maximize):
    seen = []
    res = _knapsack(maximize=maximize).solve(
        milp_backend="native", milp_cuts=False, node_callback=lambda ctx, m: seen.append(ctx)
    )
    assert res.status == "optimal"
    assert len(seen) > 0
    counts = [c.node_count for c in seen]
    assert counts == sorted(counts) and counts[-1] <= res.node_count
    last = seen[-1]
    assert last.x_relaxation.shape == (20,)
    assert last.incumbent_obj == pytest.approx(res.objective, abs=1e-4)
    assert last.best_bound is not None
    for c in seen:
        if c.best_bound is None:
            continue
        # A dual bound never crosses the optimum: above it for max, below for min.
        if maximize:
            assert c.best_bound >= res.objective - 1e-6
        else:
            assert c.best_bound <= res.objective + 1e-6


def test_native_lp_uses_the_discopt_simplex():
    ref = _lp().solve()
    res = _lp().solve(milp_backend="native")
    assert res.status == "optimal"
    assert res.objective == pytest.approx(ref.objective, abs=1e-6)
    assert res.algorithm_route.startswith("native-lp")


def test_native_with_lazy_constraints_reports_the_spatial_route():
    """#748 keeps a MILP with lazy constraints on spatial B&B; the result says so."""
    calls = []

    def lazy(ctx, model):
        calls.append(1)
        return []

    res = _knapsack(n=8).solve(milp_backend="native", lazy_constraints=lazy)
    ref = _knapsack(n=8).solve()
    assert res.objective == pytest.approx(ref.objective, abs=1e-6)
    assert res.algorithm_route.startswith("native-spatial")
    assert calls


@pytest.mark.parametrize(
    "model, kwargs, match",
    [
        (_knapsack, {"milp_backend": "gurobi"}, "expected one of"),
        (_knapsack, {"milp_cuts": False}, "milp_backend='native'"),
        (_knapsack, {"milp_backend": "highs", "milp_cuts": False}, "milp_backend='native'"),
        (_knapsack, {"milp_backend": "native", "nlp_solver": "simplex"}, "nlp_solver"),
        (_knapsack, {"milp_backend": "native", "solver": "amp"}, "different engines"),
        (_lp, {"milp_backend": "native", "milp_cuts": False}, "does not run for an LP"),
        (
            _knapsack,
            {"milp_backend": "native", "milp_cuts": True, "lazy_constraints": lambda *a: []},
            "spatial",
        ),
    ],
)
def test_conflicting_or_meaningless_requests_are_refused(model, kwargs, match):
    with pytest.raises(ValueError, match=match):
        model().solve(**kwargs)


def test_refused_for_a_nonlinear_model():
    m = dm.Model("minlp")
    x = m.continuous("x", lb=0, ub=4)
    z = m.binary("z")
    m.minimize((x - 1.5) ** 2 + z)
    m.subject_to(x <= 3 * z + 1)
    with pytest.raises(ValueError, match="LP or MILP"):
        m.solve(milp_backend="native")


# ── branching rules ─────────────────────────────────────────────────────


def _multi_knapsack(n: int = 20, k: int = 5, seed: int = 0) -> dm.Model:
    """Several rows, so a node LP has several fractional columns to choose from (a
    single-row knapsack vertex has at most one, which makes every rule agree)."""
    rng = np.random.default_rng(seed)
    W = rng.integers(5, 40, (k, n))
    v = rng.integers(10, 50, n)
    m = dm.Model("multi_knapsack")
    x = m.binary("x", shape=(n,))
    m.maximize(sum(int(v[i]) * x[i] for i in range(n)))
    for r in range(k):
        m.subject_to(sum(int(W[r, i]) * x[i] for i in range(n)) <= int(W[r].sum() // 2))
    return m


def test_branching_rules_change_the_search_but_not_the_answer(monkeypatch):
    ref = _multi_knapsack().solve()
    hinted = []
    real = S._milp_branching_hints

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        hinted.append((args[0], len(out[0])))
        return out

    monkeypatch.setattr(S, "_milp_branching_hints", spy)
    nodes = {}
    for rule in S._MILP_BRANCHING_RULES:
        res = _multi_knapsack().solve(
            milp_backend="native", milp_cuts=False, branching_rule=rule, batch_size=1
        )
        assert res.status == "optimal" and res.gap_certified
        assert res.objective == pytest.approx(ref.objective, abs=1e-6)
        assert f"branching_rule={rule!r}" in res.algorithm_route
        nodes[rule] = res.node_count
        probes = (res.solver_stats or {}).get("branching/strong_probe_lps", 0)
        assert (probes > 0) == (rule == "strong")
    # Every non-default rule handed the tree hints (the probe fired), and the
    # rules did not all explore the same tree.
    for rule in ("most_fractional", "least_fractional", "strong"):
        assert sum(n for r, n in hinted if r == rule) > 0, rule
    assert not any(r == "pseudocost" for r, _ in hinted)
    assert len(set(nodes.values())) > 1, nodes


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"branching_rule": "strong"}, "milp_backend='native'"),
        ({"milp_backend": "native", "branching_rule": "random"}, "expected one of"),
        (
            {"milp_backend": "native", "branching_rule": "strong", "cut_callback": lambda c, m: []},
            "spatial",
        ),
    ],
)
def test_branching_rule_refusals(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _knapsack().solve(**kwargs)


def test_branching_rule_refused_on_an_lp():
    with pytest.raises(ValueError, match="does not run for an LP"):
        _lp().solve(milp_backend="native", branching_rule="strong")


# ── callbacks on a MILP without milp_backend ────────────────────────────


def test_node_callback_on_a_default_milp_runs_the_tree(monkeypatch):
    """Previously the HiGHS route returned with the callback never called."""
    monkeypatch.delenv("DISCOPT_LP_MILP_BACKEND", raising=False)
    highs = _spy(monkeypatch, "_solve_milp_highs")
    seen = []
    ref = _knapsack().solve()
    highs.clear()
    res = _knapsack().solve(node_callback=lambda ctx, m: seen.append(ctx))
    assert highs == []
    assert seen and res.node_count > 0
    assert res.algorithm_route.startswith("milp-tree")
    assert res.objective == pytest.approx(ref.objective, abs=1e-6)


def test_node_callback_with_explicit_highs_is_refused():
    with pytest.raises(ValueError, match="no node or cut hook"):
        _knapsack().solve(milp_backend="highs", node_callback=lambda ctx, m: None)


def test_cut_callback_on_a_milp_is_called():
    """Neither MILP engine nor the NLP-BB auto-select has a cut hook; the callback
    used to be silently skipped (0 calls) on every pure-MILP route."""
    calls = []

    def cut(ctx, model):
        calls.append(ctx.node_id)
        return []

    ref = _knapsack(n=10).solve()
    res = _knapsack(n=10).solve(cut_callback=cut)
    assert calls
    assert res.objective == pytest.approx(ref.objective, abs=1e-6)


def test_cut_callback_with_explicit_nlp_bb_is_refused():
    with pytest.raises(ValueError, match="nlp_bb=True"):
        _knapsack(n=10).solve(cut_callback=lambda ctx, m: [], nlp_bb=True)


def test_node_callback_on_an_lp_warns_that_it_is_not_called():
    with pytest.warns(UserWarning, match="node_callback not called"):
        _lp().solve(node_callback=lambda ctx, m: None)
