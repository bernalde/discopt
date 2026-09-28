"""#1520: the relaxation core's remaining ``except Exception`` handlers.

Continues #1514 (``test_1514_bound_handlers_propagate.py``) into the relaxation
layer proper: ``mccormick_lp``, ``incremental_mccormick``, ``milp_relaxation``,
``obbt``, ``lp_spatial_bb``, ``root_reduce`` and ``node_reduce``. Each handler is
now one of:

* **removed** -- the callee declines by return value (``solve_lp_warm_std`` returns
  ``None``, ``solve_assembled_full`` a non-optimal status, ``obbt_tighten_root`` the
  input box, a separator's ``milp.solve`` a non-optimal status), so an exception is
  a defect and propagates;
* **narrowed** -- the callee documents a decline exception (``model_to_repr``'s and
  ``interval_hessian``'s ``ValueError``, the incremental structure's ``ValueError`` /
  ``_IncrementalStructureTooLarge``), which is still
  absorbed; everything else propagates;
* **kept as a sound fallback** -- reported once per solve at WARNING on
  ``discopt.solver`` through ``warn_fallback_once``.

Every test that injects a failure counts the injected calls and asserts the count
is non-zero, so a fixture that stopped reaching the site fails here instead of
passing vacuously (CLAUDE.md §6).
"""

from __future__ import annotations

import logging
import types

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt._relax import _fallback
from discopt._relax import incremental_mccormick as incm
from discopt._relax import lp_spatial_bb as lsb
from discopt._relax import mccormick_lp as mlp
from discopt._relax import milp_relaxation as mr
from discopt._relax import node_reduce as nr
from discopt._relax import obbt as obbt_mod
from discopt._relax import root_reduce as rr
from discopt._relax.discretization import DiscretizationState
from discopt._relax.term_classifier import classify_nonlinear_terms


class _Boom(RuntimeError):
    """The injected defect; a distinct type so an assertion cannot match by luck."""


def _counting_raiser(exc_factory, calls):
    def _raise(*_a, **_k):
        calls.append(1)
        raise exc_factory()

    return _raise


def _bilinear() -> dm.Model:
    m = dm.Model("bilinear_1520")
    x = m.continuous("x", lb=-1.0, ub=2.0)
    y = m.continuous("y", lb=-1.0, ub=2.0)
    m.minimize(x * y + 0.1 * x)
    m.subject_to(x + y >= 0.5)
    return m


def _box():
    return np.array([-1.0, -1.0]), np.array([2.0, 2.0])


def _milp(m=None):
    m = m if m is not None else _bilinear()
    lb, ub = _box()
    milp, varmap = mr.build_milp_relaxation(
        m, classify_nonlinear_terms(m), DiscretizationState(), bound_override=(lb, ub)
    )
    return milp, varmap


def _inc():
    m = _bilinear()
    inc = incm.IncrementalMcCormickLP(m, classify_nonlinear_terms(m))
    assert inc.ok, inc.decline_reason
    return inc


@pytest.fixture
def fresh_fallbacks(monkeypatch, caplog):
    """An empty once-per-solve record, and DEBUG capture on ``discopt.solver``."""
    monkeypatch.setattr(_fallback.FALLBACK_WARNINGS, "seen", set())
    caplog.set_level(logging.DEBUG, logger="discopt.solver")
    return caplog


def _warnings(caplog, needle):
    return [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == "discopt.solver" and needle in r.getMessage()
    ]


# ── incremental_mccormick ────────────────────────────────────────────────────


def test_incremental_structure_declines_only_on_documented_types(monkeypatch):
    m = _bilinear()
    terms = classify_nonlinear_terms(m)
    calls: list[int] = []
    for decline in (
        lambda: ValueError("row-set mismatch"),
        lambda: incm._IncrementalStructureTooLarge("lift nnz"),
    ):
        monkeypatch.setattr(
            incm.IncrementalMcCormickLP, "_validate", _counting_raiser(decline, calls)
        )
        inc = incm.IncrementalMcCormickLP(m, terms)
        assert inc.ok is False and inc.decline_reason
    assert len(calls) == 2
    monkeypatch.setattr(
        incm.IncrementalMcCormickLP, "_validate", _counting_raiser(lambda: _Boom("v"), calls)
    )
    with pytest.raises(_Boom, match="v"):
        incm.IncrementalMcCormickLP(m, terms)
    assert len(calls) == 3


def test_incremental_column_identities_defect_propagates(monkeypatch):
    m = _bilinear()
    calls: list[int] = []
    monkeypatch.setattr(mlp, "column_identities", _counting_raiser(lambda: _Boom("ids"), calls))
    with pytest.raises(_Boom, match="ids"):
        incm.IncrementalMcCormickLP(m, classify_nonlinear_terms(m))
    assert calls


def test_incremental_lp_solves_propagate_solver_defects(monkeypatch):
    from discopt.solvers import milp_simplex

    inc = _inc()
    lb, ub = _box()
    A, b, bounds = inc.assemble(lb, ub)
    calls: list[int] = []
    monkeypatch.setattr(
        milp_simplex, "solve_lp_warm_std", _counting_raiser(lambda: _Boom("lp"), calls)
    )
    with pytest.raises(_Boom):
        inc.solve_assembled(A, b, bounds)
    with pytest.raises(_Boom):
        inc.solve_assembled_full(A, b, bounds)
    assert len(calls) == 2


# ── milp_relaxation ──────────────────────────────────────────────────────────


def test_warm_lp_paths_propagate_solver_defects(monkeypatch):
    from discopt.solvers import milp_simplex

    milp, _ = _milp()
    calls: list[int] = []
    monkeypatch.setattr(
        milp_simplex, "solve_lp_warm_std", _counting_raiser(lambda: _Boom("warm"), calls)
    )
    with pytest.raises(_Boom):
        milp._solve_lp_warm()
    with pytest.raises(_Boom):
        milp._solve_lp_warm_equilibrated()
    assert len(calls) == 2


def test_equilibration_defect_propagates(monkeypatch):
    milp, _ = _milp()
    calls: list[int] = []
    monkeypatch.setattr(
        mr, "equilibrate_relaxation_lp", _counting_raiser(lambda: _Boom("equil"), calls)
    )
    with pytest.raises(_Boom, match="equil"):
        milp._solve_lp_warm_equilibrated()
    assert calls


def test_warm_lp_marginal_extraction_defect_propagates(monkeypatch):
    """The ``want_marginals`` reduced-cost block had its own catch-all."""
    from discopt.solvers import milp_simplex

    real = milp_simplex.solve_lp_warm_std
    calls: list[int] = []

    class _BadDual:
        def __array__(self, *a, **k):
            calls.append(1)
            raise _Boom("dual")

    def _fake(*a, **k):
        res, basis, cert = real(*a, **k)
        return res, basis, cert._replace(dual=_BadDual())

    monkeypatch.setattr(milp_simplex, "solve_lp_warm_std", _fake)
    milp, _ = _milp()
    with pytest.raises(_Boom, match="dual"):
        milp._solve_lp_warm(want_marginals=True)
    assert calls


def test_multivar_box_curvature_narrowed_to_value_error(monkeypatch):
    from discopt._relax.convexity import interval_ad
    from discopt._relax.convexity.interval import Interval

    m = dm.Model("curv_1520")
    x = m.continuous("x", lb=0.5, ub=2.0)
    y = m.continuous("y", lb=0.5, ub=2.0)
    expr = dm.exp(x + y)
    m.minimize(expr)
    box = {v: Interval(np.array([float(v.lb)]), np.array([float(v.ub)])) for v in m._variables}
    flb, fub = np.array([0.5, 0.5]), np.array([2.0, 2.0])
    calls: list[int] = []
    monkeypatch.setattr(
        interval_ad,
        "interval_hessian",
        _counting_raiser(lambda: interval_ad.IntervalHessianTooLarge("big"), calls),
    )
    assert mr._multivar_box_curvature(expr, m, [0, 1], flb, fub, box) is None
    # ``RecursionError`` is no longer a decline: ``interval_hessian`` runs deep
    # walks on a large stack (#1520 root fix), so one reaching here is a defect.
    monkeypatch.setattr(
        interval_ad, "interval_hessian", _counting_raiser(lambda: RecursionError("deep"), calls)
    )
    with pytest.raises(RecursionError, match="deep"):
        mr._multivar_box_curvature(expr, m, [0, 1], flb, fub, box)
    monkeypatch.setattr(
        interval_ad, "interval_hessian", _counting_raiser(lambda: _Boom("h"), calls)
    )
    with pytest.raises(_Boom, match="h"):
        mr._multivar_box_curvature(expr, m, [0, 1], flb, fub, box)
    assert len(calls) == 3
    # The box handed in is restored on the propagating exit too.
    assert float(box[x].lo[0]) == 0.5 and float(box[x].hi[0]) == 2.0


# ── mccormick_lp: construction and the incremental node path ────────────────


def test_relaxer_incremental_construction_defect_propagates(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(
        incm, "IncrementalMcCormickLP", _counting_raiser(lambda: _Boom("inc"), calls)
    )
    with pytest.raises(_Boom, match="inc"):
        mlp.MccormickLPRelaxer(_bilinear())
    assert calls


def test_composite_lift_probe_defect_propagates(monkeypatch):
    from discopt._relax import uniform_relax

    relaxer = mlp.MccormickLPRelaxer(_bilinear(), build_incremental=False)
    calls: list[int] = []
    monkeypatch.setattr(
        uniform_relax, "build_uniform_relaxation", _counting_raiser(lambda: _Boom("u"), calls)
    )
    with pytest.raises(_Boom, match="u"):
        relaxer._model_has_composite_lift()
    assert calls


def _relaxer_with_inc():
    relaxer = mlp.MccormickLPRelaxer(_bilinear())
    assert relaxer._inc is not None and relaxer._inc.ok
    return relaxer


def test_incremental_node_assemble_defect_propagates(monkeypatch):
    relaxer = _relaxer_with_inc()
    calls: list[int] = []
    monkeypatch.setattr(relaxer._inc, "assemble", _counting_raiser(lambda: _Boom("asm"), calls))
    with pytest.raises(_Boom, match="asm"):
        relaxer._try_incremental_node(*_box(), None)
    assert calls


def test_incremental_node_marginals_defect_propagates(monkeypatch):
    relaxer = _relaxer_with_inc()
    inc = relaxer._inc
    real = inc.solve_assembled_full
    calls: list[int] = []

    class _BadDual:
        def __array__(self, *a, **k):
            calls.append(1)
            raise _Boom("marg")

    class _BadCert:
        safe_bound = None
        col_status = None
        dual = _BadDual()

    def _fake(*a, **k):
        out = real(*a, **k)
        return (*out[:5], _BadCert())

    monkeypatch.setattr(inc, "solve_assembled_full", _fake)
    with pytest.raises(_Boom, match="marg"):
        relaxer._try_incremental_node(*_box(), None, want_marginals=True)
    assert calls


def test_incremental_node_cold_resolve_defect_propagates(monkeypatch):
    """The warm-started ``infeasible`` re-solve (C-38) had its own catch-all."""
    relaxer = _relaxer_with_inc()
    inc = relaxer._inc
    lb, ub = _box()
    nrows = inc.assemble(lb, ub)[0].shape[0]
    relaxer._inc_warm_basis = (np.zeros(1), np.zeros(1))
    relaxer._inc_basis_nrows = nrows
    calls: list[int] = []

    def _fake(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            return ("infeasible", None, None, None, False)
        raise _Boom("cold")

    monkeypatch.setattr(inc, "solve_assembled_full", _fake)
    with pytest.raises(_Boom, match="cold"):
        relaxer._try_incremental_node(lb, ub, None)
    assert len(calls) == 2


def test_reverify_infeasible_defects_propagate(monkeypatch):
    relaxer = _relaxer_with_inc()
    inc = relaxer._inc
    A, b, bounds = inc.assemble(*_box())
    calls: list[int] = []
    monkeypatch.setattr(sp, "csr_matrix", _counting_raiser(lambda: _Boom("csr"), calls))
    with pytest.raises(_Boom, match="csr"):
        relaxer._reverify_incremental_infeasible(inc, A, b, bounds)
    monkeypatch.undo()
    monkeypatch.setattr(
        mr, "equilibrate_relaxation_lp", _counting_raiser(lambda: _Boom("eq"), calls)
    )
    with pytest.raises(_Boom, match="eq"):
        relaxer._reverify_incremental_infeasible(inc, A, b, bounds)
    monkeypatch.undo()
    monkeypatch.setattr(inc, "solve_assembled_full", _counting_raiser(lambda: _Boom("rv"), calls))
    with pytest.raises(_Boom, match="rv"):
        relaxer._reverify_incremental_infeasible(inc, A, b, bounds)
    assert len(calls) == 3


def test_integer_ratio_partition_defect_propagates():
    relaxer = mlp.MccormickLPRelaxer(_bilinear(), build_incremental=False)
    calls: list[int] = []
    relaxer._integer_ratio_partitioner = types.SimpleNamespace(
        node_bound=_counting_raiser(lambda: _Boom("ratio"), calls)
    )
    res = mlp.MccormickLPResult(status="optimal", lower_bound=0.0, x=np.zeros(2))
    with pytest.raises(_Boom, match="ratio"):
        relaxer._apply_integer_ratio_partition(res, *_box(), None)
    assert calls


def test_cold_pool_column_identity_defects_propagate(monkeypatch):
    """Both cold-path ``column_identities`` calls (pool inheritance and capture)."""
    relaxer = mlp.MccormickLPRelaxer(_bilinear(), build_incremental=False)
    lb, ub = _box()
    milp, _ = _milp()
    ncol = int(np.size(milp._c))
    pool = (sp.csr_matrix(np.ones((1, ncol))), np.array([1e6]), tuple(range(ncol)))
    calls: list[int] = []
    monkeypatch.setattr(mlp, "column_identities", _counting_raiser(lambda: _Boom("pool"), calls))
    with pytest.raises(_Boom, match="pool"):
        relaxer.solve_at_node(lb, ub, inherited_cuts=pool, separate=False)
    assert len(calls) == 1

    # Capture: a separation round that appends rows reaches the tagging call.
    m = dm.Model("sq_1520")
    x = m.continuous("x", lb=-10.0, ub=10.0)
    y = m.continuous("y", lb=-10.0, ub=10.0)
    m.minimize(x**2 + y**2 - 4.0 * x - 2.0 * y)
    m.subject_to(x + y >= 1.0)
    relaxer2 = mlp.MccormickLPRelaxer(m, build_incremental=False)
    with pytest.raises(_Boom, match="pool"):
        relaxer2.solve_at_node(
            np.array([-10.0, -10.0]), np.array([10.0, 10.0]), separate=True, out_cuts=[]
        )
    assert len(calls) == 2


# ── mccormick_lp: the per-node separators ───────────────────────────────────


class _BoomMilp:
    """A node LP whose first attribute read raises: reached only past each
    separator's gates, where the removed ``try`` used to begin."""

    def __init__(self, calls):
        self._calls = calls

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        self._calls.append(name)
        raise _Boom(f"milp.{name}")


class _Tuning:
    def __init__(self, real, **over):
        self._real = real
        self._over = over

    def __getattr__(self, name):
        return self._over.get(name, getattr(self._real, name))


def _opt_res(n=8):
    return types.SimpleNamespace(status="optimal", x=np.zeros(n), objective=0.0)


@pytest.mark.parametrize(
    "method, varmap, setup",
    [
        ("_separate_multilinear", {}, {"tuning": {"multilinear_separate": True}}),
        (
            "_separate_univariate_square",
            {"monomial": {(0, 2): 2}},
            {"tuning": {"square_separate": True}},
        ),
        ("_separate_singular_tangent", {"singular_tangent_relaxations": [object()]}, {}),
        (
            "_separate_convex",
            {
                "composite_multivar_relaxations": [
                    types.SimpleNamespace(
                        value_fn=types.SimpleNamespace(numpy_native=True),
                        grad_fn=object(),
                        idxs=[0],
                    )
                ]
            },
            {},
        ),
        ("_separate_rlt", {}, {"attr": {"_rlt_cuts": True}}),
        ("_separate_psd", {}, {"attr": {"_psd_cuts": True}}),
        (
            "_separate_edge_concave",
            {},
            {"tuning": {"edge_concave": True}, "attr": {"_ec_blocks": [object()]}},
        ),
    ],
)
def test_separator_defects_propagate(monkeypatch, method, varmap, setup):
    relaxer = mlp.MccormickLPRelaxer(_bilinear(), build_incremental=False)
    if "tuning" in setup:
        real = mlp._tuning()
        monkeypatch.setattr(mlp, "_tuning", lambda: _Tuning(real, **setup["tuning"]))
    for k, v in setup.get("attr", {}).items():
        setattr(relaxer, k, v)
    calls: list = []
    with pytest.raises(_Boom, match="milp"):
        getattr(relaxer, method)(_BoomMilp(calls), varmap, _opt_res(), None)
    assert calls, f"{method} returned before reaching the node LP"


def test_g_convex_flag_read_defect_propagates(monkeypatch):
    from discopt._relax.convexity import g_convex_inject

    relaxer = mlp.MccormickLPRelaxer(_bilinear(), build_incremental=False)
    calls: list[int] = []
    monkeypatch.setattr(
        g_convex_inject, "g_convex_cuts_enabled", _counting_raiser(lambda: _Boom("flag"), calls)
    )
    with pytest.raises(_Boom, match="flag"):
        relaxer._g_convex_enabled()
    assert calls


def test_g_convex_separator_import_and_certify_defects_propagate(monkeypatch):
    from discopt._relax.convexity import g_convex_inject, g_convexity

    relaxer = mlp.MccormickLPRelaxer(_bilinear(), build_incremental=False)
    lb, ub = _box()
    res = _opt_res(2)
    calls: list = []
    # Import: a module-level ``__getattr__`` serves the name the separator imports.
    monkeypatch.delattr(g_convex_inject, "rigorous_g_convex_cut_coeffs")
    monkeypatch.setattr(
        g_convex_inject, "__getattr__", _counting_raiser(lambda: _Boom("imp"), calls), raising=False
    )
    with pytest.raises(_Boom, match="imp"):
        relaxer._separate_g_convex(types.SimpleNamespace(_c=np.zeros(2)), res, lb, ub, None)
    monkeypatch.undo()
    assert len(calls) == 1

    monkeypatch.setattr(
        g_convexity, "certify_g_convex", _counting_raiser(lambda: _Boom("cert"), calls)
    )
    milp = types.SimpleNamespace(_c=np.zeros(3), _A_ub=None, _b_ub=None)
    with pytest.raises(_Boom, match="cert"):
        relaxer._separate_g_convex(milp, res, lb, ub, None)
    assert len(calls) == 2


def test_convex_oa_evaluation_failure_warns_once(monkeypatch, fresh_fallbacks):
    """Kept: a composite's ``value_fn`` (native tape / JAX) raising only skips that
    tangent -- now at WARNING once, not DEBUG."""
    relaxer = mlp.MccormickLPRelaxer(_bilinear(), build_incremental=False)
    calls: list[int] = []
    fn = _counting_raiser(lambda: _Boom("tape"), calls)
    fn.numpy_native = True
    spec = types.SimpleNamespace(value_fn=fn, grad_fn=fn, idxs=[0], aux_col=2)
    varmap = {"composite_multivar_relaxations": [spec, spec]}
    milp = types.SimpleNamespace(_c=np.zeros(3), _A_ub=None, _b_ub=None)
    res = _opt_res(3)
    assert relaxer._separate_convex(milp, varmap, res, None) is res
    assert len(calls) == 2
    warned = _warnings(fresh_fallbacks, "composite OA cut evaluation")
    assert len(warned) == 1 and "_Boom" in warned[0].getMessage()


# ── obbt ─────────────────────────────────────────────────────────────────────


def test_obbt_scoring_lp_defect_propagates(monkeypatch):
    milp, _ = _milp()
    calls: list[int] = []
    boom = _counting_raiser(lambda: _Boom("score"), calls)
    monkeypatch.setattr(obbt_mod, "get_exact_dual_lp_solver", lambda: boom)
    with pytest.raises(_Boom, match="score"):
        obbt_mod.run_obbt_on_relaxation(milp, 2, top_k=0)
    assert calls


def _eq_defined() -> dm.Model:
    """``v`` is open and defined by an equality over bounded ``x``."""
    m = dm.Model("eqdef_1520")
    x = m.continuous("x", lb=1.0, ub=2.0)
    v = m.continuous("v")
    m.subject_to(v - dm.exp(x) == 0.0)
    m.minimize(v + x)
    return m


def test_equality_propagation_bound_defect_propagates(monkeypatch):
    from discopt._relax import gdp_reformulate

    m = _eq_defined()
    lb = np.array([1.0, -np.inf])
    ub = np.array([2.0, np.inf])
    lb_out, ub_out, n = obbt_mod.propagate_equality_defined_bounds(m, lb, ub)
    assert n >= 1 and np.isfinite(ub_out[1])  # the pass does reach the row

    calls: list[int] = []
    monkeypatch.setattr(
        gdp_reformulate, "_bound_expression", _counting_raiser(lambda: _Boom("bnd"), calls)
    )
    saved = [(np.copy(v.lb), np.copy(v.ub)) for v in m._variables]
    with pytest.raises(_Boom, match="bnd"):
        obbt_mod.propagate_equality_defined_bounds(m, lb, ub)
    assert len(calls) == 1
    for v, (olb, oub) in zip(m._variables, saved):  # ``finally`` still restores
        assert np.array_equal(v.lb, olb) and np.array_equal(v.ub, oub)


def test_bootstrap_finite_bounds_defect_propagates(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(
        obbt_mod, "_extract_linear_constraints", _counting_raiser(lambda: _Boom("lin"), calls)
    )
    with pytest.raises(_Boom, match="lin"):
        obbt_mod.bootstrap_finite_bounds(
            _eq_defined(), np.array([1.0, -np.inf]), np.array([2.0, np.inf])
        )
    assert calls


def test_obbt_tighten_root_dbbt_defect_propagates(monkeypatch):
    """Both the DBBT-pass handler and the whole-function catch-all are gone."""
    calls: list[int] = []
    monkeypatch.setattr(
        obbt_mod, "dbbt_on_relaxation", _counting_raiser(lambda: _Boom("dbbt"), calls)
    )
    with pytest.raises(_Boom, match="dbbt"):
        obbt_mod.obbt_tighten_root(_bilinear(), *_box(), incumbent_cutoff=10.0)
    assert calls


def test_obbt_tighten_root_probe_defect_propagates(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(
        obbt_mod, "run_obbt_on_relaxation", _counting_raiser(lambda: _Boom("probe"), calls)
    )
    with pytest.raises(_Boom, match="probe"):
        obbt_mod.obbt_tighten_root(_bilinear(), *_box())
    assert calls


def test_obbt_envelope_build_failure_warns_once(monkeypatch, fresh_fallbacks):
    """Kept: the root envelope build (same builder and policy as the node relaxer's
    cold build, which reports at WARNING)."""
    calls: list[int] = []
    monkeypatch.setattr(mlp, "build_milp_relaxation", _counting_raiser(lambda: _Boom("env"), calls))
    lb, ub = _box()
    for _ in range(2):
        r = obbt_mod.obbt_tighten_root(_bilinear(), lb, ub)
        assert r.infeasible is False
        assert np.array_equal(r.lb, lb) and np.array_equal(r.ub, ub)
    assert len(calls) == 2
    warned = _warnings(fresh_fallbacks, "root OBBT envelope build")
    assert len(warned) == 1 and "_Boom" in warned[0].getMessage()


def test_dbbt_envelope_rebuild_failure_warns_once(monkeypatch, fresh_fallbacks):
    real_build = mr.build_milp_relaxation
    calls: list[int] = []

    def _build(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            return real_build(*a, **k)
        raise _Boom("rebuild")

    lb, ub = _box()

    def _dbbt(*a, **k):
        return obbt_mod.ObbtResult(
            tightened_lb=np.array([-0.5, -1.0]),
            tightened_ub=ub.copy(),
            n_lp_solves=1,
            n_tightened=1,
            total_lp_time=0.0,
        )

    monkeypatch.setattr(mlp, "build_milp_relaxation", _build)
    monkeypatch.setattr(obbt_mod, "dbbt_on_relaxation", _dbbt)
    r = obbt_mod.obbt_tighten_root(_bilinear(), lb, ub, incumbent_cutoff=10.0)
    assert len(calls) == 2
    assert r.lb[0] == -0.5  # the DBBT tightening already applied is kept
    warned = _warnings(fresh_fallbacks, "root DBBT envelope rebuild")
    assert len(warned) == 1 and "_Boom" in warned[0].getMessage()


# ── lp_spatial_bb ────────────────────────────────────────────────────────────


def _int_bilinear() -> dm.Model:
    m = dm.Model("int_1520")
    x = m.integer("x", lb=-2, ub=3)
    y = m.integer("y", lb=-2, ub=3)
    m.minimize(x * y + 0.5 * x)
    m.subject_to(x + y >= 1)
    return m


def test_lp_spatial_cold_relax_bound_defect_propagates(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(mr, "build_milp_relaxation", _counting_raiser(lambda: _Boom("cold"), calls))
    m = _bilinear()
    with pytest.raises(_Boom, match="cold"):
        lsb._relax_bound(m, classify_nonlinear_terms(m), *_box())
    assert calls


def _node_lp():
    inc = _inc()
    lb, ub = _box()
    A, b, bounds = inc.assemble(lb, ub)
    _, x, _ = inc.solve_assembled(A, b, bounds)
    return A, b, bounds, x, inc


@pytest.mark.parametrize(
    "module, name",
    [
        ("discopt._relax.crossover", "crossover_to_vertex"),
        ("discopt._relax.cmir_cuts", "separate_cmir"),
    ],
)
def test_lp_spatial_cut_separator_defects_propagate(monkeypatch, module, name):
    import importlib

    A, b, bounds, x, inc = _node_lp()
    calls: list[int] = []
    monkeypatch.setattr(
        importlib.import_module(module), name, _counting_raiser(lambda: _Boom(name), calls)
    )
    with pytest.raises(_Boom, match=name):
        lsb._separate_node_cuts(A, b, bounds, x, inc.ncol, inc.c)
    assert calls


def test_lp_spatial_cut_separator_import_defect_propagates(monkeypatch):
    from discopt._relax import cmir_cuts

    A, b, bounds, x, inc = _node_lp()
    calls: list[int] = []
    monkeypatch.delattr(cmir_cuts, "separate_cmir")
    monkeypatch.setattr(
        cmir_cuts, "__getattr__", _counting_raiser(lambda: _Boom("imp"), calls), raising=False
    )
    with pytest.raises(_Boom, match="imp"):
        lsb._separate_node_cuts(A, b, bounds, x, inc.ncol, inc.c)
    assert calls


def test_lp_spatial_evaluator_defect_propagates(monkeypatch):
    import discopt._tape_nlp_evaluator as tne

    calls: list[int] = []
    monkeypatch.setattr(tne, "make_evaluator", _counting_raiser(lambda: _Boom("ev"), calls))
    with pytest.raises(_Boom, match="ev"):
        lsb.solve_lp_spatial_bb(_int_bilinear(), time_limit=10.0)
    assert calls


def test_lp_spatial_root_obbt_defect_propagates(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(
        obbt_mod, "obbt_tighten_root", _counting_raiser(lambda: _Boom("robbt"), calls)
    )
    with pytest.raises(_Boom, match="robbt"):
        lsb.solve_lp_spatial_bb(_int_bilinear(), time_limit=10.0, use_obbt=True)
    assert calls


# ── root_reduce / node_reduce ────────────────────────────────────────────────


def test_root_cutoff_fbbt_declines_only_on_missing_repr(monkeypatch):
    import discopt._rust as rust

    m = _bilinear()
    lb, ub = _box()
    calls: list[int] = []
    monkeypatch.setattr(rust, "model_to_repr", _counting_raiser(lambda: ValueError("x"), calls))
    out_lb, out_ub, n, infeas = rr._stage_fbbt_with_cutoff(m, lb, ub, None, max_iter=5, tol=1e-9)
    assert calls and n == 0 and infeas is False
    assert np.array_equal(out_lb, lb) and np.array_equal(out_ub, ub)
    monkeypatch.setattr(rust, "model_to_repr", _counting_raiser(lambda: _Boom("rr"), calls))
    with pytest.raises(_Boom, match="rr"):
        rr._stage_fbbt_with_cutoff(m, lb, ub, None, max_iter=5, tol=1e-9)
    assert len(calls) == 2
    assert float(m._variables[0].lb) == -1.0  # saved_bounds restored on the raise


def _raising_relaxer_init(calls):
    def _init(self, *a, **k):
        calls.append(1)
        raise _Boom("ctor")

    return _init


def test_obbt_relaxer_constructor_failure_warns_once(monkeypatch, fresh_fallbacks):
    """Kept, as in the solver's relaxer setups (#1514 / 07ceed3): a relaxer that
    cannot be constructed skips the sweep and returns the box, reported once."""
    calls: list[int] = []
    monkeypatch.setattr(mlp.MccormickLPRelaxer, "__init__", _raising_relaxer_init(calls))
    lb, ub = _box()
    for _ in range(2):
        r = obbt_mod.obbt_tighten_root(_bilinear(), lb, ub)
        assert r.infeasible is False
        assert np.array_equal(r.lb, lb) and np.array_equal(r.ub, ub)
    assert len(calls) == 2
    assert len(_warnings(fresh_fallbacks, "relaxer setup (root OBBT) failed")) == 1


def test_root_lp_bound_relaxer_constructor_failure_warns(monkeypatch, fresh_fallbacks):
    calls: list[int] = []
    monkeypatch.setattr(mlp.MccormickLPRelaxer, "__init__", _raising_relaxer_init(calls))
    assert rr._root_lp_bound(_bilinear(), *_box()) is None
    assert calls
    assert len(_warnings(fresh_fallbacks, "relaxer setup (root fixpoint) failed")) == 1


def test_root_lp_bound_defect_propagates(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(
        mlp.MccormickLPRelaxer, "solve_at_node", _counting_raiser(lambda: _Boom("lpb"), calls)
    )
    with pytest.raises(_Boom, match="lpb"):
        rr._root_lp_bound(_bilinear(), *_box())
    assert calls


def test_root_stage_obbt_defect_propagates(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(obbt_mod, "obbt_tighten_root", _counting_raiser(lambda: _Boom("s3"), calls))
    with pytest.raises(_Boom, match="s3"):
        rr._stage_obbt(_bilinear(), *_box(), None, rounds=1, deadline=None, prefer_pounce=False)
    assert calls


def test_node_cutoff_fbbt_declines_only_on_missing_repr(monkeypatch):
    import discopt._rust as rust

    m = _bilinear()
    lb, ub = _box()
    calls: list[int] = []
    monkeypatch.setattr(rust, "model_to_repr", _counting_raiser(lambda: ValueError("x"), calls))
    res = nr.reduce_node(m, lb, ub, None, None, do_fbbt=True)
    assert calls and res.n_tightened == 0 and not res.infeasible
    monkeypatch.setattr(rust, "model_to_repr", _counting_raiser(lambda: _Boom("nr"), calls))
    with pytest.raises(_Boom, match="nr"):
        nr.reduce_node(m, lb, ub, None, None, do_fbbt=True)
    assert len(calls) == 2


def test_node_cutoff_fbbt_uses_the_repr_objective_space(monkeypatch):
    """Found while narrowing: ``_fbbt_on_node`` passed the INTERNAL (minimization-
    space) cutoff straight to ``fbbt_with_cutoff``, which builds the row under the
    repr's own sense -- #1373's false-certificate mode on a MAXIMIZE model. The root
    stage already converts with ``repr_space_cutoff``; the node stage now does too."""
    import discopt._rust as rust

    real = rust.model_to_repr
    seen: list = []

    class _Spy:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def fbbt_with_cutoff(self, *, max_iter, tol, incumbent_bound):
            seen.append(incumbent_bound)
            return self._inner.fbbt_with_cutoff(
                max_iter=max_iter, tol=tol, incumbent_bound=incumbent_bound
            )

    monkeypatch.setattr(rust, "model_to_repr", lambda *a, **k: _Spy(real(*a, **k)))
    m = dm.Model("max_1520")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    z = m.continuous("z", lb=-10.0, ub=10.0)
    m.subject_to(x * y <= 50.0)
    m.subject_to(z >= x + 1.0)
    m.maximize(-z + 0.5)  # optimum -0.5 at x = 0, z = 1; internal cutoff +0.5
    res = nr.reduce_node(
        m, np.array([0.0, 0.0, -10.0]), np.array([10.0, 10.0, 10.0]), None, 0.5, do_fbbt=True
    )
    assert seen == [-0.5], seen  # the repr (maximize) space: f >= -0.5
    assert not res.infeasible


def test_bound_expression_power_overflow_is_an_infinite_range():
    """Root fix for the ``OverflowError`` the equality pass used to catch: a float
    integer power past the float range is a signed ``inf``, not a raise."""
    import discopt.modeling as dm
    from discopt._relax.gdp_reformulate import _bound_expression

    m = dm.Model("pow_1520")
    x = m.continuous("x", lb=-1e200, ub=1e200)
    assert _bound_expression(x**2, m) == (0.0, np.inf)
    assert _bound_expression(x**3, m) == (-np.inf, np.inf)
    y = m.continuous("y", lb=1e200, ub=2e200)
    assert _bound_expression(y**2, m) == (np.inf, np.inf)
