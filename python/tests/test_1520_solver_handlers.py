"""#1520: the ``solver.py`` handlers #1514 did not list no longer swallow defects.

Same three outcomes as #1514 (``test_1514_bound_handlers_propagate.py``):

* **removed** -- the callee declines by return value (or has no failure mode), so a
  raise is a defect and now fails the solve;
* **narrowed** -- only the documented decline is still absorbed;
* **kept** -- a genuine external failure with a sound fallback, now reported once
  per solve at WARNING through ``warn_fallback_once``.

Every injected fake counts its calls and the test asserts the count, so a model
that stopped reaching a site fails here instead of passing vacuously (CLAUDE.md §6).
Where a callee is shared by several sites, the fake raises only when called from
the function under test, so the test pins *that* site.
"""

from __future__ import annotations

import logging
import sys
import time

import discopt._rust as rust
import discopt.modeling as dm
import discopt.solver as solver
import numpy as np
import pytest
from discopt._relax import factorable_reform as fr
from discopt._relax import node_reduce as nr
from discopt._relax import problem_classifier as pc
from discopt._relax import term_classifier as tc


class _Boom(RuntimeError):
    """The injected defect; a distinct type so an assertion cannot match by luck."""


def _raise_from(caller: str, real, calls: list, msg: str, after_local: str | None = None):
    """A fake for ``real`` that raises ``_Boom(msg)`` only when called from ``caller``
    -- and, with ``after_local``, only once that local is bound in the caller, which
    pins one call site among several in a long function such as ``solve_model``."""

    def _fake(*a, **k):
        frame = sys._getframe(1)
        if frame.f_code.co_name == caller and (
            after_local is None or after_local in frame.f_locals
        ):
            calls.append(1)
            raise _Boom(msg)
        return real(*a, **k)

    return _fake


def _wave() -> dm.Model:
    """Nonconvex, needs branching (several node-loop iterations)."""
    m = dm.Model("wave_1520")
    x = m.continuous("x", lb=-3.0, ub=3.0)
    y = m.continuous("y", lb=-3.0, ub=3.0)
    m.minimize(dm.sin(3 * x) * y + 0.1 * x * x)
    m.subject_to(x + y >= -1.0)
    return m


def _bilinear() -> dm.Model:
    m = dm.Model("bilinear_1520")
    x = m.continuous("x", lb=-1.0, ub=2.0)
    y = m.continuous("y", lb=-1.0, ub=2.0)
    m.minimize(x * y + 0.1 * x)
    m.subject_to(x + y >= 0.5)
    return m


def _milp() -> dm.Model:
    m = dm.Model("milp_1520")
    x = m.integer("x", lb=0, ub=10)
    y = m.integer("y", lb=0, ub=10)
    m.maximize(3 * x + 2 * y)
    m.subject_to(2 * x + y <= 9.5)
    m.subject_to(x + 3 * y <= 12.5)
    return m


def _pure_integer_nonlinear() -> dm.Model:
    """No continuous variable: reaches the OBBT nonlinearity gate in ``solve_model``."""
    m = dm.Model("int_nl_1520")
    x = m.integer("x", lb=0, ub=20)
    y = m.integer("y", lb=0, ub=20)
    m.minimize(-x * y + 0.3 * x * x - y)  # nonconvex: stays on the spatial path
    m.subject_to(x + 2 * y <= 25)
    return m


class _TreeProxy:
    """Wraps a real ``PyTreeManager`` so one method can be replaced."""

    real_cls = solver.PyTreeManager

    def __init__(self, *a, **k):
        object.__setattr__(self, "_t", self.real_cls(*a, **k))

    def __getattr__(self, name):
        return getattr(self._t, name)


# ── removed ────────────────────────────────────────────────────────────────


def test_set_node_bounds_defect_fails_the_solve(monkeypatch):
    """The issue's named site: the Phase-2 child-box export used to log a failed
    ``set_node_bounds`` at DEBUG and branch on the unreduced box."""
    monkeypatch.setenv("DISCOPT_PHASE2_DBBT", "1")
    staged: list[int] = []
    exported: list[int] = []

    def _fake_reduce(model, lb, ub, lp, cutoff, do_fbbt=False):
        staged.append(1)  # a (no-op) reduction, so the box is staged for export
        return nr.NodeReduceResult(lb=lb.copy(), ub=ub.copy(), n_tightened=1)

    class _Proxy(_TreeProxy):
        def set_node_bounds(self, *a):
            exported.append(1)
            raise _Boom("set_node_bounds")

    monkeypatch.setattr(nr, "reduce_node", _fake_reduce)
    monkeypatch.setattr(solver, "PyTreeManager", _Proxy)
    with pytest.raises(_Boom, match="set_node_bounds"):
        _wave().solve(time_limit=60, deterministic=True, max_nodes=50)
    assert staged and exported, "the child-box export was never reached"


def test_milp_root_lp_bound_defect_fails_the_solve(monkeypatch):
    """The issue's other named site: the root LP relaxation bound on the monolithic
    Rust MILP route. The fake raises only for the integer-free root re-solve."""
    real = rust.solve_milp_csc_py
    main: list[int] = []
    root: list[int] = []

    def _fake(*a, **k):
        integer_cols = a[9]
        if len(integer_cols) == 0:
            root.append(1)
            raise _Boom("root lp")
        main.append(1)
        return real(*a, **k)

    monkeypatch.setattr(rust, "solve_milp_csc_py", _fake)
    with pytest.raises(_Boom, match="root lp"):
        solver._solve_milp_simplex(_milp(), 30.0, 1e-4, 1000, time.perf_counter())
    assert main and root, "the root-bound re-solve was never reached"


def test_milp_root_lp_bound_still_computed():
    """The removal is not a behaviour change on the success path."""
    res = solver._solve_milp_simplex(_milp(), 30.0, 1e-4, 1000, time.perf_counter())
    assert res.root_bound is not None and np.isfinite(res.root_bound)


def test_native_spatial_kernel_defect_fails_the_solve(monkeypatch):
    """A raise from the native kernel used to reroute the solve to the Python tree
    silently; ``build_spatial_kernel_spec`` returning ``None`` is the decline."""
    calls: list[int] = []

    def _fake(**_k):
        calls.append(1)
        raise _Boom("kernel")

    monkeypatch.setattr(rust, "solve_spatial_tree_py", _fake)
    with pytest.raises(_Boom, match="kernel"):
        _bilinear().solve(time_limit=30, deterministic=True)
    assert calls, "the native spatial kernel was never reached"


def test_problem_classification_defect_fails_the_solve(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(
        pc,
        "classify_problem",
        _raise_from("solve_model", pc.classify_problem, calls, "classify"),
    )
    with pytest.raises(_Boom, match="classify"):
        _bilinear().solve(time_limit=30, deterministic=True)
    assert calls


def test_convex_minlp_route_classification_defect_propagates(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(
        pc,
        "classify_problem",
        _raise_from("_convex_minlp_auto_route", pc.classify_problem, calls, "route"),
    )
    m = dm.Model("route_1520")
    x = m.continuous("x", lb=0.0, ub=4.0)
    b = m.binary("b")
    m.minimize((x - 1.5) ** 2 + b)
    m.subject_to(x <= 3 * b + 0.5)
    with pytest.raises(_Boom, match="route"):
        solver._convex_minlp_auto_route(m)
    assert calls


def test_obbt_nonlinearity_gate_defect_fails_the_solve(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(
        tc,
        "classify_nonlinear_terms",
        _raise_from(
            "solve_model",
            tc.classify_nonlinear_terms,
            calls,
            "terms",
            after_local="_obbt_has_continuous",
        ),
    )
    with pytest.raises(_Boom, match="terms"):
        _pure_integer_nonlinear().solve(time_limit=30, deterministic=True)
    assert calls


def test_obbt_iterate_quadratic_gate_defect_fails_the_solve(monkeypatch):
    monkeypatch.setenv("DISCOPT_OBBT_ITERATE", "1")
    calls: list[int] = []
    monkeypatch.setattr(
        tc,
        "classify_nonlinear_terms",
        _raise_from(
            "solve_model",
            tc.classify_nonlinear_terms,
            calls,
            "iterate",
            after_local="_obbt_min_impr",
        ),
    )
    with pytest.raises(_Boom, match="iterate"):
        _wave().solve(time_limit=30, deterministic=True, max_nodes=50)
    assert calls


def test_rlt_sparse_admit_classification_defect_propagates(monkeypatch):
    real_tuning = solver._tuning()

    class _Tun:
        rlt_sparse_auto = True
        rlt_sparse_max_vars = 10_000

        def __getattr__(self, name):
            return getattr(real_tuning, name)

    calls: list[int] = []
    monkeypatch.setattr(solver, "_tuning", lambda: _Tun())
    monkeypatch.setattr(
        tc,
        "classify_nonlinear_terms",
        _raise_from("_rlt_sparse_admit", tc.classify_nonlinear_terms, calls, "rlt"),
    )
    with pytest.raises(_Boom, match="rlt"):
        solver._rlt_sparse_admit(_bilinear(), 2)
    assert calls


def test_deep_solve_depth_count_defect_fails_the_solve(monkeypatch):
    """The old ``depth = 0`` fallback ran a possibly deep model on the small stack."""
    calls: list[int] = []

    def _fake(_model):
        calls.append(1)
        raise _Boom("depth")

    monkeypatch.setattr(fr, "_max_expr_node_count", _fake)
    with pytest.raises(_Boom, match="depth"):
        _bilinear().solve(time_limit=30, deterministic=True)
    assert calls


def test_convexity_refresh_import_failure_fails_the_solve(monkeypatch):
    """A broken in-tree import used to disable per-node convexity refresh silently."""
    import discopt._relax.convexity as conv

    monkeypatch.delattr(conv, "refresh_convex_mask")
    with pytest.raises(ImportError, match="refresh_convex_mask"):
        _wave().solve(time_limit=30, deterministic=True, max_nodes=50)


def test_reduce_node_import_failure_fails_the_solve(monkeypatch):
    """A broken in-tree import used to switch Phase-2 DBBT off silently."""
    monkeypatch.setenv("DISCOPT_PHASE2_DBBT", "1")
    monkeypatch.delattr(nr, "reduce_node")
    with pytest.raises(ImportError, match="reduce_node"):
        _wave().solve(time_limit=30, deterministic=True, max_nodes=50)


class _Ev:
    """Minimal evaluator stand-in carrying a model."""

    def __init__(self, model):
        self._model = model
        self._constraint_flat_sizes = None


def test_structural_linear_mask_defect_propagates(monkeypatch):
    """A raise here used to become ``None`` -- which the caller read as "use the
    numeric linearity test alone", the unsound #27a path."""
    calls: list[int] = []

    def _fake(*_a):
        calls.append(1)
        raise _Boom("mask")

    monkeypatch.setattr(solver, "_structural_linear_row_mask", _fake)
    with pytest.raises(_Boom, match="mask"):
        solver._cached_structural_linear_mask(_Ev(_bilinear()), 1)
    assert calls


def test_unaligned_structural_mask_forgoes_linear_fbbt(monkeypatch):
    """When the mask cannot be aligned (``None``), no row is treated as linear.

    ``x - y == 0`` is linear; with ``y`` in ``[0, 1]`` linear FBBT would cut ``x``'s
    ``[0, 10]`` down to ``[0, 1]``. With the mask unavailable that tightening must
    not come from the numeric test alone.
    """
    from discopt._relax.nlp_evaluator import NLPEvaluator

    m = dm.Model("lin_1520")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.minimize(x + y)
    m.subject_to(x - y == 0)
    ev = NLPEvaluator(m)
    lb, ub = np.array([0.0, 0.0]), np.array([10.0, 1.0])
    # Isolate the Jacobian-row FBBT: the structural nonlinear pass (sound on its
    # own) would otherwise tighten ``x`` too.
    monkeypatch.setattr(
        solver,
        "_apply_nonlinear_tightening_with_status",
        lambda _m, lo, hi: (np.asarray(lo, float), np.asarray(hi, float), False),
    )

    t_lb, t_ub, _ = solver._tighten_node_bounds_with_status(ev, lb, ub, [0.0], [0.0])
    assert t_ub[0] <= 1.0 + 1e-6, "control: linear FBBT should tighten x here"

    calls: list[int] = []

    def _none(*_a):
        calls.append(1)
        return None

    monkeypatch.setattr(solver, "_structural_linear_row_mask", _none)
    ev2 = NLPEvaluator(m)
    t_lb, t_ub, _ = solver._tighten_node_bounds_with_status(ev2, lb, ub, [0.0], [0.0])
    assert calls
    assert t_ub[0] == pytest.approx(10.0), "linear FBBT ran on an unverified row"


# ── narrowed: memo writes absorb only AttributeError ───────────────────────


class _Slotted:
    __slots__ = ("_model", "_constraint_flat_sizes")

    def __init__(self, model):
        self._model = model
        self._constraint_flat_sizes = None


class _Hostile:
    def __init__(self, model):
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_constraint_flat_sizes", None)
        object.__setattr__(self, "calls", [])

    def __setattr__(self, name, value):
        self.calls.append(1)
        raise _Boom("setattr")


def test_structural_mask_memo_absorbs_only_attribute_error():
    mask = solver._cached_structural_linear_mask(_Slotted(_bilinear()), 1)
    assert mask is not None and mask.shape == (1,)
    hostile = _Hostile(_bilinear())
    with pytest.raises(_Boom, match="setattr"):
        solver._cached_structural_linear_mask(hostile, 1)
    assert hostile.calls


# ── kept: sound fallbacks, reported once per solve at WARNING ──────────────


def _warnings_from(caplog, needle):
    return [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == "discopt.solver" and needle in r.getMessage()
    ]


def test_convex_quadratic_objective_test_warns_on_evaluator_failure(caplog):
    caplog.set_level(logging.DEBUG, logger="discopt.solver")
    m = dm.Model("cq_1520")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    y = m.continuous("y", lb=-1.0, ub=1.0)
    m.minimize(x * x + y * y)

    calls: list[int] = []

    class _BadEv:
        def evaluate_hessian(self, _x):
            calls.append(1)
            raise _Boom("hessian")

    solver._FALLBACK_WARNINGS.seen = set()
    assert solver._objective_is_convex_quadratic(m, _BadEv(), 2) is False
    assert solver._objective_is_convex_quadratic(m, _BadEv(), 2) is False
    assert len(calls) == 2
    warned = _warnings_from(caplog, "convex-quadratic objective test failed")
    assert len(warned) == 1 and "_Boom" in warned[0].getMessage()


def test_rlt_root_gain_probe_warns_and_declines(monkeypatch, caplog):
    from discopt._relax.mccormick_lp import MccormickLPRelaxer

    caplog.set_level(logging.DEBUG, logger="discopt.solver")
    calls: list[int] = []

    def _raise(self, *a, **k):
        calls.append(1)
        raise _Boom("rlt probe")

    monkeypatch.setattr(MccormickLPRelaxer, "solve_at_node", _raise)
    solver._FALLBACK_WARNINGS.seen = set()
    assert solver._rlt_root_gain(_bilinear()) is None
    assert calls
    assert len(_warnings_from(caplog, "RLT root-gain probe failed")) == 1
