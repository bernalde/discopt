"""#1520: relaxation-layer classifier/tightening handlers no longer swallow defects.

The ``except Exception`` handlers in the convexity package, the problem/term
classifiers, the uniform relaxation, nonlinear bound tightening and the factorable
reformulation each put into one of three states (the #1514 scheme):

* **removed** -- the callee already declines by return value, so an exception is a
  defect and propagates;
* **narrowed** -- only the callee's documented decline is absorbed;
* **kept as a sound fallback** -- a genuinely external failure (SciPy's SLSQP/NNLS,
  a user-registered operator lowering), now reported once per solve at WARNING via
  ``discopt._relax._fallback.warn_fallback_once``.

A convexity classifier that absorbs an error must answer "not convex", never
"convex": one handler (the declared-box cache invalidation in ``patterns``) could
leave a STALE box behind and was removed for that reason.

Every injection test counts the injected calls and asserts the count is non-zero,
so a fixture that stopped reaching the site fails instead of passing vacuously
(CLAUDE.md section 6).
"""

from __future__ import annotations

import builtins
import logging
import sys

import discopt._rust as _rust
import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import _fallback
from discopt._relax import factorable_reform as fr
from discopt._relax import nonlinear_bound_tightening as nbt
from discopt._relax import problem_classifier as pc
from discopt._relax import term_classifier as tc
from discopt._relax.convexity import certificate as cert_mod
from discopt._relax.convexity import g_convex_inject as gci
from discopt._relax.convexity import interval_eval as ie
from discopt._relax.convexity import patterns as pat
from discopt._relax.convexity import rules
from discopt._relax.convexity import signomial_global as sg
from discopt._relax.convexity.lattice import Curvature


class _Boom(RuntimeError):
    """The injected defect; a distinct type so an assertion cannot match by luck."""


def _raiser(calls, exc_factory=lambda: _Boom("injected #1520")):
    def _raise(*_a, **_k):
        calls.append(1)
        raise exc_factory()

    return _raise


@pytest.fixture(autouse=True)
def _fresh_fallback_record():
    _fallback.FALLBACK_WARNINGS.seen.clear()
    yield
    _fallback.FALLBACK_WARNINGS.seen.clear()


def _lp() -> dm.Model:
    m = dm.Model("lp_1520")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    m.minimize(x + 2.0 * y)
    m.subject_to(x + y >= 1.0)
    return m


def _qp() -> dm.Model:
    m = dm.Model("qp_1520")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    m.minimize(x**2 + y**2 + x * y + x)
    m.subject_to(x + y <= 1.0)
    return m


def _exp_model() -> dm.Model:
    """``exp(x^2 + y^2)``: a DCP-convex composite over two variables."""
    m = dm.Model("exp_1520")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    y = m.continuous("y", lb=-1.0, ub=1.0)
    m.minimize(dm.exp(x**2 + y**2))
    m.subject_to(x + y >= -1.5)
    return m


# ── certificate.py ─────────────────────────────────────────────────────────


def test_qp_objective_certificate_extraction_defect_propagates(monkeypatch):
    """removed: ``certify_quadratic_objective_convex`` used to answer "not proven"
    for any exception in the QP extractor."""
    calls: list[int] = []
    monkeypatch.setattr(pc, "extract_qp_data", _raiser(calls))
    with pytest.raises(_Boom):
        cert_mod.certify_quadratic_objective_convex(_qp())
    assert calls


def test_qp_objective_certificate_still_proves_a_convex_qp():
    assert cert_mod.certify_quadratic_objective_convex(_qp()) is True


def test_refresh_convex_mask_certificate_defect_propagates(monkeypatch):
    """removed: ``refresh_convex_mask`` turned a raising certificate into "None"."""
    m = dm.Model("refresh_1520")
    x = m.continuous("x", lb=0.5, ub=2.0)
    y = m.continuous("y", lb=0.5, ub=2.0)
    m.minimize(x + y)
    m.subject_to(x * y >= 1.0)
    calls: list[int] = []
    monkeypatch.setattr(cert_mod, "certify_convex", _raiser(calls))
    with pytest.raises(_Boom):
        cert_mod.refresh_convex_mask(m, [False], np.array([0.5, 0.5]), np.array([2.0, 2.0]))
    assert calls


# ── rules.py ───────────────────────────────────────────────────────────────


def test_is_nonneg_int_constant_defect_propagates():
    calls: list[int] = []

    class _BadConst:
        @property
        def value(self):
            calls.append(1)
            raise _Boom("value")

    with pytest.raises(_Boom):
        rules._is_nonneg_int_constant(_BadConst())
    assert calls


def test_hash_index_narrowed_to_asarray_refusals():
    calls: list[int] = []

    class _Unhashable:
        __hash__ = None  # type: ignore[assignment]

        def __array__(self, *a, **k):
            calls.append(1)
            raise _Boom("array")

    with pytest.raises(_Boom):
        rules._hash_index(_Unhashable())
    assert calls
    # The documented refusal (a ragged, unhashable index) keeps the identity key.
    ragged = [[1], [1, 2]]
    assert rules._hash_index(ragged) == ("id", id(ragged))


@pytest.mark.parametrize("target", ["fractional", "certificate"])
def test_classify_constraint_fallback_defects_propagate(monkeypatch, target):
    """removed: the fractional-epigraph recognizer and the numeric certificate both
    decline by value inside ``classify_constraint``."""
    m = dm.Model("cc_1520")
    x = m.continuous("x", lb=0.5, ub=2.0)
    y = m.continuous("y", lb=0.5, ub=2.0)
    m.minimize(x + y)
    m.subject_to(dm.sin(x) * y <= 1.0)  # syntactically UNKNOWN
    calls: list[int] = []
    if target == "fractional":
        monkeypatch.setattr(pat, "classify_fractional_epigraph_constraint", _raiser(calls))
    else:
        monkeypatch.setattr(cert_mod, "certify_convex", _raiser(calls))
    with pytest.raises(_Boom):
        rules.classify_constraint(m._constraints[0], m, use_certificate=True)
    assert calls


def test_objective_certificate_defect_propagates(monkeypatch):
    m = dm.Model("obj_1520")
    x = m.continuous("x", lb=0.5, ub=2.0)
    y = m.continuous("y", lb=0.5, ub=2.0)
    m.minimize(dm.sin(x) * y)
    calls: list[int] = []
    monkeypatch.setattr(cert_mod, "certify_convex", _raiser(calls))
    with pytest.raises(_Boom):
        rules._objective_is_convex(m, {}, use_certificate=True, exact_qp=False)
    assert calls


def test_recursion_headroom_defect_propagates():
    calls: list[int] = []

    class _BadVar:
        @property
        def lb(self):
            calls.append(1)
            raise _Boom("lb")

    class _FakeModel:
        _constraints: list = []
        _variables = [_BadVar()]

    with pytest.raises(_Boom):
        rules._recursion_headroom_need(_FakeModel())
    assert calls


# ── patterns.py ────────────────────────────────────────────────────────────


def test_declared_box_cache_invalidation_failure_is_not_absorbed():
    """removed (UNSOUND fallback): a failed invalidation used to leave a stale box
    behind, against which a recognizer could prove what the real box does not."""
    calls: list[int] = []

    class _Weird:
        def __delattr__(self, name):
            calls.append(1)
            raise _Boom("nope")

    w = _Weird()
    w.__dict__[pat._DECLARED_BOX_CACHE_ATTR] = (1, None, None)
    with pytest.raises(_Boom):
        pat.clear_declared_box_cache(w)
    assert calls


def test_declared_box_memo_narrowed_to_attribute_error():
    calls: list[int] = []
    m = _lp()

    class _NoMemo:
        __slots__ = ("_variables",)

    slotted = _NoMemo()
    slotted._variables = m._variables
    lo, hi = pat._box_bounds(slotted)  # AttributeError on the memo: rebuilt, fine
    np.testing.assert_array_equal(lo, [0.0, 0.0])
    np.testing.assert_array_equal(hi, [4.0, 4.0])

    class _Hostile:
        def __init__(self, variables):
            object.__setattr__(self, "_variables", variables)

        def __setattr__(self, name, value):
            calls.append(1)
            raise _Boom("setattr")

    with pytest.raises(_Boom):
        pat._box_bounds(_Hostile(m._variables))
    assert calls


def test_affine_lower_bound_narrowed(monkeypatch):
    m = _lp()
    x = m._variables[0]
    assert pat._affine_lower_bound(x * x, m) is None  # documented decline
    calls: list[int] = []
    monkeypatch.setattr(pc, "_extract_linear_coefficients", _raiser(calls))
    with pytest.raises(_Boom):
        pat._affine_lower_bound(x + 1.0, m)
    assert calls


def test_quadratic_data_and_sign_form_narrowed(monkeypatch):
    m = _lp()
    x = m._variables[0]
    assert pat._quadratic_data(dm.exp(x), m) is None  # documented decline
    assert pat._quadratic_sign_form(dm.exp(x), m) is None
    calls: list[int] = []
    monkeypatch.setattr(pc, "_extract_quadratic_coefficients", _raiser(calls))
    with pytest.raises(_Boom):
        pat._quadratic_data(x * x, m)
    assert calls
    calls2: list[int] = []
    monkeypatch.setattr(pc, "_extract_quadratic_terms", _raiser(calls2))
    with pytest.raises(_Boom):
        pat._quadratic_sign_form(x * x, m)
    assert calls2


def test_quadratic_extractors_array_constant_decline_is_absorbed():
    """Found by the narrowing: the quadratic walk's constant folding declines a
    non-scalar array coefficient with ``_NotLinearError`` (``_eval_const``), so the
    quadratic sites absorb both extraction declines and the verdict stays UNKNOWN
    (test_944's mixed-sign case)."""
    m = dm.Model("arr_1520")
    x = m.continuous("x", shape=(3,), lb=-5.0, ub=5.0)
    expr = np.array([1.0, -2.0, 3.0]) * (x * x)
    with pytest.raises(pc._NotLinearError):
        pc._extract_quadratic_terms(expr, m, 3)
    assert pat._quadratic_data(expr, m) is None
    assert pat._quadratic_sign_form(expr, m) is None
    assert rules.classify_expr(expr, m, {}) == Curvature.UNKNOWN


def test_affine_square_sum_matrix_narrowed(monkeypatch):
    m = _lp()
    x, y = m._variables
    # documented decline: a square of a non-affine base
    assert pat._affine_square_sum_matrix(dm.exp(x) ** 2 + y**2, m) is None
    calls: list[int] = []
    monkeypatch.setattr(pc, "_extract_linear_coefficients", _raiser(calls))
    with pytest.raises(_Boom):
        pat._affine_square_sum_matrix((x - y) ** 2 + (x + 1.0) ** 2, m)
    assert calls


def test_fractional_epigraph_extraction_narrowed(monkeypatch):
    m = dm.Model("frac_1520")
    x = m.continuous("x", lb=1.0, ub=2.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.minimize(y)
    m.subject_to(x**2 + 1.0 - x * y <= 0.0)
    c = m._constraints[0]
    calls: list[int] = []
    monkeypatch.setattr(pc, "_extract_linear_coefficients", _raiser(calls))
    with pytest.raises(_Boom):
        pat.classify_fractional_epigraph_constraint(c, m)
    assert calls


def test_fractional_epigraph_decline_still_absorbed(monkeypatch):
    m = dm.Model("frac_decline_1520")
    x = m.continuous("x", lb=1.0, ub=2.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.minimize(y)
    m.subject_to(x**2 + 1.0 - x * y <= 0.0)
    calls: list[int] = []

    def _decline(*_a, **_k):
        calls.append(1)
        raise pc._NotLinearError("declined")

    monkeypatch.setattr(pc, "_extract_linear_coefficients", _decline)
    assert pat.classify_fractional_epigraph_constraint(m._constraints[0], m) is None
    assert calls


# ── g_convex_inject.py ─────────────────────────────────────────────────────


def _gconv_model() -> dm.Model:
    m = dm.Model("gconv_1520")
    x = m.continuous("x", lb=0.5, ub=2.0)
    y = m.continuous("y", lb=0.5, ub=2.0)
    m.minimize(x + y)
    m.subject_to(x * y <= 2.0)
    return m


def test_g_convex_injector_certificate_defect_propagates(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(gci, "certify_g_convex", _raiser(calls))
    with pytest.raises(_Boom):
        gci.inject_g_convex_cuts(_gconv_model())
    assert calls


def test_g_convex_injector_body_defect_propagates(monkeypatch):
    class _Cert:
        kind = "g_convex"
        rho = 1.0

    monkeypatch.setattr(gci, "certify_g_convex", lambda *a, **k: _Cert())
    calls: list[int] = []
    monkeypatch.setattr(gci, "_inject_for_body", _raiser(calls))
    with pytest.raises(_Boom):
        gci.inject_g_convex_cuts(_gconv_model())
    assert calls


def test_g_convex_cut_residual_enclosure_defect_propagates(monkeypatch):
    m = _gconv_model()
    x, y = m._variables
    box = gci._declared_box(m)
    x0 = np.array([1.0, 1.0])
    # Sanity: the unpatched construction reaches the residual enclosure and succeeds.
    assert gci.rigorous_g_convex_cut_coeffs(m, x * y - 2.0, 1.0, x0, box) is not None
    calls: list[int] = []
    monkeypatch.setattr(gci, "evaluate_interval", _raiser(calls))
    with pytest.raises(_Boom):
        gci.rigorous_g_convex_cut_coeffs(m, x * y - 2.0, 1.0, x0, box)
    assert calls


# ── interval_eval.py (kept) ────────────────────────────────────────────────


def test_registered_operator_lowering_failure_warns_once(monkeypatch, caplog):
    """kept: a user-registered lowering that raises leaves the enclosure unbounded
    (sound) and is reported at WARNING on ``discopt.solver`` once per solve."""
    import discopt.operators as ops

    calls: list[int] = []

    class _Fn:
        def interval_expr(self, arg):
            calls.append(1)
            raise _Boom("user lowering")

    monkeypatch.setattr(ops, "get_registered", lambda name: _Fn())
    m = _lp()
    x = m._variables[0]
    expr = dm.exp(x)  # any single-argument call; only ``args`` is read
    caplog.set_level(logging.DEBUG, logger="discopt.solver")
    assert ie._registered_interval("myop", expr, m, {}, {}) is None
    assert ie._registered_interval("myop", expr, m, {}, {}) is None
    assert len(calls) == 2
    warnings = [
        r for r in caplog.records if r.name == "discopt.solver" and r.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert "user lowering" in warnings[0].getMessage()


# ── signomial_global.py (kept) ─────────────────────────────────────────────


def _sig_constrained():
    m = dm.Model("sig_1520")
    x = m.continuous("x", lb=0.5, ub=3.0)
    y = m.continuous("y", lb=0.5, ub=3.0)
    m.minimize(x**2 + y**2 - 3.0 * x * y)
    m.subject_to(x * y >= 1.0)
    struct = sg.classify_signomial_global(m)
    obj = sg._pack(struct.terms)
    cons = [sg._pack(t) for t in struct.constraint_terms]
    return obj, cons, struct.u_lb, struct.u_ub


def _warnings(caplog):
    return [
        r for r in caplog.records if r.name == "discopt.solver" and r.levelno == logging.WARNING
    ]


@pytest.mark.parametrize("which", ["minimize", "nnls"])
def test_signomial_node_bound_scipy_failures_warn_and_stay_valid(monkeypatch, caplog, which):
    import scipy.optimize as so

    obj, cons, u_lb, u_ub = _sig_constrained()
    ref, _ = sg._constrained_node_bound(obj, cons, u_lb, u_ub)
    calls: list[int] = []
    monkeypatch.setattr(so, which, _raiser(calls))
    caplog.set_level(logging.DEBUG, logger="discopt.solver")
    lb, _ = sg._constrained_node_bound(obj, cons, u_lb, u_ub)
    assert calls
    assert np.isfinite(lb) and lb <= ref + 1e-9  # a looser, still valid bound
    assert len(_warnings(caplog)) == 1


def test_signomial_corner_bound_nnls_failure_warns(monkeypatch, caplog):
    import scipy.optimize as so

    calls: list[int] = []
    monkeypatch.setattr(so, "nnls", _raiser(calls))
    caplog.set_level(logging.DEBUG, logger="discopt.solver")
    lb = sg._corner_lagrangian_bound(
        1.0,
        np.array([1.0, 0.0]),
        [0.5],
        [np.array([0.0, 1.0])],
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
        np.array([0.5, 0.5]),
    )
    assert calls
    assert lb == pytest.approx(0.5)  # lam = 0: L = f, corner min over the box
    assert len(_warnings(caplog)) == 1


def test_signomial_linear_certificate_slsqp_failure_warns(monkeypatch, caplog):
    import scipy.optimize as so

    obj, cons, u_lb, u_ub = _sig_constrained()
    calls: list[int] = []
    monkeypatch.setattr(so, "minimize", _raiser(calls))
    caplog.set_level(logging.DEBUG, logger="discopt.solver")
    lb = sg._cert_min_linear(np.array([1.0, 0.0]), [(p, 0.0) for p in cons], u_lb, u_ub)
    assert calls
    assert np.isfinite(lb)  # certified at the box midpoint instead
    assert len(_warnings(caplog)) == 1


# ── problem_classifier.py ──────────────────────────────────────────────────


def test_classify_problem_narrowed_to_repr_declines(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(_rust, "model_to_repr", _raiser(calls))
    with pytest.raises(_Boom):
        pc.classify_problem(_lp())
    assert calls
    for decline in (ValueError, TypeError):
        monkeypatch.setattr(_rust, "model_to_repr", _raiser(calls, lambda d=decline: d("no")))
        assert pc.classify_problem(_lp()) == pc.ProblemClass.NLP


def test_model_repr_or_decline(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(_rust, "model_to_repr", _raiser(calls, lambda: ValueError("no repr")))
    with pytest.raises(pc._NotLinearError):
        pc._model_repr_or_decline(_lp(), pc._NotLinearError)
    monkeypatch.setattr(_rust, "model_to_repr", _raiser(calls))
    with pytest.raises(_Boom):
        pc._model_repr_or_decline(_lp(), pc._NotLinearError)
    assert len(calls) == 2


def test_lp_extraction_ladder_narrowed(monkeypatch):
    ref = pc.extract_lp_data(_lp())
    calls: list[int] = []
    monkeypatch.setattr(pc, "_extract_lp_data_from_repr", _raiser(calls))
    with pytest.raises(_Boom):
        pc.extract_lp_data(_lp())
    # a documented decline on the first rung still falls through to the second
    monkeypatch.setattr(
        pc, "_extract_lp_data_from_repr", _raiser(calls, lambda: pc._NotLinearError("no"))
    )
    np.testing.assert_allclose(pc.extract_lp_data(_lp()).c, ref.c)
    monkeypatch.setattr(pc, "extract_lp_data_algebraic", _raiser(calls))
    with pytest.raises(_Boom):
        pc.extract_lp_data(_lp())
    assert len(calls) == 4


def test_qp_extraction_ladder_narrowed(monkeypatch):
    ref = pc.extract_qp_data(_qp())
    calls: list[int] = []
    monkeypatch.setattr(pc, "_extract_qp_data_symbolic", _raiser(calls))
    with pytest.raises(_Boom):
        pc.extract_qp_data(_qp())
    monkeypatch.setattr(
        pc, "_extract_qp_data_symbolic", _raiser(calls, lambda: pc._NotQuadraticError("no"))
    )
    np.testing.assert_allclose(pc.dense_Q(pc.extract_qp_data(_qp()).Q), pc.dense_Q(ref.Q))
    monkeypatch.setattr(pc, "extract_qp_data_algebraic", _raiser(calls))
    with pytest.raises(_Boom):
        pc.extract_qp_data(_qp())
    assert len(calls) == 4


def test_qcp_repr_rung_narrowed(monkeypatch):
    m = _qp()
    m.subject_to(m._variables[0] ** 2 + m._variables[1] ** 2 <= 3.0)
    ref = pc.extract_qcp_data(m)
    m._builder = object()  # only the gate reads it: the rung itself is patched
    calls: list[int] = []
    monkeypatch.setattr(pc, "_extract_qcp_data_from_repr", _raiser(calls))
    with pytest.raises(_Boom):
        pc.extract_qcp_data(m)
    monkeypatch.setattr(
        pc, "_extract_qcp_data_from_repr", _raiser(calls, lambda: pc._NotQuadraticError("no"))
    )
    got = pc.extract_qcp_data(m)
    assert len(got.quadratic_constraints) == len(ref.quadratic_constraints) == 1
    assert len(calls) == 2


# ── term_classifier.py ─────────────────────────────────────────────────────


def test_term_classifier_import_narrowed_to_import_error(monkeypatch):
    real_import = builtins.__import__
    calls: list[int] = []

    def _broken(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "discopt._rust" and fromlist and "model_to_repr" in fromlist:
            calls.append(1)
            raise _Boom("import side effect")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _broken)
    with pytest.raises(_Boom):
        tc._classify_nonlinear_terms_rust(_qp())
    assert calls


# ── uniform_relax.py ───────────────────────────────────────────────────────


def test_uniform_relax_dcp_defect_propagates(monkeypatch):
    import discopt._relax.convexity as conv
    from discopt._relax.uniform_relax import build_uniform_relaxation

    calls: list[int] = []
    monkeypatch.setattr(conv, "classify_expr", _raiser(calls))
    with pytest.raises(_Boom):
        build_uniform_relaxation(_exp_model())
    assert calls


def test_uniform_relax_convex_lift_import_failure_propagates(monkeypatch):
    import sys

    import discopt._relax.milp_relaxation as real
    from discopt._relax.uniform_relax import build_uniform_relaxation

    calls: list[int] = []
    site_names = {"_LIFT_MAX_CROSS_TERM_ARG_MAGNITUDE", "_build_convexity_box"}

    class _Stub:
        """Breaks only the names this site imports; everything else is real."""

        def __getattr__(self, name):
            if name in site_names:
                calls.append(1)
                raise AttributeError(name)
            return getattr(real, name)

    monkeypatch.setitem(sys.modules, "discopt._relax.milp_relaxation", _Stub())
    with pytest.raises(ImportError):
        build_uniform_relaxation(_exp_model())
    assert calls


def test_uniform_relax_jax_compile_narrowed(monkeypatch):
    import discopt._relax.dag_compiler as dc
    from discopt._relax.uniform_relax import build_uniform_relaxation

    monkeypatch.setenv("DISCOPT_SEPGRAD", "jax")
    calls: list[int] = []
    monkeypatch.setattr(dc, "compile_expression", _raiser(calls))
    with pytest.raises(_Boom):
        build_uniform_relaxation(_exp_model())
    assert calls
    # the documented refusal abstains from the lift; the build still succeeds
    monkeypatch.setattr(dc, "compile_expression", _raiser(calls, lambda: ValueError("op")))
    build_uniform_relaxation(_exp_model())
    assert len(calls) >= 2


def test_uniform_relax_analytic_probe_narrowed(monkeypatch):
    import discopt._relax.convexity.interval_ad as iad
    from discopt._relax.uniform_relax import build_uniform_relaxation

    monkeypatch.setenv("DISCOPT_ANALYTIC_SEPGRAD", "1")
    real = iad.interval_hessian
    calls: list[int] = []
    monkeypatch.setattr(iad, "interval_hessian", _raiser(calls))
    with pytest.raises(_Boom):
        build_uniform_relaxation(_exp_model())
    assert calls
    # the documented ValueError decline falls through to the tape; build succeeds
    declines: list[int] = []

    def _decline(*a, **k):
        declines.append(1)
        raise ValueError("array leaves")

    monkeypatch.setattr(iad, "interval_hessian", _decline)
    build_uniform_relaxation(_exp_model())
    assert declines
    monkeypatch.setattr(iad, "interval_hessian", real)


# ── nonlinear_bound_tightening.py ──────────────────────────────────────────


def test_nbt_struct_cache_setattr_narrowed():
    calls: list[int] = []

    class _Hostile(dm.Model):
        def __setattr__(self, name, value):
            if name == "_nl_struct_cache":
                calls.append(1)
                raise _Boom("setattr")
            super().__setattr__(name, value)

    m = _Hostile("hostile_1520")
    x = m.continuous("x", lb=0.0, ub=1.0)
    m.minimize(x)
    with pytest.raises(_Boom):
        nbt._get_struct_cache(m)
    assert calls

    class _ReadOnly(dm.Model):
        def __setattr__(self, name, value):
            if name == "_nl_struct_cache":
                raise AttributeError("read-only")
            super().__setattr__(name, value)

    r = _ReadOnly("ro_1520")
    y = r.continuous("y", lb=0.0, ub=1.0)
    r.minimize(y)
    cache = nbt._get_struct_cache(r)  # documented degradation: fresh cache
    assert "metadata" in cache and not hasattr(r, "_nl_struct_cache")


def _log_model() -> dm.Model:
    m = dm.Model("log_1520")
    x = m.continuous("x", lb=-5.0, ub=5.0)
    y = m.continuous("y", lb=-5.0, ub=5.0)
    m.minimize(y)
    m.subject_to(dm.log(x) + y <= 3.0)
    m.subject_to(dm.sin(y) <= 0.5)
    return m


def test_function_domain_rule_defect_propagates(monkeypatch):
    m = _log_model()
    md = nbt._cached_flat_metadata(m)
    lb, ub = np.full(2, -5.0), np.full(2, 5.0)
    calls: list[int] = []
    monkeypatch.setattr(nbt, "_match_affine_var", _raiser(calls))
    with pytest.raises(_Boom):
        nbt.FunctionDomainBoundRule().tighten(m, lb, ub, md)
    assert calls


def test_periodic_rule_defect_propagates():
    m = _log_model()
    md = nbt._cached_flat_metadata(m)
    calls: list[int] = []

    class _BadMetadata:
        def scalar_flat_index(self, expr):
            calls.append(1)
            raise _Boom("metadata")

        def __getattr__(self, name):
            return getattr(md, name)

    with pytest.raises(_Boom):
        nbt.PeriodicVariableBoundRule().tighten(
            m, np.full(2, -5.0), np.full(2, 5.0), _BadMetadata()
        )
    assert calls


def _deep_row_model(n: int = 3000) -> dm.Model:
    """One left-deep ``+`` chain of ``n`` terms, as ``.nl`` writes a long row."""
    m = dm.Model("deep_1520")
    xs = [m.continuous(f"x{i}", lb=-5.0, ub=5.0) for i in range(n)]
    body = dm.log(xs[0])
    for v in xs[1:]:
        body = body + v
    m.subject_to(body <= 10.0)
    m.minimize(xs[0])
    return m


def test_function_domain_rule_handles_deep_rows():
    """Defect the removed handler hid: the recursive walk hit RecursionError on a
    3000-term row and the handler returned the box untightened (``x0 >= 0`` from
    ``log(x0)`` was lost for the whole model)."""
    m = _deep_row_model()
    md = nbt._cached_flat_metadata(m)
    n = len(m._variables)
    lb, _ub = nbt.FunctionDomainBoundRule().tighten(m, np.full(n, -5.0), np.full(n, 5.0), md)
    assert lb[0] == pytest.approx(0.0)


def test_periodic_rule_handles_deep_rows():
    m = _deep_row_model()
    md = nbt._cached_flat_metadata(m)
    n = len(m._variables)
    lb, ub = nbt.PeriodicVariableBoundRule().tighten(m, np.full(n, -5.0), np.full(n, 5.0), md)
    np.testing.assert_array_equal(lb, np.full(n, -5.0))  # no sin/cos-only variable
    np.testing.assert_array_equal(ub, np.full(n, 5.0))


def test_nbt_rules_on_non_algebraic_rows():
    """Defect the removed handlers hid: both rules read ``.body`` on an SOS row and
    raised AttributeError. The domain rule now skips the row (and still tightens
    from the algebraic ones); the periodic rule abstains (it cannot prove a
    variable is sin/cos-only when a row it cannot read might use it)."""
    m = _log_model()
    x, y = m._variables
    m.sos1([x, y])
    md = nbt._cached_flat_metadata(m)
    lb, _ = nbt.FunctionDomainBoundRule().tighten(m, np.full(2, -5.0), np.full(2, 5.0), md)
    assert lb[0] == pytest.approx(0.0)
    plb, pub = nbt.PeriodicVariableBoundRule().tighten(m, np.full(2, -5.0), np.full(2, 5.0), md)
    np.testing.assert_array_equal(plb, [-5.0, -5.0])
    np.testing.assert_array_equal(pub, [5.0, 5.0])


# ── factorable_reform.py ───────────────────────────────────────────────────


def test_factorable_headroom_estimate_defect_propagates(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(fr, "_max_expr_node_count", _raiser(calls))
    with pytest.raises(_Boom):
        fr._recursion_headroom_need(_lp())
    assert calls


def test_canonicalize_entropy_defect_propagates(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(fr, "_canonicalize_entropy_expr", _raiser(calls))
    with pytest.raises(_Boom):
        fr.canonicalize_entropy(_lp())
    assert calls


def test_factorable_reformulate_defect_propagates(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(fr, "_has_factorable_work_inner", _raiser(calls))
    with pytest.raises(_Boom):
        fr.factorable_reformulate(_lp())
    assert calls


def test_canonicalize_entropy_handles_deep_rows():
    """Defect the removed handler hid: the recursive rewrite hit RecursionError on a
    3000-term row, and the handler returned the model with no entropy rewrite."""
    m = dm.Model("deep_entropy_1520")
    xs = [m.continuous(f"x{i}", lb=0.1, ub=5.0) for i in range(3000)]
    body = xs[0] * dm.log(xs[0])
    for v in xs[1:]:
        body = body + v
    m.subject_to(body <= 10.0)
    m.minimize(xs[0])
    out = fr.canonicalize_entropy(m)
    assert out is not m  # the x*log(x) product was rewritten


def test_convexity_verdict_unchanged_on_a_plain_model():
    """Decline paths untouched: an ordinary convex model still classifies convex."""
    m = dm.Model("plain_1520")
    x = m.continuous("x", lb=0.0, ub=2.0)
    y = m.continuous("y", lb=0.0, ub=2.0)
    m.minimize(x**2 + y**2)
    m.subject_to(x + y >= 1.0)
    is_convex, mask = rules.classify_model(m, use_certificate=True)
    assert is_convex is True and mask == [True]
    assert rules.classify_expr(dm.sin(x) * y, m) == Curvature.UNKNOWN


def _left_folded_bilinear_sum(n_terms: int):
    m = dm.Model("deep_ih_1520")
    xs = [m.continuous(f"x{i}", lb=0.0, ub=1.0) for i in range(n_terms + 1)]
    e = xs[0] * xs[1]
    for i in range(1, n_terms):
        e = e + xs[i] * xs[i + 1]
    return m, e


def test_interval_hessian_walks_a_deep_left_folded_sum():
    """Root fix for the ``RecursionError`` a call site used to catch: a 400-term
    left-folded sum is far inside the node budget but ~1600 frames deep. It now
    runs on the large-stack runner and returns the exact bilinear Hessian."""
    from discopt._relax.convexity.interval_ad import interval_hessian

    m, e = _left_folded_bilinear_sum(400)
    ad = interval_hessian(e, m)
    lo, hi = np.asarray(ad.hess.lo), np.asarray(ad.hess.hi)
    assert lo.shape == (401, 401)
    assert lo[0, 1] == pytest.approx(1.0) and hi[0, 1] == pytest.approx(1.0)
    assert lo[0, 0] == pytest.approx(0.0) and hi[5, 5] == pytest.approx(0.0)


def test_interval_hessian_shallow_body_runs_inline(monkeypatch):
    """The deep-stack worker engages only when needed: a small body runs inline."""
    import discopt._relax.convexity.rules as rules
    from discopt._relax.convexity.interval_ad import interval_hessian

    calls: list[int] = []
    real = rules._run_with_deep_recursion

    def _spy(fn, *, depth_need):
        calls.append(depth_need)
        return real(fn, depth_need=depth_need)

    monkeypatch.setattr(rules, "_run_with_deep_recursion", _spy)
    m, e = _left_folded_bilinear_sum(3)
    interval_hessian(e, m)
    assert calls and calls[0] <= sys.getrecursionlimit()
