"""#1595: the in-house simplex must not certify a vertex that is not dual feasible.

Witness family: ``min -x  s.t.  a*x - e*y <= 0,  x in [0, 1],  y >= 0``. Any
``x`` is feasible (take ``y = a*x/e``), so the minimum is ``-1``. With ``a`` around
1e9 or more, equilibration scales every cost below the absolute pricing tolerance
``1e-9``; on main the primal simplex declared ``x = 0`` optimal (objective ``0``)
with the wrong-signed reduced cost of the OPEN column ``y`` hidden under ``tol``.
The Neumaier-Shcherbina safe bound abstained there, and ``_result_basis_cert``
then published the raw objective ``0`` as a certified bound against the true ``-1``.

Two layers are pinned here:

* the simplex itself (cold primal, warm dual) now refuses such a vertex, so every
  witness returns ``-1``;
* independently, an ``optimal`` exit whose safe bound abstains is reported
  unproven (``result is None``) instead of publishing the raw objective.
"""

import discopt.solvers.milp_simplex as ms
import numpy as np
import pytest
import scipy.sparse as sp
from discopt._relax.milp_relaxation import MilpRelaxationModel
from discopt.solvers import SolveStatus
from discopt.solvers.milp_simplex import solve_lp_warm_std

TRUE_MIN = -1.0
TOL = 1e-6

# (a, e): the issue witness, the #1594 witnesses (one rescaled), the under-cap
# coefficient below 1e-9, an a = e case and the smallest a observed to fail.
WITNESSES = [
    (7.5e9, 1.0),
    (1.5e10, 1e-6),
    (7.5e9, 5e-7),
    (7.5e9, 4.5e-10),
    (7.5e9, 7.5e9),
    (1e9, 1.0),
]


def _lp(a, e):
    c = np.array([-1.0, 0.0])
    A = sp.csr_matrix(np.array([[a, -e]]))
    b = np.array([0.0])
    bounds = [(0.0, 1.0), (0.0, np.inf)]
    return c, A, b, bounds


@pytest.mark.parametrize("a,e", WITNESSES)
def test_warm_std_cold_primal_returns_true_minimum(a, e):
    c, A, b, bounds = _lp(a, e)
    result, _basis, cert = solve_lp_warm_std(c, A, b, bounds, return_cert=True)
    assert result is not None
    assert result.status == SolveStatus.OPTIMAL
    assert result.bound <= TRUE_MIN + TOL, (result.objective, result.bound)
    assert result.objective == pytest.approx(TRUE_MIN, abs=TOL)
    assert cert.safe_bound is not None and cert.safe_bound <= TRUE_MIN + TOL


@pytest.mark.parametrize("a,e", WITNESSES)
def test_relaxation_model_simplex_backend_bound_is_valid(a, e):
    c, A, b, bounds = _lp(a, e)
    res = MilpRelaxationModel(c=c, A_ub=A, b_ub=b, bounds=bounds).solve(backend="simplex")
    assert res.status == "optimal"
    assert res.bound is not None and res.bound <= TRUE_MIN + TOL, (res.objective, res.bound)
    assert res.objective == pytest.approx(TRUE_MIN, abs=TOL)


@pytest.mark.parametrize("a,e", WITNESSES)
def test_warm_dual_from_the_falsely_optimal_basis(a, e):
    """Warm-start the dual simplex from the exact basis main certified (``x`` basic,
    ``y`` and the slack at lower). It is primal feasible and dual feasible to the
    absolute ``tol``, so the dual loop stops at once; the post-check must hand it
    to the primal, which finds ``-1``."""
    c, A, b, bounds = _lp(a, e)
    col_status = np.array([1, 0, 0], dtype=np.int8)  # BASIC, AT_LOWER, AT_LOWER
    basic = np.array([0], dtype=np.int64)
    result, _basis, _cert = solve_lp_warm_std(
        c, A, b, bounds, in_basis=(col_status, basic), return_cert=True
    )
    # ``None`` (an unproven exit, caller falls back) is sound; a published bound
    # must not exceed the true minimum.
    if result is not None:
        assert result.status == SolveStatus.OPTIMAL
        assert result.bound <= TRUE_MIN + TOL, (result.objective, result.bound)
    else:
        pytest.fail("warm dual repair should finish the solve, not decline")


def test_optimal_with_abstaining_safe_bound_is_not_published(monkeypatch):
    """Layer 2, independent of the simplex fix: when the NS safe bound abstains on
    an ``optimal`` exit, the raw objective must not become the bound."""
    calls = {"n": 0}

    def _abstain(*_a, **_k):
        calls["n"] += 1
        return None

    monkeypatch.setattr(ms, "_safe_lp_lower_bound", _abstain)
    # The basis-verified fallback would otherwise certify this LP; stub it too so
    # the "both routes decline" contract is what is pinned here.
    monkeypatch.setattr(ms, "_safe_lp_lower_bound_basis", _abstain)
    c = np.array([-1.0, -1.0])
    A = sp.csr_matrix(np.array([[1.0, 1.0]]))
    b = np.array([1.0])
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    result, basis, cert = solve_lp_warm_std(c, A, b, bounds, return_cert=True)
    assert calls["n"] >= 1, "the safe-bound evaluation never ran"
    assert result is None, "an unproven optimum was published as a bound"
    assert basis is None
    assert cert.safe_bound is None and not cert.farkas_certified


# --- #1597 review B1: mixed costs ------------------------------------------------
#
# ``min -cx*x + cz*z  s.t.  a*x - e*y <= 0,  z <= 1,  x, z in [0, 1],  y >= 0``.
# The true minimum is ``-cx`` (``x = 1``, ``z = 0``). The column ``z`` has a cost of
# ordinary size, so a pricing rule judged relative to the largest scaled cost is
# inert here; on main the simplex certified ``x = 0`` (objective 0) with ``y``'s
# wrong-signed reduced cost hidden under ``tol``.
MIXED_WITNESSES = [
    (7.5e9, 1.0, 1.0, 1.0),
    (1e9, 1e-6, 1e-3, 1e3),
    (3e9, 1e-6, 1e-3, 1.0),
    (7.5e9, 1e-6, 1.0, 1e-3),
]


def _mixed_lp(a, e, cx, cz):
    c = np.array([-cx, 0.0, cz])
    A = sp.csr_matrix(np.array([[a, -e, 0.0], [0.0, 0.0, 1.0]]))
    b = np.array([0.0, 1.0])
    bounds = [(0.0, 1.0), (0.0, np.inf), (0.0, 1.0)]
    return c, A, b, bounds


@pytest.mark.parametrize("a,e,cx,cz", MIXED_WITNESSES)
def test_mixed_cost_warm_std_returns_true_minimum(a, e, cx, cz):
    c, A, b, bounds = _mixed_lp(a, e, cx, cz)
    result, _basis, _cert = solve_lp_warm_std(c, A, b, bounds, return_cert=True)
    assert result is not None
    assert result.status == SolveStatus.OPTIMAL
    tol = TOL * (1.0 + cx)
    assert result.bound <= -cx + tol, (result.objective, result.bound)
    assert result.objective == pytest.approx(-cx, abs=tol)


@pytest.mark.parametrize("a,e,cx,cz", MIXED_WITNESSES)
def test_mixed_cost_relaxation_model_bound_is_valid(a, e, cx, cz):
    c, A, b, bounds = _mixed_lp(a, e, cx, cz)
    res = MilpRelaxationModel(c=c, A_ub=A, b_ub=b, bounds=bounds).solve(backend="simplex")
    assert res.status == "optimal"
    tol = TOL * (1.0 + cx)
    assert res.bound is not None and res.bound <= -cx + tol, (res.objective, res.bound)
    assert res.objective == pytest.approx(-cx, abs=tol)


# --- #1597 review N1: the basis-verified bound ----------------------------------
#
# ``min x0  s.t.  3*x0 - x1 = 2,  x0 free,  x1 >= 0`` (standard form, no slack).
# Optimum 2/3 with ``x0`` BASIC: its computed reduced cost is zero only up to
# rounding, and FBBT leaves it the open side ``x0 <= inf``, which is exactly where
# the NS bound abstains. Weak duality at the exact basis dual needs no such sign.
OPT_BF = 2.0 / 3.0


def _basic_free_std():
    a_std = sp.csc_matrix(np.array([[3.0, -1.0]]))
    c = np.array([1.0, 0.0])
    b = np.array([2.0])
    lb = np.array([-ms._INF, 0.0])
    ub = np.array([ms._INF, ms._INF])
    col_status = np.array([1, 0], dtype=np.int8)  # x0 BASIC, x1 AT_LOWER
    return a_std, c, b, lb, ub, col_status


def test_basis_bound_certifies_a_basic_free_column():
    a_std, c, b, lb, ub, cs = _basic_free_std()
    y = np.array([1.0 / 3.0])
    assert ms._safe_lp_lower_bound(y, c, a_std, b, lb, ub) is None, (
        "NS no longer abstains here; this witness no longer exercises the fallback"
    )
    g = ms._safe_lp_lower_bound_basis(y, c, a_std, b, lb, ub, cs)
    assert g is not None
    assert g <= OPT_BF and g >= OPT_BF - 1e-9, g


@pytest.mark.parametrize("y_bad", [0.5, 0.0, -3.0, 1.0 / 3.0 + 1e-7])
def test_basis_bound_is_valid_from_a_wrong_dual(y_bad):
    """The bound encloses the EXACT basis dual; a drifted ``ŷ`` may loosen it or
    make it decline, never lift it above the optimum."""
    a_std, c, b, lb, ub, cs = _basic_free_std()
    g = ms._safe_lp_lower_bound_basis(np.array([y_bad]), c, a_std, b, lb, ub, cs)
    assert g is None or g <= OPT_BF, g


def test_basis_bound_declines_without_a_full_basis():
    a_std, c, b, lb, ub, _cs = _basic_free_std()
    no_basic = np.array([0, 0], dtype=np.int8)
    assert ms._safe_lp_lower_bound_basis(np.array([1.0]), c, a_std, b, lb, ub, no_basic) is None


def test_safe_bound_outcome_counter_records_the_route(monkeypatch):
    """Layer-2 decline counter: every ``optimal`` exit lands in exactly one bucket."""
    monkeypatch.setattr(ms, "SAFE_BOUND_OUTCOMES", __import__("collections").Counter())
    c = np.array([-1.0, -1.0])
    A = sp.csr_matrix(np.array([[1.0, 1.0]]))
    b = np.array([1.0])
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    result, _basis, _cert = solve_lp_warm_std(c, A, b, bounds, return_cert=True)
    assert result is not None
    assert sum(ms.SAFE_BOUND_OUTCOMES.values()) == 1, dict(ms.SAFE_BOUND_OUTCOMES)
