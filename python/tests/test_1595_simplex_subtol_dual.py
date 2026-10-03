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
    c = np.array([-1.0, -1.0])
    A = sp.csr_matrix(np.array([[1.0, 1.0]]))
    b = np.array([1.0])
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    result, basis, cert = solve_lp_warm_std(c, A, b, bounds, return_cert=True)
    assert calls["n"] >= 1, "the safe-bound evaluation never ran"
    assert result is None, "an unproven optimum was published as a bound"
    assert basis is None
    assert cert.safe_bound is None and not cert.farkas_certified
