"""#1509: the HiGHS MILP route certified a bound above a feasible point.

``min x**2 + n  s.t.  log(x + 1) + n >= 2,  x in [c, c + 10],  n integer`` passed
through ``dm.nonlinear_to_pwl``: at ``c = 3e3`` the HiGHS route certified the surrogate
at ``9000012.455`` (``1e4``: ``100000012.487``) while ``x = c, n = 0`` is feasible at
``c**2``. ``DISCOPT_LP_MILP_BACKEND=rust`` certified ``c**2``.

Root cause (measured on the extracted standard form): HiGHS's MIP -- not its LP --
cuts the optimum off. The ``log`` block's error-band columns are ``8.67e-8`` wide, below
``mip_feasibility_tolerance = 1e-6``; with presolve off, or with that tolerance at 1e-7,
or with those bands zeroed, HiGHS answers ``c**2``. The error its tolerance-level
reductions introduce reaches the objective through the ``x**2`` rows (coefficients
~``2*c*h``), so it shows only at large offsets. None of the route's existing guards can
see it: no column has an open side (#1295 ratio 0.64, far above ``2**-20``), the root
LP's own primal/dual pair is dual-feasible (#1410: 0 / 1.7e-16 against a round-off bound
of 1.3e-13), and the root cross-check (#1309/#1320) compares the tree bound with the
root bound only in the direction "tree below root" -- here the tree bound is too HIGH.

The fix holds the reported bound against every *verified* point in hand at an LP's
cost -- the root LP point and the NS-safe LP over the incumbent's own integer
assignment -- and a bound above one of them is refuted outright: HiGHS's tree bound is
dropped and the NS-safe root bound reported instead.
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.solver import _highs_std_form
from discopt.solvers import lp_milp_highs
from discopt.solvers.lp_milp_highs import HighsOutcome, StdForm, solve_milp_std

OFFSETS = [3e3, 1e4]
BACKENDS = ["highs", "rust"]


def _original(c: float) -> dm.Model:
    m = dm.Model("i1509")
    x = m.continuous("x", lb=c, ub=c + 10)
    n = m.integer("n")
    m.minimize(x**2 + n)
    m.subject_to(dm.log(x + 1) + n >= 2)
    return m


def _pwl_block(m, x, c, K, f, band, tag):
    """Disaggregated convex-combination PWL of ``f`` on ``[c, c + 10]`` with a
    per-segment error band ``|d_i| <= band * z_i`` -- the ``_lower_outer`` shape,
    written out by hand."""
    b = c + (10.0 / K) * np.arange(K + 1)
    v = np.array([f(t) for t in b])
    cv = v[K // 2]
    wl = m.continuous(f"{tag}_wl", shape=(K,), lb=0.0, ub=1.0)
    wr = m.continuous(f"{tag}_wr", shape=(K,), lb=0.0, ub=1.0)
    z = m.binary(f"{tag}_z", shape=(K,))
    d = m.continuous(f"{tag}_d", shape=(K,), lb=-band, ub=band)
    F = m.continuous(f"{tag}_F", lb=float(v.min()), ub=float(v.max()))  # noqa: N806
    for i in range(K):
        m.subject_to(wl[i] + wr[i] == z[i])
        m.subject_to(d[i] <= band * z[i])
        m.subject_to(d[i] >= -band * z[i])
    m.subject_to(sum(z[i] for i in range(K)) == 1)
    m.subject_to(x == c + sum((b[i] - c) * wl[i] + (b[i + 1] - c) * wr[i] for i in range(K)))
    m.subject_to(
        F == cv + sum((v[i] - cv) * wl[i] + (v[i + 1] - cv) * wr[i] + d[i] for i in range(K))
    )
    return F


def _direct(c: float) -> dm.Model:
    """The same shape as the surrogate, with no ``nonlinear_to_pwl``: ``x**2`` and
    ``log(x + 1)`` as hand-written PWL blocks, the ``log`` band below HiGHS's
    ``mip_feasibility_tolerance``. ``x = c`` (segment 0 at its left breakpoint,
    ``d = 0``, ``n = 0``) is feasible, and ``F_sq >= c**2`` by its bound, so the
    optimum is exactly ``c**2``."""
    m = dm.Model("i1509_direct")
    x = m.continuous("x", lb=c, ub=c + 10)
    n = m.integer("n", lb=0, ub=1e6)
    sq = _pwl_block(m, x, c, 8, lambda t: t * t, 1.5625, "sq")
    lg = _pwl_block(m, x, c, 8, lambda t: math.log(t + 1), 8.67e-8, "lg")
    m.minimize(sq + n)
    m.subject_to(lg + n >= 2)
    return m


def _tol(v: float) -> float:
    return 1e-9 * (1.0 + abs(v))


def _assert_sound(r, c: float) -> int:
    """The certificate invariant, with the optimum's value ``c**2`` as the oracle.

    Returns the number of assertions executed (CLAUDE.md #6)."""
    n = 0
    assert r.status == "optimal", (r.status, r.objective, r.bound)
    assert r.gap_certified
    assert r.bound is not None and r.objective is not None
    n += 1
    # The surrogate is a relaxation of a model whose optimum is c**2 (x = c, n = 0),
    # and the direct MILP's optimum is c**2 exactly; so is every sound bound below it.
    assert r.bound <= c * c + _tol(c * c), f"certified bound {r.bound!r} above {c * c!r}"
    n += 1
    assert r.bound <= r.objective + _tol(r.objective), (r.bound, r.objective)
    n += 1
    return n


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("c", OFFSETS)
def test_surrogate_certificate_is_sound(c, backend, monkeypatch):
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", backend)
    r = dm.nonlinear_to_pwl(_original(c)).model.solve(time_limit=60)
    assert _assert_sound(r, c) == 3
    if backend == "highs":
        # Prove the route that was wrong is the one that ran, and that its check fired.
        assert "milp/refutation_margin" in (r.solver_stats or {}), r.solver_stats


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("c", OFFSETS)
def test_direct_milp_certificate_is_sound(c, backend, monkeypatch):
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", backend)
    r = _direct(c).solve(time_limit=60)
    assert _assert_sound(r, c) == 3
    if backend == "highs":
        assert "milp/refutation_margin" in (r.solver_stats or {}), r.solver_stats


@pytest.mark.parametrize("model_fn", [_direct, lambda c: dm.nonlinear_to_pwl(_original(c)).model])
@pytest.mark.parametrize("c", OFFSETS)
def test_backends_agree(c, model_fn, monkeypatch):
    """HiGHS and rust certify the same optimum, and neither bound passes the other's
    verified incumbent."""
    res = {}
    for backend in BACKENDS:
        monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", backend)
        res[backend] = model_fn(c).solve(time_limit=60)
    h, r = res["highs"], res["rust"]
    assert h.gap_certified and r.gap_certified
    best = min(h.objective, r.objective)
    assert abs(h.objective - r.objective) <= 1e-6 * (1.0 + abs(best)), (h.objective, r.objective)
    assert h.bound <= best + _tol(best), (h.bound, best)
    assert r.bound <= best + _tol(best), (r.bound, best)


@pytest.mark.parametrize("c", OFFSETS)
def test_nonlinear_to_pwl_tripwire_no_longer_fires(c, monkeypatch):
    """The issue's symptom: ``TransformedModel.solve`` raised its tripwire on the
    HiGHS route. Now it certifies the original's optimum ``c**2``."""
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "highs")
    r = dm.nonlinear_to_pwl(_original(c)).solve(time_limit=60)
    assert r.gap_certified
    assert r.objective == pytest.approx(c * c, rel=1e-9)
    assert r.bound <= r.objective + _tol(r.objective)


@pytest.mark.parametrize("c", OFFSETS)
def test_refutation_repairs_the_result_on_the_standard_form(c):
    """At the route's own level: the HiGHS bound is refuted by a verified point and
    replaced by the NS-safe root bound, which here still closes the gap."""
    _, _, sf = _highs_std_form(_direct(c))
    out = solve_milp_std(sf, time_limit=60, gap_tolerance=1e-4, max_nodes=10**6)
    assert out.stats.get("milp/fixed_int_check_ran") == 1.0
    assert out.bound is not None and out.bound <= c * c + _tol(c * c), out.bound
    assert out.objective == pytest.approx(c * c, rel=1e-12)
    if out.stats.get("milp/certificate_refuted"):
        assert out.labels.get("milp/highs_certificate") == "refuted"
        assert out.labels.get("milp/bound_provenance") == "root-ns"
        assert out.bound == min(out.root_bound, out.objective)
    assert out.gap_certified and out.status == "optimal"


def _knapsack_sf() -> StdForm:
    """max x0 + x1 s.t. x0 + x1 <= 10, x0, x1 integer -- HiGHS returns kOptimal."""
    return StdForm.from_arrays(
        c=np.array([-1.0, -1.0]),
        A=np.array([[1.0, 1.0]]),
        b=np.array([10.0]),
        xl=np.array([0.0, 0.0]),
        xu=np.array([1e20, 1e20]),
        int_idx=np.array([0, 1]),
    )


def test_well_behaved_milp_is_not_refuted():
    """Regression fence: an ordinary MILP runs the check, is not refuted, and keeps
    HiGHS's certificate unchanged."""
    out = solve_milp_std(_knapsack_sf(), time_limit=30.0, gap_tolerance=1e-4, max_nodes=1000)
    assert out.stats.get("milp/fixed_int_check_ran") == 1.0
    assert out.stats.get("milp/refutation_margin") is not None
    assert not out.stats.get("milp/certificate_refuted")
    assert out.status == "optimal" and out.gap_certified
    assert out.objective == -10.0 and out.bound == -10.0
    assert out.labels.get("milp/bound_provenance") == "highs-fp"


def test_inconclusive_fixed_integer_lp_decertifies(monkeypatch):
    """The #1320 rule applied to the new check: a check that ran and settled nothing
    is not a passed check."""
    calls = []

    def inconclusive(sf, x, time_limit):
        calls.append(x)
        return HighsOutcome("error", message="kUnknown", highs_status="kUnknown")

    monkeypatch.setattr(lp_milp_highs, "_fixed_integer_lp", inconclusive)
    out = solve_milp_std(_knapsack_sf(), time_limit=30.0, gap_tolerance=1e-4, max_nodes=1000)
    assert calls, "the fixed-integer LP check never ran, so this asserts nothing"
    assert out.stats.get("milp/fixed_int_check_inconclusive") == 1.0
    assert out.status == "feasible"
    assert not out.gap_certified
    assert out.labels.get("milp/certificate") == "declined"


def test_bound_above_a_verified_point_is_refuted(monkeypatch):
    """Direct unit test of the refutation arm, independent of whether a given HiGHS
    build reproduces #1509: HiGHS's MIP answer is replaced by a false one -- the
    suboptimal ``x = (10, 0)`` "certified" at ``-10`` where ``(0, 10)`` reaches
    ``-20`` -- and the route must catch it from a verified point of its own."""
    import highspy

    sf = StdForm.from_arrays(
        c=np.array([-1.0, -2.0]),
        A=np.array([[1.0, 1.0]]),
        b=np.array([10.0]),
        xl=np.array([0.0, 0.0]),
        xu=np.array([1e20, 1e20]),
        int_idx=np.array([0, 1]),
    )
    real_solution = highspy.Highs.getSolution
    real_info = highspy.Highs.getInfo
    faked = []

    def is_mip(h) -> bool:
        return len(h.getLp().integrality_) > 0

    def false_solution(self):
        sol = real_solution(self)
        if is_mip(self):
            sol.col_value = [10.0, 0.0]
            faked.append("x")
        return sol

    def false_info(self):
        info = real_info(self)
        if is_mip(self):
            info.objective_function_value = -10.0
            info.mip_dual_bound = -10.0
            faked.append("bound")
        return info

    monkeypatch.setattr(highspy.Highs, "getSolution", false_solution)
    monkeypatch.setattr(highspy.Highs, "getInfo", false_info)
    out = solve_milp_std(sf, time_limit=30.0, gap_tolerance=1e-4, max_nodes=1000)
    assert "x" in faked and "bound" in faked, "the false HiGHS answer was never injected"
    assert out.stats.get("milp/certificate_refuted") == 1.0
    assert out.labels.get("milp/highs_certificate") == "refuted"
    assert out.objective == -20.0
    assert out.bound is not None and out.bound <= -20.0 + 1e-9
    # The NS-safe root bound (-20) closes the gap to the verified point, so the
    # repaired result is a certificate of the route's own.
    assert out.status == "optimal" and out.gap_certified
