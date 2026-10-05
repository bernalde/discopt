"""Regression tests for #1615: options accepted and silently ignored.

Each option must either take effect or be refused/declared loudly. Covered here:

* D-03 -- an explicit ``tuning=`` never routed off the native spatial kernel,
  because ``_scoped_tuning`` popped it out of ``kwargs`` before the
  ``_native_kernel_feature_safe`` test that looked for it.
* D-01/D-02/D-08/D-13 -- search levers the native kernel does not read
  (``rlt``, ``partitions``, ``strategy``, ``presolve``,
  ``in_tree_presolve_stride``, ``obbt_at_root``) now route to the Python engine.
* B-08 -- an explicit ``mu_init`` is honoured under ``warm_start=``.
* C-10a -- ``gdp_method="hull"`` on indicator constraints (``add_disjunction``
  blocks) is declared with a warning instead of silently lowered by big-M.
* C-10b -- the convex MINLP kernel (big-M only) is not attempted when the caller
  asked for another GDP reformulation.
* B-01b -- ``initial_solution`` / ``warm_start`` on the convex ``pounce`` LP/QP
  arm (a cold matrix solve) warns.
* D-12 -- a looser relative ``gap_tolerance`` is honoured by the native kernel.
* B-02a -- ``skip_convex_check`` is documented as what it does (one local NLP).
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import discopt.solver as solver_mod
import pytest
from discopt import SolverTuning


def _haverly():
    m = dm.Model("haverly")
    fA, fB, fC = (m.continuous(n, lb=0, ub=300) for n in ["fA", "fB", "fC"])
    x, y = m.continuous("x", lb=0, ub=100), m.continuous("y", lb=0, ub=200)
    cX, cY = m.continuous("cX", lb=0, ub=100), m.continuous("cY", lb=0, ub=200)
    p = m.continuous("p", lb=1, ub=3)
    m.subject_to(fA + fB == x + y)
    m.subject_to(p * (x + y) == 3 * fA + fB)
    m.subject_to(fC == cX + cY)
    m.subject_to(x + cX <= 100)
    m.subject_to(y + cY <= 200)
    m.subject_to(p * x + 2 * cX <= 2.5 * (x + cX))
    m.subject_to(p * y + 2 * cY <= 1.5 * (y + cY))
    m.maximize(9 * (x + cX) + 15 * (y + cY) - 6 * fA - 16 * fB - 10 * fC)
    return m


@pytest.fixture
def kernel_calls(monkeypatch):
    """Record native-kernel hand-offs; decline them so the Python engine solves."""
    calls: list[int] = []

    def _fake(*args, **kwargs):
        calls.append(1)
        return None

    monkeypatch.setenv("DISCOPT_NATIVE_SPATIAL_KERNEL", "1")
    monkeypatch.setattr(solver_mod, "_try_native_spatial_kernel", _fake)
    return calls


def test_default_solve_reaches_native_kernel(kernel_calls):
    """Control arm: proves the probe fires, so the routed-off tests mean something."""
    r = _haverly().solve(time_limit=60)
    assert kernel_calls, "default solve never consulted the native kernel"
    assert r.objective == pytest.approx(400.0, abs=1e-3)


def test_explicit_tuning_routes_off_native_kernel(kernel_calls):
    r = _haverly().solve(time_limit=60, tuning=SolverTuning(root_fixpoint=False))
    assert kernel_calls == []
    assert r.objective == pytest.approx(400.0, abs=1e-3)


@pytest.mark.parametrize(
    "kw",
    [
        {"rlt": True},
        {"rlt": False},
        {"partitions": 4},
        {"strategy": "depth_first"},
        {"presolve": False},
        {"in_tree_presolve_stride": 0},
        {"obbt_at_root": False},
    ],
)
def test_kernel_ignored_levers_route_to_python_engine(kernel_calls, kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # partitions=4 warns it is unused on the LP route
        r = _haverly().solve(time_limit=60, **kw)
    assert kernel_calls == [], f"{kw} still handed the solve to the native kernel"
    assert r.objective == pytest.approx(400.0, abs=1e-3)


def test_kernel_lever_defaults_are_not_flagged():
    assert (
        solver_mod._native_kernel_ignored_levers(
            strategy="best_first",
            rlt="auto",
            partitions=0,
            presolve=True,
            in_tree_presolve_stride=1,
            kwargs={},
        )
        == []
    )


def test_mu_init_honoured_under_warm_start():
    m = dm.Model("w")
    p = m.parameter("p", 1.0)
    x = m.continuous("x", shape=(2,), lb=-5, ub=5)
    m.minimize((x[0] - p) ** 2 + (x[1] - 2 * p) ** 2 + dm.exp(0.1 * x[0]))
    m.subject_to(x[0] + x[1] <= 3 * p)
    r0 = m.solve(solver="pounce")
    p.value = 1.1
    base = m.solve(solver="pounce", warm_start=r0)
    r = m.solve(solver="pounce", warm_start=r0, pounce_options={"mu_init": 0.1})
    mu0_base = base.solve_report["iterations"][0]["mu"]
    mu0 = r.solve_report["iterations"][0]["mu"]
    assert mu0_base < 1e-3  # the warm-start-derived value
    assert mu0 == pytest.approx(0.1)
    assert r.objective == pytest.approx(base.objective, abs=1e-6)


def _twobox(api):
    m = dm.Model("twobox")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    A = [x >= 4, x <= 5, y >= 6, y <= 7]
    B = [x >= 7, x <= 8, y >= 2, y <= 3]
    if api == "either_or":
        m.either_or([A, B])
    else:
        dA, dB = m.make_disjunct("A"), m.make_disjunct("B")
        for c in A:
            dA.subject_to(c)
        for c in B:
            dB.subject_to(c)
        m.add_disjunction([dA, dB])
    m.minimize(x + y)
    return m


def test_hull_on_add_disjunction_warns():
    with pytest.warns(UserWarning, match=r"gdp_method='hull' does not apply .* indicator"):
        r = _twobox("make_disjunct").solve(gdp_method="hull")
    assert r.objective == pytest.approx(9.0, abs=1e-6)


def test_hull_on_either_or_is_silent_and_takes_effect():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        r = _twobox("either_or").solve(gdp_method="hull")
    assert not [w for w in rec if "gdp_method" in str(w.message)]
    assert r.root_bound == pytest.approx(9.0, abs=1e-4)


def test_convex_kernel_not_attempted_for_non_bigm_gdp(monkeypatch):
    import discopt.solvers._convex_kernel as ck

    calls: list[str] = []
    real = ck.try_convex_solve

    def _spy(*args, **kwargs):
        calls.append("called")
        return real(*args, **kwargs)

    monkeypatch.setattr(ck, "try_convex_solve", _spy)
    _twobox("either_or").solve()
    assert calls, "control: the default solve never consulted the convex kernel"
    calls.clear()
    _twobox("either_or").solve(gdp_method="hull")
    assert calls == []


def _small_qp():
    m = dm.Model("qp")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    m.minimize((x - 1) ** 2 + (y - 2) ** 2)
    m.subject_to(x + y <= 2)
    return m, x, y


def test_pounce_qp_initial_solution_warns():
    m, x, y = _small_qp()
    with pytest.warns(UserWarning, match=r"ignores initial_solution .*qp-ipm"):
        r = m.solve(solver="pounce", initial_solution={x: 0.5, y: 1.5})
    assert r.algorithm_route == "pounce:qp-ipm"


def test_pounce_qp_without_start_does_not_warn():
    m, _, _ = _small_qp()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        m.solve(solver="pounce")
    assert not [w for w in rec if "initial_solution" in str(w.message)]


def test_skip_convex_check_docstring_states_local_solve():
    doc = dm.Model.solve.__doc__
    i = doc.index("skip_convex_check : bool")
    section = doc[i : i + 900]
    assert "local" in section and "no spatial branch-and-bound" in section


def _kernel_gap_solve(monkeypatch, **kw):
    monkeypatch.setenv("DISCOPT_NATIVE_SPATIAL_KERNEL", "1")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        r = _haverly().solve(time_limit=60, **kw)
    assert r.algorithm_route.startswith("native-spatial"), r.algorithm_route
    return r, [w for w in rec if "gap_tolerance" in str(w.message)]


def test_kernel_loose_relative_gap_is_honoured(monkeypatch):
    """D-12: a gap_tolerance looser than the default stops the kernel on the RELATIVE gap.

    It used to be applied absolutely, so 0.2 explored the same 233 nodes as 1e-4.
    """
    tight, _ = _kernel_gap_solve(monkeypatch)
    loose, warned = _kernel_gap_solve(monkeypatch, gap_tolerance=0.2)
    assert warned == []
    assert loose.node_count < tight.node_count, (loose.node_count, tight.node_count)
    # Haverly maximizes: the bound sits above the incumbent, by at most 20%.
    assert loose.status == "optimal" and loose.gap_certified
    assert loose.bound is not None and loose.objective is not None
    assert loose.bound >= 400.0 - 1e-6  # still a valid bound on the optimum 400
    gap = (loose.bound - loose.objective) / max(abs(loose.bound), abs(loose.objective))
    assert 0.0 <= gap <= 0.2
    assert loose.solver_stats["gap_criterion"] == "relative"


def test_kernel_loose_gap_never_closes_without_a_bound(monkeypatch):
    """The lifted absolute arm must stay finite: ``inf`` closed at node 0 with no bound."""
    r, _ = _kernel_gap_solve(monkeypatch, gap_tolerance=0.5)
    assert r.bound is not None and r.node_count > 0


def test_kernel_default_gap_is_silent(monkeypatch):
    r, warned = _kernel_gap_solve(monkeypatch)
    assert warned == []
    assert r.status == "optimal" and r.solver_stats["gap_criterion"] == "absolute"
