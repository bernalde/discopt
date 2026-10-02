"""``SolveResult.solve_report``: POUNCE's structured solve report (#1534).

The single-POUNCE-call routes attach the ``pounce.solve-report/v1`` document --
the same data as ``pounce --json-output <file> --json-detail full`` -- so the
iteration history can be read without raising ``print_level`` and parsing the
console table. Branch-and-bound routes report ``None``, as they do for ``kkt``.

The line-for-line tests compare the report against the table POUNCE prints at
``print_level=5`` on the same solve. With jkitchin/pounce#979 (POUNCE ``main``, which
CI installs) every printed row, ``r`` rows included, must match on every column,
``inf_pr`` included. On pounce-solver 0.12.0, which lacks it, the inner restoration
rows are absent and ``inf_pr`` is the internal slack-form residual, so only the
main-phase rows are compared and ``inf_pr`` is skipped (``_compare_main_rows``).
"""

from __future__ import annotations

import logging
import math
import re

import discopt.modeling as dm
import numpy as np
import pytest

pytestmark = [pytest.mark.smoke, pytest.mark.requires_pounce]

_SCHEMA = "pounce.solve-report/v1"

# One printed iteration row: iter[r] objective inf_pr inf_du lg(mu) ||d|| lg(rg)
# alpha_du alpha_pr[flag] ls
_ROW = re.compile(
    r"^\s*(\d+)(r?)\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+"
    r"(\d\.\d\de[+-]\d\d)([A-Za-z]?)\s+(\d+)\s*$"
)


def _printed_rows(text: str) -> list[tuple[str, ...]]:
    return [m.groups() for line in text.splitlines() if (m := _ROW.match(line))]


def _main_rows(iterations) -> list[dict]:
    """The main-phase report rows (rows without ``phase`` predate pounce#979: all main)."""
    return [it for it in iterations if it.get("phase", "main") == "main"]


def _compare_main_rows(printed, iterations) -> int:
    """Assert every printed row matches its report row; return the count compared.

    A report that carries ``phase`` (POUNCE with jkitchin/pounce#979) holds the inner
    restoration rows too and its ``inf_pr`` is the printed column, so every printed
    row is compared, ``inf_pr`` included. An older report is compared on its
    main-phase rows only, without ``inf_pr``.
    """
    full = any("phase" in it for it in iterations)
    if not full:
        printed = [row for row in printed if row[1] == ""]
    assert len(printed) == len(iterations), (len(printed), len(iterations))
    compared = 0
    for row, it in zip(printed, iterations):
        it_no, r_flag, obj, inf_pr, inf_du, lg_mu, d_norm, _rg, a_du, a_pr, flag, ls = row
        assert int(it_no) == it["iter"]
        if full:
            assert it["phase"] == ("restoration" if r_flag else "main")
            assert f"{it['inf_pr']:.2e}" == inf_pr
        assert f"{it['objective']:.7e}" == obj
        assert f"{it['inf_du']:.2e}" == inf_du
        assert f"{math.log10(it['mu']):.1f}" == lg_mu
        assert f"{it['d_norm']:.2e}" == d_norm
        assert f"{it['alpha_dual']:.2e}" == a_du
        assert f"{it['alpha_primal']:.2e}" == a_pr
        assert (flag or " ") == it["alpha_primal_char"]
        assert int(ls) == it["ls_trials"]
        compared += 1
    return compared


def _convex_nlp():
    m = dm.Model("convex_nlp")
    x = m.continuous("x", shape=(2,), lb=-3.0, ub=3.0)
    m.minimize(dm.exp(x[0]) + x[0] ** 2 + (x[1] - 1.0) ** 2)
    m.subject_to(x[0] + x[1] >= 1.0)
    return m


def _restoration_nlp():
    """A local single-NLP solve that enters restoration once from this start."""
    m = dm.Model("restoration_nlp")
    x = m.continuous("x", shape=(2,), lb=-10.0, ub=10.0)
    m.minimize(x[0] + x[1])
    m.subject_to(x[0] ** 2 + x[1] ** 2 == 1.0)
    m.subject_to(x[0] * x[1] == 0.4)
    return m, x


def test_convex_nlp_route_attaches_report():
    res = _convex_nlp().solve()
    assert res.status == "optimal" and res.convex_fast_path
    rep = res.solve_report
    assert rep is not None and rep["schema"] == _SCHEMA
    stats = rep["statistics"]
    its = rep["iterations"]
    # No restoration on this solve, so one row per iteration plus the start row.
    assert stats["restoration_calls"] == 0
    assert len(its) == stats["iteration_count"] + 1
    assert [it["iter"] for it in its] == list(range(len(its)))
    # The barrier parameter only ever decreases on a monotone-mu solve.
    mus = [it["mu"] for it in its]
    assert all(b <= a for a, b in zip(mus, mus[1:]))
    # The same solve's terminal mu, reported through both channels.
    assert its[-1]["mu"] == pytest.approx(res.kkt["barrier_parameter"])
    assert rep["solution"]["objective"] == pytest.approx(res.objective)


def test_trajectory_matches_printed_table_line_for_line(capfd):
    res = _convex_nlp().solve(ipopt_options={"print_level": 5})
    printed = _printed_rows(capfd.readouterr().out)
    assert printed, "the printed iteration table was not captured"
    assert _compare_main_rows(printed, res.solve_report["iterations"]) > 0


def test_restoration_entries_and_counts(capfd):
    m, x = _restoration_nlp()
    res = m.solve(
        time_limit=20,
        skip_convex_check=True,
        initial_solution={x: np.array([-3.0, 7.0])},
        ipopt_options={"print_level": 5},
    )
    rep = res.solve_report
    assert rep is not None
    stats = rep["statistics"]
    main = _main_rows(rep["iterations"])
    entries = [it["iter"] for it in main if it["alpha_primal_char"] == "R"]
    assert stats["restoration_calls"] >= 1
    assert len(entries) == stats["restoration_calls"]
    restoration = [it for it in rep["iterations"] if it.get("phase") == "restoration"]
    if restoration:  # POUNCE with jkitchin/pounce#979
        assert len(restoration) == stats["restoration_inner_iters"]
    printed = _printed_rows(capfd.readouterr().out)
    assert any(row[1] == "r" for row in printed)
    assert _compare_main_rows(printed, rep["iterations"]) > 0


def test_qp_route_attaches_report():
    m = dm.Model("qp")
    x = m.continuous("x", shape=(2,), lb=-3.0, ub=3.0)
    m.minimize(x[0] ** 2 + x[1] ** 2 + x[0])
    m.subject_to(x[0] + x[1] >= 1.0)
    res = m.solve()
    assert res.status == "optimal"
    rep = res.solve_report
    assert rep is not None and rep["schema"] == _SCHEMA
    assert len(rep["iterations"]) >= 1


def test_branch_and_bound_route_has_no_report():
    m = dm.Model("minlp")
    x = m.continuous("x", lb=0.0, ub=3.0)
    y = m.binary("y")
    m.minimize(dm.exp(x) - 2.0 * x + y)
    m.subject_to(x <= 2.0 * y + 0.5)
    res = m.solve()
    assert res.status == "optimal"
    assert res.solve_report is None


def test_lp_on_simplex_has_no_report():
    m = dm.Model("lp")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=3.0)
    m.minimize(x[0] + 2.0 * x[1])
    m.subject_to(x[0] + x[1] >= 1.0)
    res = m.solve()
    assert res.status == "optimal"
    assert res.solve_report is None


def test_lp_pounce_backend_report_is_opt_in():
    from discopt.solvers.lp_pounce import solve_lp

    kw = dict(c=np.array([1.0, 2.0]), A_ub=np.array([[-1.0, -1.0]]), b_ub=np.array([-1.0]))
    assert solve_lp(**kw).solve_report is None
    rep = solve_lp(**kw, solve_report=True).solve_report
    assert rep is not None and rep["schema"] == _SCHEMA


def test_lp_pounce_route_attaches_report():
    """The model-level POUNCE LP route, reached directly (the simplex leads)."""
    import time

    from discopt.solver import _solve_lp_pounce

    m = dm.Model("lp")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=3.0)
    m.minimize(x[0] + 2.0 * x[1])
    m.subject_to(x[0] + x[1] >= 1.0)
    res = _solve_lp_pounce(m, time.perf_counter(), 10.0)
    assert res.status == "optimal"
    assert res.solve_report is not None and res.solve_report["schema"] == _SCHEMA


def test_attach_requires_exactly_one_call():
    from discopt.modeling.core import SolveResult
    from discopt.solver import _attach_solve_report

    one = SolveResult(status="optimal")
    _attach_solve_report(one, [{"schema": _SCHEMA}])
    assert one.solve_report == {"schema": _SCHEMA}
    two = SolveResult(status="optimal")
    _attach_solve_report(two, [{"a": 1}, {"b": 2}])
    assert two.solve_report is None


def test_missing_report_warns_and_is_none(monkeypatch, caplog):
    """A report POUNCE did not write must not fail the solve, and must not be silent."""
    import pounce

    real_solve = pounce.Problem.solve

    def solve_without_report(self, *args, report_path=None, report_detail=None, **kwargs):
        return real_solve(self, *args, **kwargs)

    monkeypatch.setattr(pounce.Problem, "solve", solve_without_report)
    with caplog.at_level(logging.WARNING, logger="discopt.solvers._pounce_report"):
        res = _convex_nlp().solve()
    assert res.status == "optimal"
    assert res.solve_report is None
    assert "[pounce-solve-report-missing]" in caplog.text


# --- solver="pounce" (#1533): every arm attaches the report ------------------------
#
# The LP and QP arms run POUNCE's convex IPM through ``pounce.qp.solve_qp``, which
# writes no report; discopt builds the document from its ``QpResult``
# (``_pounce_report.convex_report``). Reopened #1534: those two arms returned
# ``solve_report=None``.

# One row of ``convex_ipm_pounce._print_trace``: iter objective inf_pr inf_du mu a_pr a_du
_CONVEX_ROW = re.compile(r"^\s*(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$")


def _pounce_lp():
    m = dm.Model("lp")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=10.0)
    m.minimize(-x[0] - 2.0 * x[1])
    m.subject_to(x[0] + x[1] <= 4.0)
    return m


def _pounce_qp():
    m = dm.Model("qp")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=1.0)
    m.minimize(x[0] ** 2 + x[1] ** 2 - 3.0 * x[0] - 4.0 * x[1])
    m.subject_to(x[0] + x[1] <= 1.0)
    return m


def _check_convex_report(res, route):
    assert res.algorithm_route == route
    rep = res.solve_report
    assert rep is not None and rep["schema"] == _SCHEMA
    assert rep["solution"]["engine"] == "cvx-qp"
    assert rep["fair_metadata"]["generated_by"].endswith("convex_report")
    stats = rep["statistics"]
    assert stats["iteration_count"] == res.solver_stats["pounce/iterations"]
    assert stats["restoration_calls"] == 0
    its = rep["iterations"]
    assert [it["iter"] for it in its] == list(range(len(its)))
    for key in ("objective", "inf_pr", "inf_du", "mu", "alpha_primal", "alpha_dual"):
        assert all(math.isfinite(it[key]) for it in its), key
    return rep


@pytest.mark.parametrize(
    ("build", "route", "optimum"),
    [(_pounce_lp, "pounce:lp-ipm", -8.0), (_pounce_qp, "pounce:qp-ipm", -3.125)],
)
def test_pounce_solver_convex_arms_attach_report(build, route, optimum):
    res = build().solve(solver="pounce")
    assert res.status == "optimal"
    assert res.objective == pytest.approx(optimum, abs=1e-6)
    rep = _check_convex_report(res, route)
    assert rep["statistics"]["iteration_count"] >= 1
    assert rep["solution"]["status_upstream"] == "Solve_Succeeded"
    assert rep["solution"]["x"] == pytest.approx(list(res.x["x"]), abs=1e-6)


def test_pounce_solver_nlp_arm_attaches_report():
    res = _convex_nlp().solve(solver="pounce")
    assert res.algorithm_route == "pounce:nlp"
    assert res.solve_report is not None and res.solve_report["schema"] == _SCHEMA
    assert len(res.solve_report["iterations"]) >= 1


@pytest.mark.parametrize("build", [_pounce_lp, _pounce_qp])
def test_pounce_solver_convex_trajectory_matches_printed_table(build, capsys):
    res = build().solve(solver="pounce", pounce_options={"print_level": 1})
    out = capsys.readouterr().out
    printed = [m.groups() for line in out.splitlines() if (m := _CONVEX_ROW.match(line))]
    its = res.solve_report["iterations"]
    assert len(printed) == len(its) >= 1
    for row, it in zip(printed, its):
        it_no, obj, inf_pr, inf_du, mu, a_pr, a_du = row
        assert int(it_no) == it["iter"]
        assert f"{it['objective']:.7e}" == obj
        assert f"{it['inf_pr']:.2e}" == inf_pr
        assert f"{it['inf_du']:.2e}" == inf_du
        assert f"{it['mu']:.2e}" == mu
        assert f"{it['alpha_primal']:.2e}" == a_pr
        assert f"{it['alpha_dual']:.2e}" == a_du


def test_pounce_solver_convex_report_on_a_limit():
    """The report is attached on a non-optimal outcome too, where it is most needed."""
    res = _pounce_qp().solve(solver="pounce", pounce_options={"max_iter": 1})
    assert res.status == "iteration_limit"
    rep = _check_convex_report(res, "pounce:qp-ipm")
    assert rep["solution"]["status_upstream"] == "Maximum_Iterations_Exceeded"


def test_convex_backend_report_is_opt_in():
    from discopt.solvers import convex_ipm_pounce as cvx

    kw = dict(c=np.array([-1.0, -2.0]), A_ub=np.array([[1.0, 1.0]]), b_ub=np.array([4.0]))
    assert cvx.solve_lp(**kw).solve_report is None
    rep = cvx.solve_lp(**kw, solve_report=True).solve_report
    assert rep is not None and rep["schema"] == _SCHEMA
    assert rep["problem"]["n_constraints"] == 1
