"""``SolveResult.solve_report``: POUNCE's structured solve report (#1534).

The single-POUNCE-call routes attach the ``pounce.solve-report/v1`` document --
the same data as ``pounce --json-output <file> --json-detail full`` -- so the
iteration history can be read without raising ``print_level`` and parsing the
console table. Branch-and-bound routes report ``None``, as they do for ``kkt``.

The line-for-line tests compare the report against the table POUNCE prints at
``print_level=5`` on the same solve. Every column but ``inf_pr`` must match to the
printed digits; ``inf_pr`` is a different quantity in the report (POUNCE's internal
slack-form residual), which ``SolveResult``'s docstring records, and the inner
restoration rows (``24r``) are not in the report at all. Both are POUNCE-side,
tracked as jkitchin/pounce#979; when it lands, assert ``inf_pr`` and the ``r`` rows too.
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


def _compare_main_rows(printed, iterations) -> int:
    """Assert every main-phase printed row matches its report row; return the count."""
    main = [row for row in printed if row[1] == ""]
    assert len(main) == len(iterations), (len(main), len(iterations))
    compared = 0
    for row, it in zip(main, iterations):
        it_no, _, obj, _inf_pr, inf_du, lg_mu, d_norm, _rg, a_du, a_pr, flag, ls = row
        assert int(it_no) == it["iter"]
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
    entries = [it["iter"] for it in rep["iterations"] if it["alpha_primal_char"] == "R"]
    assert stats["restoration_calls"] >= 1
    assert len(entries) == stats["restoration_calls"]
    printed = _printed_rows(capfd.readouterr().out)
    # The inner restoration rows are printed but are not in the report.
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
