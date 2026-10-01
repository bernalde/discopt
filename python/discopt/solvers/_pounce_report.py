"""POUNCE's structured solve report, captured in memory (#1534).

POUNCE writes its ``pounce.solve-report/v1`` document -- the same JSON as
``pounce --json-output <path> --json-detail full`` -- only to a path handed to
``Problem.solve(report_path=..., report_detail=...)``. There is no in-memory
variant in the pinned release (0.12.0), so this module routes the report through a
private temporary directory, parses it, and removes the file before returning.

What the report holds, measured on pounce-solver 0.12.0 (not assumed):

* ``statistics`` -- ``iteration_count``, the ``final_*`` residuals, evaluation
  counts, ``total_wallclock_time_secs``, ``restoration_calls`` /
  ``restoration_inner_iters`` / ``restoration_outer_iters`` /
  ``restoration_wall_secs``.
* ``iterations`` -- one row per **main-phase** iteration, in the same order and
  with the same ``iter`` numbers as POUNCE's printed table: ``objective``
  (unscaled, as printed), ``inf_pr``, ``inf_du``, ``mu``, ``d_norm``,
  ``regularization``, ``alpha_dual``, ``alpha_primal``, ``alpha_primal_char``
  (the line-search flag printed after ``alpha_pr``; ``"R"`` marks the iteration
  that entered restoration) and ``ls_trials``.
* ``solution``, ``problem`` and ``fair_metadata`` (solver version, timestamps).

Two places the report does **not** reproduce the printed table, both POUNCE-side:

1. The inner restoration-phase rows (printed with an ``r`` suffix, e.g. ``24r``)
   are not in ``iterations``; the restoration phase appears only as the ``"R"``
   row that entered it and the ``restoration_*`` counts in ``statistics``.
2. ``inf_pr`` is POUNCE's internal primal infeasibility -- the residual of the
   slack-reformulated, scaled problem that its filter and convergence test use --
   while the printed ``inf_pr`` column is the violation of the original
   constraints. They agree whenever the slacks are at the constraint values; at an
   iterate where a slack sits off the constraint value (the starting point after
   the bound push, the iterations after a restoration exit) they differ visibly --
   0.94 printed against 0.955 reported at iteration 0 of a one-row problem. Every
   other column matches the printed table to the digits it prints.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any, Optional

_logger = logging.getLogger(__name__)

#: The schema this module was written and tested against.
SOLVE_REPORT_SCHEMA = "pounce.solve-report/v1"

#: ``"full"`` carries the per-iteration trajectory; ``"summary"`` drops it.
_REPORT_DETAIL = "full"


def solve_with_report(problem: Any, *args: Any, **kwargs: Any) -> tuple[Any, dict, Optional[dict]]:
    """Call ``problem.solve(*args, **kwargs)`` and also return its solve report.

    Returns ``(x, info, report)``. ``report`` is the parsed
    ``pounce.solve-report/v1`` document, or ``None`` when POUNCE wrote none.

    A missing or unreadable report is logged at WARNING with the marker
    ``[pounce-solve-report-missing]`` and returned as ``None``: the report is a
    diagnostic, nothing in the solver reads it back, so it must never be able to
    turn a solve that succeeded into one that failed. Exceptions raised by the
    solve itself propagate unchanged.
    """
    with tempfile.TemporaryDirectory(prefix="discopt-pounce-report-") as tmp:
        path = os.path.join(tmp, "report.json")
        x, info = problem.solve(*args, report_path=path, report_detail=_REPORT_DETAIL, **kwargs)
        report = _read_report(path)
    return x, info, report


def _read_report(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            report = json.load(fh)
    except (OSError, ValueError) as exc:
        _logger.warning(
            "POUNCE solve report could not be read from %s: %s: %s. "
            "SolveResult.solve_report is None for this solve [pounce-solve-report-missing].",
            path,
            type(exc).__name__,
            exc,
        )
        return None
    if not isinstance(report, dict):
        _logger.warning(
            "POUNCE solve report is a %s, not a JSON object; "
            "SolveResult.solve_report is None for this solve [pounce-solve-report-missing].",
            type(report).__name__,
        )
        return None
    if report.get("schema") != SOLVE_REPORT_SCHEMA:
        # Attached anyway: the document is what POUNCE produced and carries its own
        # ``schema`` key, so a consumer can branch on it. Warn so a schema bump is
        # noticed rather than discovered as a KeyError downstream.
        _logger.warning(
            "POUNCE solve report has schema %r; discopt was written against %r. "
            "Attached as-is [pounce-solve-report-schema].",
            report.get("schema"),
            SOLVE_REPORT_SCHEMA,
        )
    return report
