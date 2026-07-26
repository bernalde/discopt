"""Incumbent-quality scoring for primal panels (issue #862).

The #844 differential panel gated on whether an incumbent **exists**, is sound and
stays in budget (``gains / lost_incumbents / cert_regressions / overshoots``). That
was the right bar for closing a "returns nothing" gap, but it never scored how
*good* the incumbent was: the fallback shipped returning 65.3 on tln6 against a
reference optimum of 15.3 (+327%), and a later change could have halved incumbent
quality with the panel still passing.

This module supplies the missing measurement. It is deliberately pure — no solver
imports, no I/O — so it is cheap to unit-test and safe to import from any panel.

Two numbers, for two different jobs:

* :func:`primal_gap` — the MIPLIB/Berthold *primal gap* in ``[0, 1]``, the standard
  literature measure (Berthold 2006, *Primal heuristics for mixed integer
  programs*). Bounded and sign-safe, so it aggregates across a corpus: a mean or a
  worst-case over instances whose objectives span many orders of magnitude is
  meaningful. This is what a panel should gate and trend on.
* :func:`relative_excess` — the signed, unbounded "how much worse than optimal"
  ratio quoted in issue #862 (tln6 ``65.3`` vs ``15.3`` -> ``+3.27``). Unbounded and
  undefined at ``optimum == 0``, so it is a *reporting* number, not an aggregation
  one.

Soundness is scored separately by :func:`is_false_primal`, and the distinction is
load-bearing: an incumbent *below* the reference optimum of a minimize is not a
high-quality incumbent, it is a **correctness failure** (CLAUDE.md §1). Quality
metrics must never absorb one — a panel that averaged a false primal into a "good
gap" would hide the very thing that can never be traded away.

Sense convention: ``sense`` is ``"min"`` or ``"max"`` and refers to the *model's*
sense. ``primal_gap`` is sense-independent by construction; ``relative_excess`` and
``is_false_primal`` are not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


def primal_gap(objective: float | None, optimum: float | None) -> float | None:
    """MIPLIB primal gap of an incumbent against a reference optimum, in ``[0, 1]``.

    ``None`` when there is no incumbent or no reference value — "unscored" is a
    distinct outcome from "scored badly" and callers must be able to tell them
    apart (a panel that silently treated a missing oracle as gap 0 would report a
    corpus as perfect precisely where it measured nothing).

    Definition (Berthold 2006), for incumbent ``p`` and reference ``o``::

        0                             if p == o
        1                             if sign(p) != sign(o), or exactly one is 0
        |p - o| / max(|p|, |o|)       otherwise

    The sign rule keeps the measure bounded and well defined when the objectives
    straddle zero, which is where a plain relative error blows up or changes sign.
    It is sense-independent: it measures distance, and which direction counts as
    "worse" is :func:`is_false_primal`'s job.
    """
    if objective is None or optimum is None:
        return None
    p, o = float(objective), float(optimum)
    if not (math.isfinite(p) and math.isfinite(o)):
        return None
    if p == o:
        return 0.0
    if p * o < 0.0 or (p == 0.0) != (o == 0.0):
        return 1.0
    return abs(p - o) / max(abs(p), abs(o))


def relative_excess(
    objective: float | None, optimum: float | None, sense: str = "min"
) -> float | None:
    """Signed fractional excess of an incumbent over the reference optimum.

    ``+3.27`` means "227% worse than optimal" in the issue's phrasing — this is the
    number the #862 table quotes. Positive is always *worse*, for either sense.

    Returns ``None`` when unscoreable, including at ``optimum == 0`` where the ratio
    is undefined; use :func:`primal_gap` for anything that has to aggregate.
    """
    if objective is None or optimum is None:
        return None
    p, o = float(objective), float(optimum)
    if not (math.isfinite(p) and math.isfinite(o)) or o == 0.0:
        return None
    diff = (p - o) if sense == "min" else (o - p)
    return diff / abs(o)


def is_false_primal(
    objective: float | None,
    optimum: float | None,
    sense: str = "min",
    tol: float = 1e-4,
) -> bool:
    """True when an incumbent beats the reference optimum — a soundness failure.

    For a minimize, a feasible point strictly below the true global optimum cannot
    exist, so reporting one means the incumbent was never feasible (or the oracle is
    wrong). This is a hard gate with zero slack beyond numerical tolerance; it is
    never a quality question.
    """
    if objective is None or optimum is None:
        return False
    p, o = float(objective), float(optimum)
    if not (math.isfinite(p) and math.isfinite(o)):
        return False
    slack = tol * (1.0 + abs(o))
    return (p < o - slack) if sense == "min" else (p > o + slack)


@dataclass(frozen=True)
class QualitySummary:
    """Aggregate incumbent quality over a corpus.

    ``scored`` counts instances that had both an incumbent and a reference optimum;
    ``unscored`` counts everything else and is reported explicitly so a panel can
    never present "no measurement" as "no problem".
    """

    scored: int
    unscored: int
    with_incumbent: int
    false_primals: int
    mean_gap: float | None
    median_gap: float | None
    worst_gap: float | None
    worst_instance: str | None

    def as_dict(self) -> dict:
        return {
            "scored": self.scored,
            "unscored": self.unscored,
            "with_incumbent": self.with_incumbent,
            "false_primals": self.false_primals,
            "mean_gap": self.mean_gap,
            "median_gap": self.median_gap,
            "worst_gap": self.worst_gap,
            "worst_instance": self.worst_instance,
        }


def summarize(rows: Iterable[dict], sense_key: str = "sense") -> QualitySummary:
    """Aggregate ``{"name", "objective", "optimum"[, "sense"]}`` rows.

    Instances with no incumbent are *not* folded in as gap 1.0: "found nothing" and
    "found something poor" are different failures with different fixes, and averaging
    them together would let a change that trades incumbents for quality (or the
    reverse) look neutral. They are counted in ``unscored`` / ``with_incumbent``
    instead.
    """
    gaps: list[float] = []
    worst_gap: float | None = None
    worst_instance: str | None = None
    scored = unscored = with_incumbent = false_primals = 0
    for row in rows:
        obj = row.get("objective")
        opt = row.get("optimum")
        if obj is not None:
            with_incumbent += 1
        if is_false_primal(obj, opt, row.get(sense_key, "min")):
            false_primals += 1
        g = primal_gap(obj, opt)
        if g is None:
            unscored += 1
            continue
        scored += 1
        gaps.append(g)
        if worst_gap is None or g > worst_gap:
            worst_gap, worst_instance = g, row.get("name")
    if not gaps:
        return QualitySummary(
            scored, unscored, with_incumbent, false_primals, None, None, None, None
        )
    ordered = sorted(gaps)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    return QualitySummary(
        scored=scored,
        unscored=unscored,
        with_incumbent=with_incumbent,
        false_primals=false_primals,
        mean_gap=sum(gaps) / len(gaps),
        median_gap=median,
        worst_gap=worst_gap,
        worst_instance=worst_instance,
    )


def quality_regressions(
    baseline: Iterable[dict],
    candidate: Iterable[dict],
    tol: float = 1e-6,
) -> list[dict]:
    """Per-instance incumbent-quality regressions of ``candidate`` against ``baseline``.

    This is the check issue #862 says the #844 panel lacked: with only existence and
    soundness gated, "a change could halve incumbent quality and the panel would
    still pass". Rows are keyed by ``name``; an instance is a regression when both
    runs produced an incumbent, both are scoreable, and the candidate's primal gap is
    strictly worse beyond ``tol``.

    Losing an incumbent outright is *not* reported here — that is the existing
    ``lost_incumbents`` gate's job, and double-counting it would obscure which gate
    actually fired.
    """
    base = {r["name"]: r for r in baseline}
    out: list[dict] = []
    for row in candidate:
        b = base.get(row["name"])
        if b is None:
            continue
        gb = primal_gap(b.get("objective"), b.get("optimum"))
        gc = primal_gap(row.get("objective"), row.get("optimum"))
        if gb is None or gc is None or gc <= gb + tol:
            continue
        out.append(
            {
                "name": row["name"],
                "baseline_objective": b.get("objective"),
                "candidate_objective": row.get("objective"),
                "optimum": row.get("optimum"),
                "baseline_gap": gb,
                "candidate_gap": gc,
                "delta": gc - gb,
            }
        )
    out.sort(key=lambda r: -r["delta"])
    return out
