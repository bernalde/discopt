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

# Repo-wide absolute numerical tolerance (conftest.py: abs=1e-6). Used to decide
# when an incumbent has *reached* the reference optimum, which matters most at a
# reference optimum of exactly zero -- see ``primal_gap``.
_ABS_TOL = 1e-6

_SENSES = ("min", "max")


def _check_sense(sense: str) -> str:
    """Validate an objective sense, raising rather than assuming.

    ``sense`` used to default to ``"min"``. That default was unsafe: the ``.nl``
    reader does NOT normalize to MINIMIZE (5 of the vendored instances --
    ``syn05m``, ``syn05hfsg``, ``bchoco06/07/08`` -- load as MAXIMIZE), so a
    maximize row reaching :func:`is_false_primal` with an assumed ``"min"`` would
    report a perfectly sound incumbent as a **correctness violation**, and
    :func:`relative_excess` would report a 10%-worse incumbent as *better than
    optimal*. A silently wrong metric is worse than no metric, because it launders
    regressions as passes. So the sense is required and an unrecognised value is a
    loud error, never a guess.
    """
    if sense not in _SENSES:
        raise ValueError(f"sense must be one of {_SENSES}, got {sense!r}")
    return sense


def primal_gap(
    objective: float | None, optimum: float | None, atol: float = _ABS_TOL
) -> float | None:
    """MIPLIB primal gap of an incumbent against a reference optimum, in ``[0, 1]``.

    ``None`` when there is no incumbent or no reference value — "unscored" is a
    distinct outcome from "scored badly" and callers must be able to tell them
    apart (a panel that silently treated a missing oracle as gap 0 would report a
    corpus as perfect precisely where it measured nothing).

    Definition (Berthold 2006), for incumbent ``p`` and reference ``o``::

        0                             if |p - o| <= atol
        1                             if sign(p) != sign(o), or exactly one is 0
        |p - o| / max(|p|, |o|)       otherwise

    The sign rule keeps the measure bounded and well defined when the objectives
    straddle zero, which is where a plain relative error blows up or changes sign.
    It is sense-independent: it measures distance, and which direction counts as
    "worse" is :func:`is_false_primal`'s job.

    ``atol`` is why the first branch is a tolerance rather than Berthold's exact
    ``p == o``, and it is not cosmetic. Berthold's rule was written for objectives
    that are integral or comfortably away from zero; against a reference optimum of
    *exactly* zero, a floating-point incumbent lands on the "exactly one is 0" branch
    and saturates at 1.0 — the maximum possible gap. The first run of the #862 panel
    scored ``gear`` (optimum 0, incumbent 2.9e-07) and ``st_test1`` (optimum 0,
    incumbent -1.6e-08) as maximally bad and dragged the corpus mean from ~0 to 0.10,
    on two instances discopt had solved to numerical exactness. A metric that reports
    an exact solution as the worst possible outcome is worse than no metric: it would
    train a reader to discount the very signal the panel exists to raise. At a zero
    optimum no relative measure is defined, so an absolute tolerance is the only
    meaningful comparison; it defaults to the repo-wide ``abs=1e-6`` (``conftest.py``).
    """
    if objective is None or optimum is None:
        return None
    p, o = float(objective), float(optimum)
    if not (math.isfinite(p) and math.isfinite(o)):
        return None
    if abs(p - o) <= atol:
        return 0.0
    if p * o < 0.0 or (p == 0.0) != (o == 0.0):
        return 1.0
    return abs(p - o) / max(abs(p), abs(o))


def relative_excess(objective: float | None, optimum: float | None, sense: str) -> float | None:
    """Signed fractional excess of an incumbent over the reference optimum.

    ``+3.27`` means "227% worse than optimal" in the issue's phrasing — this is the
    number the #862 table quotes. Positive is always *worse*, for either sense.

    Returns ``None`` when unscoreable, including at ``optimum == 0`` where the ratio
    is undefined; use :func:`primal_gap` for anything that has to aggregate.
    """
    _check_sense(sense)
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
    sense: str,
    tol: float = 1e-4,
) -> bool:
    """True when an incumbent beats the reference optimum — a soundness failure.

    For a minimize, a feasible point strictly below the true global optimum cannot
    exist, so reporting one means the incumbent was never feasible (or the oracle is
    wrong). This is a hard gate with zero slack beyond numerical tolerance; it is
    never a quality question.
    """
    _check_sense(sense)
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
    """Aggregate ``{"name", "objective", "optimum", "sense"}`` rows.

    Every row must carry ``sense_key``; a missing sense raises rather than being
    assumed to be ``"min"`` (see :func:`_check_sense` for why that default was
    unsafe).

    Two classes of row are counted but deliberately kept out of the gap statistics:

    * **No incumbent / no oracle.** "Found nothing" and "found something poor" are
      different failures with different fixes, and averaging them together would let
      a change that trades incumbents for quality (or the reverse) look neutral.
      They land in ``unscored`` / ``with_incumbent``.
    * **False primals.** An incumbent that beats the reference optimum is not a
      measurement of quality, it is a corruption of one — and because
      :func:`primal_gap` is symmetric, it scores a false primal as *better* than an
      honest incumbent the same distance away (min, opt 10: honest 12 -> 0.167;
      false 9 -> 0.100). Folding that into ``mean_gap`` would let a soundness
      failure read as a quality improvement. It is counted in ``false_primals``,
      which the panel gates on separately, and excluded from the aggregate.
    """
    gaps: list[float] = []
    worst_gap: float | None = None
    worst_instance: str | None = None
    scored = unscored = with_incumbent = false_primals = 0
    for row in rows:
        obj = row.get("objective")
        opt = row.get("optimum")
        if sense_key not in row:
            raise KeyError(
                f"row {row.get('name')!r} has no {sense_key!r}; the objective sense is "
                "required (assuming 'min' can manufacture a false soundness violation)"
            )
        sense = _check_sense(row[sense_key])
        if obj is not None:
            with_incumbent += 1
        if is_false_primal(obj, opt, sense):
            false_primals += 1
            unscored += 1
            continue
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
