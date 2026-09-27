"""Regression tests for #1506: AMP's sin/cos inflection injection.

``_apply_partition_refinement`` listed every inflection point of a ``sin``/``cos``
atom over the variable's whole box with a Python loop:

* an unbounded argument crashed on ``math.ceil(-inf)`` (``OverflowError``), which
  AMP's MILP step used to swallow into ``status="error"`` (MINLPLib ``mathopt3``);
* a wide finite argument (``[-1e9, 1e9]``, ~6.4e8 inflections) hung the relaxation
  build, where no deadline is checked, far past ``time_limit``.

``_tan_branch_safe`` had the same O(width) loop and the same crash.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import Model
from discopt._relax import uniform_relax as ur
from discopt.modeling.core import from_nl

pytestmark = pytest.mark.smoke

_DATA = Path(__file__).parent / "data" / "minlplib"

#: Oracle for the issue model: the default B&B route certifies it.
_ISSUE_OPT = {"sin": -0.3429166732, "cos": -0.1808983342}


def _issue_model(fname: str, B: float) -> Model:
    m = Model("amp_trig_1506")
    x = m.continuous("x", lb=-B, ub=B)
    y = m.continuous("y", lb=-B, ub=B)
    m.subject_to(x + y >= 1)
    m.minimize(getattr(dm, fname)(x) + 0.1 * x * x + 0.1 * y * y)
    return m


def _old_tan_branch_safe(alo: float, ahi: float) -> bool:
    """The pre-#1506 loop, kept as the reference for finite narrow intervals."""
    margin = 1e-6
    k_min = math.floor((alo - 0.5 * math.pi) / math.pi) - 1
    k_max = math.ceil((ahi - 0.5 * math.pi) / math.pi) + 1
    for k in range(k_min, k_max + 1):
        asymptote = 0.5 * math.pi + k * math.pi
        if alo - margin <= asymptote <= ahi + margin:
            return False
    return True


def _old_inflections(fname: str, alo: float, ahi: float) -> list[float]:
    start = 0.0 if fname == "sin" else 0.5 * math.pi
    out = []
    for k in range(math.ceil((alo - start) / math.pi), math.floor((ahi - start) / math.pi) + 1):
        p = start + k * math.pi
        if alo < p < ahi:
            out.append(float(p))
    return out


@pytest.mark.parametrize("fname", ["sin", "cos"])
def test_unbounded_argument_does_not_crash(fname):
    r = _issue_model(fname, math.inf).solve(solver="amp", time_limit=20)
    assert r.status in ("optimal", "feasible"), r.status
    assert r.objective == pytest.approx(_ISSUE_OPT[fname], abs=1e-6)
    if r.bound is not None:
        assert r.bound <= _ISSUE_OPT[fname] + 1e-6


@pytest.mark.parametrize("fname", ["sin", "cos"])
def test_wide_box_respects_time_limit(fname):
    """A 2e9-wide box no longer runs an O(width) loop inside the relaxation build."""
    t0 = time.perf_counter()
    r = _issue_model(fname, 1e9).solve(solver="amp", time_limit=5)
    wall = time.perf_counter() - t0
    # Before the fix this never returned (still inside the enumeration at 60 s).
    assert wall < 60.0, wall
    assert r.status in ("optimal", "feasible", "time_limit"), r.status
    if r.objective is not None:
        assert r.objective >= _ISSUE_OPT[fname] - 1e-6
    if r.bound is not None:
        assert r.bound <= _ISSUE_OPT[fname] + 1e-6


def test_mathopt3_amp_certifies():
    """MINLPLib mathopt3 (all variables free, =opt= 0) returned status=error in ~2 s."""
    m = from_nl(str(_DATA / "mathopt3.nl"))
    r = m.solve(solver="amp", time_limit=60)
    assert r.status == "optimal", (r.status, r.objective, r.bound)
    assert r.objective == pytest.approx(0.0, abs=1e-6)
    assert r.bound is not None and r.bound <= 1e-6


@pytest.mark.parametrize("fname", ["sin", "cos"])
def test_inflection_enumeration_is_bounded(fname):
    assert ur._univariate_inflection_args(fname, -math.inf, 1.0) is None
    assert ur._univariate_inflection_args(fname, 0.0, math.inf) is None
    t0 = time.perf_counter()
    assert ur._univariate_inflection_args(fname, -1e300, 1e300) is None
    assert time.perf_counter() - t0 < 1.0


@pytest.mark.parametrize("fname", ["sin", "cos"])
@pytest.mark.parametrize("lo,hi", [(-10.0, 10.0), (-3.0, 0.5), (0.1, 0.2), (-100.0, 100.0)])
def test_inflections_unchanged_under_the_cap(fname, lo, hi):
    """Within the cap the injected set is exactly the old enumeration (bound-neutral)."""
    old = _old_inflections(fname, lo, hi)
    assert len(old) <= ur._MAX_INFLECTION_POINTS
    assert ur._univariate_inflection_args(fname, lo, hi) == old
    n = 0
    for coeff, const in ((1.0, 0.0), (2.0, 0.3), (-0.5, 1.0)):
        pts = sorted([lo, 0.5 * (lo + hi), hi])
        a0, a1 = coeff * lo + const, coeff * hi + const
        expected = _old_inflections(fname, min(a0, a1), max(a0, a1))
        if len(expected) > ur._MAX_INFLECTION_POINTS:
            continue  # over the cap: the wide-box regime, tested separately
        assert ur._partition_inflection_args(fname, coeff, const, pts) == expected
        n += 1
    assert n >= 2


def test_wide_partition_injects_only_narrow_intervals():
    """Over a wide box the narrow partition intervals still get their inflections."""
    pts = [-1e9, -4.0, 4.0, 1e9]
    got = ur._partition_inflection_args("sin", 1.0, 0.0, pts)
    assert got == pytest.approx([-math.pi, 0.0, math.pi])
    # Budgeted: many moderately wide intervals never exceed the cap in total.
    many = list(np.linspace(-1e6, 1e6, 2001))
    assert len(ur._partition_inflection_args("cos", 1.0, 0.0, many)) <= ur._MAX_INFLECTION_POINTS


def test_tan_branch_safe_bounded_and_equivalent():
    assert ur._tan_branch_safe(-math.inf, 0.0) is False
    assert ur._tan_branch_safe(0.0, math.inf) is False
    t0 = time.perf_counter()
    assert ur._tan_branch_safe(-1e12, 1e12) is False
    assert time.perf_counter() - t0 < 1.0
    rng = np.random.default_rng(1506)
    n = 0
    for _ in range(4000):
        lo = float(rng.uniform(-20.0, 20.0))
        hi = lo + float(rng.choice([rng.uniform(0.0, 0.2), rng.uniform(0.0, 4.0)]))
        assert ur._tan_branch_safe(lo, hi) == _old_tan_branch_safe(lo, hi), (lo, hi)
        n += 1
    # Boundary-adjacent intervals, where the float rounding matters most.
    for k in range(-5, 6):
        a = 0.5 * math.pi + k * math.pi
        cases = ((a - 0.3, a - 2e-6), (a - 0.3, a - 5e-7), (a + 5e-7, a + 0.3), (a - 1, a + 1))
        for lo, hi in cases:
            assert ur._tan_branch_safe(lo, hi) == _old_tan_branch_safe(lo, hi), (lo, hi)
            n += 1
    assert n == 4000 + 11 * 4
