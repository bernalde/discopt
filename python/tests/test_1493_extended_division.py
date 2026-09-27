"""Extended-interval division at a pole, behind ``DISCOPT_EXTENDED_DIVISION`` (#1493).

``min 1/x`` on ``x in [0, 5]`` has optimum 0.2 (at x = 5), but the reciprocal of a
denominator range holding 0 was always enclosed as ``(-inf, inf)``, so the root had
no bound and the solve ended ``feasible`` uncertified. A denominator that only
TOUCHES the pole has a one-sided reciprocal -- ``1/[0, 5] = [0.2, +inf]`` -- and a
straddling one keeps the hull ``(-inf, inf)``. The flag is default OFF (CLAUDE.md §5);
these tests pin both arms:

* OFF is byte-identical to the historical enclosure;
* ON: every enclosure contains every sampled value (feasible-point sampling, the
  endpoint pole included via IEEE ``1/+0 = +inf``); the differential bound test --
  ON root bound >= OFF root bound AND <= the sampled box minimum -- on fixed boxes
  that are pole-free, pole-touching and pole-straddling;
* end to end: ``min 1/x`` on ``[0, 5]`` certifies 0.2 with the flag and does not
  without it, and a pole-touching constraint cuts no feasible point.
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.convexity.interval import Interval
from discopt._relax.convexity.interval_eval import evaluate_interval
from discopt._relax.discretization import DiscretizationState
from discopt._relax.milp_relaxation import build_milp_relaxation
from discopt._relax.term_classifier import classify_nonlinear_terms

pytestmark = pytest.mark.unit

FLAG = "DISCOPT_EXTENDED_DIVISION"

BOXES = [
    (0.0, 5.0),  # touches the pole from above
    (-5.0, 0.0),  # touches from below
    (-3.0, 3.0),  # straddles
    (-0.5, 4.0),  # straddles, asymmetric
    (0.5, 3.0),  # pole-free
    (-4.0, -1.0),  # pole-free, negative
]

# name -> (builder over (m, x, y), numpy evaluator over (x, y))
FORMS = {
    "1/x": (lambda x, y: 1 / x, lambda x, y: 1.0 / x),
    "y/x": (lambda x, y: y / x, lambda x, y: y / x),
    "x**-2": (lambda x, y: x**-2, lambda x, y: x**-2.0),
    "x**-3": (lambda x, y: x**-3, lambda x, y: x**-3.0),
    "x+2/x": (lambda x, y: x + 2 / x, lambda x, y: x + 2.0 / x),
    "(y-1.5)/x": (lambda x, y: (y - 1.5) / x, lambda x, y: (y - 1.5) / x),
}


def _model(form, box):
    m = dm.Model("ext")
    x = m.continuous("x", lb=box[0], ub=box[1])
    y = m.continuous("y", lb=1.0, ub=2.0)
    m.minimize(FORMS[form][0](x, y) + 0 * y)
    return m, x, y


def _samples(box, n=4001):
    xs = np.linspace(box[0], box[1], n)
    xs = xs[np.abs(xs) > 1e-12]
    ys = np.linspace(1.0, 2.0, 9)
    return np.array([(a, b) for a in xs for b in ys])


def _root_bound(m, box):
    milp, _ = build_milp_relaxation(
        m,
        classify_nonlinear_terms(m),
        DiscretizationState(),
        bound_override=(np.array([box[0], 1.0]), np.array([box[1], 2.0])),
    )
    b = milp.solve().bound
    return -math.inf if b is None else float(b)


@pytest.mark.parametrize("form", list(FORMS))
@pytest.mark.parametrize("box", BOXES)
def test_off_arm_is_the_historical_enclosure(form, box, monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    m, x, y = _model(form, box)
    enc = evaluate_interval(m._objective.expression, m)
    lo, hi = box
    if lo <= 0.0 <= hi:
        assert float(enc.lo) == -math.inf and float(enc.hi) == math.inf


@pytest.mark.parametrize("form", list(FORMS))
@pytest.mark.parametrize("box", BOXES)
def test_on_enclosure_contains_every_sampled_value(form, box, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    m, x, y = _model(form, box)
    enc = evaluate_interval(m._objective.expression, m)
    lo, hi = float(enc.lo), float(enc.hi)
    pts = _samples(box)
    vals = FORMS[form][1](pts[:, 0], pts[:, 1])
    assert np.all(vals >= lo) and np.all(vals <= hi), (form, box, lo, hi)
    # The pole endpoint itself: IEEE 1/+0 = +inf, 1/-0 = -inf, both enclosed.
    if box[0] == 0.0 or box[1] == 0.0:
        with np.errstate(divide="ignore", invalid="ignore"):
            end = FORMS[form][1](np.float64(0.0) if box[0] == 0.0 else np.float64(-0.0), 1.0)
        if not np.isnan(end):
            assert lo <= end <= hi
    assert pts.shape[0] > 30000


@pytest.mark.parametrize("form", list(FORMS))
@pytest.mark.parametrize("box", BOXES)
def test_differential_root_bound(form, box, monkeypatch):
    """ON >= OFF, and ON <= the true box minimum (sampled), on fixed boxes."""
    m, x, y = _model(form, box)
    monkeypatch.delenv(FLAG, raising=False)
    off = _root_bound(m, box)
    monkeypatch.setenv(FLAG, "1")
    m2, _, _ = _model(form, box)
    on = _root_bound(m2, box)
    pts = _samples(box, n=2001)
    fmin = float(np.min(FORMS[form][1](pts[:, 0], pts[:, 1])))
    assert on >= off - 1e-9, (form, box, off, on)
    assert on <= fmin + 1e-7 * (1.0 + abs(fmin)), (form, box, on, fmin)


@pytest.mark.parametrize("box", [(0.0, 5.0), (-5.0, 0.0)])
def test_touching_box_tightens_only_there(box, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    x = dm.Model("t").continuous("x", lb=box[0], ub=box[1])
    enc = evaluate_interval(1 / x, None, {x: Interval(np.float64(box[0]), np.float64(box[1]))})
    if box[0] == 0.0:
        assert float(enc.lo) == pytest.approx(0.2) and float(enc.hi) == math.inf
    else:
        assert float(enc.lo) == -math.inf and float(enc.hi) == pytest.approx(-0.2)


def test_min_reciprocal_on_a_pole_touching_box_certifies_only_with_the_flag(monkeypatch):
    def solve():
        m = dm.Model("e")
        x = m.continuous("x", lb=0, ub=5)
        m.minimize(1 / x)
        return m.solve(time_limit=20)

    monkeypatch.delenv(FLAG, raising=False)
    off = solve()
    assert off.bound is None and not off.gap_certified
    monkeypatch.setenv(FLAG, "1")
    on = solve()
    assert on.status == "optimal" and on.gap_certified, (on.status, on.bound)
    assert on.objective == pytest.approx(0.2, abs=1e-6)
    assert on.bound <= 0.2 + 1e-9


@pytest.mark.parametrize("box", [(0.0, 5.0), (-3.0, 3.0)])
def test_pole_constraint_cuts_no_feasible_point(box, monkeypatch):
    """``1/x <= -1`` / ``1/x >= 1`` feasible sets: the ON solve finds the true optimum."""
    monkeypatch.setenv(FLAG, "1")
    m = dm.Model("c")
    x = m.continuous("x", lb=box[0], ub=box[1])
    if box[0] == 0.0:
        m.subject_to(1 / x >= 1)  # x in (0, 1]
        m.maximize(x)
        ref = 1.0
    else:
        m.subject_to(1 / x <= -1)  # x in [-1, 0)
        m.minimize(x)
        ref = -1.0
    r = m.solve(time_limit=20)
    assert r.status == "optimal", (r.status, r.bound)
    assert r.objective == pytest.approx(ref, abs=1e-6)


@pytest.mark.parametrize(
    "form,box,expected",
    [
        ("1/x", (0.0, 5.0), 0.2),
        ("y/x", (0.0, 5.0), 0.2),
        ("x**-2", (-3.0, 3.0), 1.0 / 9.0),  # straddles x, but x**2 only touches 0
        ("x**-2", (-5.0, 0.0), 0.04),
        ("x**-3", (0.0, 5.0), 0.008),
    ],
)
def test_the_flag_actually_tightens_these_cells(form, box, expected, monkeypatch):
    """Proof the differential test above compares something (CLAUDE.md §6)."""
    monkeypatch.delenv(FLAG, raising=False)
    assert _root_bound(_model(form, box)[0], box) == -math.inf
    monkeypatch.setenv(FLAG, "1")
    assert _root_bound(_model(form, box)[0], box) == pytest.approx(expected, rel=1e-9)
