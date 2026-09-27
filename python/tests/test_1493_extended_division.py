"""Interval division at a pole: the default-path pins (#1493).

``min 1/x`` on ``x in [0, 5]`` has optimum 0.2, but the reciprocal of a denominator
range holding 0 is enclosed as ``(-inf, inf)``, so the root has no bound and the
solve ends ``feasible`` uncertified. ``DISCOPT_EXTENDED_DIVISION`` (a one-sided
reciprocal for a denominator that only TOUCHES 0) was RETIRED: its MINLPLib panel
was cert-clean but gained 0 certificates and 0 bounds on the 17 of 37 pole
instances where it fired (``docs/dev/flag-retirement-audit.md``). These tests pin
the shipped enclosure:

* a denominator range holding 0 gives the hull ``(-inf, inf)``;
* every enclosure contains every sampled value (feasible-point sampling);
* the root relaxation bound on a pole box is ``-inf`` and never above the sampled
  box minimum elsewhere;
* end to end: ``min 1/x`` on ``[0, 5]`` stays uncertified with no bound.
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
def test_pole_box_enclosure_is_the_full_line(form, box):
    m, x, y = _model(form, box)
    enc = evaluate_interval(m._objective.expression, m)
    lo, hi = box
    if lo <= 0.0 <= hi:
        assert float(enc.lo) == -math.inf and float(enc.hi) == math.inf


@pytest.mark.parametrize("form", list(FORMS))
@pytest.mark.parametrize("box", BOXES)
def test_enclosure_contains_every_sampled_value(form, box):
    m, x, y = _model(form, box)
    enc = evaluate_interval(m._objective.expression, m)
    lo, hi = float(enc.lo), float(enc.hi)
    pts = _samples(box)
    vals = FORMS[form][1](pts[:, 0], pts[:, 1])
    assert np.all(vals >= lo) and np.all(vals <= hi), (form, box, lo, hi)
    assert pts.shape[0] > 30000


@pytest.mark.parametrize("form", list(FORMS))
@pytest.mark.parametrize("box", BOXES)
def test_root_bound_is_valid(form, box):
    """The root bound never exceeds the true box minimum (sampled); -inf at a pole."""
    m, x, y = _model(form, box)
    b = _root_bound(m, box)
    pts = _samples(box, n=2001)
    fmin = float(np.min(FORMS[form][1](pts[:, 0], pts[:, 1])))
    assert b <= fmin + 1e-7 * (1.0 + abs(fmin)), (form, box, b, fmin)
    if box[0] <= 0.0 <= box[1]:
        assert b == -math.inf, (form, box, b)


@pytest.mark.parametrize("box", [(0.0, 5.0), (-5.0, 0.0)])
def test_touching_box_is_unbounded(box):
    x = dm.Model("t").continuous("x", lb=box[0], ub=box[1])
    enc = evaluate_interval(1 / x, None, {x: Interval(np.float64(box[0]), np.float64(box[1]))})
    assert float(enc.lo) == -math.inf and float(enc.hi) == math.inf


def test_min_reciprocal_on_a_pole_touching_box_is_uncertified():
    m = dm.Model("e")
    x = m.continuous("x", lb=0, ub=5)
    m.minimize(1 / x)
    r = m.solve(time_limit=20)
    assert r.bound is None and not r.gap_certified


@pytest.mark.parametrize(
    "form,box",
    [("1/x", (0.0, 5.0)), ("y/x", (0.0, 5.0)), ("x**-2", (-5.0, 0.0)), ("x**-3", (0.0, 5.0))],
)
def test_pole_touching_root_bound_is_minus_inf(form, box):
    assert _root_bound(_model(form, box)[0], box) == -math.inf


@pytest.mark.parametrize("box", [(0.0, 5.0), (-3.0, 3.0)])
def test_pole_constraint_cuts_no_feasible_point(box):
    """``1/x >= 1`` / ``1/x <= -1`` feasible sets: the solve finds the true optimum."""
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
