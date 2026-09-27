"""An objective pole inside the box is named, not blamed on a missing envelope (#1493).

``min 1/x`` and ``min y/x`` over ``x in [-5, 5]`` returned ``feasible`` at 0.2 with
``bound=None`` and the generic advice "a nonlinear term with no envelope is the usual
cause; an epigraph reformulation often gives the relaxation something to bound". Both
halves of that advice are wrong at a pole: the ratio envelope exists and correctly
refuses to bound a denominator whose range holds 0, and no reformulation can bound
an objective that diverges there.

The honest outcome, decided here:

* the STATUS stays what the solve proved -- a feasible point and no bound
  (``discopt.status`` state 3). ``unbounded`` means *certified* unboundedness, which
  no relaxation proves at a pole, and constraints may exclude the pole (``x**2 >= 1``
  makes ``min 1/x`` bounded, optimum -1), so the solver may not claim it;
* the DIAGNOSTIC names the pole, the denominator and its range, says the result
  carries no optimality claim, and says what to do (bound the denominator away from
  zero or split by sign).
"""

from __future__ import annotations

import logging

import discopt.modeling as dm
import pytest
from discopt._relax.poles import objective_poles

pytestmark = pytest.mark.unit

GENERIC = "A nonlinear term with no envelope is the usual cause"


def _one_var(lb=-5.0, ub=5.0):
    m = dm.Model("pole")
    x = m.continuous("x", lb=lb, ub=ub)
    return m, x


def test_detects_reciprocal_ratio_and_negative_power():
    m, x = _one_var()
    y = m.continuous("y", lb=1, ub=2)
    m.minimize(1 / x + y / x + x**-2 + dm.sin(1 / (x - 1)))
    dens = sorted(p.denominator for p in objective_poles(m, limit=10))
    assert dens == ["(x - 1)", "x", "x", "x"]


@pytest.mark.parametrize(
    "build",
    [
        lambda x: 1 / (x**2 + 1),  # denominator range [1, 26]
        lambda x: x / 2.0,  # constant denominator
        lambda x: x**2,  # positive power
    ],
    ids=["shifted", "constant", "positive-power"],
)
def test_no_pole_reported_when_the_denominator_avoids_zero(build):
    m, x = _one_var()
    m.minimize(build(x))
    assert objective_poles(m) == []


def test_no_pole_when_the_box_excludes_it():
    m, x = _one_var(lb=1.0, ub=5.0)
    m.minimize(1 / x)
    assert objective_poles(m) == []


def _solve_and_capture(m, caplog):
    with caplog.at_level(logging.WARNING, logger="discopt.solver"):
        r = m.solve(time_limit=15)
    msgs = [rec.getMessage() for rec in caplog.records if rec.name == "discopt.solver"]
    return r, msgs


@pytest.mark.parametrize("kind", ["recip", "ratio", "recip_max", "excluded_by_constraint"])
def test_pole_solve_is_uncertified_and_names_the_pole(kind, caplog, monkeypatch):
    # The default (pole branching OFF) exit. With DISCOPT_POLE_BRANCHING=1 the
    # constraint-excluded pole certifies -1 instead (test_1493_pole_branching.py).
    monkeypatch.setenv("DISCOPT_POLE_BRANCHING", "0")
    m, x = _one_var()
    if kind == "recip":
        m.minimize(1 / x)
    elif kind == "ratio":
        y = m.continuous("y", lb=1, ub=2)
        m.minimize(y / x)
    elif kind == "recip_max":
        m.maximize(1 / x)
    else:
        m.subject_to(x**2 >= 1)
        m.minimize(1 / x)
    r, msgs = _solve_and_capture(m, caplog)
    # Never a certificate: not optimal, no bound, and not a claimed unboundedness.
    assert r.status not in ("optimal", "unbounded", "infeasible"), r.status
    assert r.bound is None and not r.gap_certified
    pole_msgs = [s for s in msgs if "has a pole inside the variable box" in s]
    assert len(pole_msgs) == 1, msgs
    assert "divides by `x`" in pole_msgs[0] and "[-5, 5]" in pole_msgs[0]
    assert not any(GENERIC in s for s in msgs), "generic envelope advice at a pole"
    if kind == "excluded_by_constraint" and r.x is not None:
        # The only thing returned is a feasible point; here it is the optimum.
        assert float(r.x["x"]) ** 2 >= 1 - 1e-6


def test_no_pole_keeps_the_generic_diagnostic_path(caplog):
    """A pole-free model is untouched by the new branch."""
    m, x = _one_var(lb=1.0, ub=5.0)
    m.minimize(1 / x)
    r, msgs = _solve_and_capture(m, caplog)
    assert r.status == "optimal"
    assert not any("has a pole" in s for s in msgs)
