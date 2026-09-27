"""A node whose bound is unresolved at a pole: the default-path pins (#1493).

``min 1/x`` s.t. ``x**2 >= 1`` on ``x in [-3, 3]`` (optimum -1 at x = -1) has no
bound on the root box: the reciprocal of a range straddling 0 is ``(-inf, inf)``.
The Rust tree fathoms that untrusted root ``bound_unresolved`` (#467), so the solve
ends ``feasible`` with no bound. ``DISCOPT_POLE_BRANCHING`` (split such a node at
the pole instead) was RETIRED: its MINLPLib panel was cert-clean but gained 0
certificates on the 14 of 37 pole instances where it fired, at nodes 32 -> 4620
and wall 78 -> 372 s (``docs/dev/flag-retirement-audit.md``). These tests pin the
shipped exit -- never a certificate at an unresolved pole -- and keep the
differential check on :func:`discopt.solver._compute_interval_bound`, the node-box
enclosure of the objective, which is not flag-gated.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.solver import _compute_interval_bound

pytestmark = pytest.mark.unit


def _model(form, box=(-3.0, 3.0), exclude=True, sense="min"):
    m = dm.Model("pole")
    x = m.continuous("x", lb=box[0], ub=box[1])
    y = m.continuous("y", lb=1.0, ub=2.0)
    if exclude:
        m.subject_to(x**2 >= 1)
    e = form(x, y) + 0 * y
    (m.minimize if sense == "min" else m.maximize)(e)
    return m


CASES = [
    # (id, form, sense)
    ("recip_min", lambda x, y: 1 / x, "min"),
    ("recip_max", lambda x, y: 1 / x, "max"),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_pole_excluded_by_constraint_keeps_the_unresolved_exit(case):
    _, form, sense = case
    r = _model(form, sense=sense).solve(time_limit=60.0)
    assert r.status != "optimal" and not r.gap_certified and r.bound is None


@pytest.mark.parametrize("sense", ["min", "max"])
def test_reachable_pole_is_never_certified(sense):
    """``min 1/x`` on [-3, 3] with no exclusion is unbounded: never certify."""
    r = _model(lambda x, y: 1 / x, exclude=False, sense=sense).solve(time_limit=60.0)
    assert r.status != "optimal" and not r.gap_certified
    assert r.bound is None


def test_retired_flag_is_gone():
    from discopt.solver_tuning import SolverTuning

    assert not hasattr(SolverTuning(), "pole_branching")


# ---------------------------------------------------------------------------
# differential / feasible-point check on the node-box interval enclosure
# ---------------------------------------------------------------------------

FORMS = {
    "1/x": (lambda x, y: 1 / x, lambda x, y: 1.0 / x),
    "y/x": (lambda x, y: y / x, lambda x, y: y / x),
    "x+2/x": (lambda x, y: x + 2 / x, lambda x, y: x + 2.0 / x),
    "x**-2+y": (lambda x, y: x**-2 + y, lambda x, y: x**-2.0 + y),
    "x**-3": (lambda x, y: x**-3, lambda x, y: x**-3.0),
    "1/(x-1)+y": (lambda x, y: 1 / (x - 1) + y, lambda x, y: 1.0 / (x - 1.0) + y),
}
# Pole-free, touching from either side, straddling.
BOXES = [(-3.0, -1.5), (-1.5, 0.0), (0.0, 1.5), (1.5, 3.0), (-3.0, 3.0), (1.0, 4.0), (0.5, 1.0)]


@pytest.mark.parametrize("sense", ["min", "max"])
def test_interval_bound_is_valid(sense):
    """The node-box enclosure never exceeds the box's sampled optimum (internal
    minimization sense); -inf where it cannot enclose."""
    checked = finite = 0
    for name, (build, f) in FORMS.items():
        for box in BOXES:
            m = _model(build, box=box, exclude=False, sense=sense)
            negate = sense == "max"
            lo = np.array([box[0], 1.0])
            hi = np.array([box[1], 2.0])
            b = _compute_interval_bound(m, lo, hi, negate)
            xs = np.linspace(box[0], box[1], 4001)
            X, Y = np.meshgrid(xs, np.linspace(1.0, 2.0, 21))
            with np.errstate(divide="ignore", invalid="ignore"):
                F = f(X, Y)
            F = F[np.isfinite(F)]
            internal = -F if negate else F
            assert not np.isnan(b)
            assert b <= internal.min() + 1e-9 * (1 + abs(internal.min())), (name, box, sense, b)
            checked += 1
            finite += bool(np.isfinite(b))
    assert checked == len(FORMS) * len(BOXES)
    assert finite >= 10, "the differential test must compare finite bounds (§6)"
