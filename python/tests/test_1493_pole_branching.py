"""Branch at a pole instead of fathoming the node unresolved (#1493, flagged).

``min 1/x`` s.t. ``x**2 >= 1`` on ``x in [-3, 3]`` (optimum -1 at x = -1) has no
bound on the root box: the reciprocal of a range straddling 0 is ``(-inf, inf)``.
The Rust tree fathomed that untrusted root ``bound_unresolved`` (#467), so the
solve ended ``feasible`` with no bound although one split AT x = 0 gives two
children FBBT reduces to ``x <= -1`` / ``x >= 1``, each with a finite bound.

``DISCOPT_POLE_BRANCHING=1`` (default OFF, CLAUDE.md §5) splits such a node
instead: exactly at the pole when :func:`discopt._relax.poles.objective_pole_loci`
locates one inside the box, else on the pole column's midpoint / the longest edge.
Children inherit the parent's (non-finite) bound, so an unresolved region keeps
the global bound unresolved; a lineage past ``POLE_MAX_DEPTH`` or a solve past
``POLE_MAX_BRANCHES`` takes the legacy fathom, so a pole the objective genuinely
reaches still terminates uncertified. The flag also offers the node-box interval
enclosure of the objective to a node that has no finite floor -- a valid bound,
checked here by the differential/feasible-point tests.
"""

from __future__ import annotations

import logging
import re

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.poles import PoleLocus, objective_pole_loci, pole_branch_point
from discopt.solver import _compute_interval_bound

pytestmark = pytest.mark.unit

FLAG = "DISCOPT_POLE_BRANCHING"
TOL = 1e-6


def _model(form, box=(-3.0, 3.0), exclude=True, sense="min"):
    m = dm.Model("pole")
    x = m.continuous("x", lb=box[0], ub=box[1])
    y = m.continuous("y", lb=1.0, ub=2.0)
    if exclude:
        m.subject_to(x**2 >= 1)
    e = form(x, y) + 0 * y
    (m.minimize if sense == "min" else m.maximize)(e)
    return m


def _solve(m, caplog, time_limit=60.0):
    with caplog.at_level(logging.INFO, logger="discopt.solver"):
        r = m.solve(time_limit=time_limit)
    fired = None
    for rec in caplog.records:
        hit = re.search(
            r"pole branching \(#1493\): (\d+) split\(s\), (\d+) capped", rec.getMessage()
        )
        if hit:
            fired = (int(hit.group(1)), int(hit.group(2)))
    return r, fired


# ---------------------------------------------------------------------------
# end to end: certify with the flag, not without it (the regression pins)
# ---------------------------------------------------------------------------

CERT_CASES = [
    # (id, form, sense, optimum)
    ("recip_min", lambda x, y: 1 / x, "min", -1.0),
    ("recip_max", lambda x, y: 1 / x, "max", 1.0),
    ("ratio_min", lambda x, y: y / x, "min", -2.0),
    ("ratio_max", lambda x, y: y / x, "max", 2.0),
]


@pytest.mark.parametrize("case", CERT_CASES, ids=[c[0] for c in CERT_CASES])
def test_pole_excluded_model_certifies_with_flag(case, monkeypatch, caplog):
    _, form, sense, opt = case
    monkeypatch.setenv(FLAG, "1")
    r, fired = _solve(_model(form, sense=sense), caplog)
    assert fired is not None and fired[0] >= 1, "the pole arm never fired (§6)"
    assert r.status == "optimal" and r.gap_certified, (r.status, r.bound)
    assert abs(r.objective - opt) <= 1e-5 * (1 + abs(opt))
    # certificate invariant against the known optimum, sense-corrected
    if sense == "min":
        assert r.bound <= opt + TOL
    else:
        assert r.bound >= opt - TOL
    assert r.x is not None and float(r.x["x"]) ** 2 >= 1 - 1e-5


@pytest.mark.parametrize("case", CERT_CASES[:2], ids=[c[0] for c in CERT_CASES[:2]])
def test_flag_off_keeps_the_unresolved_exit(case, monkeypatch, caplog):
    _, form, sense, _opt = case
    monkeypatch.setenv(FLAG, "0")
    r, fired = _solve(_model(form, sense=sense), caplog)
    assert fired is None, "flag OFF must not reach the pole path"
    assert r.status != "optimal" and not r.gap_certified and r.bound is None


@pytest.mark.parametrize("sense", ["min", "max"])
def test_reachable_pole_terminates_uncertified(sense, monkeypatch, caplog):
    """``min 1/x`` on [-3, 3] with no exclusion is unbounded: never certify, and
    stop by the cap rather than the time limit."""
    monkeypatch.setenv(FLAG, "1")
    r, fired = _solve(_model(lambda x, y: 1 / x, exclude=False, sense=sense), caplog, 120.0)
    assert fired is not None and fired[0] >= 1 and fired[1] >= 1, fired
    assert r.status != "optimal" and not r.gap_certified
    assert r.bound is None
    assert r.wall_time < 100.0, "must terminate by the pole cap, not the time limit"


def test_flag_is_default_off():
    from discopt.solver_tuning import SolverTuning

    assert SolverTuning().pole_branching is False


# ---------------------------------------------------------------------------
# the pole locator (branch POSITION only)
# ---------------------------------------------------------------------------


def test_objective_pole_loci_forms():
    m = dm.Model("loci")
    x = m.continuous("x", lb=-3, ub=3)
    y = m.continuous("y", lb=1, ub=2)
    z = m.continuous("z", shape=(3,), lb=-1, ub=1)
    cases = [
        (1 / x, [PoleLocus(0, 0.0)]),
        (y / x, [PoleLocus(0, 0.0)]),
        (1 / (x - 1) + y, [PoleLocus(0, 1.0)]),
        (x**-2 + y, [PoleLocus(0, 0.0)]),
        (1 / (2 * x + 3), [PoleLocus(0, -1.5)]),
        (1 / (-x / 4 + 0.5), [PoleLocus(0, 2.0)]),
        (1 / z[2], [PoleLocus(4, 0.0)]),
        (1 / ((x - 1) * (y - 1.5)), [PoleLocus(0, 1.0), PoleLocus(1, 1.5)]),
        (1 / (x + y), []),  # not affine in ONE column: no exact locus
        (x / 2 + y, []),  # constant denominator: no pole
    ]
    n = 0
    for expr, want in cases:
        m.minimize(expr)
        assert objective_pole_loci(m) == want, repr(expr)
        n += 1
    assert n == len(cases)


def test_pole_branch_point():
    loci = [PoleLocus(0, 0.0)]
    lb, ub = np.array([-3.0, 1.0]), np.array([3.0, 2.0])
    assert pole_branch_point(loci, lb, ub) == PoleLocus(0, 0.0)
    # on the boundary: bisect the pole column
    assert pole_branch_point(loci, np.array([0.0, 1.0]), ub) == PoleLocus(0, 1.5)
    # degenerate column at the pole: returned as-is (the tree stops there)
    assert pole_branch_point(loci, np.array([0.0, 1.0]), np.array([0.0, 2.0])) == loci[0]
    # outside the box: no hint
    assert pole_branch_point(loci, np.array([1.0, 1.0]), ub) is None


# ---------------------------------------------------------------------------
# §5 differential / feasible-point checks on the one bound the flag adds
# ---------------------------------------------------------------------------

FORMS = {
    "1/x": (lambda x, y: 1 / x, lambda x, y: 1.0 / x),
    "y/x": (lambda x, y: y / x, lambda x, y: y / x),
    "x+2/x": (lambda x, y: x + 2 / x, lambda x, y: x + 2.0 / x),
    "x**-2+y": (lambda x, y: x**-2 + y, lambda x, y: x**-2.0 + y),
    "x**-3": (lambda x, y: x**-3, lambda x, y: x**-3.0),
    "1/(x-1)+y": (lambda x, y: 1 / (x - 1) + y, lambda x, y: 1.0 / (x - 1.0) + y),
}
# The boxes a pole split produces: pole-free, touching from either side, straddling.
BOXES = [(-3.0, -1.5), (-1.5, 0.0), (0.0, 1.5), (1.5, 3.0), (-3.0, 3.0), (1.0, 4.0), (0.5, 1.0)]


@pytest.mark.parametrize("ext", ["0", "1"])
@pytest.mark.parametrize("sense", ["min", "max"])
def test_interval_fill_is_a_valid_bound(ext, sense, monkeypatch):
    """The enclosure the flag offers an unfloored node never exceeds the box's
    sampled optimum (internal minimization sense); -inf where it cannot enclose."""
    monkeypatch.setenv("DISCOPT_EXTENDED_DIVISION", ext)
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


FAMILY = [
    (name, box, sense)
    for name in ("1/x", "y/x", "x+2/x", "1/(x-1)+y")
    for box in ((-3.0, 3.0), (0.0, 5.0), (-5.0, 0.0))
    for sense in ("min", "max")
]


@pytest.mark.parametrize("ext", ["0", "1"])
def test_family_never_certifies_past_the_grid_optimum(ext, monkeypatch, caplog):
    """Feasible-point sampling end to end: with the flag ON, a certified bound on a
    pole model with ``x**2 >= 1`` never passes the dense-grid optimum of the
    feasible set, and a certified incumbent is feasible and within tolerance."""
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv("DISCOPT_EXTENDED_DIVISION", ext)
    certified = 0
    for name, box, sense in FAMILY:
        build, f = FORMS[name]
        r, _ = _solve(_model(build, box=box, sense=sense), caplog, 30.0)
        if not (r.status == "optimal" and r.gap_certified):
            assert r.status != "optimal" or not r.gap_certified
            continue
        xs = np.linspace(box[0], box[1], 200001)
        xs = xs[xs * xs >= 1.0]
        X, Y = np.meshgrid(xs, np.linspace(1.0, 2.0, 11))
        with np.errstate(divide="ignore", invalid="ignore"):
            F = f(X, Y)
        F = F[np.isfinite(F)]
        opt = F.min() if sense == "min" else F.max()
        if abs(opt) > 1e6:  # unbounded in the optimizing direction: must not certify
            pytest.fail(f"certified an unbounded pole model {name} {box} {sense}")
        tol = 1e-4 * (1 + abs(opt))
        if sense == "min":
            assert r.bound <= opt + tol and r.objective >= opt - tol, (name, box, r.bound, opt)
        else:
            assert r.bound >= opt - tol and r.objective <= opt + tol, (name, box, r.bound, opt)
        assert float(r.x["x"]) ** 2 >= 1 - 1e-5
        certified += 1
    assert certified >= 8, f"only {certified} family cells certified -- the check measured little"
