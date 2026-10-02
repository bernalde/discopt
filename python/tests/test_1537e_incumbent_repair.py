"""#1537 workstream E: the published incumbent is repaired to float noise.

``verify_point``'s row allowance ``ABS_TOL * max(1, |rhs|, max_j |J_ij x_j|)`` is
not invariant under a change of variables that leaves the model identical (the
floor ``1`` does not scale with a row; ``|x_j|`` grows with the origin's
distance), so a solver's working-accuracy residual passed it in one set of
coordinates and failed it in the identical model in another -- and the
objective reported at that point was bought with the residual. The corpus
witnesses (all three were strict xfails in ``test_1537_invariance.py``):

* ``ex14_1_9`` with rows x1e-3: certified ``x1 = -9.98e-6`` against a true
  minimum of ~0, on a row residual of 9.98e-9 (9.98e-6 in the original units);
* ``ex1225`` shifted ~1e6: certified 30.9999959 against 31 on an equality
  residual of 2.1e-6 that the shifted row's term scale allowed;
* ``syn05hfsg`` shifted ~1e6: a published incumbent violating row 38 by 6.7e-6.

The repair (``feasibility.repair_point``) is a minimum-norm Newton projection,
itself invariant under both transforms; these tests pin that invariance and the
end-to-end behaviour on the witnesses.
"""

from __future__ import annotations

import os

import discopt.modeling as dm
import numpy as np
import pytest
from _invariance import rescale_rows, translate
from discopt._tape_nlp_evaluator import make_evaluator
from discopt.validation.feasibility import repair_point, verify_point

DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl")


def _flat(result, model):
    return np.concatenate(
        [np.ravel(np.asarray(result.x[v.name], dtype=np.float64)) for v in model._variables]
    )


def _minimax(scale: float = 1.0):
    """``min y`` s.t. ``|x^2 - 4| <= y`` written as two rows, each times ``scale``."""
    m = dm.Model("minimax")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=-10.0, ub=10.0)
    k = m.integer("k", lb=0, ub=3)
    m.subject_to(scale * (x**2 - 4 - y) <= 0)
    m.subject_to(scale * (4 - x**2 - y) <= 0)
    m.subject_to(scale * (x + k - 4.5) <= 0)
    m.minimize(y + 0.0 * k)
    return m


# A point the solver's tolerance can exploit: y slid 1e-5 below its true minimum 0.
_BAD = np.array([2.0, -1e-5, 1.0])


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e6])
def test_repair_drives_rows_to_noise_whatever_the_row_scale(scale):
    m = _minimax(scale)
    rep = repair_point(m, _BAD, evaluator=make_evaluator(m))
    assert rep.x is not None, rep.reason
    assert rep.excess_before > 0.0 and rep.excess_after == 0.0
    # Judged in the UNSCALED units, whatever units the repair ran in.
    check = verify_point(_minimax(1.0), rep.x, with_objective=True)
    assert check.ok, check.reason
    assert abs(check.objective) <= 1e-12
    assert rep.x[2] == 1.0  # the integer column never moves


def test_repair_step_is_invariant_under_row_scaling():
    """The minimum-norm step solves ``J dx = -r``; a row rescale scales both sides."""
    models = [_minimax(s) for s in (1e-3, 1.0, 1e6)]
    pts = [repair_point(m, _BAD, evaluator=make_evaluator(m)).x for m in models]
    assert all(p is not None for p in pts)
    np.testing.assert_allclose(pts[0], pts[1], rtol=0, atol=1e-12)
    np.testing.assert_allclose(pts[2], pts[1], rtol=0, atol=1e-12)


def test_repair_step_is_invariant_under_translation():
    m = _minimax(1.0)
    t = translate(m, 1e6, seed=4)
    base = repair_point(m, _BAD, evaluator=make_evaluator(m)).x
    moved = repair_point(t, _BAD + t._invariance_shift, evaluator=make_evaluator(t)).x
    assert base is not None and moved is not None
    # The shifted arithmetic carries ~1e6-sized terms: agreement to their ulps.
    np.testing.assert_allclose(moved - t._invariance_shift, base, rtol=0, atol=1e-8)
    assert verify_point(m, moved - t._invariance_shift).ok


def test_repair_declines_a_point_already_feasible_to_noise():
    m = _minimax(1.0)
    rep = repair_point(m, np.array([2.0, 0.0, 1.0]), evaluator=make_evaluator(m))
    assert rep.x is None and rep.excess_before == 0.0, rep


def test_repair_respects_the_box():
    """A column the step would push out of its box is clipped and frozen."""
    m = dm.Model("box")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=5.0)
    m.subject_to(x + y == 3.0)
    m.minimize(y)
    rep = repair_point(m, np.array([1.0, 2.0 - 1e-5]), evaluator=make_evaluator(m))
    assert rep.x is not None, rep.reason
    assert 0.0 <= rep.x[0] <= 1.0 and 0.0 <= rep.x[1] <= 5.0
    assert abs(rep.x[0] + rep.x[1] - 3.0) <= 1e-15


# ── end to end, on the corpus witnesses ───────────────────────────────────────


@pytest.mark.correctness
def test_ex14_1_9_rows_scaled_publishes_no_bought_certificate():
    """Main: ``optimal``/certified at ``-9.98e-6`` (true minimum ~0) on a point the
    original model rejects. Now the published point verifies in the ORIGINAL
    units and its objective is the honest one; the certificate that rested on the
    residual is withdrawn (the bound, a valid -9.98e-6, is untouched)."""
    orig = dm.from_nl(os.path.join(DATA, "ex14_1_9.nl"))
    scaled = rescale_rows(orig, 1e-3)
    r = scaled.solve(time_limit=20)
    assert r.x is not None
    check = verify_point(orig, _flat(r, scaled), with_objective=True)
    assert check.ok, check.reason
    assert check.objective >= -1e-6
    assert not (r.gap_certified and r.objective < -1e-6)
    assert r.solver_stats["certificate/incumbent_repaired"] == 1.0


@pytest.mark.correctness
def test_ex1225_shifted_keeps_its_certificate_on_a_repaired_point():
    """Main: certified 30.9999959 vs the true 31 on an equality residual of 2.1e-6.
    The repair lands on 31 (to float noise of the ~1e6 terms) and the certificate
    survives, because the bound was always within the relative tolerance of 31."""
    orig = dm.from_nl(os.path.join(DATA, "ex1225.nl"))
    shifted = translate(orig, 1e6, seed=2)
    r = shifted.solve(time_limit=20)
    assert r.gap_certified and r.status == "optimal"
    assert abs(r.objective - 31.0) <= 1e-6
    x = _flat(r, shifted) - shifted._invariance_shift
    check = verify_point(orig, x, with_objective=True)
    assert check.ok, check.reason


@pytest.mark.correctness
def test_flag_off_publishes_the_unrepaired_point(monkeypatch):
    """``DISCOPT_INCUMBENT_REPAIR=0`` is the documented opt-out: the solver's own
    point, as before #1537 E. The solve itself is identical either way -- the repair
    is post-solve, so the bound and node count do not move."""
    orig = dm.from_nl(os.path.join(DATA, "ex14_1_9.nl"))
    monkeypatch.setenv("DISCOPT_INCUMBENT_REPAIR", "0")
    off = rescale_rows(orig, 1e-3).solve(time_limit=20)
    monkeypatch.setenv("DISCOPT_INCUMBENT_REPAIR", "1")
    on = rescale_rows(orig, 1e-3).solve(time_limit=20)
    assert "certificate/incumbent_repaired" not in (off.solver_stats or {})
    assert on.solver_stats["certificate/incumbent_repaired"] == 1.0
    assert on.bound == off.bound and on.node_count == off.node_count
