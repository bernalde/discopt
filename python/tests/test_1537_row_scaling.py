"""#1537 witnesses under per-row scaling (row equilibration).

Multiplying a constraint row by a positive scalar changes no feasible point, so it
must change no certified answer. The per-row scaling probe (corpus + generated
families, each row scaled by its own seeded log-uniform factor in 10^[-3,3] and
10^[-6,6]) found two witnesses that this file pins:

* **A crossed single-variable pin** (fixed in ``uniform_relax``).
  ``_fix_single_var_equalities`` collapses the box of a variable fixed by
  ``a*x + c == rhs``. It admits a pin up to 1e-9 *outside* the box and then
  intersects, so a pin one ulp above ``ub`` produced ``lb = val > ub`` -- a crossed
  box, on which every ``Interval`` raises and the whole relaxation build (root
  OBBT, the native spatial kernel spec) failed with ``ValueError: Interval lo >
  hi``. A row-scaled copy computes the pin as ``(s*rhs - s*c)/(s*a)``, one ulp off
  the unscaled value, which is how 4stufen with rows scaled in 10^[-3,3] crashed
  ``Model.solve``.
* **A scale-dependent conditioning guard** (fixed in ``milp_relaxation``).
  ``sanitize_relaxation_for_conditioning`` dropped any relaxation row with an entry
  of magnitude >= 1e10 before the fallback root-bound solve. That is a statement
  about how the row was *written*, not about the constraint: ex14_1_9 with its two
  rows multiplied by 8.14 and 3.66 pushed a 2.02e9 coefficient to 1.64e10, the row
  was dropped, the fallback bound fell to -1.07e6 and the certificate the unscaled
  model earns (bound 0.0) was lost. An over-cap row is now divided by the power of
  two that brings it under the cap (exact, so the same constraint) and dropped
  unless every nonzero is still at least the 1e-9 floor afterwards. (The first
  version of the rescue checked only entries that started above the floor, so a
  row already holding a sub-floor entry was kept where it used to be dropped and
  the in-house simplex returned a false LP bound on it -- PR #1594 review; pinned
  by ``test_sanitizer_rescue_never_keeps_a_sub_floor_entry``.)
"""

from __future__ import annotations

import os

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from _invariance import _rebuild
from discopt._relax.milp_relaxation import (
    _RELAX_NUMERIC_CAP,
    _RESCUE_FLOOR,
    MilpRelaxationModel,
    sanitize_relaxation_for_conditioning,
)
from discopt._relax.uniform_relax import _fix_single_var_equalities, build_uniform_relaxation
from discopt.modeling.core import Constraint


def _pinned_model(scale: float):
    m = dm.Model("pin")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.subject_to(scale * (3.0 * x - 1.0) == 0.0)
    m.subject_to(x * y >= 0.1)
    m.minimize(y + x * x)
    return m


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_pin_one_ulp_outside_the_box_never_crosses_it(direction):
    """A pin within the 1e-9 admission tolerance but outside the box collapses to
    the hull of the two candidates instead of a crossed ``lb > ub`` box."""
    m = _pinned_model(1.0)
    lb = np.array([0.0, 0.0])
    ub = np.array([10.0, 10.0])
    lb0, ub0 = _fix_single_var_equalities(m, lb, ub)
    val = lb0[0]
    assert lb0[0] == ub0[0], "the probe's pin did not fire"
    # Put the box edge one ulp on the far side of the pin.
    if direction > 0:
        ub[0] = np.nextafter(val, -np.inf)
        lb[0] = 0.0
    else:
        lb[0] = np.nextafter(val, np.inf)
        ub[0] = 10.0
    out_lb, out_ub = _fix_single_var_equalities(m, lb, ub)
    assert out_lb[0] <= out_ub[0], (out_lb[0], out_ub[0])
    # Hull of the box edge and the pin: both candidates are kept.
    assert out_lb[0] <= val <= out_ub[0]
    assert out_ub[0] - out_lb[0] <= 2.0 * np.spacing(val)


def test_relaxation_builds_when_the_pin_is_one_ulp_past_the_box():
    """Before the fix this raised ``ValueError: Interval lo > hi``."""
    m = _pinned_model(1.0)
    lb0, _ = _fix_single_var_equalities(m, np.array([0.0, 0.0]), np.array([10.0, 10.0]))
    lb = np.array([0.0, 0.0])
    ub = np.array([np.nextafter(lb0[0], -np.inf), 10.0])
    rel = build_uniform_relaxation(m, box=(lb, ub))
    assert rel.model is not None


@pytest.mark.parametrize("scale", [8.14, 1e-3, 3.7e5])
def test_row_scaled_pin_solves_like_the_unscaled_model(scale):
    """Guard, not a before/after witness: this passes on the pre-fix sources too
    (the one-ulp crossing needs the exact 4stufen arithmetic pinned above). It keeps
    the end-to-end answer row-scale invariant for this small pinned model."""
    base = _pinned_model(1.0).solve(time_limit=20)
    scaled = _pinned_model(scale).solve(time_limit=20)
    assert base.status == "optimal" and scaled.status == "optimal"
    assert scaled.objective == pytest.approx(base.objective, rel=1e-6, abs=1e-8)


# --------------------------------------------------------------------------- #
# The fallback-bound sanitizer: an over-cap row is rescued by an exact power-of-two
# scaling instead of being dropped, so the decision is row-scale invariant.
# --------------------------------------------------------------------------- #

# ex14_1_9's relaxation row 12 as the base model writes it (max 2.02e9, min 3.4e-3,
# a dynamic range of 6e11): kept as written at every scale below.
_EX14_ROW = np.array([4.51007e6, 0.0033557, -2.02051e9, -1.0])
_EX14_RHS = 0.664


def _relax(rows, rhs):
    rows = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    n = rows.shape[1]
    return MilpRelaxationModel(
        c=np.zeros(n),
        A_ub=sp.csr_matrix(rows),
        b_ub=np.asarray(rhs, dtype=np.float64),
        bounds=[(-1.0, 1.0)] * n,
    )


@pytest.mark.parametrize("scale", [1.0, 4.9, 8.14, 1e3, 3.7e6, 1e9])
def test_sanitizer_keeps_a_row_whatever_its_scale(scale):
    """Before the fix a scale that pushed the 2.02e9 entry past 1e10 dropped the row."""
    out = sanitize_relaxation_for_conditioning(_relax(scale * _EX14_ROW, [scale * _EX14_RHS]))
    assert out._A_ub is not None and out._A_ub.shape[0] == 1, "row dropped"
    row = np.asarray(out._A_ub.todense()).ravel()
    assert np.abs(row).max() < _RELAX_NUMERIC_CAP and abs(out._b_ub[0]) < _RELAX_NUMERIC_CAP
    # The kept row is the input row times a power of two: the same constraint.
    ratio = row / (scale * _EX14_ROW)
    assert np.all(ratio == ratio[0]) and out._b_ub[0] / (scale * _EX14_RHS) == ratio[0]
    assert np.log2(ratio[0]) == np.round(np.log2(ratio[0]))


def test_sanitizer_still_drops_the_rows_the_guard_exists_for():
    """A bound-derived 1e20 entry beside a unit aux coefficient spans more than
    cap / floor: no exact rescaling keeps both representable, so it is dropped."""
    out = sanitize_relaxation_for_conditioning(
        _relax([[1e20, -1.0], [1.0, 1.0], [np.inf, 1.0]], [0.0, 1.0, 0.0])
    )
    rows = np.asarray(out._A_ub.todense())
    assert rows.shape[0] == 1 and np.array_equal(rows[0], [1.0, 1.0])
    assert 1e20 / 1.0 > _RELAX_NUMERIC_CAP / _RESCUE_FLOOR


def test_sanitizer_leaves_rows_under_the_cap_untouched():
    rows = np.array([[3.0, -1e9, 1e-31], [1.0, 2.0, 0.0]])
    out = sanitize_relaxation_for_conditioning(_relax(rows, [5.0, 7.0]))
    assert np.array_equal(np.asarray(out._A_ub.todense()), rows)
    assert np.array_equal(out._b_ub, [5.0, 7.0])


def test_row_scaled_ex14_1_9_keeps_its_certificate():
    """The end-to-end witness: ex14_1_9 with its two rows scaled by 8.14 and 3.66
    (the probe's seeded per-row factors) reported ``feasible``, bound -1.07e6, while
    the unscaled model certifies 0.0 -- the fallback root bound lost the scaled row."""
    path = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl", "ex14_1_9.nl")
    base_model = dm.from_nl(path)
    scaled = _rebuild(base_model, lambda v: np.zeros(v.lb.shape), 1.0, "ex14_rows")
    factors = [8.14, 3.66]
    assert len(scaled._constraints) == len(factors)
    scaled._constraints = [
        Constraint(body=s * c.body, sense=c.sense, rhs=0.0, name=c.name)
        for s, c in zip(factors, scaled._constraints)
    ]
    base = base_model.solve(time_limit=30)
    other = scaled.solve(time_limit=30)
    assert base.gap_certified
    assert other.gap_certified, (other.status, other.bound)
    assert other.objective == pytest.approx(base.objective, abs=1e-6)


def test_sanitizer_rescue_never_keeps_a_sub_floor_entry():
    """PR #1594 review witness. ``1.5e10*x - 9e-10*y <= 0``, x in [0, 1], y >= 0,
    min -x: the true LP minimum is -1 (y grows until the row holds). The over-cap
    row cannot be brought under the cap with every nonzero >= 1e-9 (the -9e-10 is
    already below it), so it must be dropped, as before #1537. The first rescue
    checked only the entries that started above the floor and kept the row, on
    which the in-house simplex returned the false bound 0.0."""
    rel = MilpRelaxationModel(
        c=np.array([-1.0, 0.0]),
        A_ub=sp.csr_matrix(np.array([[1.5e10, -9e-10]])),
        b_ub=np.array([0.0]),
        bounds=[(0.0, 1.0), (0.0, np.inf)],
    )
    out = sanitize_relaxation_for_conditioning(rel)
    assert out._A_ub is None or out._A_ub.shape[0] == 0, "sub-floor row kept"
    res = out.solve(backend="simplex")
    assert res.status == "optimal"
    # A valid lower bound for the min: never above the true optimum -1.
    assert res.objective <= -1.0 + 1e-9


def test_sanitizer_rescue_keeps_rows_whose_every_entry_clears_the_floor():
    """The complement: the same shape with the small entry at 1e-6 rescales exactly."""
    out = sanitize_relaxation_for_conditioning(_relax([[1.5e10, -1e-6]], [0.0]))
    row = np.asarray(out._A_ub.todense()).ravel()
    assert row.shape == (2,) and np.all(np.abs(row) >= _RESCUE_FLOOR)
    ratio = row / np.array([1.5e10, -1e-6])
    assert ratio[0] == ratio[1] and np.log2(ratio[0]) == np.round(np.log2(ratio[0]))
