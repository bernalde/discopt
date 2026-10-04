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
* **A scale-dependent conditioning guard** (``milp_relaxation``; NOT fixed --
  xfail, see #1595). ``sanitize_relaxation_for_conditioning`` drops any relaxation
  row with an entry of magnitude >= 1e10 before the fallback root-bound solve. That
  is a statement about how the row was *written*: ex14_1_9 with its two rows
  multiplied by 8.14 and 3.66 pushes a 2.02e9 coefficient to 1.64e10, the row is
  dropped, the fallback bound falls to -1.07e6 and the certificate the unscaled
  model earns (bound 0.0) is lost. An exact power-of-two "rescue" of over-cap rows
  was tried and withdrawn in the PR #1594 review: it lands the row's largest entry
  in [5e9, 1e10), where the in-house simplex returns false LP bounds (#1595) on
  rows that main drops. A lost certificate is acceptable; a false bound is not.

  Since #1595 (PR #1597) the end-to-end witness certifies again, but *not* through
  the sanitizer: the in-house simplex now solves the scaled root LP soundly, so the
  fallback that drops the row is never reached (measured: zero sanitizer calls,
  bound -1.47e-7 against the ``minlplib.solu`` optimum 0). The sanitizer still drops
  over-cap rows when it does run; that is pinned by the unit tests below.
"""

from __future__ import annotations

import os
import zlib

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from _invariance import _rebuild
from discopt._relax.milp_relaxation import (
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
# The fallback-bound sanitizer: over-cap rows are dropped (sound), even though that
# makes the fallback bound depend on row scaling. See the module docstring.
# --------------------------------------------------------------------------- #


def _lp(rows, c, bounds):
    return MilpRelaxationModel(
        c=np.asarray(c, dtype=np.float64),
        A_ub=sp.csr_matrix(np.atleast_2d(np.asarray(rows, dtype=np.float64))),
        b_ub=np.zeros(np.atleast_2d(rows).shape[0]),
        bounds=bounds,
    )


@pytest.mark.parametrize("small", [-9e-10, -1e-6, -1.0])
def test_sanitizer_drops_over_cap_rows_the_simplex_mishandles(small):
    """PR #1594 review witnesses. ``1.5e10*x + small*y <= 0``, x in [0, 1], y >= 0,
    min -x: the true LP minimum is -1. The withdrawn rescue rescaled the row to
    ``[7.5e9, small/2]`` and kept it; on that row the in-house simplex returns the
    false bound 0.0 (#1595). Main and this branch drop it, and the bound is -1."""
    rel = _lp([[1.5e10, small]], [-1.0, 0.0], [(0.0, 1.0), (0.0, np.inf)])
    out = sanitize_relaxation_for_conditioning(rel)
    assert out._A_ub is None or out._A_ub.shape[0] == 0, "over-cap row kept"
    res = out.solve(backend="simplex")
    assert res.status == "optimal"
    assert res.objective <= -1.0 + 1e-9  # a valid lower bound for the min


def test_sanitizer_leaves_rows_under_the_cap_untouched():
    rows = np.array([[3.0, -1e9, 1e-31], [1.0, 2.0, 0.0]])
    rel = MilpRelaxationModel(
        c=np.zeros(3),
        A_ub=sp.csr_matrix(rows),
        b_ub=np.array([5.0, 7.0]),
        bounds=[(-1.0, 1.0)] * 3,
    )
    out = sanitize_relaxation_for_conditioning(rel)
    assert np.array_equal(np.asarray(out._A_ub.todense()), rows)
    assert np.array_equal(out._b_ub, [5.0, 7.0])


def test_row_scaled_ex14_1_9_keeps_its_certificate():
    """The end-to-end witness: ex14_1_9 with its two rows scaled by 8.14 and 3.66
    (the probe's seeded per-row factors). Before #1595 it reported ``feasible`` with
    bound -1.07e6 (the root LP failed, and the fallback root bound lost the scaled
    row); the unscaled model certifies 0.0. It must certify, with a bound that does
    not cross the known optimum 0 (``minlplib.solu``)."""
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
    assert other.bound <= 0.0 + 1e-6, other.bound


# --------------------------------------------------------------------------- #
# Rust MILP equilibration under per-row scaling (#1537; row pre-pass retired).
# --------------------------------------------------------------------------- #


def _per_row_scaled(name: str, span: float):
    """The probe's model: every row of ``name`` multiplied by its own seeded
    log-uniform factor in ``10^[-span, span]`` (seed = crc32 of the file name)."""
    path = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl", name)
    base = dm.from_nl(path)
    new = _rebuild(base, lambda v: np.zeros(v.lb.shape), 1.0, f"{name}_pr{span:g}")
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    new._constraints = [
        Constraint(
            body=float(10.0 ** rng.uniform(-span, span)) * c.body,
            sense=c.sense,
            rhs=0.0,
            name=c.name,
        )
        for c in new._constraints
    ]
    return new


def test_row_scaled_m3_loses_its_certificate_soundly(monkeypatch):
    """m3 with rows scaled in 10^[-6,6]. An OA-master column for a variable that
    appears in a row scaled by ~1e5 and one scaled by ~1e-6 holds a 1e-11 ratio,
    which the column-first equilibration reads as noise, so the #1296 guard
    withdraws the master certificates and the solve ends uncertified.

    A pow2 row pre-pass (``DISCOPT_LP_ROW_PRESCALE``) restored this certificate but
    was retired after its §5 panel: clay0303hfsg (rows scaled in 10^[-3,3]) lost its
    certificate at 20 s, with 15x the Numerical LP verdicts of the legacy factors
    (see ``docs/dev/flag-retirement-audit.md``). This pins what main keeps: the
    certificate is lost, never replaced by a bound above the optimum. A future
    row-scale-invariant fix flips the ``gap_certified`` assertion and must pass the
    panel to do so.

    Pinned to the in-house master (``DISCOPT_CONVEX_ROUTE_OA_MASTER=auto``), which
    is where the #1296 guard lives and which remains a shipped route. The default
    HiGHS master is that row-scale-tolerant fix for this instance (performance-plan
    §25.13): see ``test_row_scaled_m3_certifies_on_the_highs_master``."""
    from discopt._rust import profile_counters_py, profile_reset_py

    monkeypatch.setenv("DISCOPT_CONVEX_ROUTE_OA_MASTER", "auto")
    monkeypatch.setenv("DISCOPT_PROFILE", "1")
    profile_reset_py()
    res = _per_row_scaled("m3.nl", 6.0).solve(time_limit=5)
    # Anti-vacuity (CLAUDE.md §6): the shape still reaches the guard.
    assert dict(profile_counters_py()).get("MilpTinyEntryDecert", 0) > 0
    assert not res.gap_certified, (res.status, res.bound)
    # minlplib.solu: m3 =opt= 37.8 (minimize). No false bound.
    assert res.bound is None or res.bound <= 37.8 + 1e-6, res.bound


def test_row_scaled_m3_certifies_on_the_highs_master(monkeypatch):
    """The same m3 at 10^[-6,6] on the default convex-route master. HiGHS fits each
    master row to its own coefficient window (``milp_highs._fit_rows_to_window``),
    so the 1e-11 column ratio that trips the in-house guard never reaches a
    certificate decision; the route certifies, with no bound above the optimum.
    Measured on the route panel: +m3 at span 6, 22/26 vs 20/26 (§25.13)."""
    monkeypatch.delenv("DISCOPT_CONVEX_ROUTE_OA_MASTER", raising=False)
    res = _per_row_scaled("m3.nl", 6.0).solve(time_limit=5)
    assert "master=highs" in (res.algorithm_route or ""), res.algorithm_route
    assert res.gap_certified, (res.status, res.bound, res.algorithm_route)
    # minlplib.solu: m3 =opt= 37.8 (minimize). Certified, never above it.
    assert res.bound <= 37.8 + 1e-6, res.bound
    assert res.objective == pytest.approx(37.8, rel=1e-4)
