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
# Rust MILP equilibration under DISCOPT_LP_ROW_PRESCALE (#1537).
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


def test_row_scaled_m3_certifies_with_row_prescale(monkeypatch):
    """m3 with rows scaled in 10^[-6,6]. An OA-master column for a variable that
    appears in a row scaled by ~1e5 and one scaled by ~1e-6 holds a 1e-11 ratio,
    which the column-first equilibration read as noise. The #1296 guard then
    (correctly) withdrew every master certificate and the solve ended uncertified.
    The pow2 row pre-pass judges the column on row-normalised entries."""
    from discopt._rust import profile_counters_py, profile_reset_py

    monkeypatch.setenv("DISCOPT_PROFILE", "1")
    # OFF arm, anti-vacuity: the shape still reaches the guard and loses the cert.
    monkeypatch.setenv("DISCOPT_LP_ROW_PRESCALE", "0")
    profile_reset_py()
    off = _per_row_scaled("m3.nl", 6.0).solve(time_limit=5)
    assert dict(profile_counters_py()).get("MilpTinyEntryDecert", 0) > 0
    assert not off.gap_certified, (off.status, off.bound)

    monkeypatch.setenv("DISCOPT_LP_ROW_PRESCALE", "1")
    profile_reset_py()
    on = _per_row_scaled("m3.nl", 6.0).solve(time_limit=60)
    assert dict(profile_counters_py()).get("MilpTinyEntryDecert", 0) == 0
    assert on.gap_certified, (on.status, on.bound)
    # minlplib.solu: m3 =opt= 37.8 (minimize).
    assert on.bound <= 37.8 + 1e-6
    assert on.objective == pytest.approx(37.8, abs=1e-4)

    # Default (#1537): the flag is default OFF (the re-panel lost a certificate),
    # so with the variable unset the legacy factors run and the guard fires.
    monkeypatch.delenv("DISCOPT_LP_ROW_PRESCALE")
    profile_reset_py()
    _per_row_scaled("m3.nl", 6.0).solve(time_limit=5)
    assert dict(profile_counters_py()).get("MilpTinyEntryDecert", 0) > 0


_HDA_ROOT_PROBE = """
import sys
import numpy as np
from discopt._relax.mccormick_lp import MccormickLPRelaxer
from discopt._rust import profile_counters_py, profile_reset_py
from discopt.modeling.core import from_nl

model = from_nl(sys.argv[1])
lb = np.concatenate([np.asarray(v.lb, float).ravel() for v in model._variables])
ub = np.concatenate([np.asarray(v.ub, float).ravel() for v in model._variables])
profile_reset_py()
res = MccormickLPRelaxer(model).solve_at_node(lb, ub)
c = dict(profile_counters_py())
print("RESULT", res.lower_bound, c.get("PrescaleLegacyRetries", 0),
      c.get("PrescaleLegacyRescues", 0))
"""


def _hda_root(arm: str):
    """hda's raw-box root bound in a fresh interpreter, so the two arms share no
    relaxer state."""
    import subprocess
    import sys
    from pathlib import Path

    path = Path(__file__).parent / "data" / "minlplib_nl" / "hda.nl"
    env = dict(os.environ, DISCOPT_LP_ROW_PRESCALE=arm, DISCOPT_PROFILE="1")
    out = subprocess.run(
        [sys.executable, "-c", _HDA_ROOT_PROBE, str(path)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    line = [ln for ln in out.splitlines() if ln.startswith("RESULT ")]
    assert len(line) == 1, out
    _, bound, retries, rescues = line[0].split()
    return float(bound), int(retries), int(rescues)


def test_hda_root_bound_does_not_regress_under_row_prescale():
    """PR #1604 CI: the pre-passed factors make hda's row-filtered root LP
    (m=2906) end ``Numerical``. The scaled range matches the legacy factors
    (1e5.8), but the pivot path's Forrest-Tomlin updates fail, and the root
    bound fell from -64675.25 to -2.9e14. Neither factor set dominates: under
    pow2 row rescalings of hda the legacy factors fail as often. So a pre-pass
    failure is retried once on the legacy factors. The ON bound must match OFF,
    and the retry must have fired."""
    off, off_retries, _ = _hda_root("0")
    on, on_retries, on_rescues = _hda_root("1")
    assert off_retries == 0  # OFF never builds pre-passed factors
    assert on_retries >= 1 and on_rescues >= 1, (on_retries, on_rescues)
    # minlplib.solu: hda =opt= -5964.534084 (minimize) -- both bounds sound.
    assert off <= -5964.534084 and on <= -5964.534084
    assert on == pytest.approx(off, rel=1e-6), (on, off)
