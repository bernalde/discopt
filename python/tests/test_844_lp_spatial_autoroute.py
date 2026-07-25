"""#844: auto-route pure-integer MINIMIZE models to the LP-per-node spatial engine.

The LP-per-node engine (`_jax/lp_spatial_bb.py`) solves a McCormick LP per node,
branches on integers, and runs LP-based dive/feasibility-pump primals. Measured on
the #844 family, the default NLP-per-node path returns **no incumbent** on
tln4/tln5/tln6/ball_mk2_30 while this engine finds one on every one of them
(tln4 9.5 vs opt 8.3; ball_mk2_30 reaches the exact optimum 0.0).

That capability was invisible for two issues — #826 and #844 both concluded "no
tractable general primal cracks this family" — purely because it sat behind the
opt-in `solve(lp_spatial=True)`. The fix is to auto-detect and route on the
engine's own *exact* structural precondition (`_is_in_scope`: pure-integer AND
minimize), the way the convex fast path already auto-detects convexity, rather
than adding another flag a user must already know about.

Safety contract under test:

* flag **default OFF** ⇒ routing decision is False even for an in-scope model, so
  every solve is bit-identical to the pre-#844 path (pending the §5 panel);
* the pre-existing explicit `lp_spatial=True` kwarg is honoured unchanged;
* scope is evaluated **before** dispatch, so an out-of-scope model is *never*
  diverted and pays no cost (the engine also self-checks scope on its first line,
  so this is defence in depth: gastrans040 reaches the same certified optimum
  either way at a fair budget);
* `DISCOPT_LP_SPATIAL_AUTO=0` is the opt-out.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import pytest  # noqa: E402
from discopt.solver import _lp_spatial_route  # noqa: E402


@pytest.fixture
def auto_flag(monkeypatch):
    """Set/clear DISCOPT_LP_SPATIAL_AUTO without leaking into other tests."""

    def _set(value):
        if value is None:
            monkeypatch.delenv("DISCOPT_LP_SPATIAL_AUTO", raising=False)
        else:
            monkeypatch.setenv("DISCOPT_LP_SPATIAL_AUTO", value)

    return _set


def _pure_integer_min() -> dm.Model:
    """In scope: every variable integer, MINIMIZE. Optimum 6 at x=2, y=0."""
    m = dm.Model("pure_int_min")
    x = m.integer("x", lb=0, ub=5)
    y = m.integer("y", lb=0, ub=5)
    m.minimize(3 * x + 5 * y)
    m.subject_to(x + y >= 2)
    return m


def _mixed() -> dm.Model:
    """Out of scope: has a continuous variable (the gastrans040/portfol shape)."""
    m = dm.Model("mixed")
    x = m.integer("x", lb=0, ub=5)
    c = m.continuous("c", lb=0, ub=5)
    m.minimize(3 * x + c)
    m.subject_to(x + c >= 2)
    return m


def _pure_integer_max() -> dm.Model:
    """Out of scope: pure integer but MAXIMIZE (the rsyn0805m04hfsg shape)."""
    m = dm.Model("pure_int_max")
    x = m.integer("x", lb=0, ub=5)
    m.maximize(x)
    m.subject_to(x <= 3)
    return m


def test_flag_defaults_off_so_solves_are_bit_identical(auto_flag):
    """Default OFF: even a perfectly in-scope model is NOT auto-routed, so the
    pre-#844 behaviour is preserved until the §5 panel graduates the flag."""
    auto_flag(None)
    assert _lp_spatial_route(_pure_integer_min(), {}) is False


def test_explicit_kwarg_still_routes_regardless_of_flag(auto_flag):
    """Back-compat: `solve(lp_spatial=True)` is honoured unchanged, flag or not."""
    for value in (None, "0", "1"):
        auto_flag(value)
        assert _lp_spatial_route(_mixed(), {"lp_spatial": True}) is True


def test_auto_routes_only_in_scope_models(auto_flag):
    """With auto-routing on, an in-scope model routes and out-of-scope models do
    not. The negative cases are the contract that matters: a mixed model
    (gastrans040/portfol shape) and a maximize model (rsyn0805m04hfsg shape) must
    never be diverted."""
    auto_flag("1")
    assert _lp_spatial_route(_pure_integer_min(), {}) is True
    assert _lp_spatial_route(_mixed(), {}) is False
    assert _lp_spatial_route(_pure_integer_max(), {}) is False


def test_opt_out_disables_auto_routing(auto_flag):
    auto_flag("0")
    assert _lp_spatial_route(_pure_integer_min(), {}) is False


def test_routing_never_raises_on_a_degenerate_model(auto_flag):
    """Routing must never break a solve: an objective-less model returns a bool."""
    auto_flag("1")
    m = dm.Model("no_obj")
    m.integer("x", lb=0, ub=3)
    assert _lp_spatial_route(m, {}) in (True, False)


@pytest.mark.parametrize("auto", ["0", "1"])
def test_in_scope_solve_is_correct_and_sound_either_way(auto_flag, auto):
    """End-to-end: the auto-routed engine returns the true optimum with a sound
    certificate, and so does the default path — routing must not change the answer."""
    auto_flag(auto)
    r = _pure_integer_min().solve(time_limit=60)
    assert r.objective == pytest.approx(6.0, abs=1e-6), f"AUTO={auto}: wrong optimum"
    if r.bound is not None:
        assert r.bound <= r.objective + 1e-6, f"AUTO={auto}: UNSOUND bound > incumbent"


def test_out_of_scope_result_is_unchanged_by_the_flag(auto_flag):
    """The no-regression sibling: a mixed model's result must be identical with the
    flag on and off, since it is never diverted."""
    auto_flag("0")
    off = _mixed().solve(time_limit=60)
    auto_flag("1")
    on = _mixed().solve(time_limit=60)
    assert off.status == on.status
    assert off.objective == pytest.approx(on.objective, rel=1e-9, abs=1e-9)
