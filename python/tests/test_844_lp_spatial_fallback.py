"""#844: no-incumbent fallback to the LP-per-node spatial engine (opt-in).

A class of pure-integer MINLPs returns NO incumbent from the default NLP-per-node
path while the LP-per-node engine finds one. #826 concluded "no tractable general
primal cracks this family" after testing the NLP-based `feasibility_pump`, `rens`
and `diving`; the LP-based variants inside `lp_spatial_bb` were never tried because
they sat behind an opt-in flag.

Measured at a 60 s budget with the fallback on:

    tln4   no incumbent -> 9.0   (opt 8.3)
    tln5   no incumbent -> 11.0  (opt 10.3)
    nvs04/nvs06/nvs09/nvs15   byte-identical, still certified

It is a FALLBACK, not a route, and that ordering is the whole safety argument.
Routing in-scope models *instead of* the default path was built and panelled first
(72 instances) and FAILED: 3 gains but 4 certification regressions (nvs04
0.72 -> 6.3e8, nvs06, nvs09, nvs15). The panel split exactly along one line -- every
gain was an instance the default path leaves empty, every regression one it already
certifies -- so as a fallback the gains are kept and the regressions cannot occur.

Still opt-in (not default-ON) because the engine can overshoot the overall wall
budget on tln6 (89.7 s against 60 s). See `_lp_spatial_fallback_enabled`.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import pytest  # noqa: E402
from discopt.modeling.core import _lp_spatial_fallback_enabled  # noqa: E402


@pytest.fixture
def fb(monkeypatch):
    def _set(value):
        if value is None:
            monkeypatch.delenv("DISCOPT_LP_SPATIAL_FALLBACK", raising=False)
        else:
            monkeypatch.setenv("DISCOPT_LP_SPATIAL_FALLBACK", value)

    return _set


def _pure_integer_min() -> dm.Model:
    """In scope for the engine: all-integer, MINIMIZE. Optimum 6 at x=2, y=0."""
    m = dm.Model("pure_int_min")
    x = m.integer("x", lb=0, ub=5)
    y = m.integer("y", lb=0, ub=5)
    m.minimize(3 * x + 5 * y)
    m.subject_to(x + y >= 2)
    return m


def _mixed() -> dm.Model:
    """Out of scope: has a continuous variable."""
    m = dm.Model("mixed")
    x = m.integer("x", lb=0, ub=5)
    c = m.continuous("c", lb=0, ub=5)
    m.minimize(3 * x + c)
    m.subject_to(x + c >= 2)
    return m


def test_flag_defaults_to_off(fb):
    """Opt-in: the default path is untouched until the residual wall overshoot is
    closed (see the module docstring)."""
    fb(None)
    assert _lp_spatial_fallback_enabled() is False


@pytest.mark.parametrize("value,expected", [("1", True), ("0", False), ("off", False)])
def test_flag_parses(fb, value, expected):
    fb(value)
    assert _lp_spatial_fallback_enabled() is expected


@pytest.mark.parametrize("flag", ["0", "1"])
def test_solve_with_an_incumbent_is_unchanged(fb, flag):
    """The fallback must not touch a solve that already found an incumbent — this is
    the property that makes the four nvs certification regressions impossible."""
    fb(flag)
    r = _pure_integer_min().solve(time_limit=60)
    assert r.objective == pytest.approx(6.0, abs=1e-6)
    if r.bound is not None:
        assert r.bound <= r.objective + 1e-6, "UNSOUND: bound above incumbent"


@pytest.mark.parametrize("flag", ["0", "1"])
def test_out_of_scope_model_is_unaffected(fb, flag):
    """A mixed model can never reach the engine, so its result and its full time
    budget are untouched regardless of the flag. Optimum is 2.0 at x=0, c=2."""
    fb(flag)
    r = _mixed().solve(time_limit=60)
    assert r.objective == pytest.approx(2.0, abs=1e-4)


def test_fallback_never_breaks_a_solve(fb):
    """Even enabled, a fallback failure must surface as a warning and keep the
    default-path result — never raise. (A bare ``except`` here previously turned a
    hard TypeError into an invisible no-op.)"""
    fb("1")
    r = _pure_integer_min().solve(time_limit=60)
    assert r.objective is not None
