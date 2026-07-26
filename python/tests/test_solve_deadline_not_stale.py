"""``model._solve_deadline`` must not outlive the solve that set it.

``solve_model`` stashes a wall-clock deadline on the model (#654) so deep
relaxation machinery can self-bound without threading a parameter through every
layer. It was written in four places and cleared in none, so once the call
returned the stash held a timestamp already in the past, and any *later* consumer
reading it saw a permanently expired budget.

That silently switched off ``IncrementalMcCormickLP``'s structure build for the
#844 no-incumbent fallback — which by construction runs *after* a primary solve has
spent its budget. The engine degraded to its slow cold path and returned nothing,
and the symptom read as a *measurement* rather than as a capability turning itself
off. A whole round of debugging chased the wrong cause because of it.

The fix restores the previous value on the way out (rather than deleting
unconditionally) so nesting stays correct: a recursive ``solve_model`` (RENS
sub-solve, local-branching sub-MIP) hands the enclosing solve its own deadline back.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
from discopt.solver import solve_model_accepted_kwargs  # noqa: E402

_SENTINEL = "_solve_deadline"


def _model() -> dm.Model:
    m = dm.Model("deadline_stash")
    x = m.integer("x", lb=0, ub=5)
    y = m.integer("y", lb=0, ub=5)
    m.minimize(3 * x + 5 * y)
    m.subject_to(x + y >= 2)
    return m


def test_no_stash_left_behind_after_solve():
    """The core property: a solve leaves no expired deadline on the model."""
    m = _model()
    assert not hasattr(m, _SENTINEL), "precondition: model starts clean"
    m.solve(time_limit=30)
    assert not hasattr(m, _SENTINEL), (
        "solve left a spent _solve_deadline on the model; a later consumer would "
        "read an already-expired budget and silently degrade"
    )


def test_any_surviving_stash_would_be_in_the_past():
    """Guards the *reason* this matters: if a stash ever survives again, it is not
    merely untidy — it is actively wrong, because it is already expired."""
    m = _model()
    m.solve(time_limit=30)
    leftover = getattr(m, _SENTINEL, None)
    if leftover is not None:
        assert leftover > time.perf_counter(), (
            f"stale deadline survived and is {time.perf_counter() - leftover:.1f}s "
            "in the past — exactly the #844 failure mode"
        )


def test_stash_is_cleared_even_if_one_was_present_before():
    """A pre-existing stash is also spent by the time the solve returns, so it must
    not survive either — a caller that set one before ``Model.solve`` cannot expect
    it back, and leaving it is the same expired-budget hazard."""
    m = _model()
    m._solve_deadline = time.perf_counter() + 12345.0
    m.solve(time_limit=30)
    assert not hasattr(m, _SENTINEL), "a pre-existing stash survived the solve"


def test_decorator_does_not_break_kwarg_validation():
    """``Model.solve`` rejects misspelled keywords by introspecting
    ``solve_model``'s signature; the wrapper must stay transparent to
    ``inspect.signature`` (via ``functools.wraps``) or every legitimate kwarg would
    start being rejected."""
    accepted = solve_model_accepted_kwargs()
    assert "gap_tolerance" in accepted and "time_limit" in accepted
    assert "model" not in accepted
    assert len(accepted) > 50, f"signature introspection looks broken: {len(accepted)} kwargs"
