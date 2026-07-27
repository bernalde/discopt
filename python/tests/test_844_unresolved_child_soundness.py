"""#844 follow-up: a child whose relaxation cannot be resolved must not be dropped.

``child()`` used to silently discard any node whose LP came back without a bound.
But ``None`` conflated two very different outcomes:

* the LP feasible set over the child's box is provably empty — a rigorous fathom,
  since the McCormick polytope is a valid OUTER approximation, so an empty
  relaxation means the subtree holds no feasible point;
* the LP solve simply **failed** (numerical error, time limit, or an ``infeasible``
  claim with no Farkas proof) — in which case the subtree is *not* ruled out.

Dropping the second kind removes live space from the search. If the heap then
exhausts, the engine declares ``status="optimal"`` over a region it never examined:
a false optimality certificate, the worst error class (CLAUDE.md §1).

The fix threads a verdict out of ``node_relax`` (``"optimal"`` / ``"fathom"`` /
``"unresolved"``, where ``"fathom"`` requires a *verified Farkas dual ray*) and
folds an unresolved child's parent bound — a valid lower bound over the child's box
— into ``unresolved_lb``, the floor that already gates the optimality claim.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt._jax.lp_spatial_bb as lpsb  # noqa: E402
import discopt.modeling as dm  # noqa: E402
import pytest  # noqa: E402


def _model() -> dm.Model:
    """Pure-integer MINIMIZE with bilinear coupling: in scope, and needs branching."""
    m = dm.Model("unresolved_child")
    xs = [m.integer(f"x{i}", lb=0, ub=8) for i in range(4)]
    m.minimize(sum((i + 1) * xs[i] for i in range(4)))
    for i in range(3):
        m.subject_to(xs[i] * xs[i + 1] >= 6)
    return m


def test_baseline_is_certifiable():
    """Control: undisturbed, this instance certifies. Without this the injection
    test below could pass vacuously (never having had a certificate to lose)."""
    res = lpsb.solve_lp_spatial_bb(_model(), time_limit=30.0, gap_tolerance=1e-4)
    assert res is not None
    assert res.status == "optimal", f"baseline did not certify: {res.status}"


def test_failed_child_relaxation_never_yields_a_false_certificate(monkeypatch):
    """Inject unresolvable child LPs into a PROVABLY FEASIBLE model and require the
    engine to make no certified claim about the space it could not examine.

    Measured before the fix, with 3 injected relaxation failures::

        without fix:  status="infeasible"     <-- FALSE INFEASIBILITY CERTIFICATE
        with fix:     status="time_limit"     <-- honest "could not resolve"

    The model is demonstrably feasible (the undisturbed solve certifies optimum
    24.0), so ``infeasible`` was a false certificate: every failed child was dropped
    silently, the heap emptied with no incumbent, and the engine concluded the
    feasible set was empty. Folding an unresolved child into ``unresolved_lb`` makes
    that conclusion impossible.
    """
    truth = lpsb.solve_lp_spatial_bb(_model(), time_limit=30.0, gap_tolerance=1e-4)
    assert truth is not None and truth.status == "optimal", "control lost its certificate"

    real = lpsb._relax_bound
    state = {"calls": 0, "failures": 0}

    def flaky(model, terms, lb, ub, **kw):
        # ``**kw`` passes through whatever optional arguments the real
        # ``_relax_bound`` grows (``deadline`` since #860) so the injection keeps
        # testing the unresolved-child path rather than failing on a signature.
        state["calls"] += 1
        if state["calls"] > 3:  # let the root and first nodes succeed
            state["failures"] += 1
            return None
        return real(model, terms, lb, ub, **kw)

    monkeypatch.setattr(lpsb, "_relax_bound", flaky)
    res = lpsb.solve_lp_spatial_bb(_model(), time_limit=30.0, gap_tolerance=1e-4)

    # The injection must actually have fired, or this asserts nothing.
    assert state["failures"] > 0, "injection never failed a solve — test would be vacuous"
    assert res is not None

    # THE soundness property: no certified verdict over unexamined space.
    assert res.status != "infeasible", (
        "FALSE INFEASIBILITY: engine declared a feasible model infeasible after "
        f"dropping {state['failures']} unresolvable children (true optimum "
        f"{truth.objective})"
    )
    if res.status == "optimal":
        assert res.objective == pytest.approx(truth.objective, rel=1e-6), (
            f"FALSE OPTIMUM: certified {res.objective} over unexamined space "
            f"(true optimum {truth.objective})"
        )


def test_verdict_contract_is_tri_state():
    """``node_relax`` must expose a verdict, and 'fathom' must never be the verdict
    for an uncertified failure — that is the distinction the fix rests on."""
    import inspect

    src = inspect.getsource(lpsb.solve_lp_spatial_bb)
    assert '"fathom"' in src and '"unresolved"' in src, "verdict contract missing"
    # a fathom must require the Farkas proof, not merely an 'infeasible' label
    assert '_st == "infeasible" and _farkas' in src, (
        "fathoming must require a verified Farkas dual ray"
    )
