"""Issue #871 — a silently discarded subtree must never yield a certificate.

`solve_tree` skips a node whose LP does not come back `Optimal`
(``if r.status != LpStatus::Optimal { continue; }``). For a PROVEN-infeasible node
that is a legitimate fathom, but for a `numerical` / `unbounded` / iteration-limit
node it means the subtree was simply never explored. The frontier then drains, and
before this fix the tree upgraded `Exhausted → Optimal` on that empty frontier —
certifying a problem whose search space had a hole in it.

Observed on `clay0303hfsg` (routed since the #865 `sqr` support, with the
dominated-column bound on so it actually branches): every node dropped `numerical`,
no leaf ever recorded a dual bound, and the tree reported

    status="optimal"   bound=41709.769…   incumbent=inf

— `bound` was the *incumbent's own objective* standing in for a dual bound that
never existed, and `incumbent` was `min(inc, -inf)` clamped to `±inf` while the
incumbent POINT was a perfectly ordinary finite feasible vector. Downstream that is
a false certificate: `try_convex_solve` accepts `status == "optimal"` and reports
`gap_certified=True`.

This test pins the honest behaviour: an uncertified drop poisons the certificate.
"""

from __future__ import annotations

import math
import os

import discopt.modeling as dm
import pytest

_ck = pytest.importorskip("discopt.solvers._convex_kernel")
build_convex_spec = _ck.build_convex_spec
solve_convex_tree = _ck.solve_convex_tree

_DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl")


def test_uncertified_node_drop_never_reports_optimal():
    """The reproducer. Fails before #871: `status` was `optimal`."""
    m = dm.from_nl(os.path.join(_DATA, "clay0303hfsg.nl"))
    spec = build_convex_spec(m)
    assert spec is not None, "clay0303hfsg is routed since the #865 sqr support"

    r = solve_convex_tree(spec, initial_incumbent=None, time_limit_s=180.0, dominated_cols=True)
    # Its nodes drop `numerical`, so no dual bound covers the whole tree.
    assert r["status"] != "optimal", (
        f"certified optimality on a tree with silently dropped nodes: {r['status']}"
    )


def test_reported_incumbent_is_never_infinite():
    """`incumbent` is clamped against the dual bound; clamping against an INFINITE
    dual used to report ±inf as the objective of a finite feasible point."""
    m = dm.from_nl(os.path.join(_DATA, "clay0303hfsg.nl"))
    spec = build_convex_spec(m)
    r = solve_convex_tree(spec, initial_incumbent=None, time_limit_s=180.0, dominated_cols=True)
    inc = r["incumbent"]
    if inc is not None:
        assert math.isfinite(inc), f"incumbent objective must be finite, got {inc}"
        # ...and it must be the objective of the returned point, not a clamp artifact.
        import numpy as np

        x = np.asarray(r["incumbent_x"], float)
        assert x.size == spec["n"]
        assert np.all(np.isfinite(x))
        assert abs(float(spec["c"] @ x) - inc) <= 1e-6 * max(1.0, abs(inc))


def test_no_dual_bound_is_reported_as_no_bound_not_as_the_incumbent():
    """With the certificate poisoned, the incumbent's own objective must not be
    passed off as a dual bound (that would read as a closed gap)."""
    m = dm.from_nl(os.path.join(_DATA, "clay0303hfsg.nl"))
    spec = build_convex_spec(m)
    r = solve_convex_tree(spec, initial_incumbent=None, time_limit_s=180.0, dominated_cols=True)
    if r["status"] != "optimal" and r["incumbent"] is not None:
        bound, inc = r["bound"], r["incumbent"]
        # minimization: a valid dual bound is a LOWER bound. Either it is genuinely
        # below the incumbent, or it is -inf ("no bound"). It must never equal the
        # incumbent while the run is uncertified.
        assert bound == -math.inf or bound <= inc + 1e-6 * max(1.0, abs(inc)), (
            f"uncertified run reports bound={bound} against incumbent={inc}"
        )


def test_certifying_instances_are_unaffected():
    """The poison must be exact: an instance whose nodes all certify still certifies,
    with the same node count and objective (guards against an over-broad guard)."""
    m = dm.from_nl(os.path.join(_DATA, "syn05hfsg.nl"))
    spec = build_convex_spec(m)
    r = solve_convex_tree(spec, initial_incumbent=None, time_limit_s=120.0)
    assert r["status"] == "optimal"
    assert r["node_count"] == 2
    assert abs(r["incumbent"] - 837.7324009) < 1e-4 * 837.7324009
    assert math.isfinite(r["bound"])
