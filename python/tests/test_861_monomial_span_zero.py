"""#861 — the monomial envelope no longer declines when the root box spans zero.

``IncrementalMcCormickLP`` gated EVERY monomial ``x_i**p`` on a sign-definite root
box, so a model whose integers straddle zero (ball_mk2_30: 30 integers, one
sign-mixed "thin shell" row) declined the whole incremental structure — and with it
the cuts, the feasibility pump and, under ``require_incremental=True``, the entire
LP-per-node solve, which then returned no incumbent at all.

The gate was wider than the mathematics. ``x**p`` for EVEN ``p`` has
``f'' = p(p-1)x**(p-2) >= 0`` on all of R, so it is convex across a sign change and
the cold build emits the *same* 4-row secant/tangent envelope in every sign regime —
measured on ``build_milp_relaxation`` before this change:

    p=2 box=[-2,3]  -> 4 rows      p=2 box=[1,3] -> 4 rows    p=2 box=[-3,-1] -> 4 rows
    p=4 box=[-2,3]  -> 4 rows      p=4 box=[1,3] -> 4 rows    p=4 box=[-3,-1] -> 4 rows
    p=3 box=[-2,3]  -> 2 rows      p=3 box=[1,3] -> 4 rows    p=3 box=[-3,-1] -> 4 rows

Only the ODD powers change facet COUNT across the sign change (the S-shaped atom's
2-facet hull vs the 4-row envelope), and only those are still unmappable by a fixed
sparsity pattern. So even powers are admitted on any root box; odd powers keep the
sign-definite requirement.

Two things had to be generalized before the gate could move, and both are pinned
here: the aux-column enclosure (the old endpoint ``min``/``max`` assumed monotonicity
and would have FLOORED ``x**2`` above zero on a straddling box — cutting off the true
point ``x=0``), and the validation gate's row comparison (a *pinned* box, reached
whenever integer branching fixes a variable, gets no envelope rows from the cold
build but four exactly-tight — hence vacuous — rows from the fixed-pattern patch).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.incremental_mccormick import IncrementalMcCormickLP, _monomial_aux_bounds
from discopt._jax.lp_spatial_bb import solve_lp_spatial_bb
from discopt._jax.term_classifier import classify_nonlinear_terms


def _monomial_model(p: int, lo: float, hi: float, n: int = 3):
    """``n`` integers over ``[lo,hi]`` carrying bare ``x_i**p`` monomials."""
    m = dm.Model(f"mono_p{p}")
    xs = [m.integer(f"x{i}", lb=lo, ub=hi) for i in range(n)]
    m.minimize(sum(x**p for x in xs))
    m.subject_to(sum(x**p for x in xs) >= 1)
    return m


def _ball_mk2_class(n: int = 30):
    """The ball_mk2_30 class: ``n`` integers whose root boxes straddle zero, a single
    sign-mixed row over all of them, MINIMIZE, optimum 0 at the origin. Named
    instances are gate probes only — this is the *shape* that #861 declined."""
    m = dm.Model("ball_mk2_class")
    xs = [m.integer(f"x{i}", lb=-1, ub=1) for i in range(n)]
    m.minimize(sum(x * x for x in xs))
    shell = sum((-1.0) ** i * x for i, x in enumerate(xs))
    m.subject_to(shell <= 1)
    m.subject_to(shell >= -1)
    return m


def _structure(model):
    return IncrementalMcCormickLP(model, classify_nonlinear_terms(model))


# --------------------------------------------------------------------------- #
# The gate itself
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("p", [2, 4, 6])
def test_even_power_monomial_maps_on_a_root_box_spanning_zero(p):
    """The #861 regression: before the fix this raised ``monomial x_0^{p}: root box
    spans zero (unmappable)`` and left ``ok=False``."""
    assert _structure(_monomial_model(p, -2, 2)).ok


@pytest.mark.parametrize("p", [2, 3, 4, 5])
@pytest.mark.parametrize("box", [(0, 3), (-3, 0)])
def test_sign_definite_root_still_maps_for_every_power(p, box):
    """Unchanged behaviour on the regime that already worked."""
    assert _structure(_monomial_model(p, *box)).ok


@pytest.mark.parametrize("p", [3, 5])
def test_odd_power_monomial_still_declines_on_a_root_box_spanning_zero(p):
    """An odd power's envelope switches between the 4-row secant/tangent hull and the
    2-facet S-hull across zero — a facet-COUNT change the fixed sparsity pattern
    cannot express, so it must keep declining (soundly: the caller cold-builds)."""
    assert not _structure(_monomial_model(p, -2, 2)).ok


# --------------------------------------------------------------------------- #
# Aux-column enclosure parity (the soundness prerequisite)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("p", [2, 3, 4, 5, 6])
@pytest.mark.parametrize(
    "box", [(-2.0, 3.0), (1.0, 3.0), (-3.0, -1.0), (0.0, 2.5), (-2.5, 0.0), (-1.5, -1.5)]
)
def test_monomial_aux_bounds_match_interval_pow(p, box):
    """``_monomial_aux_bounds`` must reproduce the enclosure the COLD build takes from
    ``Interval.__pow__`` — not merely a sound one — or the two paths describe
    different polytopes. Pins the parity so a change to the interval arithmetic
    surfaces here rather than as a silent bound difference."""
    from discopt._jax.convexity.interval import Interval

    lo, hi = box
    ref = Interval.from_bounds(np.array([lo]), np.array([hi])) ** p
    got = _monomial_aux_bounds(lo, hi, p)
    assert got[0] == pytest.approx(float(ref.lo[0]), rel=1e-9, abs=1e-9)
    assert got[1] == pytest.approx(float(ref.hi[0]), rel=1e-9, abs=1e-9)


def test_even_power_aux_floor_admits_the_origin_on_a_straddling_box():
    """The old endpoint-``min`` form returned ``min(li^2, ui^2) = 1`` on ``[-1, 3]``,
    which floors ``x**2`` above its true value at ``x=0`` and cuts off a feasible
    point. The enclosure must reach 0 wherever the box contains 0."""
    lo, hi = _monomial_aux_bounds(-1.0, 3.0, 2)
    assert lo <= 0.0 <= hi
    assert hi == pytest.approx(9.0)


# --------------------------------------------------------------------------- #
# Soundness of the admitted envelope
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("p", [2, 4])
def test_spanning_monomial_envelope_cuts_no_feasible_point(p):
    """Feasible-point sampling: every ``(x_i, x_i**p)`` with ``x_i`` in the node box
    must satisfy every patched envelope row. Run over sub-boxes that straddle zero,
    sit on either side of it, and pin the variable."""
    inc = _structure(_monomial_model(p, -2, 2))
    assert inc.ok
    boxes = [(-2.0, 2.0), (-2.0, 0.0), (0.0, 2.0), (-1.0, 1.0), (1.0, 1.0), (-2.0, -2.0)]
    for lo, hi in boxes:
        lb = np.full(inc.n, lo)
        ub = np.full(inc.n, hi)
        A, b, bounds = inc.assemble(lb, ub)
        A = A.tocsr()
        for (i, a, pw), rows in inc.mono_rows.items():
            aux_lo, aux_hi = bounds[a]
            for v in np.linspace(lo, hi, 9):
                s = v**pw
                assert aux_lo - 1e-9 <= s <= aux_hi + 1e-9, f"aux bound cuts x={v}"
                for k in rows:
                    lhs = 0.0
                    for t in range(A.indptr[k], A.indptr[k + 1]):
                        col = int(A.indices[t])
                        if col == i:
                            lhs += float(A.data[t]) * v
                        elif col == a:
                            lhs += float(A.data[t]) * s
                    assert lhs <= float(b[k]) + 1e-9, f"row {k} cuts x={v} (p={pw})"


@pytest.mark.parametrize("p", [2, 4])
def test_spanning_monomial_patch_is_bound_neutral_against_the_cold_build(p):
    """The incremental path may change speed, never the bound: the patched LP's
    optimal value must equal the value of the same-flag cold build on the same box,
    including boxes that straddle zero."""
    inc = _structure(_monomial_model(p, -2, 2))
    assert inc.ok
    rng = np.random.default_rng(861)
    compared = 0
    for _ in range(10):
        lb = rng.integers(-2, 3, size=inc.n).astype(float)
        ub = np.array([rng.integers(int(v), 3) for v in lb], dtype=float)
        patched = inc.solve_assembled(*inc.assemble(lb, ub))[0]
        Af, bf, bdf, _, _, _ = inc._full_build(lb, ub)
        cold = inc.solve_assembled(Af, bf, bdf)[0]
        if patched is None or cold is None:
            continue
        assert patched == pytest.approx(cold, rel=1e-9, abs=1e-9)
        compared += 1
    assert compared >= 5, "too few comparable boxes to call this a check"


@pytest.mark.parametrize("p", [2, 4])
def test_spanning_monomial_bound_never_exceeds_the_box_optimum(p):
    """The relaxation is a valid lower bound: on a fixed box its LP value must not
    exceed the true (brute-forced) integer optimum over that box."""
    lo, hi = -2, 2
    n = 3
    model = _monomial_model(p, lo, hi, n=n)
    inc = _structure(model)
    assert inc.ok
    for box in [(-2, 2), (-2, 0), (0, 2), (-1, 1)]:
        lb = np.full(n, float(box[0]))
        ub = np.full(n, float(box[1]))
        bound = inc.solve_assembled(*inc.assemble(lb, ub))[0]
        if bound is None:
            continue
        true = None
        for pt in itertools.product(range(box[0], box[1] + 1), repeat=n):
            v = np.array(pt, dtype=float)
            if float(np.sum(v**p)) >= 1.0 - 1e-9:
                obj = float(np.sum(v**p))
                true = obj if true is None else min(true, obj)
        if true is not None:
            assert bound <= true + 1e-6, f"bound {bound} above box optimum {true}"


# --------------------------------------------------------------------------- #
# The validation gate's vacuous-row filter
# --------------------------------------------------------------------------- #


def test_rowset_drops_only_rows_that_cannot_cut_the_box():
    """The filter that lets a *pinned* box validate must drop exactly the rows whose
    maximum over the box already satisfies them, and keep every row that bites."""
    import scipy.sparse as sp

    A = sp.csr_matrix(np.array([[1.0, 1.0], [1.0, 0.0]]))
    b = np.array([10.0, 0.5])  # row 0 vacuous over the box below, row 1 cuts it
    bounds = np.array([[0.0, 1.0], [0.0, 2.0]])
    unfiltered = IncrementalMcCormickLP._rowset(A, b)
    filtered = IncrementalMcCormickLP._rowset(A, b, bounds)
    assert len(unfiltered) == 2
    assert filtered == [(((0, 1.0),), 0.5)]


def test_pinned_variable_box_validates_for_a_spanning_root():
    """``_validation_boxes`` drives every spanning var through a degenerate
    (``lb==ub``) trial — reachable whenever integer branching fixes a variable. The
    cold build emits no envelope rows at zero width while the fixed pattern must fill
    its four reserved rows, so the structure only validates because those rows are
    exactly tight (vacuous) there. Assert the regime is actually exercised."""
    inc = _structure(_monomial_model(2, -2, 2))
    assert inc.ok
    assert {"span", "degen", "neg"} <= inc._validated_regimes


# --------------------------------------------------------------------------- #
# Gate probe: the class the issue was filed against
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [10, 30])
def test_ball_mk2_class_is_admitted_and_solved_under_require_incremental(n):
    """Before #861 this returned ``None``: the structure declined on
    ``monomial x_0^2: root box spans zero`` and ``require_incremental=True`` (PR #858)
    turned that into "no incumbent". It must now certify the optimum, 0.0."""
    result = solve_lp_spatial_bb(_ball_mk2_class(n), time_limit=60.0, require_incremental=True)
    assert result is not None, "engine still declines the ball_mk2_30 class"
    assert result.status == "optimal"
    assert result.objective == pytest.approx(0.0, abs=1e-6)
    assert result.bound <= result.objective + 1e-6  # certificate invariant
