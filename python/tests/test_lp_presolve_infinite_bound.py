"""LP presolve FBBT must not fabricate bounds from the ``INF`` sentinel.

``crates/discopt-core/src/lp/simplex/presolve.rs`` propagates row activity with
"infinity bookkeeping". It used to decide whether a column's contribution
``a_ij * x_j`` was unbounded by testing the **product** against the sentinel
``INF = 1e20``. That is wrong whenever ``|a_ij| < 1``: an unbounded column
contributes ``-0.5 * 1e20 = -5e19``, which fails a ``<= -INF`` test and is booked
as a *finite* activity. Two things then go wrong at once:

1. the row's activity range is no longer recognised as unbounded, so the pass
   derives bounds it has no right to derive; and
2. the huge magnitude annihilates every smaller term in the running sum
   (``ulp(5e19) == 8192``), so the residual ``sum_min_finite - ck_min`` evaluates
   to ``0.0`` instead of its true value.

On the minimal LP below that fabricated the tightening ``x0 >= 35.2244`` for a
column whose optimal value is ``0`` — cutting the optimum out of the box. Because
presolve is *dimension-preserving* and has no postsolve, the B&B then solved the
mutilated box and reported ``OPTIMAL`` at a value far above the true optimum: a
false dual bound, which is the one class of defect this project treats as
unconditionally fatal (see CLAUDE.md §1).

The LP here is the shrunk core of ``gear4``'s equilibrated root relaxation, where
this surfaced as a certified ``optimal`` of ``184279.32`` against a true optimum of
``1.643428474``. ``presolve.rs``'s own module docstring promises the pass "never
cuts a feasible (let alone optimal) solution"; these tests hold it to that.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from discopt.solvers.milp_simplex import solve_milp  # noqa: E402

# The shrunk gear4 core: 1 row, 2 columns. x1 alone satisfies the row (at its
# upper bound 1600 the row activity is -3051 <= -17.6), so x0 = 0 is feasible and
# optimal at objective 0.
_C = np.array([4096.0, 0.0])
_A = np.array([[-0.5, -1.9073486328125]])
_B = np.array([-17.612222262299806])
_LO = np.array([0.0, 2.5599999999999965])
_HI_FINITE = 1600.0000000000016


def _solve(hi0: float) -> tuple[str, float | None]:
    bounds = [(_LO[0], hi0), (_LO[1], _HI_FINITE)]
    r = solve_milp(c=_C, A_ub=_A, b_ub=_B, bounds=bounds, integrality=None)
    obj = None if r.objective is None else float(r.objective)
    return str(r.status), obj


@pytest.mark.parametrize(
    "hi0",
    [np.inf, 1e20, 1e21, 1e25, 1e19, 1e15, 1e8],
    ids=["inf", "sentinel_1e20", "1e21", "1e25", "1e19", "1e15", "1e8"],
)
def test_unbounded_column_does_not_fabricate_a_bound(hi0):
    """The optimum is 0.0 whatever the (large or infinite) upper bound on x0.

    ``1e20`` is the Rust sentinel itself and ``inf``/``1e21``/``1e25`` all exceed
    it, so every one of these is an "unbounded above" column. Pre-fix, exactly the
    cases at or above the sentinel returned 144279.32 with status OPTIMAL; the
    sub-sentinel ones were already correct and guard against over-correction.
    """
    status, obj = _solve(hi0)
    assert obj is not None, f"no objective returned for x0 ub={hi0}"
    assert "OPTIMAL" in status.upper(), f"unexpected status {status} for x0 ub={hi0}"
    assert obj == pytest.approx(0.0, abs=1e-6), (
        f"x0 ub={hi0}: LP optimum is 0.0 (x0=0, x1=1600 is feasible) but the solver "
        f"reported {obj!r} — presolve fabricated a lower bound on x0 and cut the optimum"
    )


def test_presolve_never_exceeds_an_independently_verified_optimum():
    """Differential fuzz against HiGHS over LPs with sentinel/infinite bounds.

    Guards the general class rather than the one shrunk instance: a *claimed*
    ``OPTIMAL`` may never sit above the true minimum. Counts its comparisons so a
    silently vacuous run (every LP infeasible/unbounded, or scipy missing) fails
    instead of passing green.
    """
    linprog = pytest.importorskip("scipy.optimize").linprog
    rng = np.random.default_rng(20260726)
    compared = 0
    violations = []

    for _ in range(300):
        n = int(rng.integers(2, 6))
        m = int(rng.integers(1, 5))
        # Coefficients below 1 in magnitude are what turn the sentinel into a
        # finite product; keep them in the mix deliberately.
        A = np.round(rng.uniform(-2.0, 2.0, size=(m, n)) * 2) / 2.0
        b = np.round(rng.uniform(-20.0, 5.0, size=m), 6)
        c = rng.integers(0, 4097, size=n).astype(float)
        bounds = []
        for _j in range(n):
            lo = float(rng.choice([0.0, rng.uniform(0.0, 5.0)]))
            hi = float(rng.choice([np.inf, 1e20, 1e21, rng.uniform(lo + 1.0, 2000.0)]))
            bounds.append((lo, hi))

        h = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        if not h.success:
            continue
        r = solve_milp(c=c, A_ub=A, b_ub=b, bounds=bounds, integrality=None)
        if r.objective is None or "OPTIMAL" not in str(r.status).upper():
            continue
        compared += 1
        obj = float(r.objective)
        truth = float(h.fun)
        if obj > truth + 1e-6 * (1.0 + abs(truth)):
            violations.append((obj, truth, c.tolist(), A.tolist(), b.tolist(), bounds))

    assert compared >= 50, (
        f"only {compared} LPs were actually compared — the fuzz is vacuous and would "
        "pass without testing anything"
    )
    assert not violations, (
        f"{len(violations)} of {compared} LPs reported OPTIMAL above the true minimum; "
        f"first: solver={violations[0][0]!r} vs highs={violations[0][1]!r}"
    )
