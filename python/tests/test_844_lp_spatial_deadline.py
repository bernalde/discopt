"""#844: the LP-per-node spatial engine must honour its wall-clock ``time_limit``.

Measured before this fix, on `ball_mk2_30` (**30 integer variables, 1 constraint**):

| config | limit | wall | overrun |
|---|---|---|---|
| default path | 30 s | 30.4 s / 30.0 s | 1.01x / 1.00x |
| `lp_spatial=True` | 30 s | 486.7 s / 500.0 s | **16.22x / 16.67x** |

Reproducible across repeats, so a deterministic engine defect rather than noise, and
specific to this engine — the default path honoured the same limit exactly, twice.
It is also **not** the residual accepted in #845 (closed NOT_PLANNED), whose rationale
is IPM + sparse-direct factorization on a *very large* single NLP; that cannot apply
to a 30-variable, 1-row model.

Three unbounded loops were responsible, none of which consulted the deadline:

* the root OBBT call, which ran ``|vars| x 2 x rounds`` LPs at up to
  ``time_limit_per_lp`` each entirely outside ``time_limit`` (up to ~150 s on
  ball_mk2_30's 30 integers) — ``obbt_tighten_root`` already accepted a ``deadline``,
  the caller simply never passed one;
* ``dive``, which solves an LP per iteration for ``2n+2`` iterations, at the root and
  again at every node;
* ``feasibility_pump``, likewise one LP per iteration for ``max_iter`` iterations.

After the fix the engine honours its budget to 1.00x while still finding the same
incumbents (tln6 83.1 and tln4 13.3 at a 14 s budget).
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import pytest  # noqa: E402
from discopt._jax.lp_spatial_bb import _is_in_scope, solve_lp_spatial_bb  # noqa: E402


def _integer_model(n: int = 12) -> dm.Model:
    """A pure-integer MINIMIZE model with bilinear coupling — in scope for the engine
    and hard enough that it will not terminate before the deadline."""
    m = dm.Model("deadline_probe")
    xs = [m.integer(f"x{i}", lb=0, ub=20) for i in range(n)]
    m.minimize(sum((i + 1) * xs[i] for i in range(n)))
    for i in range(n - 1):
        m.subject_to(xs[i] * xs[i + 1] >= 12)
    return m


def test_model_is_in_scope():
    """Guard: if this model stopped being in scope the deadline test below would
    vacuously pass by bailing out immediately."""
    assert _is_in_scope(_integer_model()) is True


@pytest.mark.parametrize("limit", [3.0, 6.0])
def test_engine_honours_its_time_limit(limit):
    """The engine must return within a small multiple of its budget.

    Pre-fix this overran by 16x on ball_mk2_30; the bound here is deliberately loose
    (3x) so the test asserts "the deadline is actually polled" rather than pinning
    machine-specific timing, yet still fails hard on a 16x regression.
    """
    t0 = time.perf_counter()
    solve_lp_spatial_bb(_integer_model(), time_limit=limit, gap_tolerance=1e-4)
    wall = time.perf_counter() - t0
    assert wall < 3.0 * limit, (
        f"engine overran its {limit}s budget: {wall:.1f}s ({wall / limit:.1f}x)"
    )


def test_root_obbt_is_budgeted():
    """The root OBBT pass must be deadline-bounded. It previously ran outside
    ``time_limit`` entirely, which was the single largest contributor."""
    t0 = time.perf_counter()
    solve_lp_spatial_bb(_integer_model(16), time_limit=3.0, gap_tolerance=1e-4, use_obbt=True)
    wall = time.perf_counter() - t0
    assert wall < 9.0, f"root OBBT escaped the budget: {wall:.1f}s against a 3s limit"


def test_result_is_sound_when_one_is_returned():
    """Honouring the deadline must not cost soundness: any incumbent returned still
    carries a dual bound that does not cross it."""
    res = solve_lp_spatial_bb(_integer_model(), time_limit=6.0, gap_tolerance=1e-4)
    if res is not None and res.objective is not None and res.bound is not None:
        assert res.bound <= res.objective + 1e-6 * (1 + abs(res.objective)), (
            f"UNSOUND: bound {res.bound} > incumbent {res.objective}"
        )
