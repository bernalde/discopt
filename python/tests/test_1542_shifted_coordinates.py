"""A model written in shifted coordinates must keep its certificate (#1542).

``x = y - c`` with ``|c| ~ 1e6`` is an exact change of variables, so the shifted
model has the same optimum as the unshifted one. Two scale-dependent defects broke
that on a 5-variable convex MINLP (``c.x + 0.2 x3**3``, three linear rows), both on
the NLP-BB route:

* **certified ``infeasible`` on a feasible model.** ``_solve_node_nlp``'s fixed-box
  pre-screen measured the node's span on bounds clipped to
  ``[-STARTING_POINT_CLIP, STARTING_POINT_CLIP] = [-100, 100]``. A box lying
  entirely outside that window, here ``[1.3e6, 1.3e6 + 5]``, collapses to span 0, so
  every variable read as "pinned". The rows were then checked at ``±100``, a point
  outside the box, and the root was returned as INFEASIBLE without the NLP ever
  running. ``constants.clip_start_box`` keeps the window inside the box.
* **certified objective below the true optimum.** The NLP-BB exit gate
  (``_nonlinear_point_excess``) forgave box violations by ``1e-9 * |x_j|``. That is
  1e-3 at ``|x_j| = 1e6``, so a RENS incumbent 9.95e-5 below its declared
  ``lb = 1e6`` cleared the gate, was not replaced by its (feasible) refinement, and
  was certified ``optimal`` 3.95e-4 below the true optimum of about 0.

Both repros are the ones in the issue. Each fails on the pre-fix tree.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.constants import STARTING_POINT_CLIP, clip_start_box
from discopt.validation.feasibility import verify_point

_MIXED_SHIFT = np.array([798491.0, -1314226.0, -1100101.0, -1228561.0, 555147.0])


def _build(seed: int, shift: np.ndarray):
    rng = np.random.default_rng(seed)
    A = rng.integers(-4, 5, size=(3, 5)).astype(float)
    b = rng.integers(2, 10, size=3).astype(float)
    c = rng.integers(-5, 6, size=5).astype(float)
    m = dm.Model("cube")
    ys = [m.integer(f"i{k}", lb=0 + shift[k], ub=4 + shift[k]) for k in range(3)] + [
        m.continuous(f"c{k}", lb=0 + shift[k], ub=5 + shift[k]) for k in range(3, 5)
    ]
    x = [ys[j] - shift[j] for j in range(5)]
    for r in range(3):
        m.subject_to(sum(A[r, j] * x[j] for j in range(5)) <= b[r])
    m.minimize(sum(c[j] * x[j] for j in range(5)) + 0.2 * x[3] ** 3)
    return m, A, b, c


def _exact_objective(x_shifted, shift, c):
    """The objective in exact rational arithmetic, so no cancellation at 1e6."""
    from fractions import Fraction as F

    u = [F(float(xi)) - F(float(si)) for xi, si in zip(x_shifted, shift)]
    return float(sum(F(float(c[j])) * u[j] for j in range(5)) + F(1, 5) * u[3] ** 3)


def _flat(result, model):
    return np.array([float(np.asarray(result.x[v.name])) for v in model._variables])


@pytest.mark.parametrize(
    "seed, shift",
    [
        pytest.param(1, _MIXED_SHIFT, id="seed1-mixed-shift"),
        pytest.param(9, np.full(5, 1e6), id="seed9-uniform-1e6"),
        pytest.param(5, np.full(5, 1e6), id="seed5-uniform-1e6"),
    ],
)
def test_shifted_model_keeps_the_unshifted_answer(seed, shift):
    m0, _, _, c = _build(seed, np.zeros(5))
    r0 = m0.solve(time_limit=20)
    assert r0.status == "optimal" and r0.gap_certified
    f_star = float(r0.objective)

    m1, _, _, _ = _build(seed, shift)
    r1 = m1.solve(time_limit=20)

    # The mapped unshifted optimum is feasible, so no "infeasible" verdict is sound.
    x0 = _flat(r0, m0)
    assert verify_point(m1, x0 + shift, with_objective=True).ok
    assert r1.status in ("optimal", "feasible"), (r1.status, r1.gap_certified)
    assert r1.x, "a feasible model must return an incumbent"

    x1 = _flat(r1, m1)
    # The incumbent sits inside the DECLARED box at the repo's absolute tolerance,
    # not at a tolerance that grows with |bound|.
    lb = np.array([v.lb for v in m1._variables], dtype=float).ravel()
    ub = np.array([v.ub for v in m1._variables], dtype=float).ravel()
    assert np.all(x1 >= lb - 1e-6) and np.all(x1 <= ub + 1e-6), x1 - shift
    # ...and is not super-optimal: its exact objective is not below the true optimum.
    f1 = _exact_objective(x1, shift, c)
    assert f1 >= f_star - 1e-6 * max(1.0, abs(f_star)), (f1, f_star)
    if r1.gap_certified:
        assert r1.status == "optimal"
        assert abs(f1 - f_star) <= 1e-4 * max(1.0, abs(f_star)), (f1, f_star)
    if r1.bound is not None:
        assert float(r1.bound) <= f_star + 1e-6 * max(1.0, abs(f_star))


@pytest.mark.unit
def test_clip_start_box_stays_inside_the_box():
    C = STARTING_POINT_CLIP
    lb = np.array([-np.inf, 0.0, -5.0, 1.3e6, -1.3e6 - 5.0, 50.0, -np.inf, 1e6])
    ub = np.array([np.inf, np.inf, 5.0, 1.3e6 + 5.0, -1.3e6, 1e6, -1e6, np.inf])
    lo, hi = clip_start_box(lb, ub)
    # Unchanged wherever the box meets [-C, C]: identical to the bare clip.
    meets = (lb <= C) & (ub >= -C)
    np.testing.assert_array_equal(lo[meets], np.clip(lb, -C, C)[meets])
    np.testing.assert_array_equal(hi[meets], np.clip(ub, -C, C)[meets])
    # Always a non-degenerate finite window inside the box.
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    assert np.all(lb <= lo) and np.all(lo <= hi) and np.all(hi <= ub)
    assert np.all(hi - lo > 0)
    np.testing.assert_array_equal(lo[3:5], lb[3:5])
    np.testing.assert_array_equal(hi[3:5], ub[3:5])
    assert (lo[6], hi[6]) == (-1e6 - 2 * C, -1e6)
    assert (lo[7], hi[7]) == (1e6, 1e6 + 2 * C)


@pytest.mark.unit
def test_exit_gate_does_not_forgive_box_violation_by_magnitude():
    from discopt.solver import _NLPBB_EXIT_ABS_TOL, _nonlinear_point_excess

    box = np.array([[1e6, 1e6 + 5.0], [0.0, 5.0]])
    # 9.95e-5 below lb = 1e6: the issue's RENS incumbent.
    x = np.array([1e6 - 9.95e-5, 1.0])
    excess, where, n_cmp = _nonlinear_point_excess(None, x, None, None, box=box)
    assert n_cmp == 4
    assert excess > _NLPBB_EXIT_ABS_TOL, (excess, where)
    assert where == "bound on x[0]"
    # The same point, shifted to the origin, gets the same verdict.
    excess0, _, _ = _nonlinear_point_excess(None, x - box[:, 0], None, None, box=box - box[:, :1])
    assert excess0 > _NLPBB_EXIT_ABS_TOL
