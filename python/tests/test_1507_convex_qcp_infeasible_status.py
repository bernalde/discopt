"""An infeasible convex QCP must end ``infeasible``, not ``error`` (#1507).

``x**2 + y**2 <= r**2`` and ``x + y >= r*sqrt(2)*(1 + eps)`` have no common point
for any ``eps > 0``: the largest ``x + y`` on the disc is ``r*sqrt(2)``. The model
is certified convex, so it takes the single-NLP fast path. For ``eps >= 0.05`` the
separable-quadratic bound tightening refutes it before the NLP runs; below that
margin the box-local rule no longer sees it (FBBT narrows the box to
``[0.679, 0.763]**2`` and stops), the NLP's restoration phase ends at Ipopt code 2
(``Infeasible_Problem_Detected``), which ``_IPOPT_STATUS_MAP`` deliberately reads
as ``error`` because a *local* infeasibility verdict proves nothing -- and the
fast path returned that bare ``error`` with ``nodes=0``.

Measured on 2b39c4c: every ``eps <= 0.02`` at every radius tried returned
``status="error", objective=None, bound=None, gap_certified=False, node_count=0``.
The fix hands a failed convex NLP that left no verified point to the spatial
B&B, whose relaxation is a sound outer approximation; its root LP is empty here,
so it proves infeasibility at node 1.

The feasible control (``eps < 0``) guards the other direction: the fall-through
must never manufacture an ``infeasible`` verdict for a model that has a point.
"""

import math

import discopt.modeling as dm
import pytest


def _disc_halfplane(r: float, eps: float, form: str, box: float = 10.0) -> dm.Model:
    m = dm.Model("disc_halfplane")
    x = m.continuous("x", lb=-box, ub=box)
    y = m.continuous("y", lb=-box, ub=box)
    sq = (lambda t: t * t) if form == "mul" else (lambda t: t**2)
    m.subject_to(sq(x) + sq(y) <= r * r)
    m.subject_to(x + y >= r * math.sqrt(2.0) * (1.0 + eps))
    m.minimize(x - y)
    return m


@pytest.mark.unit
@pytest.mark.parametrize("eps", [0.02, 0.001])
@pytest.mark.parametrize("r", [0.5, 3.0])
def test_small_margin_infeasible_convex_qcp_is_proved_infeasible(r, eps):
    res = _disc_halfplane(r, eps, "mul").solve(time_limit=30)
    assert res.status == "infeasible", (
        f"r={r} eps={eps}: expected a proved infeasibility, got status={res.status!r} "
        f"(error={res.error!r})"
    )
    assert res.gap_certified is True
    assert res.x is None and res.objective is None


@pytest.mark.unit
@pytest.mark.parametrize("form", ["pow", "mul"])
@pytest.mark.parametrize("box", [10.0, 1.0])
def test_spelling_and_box_do_not_change_the_verdict(form, box):
    res = _disc_halfplane(1.0, 0.02, form, box=box).solve(time_limit=30)
    assert res.status == "infeasible", (form, box, res.status, res.error)
    assert res.gap_certified is True


@pytest.mark.unit
@pytest.mark.parametrize("eps", [-0.02, -0.2])
def test_feasible_control_is_not_called_infeasible(eps):
    """The fall-through adds a sound search, never a verdict: a feasible twin solves."""
    res = _disc_halfplane(1.0, eps, "mul").solve(time_limit=30)
    assert res.status == "optimal", (eps, res.status, res.error)
    xv = float(res.x["x"])
    yv = float(res.x["y"])
    assert xv * xv + yv * yv <= 1.0 + 1e-6
    assert xv + yv >= math.sqrt(2.0) * (1.0 + eps) - 1e-6
