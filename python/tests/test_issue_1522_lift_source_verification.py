"""The native kernel must verify its incumbent against the pre-reform model (#1522).

MINLPLib ``prob09`` (``min t`` s.t. ``100*(y - x**2)**2 + (1 - x)**2 - t == 0``) is
factorably lifted before the native spatial kernel sees it: the square is
distributed into ``100*y*y - 200*_fr_aux*y + 100*x**4`` with ``_fr_aux = x**2``. The
kernel's #789 check ran ``verify_point`` on that LIFT only. On the lifted row the
term magnitudes are ~450, where the source row's are ~2, and ``verify_point``'s
allowance ``ABS_TOL * max(1, term scale)`` grows with them. So a point with a
1.6e-4 residual on the source row was allowed 4.5e-4 on the lifted row. The kernel
reported it (``node_limit``, 100k nodes), and ``warm_start.check_feasibility``
rejected it at its default ``tol=1e-4``.

Before the fix, :func:`test_node_limit_incumbent_passes_check_feasibility` returned
the kernel's incumbent at obj ~0.011 with ``|e1| > 1e-4``.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt import solver as S
from discopt import warm_start
from discopt._relax.factorable_reform import factorable_reformulate
from discopt.modeling.core import Model
from discopt.validation.feasibility import verify_point

pytestmark = pytest.mark.unit

# The incumbent the issue reported: obj 0.0033565, |e1| = 1.5976e-4.
_X, _Y = 1.03088, 1.05765
_E1_RESIDUAL = 1.5976e-4


def _prob09() -> Model:
    m = Model("prob09")
    x = m.continuous("x", lb=-2, ub=2)
    y = m.continuous("y", lb=-2, ub=2)
    t = m.continuous("t", lb=-100, ub=100)
    m.subject_to(100 * (y - x**2) ** 2 + (1 - x) ** 2 - t == 0, name="e1")
    m.minimize(t)
    return m


def _issue_point() -> tuple[np.ndarray, np.ndarray]:
    """``(source point, lifted point)``: the issue's point, with the aux exact."""
    f = 100 * (_Y - _X**2) ** 2 + (1 - _X) ** 2
    src = np.array([_X, _Y, f - _E1_RESIDUAL])
    return src, np.append(src, _X**2)


def _flat(result, names=("x", "y", "t")) -> np.ndarray:
    return np.array([float(result.x[n]) for n in names])


def test_lift_alone_accepts_a_point_the_source_rejects():
    """The mechanism itself: the lift's scale-keyed allowance covers the residual."""
    src_model = _prob09()
    lifted = factorable_reformulate(_prob09())
    assert [v.name for v in lifted._variables] == ["x", "y", "t", "_fr_aux_0"]
    src, lift = _issue_point()

    assert verify_point(lifted, lift).ok
    assert not verify_point(src_model, src).ok
    ok, viols = warm_start.check_feasibility(src_model, src)
    assert not ok and any("e1" in v for v in viols)


def test_kernel_verifier_rejects_it_when_given_the_source():
    lifted = factorable_reformulate(_prob09())
    src, lift = _issue_point()

    # Without a source this is the pre-#1522 check, and it accepts.
    assert S._native_kernel_verify_point(lifted, lift)[0] is True
    ok, obj = S._native_kernel_verify_point(lifted, lift, source=(_prob09(), 3))
    assert ok is False and obj is None
    assert S._native_kernel_source_only_failure(lifted, lift, (_prob09(), 3))

    # A genuinely feasible point still verifies against both, objective from the lift.
    good = np.array([1.0, 1.0, 0.0, 1.0])
    ok, obj = S._native_kernel_verify_point(lifted, good, source=(_prob09(), 3))
    assert ok is True and obj == pytest.approx(0.0, abs=1e-12)
    assert not S._native_kernel_source_only_failure(lifted, good, (_prob09(), 3))


def test_repair_returns_a_source_verified_point():
    lifted = factorable_reformulate(_prob09())
    _, lift = _issue_point()
    out = S._native_kernel_repair_point(lifted, lift, (_prob09(), 3), None)
    assert out is not None
    x_new, obj = out
    assert S._native_kernel_verify_point(lifted, x_new, source=(_prob09(), 3))[0]
    assert warm_start.check_feasibility(_prob09(), x_new[:3])[0]
    assert obj == pytest.approx(float(x_new[2]))


@pytest.mark.parametrize("max_nodes", [2000])
def test_node_limit_incumbent_passes_check_feasibility(monkeypatch, max_nodes):
    """The issue's exit shape: an unseeded kernel stopping at ``node_limit``.

    On the reporting machine the kernel's NLP seed found nothing and the tree ran to
    its node limit on its own McCormick incumbent. The seed is disabled here to reach
    that exit deterministically; everything downstream of it is the default path.
    """
    monkeypatch.setattr(S, "_native_kernel_seed", lambda *a, **k: (None, None))
    m = _prob09()
    r = m.solve(deterministic=True, max_nodes=max_nodes)
    assert r.status == "node_limit"
    assert r.bound is not None and r.bound <= 1e-9
    assert r.x, "the repair found a verified point here; losing it is a regression"
    ok, viols = warm_start.check_feasibility(_prob09(), _flat(r))
    assert ok, viols
    assert verify_point(_prob09(), _flat(r)).ok
    assert r.objective >= r.bound - 1e-6


def test_time_limited_exit_never_reports_a_source_infeasible_point():
    """With no budget left to repair, the kernel keeps its bound and drops the point."""
    m = _prob09()
    r = m.solve(deterministic=True, time_limit=0.5)
    assert r.bound is not None
    if r.x:
        ok, viols = warm_start.check_feasibility(_prob09(), _flat(r))
        assert ok, viols
    else:
        assert r.objective is None
