"""#1537 C: the convexity recognisers must not depend on how a sum is spelled.

Root cause of the one certificate ``DISCOPT_RECENTRE`` lost on the graduation
panel (``clay0303hfsg`` under ``x = y - c``, c ~ 1e3): the recentring rewrite
folds every scalar affine subtree into an n-ary :class:`SumOverExpression`, so the
perspective denominator ``0.001 + 0.999 * y`` became ``Σ[2 terms]``. The
structural equality / hash the perspective recogniser uses to match
``((g / L) ** 2) * L`` handled ``BinaryOp`` only, so every one of the 36 hull
rows lost its CONVEX verdict (150 -> 114 convex rows), the model left the OA
route and timed out with bound 0.0 -- *with no variable moved at all* (rebuilt
through ``_recentre._build`` with an empty plan) and equally with all 99 moved.
The translation itself was harmless: ``translate(clay, 10)`` certifies.

The same miss hits a user who writes ``L`` with ``dm.sum([...])``, so the fix is
in the recognisers (``patterns._expr_struct_eq``, ``rules._struct_hash``,
``_flatten_sum_terms``, ``_has_positive_lower_bound``, ``_is_nonneg_domain``),
not in the recentring rewrite.
"""

from __future__ import annotations

import os

import discopt.modeling as dm
import numpy as np
import pytest
from _invariance import translate
from discopt._relax.convexity.patterns import _expr_struct_eq
from discopt._relax.convexity.rules import _struct_hash, classify_constraint, classify_model
from discopt.modeling import _recentre
from discopt.modeling.core import SumOverExpression

CORPUS = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl")
CLAY = os.path.join(CORPUS, "clay0303hfsg.nl")


def _perspective_row(spell_sum: bool):
    """The ``clay*hfsg`` hull row ``((x/L)^2 - 35 x/L + (z/L)^2 - 14 z/L) * L
    + 270.25 y <= 0`` with ``L = 0.001 + 0.999 y`` built afresh at every use (as
    ``from_nl`` does), spelled as a binary ``+`` chain or as ``dm.sum``."""
    m = dm.Model("persp")
    x = m.continuous("x", lb=0, ub=50)
    z = m.continuous("z", lb=0, ub=50)
    y = m.continuous("y", lb=0, ub=1)

    def L():
        return dm.sum([0.001, 0.999 * y]) if spell_sum else 0.001 + 0.999 * y

    body = ((x / L()) ** 2 - (35 * x) / L() + (z / L()) ** 2 - (14 * z) / L()) * L() + 270.25 * y
    m.subject_to(body <= 0)
    m.minimize(x + z)
    if spell_sum:
        assert isinstance(L(), SumOverExpression)
    return m


@pytest.mark.parametrize("spell_sum", [False, True], ids=["binary_chain", "dm_sum"])
def test_perspective_row_is_convex_however_the_sum_is_spelled(spell_sum):
    m = _perspective_row(spell_sum)
    assert classify_constraint(m._constraints[0], m) is True


def test_sumover_structural_equality_and_hash():
    m = dm.Model("eq")
    y = m.continuous("y", lb=0, ub=1)
    w = m.continuous("w", lb=0, ub=1)
    a = dm.sum([0.001, 0.999 * y])
    b = dm.sum([0.001, 0.999 * y])
    c = dm.sum([0.999 * y, 0.001])  # same sum, other order: declines, never mis-fires
    d = dm.sum([0.001, 0.999 * w])
    assert a is not b
    assert _expr_struct_eq(a, b)
    assert _struct_hash(a, {}) == _struct_hash(b, {})
    assert not _expr_struct_eq(a, c)
    assert not _expr_struct_eq(a, d)
    assert not _expr_struct_eq(a, 0.001 + 0.999 * y)  # different node type


def test_sumover_does_not_prove_a_nonconvex_row_convex():
    """Soundness side: a perspective-shaped row with a NEGATIVE square weight is
    not convex and must stay unproven with the n-ary spelling too."""
    m = dm.Model("neg")
    x = m.continuous("x", lb=0, ub=50)
    y = m.continuous("y", lb=0, ub=1)

    def L():
        return dm.sum([0.001, 0.999 * y])

    m.subject_to((-((x / L()) ** 2) + x / L()) * L() <= 0)
    m.minimize(x)
    assert classify_constraint(m._constraints[0], m) is False


def test_recentred_clay_keeps_every_convex_row():
    """The recentred ``clay0303hfsg`` (1e3 shift) proves exactly the rows the
    model as written proves. Before the fix: 150 vs 114."""
    as_written = dm.from_nl(CLAY)
    _, mask_a = classify_model(as_written)
    assert sum(mask_a) == len(mask_a) == 150
    folded = _recentre._build(dm.from_nl(CLAY), {}).model  # fold only, nothing moved
    rc = _recentre.recentre(translate(dm.from_nl(CLAY), 1e3, seed=1))
    assert rc is not None and len(rc.shifts) > 0
    for m in (folded, rc.model):
        _, mask = classify_model(m)
        assert mask == mask_a


@pytest.mark.slow
def test_recentred_clay_certifies(monkeypatch):
    """The panel cell itself: clay0303hfsg under x = y - c (c ~ 1e3) with
    ``DISCOPT_RECENTRE=1`` certifies the reference optimum. Before the fix it
    timed out uncertified (bound 0.0)."""
    monkeypatch.setenv("DISCOPT_RECENTRE", "1")
    r = translate(dm.from_nl(CLAY), 1e3, seed=1).solve(time_limit=120)
    assert (r.solver_stats or {}).get("recentre/variables_moved", 0) > 0
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(26669.1095724859, rel=1e-6)
    assert r.bound <= 26669.1095724859 + 1e-6 * 26669.1095724859


# --------------------------------------------------------------------------- #
# PR #1594 review: structural equality must be EXACT. ``np.allclose`` (atol 1e-8)
# in ``_expr_struct_eq`` and ``round(v, 12)`` in ``_struct_hash`` equated
# different constants, so a recogniser matched ``L1`` against ``L2 != L1`` and the
# verdict cache reused one row's CONVEX verdict for another row.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("spell_sum", [False, True], ids=["binary_chain", "dm_sum"])
def test_perspective_with_different_tiny_constants_is_not_convex(spell_sum):
    """``((x / (1e-9 + y))**2) * (2e-9 + y)`` is not a perspective: with ``t = y +
    1e-9`` it is ``x^2 g(t)``, ``g = (t + 1e-9)/t^2``, and ``g g'' - 2 g'^2 =
    -2e-18/t^6 < 0``, so it is nonconvex. ``allclose`` called the two denominators
    equal."""
    m = dm.Model("persp_tiny")
    x = m.continuous("x", lb=-1, ub=1)
    y = m.continuous("y", lb=0, ub=1)

    def L(c):
        return dm.sum([c, y]) if spell_sum else c + y

    m.subject_to(((x / L(1e-9)) ** 2) * L(2e-9) - 5 <= 0)
    m.minimize(x)
    assert classify_constraint(m._constraints[0], m) is not True


def test_struct_eq_and_hash_are_exact_on_constants():
    m = dm.Model("c")
    x = m.continuous("x", lb=-1, ub=1)
    a, b = 4e-13 * x**2, -4e-13 * x**2
    assert not _expr_struct_eq(a, b)
    assert _struct_hash(a, {}) != _struct_hash(b, {})
    assert not _expr_struct_eq(dm.sum([1e-9, x]), dm.sum([2e-9, x]))
    # Bit-identical constants still match (``from_nl`` rebuilds them that way).
    assert _expr_struct_eq(dm.sum([1e-9, x]), dm.sum([1e-9, x]))


def test_norm_of_two_near_equal_affine_maps_is_not_convex():
    """``sqrt(sum((A x) * (B x)))`` is the norm ``||A x||`` only when ``B == A``.
    ``B = A + [0, 1e-6]`` passed ``allclose`` (rtol 1e-5), but ``(x1 + x2)(x1 +
    (1 + 1e-6) x2)`` has Hessian determinant ``-1e-12``: indefinite."""
    A = np.array([[1.0, 1.0]])
    for B, want in [(np.array([[1.0, 1.0 + 1e-6]]), False), (A.copy(), True)]:
        m = dm.Model("norm")
        x = m.continuous("x", shape=(2,), lb=-1, ub=1)
        m.subject_to(dm.sqrt(dm.sum((A @ x) * (B @ x))) <= 1)
        m.minimize(x[0])
        assert classify_constraint(m._constraints[0], m) is want


@pytest.mark.parametrize("convex_row_first", [True, False])
def test_verdict_cache_does_not_reuse_across_sign_flipped_constants(convex_row_first):
    """End-to-end witness: ``4e-13 x^2 - z <= 1`` (convex) and ``-4e-13 x^2 + y <=
    1`` (nonconvex feasible set) hashed and compared equal, so whichever row came
    first fixed both verdicts. max ``y - 1e-8 x`` on x in [-1e6, 2e6]: the global
    optimum is 2.58 at x = 2e6; the false certificate was 1.41 (x = -1e6). A
    certificate, if issued, must be the true optimum."""
    m = dm.Model("e2e")
    x = m.continuous("x", lb=-1e6, ub=2e6)
    y = m.continuous("y", lb=-10, ub=10)
    z = m.continuous("z", lb=0, ub=10)
    r_cvx = 4e-13 * x**2 - z <= 1
    r_ccv = -4e-13 * x**2 + y <= 1
    for r in [r_cvx, r_ccv] if convex_row_first else [r_ccv, r_cvx]:
        m.subject_to(r)
    m.maximize(y - 1e-8 * x)
    res = m.solve(time_limit=30)
    if res.gap_certified:
        assert res.objective == pytest.approx(2.58, abs=1e-4)
        assert res.bound >= 2.58 - 1e-4  # max sense: a valid bound is >= the optimum
