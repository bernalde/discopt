"""Regression tests for issue #1490.

A pure-integer model whose only nonlinearity is a unary transcendental intrinsic
(``sin``, ``cos``, ``exp``, ...) was certified optimal at a non-optimal integer
point: ``min sin(x)``, ``x`` integer in ``[-8, 8]`` returned ``-0.909`` at
``x = -2`` with ``gap_certified=True`` (true optimum ``-0.989`` at ``x = -8``).

Root cause: in nonconvex mode the B&B tree fathoms a node whose relaxation
point is integral once every *registered* spatial dimension is tight. Integer
columns were registered for spatial domain-partition branching only on the
McCormick-LP route and only when they appeared in a product / monomial /
fractional-power term, so an integer inside an intrinsic (or any integer on the
alphaBB fallback route) was treated as "resolved" and its node's valid but
below-incumbent bound was dropped.

Every case is checked against exhaustive enumeration of the integer lattice.
"""

from __future__ import annotations

import itertools
import math
import random

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.sparsity import nonlinear_columns

_TOL = 1e-6


def _assert_matches_enumeration(r, true_opt, name):
    assert r.status == "optimal", (name, r.status)
    assert r.objective == pytest.approx(true_opt, abs=1e-6), (name, r.objective, true_opt)
    # A certified bound must never exceed the true optimum (min sense).
    if r.bound is not None:
        assert r.bound <= true_opt + 1e-6, (name, r.bound, true_opt)


def test_min_sin_integer_issue_repro():
    m = dm.Model("sin")
    x = m.integer("x", lb=-8, ub=8)
    m.minimize(dm.sin(x))
    r = m.solve(time_limit=30)
    true_opt = min(math.sin(k) for k in range(-8, 9))
    _assert_matches_enumeration(r, true_opt, "sin")
    assert float(np.asarray(r.x["x"])) == pytest.approx(-8.0)


def test_min_cos_integer():
    m = dm.Model("cos")
    x = m.integer("x", lb=-8, ub=8)
    m.minimize(dm.cos(x))
    r = m.solve(time_limit=30)
    true_opt = min(math.cos(k) for k in range(-8, 9))
    _assert_matches_enumeration(r, true_opt, "cos")


def test_sin_plus_linear_wide_domain():
    m = dm.Model("sinlin")
    x = m.integer("x", lb=-20, ub=20)
    m.minimize(dm.sin(x) + 0.01 * x)
    r = m.solve(time_limit=60)
    true_opt = min(math.sin(k) + 0.01 * k for k in range(-20, 21))
    _assert_matches_enumeration(r, true_opt, "sin+0.01x")


def test_two_variable_separable_sin():
    m = dm.Model("sin2")
    x = m.integer("x", lb=-8, ub=8)
    y = m.integer("y", lb=-8, ub=8)
    m.minimize(dm.sin(x) + dm.sin(y))
    r = m.solve(time_limit=60)
    true_opt = 2 * min(math.sin(k) for k in range(-8, 9))
    _assert_matches_enumeration(r, true_opt, "sin(x)+sin(y)")


def test_exp_sin_cos_two_variable():
    m = dm.Model("expsin")
    x = m.integer("x", lb=-8, ub=8)
    y = m.integer("y", lb=-8, ub=8)
    m.minimize(dm.exp(0.1 * x) * dm.sin(y) + dm.cos(x))
    r = m.solve(time_limit=60)
    true_opt = min(
        math.exp(0.1 * a) * math.sin(b) + math.cos(a)
        for a, b in itertools.product(range(-8, 9), repeat=2)
    )
    _assert_matches_enumeration(r, true_opt, "exp(0.1x)*sin(y)+cos(x)")


def test_mixed_integer_intrinsic_with_continuous():
    """The same defect on the McCormick-LP route: a continuous variable makes the
    model spatially branchable, but the integer inside ``cos`` was never
    registered, so the tree could still close a box over it."""
    m = dm.Model("mixed")
    x = m.integer("x", lb=-8, ub=8)
    y = m.continuous("y", lb=-1, ub=1)
    m.minimize(dm.cos(x) + (y - 0.3) ** 2)
    r = m.solve(time_limit=60)
    true_opt = min(math.cos(k) for k in range(-8, 9))
    _assert_matches_enumeration(r, true_opt, "cos(x)+(y-0.3)^2")


_INTRINSICS = {
    "sin": (dm.sin, math.sin),
    "cos": (dm.cos, math.cos),
    "atan": (dm.atan, math.atan),
    "tanh": (dm.tanh, math.tanh),
}


def test_fuzz_separable_unary_intrinsic_vs_enumeration():
    rng = random.Random(1490)
    n_checked = 0
    for case in range(12):
        nvar = rng.choice([1, 2])
        m = dm.Model(f"fz{case}")
        xs, spec = [], []
        for i in range(nvar):
            lo = rng.randint(-10, -2)
            hi = rng.randint(2, 10)
            xs.append(m.integer(f"x{i}", lb=lo, ub=hi))
            fname = rng.choice(sorted(_INTRINSICS))
            a = rng.choice([0.5, 1.0, 1.7, 2.3])
            c = rng.choice([-1.0, 1.0, 2.0])
            lin = rng.choice([0.0, 0.03, -0.05])
            spec.append((lo, hi, fname, a, c, lin))
        obj = 0
        for xv, (_lo, _hi, fname, a, c, lin) in zip(xs, spec):
            obj = obj + c * _INTRINSICS[fname][0](a * xv) + lin * xv
        m.minimize(obj)
        r = m.solve(time_limit=60)
        true_opt = sum(
            min(c * _INTRINSICS[f][1](a * k) + lin * k for k in range(lo, hi + 1))
            for (lo, hi, f, a, c, lin) in spec
        )
        _assert_matches_enumeration(r, true_opt, (case, spec))
        n_checked += 1
    # Prove the probe fired (CLAUDE.md §6).
    assert n_checked == 12


def test_nonlinear_columns_covers_intrinsics_and_skips_affine():
    m = dm.Model("cols")
    x = m.integer("x", lb=0, ub=3)
    y = m.integer("y", lb=0, ub=3)
    z = m.integer("z", lb=0, ub=3)
    w = m.integer("w", lb=0, ub=3)
    m.minimize(dm.sin(x) + 2 * y - z / 4)
    m.subject_to(-(z) + w * 3 <= 10)
    assert nonlinear_columns(m) == {0}
    m.subject_to(y / (w + 1) <= 5)
    assert nonlinear_columns(m) == {0, 1, 3}
