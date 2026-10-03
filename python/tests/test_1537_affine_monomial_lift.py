"""#1537 C: a multilinear monomial written in translated coordinates is lifted, not
multiplied out (``DISCOPT_LIFT_AFFINE_MONOMIALS``).

Witness (the entry experiment): nvs09 -- ``- (prod_k x_k)**0.2`` over ten integers
on [3, 9] -- certifies -43.1343 in 31 nodes as written, but under the exact change
of variables ``x = y - 3`` (box [6, 12]) or ``x = y + 3`` (box [0, 6]) the product
of ten ``y_k -/+ 3`` factors was distributed into 1,024 multilinear terms (exactly
AT ``_DISTRIBUTE_TERM_LIMIT``, so not lifted). Its McCormick LP exceeded the dense
cap and the solve fell back to interval/alphaBB bounds: -81.0 / -48.0 at the 30 s
limit, no certificate. Lifting each translated factor to an exact aux gives back
the monomial the unshifted model has, and both shifts certify in 39 nodes.
"""

from __future__ import annotations

import itertools
import os

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import factorable_reform as fr
from discopt.modeling.core import Constant, SumOverExpression

HERE = os.path.dirname(os.path.abspath(__file__))
NVS09 = os.path.join(HERE, "data", "minlplib_nl", "nvs09.nl")
NVS09_OPT = -43.1343369200  # minlplib.solu


def _factors(expr):
    return fr._collect_mul_factors(expr)


# ---------------------------------------------------------------- the rule itself


def test_rule_fires_on_a_translated_monomial_and_nowhere_else():
    m = dm.Model("r")
    x, y, z, u = (m.continuous(n, lb=0, ub=5) for n in "xyzu")
    checks = 0
    cases = [
        ((x - 3) * (y - 3) * (z - 3), True),  # the class
        (2.0 * (x + 1) * y * z, True),  # one offset factor is enough
        ((2 * x - 1) * (3 - y) * (0.5 * z + 7) * u, True),  # scaled, reversed
        ((x - 3) * (y - 3), False),  # bilinear: McCormick is already exact here
        (x * y * z, False),  # no offset: already a monomial
        ((x - 1) * (x - 2) * (y - 3), False),  # repeated variable: a polynomial
        ((x + y) * (z - 1) * (u - 1), False),  # a multivariate factor
        (dm.exp(x) * (y - 1) * (z - 1), False),  # a non-affine factor
    ]
    for expr, want in cases:
        assert fr._is_translated_monomial(_factors(expr)) is want, expr
        checks += 1
    # recentring writes ``y + c`` as a two-term SumOverExpression
    sx = SumOverExpression([x, Constant(3.0)])
    sy = SumOverExpression([y, Constant(-2.0)])
    assert fr._is_translated_monomial([sx, sy, z])
    checks += 1
    assert checks == 9


def _lifted_auxes(model) -> int:
    out = fr.factorable_reformulate(model)
    return len(out._variables) - len(model._variables)


def test_flag_off_leaves_the_product_distributed(monkeypatch):
    m = dm.Model("p")
    v = [m.integer(f"v{k}", lb=0, ub=6) for k in range(4)]
    m.minimize(-(((v[0] + 3) * (v[1] + 3) * (v[2] + 3) * (v[3] + 3)) ** 0.5))
    monkeypatch.setenv("DISCOPT_LIFT_AFFINE_MONOMIALS", "0")
    off = _lifted_auxes(m)
    monkeypatch.setenv("DISCOPT_LIFT_AFFINE_MONOMIALS", "1")
    on = _lifted_auxes(m)
    # ON lifts one aux per translated factor on top of whatever OFF lifts.
    assert on >= off + 4, (off, on)


def test_default_is_on_and_zero_opts_out(monkeypatch):
    monkeypatch.delenv("DISCOPT_LIFT_AFFINE_MONOMIALS", raising=False)
    assert fr._lift_affine_monomials_enabled() is True
    monkeypatch.setenv("DISCOPT_LIFT_AFFINE_MONOMIALS", "0")
    assert fr._lift_affine_monomials_enabled() is False


def test_lifted_factor_keeps_integrality(monkeypatch):
    monkeypatch.setenv("DISCOPT_LIFT_AFFINE_MONOMIALS", "1")
    m = dm.Model("p")
    v = [m.integer(f"v{k}", lb=0, ub=6) for k in range(3)]
    m.minimize(-(((v[0] + 3) * (v[1] + 3) * (v[2] + 3)) ** 0.5))
    out = fr.factorable_reformulate(m)
    new = out._variables[len(m._variables) :]
    # one INTEGER aux per ``v_k + 3`` factor; the product/power auxes are continuous
    assert sum(w.var_type.name == "INTEGER" for w in new) == 3, [(w.name, w.var_type) for w in new]


# ------------------------------------------------- end to end: certified and sound


def _shifted_nvs09(d: float):
    import sys

    sys.path.insert(0, HERE)
    from _invariance import _rebuild

    base = dm.from_nl(NVS09)
    return _rebuild(base, lambda v: np.full(v.lb.shape, -d), 1.0, f"nvs09_d{d:g}")


@pytest.mark.slow
@pytest.mark.parametrize("d", [3.0, -3.0])
def test_translated_nvs09_certifies_the_true_optimum(d):
    """Fails before the fix (and with ``DISCOPT_LIFT_AFFINE_MONOMIALS=0`` in the
    environment): d=+3 publishes bound -48.0, d=-3 bound -81.0, neither certified
    within 30 s. Runs on the DEFAULT, so it also pins the graduated default."""
    r = _shifted_nvs09(d).solve(time_limit=60)
    assert r.gap_certified, (r.status, r.objective, r.bound, r.node_count)
    assert r.objective == pytest.approx(NVS09_OPT, abs=1e-5)
    assert r.bound <= NVS09_OPT + 1e-6


def _brute(c, a, lo, hi, shift):
    best = np.inf
    for z in itertools.product(*(range(int(lo[k]), int(hi[k]) + 1) for k in range(len(lo)))):
        z = np.asarray(z, dtype=float)
        val = float(c @ z + a * np.prod(z - shift))
        best = min(best, val)
    return best


@pytest.mark.parametrize("seed", range(6))
def test_translated_integer_monomials_certify_the_enumerated_optimum(monkeypatch, seed):
    """Soundness: generated ``c.z + a * prod_k (z_k - s_k)`` over small integer
    boxes, certified answer against exhaustive enumeration, flag ON and OFF."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 5))
    lo = rng.integers(-3, 3, size=n).astype(float)
    hi = lo + rng.integers(2, 4, size=n)
    shift = rng.integers(-4, 5, size=n).astype(float)
    shift[shift == 0] = 1.0
    c = rng.integers(-5, 6, size=n).astype(float)
    a = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.2, 2.0))
    truth = _brute(c, a, lo, hi, shift)
    checks = 0
    for flag in ("0", "1"):
        monkeypatch.setenv("DISCOPT_LIFT_AFFINE_MONOMIALS", flag)
        m = dm.Model(f"g{seed}")
        z = [m.integer(f"z{k}", lb=lo[k], ub=hi[k]) for k in range(n)]
        prod = z[0] - shift[0]
        for k in range(1, n):
            prod = prod * (z[k] - shift[k])
        m.minimize(sum(c[k] * z[k] for k in range(n)) + a * prod)
        r = m.solve(time_limit=30)
        assert r.gap_certified, (flag, r.status)
        assert r.objective == pytest.approx(truth, abs=1e-6, rel=1e-6), (flag, truth)
        assert r.bound <= truth + 1e-6, (flag, r.bound, truth)
        checks += 1
    assert checks == 2
