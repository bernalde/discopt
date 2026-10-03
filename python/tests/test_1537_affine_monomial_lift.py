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


# ------------------------------------------- #1588 review: grouping, reach, heuristics


def _auxes_with(monkeypatch, flag, build):
    monkeypatch.setenv("DISCOPT_LIFT_AFFINE_MONOMIALS", flag)
    m = build()
    return len(fr.factorable_reformulate(m)._variables) - len(m._variables)


@pytest.mark.parametrize(
    "first, last",
    [
        # a repeated variable excludes the product however it is parenthesised
        (
            lambda y, z, u, v: (y - 4) * (y - 1) * (z - 2) * (u - 3),
            lambda y, z, u, v: (y - 1) * (z - 2) * (u - 3) * (y - 4),
        ),
        # so does a multivariate factor
        (
            lambda y, z, u, v: (u + v) * (y - 1) * (z - 2) * (u - 3),
            lambda y, z, u, v: (y - 1) * (z - 2) * (u - 3) * (u + v),
        ),
    ],
)
def test_rule_is_tested_on_the_maximal_chain_only(monkeypatch, first, last):
    """Fails before the fix: the excluded product written with the offending factor
    LAST lifted three factors (a sub-chain passed), written FIRST it lifted none."""
    counts = {}
    for label, f in (("first", first), ("last", last)):

        def build(f=f):
            m = dm.Model(label)
            y, z, u, v = (m.continuous(n, lb=0, ub=5) for n in "yzuv")
            m.subject_to(y + z <= 9)
            m.minimize(f(y, z, u, v))
            return m

        counts[label] = (_auxes_with(monkeypatch, "1", build), _auxes_with(monkeypatch, "0", build))
    # Same answer for both orderings, and the lift adds nothing to the =0 reform.
    assert counts["first"] == counts["last"], counts
    assert counts["last"][0] == counts["last"][1], counts


@pytest.mark.parametrize("where", ["objective", "row", "sum"])
def test_a_translated_monomial_alone_is_factorable_work(monkeypatch, where):
    """Fails before the fix: with no OTHER lift in the model the pass was gated off
    and ``min (x-3)(y-1)(z-2)(u-4)`` came back unchanged (and a monomial inside
    ``dm.sum`` was never reached)."""
    monkeypatch.setenv("DISCOPT_LIFT_AFFINE_MONOMIALS", "1")
    m = dm.Model(where)
    x, y, z, u = (m.continuous(n, lb=0, ub=5) for n in "xyzu")
    mono = (x - 3) * (y - 1) * (z - 2) * (u - 4)
    if where == "objective":
        m.minimize(mono)
    elif where == "row":
        m.subject_to(mono <= 1)
        m.minimize(x + y)
    else:
        m.minimize(dm.sum([mono, x]))
    assert fr.has_factorable_work(m)
    out = fr.factorable_reformulate(m)
    assert len(out._variables) - len(m._variables) >= 4
    # and the opt-out leaves the model alone, as before
    monkeypatch.setenv("DISCOPT_LIFT_AFFINE_MONOMIALS", "0")
    assert fr.factorable_reformulate(m) is m


def test_reach_lift_is_exact_and_solves_to_the_true_optimum(monkeypatch):
    """The lifted form is the same problem: min (x-3)(y-1)(z-2)(u-4) on [0,5]^4 has
    its minimum at a box vertex; enumerate the 16 vertices for the truth."""
    monkeypatch.setenv("DISCOPT_LIFT_AFFINE_MONOMIALS", "1")
    m = dm.Model("reach")
    x, y, z, u = (m.continuous(n, lb=0, ub=5) for n in "xyzu")
    m.minimize((x - 3) * (y - 1) * (z - 2) * (u - 4))
    truth = min(
        (a - 3) * (b - 1) * (c - 2) * (d - 4) for a, b, c, d in itertools.product((0, 5), repeat=4)
    )
    r = m.solve(time_limit=60)
    assert r.gap_certified, (r.status, r.objective, r.bound)
    assert r.objective == pytest.approx(truth, abs=1e-5)
    assert r.bound <= truth + 1e-6


def test_implied_integer_auxes_are_recorded_and_left_to_the_repair(monkeypatch):
    """Fails before the fix: ``subnlp`` pinned each lifted INTEGER aux at its
    (stale) relaxation value, so ``w == x - 3`` was violated whatever the NLP did
    and no point came back. The aux is integral because ``x`` is; the heuristics
    now leave it to the continuous repair."""
    from discopt._relax.primal_heuristics import _get_integer_mask, subnlp

    monkeypatch.setenv("DISCOPT_LIFT_AFFINE_MONOMIALS", "1")
    m = dm.Model("imp")
    x, y, z = (m.integer(n, lb=0, ub=4) for n in "xyz")
    m.subject_to(x + y + z >= 4)
    m.minimize((x - 3) * (y - 1) * (z - 2) + x + y + z)
    out = fr.factorable_reformulate(m)
    implied = out._implied_integer_auxes
    assert len(implied) == 3, implied
    names = [v.name for v in out._variables]
    mask = _get_integer_mask(out)
    assert [names[k] for k in np.flatnonzero(mask)] == ["x", "y", "z"]
    # originals at an integer point, every aux deliberately inconsistent
    x0 = np.array([1.0, 2.0, 3.0] + [0.4] * (len(names) - 3))
    res = subnlp(out, x0)
    assert res is not None
    pt, _obj = res
    vals = dict(zip(names, pt))
    got = sorted(round(vals[n], 6) for n in implied)
    assert got == sorted([1.0 - 3.0, 2.0 - 1.0, 3.0 - 2.0]), vals


def _nvs22() -> dm.Model:
    """nvs22 (MINLPLib) written out, as in ``test_obbt_cascade_varmap``."""
    m = dm.Model("nvs22")
    i1 = m.integer("i1", lb=1, ub=200)
    i2 = m.integer("i2", lb=1, ub=200)
    i3 = m.integer("i3", lb=1, ub=20)
    i4 = m.integer("i4", lb=1, ub=20)
    x5 = m.continuous("x5", lb=-1e7, ub=1e7)
    x6 = m.continuous("x6", lb=-1e7, ub=1e7)
    x7 = m.continuous("x7", lb=-1e7, ub=1e7)
    x8 = m.continuous("x8", lb=-1e7, ub=1e7)
    m.subject_to(-4243.28147100424 / (i3 * i4) + x5 == 0)
    m.subject_to(-dm.sqrt(0.25 * i4**2 + (0.5 * i1 + 0.5 * i3) ** 2) + x7 == 0)
    m.subject_to(
        -(59405.9405940594 + 2121.64073550212 * i4)
        * x7
        / (i3 * i4 * (0.0833333333333333 * i4**2 + (0.5 * i1 + 0.5 * i3) ** 2))
        + x6
        == 0
    )
    m.subject_to(-0.5 * i4 / x7 + x8 == 0)
    m.subject_to(-dm.sqrt(x5**2 + 2 * x5 * x6 * x8 + x6**2) >= -13600)
    m.subject_to(-504000 / (i1**2 * i2) >= -30000)
    m.subject_to(i2 - i3 >= 0)
    m.subject_to(
        0.0204744897959184 * dm.sqrt(1e13 * i2**3 * i1 * i1 * i2**3) * (1 - 0.0282346219657891 * i1)
        >= 6000
    )
    m.subject_to(-0.25 + 2.1952 / (i1**3 * i2) <= 0)
    m.minimize(1.10471 * i3**2 * i4 + 0.04811 * i1 * i2 * (14 + i4))
    return m


@pytest.mark.slow
def test_translated_nvs22_keeps_its_incumbent():
    """#1588 review: nvs22 under the 1e3 translation, lift ON (the default), lost
    its incumbent -- the three INTEGER auxes the lift adds broke integer local
    search. Fails before the fix: ``time_limit`` with no point."""
    import sys

    sys.path.insert(0, HERE)
    from _invariance import translate
    from discopt.validation.feasibility import verify_point

    m = translate(_nvs22(), 1e3)
    r = m.solve(time_limit=20)
    assert r.objective is not None, (r.status, r.node_count, r.bound)
    flat = np.concatenate([np.ravel(np.asarray(r.x[v.name], float)) for v in m._variables])
    assert verify_point(m, flat).ok
    assert r.objective >= 6.05822 - 1e-4  # minlplib.solu: never below the optimum
