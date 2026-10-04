"""#1610 C-14 remainder: ``DISCOPT_COEF_TIGHTEN`` on array variables, and implied fixings.

PR #1637 fixed C2 and the in-place-mutation half of C-14 (the solve now owns and
restores the caller's model). Two C-14 items were left:

* **Array variables.** ``tighten_bigm_coefficients`` returned 0 on any model with a
  ``size > 1`` block, so the bottling model written with ``shape=(T,)`` variables got
  root bound 1.775 while its scalar twin got 22.635.
* **Implied fixing.** ``x1 >= 2`` (demand 12, initial stock 10) and ``x1 <= 40·y1``
  imply ``y1 = 1``; root probing even derived it, but the bound was discarded, so the
  root bound stayed at 22.635 instead of 27.625.

Bound-changing (CLAUDE.md §5): besides the end-to-end numbers, the per-corner exact
LP check below proves the rewritten model (rows + fixings) has the same
integer-feasible set as the original, in both directions.
"""

from __future__ import annotations

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.problem_classifier import (
    _extract_linear_coefficients_sparse,
    _NotLinearError,
)
from discopt.export._arrays import scalarize_body
from discopt.modeling.core import VarType
from discopt.solvers._root_presolve import tighten_bigm_coefficients

D = [12, 18, 0, 25, 30, 8, 0, 22, 35, 15.0]
C = [40, 40, 40, 20, 40, 40, 40, 40, 40, 40.0]
T = 10
OPT = 38.2
ROOT_WITH_FIXING = 27.625  # LP bound with tightened rows and y1 fixed to 1


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv("DISCOPT_COEF_TIGHTEN", "1")


def _bottling(kind: str) -> dm.Model:
    """The issue's lot-sizing model: scalar vars, indexed array vars, or vector rows."""
    m = dm.Model(f"bottling_{kind}")
    if kind == "scalar":
        x = [m.continuous(f"x{t}", lb=0, ub=1000) for t in range(1, T + 1)]
        s = [m.continuous(f"s{t}", lb=0, ub=1000) for t in range(T + 1)]
        y = [m.binary(f"y{t}") for t in range(1, T + 1)]
    else:
        x = m.continuous("x", shape=(T,), lb=0, ub=1000)
        s = m.continuous("s", shape=(T + 1,), lb=0, ub=1000)
        y = m.binary("y", shape=(T,))
    m.minimize(dm.sum(lambda t: 5 * y[t] + 0.2 * s[t + 1], over=range(T)))
    m.subject_to(s[0] == 10)
    for t in range(T):
        m.subject_to(s[t] + x[t] - s[t + 1] == D[t])
    if kind == "vector":
        m.subject_to(x <= np.array(C))
        m.subject_to(x <= 1000 * y, name="co")
    else:
        for t in range(T):
            m.subject_to(x[t] <= C[t])
            m.subject_to(x[t] <= 1000 * y[t], name=f"co{t + 1}")
    return m


def _y1(m: dm.Model):
    for v in m._variables:
        if v.name == "y1":
            return float(v.lb), float(v.ub)
        if v.name == "y":
            return float(v.lb.ravel()[0]), float(v.ub.ravel()[0])
    raise AssertionError("no y variable")


def _flat(m: dm.Model):
    """``(n, lb, ub, discrete)`` over flat scalar slots, read off the variables."""
    vs = m._variables
    lb = np.concatenate([np.asarray(v.lb, float).ravel() for v in vs])
    ub = np.concatenate([np.asarray(v.ub, float).ravel() for v in vs])
    disc = np.concatenate(
        [np.full(v.size, v.var_type in (VarType.BINARY, VarType.INTEGER)) for v in vs]
    )
    return lb.size, lb, ub, disc


def _rows(m: dm.Model, n: int):
    """Every linear scalar row as ``(terms, const, sense, from_vector_body)``."""
    out = []
    for con in m._constraints:
        try:
            terms, const = _extract_linear_coefficients_sparse(con.body, m, n)
            out.append((dict(terms), float(const), con.sense, False))
            continue
        except _NotLinearError:
            pass
        for e in scalarize_body(con.body):
            terms, const = _extract_linear_coefficients_sparse(e, m, n)
            out.append((dict(terms), float(const), con.sense, True))
    return out


def _bigm_coeffs(m: dm.Model) -> list[float]:
    """|y_t coefficient| in each scalar ``x_t - M·y_t <= 0`` row, ordered by t."""
    n, _, _, disc = _flat(m)
    out = {}
    for terms, _c, sense, vec in _rows(m, n):
        if vec or sense != "<=" or len(terms) != 2:
            continue
        (j1, _), (j2, _) = sorted(terms.items())
        xj, yj = (j1, j2) if disc[j2] else (j2, j1)
        if disc[xj] or not disc[yj]:
            continue
        a_x, a_y = terms[xj], terms[yj]
        if a_x == 1.0 and a_y < 0:
            out[yj] = -a_y
    return [out[k] for k in sorted(out)]


# ── array variables ──────────────────────────────────────────────────────────


@pytest.mark.smoke
def test_array_model_is_tightened_like_its_scalar_twin(flag_on):
    ms, ma = _bottling("scalar"), _bottling("array")
    n_s = tighten_bigm_coefficients(ms)
    n_a = tighten_bigm_coefficients(ma)
    assert n_s > 0
    assert n_a == n_s, f"array model tightened {n_a} rows, scalar twin {n_s}"
    # Per-element bounds: co4 sees x4 <= 20 (block hull would say 40).
    cs, ca = _bigm_coeffs(ms), _bigm_coeffs(ma)
    assert len(cs) == len(ca) == T
    assert cs == pytest.approx(ca)
    assert ca[0] == 1000.0  # y1 is fixed to 1, so co1 is left as written
    assert ca[1:] == pytest.approx(C[1:])  # incl. co4 at 20, below the block hull 40


def test_vector_valued_rows_are_tightened_per_element(flag_on):
    m = _bottling("vector")
    n_rows = len(m._constraints)
    n = tighten_bigm_coefficients(m)
    assert n == T - 1
    appended = m._constraints[n_rows:]
    assert len(appended) == T - 1, "one appended scalar row per tightened element"
    assert all(c.rhs == 0.0 for c in appended)
    # The original vector row is kept, so no constraint index moves.
    assert m._constraints[n_rows - 1].name == "co"
    coeffs = _bigm_coeffs(m)
    assert coeffs == pytest.approx(C[1:])


def test_two_dimensional_block_slot_expressions(flag_on):
    m = dm.Model("grid")
    x = m.continuous("x", shape=(2, 3), lb=0, ub=500)
    y = m.binary("y", shape=(2, 3))
    cap = np.array([[5.0, 6, 7], [8, 9, 10]])
    m.subject_to(x <= cap)
    m.subject_to(x <= 500 * y)
    m.maximize(dm.sum(x) - dm.sum(y))
    n0 = len(m._constraints)
    assert tighten_bigm_coefficients(m) == 6
    got = sorted(_bigm_coeffs(m))
    assert got == pytest.approx(sorted(cap.ravel()))
    assert len(m._constraints) == n0 + 6


# ── implied fixing ───────────────────────────────────────────────────────────


@pytest.mark.smoke
@pytest.mark.parametrize("kind", ["scalar", "array", "vector"])
def test_implied_fixing_reaches_the_tree(flag_on, kind):
    m = _bottling(kind)
    r = m.solve()
    assert r.status == "optimal"
    assert r.objective == pytest.approx(OPT, abs=1e-6)
    # Certificate invariant, and the fixing actually lifted the root bound.
    assert r.root_bound <= r.objective + 1e-6
    assert r.root_bound == pytest.approx(ROOT_WITH_FIXING, abs=1e-5)
    # The fixing is solve-local: the caller's declared bound is untouched (#1610 C2).
    assert _y1(m) == (0.0, 1.0)


def test_implied_fixing_is_written_to_the_bounds(flag_on):
    m = _bottling("array")
    tighten_bigm_coefficients(m)
    assert _y1(m) == (1.0, 1.0)


# ── soundness: exact per-corner feasible-set equivalence (§5) ────────────────


def _rows_dense(m, n):
    out = []
    for terms, const, sense, _vec in _rows(m, n):
        a = np.zeros(n)
        for j, c in terms.items():
            a[j] = c
        out.append((a, const, sense))
    return out


def _lp_max(obj, const, rows, bounds):
    from scipy.optimize import linprog

    a_ub, b_ub, a_eq, b_eq = [], [], [], []
    for a, c, s in rows:
        if s == "<=":
            a_ub.append(a)
            b_ub.append(-c)
        elif s == ">=":
            a_ub.append(-a)
            b_ub.append(c)
        else:
            a_eq.append(a)
            b_eq.append(-c)
    res = linprog(
        -obj,
        A_ub=np.array(a_ub) if a_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=np.array(a_eq) if a_eq else None,
        b_eq=np.array(b_eq) if b_eq else None,
        bounds=bounds,
        method="highs",
    )
    if res.status == 2:
        return None
    assert res.success, res.message
    return float(obj @ res.x) + const


def _small_lot_sizing():
    m = dm.Model("small")
    d, cap = [12.0, 5.0, 9.0], [30.0, 6.0, 30.0]
    x = m.continuous("x", shape=(3,), lb=0, ub=100)
    s = m.continuous("s", shape=(4,), lb=0, ub=100)
    y = m.binary("y", shape=(3,))
    m.subject_to(s[0] == 10)
    for t in range(3):
        m.subject_to(s[t] + x[t] - s[t + 1] == d[t])
    m.subject_to(x <= np.array(cap))
    m.subject_to(x <= 100 * y)
    m.subject_to(30 * y[0] + x[1] + x[2] <= 40)  # Savelsbergh (a_k > 0) case
    m.minimize(dm.sum(y) + 0.1 * dm.sum(s))
    return m


def test_rewrite_and_fixings_preserve_integer_feasible_set(flag_on):
    orig, tight = _small_lot_sizing(), _small_lot_sizing()
    assert tighten_bigm_coefficients(tight) > 0
    n, lb_o, ub_o, disc = _flat(orig)
    _, lb_t, ub_t, _ = _flat(tight)
    rows_o, rows_t = _rows_dense(orig, n), _rows_dense(tight, n)
    bins = [j for j in range(n) if disc[j]]
    checked = 0
    for corner in itertools.product((0.0, 1.0), repeat=len(bins)):
        b_o = [(lb_o[j], ub_o[j]) for j in range(n)]
        b_t = [(lb_t[j], ub_t[j]) for j in range(n)]
        for j, v in zip(bins, corner):
            b_o[j] = (v, v)
            b_t[j] = (v, v) if lb_t[j] <= v <= ub_t[j] else None
        zero = np.zeros(n)
        feas_o = _lp_max(zero, 0.0, rows_o, b_o) is not None
        feas_t = None not in b_t and _lp_max(zero, 0.0, rows_t, b_t) is not None
        assert feas_o == feas_t, f"corner {corner}: original {feas_o}, tightened {feas_t}"
        if not feas_o:
            continue
        for src, dst, b_dst in ((rows_o, rows_t, b_t), (rows_t, rows_o, b_o)):
            for a, c, s in src:
                for sign in (1.0,) if s == "<=" else (-1.0,) if s == ">=" else (1.0, -1.0):
                    v = _lp_max(sign * a, sign * c, dst, b_dst)
                    assert v is not None and v <= 1e-6, f"corner {corner}: violation {v}"
                    checked += 1
    assert checked > 0, "probe compared nothing"
