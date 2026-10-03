"""Closed lower domain edge envelopes for asin/acos/acosh (issue #1510, item 3),
extended to xlogx/entropy and sqrt (#1242).

``uniform_relax._UNIVARIATE_FN``'s domain guard admitted ``hi = +1`` for
``asin``/``acos`` but demanded ``lo > -1`` (and ``lo > 1`` for ``acosh``), so a box
whose argument starts exactly at the closed lower edge -- the box holding an edge
optimum (#1492) -- got no envelope at all, only the aux column's interval floor.
``DISCOPT_CLOSED_EDGE_ENVELOPES`` admits the edge (an outward-rounded ``lo``
within the argument's rounding scale -- at least 1e-12 -- below it reads as the
edge); it graduated default-ON, and ``=0``
keeps the legacy interval floor. This file locks the CLAUDE.md section-5
properties for that bound-changing gate:

1. **Unset is ON**, ``=0`` is the legacy path, and the flag is a no-op on a box
   that does not touch an edge (or reaches genuinely outside the domain).
2. **Soundness -- no feasible point is cut**: every row AND column bound of the
   flag-ON relaxation holds at the exact lifted graph point ``(t, f(t))``,
   densely sampled over the box including the edge itself.
3. **Differential bound**: over a sweep of objective directions, flag-ON bound
   ``>=`` flag-OFF bound AND ``<=`` the true optimum over the same fixed box, with
   at least one direction strictly improved (the gate is not a no-op).
"""

from __future__ import annotations

import os

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt._relax.uniform_relax import build_uniform_relaxation
from scipy.optimize import linprog

pytestmark = [pytest.mark.relaxation]

FLAG = "DISCOPT_CLOSED_EDGE_ENVELOPES"


def _xlogx_np(t):
    """``t ln t`` with its continuous extension ``0`` at ``t = 0`` (nan below)."""
    t = np.asarray(t, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(t == 0.0, 0.0, t * np.log(t))


#: The lower bound interval arithmetic gives ``1 - y`` over ``y in [0, 1]``.
_AFFINE_ROUNDED_ZERO = 1.0 + np.nextafter(-1.0, -2.0)

#: ``(label, atom, numpy f, lo, hi)`` -- boxes that START at a closed lower edge.
EDGE_CASES = [
    ("asin[-1,0]", dm.asin, np.arcsin, -1.0, 0.0),
    ("asin[-1,-0.5]", dm.asin, np.arcsin, -1.0, -0.5),
    ("asin[-1,-1+1e-6]", dm.asin, np.arcsin, -1.0, -1.0 + 1e-6),
    ("acos[-1,0]", dm.acos, np.arccos, -1.0, 0.0),
    ("acosh[1,2]", dm.acosh, np.arccosh, 1.0, 2.0),
    ("acosh[1,50]", dm.acosh, np.arccosh, 1.0, 50.0),
    ("acosh[1,1+1e-6]", dm.acosh, np.arccosh, 1.0, 1.0 + 1e-6),
    # Outward rounding puts an affine argument's lower bound one ulp BELOW the
    # edge (``asin(x - 1)`` over ``x in [0, 1]``); read as the edge.
    ("asin[-1-ulp,0]", dm.asin, np.arcsin, np.nextafter(-1.0, -2.0), 0.0),
    ("acosh[1-ulp,2]", dm.acosh, np.arccosh, np.nextafter(1.0, 0.0), 2.0),
    # #1242: ``xlogx`` (the ``entropy`` atom) is finite at its closed edge 0
    # (``0 ln 0 = 0``) but ``dom_ok`` demanded ``lo > 0``; and ``1 - y`` over
    # ``y in [0, 1]`` arrives as ``lo = -2.2e-16`` (``-1 * [0, 1]`` is rounded
    # outward), which also defeated ``sqrt``'s ``lo >= 0``.
    ("xlogx[0,1]", dm.xlogx, _xlogx_np, 0.0, 1.0),
    ("xlogx[0,0.2]", dm.xlogx, _xlogx_np, 0.0, 0.2),
    ("xlogx[0,1e-6]", dm.xlogx, _xlogx_np, 0.0, 1e-6),
    ("xlogx[-2.2e-16,1]", dm.xlogx, _xlogx_np, _AFFINE_ROUNDED_ZERO, 1.0),
    ("xlogx[-ulp,3]", dm.xlogx, _xlogx_np, np.nextafter(0.0, -1.0), 3.0),
    ("sqrt[-2.2e-16,1]", dm.sqrt, np.sqrt, _AFFINE_ROUNDED_ZERO, 1.0),
    ("sqrt[-ulp,4]", dm.sqrt, np.sqrt, np.nextafter(0.0, -1.0), 4.0),
    # #1242 review: the rounding error scales with the argument. ``K - K*y`` at
    # ``K = 1e4`` arrives as ``lo = -1.8e-12`` (``-1.2e-10`` at ``1e6``), past an
    # absolute 1e-12, so the tolerance is the argument's own rounding scale.
    ("xlogx[-1.8e-12,1e4]", dm.xlogx, _xlogx_np, -1.8e-12, 1e4),
    ("sqrt[-1.2e-10,1e6]", dm.sqrt, np.sqrt, -1.2e-10, 1e6),
    ("acosh[1-1.2e-10,1e6]", dm.acosh, np.arccosh, 1.0 - 1.2e-10, 1e6),
]

#: Boxes where the flag must change nothing: interior boxes, the already-admitted
#: upper edge, a box reaching genuinely OUTSIDE the domain (beyond rounding), and
#: an edge box straddling the inflection at 0 (no single curvature -> no envelope
#: either way).
NOOP_CASES = [
    ("asin[-1,1]", dm.asin, np.arcsin, -1.0, 1.0),
    ("acos[-1,0.7]", dm.acos, np.arccos, -1.0, 0.7),
    ("asin[-0.9,0.5]", dm.asin, np.arcsin, -0.9, 0.5),
    ("asin[0.2,1]", dm.asin, np.arcsin, 0.2, 1.0),
    ("asin[-1.001,0]", dm.asin, np.arcsin, -1.001, 0.0),
    ("acos[-0.5,1]", dm.acos, np.arccos, -0.5, 1.0),
    ("acosh[1.5,3]", dm.acosh, np.arccosh, 1.5, 3.0),
    ("acosh[0.999,2]", dm.acosh, np.arccosh, 0.999, 2.0),
    # sqrt's exact edge was already admitted; a box genuinely below 0 keeps the floor.
    ("sqrt[0,1]", dm.sqrt, np.sqrt, 0.0, 1.0),
    ("sqrt[-1e-9,1]", dm.sqrt, np.sqrt, -1e-9, 1.0),
    ("xlogx[-1e-9,1]", dm.xlogx, _xlogx_np, -1e-9, 1.0),
    ("xlogx[0.1,2]", dm.xlogx, _xlogx_np, 0.1, 2.0),
    # ...but a rounding-SCALED tolerance is still a rounding tolerance: these sit
    # ~1000x further below the edge than evaluating the argument could round.
    ("xlogx[-1e-9,1e4]", dm.xlogx, _xlogx_np, -1e-9, 1e4),
    ("sqrt[-1e-6,1e6]", dm.sqrt, np.sqrt, -1e-6, 1e6),
    ("acosh[1-1e-6,1e6]", dm.acosh, np.arccosh, 1.0 - 1e-6, 1e6),
]


@pytest.fixture
def flag():
    prev = os.environ.get(FLAG)

    def _set(value):
        if value is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = value

    yield _set
    if prev is None:
        os.environ.pop(FLAG, None)
    else:
        os.environ[FLAG] = prev


def _atom_model(atom, lo, hi, obj=(0.0, -1.0)):
    """``y == atom(x)`` over ``x in [lo, hi]``, minimizing ``a*x + b*y``."""
    m = dm.Model()
    span = max(abs(lo), abs(hi), 1.0)
    x = m.continuous("x", lb=lo, ub=hi)
    y = m.continuous("y", lb=-1e3 * span, ub=1e3 * span)
    m.subject_to(y == atom(x))
    m.minimize(obj[0] * x + obj[1] * y)
    return m


def _rows(model):
    rel = build_uniform_relaxation(model)
    A = sp.csr_matrix(rel.model._A_ub, dtype=float)
    A.sort_indices()
    return rel, A.toarray(), np.asarray(rel.model._b_ub, dtype=float).ravel()


def _lp_bound(model):
    rel = build_uniform_relaxation(model)
    M = rel.model
    bnds = [
        (float(lo) if np.isfinite(lo) else None, float(hi) if np.isfinite(hi) else None)
        for lo, hi in np.asarray(M._bounds, dtype=float)
    ]
    res = linprog(
        np.asarray(M._c, dtype=float).ravel(),
        A_ub=sp.csr_matrix(M._A_ub),
        b_ub=np.asarray(M._b_ub, dtype=float).ravel(),
        bounds=bnds,
        method="highs",
    )
    assert res.status == 0, res.message
    return float(res.fun)


def _graph_point(rel, t, fval):
    specs = list(rel.univariate_atom_specs)
    assert len(specs) == 1, f"expected a single univariate atom, got {specs}"
    _fname, w, _var, _coeff, _cst = specs[0]
    z = np.zeros(len(rel.model._bounds), dtype=float)
    z[0] = t
    z[1] = fval
    z[int(w)] = fval
    return z


@pytest.mark.parametrize("label,atom,fnp,lo,hi", EDGE_CASES + NOOP_CASES)
def test_flag_unset_equals_flag_one(flag, label, atom, fnp, lo, hi):
    """Graduated: the default is the closed-edge envelope."""
    flag(None)
    _r0, a0, b0 = _rows(_atom_model(atom, lo, hi))
    flag("1")
    _r1, a1, b1 = _rows(_atom_model(atom, lo, hi))
    assert np.array_equal(a0, a1) and np.array_equal(b0, b1)


@pytest.mark.parametrize("label,atom,fnp,lo,hi", EDGE_CASES)
def test_opt_out_restores_the_interval_floor(flag, label, atom, fnp, lo, hi):
    """``=0`` is the legacy path: an edge box gets no envelope rows for the atom
    (the aux column keeps only its interval floor), i.e. fewer rows than the
    default."""
    flag("0")
    _r0, a0, _ = _rows(_atom_model(atom, lo, hi))
    flag(None)
    _r1, a1, _ = _rows(_atom_model(atom, lo, hi))
    assert a0.shape[0] < a1.shape[0], f"{label}: {a0.shape[0]} vs {a1.shape[0]} rows"


@pytest.mark.parametrize("label,atom,fnp,lo,hi", NOOP_CASES)
def test_flag_is_a_noop_off_the_edge(flag, label, atom, fnp, lo, hi):
    flag("0")
    _r0, a0, b0 = _rows(_atom_model(atom, lo, hi))
    flag("1")
    _r1, a1, b1 = _rows(_atom_model(atom, lo, hi))
    assert np.array_equal(a0, a1) and np.array_equal(b0, b1)


@pytest.mark.parametrize("label,atom,fnp,lo,hi", EDGE_CASES)
def test_edge_box_gains_envelope_rows(flag, label, atom, fnp, lo, hi):
    flag("0")
    _r0, a0, _ = _rows(_atom_model(atom, lo, hi))
    flag("1")
    _r1, a1, _ = _rows(_atom_model(atom, lo, hi))
    assert a1.shape[0] > a0.shape[0], f"{label}: {a0.shape[0]} -> {a1.shape[0]} rows"
    assert np.isfinite(a1).all(), f"{label}: non-finite coefficient"


@pytest.mark.parametrize("label,atom,fnp,lo,hi", EDGE_CASES)
def test_no_graph_point_is_cut(flag, label, atom, fnp, lo, hi):
    flag("1")
    rel = build_uniform_relaxation(_atom_model(atom, lo, hi))
    A = sp.csr_matrix(rel.model._A_ub, dtype=float)
    b = np.asarray(rel.model._b_ub, dtype=float).ravel()
    ts = [lo, hi, 0.5 * (lo + hi)] + list(np.linspace(lo, hi, 400))
    for k in range(1, 40):
        d = 0.5 * 2.0**-k
        ts += [lo + d * (hi - lo), hi - d * (hi - lo)]
    # The column bounds are part of the relaxation too: ``_pin_closed_edge_aux``
    # (#1242) narrows the aux column, so a graph point must also lie inside them.
    bnds = np.asarray(rel.model._bounds, dtype=float)
    checked = 0
    for t in ts:
        if not (lo <= t <= hi):
            continue
        with np.errstate(invalid="ignore"):
            fval = float(fnp(t))
        if not np.isfinite(fval):
            continue
        z = _graph_point(rel, t, fval)
        resid = A @ z - b
        viol = float(np.max(resid / np.maximum(1.0, np.abs(b))))
        assert viol <= 1e-9, f"{label}: graph point t={t!r} cut by {viol:.3e}"
        slack = 1e-9 * np.maximum(1.0, np.abs(z))
        assert np.all(z >= bnds[:, 0] - slack), f"{label}: t={t!r} below a column bound"
        assert np.all(z <= bnds[:, 1] + slack), f"{label}: t={t!r} above a column bound"
        checked += 1
    assert checked >= 400, f"{label}: only {checked} graph points evaluated"


@pytest.mark.parametrize("label,atom,fnp,lo,hi", EDGE_CASES)
def test_bound_tightens_and_never_crosses(flag, label, atom, fnp, lo, hi):
    """``bound_ON >= bound_OFF`` and ``bound_ON <= true box optimum`` over 32
    normalized objective directions; at least one direction strictly improves."""
    # The graph over the DOMAIN part of the box (an outward-rounded ``lo`` one ulp
    # outside it has no graph point there). ``round(lo, 6)`` is the edge itself
    # (every edge here is an integer, and every edge case's ``lo`` is within
    # rounding of it), which a grid from a ``lo`` that sits outside would otherwise
    # step over.
    with np.errstate(invalid="ignore"):
        grid = np.append(
            np.linspace(lo, hi, 200001), [np.nextafter(lo, hi), float(np.round(lo, 6))]
        )
        fv = fnp(grid)
    keep = np.isfinite(fv)
    grid, fv = grid[keep], fv[keep]
    assert grid.size >= 200001
    xs = max(hi - lo, 1e-300)
    fs = max(float(np.max(fv) - np.min(fv)), 1e-300)
    compared = improved = 0
    for k in range(32):
        th = 2.0 * np.pi * k / 32.0
        cx, cy = float(np.cos(th)) / xs, float(np.sin(th)) / fs
        flag("0")
        off = _lp_bound(_atom_model(atom, lo, hi, (cx, cy)))
        flag("1")
        on = _lp_bound(_atom_model(atom, lo, hi, (cx, cy)))
        true_min = float(np.min(cx * grid + cy * fv))
        assert on >= off - 1e-9 * (1.0 + abs(off)), f"{label} dir={k}: loosened {off} -> {on}"
        assert on <= true_min + 1e-6 * (1.0 + abs(true_min)), (
            f"{label} dir={k}: bound {on:.12g} crossed the true optimum {true_min:.12g}"
        )
        compared += 1
        improved += on > off + 1e-9 * (1.0 + abs(off))
    assert compared == 32
    assert improved > 0, f"{label}: the closed-edge envelope never moved the bound (no-op)"


def test_affine_argument_reaching_the_edge_is_enveloped(flag):
    """The class member the exact-edge test alone would miss: ``asin(x - 1)`` over
    ``x in [0, 1]`` reaches -1 only through outward-rounded interval arithmetic
    (``lo = -1 - ulp``). With the flag the edge box is enveloped and the bound of
    ``min asin(x - 1) + x/2`` (true optimum -pi/2 at x = 0) tightens without
    crossing it."""
    import math

    def model():
        m = dm.Model()
        x = m.continuous("x", lb=0.0, ub=1.0)
        y = m.continuous("y", lb=-10.0, ub=10.0)
        m.subject_to(y == dm.asin(x - 1.0))
        m.minimize(y + 0.5 * x)
        return m

    flag("0")
    _r0, a0, _ = _rows(model())
    off = _lp_bound(model())
    flag("1")
    _r1, a1, _ = _rows(model())
    on = _lp_bound(model())
    opt = -math.pi / 2
    assert a1.shape[0] > a0.shape[0]
    assert off - 1e-9 <= on <= opt + 1e-9, (off, on, opt)


@pytest.mark.parametrize(
    "label,atom,true_min",
    [("xlogx(1-y)", dm.xlogx, -1.0 / np.e), ("sqrt(1-y)", dm.sqrt, 0.0)],
)
def test_root_lp_of_an_affine_edge_argument_is_bounded(flag, label, atom, true_min):
    """#1242: ``min atom(1 - y)`` over ``y in [0, 1]``. The argument's interval is
    ``[-2.2e-16, 1 + 2.2e-16]``; without the edge the atom got no envelope AND an
    unbounded aux column, the root LP came back ``unbounded``, and the solver
    dropped the McCormick relaxer for the WHOLE tree (``_mc_mode = "none"``) --
    the entropy acceptance solve then converged only first-order and could not
    certify to 1e-9. With the edge the aux column is also pinned to the image of
    the clamped argument, so the safe LP bound certifies."""
    from discopt._relax.mccormick_lp import MccormickLPRelaxer

    def solve():
        m = dm.Model()
        y = m.continuous("y", lb=0.0, ub=1.0)
        m.minimize(atom(1 - y))
        return MccormickLPRelaxer(m).solve_at_node(np.array([0.0]), np.array([1.0]))

    flag("0")
    off = solve()
    flag(None)
    on = solve()
    assert off.status != "optimal" and off.lower_bound is None, (off.status, off.lower_bound)
    assert on.status == "optimal", on.status
    assert on.lower_bound is not None and np.isfinite(on.lower_bound)
    assert on.lower_bound <= true_min + 1e-12, (on.lower_bound, true_min)
    assert on.lower_bound >= true_min - 1e-9, "edge envelope bound is needlessly loose"


@pytest.mark.parametrize("K", [1e4, 1e6])
@pytest.mark.parametrize(
    "label,build,true_min",
    [
        ("xlogx(K-K*y)", lambda y, K: dm.xlogx(K - K * y), -1.0 / np.e),
        ("sqrt(K-K*y)", lambda y, K: dm.sqrt(K - K * y), 0.0),
        ("acosh(1+K-K*y)", lambda y, K: dm.acosh(1 + K - K * y), 0.0),
        ("asin(y-1) via K", lambda y, K: dm.asin((K * y - K) / K), -np.pi / 2),
    ],
)
def test_root_lp_of_a_scaled_edge_argument_is_bounded(flag, K, label, build, true_min):
    """#1242 review: outward rounding of ``K - K*y`` is ``O(K * eps)`` below the
    edge (``-1.8e-12`` at ``K = 1e4``, ``-1.2e-10`` at ``1e6``). An absolute 1e-12
    clamp tolerance declined it and the root LP came back unbounded again; the
    tolerance now scales with the argument's own rounding."""
    from discopt._relax.mccormick_lp import MccormickLPRelaxer

    flag(None)
    m = dm.Model()
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.minimize(build(y, K))
    on = MccormickLPRelaxer(m).solve_at_node(np.array([0.0]), np.array([1.0]))
    assert on.status == "optimal", (label, K, on.status)
    assert on.lower_bound is not None and np.isfinite(on.lower_bound)
    assert on.lower_bound <= true_min + 1e-12, (on.lower_bound, true_min)


@pytest.mark.slow
@pytest.mark.parametrize("K", [1e4, 1e6])
def test_scaled_binary_mixture_certifies(flag, K):
    """End to end: ``min xlogx(K y) + xlogx(K - K y) + 3K y (1 - y)`` certifies at
    ``K = 1e4`` and ``1e6`` with a bound at or below the true optimum (from a dense
    grid; the optimum is interior, so the grid minimum is within 1e-6 relative)."""
    flag(None)
    m = dm.Model()
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.minimize(dm.xlogx(K * y) + dm.xlogx(K - K * y) + 3 * K * y * (1 - y))
    res = m.solve(time_limit=120)
    g = np.linspace(0.0, 1.0, 4_000_001)
    opt = float(np.min(_xlogx_np(K * g) + _xlogx_np(K - K * g) + 3 * K * g * (1 - g)))
    assert res.status == "optimal", res.status
    assert res.bound is not None and np.isfinite(res.bound)
    assert res.bound <= opt + 1e-7 * (1 + abs(opt)), (res.bound, opt)
    assert abs(res.objective - opt) <= 1e-4 * (1 + abs(opt)), (res.objective, opt)
