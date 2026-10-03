"""Issue #1591: array quadratic forms (``x @ Q @ x`` and friends) get the LP relaxation.

Before the fix the term classifier saw ``x @ Q @ x`` as "a matmul, so linear" and
catalogued nothing; ``MccormickLPRelaxer.has_relaxable_nonlinearity`` was False, the
solver fell to the ``none`` relaxation mode and stopped at the root with
``status="feasible"`` and no certificate.  The Rust classifier had the same blind
spot for every array-valued product (``x * y``, ``x * (Q @ x)``, ``x ** 3`` over
vectors/matrices) and, next to any scalar product, returned an *incomplete* catalog
as if it were complete.

The relaxation engine itself already scalarizes these forms (#1502); the fix routes
them through the Python walk, which now expands each array-nonlinear node into its
scalar elements.  The tests pin, for every array quadratic spelling the modeling layer
produces:

* the catalog equals the one of the literal scalar transliteration, with no
  ``general_nl`` residue;
* the LP relaxation bound on random sub-boxes is sound (never above the sampled
  minimum of the true function) and equal to the scalar transliteration's bound;
* the issue repro now certifies optimality at the scalar spelling's objective.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import term_classifier as tc
from discopt._relax.mccormick_lp import MccormickLPRelaxer
from discopt._relax.model_utils import flat_variable_bounds

_RNG = np.random.default_rng(1591)
_N = 3
_Q = _RNG.normal(size=(_N, _N))
_Q = 0.5 * (_Q + _Q.T)
_QA = _RNG.normal(size=(_N, _N))  # asymmetric
_A = _RNG.normal(size=(2, _N))
_B = _RNG.normal(size=(2, _N))
_C = _RNG.normal(size=_N)


def _s(v):
    return [v[i] for i in range(v.shape[0])]


def _qform(M, xs, ys):
    """Literal transliteration of ``xs @ M @ ys``: sum_k (sum_i M[i,k] xs[i]) * ys[k]."""
    return sum(sum(float(M[i, k]) * xs[i] for i in range(len(xs))) * ys[k] for k in range(len(ys)))


def _lin(M, xs):
    return [sum(float(M[r, i]) * xs[i] for i in range(len(xs))) for r in range(M.shape[0])]


def _dot(us, vs):
    return sum(u * v for u, v in zip(us, vs))


# name -> (array spelling, scalar transliteration, numpy evaluator)
FORMS = {
    "x@Q@x": (
        lambda x, y, X, Y: x @ _Q @ x,
        lambda x, y, X, Y: _qform(_Q, _s(x), _s(x)),
        lambda x, y, X, Y: x @ _Q @ x,
    ),
    "x@(Qa@x)": (
        lambda x, y, X, Y: x @ (_QA @ x),
        lambda x, y, X, Y: _dot(_s(x), _lin(_QA, _s(x))),
        lambda x, y, X, Y: x @ _QA @ x,
    ),
    "(Qa@x)@x": (
        lambda x, y, X, Y: (_QA @ x) @ x,
        lambda x, y, X, Y: _dot(_lin(_QA, _s(x)), _s(x)),
        lambda x, y, X, Y: (_QA @ x) @ x,
    ),
    "x@Qa@y": (
        lambda x, y, X, Y: x @ _QA @ y,
        lambda x, y, X, Y: _qform(_QA, _s(x), _s(y)),
        lambda x, y, X, Y: x @ _QA @ y,
    ),
    "(A@x)@(B@y)": (
        lambda x, y, X, Y: (_A @ x) @ (_B @ y),
        lambda x, y, X, Y: _dot(_lin(_A, _s(x)), _lin(_B, _s(y))),
        lambda x, y, X, Y: (_A @ x) @ (_B @ y),
    ),
    "sum(x*(Q@x))": (
        lambda x, y, X, Y: dm.sum(x * (_Q @ x)),
        lambda x, y, X, Y: _dot(_s(x), _lin(_Q, _s(x))),
        lambda x, y, X, Y: np.sum(x * (_Q @ x)),
    ),
    "-sum(x**2)+c@x": (
        lambda x, y, X, Y: -dm.sum(x**2) + _C @ x,
        lambda x, y, X, Y: sum(-xi * xi + float(ci) * xi for xi, ci in zip(_s(x), _C)),
        lambda x, y, X, Y: -np.sum(x**2) + _C @ x,
    ),
    "sum(x*y)": (
        lambda x, y, X, Y: dm.sum(x * y),
        lambda x, y, X, Y: _dot(_s(x), _s(y)),
        lambda x, y, X, Y: np.sum(x * y),
    ),
    "sum((A@x)*(B@x))": (
        lambda x, y, X, Y: dm.sum((_A @ x) * (_B @ x)),
        lambda x, y, X, Y: _dot(_lin(_A, _s(x)), _lin(_B, _s(x))),
        lambda x, y, X, Y: np.sum((_A @ x) * (_B @ x)),
    ),
    "2(x@Q@x)-x@y": (
        lambda x, y, X, Y: 2.0 * (x @ _Q @ x) - x @ y,
        lambda x, y, X, Y: 2.0 * _qform(_Q, _s(x), _s(x)) - _dot(_s(x), _s(y)),
        lambda x, y, X, Y: 2.0 * (x @ _Q @ x) - x @ y,
    ),
    "sum(X@Y)": (
        lambda x, y, X, Y: dm.sum(X @ Y),
        lambda x, y, X, Y: sum(
            X[i, k] * Y[k, j] for i in range(2) for j in range(2) for k in range(2)
        ),
        lambda x, y, X, Y: np.sum(X @ Y),
    ),
    "sum(X*Y)-sum(X*X)": (
        lambda x, y, X, Y: dm.sum(X * Y) - dm.sum(X * X),
        lambda x, y, X, Y: sum(
            X[i, j] * Y[i, j] - X[i, j] * X[i, j] for i in range(2) for j in range(2)
        ),
        lambda x, y, X, Y: np.sum(X * Y) - np.sum(X * X),
    ),
    "sum(x**3)": (
        lambda x, y, X, Y: dm.sum(x**3),
        lambda x, y, X, Y: sum(xi**3 for xi in _s(x)),
        lambda x, y, X, Y: np.sum(x**3),
    ),
}


def _build(f, *, extra_scalar_product: bool = False) -> dm.Model:
    m = dm.Model("i1591")
    x = m.continuous("x", shape=(_N,), lb=-1.0, ub=2.0)
    y = m.continuous("y", shape=(_N,), lb=-2.0, ub=1.0)
    X = m.continuous("X", shape=(2, 2), lb=-1.0, ub=1.5)
    Y = m.continuous("Y", shape=(2, 2), lb=-1.5, ub=1.0)
    obj = f(x, y, X, Y)
    if extra_scalar_product:
        z = m.continuous("z", lb=0.0, ub=1.0)
        w = m.continuous("w", lb=0.0, ub=1.0)
        m.subject_to(z * w <= 0.5)
        obj = obj + z
    m.minimize(obj)
    return m


def _split(p):
    return p[0:3], p[3:6], p[6:10].reshape(2, 2), p[10:14].reshape(2, 2)


def _catalog(t):
    return (
        sorted(tuple(sorted(p)) for p in t.bilinear),
        sorted(t.trilinear),
        sorted(t.monomial),
    )


@pytest.mark.parametrize("name", list(FORMS))
@pytest.mark.parametrize("extra", [False, True], ids=["alone", "with_scalar_product"])
def test_array_form_catalog_matches_scalar_transliteration(name, extra):
    """The array spelling yields exactly the scalar spelling's catalog, nothing opaque.

    ``extra=True`` adds an unrelated scalar ``z * w``: on main the Rust route then
    returned only ``(z, w)`` and reported the catalog complete.
    """
    fa, fs, _ = FORMS[name]
    ta = tc.classify_nonlinear_terms(_build(fa, extra_scalar_product=extra))
    ts = tc.classify_nonlinear_terms(_build(fs, extra_scalar_product=extra))
    assert ta.general_nl == []
    assert _catalog(ta) == _catalog(ts)
    assert _catalog(ta) != ([], [], [])


def test_issue_form_has_relaxable_nonlinearity():
    m = _build(FORMS["x@Q@x"][0])
    assert MccormickLPRelaxer(m).has_relaxable_nonlinearity


def test_constant_matmul_stays_linear():
    """A matmul with a variable-free side is affine: nothing is catalogued."""
    m = dm.Model("lin")
    x = m.continuous("x", shape=(_N,), lb=-1.0, ub=2.0)
    p = m.parameter("p", value=_C)
    m.subject_to(_A @ x <= 1.0)
    m.minimize(p @ x + _C @ (_Q @ x))
    t = tc.classify_nonlinear_terms(m)
    assert _catalog(t) == ([], [], [])
    assert t.general_nl == []


def test_array_forms_relaxation_sound_and_equal_to_scalar_on_sub_boxes():
    """Differential bound test + feasible-point sampling on fixed random sub-boxes."""
    rng = np.random.default_rng(15910)
    executed = 0
    for name, (fa, fs, fe) in FORMS.items():
        ma, ms = _build(fa), _build(fs)
        ra, rs = MccormickLPRelaxer(ma), MccormickLPRelaxer(ms)
        lb0, ub0 = flat_variable_bounds(ma)
        for b in range(8):
            if b == 0:
                lo, hi = lb0.copy(), ub0.copy()
            else:
                u = rng.uniform(size=(2, lb0.size))
                lo = lb0 + (ub0 - lb0) * np.minimum(u[0], u[1])
                hi = lb0 + (ub0 - lb0) * np.maximum(u[0], u[1])
            resa = ra.solve_at_node(lo, hi)
            ress = rs.solve_at_node(lo, hi)
            assert resa.status == "optimal", (name, b, resa.status)
            assert ress.status == "optimal", (name, b, ress.status)
            pts = lo + (hi - lo) * rng.uniform(size=(1500, lo.size))
            fmin = min(fe(*_split(p)) for p in pts)
            tol = 1e-6 * max(1.0, abs(fmin))
            assert resa.lower_bound <= fmin + tol, (name, b, resa.lower_bound, fmin)
            assert resa.lower_bound == pytest.approx(ress.lower_bound, rel=1e-6, abs=1e-6), (
                name,
                b,
            )
            executed += 1
    assert executed == len(FORMS) * 8


def _issue_model(spelling: str) -> dm.Model:
    n = 8
    rng = np.random.default_rng(0)
    q = rng.normal(size=(n, n))
    q = 0.5 * (q + q.T)
    a = rng.uniform(0, 1, size=(2, n))
    b = a.sum(1) * 0.4
    m = dm.Model(spelling)
    x = m.continuous("x", shape=(n,), lb=0, ub=1)
    m.subject_to(a @ x <= b)
    if spelling == "xqx":
        m.minimize(x @ q @ x)
    elif spelling == "x(qx)":
        m.minimize(x @ (q @ x))
    else:
        xs = [x[i] for i in range(n)]
        m.minimize(sum(float(q[i, j]) * xs[i] * xs[j] for i in range(n) for j in range(n)))
    return m


@pytest.mark.parametrize("spelling", ["xqx", "x(qx)"])
def test_issue_repro_certifies_like_scalar_spelling(spelling):
    ref = _issue_model("scalar").solve(time_limit=60, deterministic=True)
    r = _issue_model(spelling).solve(time_limit=60, deterministic=True)
    assert ref.status == "optimal" and ref.gap_certified
    assert r.status == "optimal", r.status
    assert r.gap_certified
    assert r.objective == pytest.approx(ref.objective, rel=1e-4, abs=1e-6)
    assert r.bound <= r.objective + 1e-6
