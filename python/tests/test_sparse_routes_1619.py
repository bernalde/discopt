"""#1619: the LP route and the post-solve screen stay sparse; dm.external memoizes.

B-13: a ``dm.external`` block was called once per *use* of its output, so a
point used in the objective and a constraint cost two simulator calls (400 calls
for 199 points). The memo makes it one per distinct point.

A-12: ``solver="pounce"`` on an LP densified ``A`` (and pounce then densified it
again into ``G``), so a 2000-row chain LP peaked at 542 MB of traced memory.
B-12a: the post-solve screen scattered the natively sparse tape Jacobian into a
dense ``(m, n)`` array (plus four same-sized intermediates), quadratic in the
collocation mesh: 667 MB at nfe=400, 10.5 GB at nfe=1600.

The memory tests assert a peak bound well under the origin/main figure and well
over the fixed one (21.5 MB / 14.8 MB measured), so they fail on main. The parity
tests pin that every sparse arm computes exactly what the dense arm computes —
the screen is a soundness gate, so "sparse" must never mean "weaker".
"""

from __future__ import annotations

import tracemalloc

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt.validation import feasibility as F

PEAK_LIMIT_MB = 120.0


def _chain_lp(n: int) -> dm.Model:
    m = dm.Model("chain")
    x = m.continuous("x", shape=(2 * n,), lb=0, ub=100)
    for i in range(n):
        m.subject_to(x[2 * i] + x[2 * i + 1] - 0.5 * x[(2 * i + 2) % (2 * n)] >= 1)
    m.minimize(dm.sum(x))
    return m


def _batch_dae(nfe: int) -> dm.Model:
    from discopt.dae import ContinuousSet, DAEBuilder

    m = dm.Model("batch")
    dae = DAEBuilder(m, ContinuousSet("t", bounds=(0.0, 60.0), nfe=nfe, ncp=3))
    dae.add_state("cA", initial=2.0, bounds=(0, 2))
    dae.add_state("cB", initial=0.0, bounds=(0, 2))
    dae.add_control("T", bounds=(300, 360))

    def rhs(t, s, a, c):
        k1 = 1e6 * dm.exp(-5e4 / (8.314 * c["T"]))
        k2 = 5.6e10 * dm.exp(-8e4 / (8.314 * c["T"]))
        return {"cA": -k1 * s["cA"], "cB": k1 * s["cA"] - k2 * s["cB"]}

    dae.set_ode(rhs)
    v = dae.discretize()
    m.maximize(v["cB"][-1, -1])
    return m


def _peak_mb(fn):
    tracemalloc.start()
    try:
        out = fn()
        peak = tracemalloc.get_traced_memory()[1] / 1e6
    finally:
        tracemalloc.stop()
    return out, peak


# ---------------------------------------------------------------- A-12 (LP route)


def test_extract_lp_data_sparse_context_forces_csr():
    from discopt._relax.problem_classifier import extract_lp_data, sparse_constraint_matrices

    m = _chain_lp(5)
    assert not sp.issparse(extract_lp_data(m).A_eq)  # default: small A stays dense
    with sparse_constraint_matrices():
        data = extract_lp_data(m)
    assert sp.issparse(data.A_eq)
    dense = extract_lp_data(m)
    assert not sp.issparse(dense.A_eq)  # the context var was reset on exit
    np.testing.assert_array_equal(data.A_eq.toarray(), np.asarray(dense.A_eq))
    np.testing.assert_array_equal(np.asarray(data.b_eq), np.asarray(dense.b_eq))
    np.testing.assert_array_equal(np.asarray(data.c), np.asarray(dense.c))


def test_pounce_lp_route_peak_memory_is_linear():
    pytest.importorskip("pounce")
    n = 1500
    res, peak = _peak_mb(lambda: _chain_lp(n).solve(solver="pounce"))
    assert res.status == "optimal"
    # checked against the default (exact simplex) route, not derived by hand
    ref = _chain_lp(n).solve()
    assert ref.status == "optimal"
    assert res.objective == pytest.approx(ref.objective, rel=1e-8, abs=1e-8)
    assert peak < PEAK_LIMIT_MB, f"LP pounce route peaked at {peak:.1f} MB (n={n})"


# ------------------------------------------------------- B-12a (post-solve screen)


def test_screen_jacobian_is_sparse_and_exact_for_tape_evaluator():
    from discopt._tape_nlp_evaluator import TapeNLPEvaluator

    m = _batch_dae(10)
    ev = TapeNLPEvaluator(m)
    rng = np.random.default_rng(0)
    lb, ub = ev.variable_bounds
    x = lb + rng.random(lb.size) * (np.minimum(ub, 1e3) - lb)
    J = F.screen_jacobian(ev, x)
    assert sp.issparse(J)
    np.testing.assert_array_equal(J.toarray(), np.asarray(ev.evaluate_jacobian(x)))


def test_dae_post_solve_screen_peak_memory_is_linear():
    pytest.importorskip("pounce")
    fired = {"n": 0, "sparse": 0}
    orig = F.improving_gradient_norms

    def counting(J, *a, **k):
        fired["n"] += 1
        fired["sparse"] += int(sp.issparse(J))
        return orig(J, *a, **k)

    mp = pytest.MonkeyPatch()
    mp.setattr(F, "improving_gradient_norms", counting)
    try:
        res, peak = _peak_mb(lambda: _batch_dae(400).solve(solver="pounce"))
    finally:
        mp.undo()
    assert res.status in ("optimal", "local_optimal")
    assert fired["n"] > 0, "the post-solve screen never ran"
    assert fired["sparse"] == fired["n"]
    assert peak < PEAK_LIMIT_MB, f"DAE solve peaked at {peak:.1f} MB (nfe=400)"


# ------------------------------------------------------------ sparse/dense parity


def _random_jac(rng, m, n, density=0.2, special=True):
    A = sp.random(m, n, density=density, random_state=rng, format="csr")
    A.data = rng.normal(size=A.data.size) * 10.0 ** rng.integers(-3, 3, size=A.data.size)
    if special:
        A = A.tolil()
        A[0, 0] = np.inf
        A[1, 1] = np.nan
        A[2, :] = 0.0  # an all-zero (structurally empty) row
        A = A.tocsr()
        A.eliminate_zeros()
    return A


@pytest.mark.parametrize("seed", range(8))
def test_row_gradient_norms_sparse_matches_dense(seed):
    rng = np.random.default_rng(seed)
    J = _random_jac(rng, 30, 25)
    np.testing.assert_array_equal(
        F.jacobian_row_gradient_norms(J), F.jacobian_row_gradient_norms(J.toarray())
    )


@pytest.mark.parametrize("seed", range(8))
def test_improving_gradient_norms_sparse_matches_dense(seed):
    rng = np.random.default_rng(seed)
    m, n = 30, 25
    J = _random_jac(rng, m, n)
    lb = rng.uniform(-5, 0, n)
    ub = lb + rng.uniform(0, 5, n)
    x = np.clip(rng.uniform(-6, 6, n), lb - 1e-7, ub + 1e-7)
    # put some columns exactly on a bound, some integers on / off integers
    x[:4] = lb[:4]
    x[4:8] = ub[4:8]
    mask = np.zeros(n, dtype=bool)
    mask[8:14] = True
    x[8:11] = np.round(x[8:11])
    x[11:14] = np.round(x[11:14]) + rng.uniform(-3e-7, 3e-7, 3)
    lb[8:14] = np.floor(lb[8:14])
    ub[8:14] = np.ceil(ub[8:14])
    d = rng.integers(-1, 2, m).astype(float)
    for imask in (None, mask):
        got = F.improving_gradient_norms(J, x, lb, ub, d, imask)
        want = F.improving_gradient_norms(J.toarray(), x, lb, ub, d, imask)
        np.testing.assert_allclose(got, want, rtol=1e-13, atol=0.0)
        np.testing.assert_array_equal(np.isinf(got), np.isinf(want))


@pytest.mark.parametrize("seed", range(4))
def test_scale_from_jacobian_sparse_matches_dense(seed):
    from discopt._relax.primal_heuristics import _scale_from_jacobian

    rng = np.random.default_rng(seed)
    J = _random_jac(rng, 20, 15)
    x = rng.normal(size=15)
    x[0] = 0.0
    np.testing.assert_allclose(
        _scale_from_jacobian(J, x), _scale_from_jacobian(J.toarray(), x), rtol=1e-13
    )


def test_matrix_solution_feasible_sparse_matches_dense_and_rejects():
    from discopt.solver import _matrix_solution_feasible

    rng = np.random.default_rng(3)
    n = 12
    A_ub = sp.random(8, n, density=0.4, random_state=rng, format="csr")
    A_eq = sp.random(4, n, density=0.4, random_state=rng, format="csr")
    x = rng.uniform(0, 1, n)
    b_ub = A_ub @ x + 1e-3
    b_eq = A_eq @ x
    bounds = [(0.0, 1.0)] * n
    for args in ((A_ub, A_eq), (A_ub.toarray(), A_eq.toarray())):
        assert _matrix_solution_feasible(x, args[0], b_ub, args[1], b_eq, bounds)
    # violate one inequality row and one equality row: both forms must reject
    bad_ub = b_ub.copy()
    bad_ub[2] -= 1.0
    bad_eq = b_eq.copy()
    bad_eq[1] += 1e-3
    for A1, A2 in ((A_ub, A_eq), (A_ub.toarray(), A_eq.toarray())):
        assert not _matrix_solution_feasible(x, A1, bad_ub, A2, b_eq, bounds)
        assert not _matrix_solution_feasible(x, A1, b_ub, A2, bad_eq, bounds)


# --------------------------------------------------------- B-13 (dm.external memo)


def _counted_external():
    calls = {"fn": 0}

    def f(x):
        calls["fn"] += 1
        return np.array([x[0] ** 2 + x[1] ** 2, x[0] - x[1]])

    sim = dm.external(
        f,
        jac=lambda x: np.array([[2 * x[0], 2 * x[1]], [1.0, -1.0]]),
        hess=lambda x: np.stack([2 * np.eye(2), np.zeros((2, 2))]),
        shape=(2,),
    )
    return sim, calls


def test_external_called_once_per_distinct_point():
    sim, calls = _counted_external()
    m = dm.Model("e")
    x = m.continuous("x", shape=(2,), lb=-3, ub=3)
    y = sim(x)
    m.minimize(y[0] + 0.5 * y[1])
    m.subject_to(y[1] >= 0.2)
    r = m.solve(solver="direct", max_evals=200, local_refine=False)
    evals = int(r.solver_stats["direct/evals"])
    assert evals > 0 and r.objective is not None
    # two uses of y per point: main called fn twice per evaluation (400 for 199)
    assert calls["fn"] <= evals + 1, (calls["fn"], evals)


def test_external_memo_returns_fresh_copies_and_distinguishes_points():
    from discopt.modeling.external import _checked

    seen = []

    def f(x):
        seen.append(np.array(x))
        return np.array([x[0] * 2.0])

    call = _checked(f, role="fn", label="t", want=lambda xs: (1,))
    a = call(np.array([1.0]))
    a[0] = 99.0  # a consumer mutating its result must not poison the memo
    assert call(np.array([1.0]))[0] == 2.0
    assert len(seen) == 1
    # a different point -- even by one ulp -- is a fresh call
    nxt = np.nextafter(1.0, 2.0)
    assert call(np.array([nxt]))[0] == 2.0 * nxt
    assert len(seen) == 2


def test_external_memo_never_caches_a_failure():
    from discopt.modeling.external import _checked

    n = {"k": 0}

    def flaky(x):
        n["k"] += 1
        if n["k"] == 1:
            raise RuntimeError("transient")
        return np.array([1.0])

    call = _checked(flaky, role="fn", label="t", want=lambda xs: (1,))
    with pytest.raises(RuntimeError):
        call(np.array([0.0]))
    assert call(np.array([0.0]))[0] == 1.0
    assert n["k"] == 2
