"""Issue #865 — perspective terms in the convex-kernel producer.

The hull-reformulated (``*hfsg``) family writes its disjunctive nonlinearities as
the perspective ``s·f(a/s)`` with ``s = 0.001 + 0.999·y``. Syntactically that is a
product of two non-constant subexpressions, so the pre-#865 gate rejected it as a
"bilinear product"; mathematically the perspective of a convex ``f`` is *jointly
convex* on ``s > 0``, so admitting it recognises convexity the gate missed rather
than loosening anything.

The tests below pin, in order of what actually protects the certificate:

1. **exactness** — the marshaled row equals the pristine model's row pointwise
   (the lift ``s·h(·/s) → affine + perspective`` is an algebraic identity, not an
   approximation);
2. **convexity** — every routed row satisfies the midpoint inequality, so its OA
   tangent is a valid relaxation;
3. **soundness gates** — a scale not provably positive on the box, a
   wrong-curvature perspective, and a genuine bilinear product all still fall back;
4. **no drift** — models routed before #865 marshal byte-identically (their terms
   carry an all-zero scale, i.e. the plain composite form).
"""

from __future__ import annotations

import os

import discopt.modeling as dm
import numpy as np
import pytest

_ck = pytest.importorskip("discopt.solvers._convex_kernel")
build_convex_spec = _ck.build_convex_spec
solve_convex_tree = _ck.solve_convex_tree

_DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl")
# BARON-confirmed optimum from minlplib.solu (=opt=); syn05hfsg is MAXIMIZE, so the
# dual bound is an UPPER bound and `bound >= opt` is the correct-side invariant
# (see test_issue759_syn05hfsg_bound_sense.py).
_SYN05HFSG_OPT = 837.7324009

_FUNC_NP = {
    "log": np.log,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "log1p": np.log1p,
    "sqr": np.square,
}


def _scalar(m, expr_fn, name):
    m.constraint(dm.RangeSet(1), lambda _i: expr_fn(), name=name, fast=False)


def _row_value(d, x):
    """g(x) for a `_Decomp`, mirroring the Rust kernel's term semantics exactly."""
    v = d.const + sum(k * x[c] for c, k in d.aff.items())
    for t in d.terms:
        a = t["arg_const"] + sum(k * x[c] for c, k in t["arg_aff"].items())
        f = _FUNC_NP[t["func"]]
        if t["sc_aff"] is None:
            v += t["coeff"] * f(a)
        else:
            s = t["sc_const"] + sum(k * x[c] for c, k in t["sc_aff"].items())
            v += t["coeff"] * s * f(a / s)
    return float(v)


def _nl_decomps(model):
    """Re-run the producer's row loop: [(row_index, sign, _Decomp)] for nonlinear rows."""
    from discopt._jax.gdp_reformulate import reformulate_gdp
    from discopt._jax.model_utils import flat_variable_bounds
    from discopt._jax.nlp_evaluator import NLPEvaluator

    m = reformulate_gdp(model, method="big-m")
    lb, ub = flat_variable_bounds(m)
    lb, ub = lb.astype(float), ub.astype(float)
    ev = NLPEvaluator(m)
    n = len(lb)
    rng = np.random.default_rng(0)
    lo = np.where(np.isfinite(lb), lb, 0.0)
    hi = np.where(np.isfinite(ub), ub, lo + 5.0)
    xa = lo + rng.random(n) * (hi - lo)
    xb = lo + rng.random(n) * (hi - lo)
    lin = np.all(np.isclose(ev.evaluate_jacobian(xa), ev.evaluate_jacobian(xb), atol=1e-9), axis=1)
    offsets = _ck._flat_offsets(m)
    rows = []
    for i, con in enumerate(m._constraints):
        if lin[i]:
            continue
        sense = con.sense if isinstance(con.sense, str) else con.sense.value
        d = _ck._decompose(_ck._constraint_expr(m, i), offsets)
        sign = -1.0 if sense == ">=" else 1.0
        if sign < 0:
            d.scale(-1.0)
        rows.append((i, sign, d))
    return m, lb, ub, ev, rows


def _box_sample(lb, ub, rng, n_pts):
    lo = np.where(np.isfinite(lb), lb, -10.0)
    hi = np.where(np.isfinite(ub), ub, lo + 20.0)
    return lo + rng.random((n_pts, len(lb))) * (hi - lo)


# ── the real-corpus instance the issue is about ───────────────────────────────


def test_syn05hfsg_is_routed_after_865():
    """The `*hfsg` perspective family reaches the kernel (it did not before #865)."""
    m = dm.from_nl(os.path.join(_DATA, "syn05hfsg.nl"))
    spec = build_convex_spec(m)
    assert spec is not None, "syn05hfsg's smoothed perspective rows must be routed"
    # Its three nonlinear rows are perspectives, i.e. they carry a nonzero scale.
    assert int(np.count_nonzero(spec["term_scale_const"])) == 3


def test_syn05hfsg_certifies_the_true_optimum():
    """Certified objective == the BARON optimum, and the bound is on the sound side."""
    m = dm.from_nl(os.path.join(_DATA, "syn05hfsg.nl"))
    spec = build_convex_spec(m)
    assert spec is not None
    r = solve_convex_tree(spec, initial_incumbent=None, time_limit_s=120.0)
    assert r["status"] == "optimal"
    inc, bound = r["incumbent"], r["bound"]
    tol = 1e-4 * max(1.0, abs(_SYN05HFSG_OPT))
    assert abs(inc - _SYN05HFSG_OPT) < tol, f"incumbent {inc} != {_SYN05HFSG_OPT}"
    # MAXIMIZE: the dual bound is an UPPER bound — it must never fall below the
    # true optimum (that would be a too-tight, unsound bound).
    assert bound >= _SYN05HFSG_OPT - tol, f"unsound dual bound {bound} < {_SYN05HFSG_OPT}"
    assert bound >= inc - tol, "certificate invariant: bound ≥ incumbent (max sense)"


@pytest.mark.parametrize(
    ("instance", "func", "n_pts"),
    [
        ("syn05hfsg", "log", 120),  # log perspective  −s·ln(a/s)
        ("clay0303hfsg", "sqr", 40),  # quadratic perspective  s·(a/s)² = a²/s
    ],
)
def test_perspective_lift_is_exact_and_convex(instance, func, n_pts):
    """The two properties the certificate rests on, checked over the box.

    Exactness: the marshaled row must equal the PRISTINE model's row pointwise —
    any drift would mean the lift is an approximation, not an identity.
    Convexity: the midpoint inequality must hold on every routed row, else the OA
    tangent is not a valid relaxation.
    """
    model = dm.from_nl(os.path.join(_DATA, f"{instance}.nl"))
    assert build_convex_spec(model) is not None
    m, lb, ub, ev, rows = _nl_decomps(model)
    assert any(t["sc_aff"] is not None for _i, _s, d in rows for t in d.terms)
    assert {t["func"] for _i, _s, d in rows for t in d.terms} == {func}

    rng = np.random.default_rng(12345)
    X = _box_sample(lb, ub, rng, n_pts)
    for x in X:
        g = np.asarray(ev.evaluate_constraints(x), float)
        for i, sign, d in rows:
            ref, got = sign * g[i], _row_value(d, x)
            if np.isfinite(ref) and np.isfinite(got):
                assert abs(ref - got) <= 1e-9 * max(1.0, abs(ref)), (
                    f"row {i}: marshaled {got} != model {ref}"
                )

    A, B = _box_sample(lb, ub, rng, n_pts), _box_sample(lb, ub, rng, n_pts)
    for a, b in zip(A, B):
        for lam in (0.25, 0.5, 0.75):
            mid = lam * a + (1 - lam) * b
            for i, _s, d in rows:
                gm, ga, gb = _row_value(d, mid), _row_value(d, a), _row_value(d, b)
                if not all(np.isfinite(v) for v in (gm, ga, gb)):
                    continue
                assert gm - (lam * ga + (1 - lam) * gb) <= 1e-9 * max(1.0, abs(gm)), (
                    f"row {i} is not convex at lambda={lam}"
                )


# ── quadratic inner function: `** 2` → sqr (#865 follow-up) ───────────────────


def test_clay0303hfsg_is_routed_with_quadratic_perspectives():
    """`clay*hfsg`'s hull rows are `ε·((x/ε)² − c·x/ε + …)`, i.e. the quadratic
    perspective `x²/ε`. They were declined before `sqr` existed."""
    m = dm.from_nl(os.path.join(_DATA, "clay0303hfsg.nl"))
    spec = build_convex_spec(m)
    assert spec is not None, "clay0303hfsg's quadratic perspective rows must be routed"
    assert set(spec["term_func"].tolist()) == {_ck._FUNC_CODE["sqr"]}
    # every one of its 72 terms is a perspective (carries a nonzero scale)
    assert int(np.count_nonzero(spec["term_scale_const"])) == len(spec["term_coeff"]) == 72


def test_plain_square_row_is_routed_and_certified():
    """A plain (non-perspective) `x² ≤ c` row is convex and must certify.

    max x + k  s.t.  x² ≤ 4 (→ x ≤ 2),  k ≤ x,  x∈[0,10], k∈{0..3} int.
    Optimum: x = 2, k = 2 → 4.
    """
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=10.0)
    k = m.integer("k", lb=0, ub=3)
    _scalar(m, lambda: x**2 <= 4.0, "sq")
    _scalar(m, lambda: k - x <= 0, "kx")
    m.maximize(x + k)

    spec = build_convex_spec(m)
    assert spec is not None
    assert set(spec["term_func"].tolist()) == {_ck._FUNC_CODE["sqr"]}
    assert np.all(spec["term_scale_const"] == 0.0)  # plain, not a perspective

    r = solve_convex_tree(spec, initial_incumbent=None, time_limit_s=30.0)
    assert r["status"] == "optimal"
    assert abs(r["incumbent"] - 4.0) < 1e-3, f"incumbent {r['incumbent']} != 4"
    assert r["bound"] >= 4.0 - 1e-6, "sound: dual bound never below the true optimum"


def test_wrong_curvature_square_falls_back():
    """`x² ≥ 1` → its ≤-normal form is `−x² + 1 ≤ 0`, a CONCAVE term → nonconvex
    feasible set → must fall back."""
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=5.0)
    z = m.binary("z")
    _scalar(m, lambda: x**2 >= 1.0, "sqge")
    m.maximize(x + z)
    assert build_convex_spec(m) is None


@pytest.mark.parametrize("exponent", [3, 4, 0.5, 1.5, -1, -2])
def test_only_exponent_two_is_admitted(exponent):
    """Every other power is nonconvex, domain-restricted, or signomial — the gate
    must keep refusing them rather than wave them through with `sqr`."""
    m = dm.Model()
    x = m.continuous("x", lb=0.5, ub=5.0)
    z = m.binary("z")
    _scalar(m, lambda: x**exponent <= 4.0, "pow")
    m.maximize(x + z)
    assert build_convex_spec(m) is None, f"exponent {exponent} must not be routed"


def test_variable_exponent_falls_back():
    m = dm.Model()
    x = m.continuous("x", lb=0.5, ub=5.0)
    y = m.continuous("y", lb=1.0, ub=3.0)
    z = m.binary("z")
    _scalar(m, lambda: x**y <= 4.0, "varpow")
    m.maximize(x + y + z)
    assert build_convex_spec(m) is None


def test_square_of_nonaffine_base_falls_back():
    """`(log x)² ≤ c` — the base is not affine, so the composite is not a `sqr` of
    an affine form and its convexity is not established by this gate."""
    m = dm.Model()
    x = m.continuous("x", lb=1.0, ub=5.0)
    z = m.binary("z")
    _scalar(m, lambda: dm.log(x) ** 2 <= 4.0, "logsq")
    m.maximize(x + z)
    assert build_convex_spec(m) is None


# ── soundness gates ───────────────────────────────────────────────────────────


def _perspective_model(*, scale_lb: float, sense_ge: bool = False):
    """`s·log(u/s + 1)` with `s = scale_lb + (1-scale_lb)·y`, y binary.

    With ``scale_lb > 0`` the perspective is convex and the ``≤`` row is routable;
    the caller varies ``scale_lb`` (positivity of the scale) and the row sense
    (curvature) to exercise each gate.
    """
    m = dm.Model()
    u = m.continuous("u", lb=0.0, ub=10.0)
    y = m.binary("y")
    w = m.continuous("w", lb=0.0, ub=10.0)

    def body():
        s = scale_lb + (1.0 - scale_lb) * y
        expr = (w / s - dm.log(u / s + 1.0)) * s
        return expr >= 0.0 if sense_ge else expr <= 0.0

    _scalar(m, body, "persp")
    _scalar(m, lambda: u + w <= 8.0, "lin")
    m.maximize(u + w + y)
    return m


def test_positive_scale_perspective_is_routed():
    assert build_convex_spec(_perspective_model(scale_lb=0.001)) is not None


def test_scale_not_provably_positive_falls_back():
    """`s = 0 + 1·y` touches 0 on the box → the perspective is not convex there →
    the gate must refuse rather than emit an invalid tangent."""
    assert build_convex_spec(_perspective_model(scale_lb=0.0)) is None


def test_wrong_curvature_perspective_falls_back():
    """The same row as `>=`: its ≤-normal form flips every sign, making the
    perspective term concave → not routable."""
    assert build_convex_spec(_perspective_model(scale_lb=0.001, sense_ge=True)) is None


def test_genuine_bilinear_still_falls_back():
    """A real var*var product has no perspective structure — the #865 path must not
    admit it by mistake."""
    m = dm.Model()
    a = m.continuous("a", lb=0.5, ub=5.0)
    b = m.continuous("b", lb=0.5, ub=5.0)
    z = m.binary("z")
    _scalar(m, lambda: a * b <= 4.0, "bilin")
    m.maximize(a + b + z)
    assert build_convex_spec(m) is None


def test_scaled_product_without_matching_denominator_falls_back():
    """`(w/s2 - log(u/s2 + 1)) * s1` with s1 != s2 is NOT a perspective."""
    m = dm.Model()
    u = m.continuous("u", lb=0.0, ub=10.0)
    w = m.continuous("w", lb=0.0, ub=10.0)
    y = m.binary("y")
    z = m.binary("z")
    s1 = 0.001 + 0.999 * y
    s2 = 0.001 + 0.999 * z
    _scalar(m, lambda: (w / s2 - dm.log(u / s2 + 1.0)) * s1 <= 0.0, "notpersp")
    m.maximize(u + w + y + z)
    assert build_convex_spec(m) is None


# ── no drift on models routed before #865 ─────────────────────────────────────


def test_plain_composite_terms_carry_a_zero_scale():
    """A pre-#865 routable model marshals unchanged: every term is the plain form,
    flagged by an empty scale CSR row with a zero constant."""
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=10.0)
    k = m.integer("k", lb=0, ub=3)
    _scalar(m, lambda: k - x <= 0, "kx")
    _scalar(m, lambda: dm.exp(x) <= 5.0, "expc")
    m.maximize(x + k)

    spec = build_convex_spec(m)
    assert spec is not None
    n_terms = len(spec["term_coeff"])
    assert n_terms == 1
    assert np.all(spec["term_scale_const"] == 0.0)
    assert len(spec["term_scale_cols"]) == 0
    assert list(spec["term_scale_ptr"]) == [0] * (n_terms + 1)

    # ...and it still certifies the analytic optimum ln(5) + 1.
    r = solve_convex_tree(spec, initial_incumbent=None, time_limit_s=30.0)
    assert r["status"] == "optimal"
    truth = float(np.log(5.0)) + 1.0
    assert abs(r["incumbent"] - truth) < 1e-3
    assert r["bound"] >= truth - 1e-6


def test_model_solve_routes_syn05hfsg_when_flag_on(monkeypatch):
    """End-to-end: with the kernel flag on, `Model.solve()` returns the certified
    optimum, incumbent-verified against the pristine model (the #779 guard)."""
    monkeypatch.setenv("DISCOPT_CONVEX_KERNEL", "1")
    m = dm.from_nl(os.path.join(_DATA, "syn05hfsg.nl"))
    r = m.solve(time_limit=120, gap_tolerance=1e-4)
    tol = 1e-4 * max(1.0, abs(_SYN05HFSG_OPT))
    assert r.objective is not None
    assert abs(r.objective - _SYN05HFSG_OPT) < tol, f"objective {r.objective}"
    assert r.bound >= _SYN05HFSG_OPT - tol, f"unsound dual bound {r.bound}"
