"""#1616 A-01: a public convexity verdict that matches what the solve dispatches on.

The solve classifies convexity after exact rewrites (entropy canonicalisation,
objective epigraph). The private classifier sees the model as written, so it
could report a model nonconvex that the solve then put on its convex fast path.
"""

from __future__ import annotations

import discopt
import discopt.modeling as dm
import pytest
from discopt._relax.convexity import classify_model


def _xlogx():
    m = dm.Model("xlogx_product")
    x = m.continuous("x", lb=0.01, ub=0.99)
    m.minimize(x * dm.log(x))
    return m


def _xlogx_intrinsic():
    m = dm.Model("xlogx_intrinsic")
    y = m.continuous("y", lb=0.01, ub=0.99)
    m.minimize(dm.xlogx(y))
    return m


def _epigraph():
    # min z s.t. z == x^2 + y^2: the equality is nonconvex *as written*; the
    # solve relaxes it to z >= x^2 + y^2, which is exact at the optimum.
    m = dm.Model("epigraph")
    x = m.continuous("x", lb=-2, ub=2)
    y = m.continuous("y", lb=-2, ub=2)
    z = m.continuous("z")  # free: the rewrite needs z unbounded below
    m.minimize(z)
    m.subject_to(z == (x - 1) ** 2 + (y + 0.5) ** 2, name="defn")
    m.subject_to(x + y >= 1, name="lin")
    return m


def _bilinear():
    m = dm.Model("bilinear")
    u = m.continuous("u", lb=-1, ub=1)
    v = m.continuous("v", lb=-1, ub=1)
    m.minimize(u * v)
    m.subject_to(u**2 + v**2 <= 1, name="disk")
    m.subject_to(u * v >= -0.5, name="hyp")
    return m


def test_report_type_is_public():
    assert discopt.ConvexityReport is type(_xlogx().convexity())


@pytest.mark.parametrize("build", [_xlogx, _xlogx_intrinsic])
def test_entropy_witnesses_are_convex(build):
    rep = build().convexity()
    assert rep.is_convex is True
    assert rep.objective_convex is True
    assert bool(rep)


def test_product_spelling_reports_the_rewrite():
    assert _xlogx().convexity().rewrites == ("entropy",)
    assert _xlogx_intrinsic().convexity().rewrites == ()


def test_epigraph_rewrite_is_what_proves_convexity():
    m = _epigraph()
    # The bare classifier, on the model as written, does not prove it.
    assert classify_model(m, use_certificate=True)[0] is False
    rep = m.convexity()
    assert rep.is_convex is True
    assert rep.rewrites == ("objective_epigraph",)
    assert rep.constraints == (("defn", True), ("lin", True))
    # The caller's model is not rewritten.
    assert m._constraints[0].sense == "=="


def test_nonconvex_constraint_is_named():
    rep = _bilinear().convexity()
    assert rep.is_convex is False
    assert rep.objective_convex is False
    assert rep.nonconvex_constraints() == ["hyp"]
    assert not bool(rep)


@pytest.mark.parametrize("build", [_xlogx, _xlogx_intrinsic, _epigraph, _bilinear])
def test_verdict_matches_the_solve(build, monkeypatch):
    """The report equals the classification ``solve_model`` dispatches on."""
    import discopt.solver as solver

    seen: list[bool] = []
    real = solver._classify_model_convexity

    def spy(model, **kw):
        out = real(model, **kw)
        seen.append(out[1])
        return out

    monkeypatch.setattr(solver, "_classify_model_convexity", spy)
    m = build()
    expected = m.convexity().is_convex
    m.solve(time_limit=20)
    assert seen, "solve never classified convexity"
    assert seen[0] is expected


def test_budget_exhaustion_is_reported_as_unknown():
    rep = _bilinear().convexity(time_limit=0.0)
    assert rep.is_convex is None
    assert rep.objective_convex is None
    assert rep.constraints == ()
    assert not bool(rep)
