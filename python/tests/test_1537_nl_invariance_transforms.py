"""#1537: the ``.nl`` changes of representation the invariance panel applies must be
exact -- otherwise a "false certificate" the panel reports is the transform's.

Each transformed model is parsed back with ``from_nl`` and evaluated at the mapped
point: the objective must match (times ``k`` for an objective scale) and every
row's distance to each finite side must match (times ``k`` for a row scale).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from discopt.modeling.core import from_nl
from discopt.validation.nl_invariance import scale_objective_nl, scale_rows_nl, translate_nl

DATA = Path(__file__).parent / "data" / "minlplib_nl"
# Linear, quadratic, signomial, n-ary sums (o54 count lines), integer and binary.
INSTANCES = ["ex1221", "nvs03", "st_e13", "4stufen", "gkocis", "tls2", "st_miqp1"]


def _eval(model, x):
    from discopt._tape_nlp_evaluator import make_evaluator
    from discopt.solver import _infer_constraint_bounds

    ev = make_evaluator(model)
    cl, cu = (np.asarray(a, dtype=float) for a in _infer_constraint_bounds(model, ev))
    c = np.asarray(ev.evaluate_constraints(x), dtype=float)
    return float(ev.evaluate_objective(x)), c - cl, cu - c


def _point(model):
    lb = np.concatenate([np.atleast_1d(v.lb).ravel() for v in model._variables])
    ub = np.concatenate([np.atleast_1d(v.ub).ravel() for v in model._variables])
    x = np.random.default_rng(0).uniform(-2.0, 2.0, lb.size)
    return np.clip(x, np.maximum(lb, -10.0), np.minimum(ub, 10.0))


@pytest.mark.parametrize("name", INSTANCES)
@pytest.mark.parametrize(
    "kind,arg", [("shift", 1000.0), ("shift", 1e6), ("rows", 1e3), ("rows", 1e-3), ("obj", 1e3)]
)
def test_transform_is_exact(tmp_path, name, kind, arg):
    src = DATA / f"{name}.nl"
    text = src.read_text(encoding="latin-1")
    fn = {"shift": translate_nl, "rows": scale_rows_nl, "obj": scale_objective_nl}[kind]
    out = tmp_path / f"{name}_{kind}.nl"
    out.write_text(fn(text, arg), encoding="latin-1")
    m0, m1 = from_nl(str(src)), from_nl(str(out))
    assert [str(v.var_type) for v in m1._variables] == [str(v.var_type) for v in m0._variables]
    x0 = _point(m0)
    is_bin = np.concatenate(
        [np.full(v.size, str(v.var_type).endswith("BINARY")) for v in m0._variables]
    )
    t = np.where(is_bin, 0.0, arg)  # a linear binary is not moved (positional type)
    x1 = x0 + t if kind == "shift" else x0
    f0, lo0, hi0 = _eval(m0, x0)
    f1, lo1, hi1 = _eval(m1, x1)
    fk = arg if kind == "obj" else 1.0
    rk = arg if kind == "rows" else 1.0
    # A shift by 1e6 legitimately costs ~ulp(1e6) per variable reference.
    tol = 1e-6 if kind != "shift" else 1e-9 * arg * (1.0 + abs(f0))
    assert f1 == pytest.approx(fk * f0, rel=1e-6, abs=tol)
    compared = 0
    for a0, a1 in ((lo0, lo1), (hi0, hi1)):
        fin = np.isfinite(a0) & (np.abs(a0) < 1e15)
        assert np.allclose(a1[fin], rk * a0[fin], rtol=1e-6, atol=max(tol, 1e-6 * rk))
        compared += int(fin.sum())
    assert compared > 0, "no finite row side was compared"
    if kind == "shift":
        lb0 = np.concatenate([np.atleast_1d(v.lb).ravel() for v in m0._variables])
        lb1 = np.concatenate([np.atleast_1d(v.lb).ravel() for v in m1._variables])
        fin = np.abs(lb0) < 1e15
        assert np.allclose(lb1[fin], lb0[fin] + t[fin])
        assert np.any(t[fin] != 0.0), "nothing was shifted"


def test_refusals_are_loud():
    text = (DATA / "ex1221.nl").read_text(encoding="latin-1")
    with pytest.raises(NotImplementedError, match="non-integral shift"):
        translate_nl(text, 0.5)  # ex1221 has binaries
    with pytest.raises(ValueError, match="positive"):
        scale_rows_nl(text, -2.0)
    with pytest.raises(NotImplementedError, match="ASCII"):
        translate_nl("b3 0 1 0\n" + "\n".join(["0"] * 9), 1.0)
