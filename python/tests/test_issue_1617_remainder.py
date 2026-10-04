"""Regression tests for the items of #1617 left open by PR #1643.

* B-07a -- ``Model.sensitivity`` refused a point that ``solve`` returned when the
  model was written with multiplied-out (badly scaled) rows, while the divided
  form of the same model differentiated without complaint. The KKT gates in
  ``modeling/argmin.py`` compared row slacks and multipliers in absolute units,
  so a row with gradient norm ~1e13 sitting 4e-11 (in x) from its bound was
  called inactive and its multiplier's pull on stationarity (~80) became the
  residual.
* The logic-layer spelling gaps noted under C-02: no ``xor`` and only
  ``dm.atleast`` / ``dm.atmost`` (against ``Model.at_least`` / ``at_most``).
"""

from __future__ import annotations

import itertools
import warnings

import discopt.modeling as dm
import numpy as np
import pytest

L_BEAM, E_MOD, D_MAX = 2000.0, 2e5, 10.0


def _beam(form: str):
    m = dm.Model(f"beam_{form}")
    P = m.parameter("P", 1e4)
    sig = m.parameter("sig", 160.0)
    b = m.continuous("b", lb=1, ub=500)
    h = m.continuous("h", lb=1, ub=2000)
    if form == "multiplied":
        m.subject_to(6 * P * L_BEAM <= sig * b * h**2)
        m.subject_to(4 * P * L_BEAM**3 <= E_MOD * D_MAX * b * h**3)
    else:
        m.subject_to(6 * P * L_BEAM / (b * h**2) <= sig)
        m.subject_to(4 * P * L_BEAM**3 / (E_MOD * D_MAX * b * h**3) <= 1.0)
    m.subject_to(h <= 4 * b)
    m.minimize(b * h)
    r = m.solve(solver="pounce", initial_solution={b: 50.0, h: 150.0})
    assert r.status in ("optimal", "local_optimal"), r.status
    return m, P, sig


def test_sensitivity_on_multiplied_out_rows_matches_divided_form():
    """B-07a: the two spellings of one model have the same dx*/dp."""
    m_div, P_d, s_d = _beam("divided")
    m_mul, P_m, s_m = _beam("multiplied")
    sens_div = m_div.sensitivity(wrt=[P_d, s_d])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # no KKT-gate refusal warning
        sens_mul = m_mul.sensitivity(wrt=[P_m, s_m])
    d_div = np.asarray(sens_div.dx_dp)
    d_mul = np.asarray(sens_mul.dx_dp)
    assert np.all(np.isfinite(d_mul))
    # The issue's divided-form reference: dx/dP = [9.94e-4, 3.98e-3].
    np.testing.assert_allclose(d_div[:, 0], [9.94e-4, 3.98e-3], rtol=2e-3)
    np.testing.assert_allclose(d_mul, d_div, rtol=1e-5, atol=1e-10)


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6, 1e12])
def test_sensitivity_is_invariant_to_row_scaling(scale):
    """Multiplying an active row by a constant changes nothing about dx*/dp."""
    m = dm.Model("scaled")
    p = m.parameter("p", 2.0)
    x = m.continuous("x", lb=-10, ub=10)
    y = m.continuous("y", lb=-10, ub=10)
    m.subject_to(scale * (x + y) >= scale * p)
    m.minimize(x**2 + 2 * y**2)
    r = m.solve(solver="pounce")
    # The 1e-6 row's QP route reports an uncertified "feasible" (#1596); the
    # sensitivity only needs the KKT point.
    assert r.status in ("optimal", "local_optimal", "feasible")
    sens = m.sensitivity(wrt=[p])
    # min x^2 + 2y^2 s.t. x + y = p  ->  x = 2p/3, y = p/3.
    np.testing.assert_allclose(np.asarray(sens.dx_dp).ravel(), [2 / 3, 1 / 3], rtol=1e-5)


def _solve_with_fixed(build, fixes):
    m, bools, obj = build()
    for bv, val in zip(bools, fixes):
        m.subject_to(bv.variable == val)
    m.minimize(obj(bools))
    return m.solve()


@pytest.mark.parametrize("spelling", ["function", "operator"])
def test_xor_lowers_to_its_truth_table(spelling):
    """``dm.xor(A, B)`` / ``A ^ B`` holds exactly when one of A, B is true."""

    def build():
        m = dm.Model("xor")
        A, B = m.boolean("A"), m.boolean("B")
        m.logical(dm.xor(A, B) if spelling == "function" else A ^ B)
        return m, (A, B), lambda bs: bs[0].variable + bs[1].variable

    checked = 0
    for a, b in itertools.product([0, 1], repeat=2):
        r = _solve_with_fixed(build, (a, b))
        feasible = r.status == "optimal"
        assert feasible == (a != b), (a, b, r.status)
        checked += 1
    assert checked == 4


def test_chained_xor_is_parity():
    def build():
        m = dm.Model("xor3")
        A, B, C = m.boolean("A"), m.boolean("B"), m.boolean("C")
        m.logical(A ^ B ^ C)
        return m, (A, B, C), lambda bs: bs[0].variable

    checked = 0
    for bits in itertools.product([0, 1], repeat=3):
        r = _solve_with_fixed(build, bits)
        assert (r.status == "optimal") == (sum(bits) % 2 == 1), (bits, r.status)
        checked += 1
    assert checked == 8


def test_xor_rejects_a_non_logical_operand():
    m = dm.Model("bad")
    A = m.boolean("A")
    x = m.binary("x")
    with pytest.raises(TypeError):
        dm.xor(A, x)


def test_logical_cardinality_accepts_the_model_method_spelling():
    assert dm.at_least is dm.atleast
    assert dm.at_most is dm.atmost
    m = dm.Model("card")
    A, B, C = m.boolean("A"), m.boolean("B"), m.boolean("C")
    m.logical(dm.at_least(2, A, B, C))
    m.logical(dm.at_most(2, A, B, C))
    m.minimize(A.variable + B.variable + C.variable)
    r = m.solve()
    assert r.status == "optimal"
    assert r.objective == pytest.approx(2.0)
