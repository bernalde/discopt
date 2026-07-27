"""E1 independent refresh reference (numpy, no JAX).

Pure closed-form ``refresh(box) -> row coefficients`` reimplementations of the
box-parametric envelope families the uniform relaxation emits. These are written
FROM THE ENVELOPE MATH (McCormick 1976 inequalities; secant/tangent line
constructions), NOT by copying the builder's dict-assembly, so that comparing their
output to the production rows is a genuine parity test: a wrong formula fails the
ulp bar. The Rust port (P1) is out of scope; E1 only establishes the mapping exists
and is exact.

Refresh inputs are the STATIC template (column identity, the atom stencil f/f' which
close over only atom identity / fixed exponent — never the box) plus the node
interval bounds produced by the interval/FBBT pass (``lo, hi`` for a 1-D atom; the
four factor endpoints for a product). Output is the list of ``(coeffs, rhs)`` rows
meaning ``sum_j coeffs[j]*col_j <= rhs`` — the exact contract of the builder.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

Row = tuple[dict[int, float], float]

_BIG = 1e19
_MIN_WIDTH = 1e-12


def _finite(*vals: float) -> bool:
    return all(math.isfinite(v) and abs(v) < _BIG for v in vals)


def _scaled_into(coeffs: dict[int, float], src: dict[int, float], s: float) -> None:
    """Accumulate ``s * src`` into ``coeffs`` (mirrors LinForm.scaled + fold)."""
    for j, c in src.items():
        cs = c * s
        coeffs[j] = coeffs.get(j, 0.0) + cs


def refresh_mccormick(
    w: int,
    a_coeffs: dict[int, float],
    a_const: float,
    b_coeffs: dict[int, float],
    b_const: float,
    a_lo: float,
    a_hi: float,
    b_lo: float,
    b_hi: float,
) -> list[Row]:
    """Four McCormick rows for ``w = A*B`` over ``A in [a_lo,a_hi]``, ``B in [b_lo,b_hi]``.

    The valid inequalities (McCormick 1976), each touching the bilinear graph:
        w >= a_lo*B + b_lo*A - a_lo*b_lo     w >= a_hi*B + b_hi*A - a_hi*b_hi   (under)
        w <= a_hi*B + b_lo*A - a_hi*b_lo     w <= a_lo*B + b_hi*A - a_lo*b_hi   (over)
    Written here as ``E = coef_a*A + coef_b*B + cc`` then converted to a ``<= 0``
    row: ``(-sign)*w + sign*E <= 0`` with sign=+1 for under, -1 for over.
    """
    rows: list[Row] = []
    # (a_endpoint, b_endpoint, sign): coef_a = b_endpoint, coef_b = a_endpoint,
    # cc = -a_endpoint*b_endpoint.
    for a_pt, b_pt, sign in (
        (a_lo, b_lo, +1.0),
        (a_hi, b_hi, +1.0),
        (a_hi, b_lo, -1.0),
        (a_lo, b_hi, -1.0),
    ):
        coef_a = b_pt
        coef_b = a_pt
        cc = -a_pt * b_pt
        if not _finite(coef_a, coef_b, cc):
            continue
        coeffs: dict[int, float] = {w: -sign}
        _scaled_into(coeffs, a_coeffs, sign * coef_a)
        _scaled_into(coeffs, b_coeffs, sign * coef_b)
        rhs = -sign * (cc + coef_a * a_const + coef_b * b_const)
        rows.append(({j: c for j, c in coeffs.items() if c != 0.0}, rhs))
    return rows


def _secant_row(
    w: int,
    lt_coeffs: dict[int, float],
    lt_const: float,
    lo: float,
    hi: float,
    f: Callable[[float], float],
    sign: float,
) -> Row | None:
    """``sign*w <= sign*chord(t)`` — the secant of ``f`` between the endpoint images."""
    if not _finite(lo, hi) or (hi - lo) < _MIN_WIDTH:
        return None
    flo, fhi = float(f(lo)), float(f(hi))
    if not _finite(flo, fhi):
        return None
    slope = (fhi - flo) / (hi - lo)
    a = flo - slope * lo
    coeffs: dict[int, float] = {w: sign}
    _scaled_into(coeffs, lt_coeffs, -sign * slope)
    return ({j: c for j, c in coeffs.items() if c != 0.0}, sign * (a + slope * lt_const))


def _tangent_row(
    w: int,
    lt_coeffs: dict[int, float],
    lt_const: float,
    t0: float,
    f: Callable[[float], float],
    fp: Callable[[float], float],
    sign: float,
) -> Row | None:
    """``sign*w >= sign*tangent(t)`` at ``t0`` — a supporting line of ``f``."""
    try:
        g, gp = float(f(t0)), float(fp(t0))
    except (ValueError, ArithmeticError):
        return None
    if not _finite(g, gp):
        return None
    intercept = g - gp * t0
    coeffs: dict[int, float] = {w: -sign}
    _scaled_into(coeffs, lt_coeffs, sign * gp)
    return (
        {j: c for j, c in coeffs.items() if c != 0.0},
        -sign * intercept - sign * gp * lt_const,
    )


def refresh_univariate(
    w: int,
    lt_coeffs: dict[int, float],
    lt_const: float,
    lo: float,
    hi: float,
    f: Callable[[float], float],
    fp: Callable[[float], float],
    curv: str | None,
) -> list[Row]:
    """Secant + three tangents of ``w = f(t)`` over ``t in [lo,hi]`` (convex/concave).

    Reproduces ``_emit_1d``: convex -> ``w <= secant`` and ``w >= tangent`` at
    (lo, mid, hi); concave -> the mirror. ``curv is None`` -> no rows (aux floor).
    """
    if curv is None or not _finite(lo, hi) or (hi - lo) < _MIN_WIDTH:
        return []
    # Mirror _emit_1d: the endpoint eval gates the ENTIRE envelope (secant AND
    # tangents) — a raise / non-finite there emits zero rows, not just no secant.
    try:
        flo, fhi = float(f(lo)), float(f(hi))
    except (ValueError, ArithmeticError):
        return []
    if not _finite(flo, fhi):
        return []
    sign = +1.0 if curv == "convex" else -1.0
    mid = 0.5 * (lo + hi)
    rows: list[Row] = []
    sec = _secant_row(w, lt_coeffs, lt_const, lo, hi, f, sign)
    if sec is not None:
        rows.append(sec)
    for t0 in (lo, mid, hi):
        tan = _tangent_row(w, lt_coeffs, lt_const, t0, f, fp, sign)
        if tan is not None:
            rows.append(tan)
    return rows


def refresh_obj_floor(
    obj_coeffs: dict[int, float],
    obj_const: float,
    sep_lb: float,
) -> list[Row]:
    """The separable objective-floor cut ``obj_lin >= sep_lb`` as a ``<= 0`` row.

    Coefficients are box-independent (the negated objective's variable part); only
    the RHS moves with the box, through ``sep_lb`` — a separable interval lower
    bound of the objective over the box (closed-form per term, no JAX). Here
    ``sep_lb`` is the interval/FBBT-layer input, exactly as the McCormick endpoints
    are; refresh reassembles the row.
    """
    coeffs = {j: -c for j, c in obj_coeffs.items()}
    return [({j: c for j, c in coeffs.items() if c != 0.0}, obj_const - sep_lb)]


def refresh_secant_only(
    w: int,
    lt_coeffs: dict[int, float],
    lt_const: float,
    lo: float,
    hi: float,
    f: Callable[[float], float],
    sign: float,
) -> list[Row]:
    """Single secant row (abs over-side, odd-power hull secant)."""
    sec = _secant_row(w, lt_coeffs, lt_const, lo, hi, f, sign)
    return [] if sec is None else [sec]
