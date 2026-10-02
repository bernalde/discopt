"""Scalar directed (outward) rounding for Python-side bound tightening (#1537 D).

The Python counterpart of ``crates/discopt-core/src/presolve/directed.rs`` (#1504),
with the same contract. Each ``*_down`` / ``*_up`` returns the round-to-nearest
result when that result is exact, and the adjacent float in the outward direction
when it is not. That is true directed rounding: integer data, ``0`` and ``2 * 0.5``
are left untouched, so bounds on well-scaled models do not move, and every inexact
endpoint lands on the safe side of the exact value.

The rounding error is recovered exactly by error-free transforms: TwoSum for
``+``/``-``, and for ``*``, ``/`` and ``sqrt`` the product remainder ``a*b - fl(a*b)``.
Python 3.12 has no ``math.fma``, so the remainder is computed with Dekker's
TwoProduct (Veltkamp split), which is exact when nothing overflows or underflows;
outside that range the exactness test is skipped and the result is stepped one ulp
outward unconditionally. Stepping is always sound -- round-to-nearest is within
half an ulp of the exact value -- it is only one ulp looser.

Why it is needed (measured, #1537 D): ``(arg_lb - offset) / coeff`` rounded to
nearest, then ``ceil(lb - 1e-9)``, cut the only feasible integer of
``R2 <= exp(a*x + b) <= R1`` at ``x ~ 4e8`` -- one rounding ulp of the
4e8-magnitude quotient is 6e-8, far above the 1e-9 integrality slack -- and the
solve returned a certified ``infeasible``.

libm results (``log``, ``exp``, ``**``) are not correctly rounded; :func:`lib_down`
/ :func:`lib_up` widen them by :data:`LIB_ULPS` ulps plus one, as ``directed.rs``
does.
"""

from __future__ import annotations

import math
import sys

#: Relative widening, in units of machine epsilon, applied to libm results
#: (matches ``LIB_ULPS`` in ``directed.rs``).
LIB_ULPS = 4.0

#: Below this magnitude a product remainder may underflow and stop being exact,
#: so the exactness test is skipped and the result widened unconditionally.
_TINY = 1.0e-290
#: Above this magnitude the Veltkamp split (or the product it feeds) may overflow.
_HUGE = 1.0e290
_EPS = sys.float_info.epsilon
_SPLIT = 134217729.0  # 2**27 + 1
_INF = math.inf
_FMA = getattr(math, "fma", None)


def next_up(x: float) -> float:
    """The next float above ``x``."""
    return math.nextafter(x, _INF)


def next_down(x: float) -> float:
    """The next float below ``x``."""
    return math.nextafter(x, -_INF)


def _two_sum_err(a: float, b: float, s: float) -> float:
    """Exact rounding error of ``s = fl(a + b)`` (TwoSum): ``a + b = s + err``."""
    bb = s - a
    return (a - (s - bb)) + (b - bb)


def _split(a: float) -> tuple[float, float]:
    c = _SPLIT * a
    hi = c - (c - a)
    return hi, a - hi


def _prod_err(a: float, b: float, p: float) -> float | None:
    """Exact ``a*b - p`` for ``p = fl(a*b)``, or ``None`` when it may not be exact."""
    if not (_TINY <= abs(p) <= _HUGE and abs(a) <= _HUGE and abs(b) <= _HUGE):
        return None
    if _FMA is not None:
        return float(_FMA(a, b, -p))
    ah, al = _split(a)
    bh, bl = _split(b)
    return ((ah * bh - p) + ah * bl + al * bh) + al * bl


def add_down(a: float, b: float) -> float:
    """``a + b`` rounded toward ``-inf``."""
    s = a + b
    if not math.isfinite(s):
        return s
    return next_down(s) if _two_sum_err(a, b, s) < 0.0 else s


def add_up(a: float, b: float) -> float:
    """``a + b`` rounded toward ``+inf``."""
    s = a + b
    if not math.isfinite(s):
        return s
    return next_up(s) if _two_sum_err(a, b, s) > 0.0 else s


def sub_down(a: float, b: float) -> float:
    """``a - b`` rounded toward ``-inf``."""
    return add_down(a, -b)


def sub_up(a: float, b: float) -> float:
    """``a - b`` rounded toward ``+inf``."""
    return add_up(a, -b)


def mul_down(a: float, b: float) -> float:
    """``a * b`` rounded toward ``-inf`` (a NaN, e.g. ``0 * inf``, stays NaN)."""
    p = a * b
    if not math.isfinite(p) or a == 0.0 or b == 0.0:
        return p
    err = _prod_err(a, b, p)
    if err is None or err < 0.0:
        return next_down(p)
    return p


def mul_up(a: float, b: float) -> float:
    """``a * b`` rounded toward ``+inf`` (a NaN, e.g. ``0 * inf``, stays NaN)."""
    p = a * b
    if not math.isfinite(p) or a == 0.0 or b == 0.0:
        return p
    err = _prod_err(a, b, p)
    if err is None or err > 0.0:
        return next_up(p)
    return p


def _div_err_sign(a: float, b: float, q: float) -> int | None:
    """Sign of ``a/b - fl(a/b)``; ``None`` when not decidable exactly.

    ``r = a - q*b`` is exactly representable for ``q = RN(a/b)``; with
    ``q*b = p + e`` exactly, ``a - p`` is exact (Sterbenz: ``p`` is within a factor
    two of ``a``), so ``r = (a - p) - e`` is computed exactly, and ``a/b - q = r/b``.
    """
    if abs(q) < _TINY or abs(a) < _TINY or abs(b) < _TINY:
        return None
    p = q * b
    e = _prod_err(q, b, p)
    if e is None:
        return None
    r = (a - p) - e
    if r == 0.0:
        return 0
    return 1 if (r > 0.0) == (b > 0.0) else -1


def div_down(a: float, b: float) -> float:
    """``a / b`` rounded toward ``-inf`` (``b != 0``)."""
    q = a / b
    if not math.isfinite(q) or a == 0.0 or math.isinf(b):
        return q
    s = _div_err_sign(a, b, q)
    return q if s is not None and s >= 0 else next_down(q)


def div_up(a: float, b: float) -> float:
    """``a / b`` rounded toward ``+inf`` (``b != 0``)."""
    q = a / b
    if not math.isfinite(q) or a == 0.0 or math.isinf(b):
        return q
    s = _div_err_sign(a, b, q)
    return q if s is not None and s <= 0 else next_up(q)


def lib_down(v: float) -> float:
    """Widen a libm result down by :data:`LIB_ULPS` ulps plus one (non-finite passes)."""
    if not math.isfinite(v):
        return v
    return next_down(v - LIB_ULPS * _EPS * abs(v))


def lib_up(v: float) -> float:
    """Widen a libm result up by :data:`LIB_ULPS` ulps plus one (non-finite passes)."""
    if not math.isfinite(v):
        return v
    return next_up(v + LIB_ULPS * _EPS * abs(v))


def affine_preimage(coeff: float, offset: float, target: float, *, lower: bool) -> float:
    """Directed bound on ``(target - offset) / coeff``.

    A lower bound when ``lower`` is true, an upper bound otherwise. The caller picks
    the side from where the deduction lands: for ``coeff > 0``, ``arg >= L`` gives
    ``x >= (L - offset) / coeff`` (a lower bound); for ``coeff < 0`` the same
    deduction gives an upper bound.
    """
    if coeff > 0.0:
        # The quotient increases with the numerator.
        num = sub_down(target, offset) if lower else sub_up(target, offset)
    else:
        # The quotient decreases with the numerator.
        num = sub_up(target, offset) if lower else sub_down(target, offset)
    return div_down(num, coeff) if lower else div_up(num, coeff)
