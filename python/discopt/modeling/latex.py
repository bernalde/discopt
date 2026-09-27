"""Rich rendering of :class:`~discopt.modeling.core.Model` objects.

Produces LaTeX (and MathJax-backed HTML) in **standard PSE problem form**::

    minimize/maximize   f(x)
    subject to          g_i(x)  <=/>=/=  c_i
                        bounds and integrality on the variables

The output is meant to be pasted into a paper, so it follows the usual
typographic conventions of mathematical writing rather than echoing the DAG:

* implicit multiplication with the numeric coefficient first (``2 x y``, not
  ``x \\cdot 2 \\cdot y``) and signs folded (``x - 2y``, not ``x + -2 \\cdot y``);
* standard operator names (``\\arcsin``, ``\\log_{2}``, ``\\operatorname{erf}``,
  ``\\max\\{\\cdot\\}``, ``\\lVert x \\rVert_{2}``, ``\\prod_{i} x_{i}``) and the
  textbook form of composite atoms (``\\ln(1 + x)`` for ``log1p``,
  ``x \\ln x`` for ``xlogx``);
* scientific notation as ``2.5 \\times 10^{7}``, never ``2.5e+07``;
* identifiers typeset as symbols: ``alpha`` is ``\\alpha``, ``flow_in`` is
  ``\\mathit{flow}_{\\mathrm{in}}``, ``x3`` is ``x_{3}``;
* indexed sums recovered symbolically where the terms share one structure
  (``\\sum_{i=0}^{11} v_{i}^{2}``) and a whole-array sum as ``\\sum_{i} x_{i}``;
* constraints read with the variables on the left (``x + 5y \\ge 4``, not
  ``4 \\le x + 5y``) and vector domains as ``x \\in [0, 10]^{3}``.

Everything emitted compiles under ``pdflatex`` with ``amsmath``/``amssymb`` and
typesets under MathJax 3 (``base`` + ``ams``); ``test_model_latex.py`` checks
both engines over a corpus of models when they are installed.

Every transformation is display-only and exact: a symbolic sum is emitted only
when *every* term matches the template at the stated index, otherwise the terms
are written out. Large models are summarised so an auto-rendered ``_repr_``
cannot flood a notebook; the public ``model_to_latex`` / ``model_to_html`` take
``max_rows=None`` for the full, untruncated form.

Imported lazily by ``core.Model`` (the ``to_latex``/``_repr_latex_`` methods) to
avoid a circular import.
"""

from __future__ import annotations

import contextlib
import contextvars
import math
import re
from fractions import Fraction
from typing import Any, Iterable, Optional

import numpy as np

from discopt.modeling.core import (
    _ELEMENTWISE_FUNCS,
    BinaryOp,
    Constant,
    CustomCall,
    Expression,
    FunctionCall,
    IndexExpression,
    MatMulExpression,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
    _known_shape,
)

# Default number of constraint / variable rows shown in an auto ``_repr_``.
_DEFAULT_MAX_ROWS = 24

# Precedence levels (higher binds tighter) for minimal parenthesisation.
_P_ADD = 1  # a + b, a - b, unary minus
_P_MUL = 2  # juxtaposed products, \circ, matrix products, big operators
_P_FRAC = 3  # \frac: safe inside a product, not as a power base
_P_POW = 5  # a power base must bind at least this tightly
_P_ATOM = 9

# A written-out sum (one that is not a recovered \sum) longer than this is shown
# as `t_1 + t_2 + t_3 + \cdots + t_n`: no page fits a 3000-term row, and the
# elision is visible, never silent.
_EXPLICIT_SUM_MAX_TERMS = 12
_EXPLICIT_SUM_HEAD = 3

# Magnitude at or above which a bound is treated as infinite (the default
# continuous bound is +/-9.999e19).
_BIG = 1e15

# A +/- chain of at least this many same-structure terms reads as a \sum.
_CHAIN_SUM_MIN_TERMS = 4

# Index symbols for recovered sums, in order of preference.
_INDEX_SYMBOLS = ("i", "j", "k", "l", "p", "q", "r", "s", "t")

# ---------------------------------------------------------------------- numbers

# Significant figures for printed decimals, or ``None`` (the default) for the
# shortest decimal that round-trips to the exact float. Exact is the default
# because rounding is not cosmetic: measured 2026-09-27 over the 66 in-repo
# MINLPLib instances, 6 significant figures changed the value of 82 of 11 294
# evaluated rows beyond 1e-4 relative (cancellation, e.g. tspn05's
# `0.444444 x_0^2 - 61.7778 x_0 <= -2193.51`), i.e. the display stated a
# different constraint than the model. Rounding is an explicit opt-in.
_PRECISION: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "discopt_latex_precision", default=None
)


@contextlib.contextmanager
def _numbers(precision: Optional[int]):
    if precision is not None and (
        isinstance(precision, bool) or not isinstance(precision, int) or precision < 1
    ):
        raise ValueError(f"precision must be a positive int or None, got {precision!r}")
    token = _PRECISION.set(precision)
    try:
        yield
    finally:
        _PRECISION.reset(token)


def _fmt_num(v: float) -> str:
    """Format a scalar: integers exactly, decimals exactly (shortest round-trip
    form) or to ``precision`` significant figures when one is set, and
    scientific notation as ``m \\times 10^{e}``."""
    f = float(v)
    if not np.isfinite(f):  # int(inf) raises OverflowError (L6)
        if np.isnan(f):
            return r"\mathrm{nan}"
        return r"\infty" if f > 0 else r"-\infty"
    if f == int(f) and abs(f) < 1e15:
        s = str(int(f))
        digits = s.lstrip("-")
        # Round numbers with six or more trailing zeros read better in
        # scientific notation (25000000 -> 2.5 \times 10^{7}); the mantissa is
        # exact because only zeros are dropped.
        stripped = digits.rstrip("0")
        if len(digits) - len(stripped) >= 6:
            exp = len(digits) - 1
            mant = stripped[0] + ("." + stripped[1:] if len(stripped) > 1 else "")
            return _sci(s.startswith("-"), mant, exp)
        return s
    precision = _PRECISION.get()
    s = repr(f) if precision is None else f"{f:.{precision}g}"
    if "e" in s:
        mant, exp_s = s.split("e")
        neg = mant.startswith("-")
        return _sci(neg, mant.lstrip("-"), int(exp_s))
    return s


def _sci(neg: bool, mant: str, exp: int) -> str:
    body = rf"10^{{{exp}}}" if mant == "1" else rf"{mant} \times 10^{{{exp}}}"
    return ("-" if neg else "") + body


def _exponent_num(v: float) -> str:
    """An exponent: small exact rationals as ``p/q`` (``x^{1/3}``)."""
    f = float(v)
    if np.isfinite(f) and f != int(f):
        fr = Fraction(f).limit_denominator(12)
        if float(fr) == f:
            return f"{fr.numerator}/{fr.denominator}"
    return _fmt_num(f)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating))


def _scalar_value(e: Any) -> Optional[float]:
    """The value of a scalar numeric constant, else ``None``."""
    if _is_number(e):
        return float(e)
    if isinstance(e, Constant):
        arr = np.asarray(e.value)
        if arr.ndim == 0:
            try:
                return float(arr)
            except (TypeError, ValueError):
                return None
    return None


def _const_to_latex(value: Any) -> str:
    """Render a constant: scalars as numbers, small vectors/matrices inline, a
    uniform array as ``c\\,\\mathbf{1}``, and larger arrays as a shape-annotated
    bold placeholder (the model does not name numpy coefficients, so a shaped
    placeholder is the most honest compact form -- use a ``Parameter`` to give
    data a symbol)."""
    arr = np.asarray(value)
    if arr.ndim == 0:
        f = float(arr)
        if f == math.pi:
            return r"\pi"
        if f == math.e:
            return r"\mathrm{e}"
        return _fmt_num(f)
    if arr.size <= 6 and arr.ndim == 1:
        return r"\begin{bmatrix}" + r" \\ ".join(_fmt_num(x) for x in arr) + r"\end{bmatrix}"
    if arr.size <= 6 and arr.ndim == 2:
        return (
            r"\begin{bmatrix}"
            + r" \\ ".join(" & ".join(_fmt_num(x) for x in row) for row in arr)
            + r"\end{bmatrix}"
        )
    if arr.size and np.all(arr == arr.reshape(-1)[0]):
        c = float(arr.reshape(-1)[0])
        return r"\mathbf{1}" if c == 1.0 else rf"{_fmt_num(c)}\,\mathbf{{1}}"
    shape = r"{\times}".join(str(d) for d in arr.shape)
    return rf"\mathbf{{C}}_{{[{shape}]}}"


# ------------------------------------------------------------------ identifiers

_GREEK = frozenset(
    "alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa "
    "lambda mu nu xi pi varpi rho varrho sigma varsigma tau upsilon phi varphi chi "
    "psi omega Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega".split()
)
_IDENT = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*\Z")


def _math_escape(s: str) -> str:
    """Escape arbitrary text for math mode (LaTeX specials as math atoms)."""
    return "".join(_MATH_ATOMS.get(ch, r"\;" if ch == " " else ch) for ch in s)


def _name_parts(name: str) -> tuple[str, list[str]]:
    """Split an identifier into its typeset head and subscript parts.

    ``x`` -> (``x``, []), ``x3`` -> (``x``, [``3``]), ``flow_in`` ->
    (``\\mathit{flow}``, [``\\mathrm{in}``]), ``theta_1`` -> (``\\theta``, [``1``]).
    Anything that is not a plain identifier is typeset whole, escaped.
    """
    name = str(name)
    if not _IDENT.match(name):
        return rf"\mathit{{{_math_escape(name)}}}", []
    head, *subs = name.split("_")
    if not subs:
        m = re.fullmatch(r"([A-Za-z]+)(\d+)", head)
        if m:
            head, subs = m.group(1), [m.group(2)]
    return _head_tex(head), [_sub_tex(s) for s in subs]


def _head_tex(h: str) -> str:
    if h in _GREEK:
        return "\\" + h
    if len(h) == 1:
        return h
    return rf"\mathit{{{h}}}"


def _sub_tex(s: str) -> str:
    if s.isdigit() or len(s) == 1:
        return s
    if s in _GREEK:
        return "\\" + s
    return rf"\mathrm{{{s}}}"


def _sym(name: str, extra: Iterable[str] = ()) -> str:
    """Typeset an identifier with optional extra subscripts (element indices),
    merged into ONE subscript group so ``y1[0]`` is ``y_{1,0}``, never the
    invalid double subscript ``y_{1}_{0}`` (L5)."""
    head, subs = _name_parts(name)
    subs = subs + [str(x) for x in extra]
    return head + (f"_{{{','.join(subs)}}}" if subs else "")


def _is_wordy(tex: str) -> bool:
    """A factor led by a multi-letter name. Math mode ignores source spaces, so
    juxtaposing one with another factor needs an explicit thin space, or
    ``price_i x_i`` would set as the single word ``price_ix_i``."""
    return tex.startswith((r"\mathit", r"\operatorname"))


_TALL = (r"\frac", r"\sum", r"\prod", r"\begin", r"\sqrt", r"\left", r"\dfrac")


def _delim(inner: str, left: str = "(", right: str = ")") -> str:
    """Wrap in delimiters, auto-sized only when the content is tall."""
    if any(t in inner for t in _TALL):
        return rf"\left{left}{inner}\right{right}"
    return f"{left}{inner}{right}"


def _paren(inner: str, own_prec: int, parent_prec: int) -> str:
    """Parenthesise *inner* when a tighter-binding parent needs it."""
    return _delim(inner) if own_prec < parent_prec else inner


# ------------------------------------------------------------ display-only nodes


class _SymIndex:
    """Display-only element reference ``base[idx]`` whose index components may
    be symbolic strings (``"i"``, ``"i+1"``). Built by the sum-recovery and
    element-wise rewrites below; never reaches the solver."""

    _shape: tuple = ()

    def __init__(self, base: Any, index: tuple):
        self.base = base
        self.index = tuple(index)


def _norm_index(idx: Any) -> Optional[tuple]:
    """An index as a tuple of ints / symbolic strings, or ``None`` if it is
    anything else (a slice, a fancy index, ...)."""
    comps = idx if isinstance(idx, tuple) else (idx,)
    out: list[int | str] = []
    for c in comps:
        if isinstance(c, (bool, np.bool_)):
            return None
        if isinstance(c, (int, np.integer)):
            out.append(int(c))
        elif isinstance(c, str):
            out.append(c)
        else:
            return None
    return tuple(out)


def _leaf_base(node: Any) -> Any:
    """The Variable/Parameter an element reference points into, else ``None``."""
    base = getattr(node, "base", None)
    return base if isinstance(base, (Variable, Parameter)) else None


# -------------------------------------------------------------- sum recovery


def _signature(node: Any, slots: list) -> Any:
    """Structural signature of *node*, with every element reference's index
    abstracted into *slots* (in traversal order). Two terms with equal
    signatures differ only in those indices."""
    if isinstance(node, (IndexExpression, _SymIndex)) and _leaf_base(node) is not None:
        idx = _norm_index(node.index)
        if idx is not None:
            slots.append(idx)
            return ("ix", id(node.base), len(idx))
        return ("node", id(node))
    if isinstance(node, Constant):
        arr = np.asarray(node.value)
        return ("c", arr.shape, arr.tobytes())
    if isinstance(node, BinaryOp):
        return ("b", node.op, _signature(node.left, slots), _signature(node.right, slots))
    if isinstance(node, UnaryOp):
        return ("u", node.op, _signature(node.operand, slots))
    if isinstance(node, FunctionCall):
        return ("f", node.func_name, tuple(_signature(a, slots) for a in node.args))
    if isinstance(node, SumOverExpression):
        return ("so", tuple(_signature(t, slots) for t in node.terms))
    if isinstance(node, SumExpression):
        return ("s", node.axis, _signature(node.operand, slots))
    if isinstance(node, MatMulExpression):
        return ("mm", _signature(node.left, slots), _signature(node.right, slots))
    return ("node", id(node))


def _rebuild(node: Any, it: Any) -> Any:
    """Copy *node*, replacing each abstracted element reference (same traversal
    order as :func:`_signature`) with the next index from *it*."""
    if isinstance(node, (IndexExpression, _SymIndex)) and _leaf_base(node) is not None:
        if _norm_index(node.index) is not None:
            return _SymIndex(node.base, next(it))
        return node
    if isinstance(node, BinaryOp):
        left = _rebuild(node.left, it)
        return BinaryOp(node.op, left, _rebuild(node.right, it))
    if isinstance(node, UnaryOp):
        return UnaryOp(node.op, _rebuild(node.operand, it))
    if isinstance(node, FunctionCall):
        return FunctionCall(node.func_name, *[_rebuild(a, it) for a in node.args])
    if isinstance(node, SumOverExpression):
        return SumOverExpression([_rebuild(t, it) for t in node.terms])
    if isinstance(node, SumExpression):
        return SumExpression(_rebuild(node.operand, it), axis=node.axis)
    if isinstance(node, MatMulExpression):
        left = _rebuild(node.left, it)
        return MatMulExpression(left, _rebuild(node.right, it))
    return node


def _offset(sym: str, c: tuple) -> str:
    """Index text for ``c = (+1, k)`` (``i + k``) or ``c = (-1, k)`` (``k - i``)."""
    slope, k = c
    if slope < 0:
        return f"{k}-{sym}"
    return sym if k == 0 else (f"{sym}+{k}" if k > 0 else f"{sym}-{-k}")


def _recover_sum(terms: list, sym: str) -> Optional[tuple[int, int, Any]]:
    """Recognise ``terms`` as ``f(lo), f(lo+1), ..., f(hi)`` for ONE template f.

    Returns ``(lo, hi, template)`` or ``None``. Exact by construction: every
    term must have the same structure, and every index component that varies
    across terms must equal ``i + c`` for the one summation index ``i`` (which
    must itself run over consecutive integers) and a fixed offset ``c``.
    Differing constants (unnamed data) defeat recovery; name data with a
    ``Parameter`` to have it indexed.
    """
    if len(terms) < 2:
        return None
    all_slots: list[list] = []
    sig0 = None
    for t in terms:
        slots: list = []
        sig = _signature(t, slots)
        if sig0 is None:
            sig0 = sig
        elif sig != sig0:
            return None
        all_slots.append(slots)
    # Columns: one per (slot, component); a column varies if any term differs.
    n_slots = len(all_slots[0])
    driver: Optional[list[int]] = None
    plan: list[list] = []  # per slot: per component, None (fixed) or offset
    for s in range(n_slots):
        comps: list[Optional[tuple[int, int]]] = []
        for d in range(len(all_slots[0][s])):
            col = [slots[s][d] for slots in all_slots]
            if all(v == col[0] for v in col):
                comps.append(None)
                continue
            if not all(isinstance(v, int) for v in col):
                return None
            if driver is None:
                if any(col[k] != col[0] + k for k in range(len(col))):
                    return None
                driver = col
            # Either i + k (same direction) or k - i (reversed, a convolution).
            diffs = {col[k] - driver[k] for k in range(len(col))}
            sums = {col[k] + driver[k] for k in range(len(col))}
            if len(diffs) == 1:
                comps.append((1, diffs.pop()))
            elif len(sums) == 1:
                comps.append((-1, sums.pop()))
            else:
                return None
        plan.append(comps)
    if driver is None:
        return None
    lo = driver[0]
    new_idx = []
    for s, comps in enumerate(plan):
        base_idx = all_slots[0][s]
        new_idx.append(
            tuple(base_idx[d] if c is None else _offset(sym, c) for d, c in enumerate(comps))
        )
    template = _rebuild(terms[0], iter(new_idx))
    # Sound-or-refuse: re-instantiate the template at every index value and
    # require it to reproduce the original term exactly (same structure, same
    # leaves, same constants, same indices). Any mismatch means the recovery
    # logic above is wrong for this input, so write the terms out instead.
    for k, term in enumerate(terms):
        want: list = []
        got: list = []
        if _signature(_instantiate(template, sym, lo + k), got) != _signature(term, want):
            return None
        if got != want:
            return None
    return lo, lo + len(terms) - 1, template


def _instantiate(template: Any, sym: str, value: int) -> Any:
    """Substitute ``sym = value`` into a recovered template's indices."""
    fwd = re.compile(rf"{re.escape(sym)}([+-]\d+)?\Z")
    rev = re.compile(rf"(-?\d+)-{re.escape(sym)}\Z")

    def comp(c: Any) -> Any:
        if isinstance(c, str):
            m = fwd.match(c)
            if m:
                return value + int(m.group(1) or 0)
            m = rev.match(c)
            if m:
                return int(m.group(1)) - value
        return c

    slots: list = []
    _signature(template, slots)
    return _rebuild(template, iter(tuple(comp(c) for c in idx) for idx in slots))


def _elementize(node: Any, shape: tuple, syms: tuple) -> Any:
    """Rewrite an element-wise expression over arrays of ``shape`` into its
    generic element (``price * flow`` -> ``price_i flow_i``), or ``None`` when
    that is not exact (a non-element-wise node, a broadcast, unnamed
    non-uniform data)."""
    s = _known_shape(node)
    if (
        s == ()
        and not isinstance(node, (Variable, Parameter))
        or (isinstance(node, (Variable, Parameter)) and tuple(node.shape) == ())
    ):
        return node
    if isinstance(node, (Variable, Parameter)):
        return _SymIndex(node, syms) if tuple(node.shape) == shape else None
    if isinstance(node, Constant):
        arr = np.asarray(node.value)
        if arr.size and np.all(arr == arr.reshape(-1)[0]):
            return Constant(float(arr.reshape(-1)[0]))
        return None
    if s is None or tuple(s) != shape:
        return None
    if isinstance(node, BinaryOp):
        left = _elementize(node.left, shape, syms)
        right = _elementize(node.right, shape, syms)
        if left is None or right is None:
            return None
        return BinaryOp(node.op, left, right)
    if isinstance(node, UnaryOp):
        inner = _elementize(node.operand, shape, syms)
        return None if inner is None else UnaryOp(node.op, inner)
    if isinstance(node, FunctionCall) and node.func_name in _ELEMENTWISE_FUNCS:
        args = [_elementize(a, shape, syms) for a in node.args]
        return None if any(a is None for a in args) else FunctionCall(node.func_name, *args)
    return None


def _name_symbols(name: str) -> set[str]:
    """Every letter group an identifier typesets as (``x_i_j`` -> x, i, j;
    ``q12`` -> q), so a recovered index symbol never reads as part of a name."""
    parts = str(name).split("_")
    out = {p for p in parts[1:] if p}
    out.add(parts[0].rstrip("0123456789"))
    return out


def _leaf_names(node: Any) -> set[str]:
    """Identifier symbols under *node*. Iterative and DAG-aware: expression
    graphs share subtrees and can be thousands of nodes deep."""
    out: set[str] = set()
    stack = list(node) if isinstance(node, (list, tuple)) else [node]
    seen: set[int] = set()
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        if isinstance(n, (Variable, Parameter)) and isinstance(getattr(n, "name", None), str):
            out |= _name_symbols(n.name)
        for attr in ("left", "right", "operand", "base"):
            child = getattr(n, attr, None)
            if isinstance(child, (Expression, _SymIndex)):
                stack.append(child)
        stack.extend(getattr(n, "args", ()) or ())
        stack.extend(getattr(n, "terms", ()) or ())
    return out


# ------------------------------------------------------------------ renderer

_FUNC_OPS = {
    "exp": r"\exp",
    "log": r"\ln",
    "log2": r"\log_{2}",
    "log10": r"\log_{10}",
    "sin": r"\sin",
    "cos": r"\cos",
    "tan": r"\tan",
    "sinh": r"\sinh",
    "cosh": r"\cosh",
    "tanh": r"\tanh",
    "asin": r"\arcsin",
    "acos": r"\arccos",
    "atan": r"\arctan",
    "asinh": r"\operatorname{arsinh}",
    "acosh": r"\operatorname{arcosh}",
    "atanh": r"\operatorname{artanh}",
    "erf": r"\operatorname{erf}",
    "sign": r"\operatorname{sgn}",
    "atan2": r"\operatorname{atan2}",
}
# Functions whose integer powers are written ``\sin^{2}(x)``.
_TRIG = frozenset({"sin", "cos", "tan", "sinh", "cosh", "tanh"})
_NORM_RE = re.compile(r"norm(inf|fro|[0-9.]+)\Z")


class _Renderer:
    """Expression-DAG -> LaTeX visitor. ``bound`` holds the index symbols in
    scope (enclosing sums) plus identifiers they must not collide with."""

    def __init__(self, reserved: Iterable[str] = ()):
        self.bound: set[str] = set(reserved)

    # -- entry -----------------------------------------------------------

    def render(self, e: Any, parent: int = 0) -> str:
        neg, mag, prec = self.signed(e)
        if neg:
            return _paren(f"-{mag}", _P_ADD, parent)
        return _paren(mag, prec, parent)

    def signed(self, e: Any) -> tuple[bool, str, int]:
        """``(negative, magnitude_tex, magnitude_precedence)`` -- the sign is
        split off so sums can fold it into their operator (``a - 2b``)."""
        v = _scalar_value(e)
        if v is not None and np.isfinite(v) and v < 0 and isinstance(e, (Constant, float, int)):
            return True, _const_to_latex(-v), _P_ATOM
        if isinstance(e, UnaryOp) and e.op == "neg":
            neg, mag, prec = self.signed(e.operand)
            if neg:
                return False, mag, prec
            if prec < _P_MUL:
                return True, _delim(mag), _P_ATOM
            return True, mag, prec
        if isinstance(e, BinaryOp) and e.op == "*" and not self._hadamard(e):
            return self._product(e)
        if isinstance(e, SumOverExpression):
            return self._sum_over(e)
        if isinstance(e, BinaryOp) and e.op == "/":
            nneg, nmag, _ = self.signed(e.left)
            return nneg, rf"\frac{{{nmag}}}{{{self.render(e.right)}}}", _P_FRAC
        return False, *self._unsigned(e)

    # -- nodes ------------------------------------------------------------

    def _unsigned(self, e: Any) -> tuple[str, int]:
        if isinstance(e, Constant):
            v = _scalar_value(e)
            return _const_to_latex(e.value), (_P_ATOM if v is None or v >= 0 else _P_ADD)
        if _is_number(e):
            return _fmt_num(e), (_P_ATOM if e >= 0 else _P_ADD)
        if isinstance(e, (Variable, Parameter)):
            return _sym(e.name), _P_ATOM
        if isinstance(e, (IndexExpression, _SymIndex)):
            return self._index(e), _P_ATOM
        if isinstance(e, BinaryOp):
            if e.op in ("+", "-"):
                return self._sum_chain(e), _P_ADD
            if e.op == "*":  # Hadamard (shaped operands)
                return (
                    rf"{self.render(e.left, _P_MUL + 1)} \circ {self.render(e.right, _P_MUL + 1)}",
                    _P_MUL,
                )
            if e.op == "**":
                return self._power(e)
            return self._fallback(e), _P_ATOM
        if isinstance(e, UnaryOp):
            if e.op == "abs":
                return _delim(self.render(e.operand), "|", "|"), _P_ATOM
            return rf"\operatorname{{{_math_escape(e.op)}}}{_delim(self.render(e.operand))}", (
                _P_ATOM
            )
        if isinstance(e, FunctionCall):
            return self._call(e)
        if isinstance(e, CustomCall):
            args = ", ".join(self.render(a) for a in e.args)
            return rf"\operatorname{{{_math_escape(str(e.name))}}}{_delim(args)}", _P_ATOM
        if isinstance(e, MatMulExpression):
            # A 1-D left operand of `@` acts as a row vector: `x^T y`, `x^T A`.
            ls = _known_shape(e.left)
            row = ls is not None and len(ls) == 1
            left = self.render(e.left, _P_POW if row else _P_MUL)
            right = self.render(e.right, _P_MUL + 1)
            if row:
                return rf"{left}^{{\top}} {right}", _P_MUL
            return rf"{left}\,{right}", _P_MUL
        if isinstance(e, SumExpression):
            return self._array_sum(e)
        if isinstance(e, SumOverExpression):
            neg, mag, prec = self._sum_over(e)
            return (f"-{mag}", _P_ADD) if neg else (mag, prec)
        if isinstance(e, Expression):
            return self._fallback(e), _P_ATOM
        return _latex_text(str(e)), _P_ATOM

    def _fallback(self, e: Any) -> str:
        return _latex_text(repr(e))

    def _hadamard(self, e: Any) -> bool:
        ls, rs = _known_shape(e.left), _known_shape(e.right)
        return bool(ls) and bool(rs)

    def _index(self, e: Any) -> str:
        comps = e.index if isinstance(e.index, tuple) else (e.index,)
        base = e.base
        shape = _known_shape(base)
        parts = [
            self._index_part(c, shape[k] if shape and k < len(shape) else None)
            for k, c in enumerate(comps)
        ]
        if isinstance(base, (Variable, Parameter)):
            return _sym(base.name, parts)
        # Any other base is grouped so its own subscripts nest validly.
        inner = self.render(base, _P_POW)
        return f"{{{inner}}}_{{{','.join(parts)}}}"

    @staticmethod
    def _index_part(c: Any, dim: Optional[int]) -> str:
        if isinstance(c, (int, np.integer)) and not isinstance(c, bool):
            return str(int(c))
        if isinstance(c, str):
            return c
        if isinstance(c, slice):
            if c == slice(None):
                return r"\cdot"
            if dim is not None:
                idx = list(range(dim))[c]
                if not idx:
                    return r"\emptyset"
                if len(idx) <= 3:
                    return r"\{" + ",".join(map(str, idx)) + r"\}"
                step = idx[1] - idx[0]
                second = f"{idx[1]}," if step != 1 else ""
                return rf"\{{{idx[0]},{second}\ldots,{idx[-1]}\}}"
            return _math_escape(
                ":".join("" if x is None else str(x) for x in (c.start, c.stop, c.step))
            )
        if c is Ellipsis:
            return r"\ldots"
        arr = np.asarray(c)
        if arr.ndim == 1 and arr.dtype.kind in "iu":
            vals = [str(int(x)) for x in arr]
            if len(vals) > 5:
                vals = vals[:3] + [r"\ldots", vals[-1]]
            return r"\{" + ",".join(vals) + r"\}"
        return _math_escape(str(c))

    def _product(self, e: Any) -> tuple[bool, str, int]:
        factors: list = []
        stack = [e]
        while stack:  # iterative, left-to-right
            n = stack.pop()
            if isinstance(n, BinaryOp) and n.op == "*" and not self._hadamard(n):
                stack.append(n.right)
                stack.append(n.left)
            else:
                factors.append(n)
        coeff = 1.0
        rest = []
        for f in factors:
            v = _scalar_value(f)
            if v is not None and np.isfinite(v) and f is not None and not _known_shape(f):
                coeff *= v
            elif isinstance(f, UnaryOp) and f.op == "neg":
                coeff = -coeff
                rest.append(f.operand)
            else:
                rest.append(f)
        neg = coeff < 0
        coeff = abs(coeff)
        if not rest:
            return neg, _fmt_num(coeff), _P_ATOM
        texts = []
        for k, f in enumerate(rest):
            # A leading big operator (\sum) would swallow the factors after it.
            is_big = isinstance(f, (SumOverExpression, SumExpression))
            if is_big:
                prec = _P_POW if k < len(rest) - 1 else _P_MUL
            else:
                prec = _P_MUL + 1
            fneg, fmag, fprec = self.signed(f)
            if fneg:
                texts.append(_delim(f"-{fmag}"))
            else:
                texts.append(_paren(fmag, fprec, prec))
        out = "" if coeff == 1.0 else _fmt_num(coeff)
        prev = out
        for t in texts:
            if not out:
                out = prev = t
                continue
            # `\cdot` where juxtaposition would misread: two numbers (`2 3`),
            # or a number before a numeric fraction (`2\frac{1}{3}` reads as a
            # mixed number).
            if (
                t[:1].isdigit()
                or t[:1] in "-."
                or (prev[-1:].isdigit() and t.startswith(r"\frac{") and t[6:7].isdigit())
            ):
                sep = r" \cdot "
            elif _is_wordy(t) or _is_wordy(prev):
                sep = r"\,"
            else:
                sep = " "
            out = f"{out}{sep}{t}"
            prev = t
        return neg, out, _P_MUL

    def _terms(self, e: Any, sign: bool, out: list) -> None:
        """Flatten a +/- chain into signed terms. The right operand of ``-`` is
        only flattened through a sign flip, so ``a - (b - c)`` keeps its
        grouping as ``a - (b - c)``."""
        # Walk the left spine iteratively: `x0 + x1 + ... + xn` built with
        # Python's builtin sum() is n levels deep, beyond the recursion limit.
        spine = []
        node = e
        while isinstance(node, BinaryOp) and node.op in ("+", "-"):
            spine.append(node)
            node = node.left
        out.append((sign, node, False))
        for parent in reversed(spine):
            right = parent.right
            nested = isinstance(right, BinaryOp) and right.op in ("+", "-")
            if parent.op == "+" and nested:
                self._terms(right, sign, out)
            else:
                out.append(
                    (sign if parent.op == "+" else not sign, right, nested and parent.op == "-")
                )

    def _sum_chain(self, e: Any) -> str:
        terms: list = []
        self._terms(e, False, terms)
        return self._join(terms)

    def _join(self, terms: list) -> str:
        """Join ``(negated, node, group)`` terms with folded signs. A long
        chain of same-sign terms that share one structure (``x_0 + x_1 + ...``
        from Python's builtin ``sum``) is recovered as ``\\sum``, exactly as
        for ``dm.sum`` (see :func:`_recover_sum`)."""
        # An additive zero (builtin ``sum`` starts from ``0``) is dropped.
        nonzero = [t for t in terms if t[2] or _scalar_value(t[1]) != 0.0]
        terms = nonzero or terms[:1]
        if len(terms) >= _CHAIN_SUM_MIN_TERMS and all(not s and not g for s, _, g in terms):
            recovered = self._recovered([n for _, n, _ in terms])
            if recovered is not None:
                neg, tex, _ = recovered
                return f"-{tex}" if neg else tex
        if len(terms) > _EXPLICIT_SUM_MAX_TERMS:
            head = self._join(terms[:_EXPLICIT_SUM_HEAD])
            tail = self._join([(False, Constant(1.0), False), terms[-1]])
            # `tail` is "1 + t" or "1 - t"; keep its operator and term.
            return f"{head} + \\cdots {tail[2:]}"
        out = ""
        for k, (sign, node, group) in enumerate(terms):
            if group:
                neg, mag = False, _delim(self._sum_chain(node))
            else:
                neg, mag, prec = self.signed(node)
                mag = _paren(mag, prec, _P_ADD + 1) if prec <= _P_ADD else mag
            neg = neg != sign
            if k == 0:
                out = f"-{mag}" if neg else mag
            else:
                out += f" - {mag}" if neg else f" + {mag}"
        return out or "0"

    def _power(self, e: Any) -> tuple[str, int]:
        base, ex = e.left, e.right
        ev = _scalar_value(ex)
        if ev is not None and ev == 0.5:
            return rf"\sqrt{{{self.render(base)}}}", _P_ATOM
        if (
            isinstance(base, FunctionCall)
            and base.func_name in _TRIG
            and ev is not None
            and ev == int(ev)
            and ev > 0
            and len(base.args) == 1
        ):
            arg = self.render(base.args[0])
            return rf"{_FUNC_OPS[base.func_name]}^{{{int(ev)}}}{_delim(arg)}", _P_ATOM
        exp_tex = _exponent_num(ev) if ev is not None else self.render(ex)
        return f"{self.render(base, _P_POW)}^{{{exp_tex}}}", _P_POW

    def _call(self, e: Any) -> tuple[str, int]:
        name = e.func_name
        args = list(e.args)
        if name == "sqrt" and len(args) == 1:
            return rf"\sqrt{{{self.render(args[0])}}}", _P_ATOM
        if name == "abs" and len(args) == 1:
            return _delim(self.render(args[0]), "|", "|"), _P_ATOM
        if name in ("min", "max"):
            flat: list = []

            def gather(n: Any) -> None:
                if isinstance(n, FunctionCall) and n.func_name == name:
                    for a in n.args:
                        gather(a)
                else:
                    flat.append(n)

            gather(e)
            inner = ", ".join(self.render(a) for a in flat)
            return rf"\{name}" + _delim(inner, r"\{", r"\}"), _P_ATOM
        if name == "log1p" and len(args) == 1:
            one_plus = self._join([(False, Constant(1.0), False), (False, args[0], False)])
            return rf"\ln{_delim(one_plus)}", _P_ATOM
        if name == "softplus" and len(args) == 1:
            return rf"\ln{_delim(r'1 + \mathrm{e}^{' + self.render(args[0]) + '}')}", _P_ATOM
        if name == "sigmoid" and len(args) == 1:
            neg, mag, prec = self.signed(args[0])
            mag = _paren(mag, prec, _P_ADD + 1)
            ex = mag if neg else f"-{mag}"
            return rf"\frac{{1}}{{1 + \mathrm{{e}}^{{{ex}}}}}", _P_FRAC
        if name == "entropy" and len(args) == 1:
            a = args[0]
            return rf"{self.render(a, _P_MUL + 1)} \ln{self._fn_arg(a)}", _P_MUL
        if name == "centropy" and len(args) == 2:
            a, b = args
            frac = rf"\frac{{{self.render(a)}}}{{{self.render(b)}}}"
            return rf"{self.render(a, _P_MUL + 1)} \ln\left({frac}\right)", _P_MUL
        if name == "signpower" and len(args) == 2:
            a = self.render(args[0])
            abs_a = _delim(a, "|", "|")
            return rf"\operatorname{{sgn}}{_delim(a)}\,{abs_a}^{{{self.render(args[1])}}}", _P_MUL
        if name == "prod" and len(args) == 1:
            big = self._big_op(r"\prod", args[0])
            if big is not None:
                return big, _P_MUL
        m = _NORM_RE.match(name)
        if m and len(args) == 1:
            p = {"inf": r"\infty", "fro": "F"}.get(m.group(1), m.group(1))
            inner = self.render(args[0])
            return _delim(f" {inner} ", r"\lVert", r"\rVert") + f"_{{{p}}}", _P_ATOM
        op = _FUNC_OPS.get(name) or rf"\operatorname{{{_math_escape(str(name))}}}"
        if len(args) == 1:
            return f"{op}{self._fn_arg(args[0])}", _P_ATOM
        return f"{op}{_delim(', '.join(self.render(a) for a in args))}", _P_ATOM

    def _fn_arg(self, a: Any) -> str:
        return _delim(self.render(a))

    # -- sums -------------------------------------------------------------

    def _fresh(self, n: int, node: Any) -> Optional[tuple[str, ...]]:
        avoid = self.bound | _leaf_names(node)
        free = [s for s in _INDEX_SYMBOLS if s not in avoid]
        return tuple(free[:n]) if len(free) >= n else None

    def _big_op(self, op: str, operand: Any, axis: Optional[int] = None) -> Optional[str]:
        """``\\sum_{i} f_i`` / ``\\prod_{i} f_i`` for an element-wise array
        expression, or ``None`` when its generic element is not exact."""
        shape = _known_shape(operand)
        if not shape:
            return None
        shape = tuple(shape)
        syms = self._fresh(len(shape), operand)
        if syms is None:
            return None
        if axis is not None:
            if not isinstance(axis, (int, np.integer)) or isinstance(axis, bool):
                return None
            ax = int(axis) % len(shape)
            comps = tuple(syms[k] if k == ax else r"\cdot" for k in range(len(shape)))
            below = syms[ax]
        else:
            comps = syms
            below = ",".join(syms)
        elem = _elementize(operand, shape, comps)
        if elem is None:
            return None
        self.bound |= set(syms)
        try:
            body = self.render(elem, _P_MUL)
        finally:
            self.bound -= set(syms)
        return rf"{op}_{{{below}}} {body}"

    def _recovered(self, terms: list) -> Optional[tuple[bool, str, int]]:
        """``\\sum_{i=lo}^{hi} f(i)`` for *terms*, or ``None`` if they are not
        one template at consecutive indices."""
        syms = self._fresh(1, terms)
        if syms is None:
            return None
        (sym,) = syms
        rec = _recover_sum(terms, sym)
        if rec is None:
            return None
        lo, hi, template = rec
        self.bound.add(sym)
        try:
            neg, mag, prec = self.signed(template)
        finally:
            self.bound.discard(sym)
        body = _paren(mag, prec, _P_MUL)
        return neg, rf"\sum_{{{sym}={lo}}}^{{{hi}}} {body}", _P_MUL

    def _array_sum(self, e: Any) -> tuple[str, int]:
        big = self._big_op(r"\sum", e.operand, e.axis)
        if big is not None:
            return big, _P_MUL
        inner = self.render(e.operand)
        sub = "" if e.axis is None else rf"_{{\text{{axis }} {_math_escape(str(e.axis))}}}"
        return rf"\operatorname{{sum}}{sub}{_delim(inner)}", _P_ATOM

    def _sum_over(self, e: Any) -> tuple[bool, str, int]:
        """``(negative, tex, precedence)``: a recovered sum whose every term is
        negated reads ``-\\sum_{i} v_{i}``, not ``\\sum_{i} (-v_{i})``."""
        terms = list(getattr(e, "terms", []) or [])
        if not terms:
            return False, "0", _P_ATOM
        if len(terms) == 1:
            return self.signed(terms[0])
        recovered = self._recovered(terms)
        if recovered is not None:
            return recovered
        # Write the terms out; nested explicit sums flatten (associativity).
        flat: list = []

        def gather(n: Any) -> None:
            if isinstance(n, SumOverExpression) and _recover_sum(list(n.terms), "i") is None:
                for t in n.terms:
                    gather(t)
            else:
                flat.append(n)

        for t in terms:
            gather(t)
        return False, self._join([(False, t, False) for t in flat]), _P_ADD


def expr_to_latex(
    e: Any,
    parent_prec: int = 0,
    reserved: Iterable[str] = (),
    precision: Optional[int] = None,
) -> str:
    """Render an expression DAG node to LaTeX. Never raises on an unusual node --
    unknown nodes fall back to an escaped string form so a display can't crash a
    model build (an invalid ``precision`` does raise).

    ``reserved`` names identifiers that recovered summation indices must avoid
    (``model_to_latex`` passes every variable and parameter name).
    ``precision`` rounds printed decimals to that many significant figures;
    the default ``None`` prints each number exactly."""
    with _numbers(precision):
        return _expr_to_latex(e, parent_prec, reserved)


def _expr_to_latex(e: Any, parent_prec: int, reserved: Iterable[str]) -> str:
    try:
        return _Renderer(reserved).render(e, parent_prec)
    except Exception:
        try:
            return _latex_text(repr(e))
        except Exception:
            # Never an empty display: say that something is not shown.
            return _latex_text(f"[{type(e).__name__}: not renderable]")


# ---------------------------------------------------------------- constraints

_SENSE = {"<=": r"\le", ">=": r"\ge", "==": "="}
_FLIP = {"<=": ">=", ">=": "<=", "==": "=="}


def _is_const(n: Any) -> bool:
    return isinstance(n, Constant) or _is_number(n)


def _constraint_to_latex(c: Any, reserved: Iterable[str] = ()) -> str:
    """Render a constraint. Constraints are normalised to ``body sense rhs``
    (``rhs`` usually 0); this un-normalises to the natural reading: a top-level
    ``L - R`` becomes ``L sense R``, a constant ``L`` flips so the variables
    read on the left (``x + 5y >= 4``, not ``4 <= x + 5y``), and constant terms
    of a sum move to the right-hand side (``x_0 = 4``, not ``x_0 - 4 = 0``)."""
    # Non-arithmetic constraint types (indicator / disjunctive / SOS / logical,
    # from if_then/either_or/m.logical) carry no .body/.sense — render a readable
    # placeholder instead of raising AttributeError and crashing the whole model
    # display / Jupyter _repr_ (L1).
    if not (hasattr(c, "sense") and hasattr(c, "body")):
        kind = type(c).__name__.lstrip("_")
        label = getattr(c, "name", None)
        text = f"{kind}: {label}" if label else kind
        # Route through the shared escaper rather than hand-rolling one: a
        # constraint name is arbitrary user text, and `_` was the only special it
        # handled — and handled the way MathJax prints literally (_MATH_ATOMS).
        return _latex_text(f"[{text}]")
    try:
        return _relation_to_latex(c, reserved)
    except Exception:
        return _latex_text(repr(c))


def _relation_to_latex(c: Any, reserved: Iterable[str]) -> str:
    sense = c.sense
    body = c.body
    rhs_val = getattr(c, "rhs", 0.0)
    r = _Renderer(reserved)
    rhs_is_zero = _is_number(rhs_val) and float(rhs_val) == 0.0
    if isinstance(body, BinaryOp) and body.op == "-" and rhs_is_zero and not _is_const(body.right):
        left, right = body.left, body.right
        if _is_const(left):
            left, right, sense = right, left, _FLIP.get(sense, sense)
        return f"{r.render(left)} {_SENSE.get(sense, sense)} {r.render(right)}"
    # Move constant terms of a top-level sum to the right-hand side.
    terms: list = []
    r._terms(body, False, terms)
    const_terms = [(s, n) for s, n, g in terms if not g and _is_const(n)]
    var_terms = [t for t in terms if t[2] or not _is_const(t[1])]
    if const_terms and var_terms:
        # body = V + C with C the signed constants, so `V + C sense rhs` reads
        # `V sense rhs - C`.
        new_rhs = np.asarray(rhs_val, dtype=float)
        for negated, n in const_terms:
            v = np.asarray(n.value if isinstance(n, Constant) else n, dtype=float)
            new_rhs = new_rhs + v if negated else new_rhs - v
        rhs_tex = r.render(Constant(new_rhs))
        return f"{r._join(var_terms)} {_SENSE.get(sense, sense)} {rhs_tex}"
    rhs_tex = "0" if rhs_is_zero else r.render(Constant(np.asarray(rhs_val, dtype=float)))
    return f"{r.render(body)} {_SENSE.get(sense, sense)} {rhs_tex}"


def _linear_row_to_latex(row: Any) -> str:
    """Render a builder-resident linear row (``export._common.LinearRow``) as
    ``sum(coeff * var[idx]) sense rhs`` (L2).

    These rows come from the fast-construction API (``Model.constraint`` fast path /
    ``add_linear_constraints``); they live only in ``model._builder_linear_blocks``
    and never reach ``model._constraints``. Rendering them keeps the display honest.
    """
    sense = _SENSE.get(row.sense, row.sense)
    terms = getattr(row, "terms", None)
    pieces: list[str] = []
    for var, local, coeff in terms or ():
        # A vector variable's element uses its flat local index as a subscript;
        # a scalar carries no index.
        sym = _sym(var.name) if getattr(var, "size", 1) <= 1 else _sym(var.name, [str(local)])
        mag = abs(float(coeff))
        term = sym if mag == 1.0 else rf"{_fmt_num(mag)}{r'\,' if _is_wordy(sym) else ' '}{sym}"
        if coeff < 0:
            pieces.append(f"- {term}" if pieces else f"-{term}")
        else:
            pieces.append(f"+ {term}" if pieces else term)
    body = " ".join(pieces) if pieces else "0"
    return f"{body} {sense} {_fmt_num(row.rhs)}"


# ------------------------------------------------------------------ variables


def _dims_tex(shape: tuple) -> str:
    return r" \times ".join(str(d) for d in shape)


def _vec_tex(vals: np.ndarray) -> str:
    return "(" + ", ".join(_fmt_num(x) for x in vals) + r")^{\top}"


def _var_domain_to_latex(v: Any) -> str:
    vtype = v.var_type.value if hasattr(v.var_type, "value") else str(v.var_type)
    lb_arr = np.atleast_1d(np.asarray(v.lb, dtype=float)).reshape(-1)
    ub_arr = np.atleast_1d(np.asarray(v.ub, dtype=float)).reshape(-1)
    shape = tuple(getattr(v, "shape", ()) or ())
    name = _sym(v.name)
    lb, ub = float(np.min(lb_arr)), float(np.max(ub_arr))
    has_lb, has_ub = lb > -_BIG, ub < _BIG

    if int(np.prod(shape)) <= 1 or not shape:
        if vtype == "binary":
            return rf"{name} \in \{{0, 1\}}"
        if has_lb and has_ub:
            rng = rf"{_fmt_num(lb)} \le {name} \le {_fmt_num(ub)}"
        elif has_lb:
            rng = rf"{name} \ge {_fmt_num(lb)}"
        elif has_ub:
            rng = rf"{name} \le {_fmt_num(ub)}"
        else:
            rng = ""
        if vtype == "integer":
            return rf"{rng},\; {name} \in \mathbb{{Z}}" if rng else rf"{name} \in \mathbb{{Z}}"
        return rng or rf"{name} \in \mathbb{{R}}"

    dims = _dims_tex(shape)
    if vtype == "binary":
        return rf"{name} \in \{{0, 1\}}^{{{dims}}}"
    uniform = bool(np.all(lb_arr == lb_arr[0]) and np.all(ub_arr == ub_arr[0]))
    if uniform:
        if vtype == "integer":
            if has_lb and has_ub and ub - lb <= 1:
                vals = ", ".join(_fmt_num(x) for x in np.arange(lb, ub + 1))
                return rf"{name} \in \{{{vals}\}}^{{{dims}}}"
            if has_lb and has_ub:
                return rf"{name} \in \{{{_fmt_num(lb)}, \ldots, {_fmt_num(ub)}\}}^{{{dims}}}"
            dom = rf"{name} \in \mathbb{{Z}}^{{{dims}}}"
        else:
            if has_lb and has_ub:
                return rf"{name} \in [{_fmt_num(lb)}, {_fmt_num(ub)}]^{{{dims}}}"
            if has_lb:
                return rf"{name} \in [{_fmt_num(lb)}, \infty)^{{{dims}}}"
            if has_ub:
                return rf"{name} \in (-\infty, {_fmt_num(ub)}]^{{{dims}}}"
            return rf"{name} \in \mathbb{{R}}^{{{dims}}}"
        if has_lb:
            return rf"{dom},\; {name} \ge {_fmt_num(lb)}"
        if has_ub:
            return rf"{dom},\; {name} \le {_fmt_num(ub)}"
        return dom

    # Element-wise bounds that differ: list them when short, else state the
    # enclosing box and say the bounds vary (never claim a box it isn't).
    space = r"\mathbb{Z}" if vtype == "integer" else r"\mathbb{R}"
    dom = rf"{name} \in {space}^{{{dims}}}"
    if len(shape) == 1 and lb_arr.size <= 6:

        def side(arr: np.ndarray, finite: np.ndarray) -> Optional[str]:
            if not finite.any():
                return None
            if not finite.all():
                return None
            return _fmt_num(arr[0]) if np.all(arr == arr[0]) else _vec_tex(arr)

        lo = side(lb_arr, lb_arr > -_BIG)
        hi = side(ub_arr, ub_arr < _BIG)
        if (lo is not None or not (lb_arr > -_BIG).any()) and (
            hi is not None or not (ub_arr < _BIG).any()
        ):
            rel = name
            if lo is not None:
                rel = rf"{lo} \le {rel}"
            if hi is not None:
                rel = rf"{rel} \le {hi}"
            return rf"{dom},\; {rel}"
    parts = []
    if has_lb:
        parts.append(rf"{_fmt_num(lb)} \le {name}")
    if has_ub:
        parts.append(rf"{name} \le {_fmt_num(ub)}")
    rng = r",\; ".join(parts)
    note = r"\;\text{(bounds vary by element)}"
    return rf"{dom},\; {rng}{note}" if rng else dom


# ---------------------------------------------------------------------- model


def _all_constraint_rows(model: Any) -> tuple[list, list]:
    """Return ``(expression_constraint_rows, builder_linear_rows)`` for *model*.

    The expression rows are the *full* ``model._constraints`` list — including the
    non-arithmetic GDP/indicator/SOS/logical constraint objects that render as an
    L1 placeholder (``iter_all_rows`` filters those out, but the display must keep
    showing them). The builder rows come from the shared X-1 primitive
    ``export._common.iter_all_rows`` so builder-resident fast-path rows appear too
    (L2). The import is local to avoid a heavyweight import at module load and to
    keep the display resilient: if the export helper is unavailable or the model is
    malformed, fall back to the expression-path constraints alone rather than
    crashing a Jupyter ``_repr_``.
    """
    expr_rows = list(getattr(model, "_constraints", []) or [])
    try:
        from discopt.export._common import iter_builder_linear_rows

        builder_rows = list(iter_builder_linear_rows(model))
    except Exception:
        builder_rows = []
    return expr_rows, builder_rows


def _reserved_names(model: Any) -> set[str]:
    names = set()
    for v in list(getattr(model, "_variables", []) or []) + list(
        getattr(model, "_parameters", []) or []
    ):
        n = getattr(v, "name", None)
        if isinstance(n, str):
            names |= _name_symbols(n)
    return names


_PREAMBLE = "\\documentclass{article}\n\\usepackage{amsmath,amssymb}\n\\begin{document}\n"


def model_to_latex(
    model: Any,
    max_rows: int | None = None,
    env: str = "aligned",
    standalone: bool = False,
    precision: int | None = None,
) -> str:
    """Render *model* to a LaTeX ``aligned`` block in standard PSE form.

    ``max_rows`` caps the number of constraint and variable rows shown (``None`` =
    no limit); excess is replaced with a ``\\vdots`` summary row. With
    ``standalone=True`` the block is wrapped in a minimal ``article`` document
    (``amsmath`` + ``amssymb``) that compiles as-is with ``pdflatex``.
    ``precision`` rounds printed decimals to that many significant figures; the
    default ``None`` prints every number exactly, so the display is the model.
    """
    with _numbers(precision):
        return _model_to_latex(model, max_rows, env, standalone)


def _model_to_latex(model: Any, max_rows: int | None, env: str, standalone: bool) -> str:
    reserved = _reserved_names(model)
    rows: list[str] = []
    obj = getattr(model, "_objective", None)
    if obj is not None:
        sense = "minimize" if obj.sense.value == "minimize" else "maximize"
        objective = _expr_to_latex(obj.expression, 0, reserved)
        rows.append(rf"& \text{{{sense}}} \quad && {objective}")
    else:
        rows.append(r"& \text{find} \quad && x")

    # Route through the shared X-1 primitive so builder-resident fast-path rows
    # (Model.constraint fast path / add_linear_constraints) render too — reading only
    # `_constraints` would silently hide them and misrepresent the model (L2).
    expr_rows, builder_rows = _all_constraint_rows(model)
    renderers = [(lambda c: _constraint_to_latex(c, reserved), c) for c in expr_rows] + [
        (_linear_row_to_latex, r) for r in builder_rows
    ]
    n_cons = len(renderers)
    shown = renderers if max_rows is None else renderers[:max_rows]
    for i, (render, c) in enumerate(shown):
        lead = r"\text{subject to}" if i == 0 else ""
        rows.append(rf"& {lead} \quad && {render(c)}")
    if max_rows is not None and n_cons > max_rows:
        lead = r"\text{subject to}" if not shown else ""
        rows.append(rf"& {lead} \quad && \vdots \quad (\text{{{n_cons} constraints}})")

    variables = list(getattr(model, "_variables", []) or [])
    rows.extend(_variable_rows(variables, max_rows))
    body = " \\\\\n".join(rows)
    block = f"\\begin{{{env}}}\n{body}\n\\end{{{env}}}"
    if standalone:
        return f"{_PREAMBLE}\\[\n{block}\n\\]\n\\end{{document}}\n"
    return block


def _variable_rows(variables: list, max_rows: int | None) -> list[str]:
    if not variables:
        return []
    if max_rows is not None and len(variables) > max_rows:
        # Summarise by type rather than listing hundreds of declarations.
        counts: dict[str, int] = {}
        for v in variables:
            t = v.var_type.value if hasattr(v.var_type, "value") else str(v.var_type)
            counts[t] = counts.get(t, 0) + 1
        parts = ", ".join(f"{n}\\;\\text{{{t}}}" for t, n in counts.items())
        return [rf"& \text{{with}} \quad && {parts} \text{{ variables}}"]
    out = []
    for j, v in enumerate(variables):
        lead = r"\text{with}" if j == 0 else ""
        out.append(rf"& {lead} \quad && {_var_domain_to_latex(v)}")
    return out


def model_to_html(model: Any, max_rows: int | None = None, precision: int | None = None) -> str:
    """Standalone HTML rendering: a titled block with the PSE LaTeX rendered via the
    notebook's MathJax. Falls back gracefully to showing the LaTeX source."""
    name = getattr(model, "name", "model")
    n_v = len(getattr(model, "_variables", []) or [])
    # Count builder-resident fast-path rows too, so the header count matches the
    # rendered body and does not misreport a fast-API model as "0 constraints" (L2).
    expr_rows, builder_rows = _all_constraint_rows(model)
    n_c = len(expr_rows) + len(builder_rows)
    latex = model_to_latex(model, max_rows=max_rows, precision=precision)
    return (
        f'<div class="discopt-model">'
        f'<div style="font-weight:600">Model <code>{_escape_html(str(name))}</code>'
        f' <span style="color:#888;font-weight:400">'
        f"({n_v} variable{'s' if n_v != 1 else ''}, "
        f"{n_c} constraint{'s' if n_c != 1 else ''})</span></div>"
        f"$$\n{latex}\n$$"
        f"</div>"
    )


# Spacing: never emit the control space `\ `. MathJax 3 defines it as a *macro*
# (`\text{ }`), and expanding any macro concatenates its expansion with the
# entire rest of the formula, throwing "internal buffer size exceeded" once that
# is over 5120 characters -- so one `\ ` early in a large model fails the whole
# display. Measured 2026-09-27 on MINLPLib casctanks (45 KB, a `,\ ` in an
# integer-domain row). `\,` `\;` `\quad` are spacers, not macros, and are safe.

# LaTeX specials, rendered as MATH-mode atoms rather than escaped inside
# `\text{...}`. This output has two consumers that disagree about text mode:
# MathJax typesets it in the notebook and on the docs site, and `to_latex()` is
# documented as markup to paste into a paper. Inside `\text{}`, pdflatex REQUIRES
# `\_` while MathJax 3 does not implement it and prints the backslash literally —
# which is the `global\_opt` that shipped to the docs site. There is no text-mode
# spelling both engines accept.
#
# Math mode has one. Measured 2026-09-24 against pdflatex 3.141592653 and MathJax
# 3 in Chromium (scratchpad/dual_probe.py, 24 compiles x 24 renderings): each
# spelling below compiles under pdflatex AND renders as exactly its character
# under MathJax. `\hat{}` and `\tilde{}` are the accents with no base glyph, which
# is how both engines draw a bare caret and tilde.
_MATH_ATOMS = {
    "_": r"\_",
    "%": r"\%",
    "#": r"\#",
    "&": r"\&",
    "$": r"\$",
    "{": r"\{",
    "}": r"\}",
    "^": r"\hat{}",
    "~": r"\tilde{}",
    # The one character with no exact dual-target spelling: `\backslash` compiles
    # in both, but MathJax draws U+2216 SET MINUS, not U+005C. Every text-mode
    # alternative (`\textbackslash`, `\char92`) renders as its own source under
    # MathJax, which is strictly worse. Disclosed rather than silent.
    "\\": r"\backslash",
}


def _latex_text(s: str) -> str:
    """Render arbitrary prose safely into a math environment.

    Emits runs of ordinary characters as ``\\text{...}`` and every LaTeX special
    (``% _ # & $ { } \\ ~ ^``) as a math-mode atom between those runs, so an
    unknown-node repr or a stray string can neither break math mode nor print a
    literal backslash under MathJax (see ``_MATH_ATOMS``). Never injects HTML
    entities (``&amp;``) into a LaTeX ``aligned`` block (L7).
    """
    parts: list[str] = []
    run: list[str] = []
    for ch in s:
        atom = _MATH_ATOMS.get(ch)
        if atom is None:
            run.append(ch)
            continue
        if run:
            parts.append(rf"\text{{{''.join(run)}}}")
            run = []
        parts.append(atom)
    if run:
        parts.append(rf"\text{{{''.join(run)}}}")
    # An empty string still has to be a math-mode fragment, not nothing: callers
    # interpolate the result straight into an `aligned` row.
    return "".join(parts) or r"\text{}"


def _escape_html(s: str) -> str:
    """HTML-escape ``< > &`` for text placed in the HTML chrome (not math) (L7)."""
    return (
        s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if "<" in s or ">" in s or "&" in s
        else s
    )
