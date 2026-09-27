"""Publication-quality LaTeX rendering of models and expressions.

Three layers of evidence, from strongest to weakest:

1. **Semantic round trip.** ``_TexEval`` is an independent evaluator for the
   LaTeX subset the renderer emits (implicit products, folded signs, ``\\frac``,
   powers, ``\\sqrt``, named functions, ``\\max\\{\\}``, recovered ``\\sum``).
   Randomised expression trees and indexed sums are rendered, the *text* is
   evaluated, and the value must match the expression DAG at random points.
   This is what proves a display-only rewrite (sign folding, constants moved
   across a relation, a recovered ``\\sum_{i=lo}^{hi}``) never misstates the
   model.
2. **Structural lint** over a corpus of models: balanced braces, paired
   ``\\left``/``\\right``, no double sub/superscripts, and every macro drawn from
   a whitelist that both pdflatex (amsmath/amssymb) and MathJax 3 implement.
3. **Engine checks** (skipped when the engine is absent): the corpus compiles
   under ``pdflatex -halt-on-error`` and typesets under MathJax 3 with
   undefined-macro errors fatal.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import subprocess
import tempfile

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling import latex
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    FunctionCall,
    IndexExpression,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
)

pytestmark = pytest.mark.smoke

# --------------------------------------------------------------------------
# An independent evaluator for the emitted LaTeX subset.
# --------------------------------------------------------------------------

_TOKEN = re.compile(r"\\[A-Za-z]+|\\.|\d+(?:\.\d+)?|[A-Za-z]|[{}^_()|+\-,=]|\S")

_FUNCS = {
    r"\exp": math.exp,
    r"\ln": math.log,
    r"\sin": math.sin,
    r"\cos": math.cos,
    r"\tan": math.tan,
    r"\sinh": math.sinh,
    r"\cosh": math.cosh,
    r"\tanh": math.tanh,
    r"\arcsin": math.asin,
    r"\arccos": math.acos,
    r"\arctan": math.atan,
}
_OPNAMES = {
    "erf": math.erf,
    "arsinh": math.asinh,
    "arcosh": math.acosh,
    "artanh": math.atanh,
    "sgn": lambda v: float(np.sign(v)),
}
_GREEK = {"\\" + g for g in latex._GREEK}


class _TexEval:
    """Parse a LaTeX string into a closure ``env -> float``.

    ``env`` maps a rendered element symbol (``x_{0}``, ``\\mathit{flow}_{\\mathrm{in},1}``)
    to its value. Bound summation indices are substituted into subscripts
    before lookup, so ``v_{i-1}`` under ``i = 3`` reads ``v_{2}``.
    """

    def __init__(self, text: str):
        self.toks = [t for t in _TOKEN.findall(text) if t not in (r"\,", r"\;", r"\ ", r"\!")]
        self.pos = 0
        self.abs_depth = 0

    # -- token helpers --
    def peek(self, k: int = 0):
        i = self.pos + k
        return self.toks[i] if i < len(self.toks) else None

    def take(self, want=None):
        t = self.peek()
        if want is not None and t != want:
            raise SyntaxError(f"expected {want!r}, got {t!r} at {self.pos}: {self.toks}")
        self.pos += 1
        return t

    def raw_group(self) -> str:
        self.take("{")
        depth, out = 1, []
        while True:
            t = self.take()
            if t == "{":
                depth += 1
            elif t == "}":
                depth -= 1
                if depth == 0:
                    return "".join(out)
            out.append(t)

    def group(self):
        self.take("{")
        e = self.expr()
        self.take("}")
        return e

    # -- grammar --
    def parse(self):
        e = self.expr()
        rel = self.peek()
        if rel in (r"\le", r"\ge", "="):
            self.take()
            rhs = self.expr()
            if self.peek() is not None:
                raise SyntaxError(f"trailing tokens {self.toks[self.pos :]}")
            return rel, e, rhs
        if self.peek() is not None:
            raise SyntaxError(f"trailing tokens {self.toks[self.pos :]}")
        return None, e, None

    def expr(self):
        terms = []
        sign = 1.0
        if self.peek() == "-":
            self.take()
            sign = -1.0
        terms.append((sign, self.term()))
        while self.peek() in ("+", "-"):
            s = 1.0 if self.take() == "+" else -1.0
            terms.append((s, self.term()))
        return lambda env: sum(s * f(env) for s, f in terms)

    _STOP = {"+", "-", ")", r"\right", "}", ",", "=", r"\le", r"\ge", r"\}", None}

    def starts_factor(self) -> bool:
        t = self.peek()
        if t in self._STOP:
            return False
        if t == "|":
            return self.abs_depth == 0
        return True

    def term(self):
        factors = [self.postfix()]
        while True:
            t = self.peek()
            if t in (r"\cdot", r"\circ"):
                self.take()
                factors.append(self.postfix())
            elif self.starts_factor():
                factors.append(self.postfix())
            else:
                break
        return lambda env: math.prod(f(env) for f in factors)

    def postfix(self):
        a = self.atom()
        while self.peek() == "^":
            self.take()
            window = self.toks[self.pos : self.pos + 6]
            m = re.fullmatch(r"\{(-?)(\d+)/(\d+)\}", "".join(window[:6]))
            m = m or re.fullmatch(r"\{(-?)(\d+)/(\d+)\}", "".join(window[:5]))
            if m:  # a rational exponent `p/q`
                self.pos += 6 if m.group(1) else 5
                q = (-1 if m.group(1) else 1) * int(m.group(2)) / int(m.group(3))
                b = (lambda q: lambda env: q)(q)
            else:
                b = self.group()
            a = (lambda a, b: lambda env: _real_pow(a(env), b(env)))(a, b)
        return a

    def fn_arg(self):
        if self.peek() == r"\left":
            self.take()
            self.take("(")
            e = self.expr()
            self.take(r"\right")
            self.take(")")
            return e
        self.take("(")
        e = self.expr()
        self.take(")")
        return e

    def symbol(self, head: str):
        sub = self.raw_group() if self.peek() == "_" and self.take("_") else None

        def look(env, head=head, sub=sub):
            if sub is None:
                return env[head]
            comps = []
            bound = env.get("__bound__", {})
            for c in sub.split(","):
                m = re.fullmatch(r"([a-z])([+-]\d+)?", c)
                r = re.fullmatch(r"(-?\d+)-([a-z])", c)
                if m and m.group(1) in bound:
                    c = str(bound[m.group(1)] + int(m.group(2) or 0))
                elif r and r.group(2) in bound:
                    c = str(int(r.group(1)) - bound[r.group(2)])
                comps.append(c)
            return env[f"{head}_{{{','.join(comps)}}}"]

        return look

    def big_op(self, combine):
        self.take("_")
        spec = self.raw_group()
        hi = None
        if self.peek() == "^":
            self.take()
            hi = int(self.raw_group())
        body = self.term()
        if "=" in spec:
            sym, lo = spec.split("=")
            syms, ranges = [sym], [range(int(lo), hi + 1)]
        else:
            syms, ranges = spec.split(","), None

        def run(env):
            def rec(k, bound):
                if k == len(syms):
                    return [body({**env, "__bound__": bound})]
                out = []
                if ranges:
                    for v in ranges[k]:
                        out += rec(k + 1, {**bound, syms[k]: v})
                    return out
                # Unbounded (`\sum_{i}`): run until the index leaves the array.
                v = 0
                while True:
                    try:
                        part = rec(k + 1, {**bound, syms[k]: v})
                    except KeyError:
                        if v == 0:
                            raise
                        return out
                    out += part
                    v += 1

            return combine(rec(0, dict(env.get("__bound__", {}))))

        return run

    def atom(self):
        t = self.take()
        if t is None:
            raise SyntaxError("unexpected end")
        if re.fullmatch(r"\d+(?:\.\d+)?", t):
            v = float(t)
            if self.peek() == r"\times":
                self.take()
                self.take("10")
                self.take("^")
                ex = self.raw_group()
                v = v * 10.0 ** int(ex)
            return lambda env, v=v: v
        if t == "(":
            e = self.expr()
            self.take(")")
            return e
        if t == "|":
            self.abs_depth += 1
            e = self.expr()
            self.take("|")
            self.abs_depth -= 1
            return lambda env: abs(e(env))
        if t == r"\left":
            d = self.take()
            if d == "(":
                e = self.expr()
                self.take(r"\right")
                self.take(")")
                return e
            if d == "|":
                self.abs_depth += 1
                e = self.expr()
                self.take(r"\right")
                self.take("|")
                self.abs_depth -= 1
                return lambda env: abs(e(env))
            raise SyntaxError(f"\\left{d}")
        if t == r"\frac":
            n, d = self.group(), self.group()
            return lambda env: n(env) / d(env)
        if t == r"\sqrt":
            a = self.group()
            return lambda env: math.sqrt(a(env))
        if t in _FUNCS or t in (r"\log", r"\operatorname"):
            if t == r"\log":
                self.take("_")
                base = float(self.raw_group())
                fn = lambda v, b=base: math.log(v, b)  # noqa: E731
            elif t == r"\operatorname":
                fn = _OPNAMES[self.raw_group()]
            else:
                fn = _FUNCS[t]
            power = None
            if self.peek() == "^":
                self.take()
                power = self.group()
            arg = self.fn_arg()
            if power is None:
                return lambda env: fn(arg(env))
            return lambda env: fn(arg(env)) ** power(env)
        if t in (r"\max", r"\min"):
            agg = max if t == r"\max" else min
            left = self.take()
            if left == r"\left":
                self.take(r"\{")
            items = [self.expr()]
            while self.peek() == ",":
                self.take()
                items.append(self.expr())
            if self.peek() == r"\right":
                self.take()
            self.take(r"\}")
            return lambda env: agg(f(env) for f in items)
        if t == r"\sum":
            return self.big_op(sum)
        if t == r"\prod":
            return self.big_op(math.prod)
        if t == r"\mathrm":
            name = self.raw_group()
            if name == "e":
                return lambda env: math.e
            raise SyntaxError(f"bare \\mathrm{{{name}}}")
        if t == r"\pi":
            return lambda env: math.pi
        if t == r"\mathit":
            return self.symbol(r"\mathit{" + self.raw_group() + "}")
        if t in _GREEK or re.fullmatch(r"[A-Za-z]", t):
            return self.symbol(t)
        raise SyntaxError(f"unexpected token {t!r} at {self.pos}: {self.toks}")


def _real_pow(a: float, b: float) -> float:
    v = a**b
    if isinstance(v, complex):
        raise ValueError("complex power")
    return v


def tex_value(text: str, env: dict) -> float:
    rel, e, rhs = _TexEval(text).parse()
    assert rel is None, text
    return e(env)


# --------------------------------------------------------------------------
# A reference evaluator for the expression DAG (numpy, no solver code).
# --------------------------------------------------------------------------

_NP_FUNCS = {
    "exp": np.exp,
    "log": np.log,
    "log2": np.log2,
    "log10": np.log10,
    "log1p": np.log1p,
    "sqrt": np.sqrt,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "asinh": np.arcsinh,
    "acosh": np.arccosh,
    "atanh": np.arctanh,
    "erf": np.vectorize(math.erf),
    "abs": np.abs,
    "sign": np.sign,
    "sigmoid": lambda v: 1.0 / (1.0 + np.exp(-v)),
    "softplus": lambda v: np.log1p(np.exp(v)),
    "entropy": lambda v: v * np.log(v),
    "prod": np.prod,
    "norm2": np.linalg.norm,
}


def dag_value(e, vals: dict):
    if isinstance(e, Constant):
        return np.asarray(e.value, dtype=float)
    if isinstance(e, Variable):
        return np.asarray(vals[e.name], dtype=float)
    if isinstance(e, IndexExpression):
        return dag_value(e.base, vals)[e.index]
    if isinstance(e, BinaryOp) and e.op in ("+", "-"):
        spine = []  # iterative over the left spine: builtin sum() nests deeply
        while isinstance(e, BinaryOp) and e.op in ("+", "-"):
            spine.append(e)
            e = e.left
        acc = dag_value(e, vals)
        for node in reversed(spine):
            r = dag_value(node.right, vals)
            acc = acc + r if node.op == "+" else acc - r
        return acc
    if isinstance(e, BinaryOp):
        a, b = dag_value(e.left, vals), dag_value(e.right, vals)
        return {"+": a + b, "-": a - b, "*": a * b, "/": a / b, "**": a**b}[e.op]
    if isinstance(e, UnaryOp):
        v = dag_value(e.operand, vals)
        return -v if e.op == "neg" else np.abs(v)
    if isinstance(e, FunctionCall):
        args = [dag_value(a, vals) for a in e.args]
        if e.func_name == "max":
            return np.maximum(*args)
        if e.func_name == "min":
            return np.minimum(*args)
        return _NP_FUNCS[e.func_name](*args)
    if isinstance(e, SumOverExpression):
        return sum(dag_value(t, vals) for t in e.terms)
    if isinstance(e, SumExpression):
        return np.sum(dag_value(e.operand, vals), axis=e.axis)
    raise TypeError(type(e))


def tex_env(model_vars, vals: dict) -> dict:
    """Map every rendered element symbol to its value."""
    env = {}
    for v in model_vars:
        arr = np.asarray(vals[v.name], dtype=float)
        if arr.ndim == 0:
            env[latex._sym(v.name)] = float(arr)
        else:
            for idx in np.ndindex(arr.shape):
                env[latex._sym(v.name, [str(k) for k in idx])] = float(arr[idx])
    return env


def _close(a: float, b: float) -> bool:
    # The renderer prints decimals to 6 significant figures, so compare there.
    return math.isclose(a, b, rel_tol=5e-6, abs_tol=1e-9)


# --------------------------------------------------------------------------
# 1. Semantic round trip
# --------------------------------------------------------------------------

_CONSTS = [-3.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.5, 4.0, 2.5e7, 1e-7, 0.25]
_UNARY = [
    "exp",
    "log",
    "sin",
    "cos",
    "tanh",
    "sqrt",
    "atan",
    "log1p",
    "softplus",
    "sigmoid",
    "entropy",
    "erf",
    "asinh",
    "log2",
    "log10",
]


def _random_expr(rng: random.Random, leaves: list, depth: int):
    if depth == 0 or rng.random() < 0.25:
        if rng.random() < 0.3:
            return Constant(rng.choice(_CONSTS))
        return rng.choice(leaves)
    kind = rng.random()
    if kind < 0.45:
        op = rng.choice(["+", "-", "*", "/", "*", "-"])
        return BinaryOp(
            op, _random_expr(rng, leaves, depth - 1), _random_expr(rng, leaves, depth - 1)
        )
    if kind < 0.55:
        return BinaryOp(
            "**",
            _random_expr(rng, leaves, depth - 1),
            Constant(rng.choice([2.0, 3.0, 0.5, -1.0, 1.0 / 3.0])),
        )
    if kind < 0.65:
        return UnaryOp("neg", _random_expr(rng, leaves, depth - 1))
    if kind < 0.72:
        return UnaryOp("abs", _random_expr(rng, leaves, depth - 1))
    if kind < 0.8:
        name = rng.choice(["max", "min"])
        return FunctionCall(
            name, _random_expr(rng, leaves, depth - 1), _random_expr(rng, leaves, depth - 1)
        )
    return FunctionCall(rng.choice(_UNARY), _random_expr(rng, leaves, depth - 1))


def _check_roundtrip(e, model_vars, rng: random.Random, n_points: int = 4) -> int:
    """Render *e*, evaluate the text and the DAG at random points; return the
    number of points actually compared (domain errors on both sides skip)."""
    text = latex.expr_to_latex(e)
    compared = 0
    for _ in range(n_points):
        vals = {}
        for v in model_vars:
            shape = tuple(v.shape)
            vals[v.name] = (
                np.asarray(
                    [rng.uniform(0.2, 1.8) for _ in range(int(np.prod(shape)) or 1)]
                ).reshape(shape)
                if shape
                else rng.uniform(0.2, 1.8)
            )
        with np.errstate(all="ignore"):
            ref = float(dag_value(e, vals))
        if not np.isfinite(ref) or abs(ref) > 1e12:
            continue
        try:
            got = tex_value(text, tex_env(model_vars, vals))
        except (ValueError, ZeroDivisionError, OverflowError):
            continue  # a domain error the numpy side turned into a finite value
        assert _close(got, ref), f"{text!r}: tex={got} dag={ref} expr={e!r}"
        compared += 1
    return compared


def test_random_expression_trees_round_trip():
    rng = random.Random(20260927)
    m = dm.Model("rt")
    a = m.continuous("a", lb=0, ub=2)
    b = m.continuous("b", lb=0, ub=2)
    x = m.continuous("x", shape=(3,), lb=0, ub=2)
    flow = m.continuous("flow_in", lb=0, ub=2)
    leaves = [a, b, x[0], x[1], x[2], flow]
    model_vars = [a, b, x, flow]
    compared = 0
    for _ in range(600):
        compared += _check_roundtrip(_random_expr(rng, leaves, 4), model_vars, rng)
    # Prove the probe fired: most trees must actually have been compared.
    assert compared > 1000, compared


def test_recovered_sums_round_trip():
    rng = random.Random(7)
    m = dm.Model("sums")
    n = 9
    v = m.continuous("v", shape=(n,), lb=0, ub=2)
    w = m.continuous("w", shape=(n,), lb=0, ub=2)
    X = m.continuous("X", shape=(4, 3), lb=0, ub=2)
    a = m.continuous("a", lb=0, ub=2)
    cases = [
        dm.sum(v[k] ** 2 for k in range(n)),
        dm.sum(v[k + 1] - v[k] for k in range(n - 1)),
        dm.sum(v[k] * w[n - 1 - k] for k in range(n)),  # not affine in one index
        dm.sum(a * v[k] - w[k] for k in range(2, 7)),
        dm.sum(dm.exp(v[k]) * w[k] for k in range(n)) ** 2,
        dm.sum(lambda i: dm.sum(lambda j: X[i, j] * v[j], over=range(3)), over=range(4)),
        2 * dm.sum(X[i, 1] for i in range(4)) - dm.sum(X[1, j] for j in range(3)),
        dm.sum(v[k] for k in [0, 2, 4, 6]),  # non-contiguous: written out
        dm.sum(v[k] * 2.0**k for k in range(4)),  # differing data: written out
        dm.sum(v) + dm.sum(X) - dm.sum(w * v),
        dm.sum(-v[k] for k in range(n)),
        sum(v[k] * w[k] for k in range(n)),  # builtin sum: a nested +-chain
        sum(2.0 * v[k] for k in range(1, 5)) - a,
    ]
    compared = 0
    for e in cases:
        compared += _check_roundtrip(e, [v, w, X, a], rng)
    assert compared >= 4 * len(cases) - 2, compared


def test_deep_builtin_sum_renders_and_round_trips():
    """`sum(x[i] for i in range(3000))` is a 3000-deep left-nested chain; the
    old renderer raised RecursionError into the notebook and a naive fix
    rendered an empty `\\text{}`. It must render (as a recovered sum) and mean
    the same thing."""
    m = dm.Model("deep")
    x = m.continuous("x", shape=(3000,), lb=0, ub=1)
    e = sum(x[i] for i in range(3000))
    assert latex.expr_to_latex(e) == r"\sum_{i=0}^{2999} x_{i}"
    m.minimize(e)
    m.subject_to(sum(float(i % 7) * x[i] for i in range(3000)) <= 5)
    tex = m.to_latex()
    assert r"\sum_{i=0}^{2999} x_{i}" in tex
    assert r"\cdots" in tex  # differing data: written out, elided
    compared = _check_roundtrip(e, [x], random.Random(0), n_points=2)
    assert compared == 2


def test_index_symbol_avoids_name_subscripts():
    m = dm.Model("names")
    x = m.continuous("x", shape=(3,), lb=0, ub=1)
    xi = m.continuous("x_i", lb=0, ub=1)  # typesets as x_{i}
    tex = latex.expr_to_latex(dm.sum(xi * x[k] for k in range(3)))
    assert tex.startswith(r"\sum_{j=0}^{2}"), tex


def test_constraint_orientation_round_trip():
    """A rendered relation must hold exactly when the model's constraint does:
    flipping a constant left side or moving constants across must preserve
    the direction."""
    rng = random.Random(3)
    m = dm.Model("rel")
    x = m.continuous("x", shape=(3,), lb=0, ub=2)
    y = m.continuous("y", lb=0, ub=2)
    cons = [
        x[0] + 5 * y >= 4,
        4 <= x[0] + x[1],
        x[0] >= x[1],
        x[0] - 4 == 0,
        x[0] + 2 * x[1] - 3 <= 0,
        dm.sum(x) >= 2,
        -1.5 - x[2] + y <= 0.5 * x[0],
        x[0] * y - (x[1] - 3) <= 1,
        2 - y >= x[2],
    ]
    checked = 0
    for c in cons:
        text = latex._constraint_to_latex(c)
        rel, lhs, rhs = _TexEval(text).parse()
        assert rel in (r"\le", r"\ge", "="), text
        for _ in range(20):
            vals = {"x": np.array([rng.uniform(0, 2) for _ in range(3)]), "y": rng.uniform(0, 2)}
            body = float(dag_value(c.body, vals)) - float(getattr(c, "rhs", 0.0))
            want = {"<=": body <= 0, ">=": body >= 0, "==": abs(body) < 1e-12}[c.sense]
            env = tex_env([x, y], vals)
            d = lhs(env) - rhs(env)
            if c.sense == "==":
                assert _close(abs(d), abs(body)), (text, d, body)
            else:
                got = {r"\le": d <= 0, r"\ge": d >= 0}[rel]
                if abs(body) > 1e-9:
                    assert got == want, (text, vals, body, d)
            checked += 1
    assert checked == 20 * len(cons)


# --------------------------------------------------------------------------
# 2. Conventions (golden fragments)
# --------------------------------------------------------------------------


@pytest.fixture
def xm():
    m = dm.Model("g")
    x = m.continuous("x", shape=(3,), lb=0.1, ub=5)
    y = m.continuous("y", lb=0, ub=1)
    return m, x, y


def test_implicit_multiplication_and_sign_folding(xm):
    _, x, y = xm
    r = latex.expr_to_latex
    assert r(2 * y) == "2 y"
    assert r(y * 3) == "3 y"
    assert r(x[0] + (-2) * x[1]) == "x_{0} - 2 x_{1}"
    assert r(x[0] - (-3) * x[1]) == "x_{0} + 3 x_{1}"
    assert r(-1 * y + 1 * x[0]) == "-y + x_{0}"
    assert r(x[0] * (x[1] + x[2]) * 3) == "3 x_{0} (x_{1} + x_{2})"
    assert r(x[0] - (x[1] - x[2])) == "x_{0} - (x_{1} - x_{2})"
    assert r((-y) ** 2) == "(-y)^{2}"
    assert r(Constant(2.0) * Constant(3.0)) == "6"


def test_numbers():
    f = latex._fmt_num
    assert f(3.0) == "3"
    assert f(1e-6) == "10^{-6}"
    assert f(2.5e7) == r"2.5 \times 10^{7}"
    assert f(-3.2e-9) == r"-3.2 \times 10^{-9}"
    assert f(123456.0) == "123456"
    assert f(0.25) == "0.25"
    assert latex._exponent_num(1 / 3) == "1/3"
    # Exact by default: the shortest decimal that round-trips to the float.
    assert f(4 / 9) == "0.4444444444444444"
    assert float(f(2193.5123456789)) == 2193.5123456789


def test_precision_is_opt_in(xm):
    m, x, y = xm
    e = (4 / 9) * y + 2193.5123456789
    assert latex.expr_to_latex(e) == "0.4444444444444444 y + 2193.5123456789"
    assert latex.expr_to_latex(e, precision=4) == "0.4444 y + 2194"
    m.minimize(e)
    assert "0.4444 y" in m.to_latex(precision=4)
    assert "0.4444444444444444 y" in m.to_latex()
    # The setting is scoped to the call.
    assert "0.4444444444444444 y" in m.to_latex()
    with pytest.raises(ValueError):
        m.to_latex(precision=0)


def test_function_names(xm):
    _, x, y = xm
    r = latex.expr_to_latex
    assert r(dm.asin(y)) == r"\arcsin(y)"
    assert r(dm.acos(y)) == r"\arccos(y)"
    assert r(dm.atan(y)) == r"\arctan(y)"
    assert r(dm.asinh(y)) == r"\operatorname{arsinh}(y)"
    assert r(dm.erf(y)) == r"\operatorname{erf}(y)"
    assert r(dm.log2(y)) == r"\log_{2}(y)"
    assert r(dm.log10(y)) == r"\log_{10}(y)"
    assert r(dm.log(y)) == r"\ln(y)"
    assert r(dm.log1p(y)) == r"\ln(1 + y)"
    assert r(dm.softplus(y)) == r"\ln(1 + \mathrm{e}^{y})"
    assert r(dm.sigmoid(y)) == r"\frac{1}{1 + \mathrm{e}^{-y}}"
    assert r(dm.xlogx(y)) == r"y \ln(y)"
    assert r(dm.sign(y)) == r"\operatorname{sgn}(y)"
    assert r(dm.abs(y)) == "|y|"
    assert r(dm.maximum(x[0], x[1], x[2])) == r"\max\{x_{0}, x_{1}, x_{2}\}"
    assert r(dm.norm(x, 2)) == r"\lVert x \rVert_{2}"
    assert r(dm.norm(x / 2, 1)) == r"\left\lVert \frac{x}{2} \right\rVert_{1}"
    assert r(dm.prod(x)) == r"\prod_{i} x_{i}"
    assert r(dm.sin(y) ** 2) == r"\sin^{2}(y)"
    assert r(y**0.5) == r"\sqrt{y}"
    assert r((x[0] / x[1]) ** 2) == r"\left(\frac{x_{0}}{x_{1}}\right)^{2}"
    assert r(x @ x) == r"x^{\top} x"
    A = Constant(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    assert r(x @ A) == r"x^{\top} \begin{bmatrix}1 & 2 \\ 3 & 4 \\ 5 & 6\end{bmatrix}"
    # GAMS intrinsics that arrive through the .gms/.nl readers.
    assert r(FunctionCall("signpower", y, Constant(3.0))) == r"\operatorname{sgn}(y)\,|y|^{3}"
    assert r(FunctionCall("centropy", y, x[0])) == r"y \ln\left(\frac{y}{x_{0}}\right)"
    assert r(FunctionCall("atan2", y, x[0])) == r"\operatorname{atan2}(y, x_{0})"


def test_identifiers():
    s = latex._sym
    assert s("alpha") == r"\alpha"
    assert s("Gamma") == r"\Gamma"
    assert s("x3") == "x_{3}"
    assert s("flow_in") == r"\mathit{flow}_{\mathrm{in}}"
    assert s("T_max") == r"T_{\mathrm{max}}"
    assert s("theta_1") == r"\theta_{1}"
    assert s("x_i_j") == "x_{i,j}"
    assert s("y1", ["0"]) == "y_{1,0}"
    assert s("price") == r"\mathit{price}"
    assert s("_ifelse_1") == r"\mathit{\_ifelse\_1}"


def test_sums(xm):
    m, x, y = xm
    v = m.continuous("v", shape=(12,), lb=-1, ub=1)
    r = latex.expr_to_latex
    assert r(dm.sum(v[k] ** 2 for k in range(12))) == r"\sum_{i=0}^{11} v_{i}^{2}"
    assert r(dm.sum(v[k + 1] - v[k] for k in range(11))) == r"\sum_{i=1}^{11} (v_{i} - v_{i-1})"
    assert r(dm.sum(x)) == r"\sum_{i} x_{i}"
    p = m.parameter("price", value=np.array([1.0, 2.0, 3.0]))
    assert r(dm.sum(p * x)) == r"\sum_{i} \mathit{price}_{i}\,x_{i}"
    # A sum as the left factor of a product is grouped so it cannot swallow it.
    assert r(dm.sum(x) * y) == r"\left(\sum_{i} x_{i}\right) y"
    # Unnamed, differing data cannot be indexed: the terms are written out.
    assert r(dm.sum(float(k + 1) * x[k] for k in range(3))) == "x_{0} + 2 x_{1} + 3 x_{2}"
    assert r(dm.sum(float(k + 1) * v[k] for k in range(12))).endswith(r"11 v_{10} + 12 v_{11}")
    w = m.continuous("w", shape=(20,), lb=-1, ub=1)
    long = r(dm.sum(float(k + 1) * w[k] for k in range(20)))
    assert long == r"w_{0} + 2 w_{1} + 3 w_{2} + \cdots + 20 w_{19}"
    assert r(dm.sum(-float(k + 1) * w[k] for k in range(20))).endswith(r"\cdots - 20 w_{19}")
    # An index symbol never collides with a variable of the same name.
    i = m.continuous("i", lb=0, ub=1)
    assert r(dm.sum(i * x[k] for k in range(3)), reserved={"i"}).startswith(r"\sum_{j=0}^{2}")


def test_constraints_read_naturally(xm):
    _, x, y = xm
    c = latex._constraint_to_latex
    assert c(x[0] + 5 * y >= 4) == r"x_{0} + 5 y \ge 4"
    assert c(4 <= x[0]) == r"x_{0} \ge 4"
    assert c(x[0] == 4) == "x_{0} = 4"
    assert c(x[0] + 2 * x[1] - 3 <= 0) == r"x_{0} + 2 x_{1} \le 3"
    assert c(dm.sum(x) >= 2) == r"\sum_{i} x_{i} \ge 2"


def test_vector_domains():
    m = dm.Model("dom")
    m.continuous("x", shape=(3,), lb=0, ub=10)
    m.continuous("X", shape=(2, 2), lb=0, ub=1)
    m.integer("n", shape=(3,), lb=0, ub=9)
    m.binary("b", shape=(4,))
    m.continuous("w", shape=(4,), lb=[0, 1, 2, 3], ub=10)
    m.continuous("f", shape=(2,))
    m.continuous("h", shape=(20,), lb=np.arange(20.0), ub=100)
    rows = [latex._var_domain_to_latex(v) for v in m._variables]
    assert rows[0] == r"x \in [0, 10]^{3}"
    assert rows[1] == r"X \in [0, 1]^{2 \times 2}"
    assert rows[2] == r"n \in \{0, \ldots, 9\}^{3}"
    assert rows[3] == r"b \in \{0, 1\}^{4}"
    assert rows[4] == r"w \in \mathbb{R}^{4},\; (0, 1, 2, 3)^{\top} \le w \le 10"
    assert rows[5] == r"f \in \mathbb{R}^{2}"
    # Too many distinct bounds to list: the enclosing box, disclosed as such.
    assert rows[6].endswith(r"\text{(bounds vary by element)}")
    assert r"0 \le h" in rows[6] and r"h \le 100" in rows[6]


def test_standalone_document():
    m = dm.Model("sa")
    x = m.continuous("x", lb=0, ub=1)
    m.minimize(x)
    doc = m.to_latex(standalone=True)
    assert doc.startswith(r"\documentclass{article}")
    assert r"\usepackage{amsmath,amssymb}" in doc
    assert doc.rstrip().endswith(r"\end{document}")


# --------------------------------------------------------------------------
# 3. Structural lint + engines over a corpus
# --------------------------------------------------------------------------


def corpus() -> dict:
    out = {}
    m = dm.Model("funcs")
    x = m.continuous("x", shape=(3,), lb=0.1, ub=5)
    z = m.integer("z", lb=0, ub=5)
    m.minimize(dm.exp(x[0]) + x[1] ** 2 / (1 + x[2]) - dm.log(z + 1))
    m.subject_to(dm.sqrt(x[0] * x[1]) <= 3 * z)
    m.subject_to(dm.sum(x) >= 2)
    m.subject_to(dm.asin(x[0] / 5) + dm.acos(x[1] / 5) + dm.atan(x[2]) <= 3)
    m.subject_to(dm.erf(x[0]) + dm.sigmoid(x[1]) + dm.softplus(x[2]) <= 3)
    m.subject_to(dm.log2(x[0]) + dm.log10(x[1]) + dm.log1p(x[2]) >= -5)
    m.subject_to(dm.xlogx(x[0]) + dm.sign(x[1]) * x[2] <= 7)
    m.subject_to(dm.maximum(x[0], x[1], x[2]) - dm.minimum(x[0], x[1]) <= 2)
    m.subject_to(dm.norm(x, 2) + dm.norm(x, 1) + dm.prod(x) <= 40)
    m.subject_to(dm.asinh(x[0]) + dm.acosh(x[1] + 1) + dm.atanh(x[2] / 6) <= 9)
    m.subject_to((x[0] / x[1]) ** 2 + x[2] ** 0.5 + x[0] ** -1 <= 30)
    m.subject_to(2.5e7 * x[0] - 1 * x[1] + (-1) * x[2] >= -1.25e-4)
    m.subject_to(x @ x <= 12)
    m.subject_to(dm.exp(x[0]) ** 2 + dm.sin(x[1]) ** 2 <= 30)
    out["funcs"] = m

    m = dm.Model("names_and_sums")
    a = m.continuous("alpha", lb=0, ub=1)
    fi = m.continuous("flow_in", shape=(2,), lb=0, ub=[3, 4])
    t = m.continuous("T_max", lb=-5, ub=5)
    p = m.parameter("price", value=np.array([1.0, 2.0]))
    aux = m.continuous("_aux_1", lb=0, ub=1)
    m.maximize(dm.sum(p * fi) - a * t + aux)
    ship = m.continuous("ship", shape=(3, 2), lb=0, ub=100)
    cost = m.parameter("cost", value=np.arange(6.0).reshape(3, 2))
    m.subject_to(
        dm.sum(lambda i: dm.sum(lambda j: cost[i, j] * ship[i, j], over=range(2)), over=range(3))
        <= 100
    )
    v = m.continuous("v", shape=(12,), lb=-1, ub=1)
    m.subject_to(dm.sum(v[k + 1] - v[k] for k in range(11)) >= -1)
    m.subject_to(dm.sum(float(k) * v[k] for k in range(12)) <= 3)
    m.subject_to(dm.sum(ship, axis=1) <= 50)
    m.subject_to(ship[:, 0] <= 20)
    m.continuous("w", shape=(4,), lb=[0, 1, 2, 3], ub=10)
    m.integer("n", shape=(3,), lb=0, ub=9)
    m.binary("b", shape=(2,))
    out["names_and_sums"] = m

    m = dm.Model("fast")
    xs = m.continuous("xs", shape=(3,), lb=0, ub=1)
    m.add_linear_constraints(np.array([[1.0, -1.0, 2.5]]), xs, "<=", np.array([3.0]))
    m.minimize(dm.sum(xs))
    out["fast"] = m

    # Large enough (> 5120 chars after its first integer-domain row) to trip
    # MathJax's macro-expansion buffer if any macro such as `\ ` is emitted.
    m = dm.Model("large")
    zs = [m.integer(f"z{k}", lb=0, ub=3) for k in range(150)]
    for k in range(149):
        m.subject_to(zs[k] - 2 * zs[k + 1] <= 1)
    m.minimize(dm.sum(zs[k] for k in range(150)))
    out["large"] = m
    return out


# Every macro the renderer may emit. Each is implemented by pdflatex with
# amsmath+amssymb AND by MathJax 3's base+ams packages.
_MACROS = {
    r"\alpha",
    r"\beta",
    r"\gamma",
    r"\delta",
    r"\epsilon",
    r"\varepsilon",
    r"\zeta",
    r"\eta",
    r"\theta",
    r"\vartheta",
    r"\iota",
    r"\kappa",
    r"\lambda",
    r"\mu",
    r"\nu",
    r"\xi",
    r"\pi",
    r"\varpi",
    r"\rho",
    r"\varrho",
    r"\sigma",
    r"\varsigma",
    r"\tau",
    r"\upsilon",
    r"\phi",
    r"\varphi",
    r"\chi",
    r"\psi",
    r"\omega",
    r"\Gamma",
    r"\Delta",
    r"\Theta",
    r"\Lambda",
    r"\Xi",
    r"\Pi",
    r"\Sigma",
    r"\Upsilon",
    r"\Phi",
    r"\Psi",
    r"\Omega",
    r"\begin",
    r"\end",
    r"\text",
    r"\quad",
    r"\le",
    r"\ge",
    r"\in",
    r"\mathbb",
    r"\mathit",
    r"\mathrm",
    r"\mathbf",
    r"\operatorname",
    r"\frac",
    r"\sqrt",
    r"\sum",
    r"\prod",
    r"\left",
    r"\right",
    r"\cdot",
    r"\cdots",
    r"\ldots",
    r"\vdots",
    r"\times",
    r"\top",
    r"\circ",
    r"\infty",
    r"\exp",
    r"\ln",
    r"\log",
    r"\sin",
    r"\cos",
    r"\tan",
    r"\sinh",
    r"\cosh",
    r"\tanh",
    r"\arcsin",
    r"\arccos",
    r"\arctan",
    r"\max",
    r"\min",
    r"\lVert",
    r"\rVert",
    r"\emptyset",
    r"\hat",
    r"\tilde",
    r"\backslash",
}


def lint(tex: str) -> list[str]:
    problems = []
    depth = 0
    for ch in re.sub(r"\\[{}]", "", tex):
        depth += {"{": 1, "}": -1}.get(ch, 0)
        if depth < 0:
            problems.append("unbalanced }")
            break
    if depth > 0:
        problems.append("unbalanced {")
    if len(re.findall(r"\\left\b", tex)) != len(re.findall(r"\\right\b", tex)):
        problems.append("unpaired \\left/\\right")
    for macro in set(re.findall(r"\\[A-Za-z]+", tex)):
        if macro not in _MACROS:
            problems.append(f"unknown macro {macro}")
    # The control space `\ ` is a MathJax *macro*: past 5120 chars of remaining
    # input its expansion throws "internal buffer size exceeded" (casctanks).
    if re.search(r"(?<!\\)(?:\\\\)*\\ ", tex):
        problems.append("control space \\ (use \\; or \\,)")
    # Double sub/superscript: a script group directly followed by the same script.
    for script in ("_", "^"):
        if re.search(re.escape(script) + r"\{[^{}]*\}" + re.escape(script), tex):
            problems.append(f"double {script}")
    return problems


def test_corpus_lints_clean():
    checked = 0
    for name, m in corpus().items():
        tex = m.to_latex()
        assert not lint(tex), (name, lint(tex), tex)
        checked += 1
    assert checked == len(corpus())


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_corpus_compiles_with_pdflatex():
    checked = 0
    for name, m in corpus().items():
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "t.tex"), "w") as f:
                f.write(m.to_latex(standalone=True))
            r = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "t.tex"],
                cwd=d,
                capture_output=True,
                text=True,
                timeout=120,
            )
            errors = [ln for ln in r.stdout.splitlines() if ln.startswith("!")]
            assert r.returncode == 0, (name, errors)
            checked += 1
    assert checked == len(corpus())


_MATHJAX_SCRIPT = r"""
const {mathjax} = require('mathjax-full/js/mathjax.js');
const {TeX} = require('mathjax-full/js/input/tex.js');
const {SVG} = require('mathjax-full/js/output/svg.js');
const {liteAdaptor} = require('mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler} = require('mathjax-full/js/handlers/html.js');
require('mathjax-full/js/input/tex/ams/AmsConfiguration.js');
RegisterHTMLHandler(liteAdaptor());
const tex = new TeX({packages: ['base', 'ams'], formatError: (j, e) => { throw e; }});
const doc = mathjax.document('', {InputJax: tex, OutputJax: new SVG({fontCache: 'none'})});
const inputs = JSON.parse(require('fs').readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(inputs.map(s => {
  try { doc.convert(s, {display: true}); return null; } catch (e) { return String(e.message); }
})));
"""


def _mathjax_available() -> bool:
    if shutil.which("node") is None:
        return False
    r = subprocess.run(
        ["node", "-e", "require.resolve('mathjax-full/js/mathjax.js')"],
        capture_output=True,
        env={**os.environ},
    )
    return r.returncode == 0


@pytest.mark.skipif(not _mathjax_available(), reason="node + mathjax-full not installed")
def test_corpus_typesets_with_mathjax():
    texs = [m.to_latex() for m in corpus().values()]
    r = subprocess.run(
        ["node", "-e", _MATHJAX_SCRIPT],
        input=json.dumps(texs),
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    errors = json.loads(r.stdout)
    assert len(errors) == len(texs)
    assert all(e is None for e in errors), errors
