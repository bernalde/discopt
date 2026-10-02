"""Exact changes of representation of an ASCII ``.nl`` model (#1537, workstream B).

A certified answer must not change under a change of variables that leaves the
model identical. This module writes such changes directly into the ``.nl`` text, so
the invariance panel (``discopt_benchmarks/scripts/invariance_panel.py``) can apply
them to any MINLPLib instance without a modeling-layer round trip:

* :func:`translate_nl` -- ``x = y - t``: every variable (except a linear binary,
  whose type is positional) is replaced by a shifted one with the same box moved
  by ``t``. Nonlinear references ``v j`` become
  ``(v j - t)``; the linear parts keep their coefficients and their constant moves
  into the row ranges and the objective.
* :func:`scale_rows_nl` -- every algebraic row (body, linear part and both range
  sides) multiplied by ``k > 0``.
* :func:`scale_objective_nl` -- the objective multiplied by ``k > 0``; the optimum
  scales by ``k``.

All three are exact in real arithmetic. A construct whose transformation is not
implemented (binary ``.nl``, defined variables, complementarity rows, a
non-integral shift of a discrete variable) raises :class:`NotImplementedError`
rather than producing a different model: a probe that silently transforms the
wrong thing measures nothing (CLAUDE.md §6-7).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

#: First characters of an expression-graph line inside a C/O/L segment. The digits
#: are the operand-count lines of the n-ary operators (``o54`` sumlist, ``o11`` min,
#: ...); no segment header starts with one.
_EXPR_HEADS = frozenset("onvfhls0123456789")


def _num(x: float) -> str:
    return repr(float(x))


def _strip(line: str) -> str:
    return line.split("#", 1)[0].strip()


@dataclass
class _Header:
    n_vars: int
    n_cons: int
    n_objs: int
    n_binary: int
    n_integer: int
    n_discrete_nl: tuple[int, int, int]
    n_nl_vars: tuple[int, int, int]


def _parse_header(lines: list[str]) -> _Header:
    if not lines or not lines[0].startswith("g"):
        raise NotImplementedError("only ASCII ('g') .nl files are supported")
    if len(lines) < 10:
        raise ValueError(".nl header is shorter than 10 lines")
    h1 = [int(t) for t in _strip(lines[1]).split()]
    h4 = [int(t) for t in _strip(lines[4]).split()]
    h6 = [int(t) for t in _strip(lines[6]).split()]
    hv = [int(t) for t in _strip(lines[9]).split()]
    if any(hv):
        raise NotImplementedError(".nl defined variables (common expressions) are not supported")
    return _Header(
        n_vars=h1[0],
        n_cons=h1[1],
        n_objs=h1[2],
        n_binary=h6[0],
        n_integer=h6[1],
        n_discrete_nl=(h6[2], h6[3], h6[4]),
        n_nl_vars=(h4[0], h4[1], h4[2]),
    )


def _discrete_mask(h: _Header) -> list[bool]:
    """Which variables are binary/integer, from the .nl variable ordering.

    AMPL orders variables: nonlinear in both, nonlinear in constraints only,
    nonlinear in objectives only (each block ending with its nonlinear discrete
    ones), then linear arcs, other linear, linear binary, linear integer. Rather
    than reproduce that ordering, every variable is treated as discrete when any
    discrete variable exists -- the caller then needs an integral shift, which is
    the only safe requirement anyway.
    """
    any_discrete = h.n_binary + h.n_integer + sum(h.n_discrete_nl) > 0
    return [any_discrete] * h.n_vars


def _segments(lines: list[str]) -> list[tuple[str, list[str]]]:
    """Split the body (after the 10-line header) into ``(header, payload)`` pairs."""
    out: list[tuple[str, list[str]]] = []
    i = 10
    n = len(lines)
    while i < n:
        head = lines[i]
        tag = head[:1]
        i += 1
        payload: list[str] = []
        if tag in ("C", "O", "L"):
            while i < n and lines[i][:1] in _EXPR_HEADS:
                payload.append(lines[i])
                i += 1
        elif tag == "V":
            raise NotImplementedError(".nl defined variables (V segments) are not supported")
        elif tag in ("x", "d", "k"):
            count = int(_strip(head)[1:].split()[0])
            payload = lines[i : i + count]
            i += count
        elif tag in ("r", "b"):
            # One line per row / variable, each opening with its kind digit.
            while i < n and lines[i][:1].isdigit():
                payload.append(lines[i])
                i += 1
        elif tag in ("J", "G"):
            count = int(_strip(head)[1:].split()[1])
            payload = lines[i : i + count]
            i += count
        elif tag == "S":
            count = int(_strip(head)[1:].split()[1])
            payload = lines[i : i + count]
            i += count
        elif tag == "F":
            pass
        elif _strip(head) == "":
            continue
        else:
            raise ValueError(f"unrecognised .nl segment header {head!r}")
        out.append((head, payload))
    return out


def _render(lines: list[str], segs: list[tuple[str, list[str]]]) -> str:
    body: list[str] = list(lines[:10])
    for head, payload in segs:
        body.append(head)
        body.extend(payload)
    return "\n".join(body) + "\n"


def _linear_parts(segs, h: _Header):
    """``J`` rows and ``G`` objectives as ``{index: [(var, coef), ...]}``."""
    jac: dict[int, list[tuple[int, float]]] = {}
    grad: dict[int, list[tuple[int, float]]] = {}
    for head, payload in segs:
        tag = head[:1]
        if tag not in ("J", "G"):
            continue
        idx = int(_strip(head)[1:].split()[0])
        terms = []
        for ln in payload:
            j, a = _strip(ln).split()[:2]
            terms.append((int(j), float(a)))
        (jac if tag == "J" else grad)[idx] = terms
    return jac, grad


def _check_ranges(payload: list[str]) -> None:
    for ln in payload:
        if _strip(ln).split()[0] == "5":
            raise NotImplementedError("complementarity rows are not supported")


def _shift_range(ln: str, K: float) -> str:  # noqa: N803
    t = _strip(ln).split()
    kind = t[0]
    if kind == "0":
        return f"0 {_num(float(t[1]) + K)} {_num(float(t[2]) + K)}"
    if kind in ("1", "2", "4"):
        return f"{kind} {_num(float(t[1]) + K)}"
    if kind == "3":
        return "3"
    raise NotImplementedError(f"range kind {kind} is not supported")


def _scale_range(ln: str, k: float) -> str:
    t = _strip(ln).split()
    kind = t[0]
    if kind == "0":
        return f"0 {_num(float(t[1]) * k)} {_num(float(t[2]) * k)}"
    if kind in ("1", "2", "4"):
        return f"{kind} {_num(float(t[1]) * k)}"
    if kind == "3":
        return "3"
    raise NotImplementedError(f"range kind {kind} is not supported")


def translate_nl(text: str, shift: float | Sequence[float]) -> str:
    """Rewrite ``text`` in the variables ``y = x + shift`` (exact in real arithmetic).

    ``shift`` is a scalar or one value per variable. A model with any discrete
    variable requires an integral shift, so that ``y`` is integral exactly when
    ``x`` is.
    """
    lines = text.splitlines()
    h = _parse_header(lines)
    if isinstance(shift, (int, float)):
        # A *linear binary* is typed by its position in the .nl ordering and its box
        # is clamped to [0, 1] on read, so it cannot be moved without changing its
        # type; a scalar shift leaves that block where it is. (Nonlinear discrete
        # columns are read as general integers and shift like any other.)
        t = [float(shift)] * h.n_vars
        start = h.n_vars - h.n_integer - h.n_binary
        for j in range(start, start + h.n_binary):
            t[j] = 0.0
    else:
        t = [float(s) for s in shift]
        if len(t) != h.n_vars:
            raise ValueError(f"shift has {len(t)} entries for {h.n_vars} variables")
    for j, (tj, disc) in enumerate(zip(t, _discrete_mask(h))):
        if disc and tj != int(tj):
            raise NotImplementedError(f"non-integral shift {tj} of discrete variable {j}")
    segs = _segments(lines)
    jac, grad = _linear_parts(segs, h)
    out: list[tuple[str, list[str]]] = []
    for head, payload in segs:
        tag = head[:1]
        if tag in ("C", "O", "L"):
            new: list[str] = []
            for ln in payload:
                s = _strip(ln)
                if s.startswith("v") and int(s[1:]) < h.n_vars and t[int(s[1:])] != 0.0:
                    new += ["o1", s, "n" + _num(t[int(s[1:])])]
                else:
                    new.append(ln)
            if tag == "O":
                idx = int(_strip(head)[1:].split()[0])
                K = sum(a * t[j] for j, a in grad.get(idx, []))  # noqa: N806
                if K != 0.0:
                    new = ["o0", *new, "n" + _num(-K)]
            out.append((head, new))
        elif tag == "r":
            _check_ranges(payload)
            rows = []
            for i, ln in enumerate(payload):
                K = sum(a * t[j] for j, a in jac.get(i, []))  # noqa: N806
                rows.append(_shift_range(ln, K))
            out.append((head, rows))
        elif tag == "b":
            new_b = []
            for j, ln in enumerate(payload):
                p = _strip(ln).split()
                kind = p[0]
                if kind == "0":
                    new_b.append(f"0 {_num(float(p[1]) + t[j])} {_num(float(p[2]) + t[j])}")
                elif kind in ("1", "2", "4"):
                    new_b.append(f"{kind} {_num(float(p[1]) + t[j])}")
                elif kind == "3":
                    new_b.append("3")
                else:
                    raise NotImplementedError(f"bound kind {kind} is not supported")
            out.append((head, new_b))
        elif tag == "x":
            new_x = []
            for ln in payload:
                j_s, v_s = _strip(ln).split()[:2]
                new_x.append(f"{j_s} {_num(float(v_s) + t[int(j_s)])}")
            out.append((head, new_x))
        else:
            out.append((head, payload))
    return _render(lines, out)


def scale_rows_nl(text: str, k: float) -> str:
    """Multiply every algebraic row of ``text`` by ``k > 0`` (body and both sides)."""
    if not k > 0.0:
        raise ValueError("row scale must be positive (a negative one flips the sense)")
    lines = text.splitlines()
    _parse_header(lines)
    out: list[tuple[str, list[str]]] = []
    for head, payload in _segments(lines):
        tag = head[:1]
        if tag == "C":
            out.append((head, ["o2", "n" + _num(k), *payload]))
        elif tag == "J":
            terms = []
            for ln in payload:
                j, a = _strip(ln).split()[:2]
                terms.append(f"{j} {_num(float(a) * k)}")
            out.append((head, terms))
        elif tag == "r":
            _check_ranges(payload)
            out.append((head, [_scale_range(ln, k) for ln in payload]))
        else:
            out.append((head, payload))
    return _render(lines, out)


def scale_objective_nl(text: str, k: float) -> str:
    """Multiply the objective of ``text`` by ``k > 0``; the optimum scales by ``k``."""
    if not k > 0.0:
        raise ValueError("objective scale must be positive (a negative one flips the sense)")
    lines = text.splitlines()
    _parse_header(lines)
    out: list[tuple[str, list[str]]] = []
    for head, payload in _segments(lines):
        tag = head[:1]
        if tag == "O":
            out.append((head, ["o2", "n" + _num(k), *payload]))
        elif tag == "G":
            terms = []
            for ln in payload:
                j, a = _strip(ln).split()[:2]
                terms.append(f"{j} {_num(float(a) * k)}")
            out.append((head, terms))
        else:
            out.append((head, payload))
    return _render(lines, out)
