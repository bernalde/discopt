"""#1562: ``to_nl`` must put the nonlinear rows first.

The ``.nl`` header's ``nlc`` (line 3) names rows ``C0 .. C{nlc-1}`` as the
nonlinear block; every later row must carry an ``n0`` body. Both writers (the
Rust fast path and the Python ``_NLWriter``, ``DISCOPT_RUST_NL=0``) used to emit
rows in declaration order while still declaring ``nlc``, so the ordinary way to
write a model -- a linear row declared before a nonlinear one -- produced a file
whose header named the wrong rows. ``pyscipopt``'s reader segfaulted on it (rc
-11). The MINLPLib corpus never caught it: AMPL wrote those files
nonlinear-first and ``from_nl`` preserves row order.

The fix permutes rows to ``[nonlinear | linear]`` (stable) in both writers and
exposes the permutation as :func:`discopt.export.nl_row_order`, so per-row reader
output (``.sol`` duals, constraint activities) maps back to the model. Header
line 2's ``neqns`` (written as a constant ``0``) is now the true count.
"""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.export import nl_row_order
from discopt.export.nl import _NLWriter, _rust_nl_text

pytestmark = pytest.mark.unit

SQRT2 = math.sqrt(2.0)


# ── models: each declares a linear row BEFORE a nonlinear one ────────────────


def _lin_first() -> dm.Model:
    """The issue's repro. Optimum x = sqrt(2), y = 0, objective sqrt(2)."""
    m = dm.Model("lin_first")
    x = m.continuous("x", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=4)
    m.subject_to(x + y <= 3)  # linear, declared first
    m.subject_to(x * x + y >= 2)  # nonlinear, declared second
    m.minimize(x + 2 * y)
    return m


def _sigmoid_layer() -> dm.Model:
    """The OMLT-panel shape: affine equality rows, THEN sigmoid equality rows."""
    m = dm.Model("sigmoid_layer")
    x = m.continuous("x", shape=(2,), lb=-1, ub=1)
    z = m.continuous("z", shape=(4,), lb=-10, ub=10)
    a = m.continuous("a", shape=(4,), lb=0, ub=1)
    w = np.array([[1.0, -0.5], [0.3, 0.8], [-1.2, 0.4], [0.7, 0.7]])
    bias = np.array([0.1, -0.2, 0.0, 0.3])
    for i in range(4):
        m.subject_to(z[i] == w[i, 0] * x[0] + w[i, 1] * x[1] + bias[i])
    for i in range(4):
        m.subject_to(a[i] == dm.sigmoid(z[i]))
    m.subject_to(x[0] + x[1] <= 1.5)
    m.minimize(a[0] - a[1] + a[2] - a[3] + 0.1 * (x[0] * x[0] + x[1] * x[1]))
    return m


def _mixed_with_builder_rows() -> dm.Model:
    """Expression rows (lin, nl, lin, nl) plus bulk builder rows (always linear)."""
    m = dm.Model("mixed_builder")
    x = m.continuous("x", shape=(3,), lb=0.1, ub=5)
    m.subject_to(x[0] + x[1] >= 0.5)
    m.subject_to(dm.exp(x[0]) + x[2] <= 20)
    m.subject_to(x[1] - x[2] <= 2)
    m.subject_to(x[0] * x[1] == 1)
    m.add_linear_constraints(np.array([[1.0, 1.0, 1.0], [1.0, -1.0, 0.0]]), x, "<=", [9.0, 3.0])
    m.minimize(x[0] + x[1] + x[2])
    return m


def _parameter_rows() -> dm.Model:
    """Parameter rows that are LINEAR at export time (PR #1578 review).

    A Parameter is a constant in the file, so ``p*x + y >= 1`` and
    ``x + y >= p`` are ``n0`` rows. The Python writer used to call them
    nonlinear while Rust (the default) called them linear. Since the
    nonlinear-first reorder, that gave the two writers different row orders,
    and ``nl_row_order`` (computed by Python) mapped the default file's rows,
    and so its duals, onto the wrong constraints.
    """
    m = dm.Model("parameter_rows")
    x = m.continuous("x", lb=0.5, ub=2)
    y = m.continuous("y", lb=0.5, ub=2)
    p = m.parameter("p", value=2.0)
    q = m.parameter("q", value=np.array([1.5, -3.0]))
    m.subject_to(x + y <= 3, name="lin")
    m.subject_to(p * x + y >= 1, name="param_coeff")
    m.subject_to(x * x + y >= 1, name="nl")
    m.subject_to(x + y >= p, name="param_rhs")
    m.subject_to(q[0] * (x + y) + q[1] <= 4, name="param_index")
    m.minimize(x + y)
    return m


MODELS = {
    "lin_first": _lin_first,
    "sigmoid_layer": _sigmoid_layer,
    "mixed_builder": _mixed_with_builder_rows,
    "parameter_rows": _parameter_rows,
}


# ── .nl text helpers ─────────────────────────────────────────────────────────


def _parse(text: str) -> tuple[list[int], list[list[str]], list[str]]:
    """``(header line 2 ints, C-body token lists by row, r-section lines)``."""
    lines = text.splitlines()
    line2 = [int(t) for t in lines[1].split("#")[0].split()]
    nlc = int(lines[2].split()[0])
    n_cons = line2[1]
    bodies: list[list[str]] = []
    r_lines: list[str] = []
    i = 10
    current: list[str] | None = None
    while i < len(lines):
        tok = lines[i]
        if tok.startswith("C"):
            current = []
            bodies.append(current)
        elif tok[0] in "OrbkJGx":
            current = None
            if tok == "r":
                r_lines = lines[i + 1 : i + 1 + n_cons]
                i += n_cons
        elif current is not None:
            current.append(tok)
        i += 1
    assert len(bodies) == n_cons, (len(bodies), n_cons)
    return [nlc, *line2], bodies, r_lines


def _text(model: dm.Model, rust: bool) -> str:
    if rust:
        text = _rust_nl_text(model, None)
        assert text is not None, "the Rust writer declined -- this case proves nothing about it"
        return text
    return _NLWriter(model).write()


# ── 1. the header's claim matches the row bodies, in both writers ────────────


@pytest.mark.parametrize("name", sorted(MODELS))
@pytest.mark.parametrize("rust", [True, False], ids=["rust", "python"])
def test_nlc_names_exactly_the_non_n0_rows(name, rust):
    model = MODELS[name]()
    (nlc, _n_vars, n_cons, _n_objs, n_ranges, n_eqns), bodies, r_lines = _parse(_text(model, rust))
    nonlinear = [b != ["n0"] for b in bodies]
    assert nlc >= 1 and nlc < n_cons, "fixture must mix both kinds of rows"
    assert nonlinear[:nlc] == [True] * nlc, f"C0..C{nlc - 1} must all be nonlinear"
    assert nonlinear[nlc:] == [False] * (n_cons - nlc), "rows past nlc must be n0"
    # Header line 2: neqns / nranges match the r section.
    assert n_eqns == sum(1 for r in r_lines if r.split()[0] == "4")
    assert n_ranges == sum(1 for r in r_lines if r.split()[0] == "0")


def test_fixtures_really_declare_a_linear_row_first():
    """Guard the premise: with an identity row order these tests are vacuous."""
    checked = 0
    for name, build in MODELS.items():
        order = nl_row_order(build())
        assert order.tolist() != sorted(order.tolist()), name
        assert sorted(order.tolist()) == list(range(len(order)))
        checked += 1
    assert checked == len(MODELS)


def test_sigmoid_layer_counts_its_equalities():
    """The issue's second defect: ``neqns`` was written as 0."""
    (_nlc, _nv, _nc, _no, n_ranges, n_eqns), _b, _r = _parse(_text(_sigmoid_layer(), True))
    assert (n_ranges, n_eqns) == (0, 8)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_writers_stay_byte_identical_on_the_new_order(name):
    model = MODELS[name]()
    assert _text(model, True) == _text(model, False)


def test_nonlinear_first_model_is_unchanged():
    """A model already declared nonlinear-first gets the identity permutation."""
    m = dm.Model("nl_first")
    x = m.continuous("x", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=4)
    m.subject_to(x * x + y >= 2)
    m.subject_to(x + y <= 3)
    m.minimize(x + 2 * y)
    assert nl_row_order(m).tolist() == [0, 1]


# ── 2. an external reader: SCIP reads the file and agrees ────────────────────


_SCIP_PROBE = """
import sys, pyscipopt
m = pyscipopt.Model()
m.hideOutput()
m.readProblem(sys.argv[1])
m.optimize()
print(m.getStatus(), repr(m.getObjVal()))
"""


@pytest.mark.skipif(importlib.util.find_spec("pyscipopt") is None, reason="pyscipopt absent")
@pytest.mark.parametrize("rust", [True, False], ids=["rust", "python"])
def test_scip_reads_the_file_and_finds_the_optimum(tmp_path, rust, monkeypatch):
    # A subprocess: the pre-fix file SEGFAULTED the reader (rc -11), which would
    # take the whole test session down with it.
    monkeypatch.setenv("DISCOPT_RUST_NL", "1" if rust else "0")
    model = _lin_first()
    path = tmp_path / "lin_first.nl"
    model.to_nl(str(path))
    proc = subprocess.run(
        [sys.executable, "-c", _SCIP_PROBE, str(path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"SCIP reader rc={proc.returncode}\n{proc.stderr[-2000:]}"
    status, obj = proc.stdout.split()
    assert status == "optimal"
    assert float(obj) == pytest.approx(SQRT2, abs=1e-5)
    assert model.solve().objective == pytest.approx(float(obj), abs=1e-5)


# ── 3. from_nl round trip reaches the same optimum ───────────────────────────


@pytest.mark.parametrize("rust", [True, False], ids=["rust", "python"])
def test_from_nl_round_trip_same_optimum(tmp_path, rust, monkeypatch):
    monkeypatch.setenv("DISCOPT_RUST_NL", "1" if rust else "0")
    model = _lin_first()
    path = tmp_path / "lin_first.nl"
    model.to_nl(str(path))
    reloaded = dm.from_nl(str(path))
    # Row i of the reloaded model is model row nl_row_order(model)[i].
    order = nl_row_order(model)
    assert [reloaded._constraints[i].sense for i in range(2)] == [
        model._constraints[j].sense for j in order
    ]
    r1, r2 = model.solve(), reloaded.solve()
    assert r1.status == r2.status == "optimal"
    assert r1.objective == pytest.approx(SQRT2, abs=1e-6)
    assert r2.objective == pytest.approx(r1.objective, abs=1e-6)


# ── 4. duals and constraint values map back to the right constraints ─────────


def _pounce():
    return pytest.importorskip("pounce")


def _flat_perm(model: dm.Model) -> np.ndarray:
    """``.nl`` column -> model flat variable index."""
    offsets, off = {}, 0
    for v in model._variables:
        offsets[v.name] = off
        off += v.size
    writer = _NLWriter(model)
    writer.write()
    return np.array([offsets[v.name] + e for v, e in writer._flat_vars], dtype=int)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_constraint_values_map_back_through_nl_row_order(tmp_path, name):
    """A reader's per-row output, permuted by ``nl_row_order``, is the model's.

    The reference is discopt's own evaluator on the model, row by row; the
    ``.nl`` side is POUNCE's reader. ``.nl`` bodies carry their constant in the
    bound, so both sides are compared as ``body - rhs`` (the constant cancels).
    """
    pounce = _pounce()
    from discopt._tape_nlp_evaluator import make_evaluator

    model = MODELS[name]()
    path = tmp_path / f"{name}.nl"
    model.to_nl(str(path))
    base = pounce.read_nl(str(path))
    order = nl_row_order(model)
    perm = _flat_perm(model)
    ev = make_evaluator(model)
    lo, hi = ev.variable_bounds
    g_l, g_u = np.asarray(base.g_l), np.asarray(base.g_u)
    rng = np.random.default_rng(1562)
    compared = 0
    for _ in range(3):
        x_model = lo + rng.random(lo.size) * (hi - lo)
        g_nl = np.asarray(base.constraints(x_model[perm]))
        # The evaluator compiles each row as `body - rhs`; on the .nl side the
        # rhs is whichever bound the row carries (both, for an equality).
        g_model = np.asarray(ev.evaluate_constraints(x_model))
        assert g_model.shape == g_nl.shape == order.shape
        for i, j in enumerate(order):
            bound = g_u[i] if abs(g_u[i]) < 1e19 else g_l[i]
            assert g_nl[i] - bound == pytest.approx(g_model[j], abs=1e-9), (i, j)
            compared += 1
    assert compared == 3 * len(order) > 0


def test_duals_map_back_to_the_right_constraint(tmp_path):
    """The active nonlinear row gets the multiplier, the slack linear row none.

    At the optimum x = sqrt(2), y = 0 of the repro, ``x + y <= 3`` is slack
    (dual 0) and ``x^2 + y >= 2`` is active with |dual| = 1 / (2 sqrt(2)). The
    ``.nl`` puts the nonlinear row first, so reading ``mult_g`` in model order
    without :func:`nl_row_order` would hand the multiplier to the wrong row.
    """
    pounce = _pounce()
    model = _lin_first()
    path = tmp_path / "lin_first.nl"
    model.to_nl(str(path))
    base = pounce.read_nl(str(path))
    perm = _flat_perm(model)
    x0_model = np.array([1.5, 0.5])  # in the basin of the global optimum
    x_nl, info = pounce.solve_nlp_batch([base], x0s=[x0_model[perm]])[0]
    assert info["status"] == 0, info.get("status_msg")
    x_model = np.empty_like(x_nl)
    x_model[perm] = x_nl
    np.testing.assert_allclose(x_model, [SQRT2, 0.0], atol=1e-6)

    order = nl_row_order(model)
    duals = np.empty(len(order))
    duals[order] = np.asarray(info["mult_g"])
    assert abs(duals[0]) < 1e-6, f"slack linear row got dual {duals[0]}"
    assert abs(duals[1]) == pytest.approx(1.0 / (2.0 * SQRT2), abs=1e-6)


# ── 5. companion block-label file follows the row permutation ────────────────


def test_block_structure_rows_follow_the_nl_row_order(tmp_path):
    """``to_nl(block_structure_file=...)`` labels ``.nl`` rows, not model rows."""
    from discopt.export.nl import to_nl

    m = dm.Model("blocks_lin_first")
    lin = m.continuous("lin", shape=(2,), lb=0, ub=10)
    nl = m.continuous("nl", shape=(2,), lb=0.1, ub=10)
    shared = m.continuous("shared", lb=0, ub=10)
    m.minimize(dm.log(nl[0]) + dm.log(nl[1]) + lin[0] + lin[1] + shared)
    m.subject_to(lin[1] + shared >= 1, name="r_lin_block1")  # linear, block 1
    m.subject_to(nl[0] ** 2 + lin[0] + shared >= 1, name="r_nl_block0")  # nonlinear, block 0
    m.set_block(lin, [0, 1])
    m.set_block(nl, [0, 1])
    m.set_block(shared, -1)

    nl_path, blocks_path = tmp_path / "m.nl", tmp_path / "m.blocks"
    to_nl(m, nl_path, block_structure_file=blocks_path)
    con_labels = [int(t) for t in blocks_path.read_text().split("\n")[2].split()]
    _hdr, bodies, _r = _parse(nl_path.read_text())
    assert bodies[0] != ["n0"] and bodies[1] == ["n0"]  # .nl row 0 is the nonlinear one
    # .nl row 0 is `r_nl_block0`, row 1 is `r_lin_block1`.
    assert con_labels == [0, 1]


# ── 6. nl_row_order describes the file the DEFAULT writer wrote (#1578 review) ─


def test_review_repro_parameter_row_is_linear_and_unmoved(tmp_path, monkeypatch):
    """The reviewer's exact repro: on main this returned ``[1, 0]`` for a file
    the Rust writer had written in identity order, swapping the two rows' duals.
    """
    monkeypatch.delenv("DISCOPT_RUST_NL", raising=False)
    m = dm.Model("pl")
    x = m.continuous("x", lb=0.5, ub=2)
    y = m.continuous("y", lb=0.5, ub=2)
    m.subject_to(x + y <= 3, name="lin")
    p = m.parameter("p", value=2.0)
    m.subject_to(p * x + y >= 1, name="param_row")
    m.minimize(x + y)
    path = tmp_path / "pl.nl"
    m.to_nl(str(path))
    text = path.read_text()
    (nlc, *_rest), bodies, r_lines = _parse(text)
    assert nlc == 0 and bodies == [["n0"], ["n0"]]
    assert nl_row_order(m).tolist() == [0, 1]
    # .nl row 0 is `x + y <= 3`. Row 1 is `p*x + y >= 1`, which discopt stores
    # as `1 - (p*x + y) <= 0`, so its bound is -1.
    assert r_lines == ["1 3.0", "1 -1.0"]
    # The forced Python writer writes the same file, so the same map holds.
    monkeypatch.setenv("DISCOPT_RUST_NL", "0")
    assert m.to_nl() == text
    assert nl_row_order(m).tolist() == [0, 1]


@pytest.mark.parametrize("name", sorted(MODELS))
def test_nl_row_order_is_the_running_writers_own_map(name, monkeypatch):
    """Default route -> the Rust writer's map; ``DISCOPT_RUST_NL=0`` -> Python's.

    Both must also be the same map, because the two files are byte-identical.
    """
    from discopt.export.nl import _rust_nl_write

    model = MODELS[name]()
    monkeypatch.delenv("DISCOPT_RUST_NL", raising=False)
    written = _rust_nl_write(model, None)
    assert written is not None, "the Rust writer declined -- this case proves nothing"
    default = nl_row_order(model).tolist()
    assert default == written[1]
    monkeypatch.setenv("DISCOPT_RUST_NL", "0")
    forced = nl_row_order(model).tolist()
    writer = _NLWriter(model)
    writer.write()
    assert forced == writer._row_order
    assert default == forced
