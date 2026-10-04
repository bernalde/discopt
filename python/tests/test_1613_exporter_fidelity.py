"""#1613: an exporter/reader must write the user's model or refuse loudly.

Three witnessed ways the model changed on the way out of (or into) discopt:

* **C-03a** -- MPS/LP are whitespace-tokenised; a name with a space became two
  tokens and HiGHS read a different model (objective 8 vs the true 24). Names
  are now mapped into each format's legal alphabet *and* de-duplicated, so two
  distinct names can never merge into one column or row.
* **C-03c** -- ``from_gams`` read ``Semicont``/``Semiint`` as plain variables,
  dropping ``x = 0 or lo <= x <= up``; ``SOS1``/``SOS2`` variables fell through
  to ``free``; and an indexed bound with a parameter right-hand side
  (``x.lo(i) = lo(i)``) was silently dropped. Semicontinuity is now lowered
  exactly, SOS variables are refused, and unevaluable bounds raise.
* **B-15b** -- the ``.nl`` writers emitted a sum as ``n - 1`` nested ``o0``,
  which stack-overflowed a downstream reader (the POUNCE CLI) at n = 2000. A
  chain of three or more ``+`` is now one n-ary ``o54``, in both writers.

Each check is a round-trip: export and re-read with the target tool (HiGHS,
discopt's own ``.nl`` reader, POUNCE when installed), compare against the model.
Reference optima for the GAMS cases were taken from GAMS 53.2 / SCIP on the same
source (BARON refuses semicontinuous variables).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import textwrap

import discopt.modeling as dm
import pytest
from discopt.export._common import legal_unique_names
from discopt.modeling.gams_parser import GamsParseError, parse_gams

highspy = pytest.importorskip("highspy")


# ── C-03a: MPS / LP names ─────────────────────────────────────────────────────

_NASTY = [
    "x[LSR naphtha,regular]",
    "a b",
    "a_b",
    "free",
    "inf",
    "info",
    "nan",
    "end",
    "st",
    "bin",
    "1st",
    ".dot",
    "p+q",
    "r-s",
    "t*u^2",
    "v<=w",
    "c:d",
    "a/b",
    "OBJ",
    "obj",
    "x_0",
    "ünï",
]


def _highs_read(model, fmt, tmp_path):
    p = tmp_path / f"m.{fmt}"
    getattr(model, f"to_{fmt}")(str(p))
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    assert h.readModel(str(p)) == highspy.HighsStatus.kOk, p.read_text()
    h.run()
    lp = h.getLp()
    return h.getInfo().objective_function_value, lp.num_col_, lp.num_row_


@pytest.mark.parametrize("fmt", ["mps", "lp"])
def test_issue_witness_space_in_name_round_trips_through_highs(fmt, tmp_path):
    m = dm.Model("mps")
    a = m.continuous("x[LSR naphtha,regular]", lb=0, ub=10)
    b = m.continuous("y", lb=0, ub=10)
    m.subject_to(a + b <= 8, name="cap row")
    m.maximize(3 * a + b)
    obj, ncol, nrow = _highs_read(m, fmt, tmp_path)
    assert (ncol, nrow) == (2, 1)
    assert obj == pytest.approx(24.0)


def _nasty_model(clean: bool):
    m = dm.Model("nasty")
    vs = []
    for k, nm in enumerate(_NASTY):
        nm = f"c{k}" if clean else nm
        vs.append(m.integer(nm, lb=0, ub=5) if k % 3 == 0 else m.continuous(nm, lb=0, ub=10))
    # An array `x` flattens to `x_0`, `x_1` -- colliding with the scalar `x_0`.
    arr = m.continuous("X" if clean else "x", shape=(2,), lb=0, ub=3)
    for k in range(len(vs) - 1):
        cn = f"r{k}" if clean else _NASTY[(k + 3) % len(_NASTY)]
        m.subject_to(vs[k] + 2 * vs[k + 1] + arr[k % 2] <= 7 + k, name=cn)
    m.subject_to(dm.sum(vs) <= 25, name="r_tot" if clean else "OBJ")
    m.maximize(sum((k % 7 + 1) * vs[k] for k in range(len(vs))) + arr[0] - arr[1])
    return m


@pytest.mark.parametrize("fmt", ["mps", "lp"])
@pytest.mark.filterwarnings("ignore:Duplicate constraint name")
def test_hostile_names_export_the_same_model_as_clean_names(fmt, tmp_path):
    clean = _highs_read(_nasty_model(True), fmt, tmp_path)
    nasty = _highs_read(_nasty_model(False), fmt, tmp_path)
    assert nasty[1:] == clean[1:]  # no merged column or row
    assert nasty[0] == pytest.approx(clean[0])
    assert nasty[0] == pytest.approx(_nasty_model(False).solve().objective)


@pytest.mark.parametrize("fmt", ["mps", "lp"])
def test_legal_unique_names_never_merges(fmt):
    raw = _NASTY + ["a b", "a_b_1", "a_b"]
    out = legal_unique_names(raw, fmt, ("OBJ", "obj"))
    assert len(set(out)) == len(out) == len(raw)
    assert not {"OBJ", "obj"} & set(out)
    assert all(" " not in s and s for s in out)
    # Already-legal names are kept verbatim.
    assert legal_unique_names(["x", "y_1", "flow"], fmt) == ["x", "y_1", "flow"]


# ── C-03c: GAMS reader ────────────────────────────────────────────────────────

_SEMI_CASES = {
    # x in {0} U [2, 10]: min (x - 0.8)^2 is 0.64 at x = 0.
    "scalar": (
        """
        Semicont Variable x; Variable obj;
        x.lo = 2; x.up = 10;
        Equation e; e.. obj =e= sqr(x - 0.8);
        Model mm /all/; Solve mm using minlp minimizing obj;
        """,
        0.64,
    ),
    # Per-element bounds from a parameter; b's box contains 0 (no switch).
    "array": (
        """
        Set i /a, b, c/;
        Parameter lo(i) /a 2, b 0, c 1.5/;
        Semicont Variable x(i); Variable obj;
        x.lo(i) = lo(i); x.up(i) = 5;
        Equation e, c; e.. obj =e= sum(i, sqr(x(i) - 1)); c.. sum(i, x(i)) =g= 2.5;
        Model mm /all/; Solve mm using minlp minimizing obj;
        """,
        1.25,
    ),
    "semiint": (
        """
        Semiint Variable n; Variable obj;
        n.lo = 3; n.up = 9;
        Equation e; e.. obj =e= sqr(n - 1.4);
        Model mm /all/; Solve mm using minlp minimizing obj;
        """,
        1.96,
    ),
}


@pytest.mark.parametrize("case", sorted(_SEMI_CASES))
def test_semicontinuous_matches_gams_scip(case):
    src, ref = _SEMI_CASES[case]
    m = parse_gams(textwrap.dedent(src))
    r = m.solve(time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(ref, abs=1e-6)


def test_semicontinuous_zero_is_feasible_and_gap_is_excluded():
    src = """
    Semicont Variable x; Variable obj;
    x.lo = 2; x.up = 10;
    Equation e; e.. obj =e= x;
    Model mm /all/; Solve mm using minlp maximizing obj;
    """
    m = parse_gams(textwrap.dedent(src))
    assert m.solve(time_limit=60).objective == pytest.approx(10.0)
    m.subject_to(m._variables[0] <= 1.5)  # only x = 0 is left
    assert m.solve(time_limit=60).objective == pytest.approx(0.0, abs=1e-7)


def test_indexed_parameter_bound_is_applied():
    m = parse_gams(
        textwrap.dedent(
            """
            Set i /a, b/;
            Parameter lo(i) /a 2, b 3/;
            Positive Variable x(i); Variable obj;
            x.lo(i) = lo(i);
            Equation e; e.. obj =e= sum(i, x(i));
            Model mm /all/; Solve mm using lp minimizing obj;
            """
        )
    )
    x = next(v for v in m._variables if v.name == "x")
    assert list(x.lb) == [2.0, 3.0]


def test_semicontinuous_without_finite_upper_bound_refuses():
    with pytest.raises(GamsParseError, match="exact bounded lowering"):
        parse_gams(
            "Semicont Variable x; Variable obj; x.lo = 2; Equation e; e.. obj =e= x;"
            " Model mm /all/; Solve mm using minlp minimizing obj;"
        )


@pytest.mark.parametrize("kind", ["SOS1", "SOS2"])
def test_sos_variables_refuse(kind):
    with pytest.raises(GamsParseError, match="SOS restriction"):
        parse_gams(
            f"Set i /1*3/; {kind} Variable s(i); Variable obj;"
            " Equation e; e.. obj =e= sum(i, s(i));"
            " Model mm /all/; Solve mm using mip minimizing obj;"
        )


# ── B-15b: .nl sums ───────────────────────────────────────────────────────────


def _nested_add_pairs(text: str) -> int:
    """Number of `o0` lines immediately followed by another `o0` line."""
    lines = text.split("\n")
    return sum(1 for a, b in zip(lines, lines[1:]) if a == "o0" and b == "o0")


def _python_nl(model) -> str:
    prev = os.environ.get("DISCOPT_RUST_NL")
    os.environ["DISCOPT_RUST_NL"] = "0"
    try:
        return model.to_nl()
    finally:
        if prev is None:
            os.environ.pop("DISCOPT_RUST_NL", None)
        else:
            os.environ["DISCOPT_RUST_NL"] = prev


def test_issue_witness_sum_is_one_sumlist_in_both_writers():
    m = dm.Model("s")
    x = m.continuous("x", shape=(8,), lb=0, ub=1)
    m.minimize(dm.sum(dm.exp(x[i]) for i in range(8)))
    text = m.to_nl()
    body = text.split("O0")[1]
    assert body.split("\n")[1:3] == ["o54", "8"]
    assert "o0\n" not in body
    assert _python_nl(m) == text


def test_plus_chain_inside_a_function_is_one_sumlist():
    m = dm.Model("s")
    x = m.continuous("x", shape=(5,), lb=0.1, ub=1)
    # Python's `+` builds a left-nested chain under `log`; it is not split.
    m.minimize(dm.log(x[0] * x[1] + x[1] * x[2] + x[2] * x[3] + x[3] * x[4]))
    text = m.to_nl()
    assert _nested_add_pairs(text) == 0
    assert "\no54\n4\n" in text
    assert _python_nl(m) == text


def test_two_term_sum_keeps_binary_add():
    m = dm.Model("s")
    x = m.continuous("x", shape=(2,), lb=0, ub=1)
    m.minimize(dm.exp(x[0]) + dm.exp(x[1]))
    text = m.to_nl()
    assert "o54" not in text and "o0\n" in text
    assert _python_nl(m) == text


def test_deep_sum_round_trips_through_nl_reader(tmp_path):
    n = 2000
    m = dm.Model("deep")
    x = m.continuous("x", shape=(n,), lb=-2, ub=2)
    m.minimize(dm.sum(dm.exp(x[i]) + (x[i] - 1) ** 2 for i in range(n)))
    p = tmp_path / "deep.nl"
    m.to_nl(str(p))
    text = p.read_text()
    assert _nested_add_pairs(text) == 0
    # exp(x_i) and (x_i - 1)**2 are both nonlinear terms: one 2n-ary sum.
    assert f"\no54\n{2 * n}\n" in text
    back = dm.from_nl(str(p))
    r0 = m.solve(time_limit=120)
    r1 = back.solve(time_limit=120)
    assert r1.objective == pytest.approx(r0.objective, rel=1e-8)

    pounce = shutil.which("pounce")
    if pounce is not None:
        # The witness's downstream reader: on the o0 chain it died with
        # "thread 'main' has overflowed its stack" at exactly this size.
        res = subprocess.run(
            [pounce, str(p)], capture_output=True, text=True, timeout=300, cwd=tmp_path
        )
        assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
