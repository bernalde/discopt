"""Tests for the optional Pyomo ``SolverFactory('discopt')`` plugin.

The bridge round-trips a Pyomo model through a temporary AMPL ``.nl`` file into
discopt and maps the solution back by column order. These tests skip when Pyomo is
not installed (``pip install discopt[pyomo]``).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import pytest  # noqa: E402

pyo = pytest.importorskip("pyomo.environ")

import discopt.pyomo  # noqa: E402,F401  (registers the solver)


@pytest.fixture()
def opt():
    return pyo.SolverFactory("discopt")


def _tiny_minlp(sense=None):
    """min (x-3)^2 + 2y  s.t. x + 5y >= 4 ; x in [0,10], y binary -> x=4, y=0, obj=1."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 10))
    m.y = pyo.Var(domain=pyo.Binary)
    m.obj = pyo.Objective(expr=(m.x - 3) ** 2 + 2 * m.y, sense=sense or pyo.minimize)
    m.c = pyo.Constraint(expr=m.x + 5 * m.y >= 4)
    return m


def test_registration(opt):
    assert "discopt" in pyo.SolverFactory
    assert opt.available() is True


def test_roundtrip_matches_from_nl(opt, tmp_path):
    """The plugin must match a direct from_nl().solve() on the same .nl: same status,
    objective, and variable values aligned by column order (names differ)."""
    import discopt.modeling as dm

    m = _tiny_minlp()
    res = opt.solve(m)
    assert res.solver.termination_condition == pyo.TerminationCondition.optimal
    assert pyo.value(m.obj) == pytest.approx(1.0, abs=1e-3)
    assert m.x.value == pytest.approx(4.0, abs=1e-3)
    assert m.y.value == 0  # exact integer, not 1e-9 drift

    # Independent reference: write the same model and solve via from_nl directly.
    from pyomo.repn.plugins.nl_writer import NLWriter

    nl = str(tmp_path / "ref.nl")
    with open(nl, "w") as f:
        info = NLWriter().write(m, f, linear_presolve=False, scale_model=False)
    ref = dm.from_nl(nl).solve(time_limit=30, gap_tolerance=1e-4)
    assert ref.status == "optimal"
    assert ref.objective == pytest.approx(pyo.value(m.obj), rel=1e-4, abs=1e-6)
    # Column-order alignment: the i-th .nl column value equals the plugin-loaded var.
    flat = []
    import numpy as np

    dref = dm.from_nl(nl)
    for v in dref._variables:
        flat.extend(np.asarray(ref.x[v.name]).ravel())
    pyomo_vals = [info.variables[i].value for i in range(len(info.variables))]
    assert pyomo_vals == pytest.approx(flat, abs=1e-3)


def test_maximize_sense(opt):
    """max -(x-3)^2 over [0,10] -> x=3, obj=0 (no sign flip)."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 10))
    m.o = pyo.Objective(expr=-((m.x - 3) ** 2), sense=pyo.maximize)
    res = opt.solve(m)
    assert res.solver.termination_condition == pyo.TerminationCondition.optimal
    assert m.x.value == pytest.approx(3.0, abs=1e-3)
    assert pyo.value(m.o) == pytest.approx(0.0, abs=1e-3)


def test_integer_rounding(opt):
    """An integral optimum loads as an exact integer."""
    m = _tiny_minlp()
    opt.solve(m)
    assert m.y.value in (0, 1)
    assert float(m.y.value).is_integer()


def test_infeasible(opt):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 1))
    m.o = pyo.Objective(expr=m.x)
    m.c = pyo.Constraint(expr=m.x >= 2)
    res = opt.solve(m)
    assert res.solver.termination_condition == pyo.TerminationCondition.infeasible


def test_options_passthrough(opt, monkeypatch):
    """Pyomo's `timelimit`/options reach Model.solve as the right kwargs."""
    import discopt.modeling as dm

    captured = {}
    orig = dm.Model.solve

    def spy(self, *a, **k):
        captured.update(k)
        return orig(self, *a, **k)

    monkeypatch.setattr(dm.Model, "solve", spy)
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 5))
    m.o = pyo.Objective(expr=m.x)
    opt.solve(m, options={"timelimit": 7, "mipgap": 1e-3})
    assert captured.get("time_limit") == 7
    assert captured.get("gap_tolerance") == 1e-3


def test_tee_matches_no_tee(opt, capsys):
    """`tee=True` solves identically to `tee=False` and streams the log (issue #1178).

    Before the fix, `tee` was mapped onto `Model.solve(stream=True)` — a different
    feature (an iterator of SolveUpdate) that raises NotImplementedError — so every
    `tee=True` solve came back as termination `error` with no solution loaded.
    """
    quiet = _tiny_minlp()
    res_quiet = opt.solve(quiet, tee=False)
    capsys.readouterr()  # drop anything the quiet solve emitted

    loud = _tiny_minlp()
    res_loud = opt.solve(loud, tee=True)
    out = capsys.readouterr().out

    assert res_loud.solver.termination_condition == res_quiet.solver.termination_condition
    assert res_loud.solver.termination_condition == pyo.TerminationCondition.optimal
    assert pyo.value(loud.obj) == pytest.approx(pyo.value(quiet.obj), abs=1e-6)
    assert loud.x.value == pytest.approx(quiet.x.value, abs=1e-6)
    assert loud.y.value == pytest.approx(quiet.y.value, abs=1e-6)
    assert out.strip(), "tee=True produced no solver log on stdout"


def test_tee_attaches_a_stdout_handler_then_restores_it(opt, monkeypatch):
    """`tee=True` attaches a stdout handler for the solve and removes it after."""
    import logging
    import sys

    import discopt.modeling as dm

    discopt_logger = logging.getLogger("discopt")
    before_handlers = list(discopt_logger.handlers)
    before_level = discopt_logger.level

    during: dict[str, object] = {}
    orig = dm.Model.solve

    def spy(self, *a, **k):
        added = [h for h in discopt_logger.handlers if h not in before_handlers]
        during["added"] = added
        during["effective_level"] = discopt_logger.getEffectiveLevel()
        return orig(self, *a, **k)

    monkeypatch.setattr(dm.Model, "solve", spy)
    opt.solve(_tiny_minlp(), tee=True)

    assert "added" in during, "Model.solve was never reached — the probe measured nothing"
    added = during["added"]
    assert len(added) == 1, f"expected exactly one tee handler, saw {added}"
    assert isinstance(added[0], logging.StreamHandler)
    assert added[0].stream is sys.stdout
    assert during["effective_level"] <= logging.INFO

    # ...and the logger is left exactly as it was found.
    assert discopt_logger.handlers == before_handlers
    assert discopt_logger.level == before_level


def test_tee_does_not_request_streaming(opt, monkeypatch):
    """`tee=True` must not reach `Model.solve` as `stream=True` (issue #1178)."""
    import discopt.modeling as dm

    captured = {}
    orig = dm.Model.solve

    def spy(self, *a, **k):
        captured.update(k)
        return orig(self, *a, **k)

    monkeypatch.setattr(dm.Model, "solve", spy)
    opt.solve(_tiny_minlp(), tee=True)
    assert "stream" not in captured


def test_duals_when_exposed(opt):
    """Convex NLP min (x-3)^2 s.t. x>=4 -> KKT multiplier ~2; sign matches the
    AMPL/Pyomo convention (cross-checked against ipopt's value 2.0)."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 10))
    m.o = pyo.Objective(expr=(m.x - 3) ** 2, sense=pyo.minimize)
    m.c = pyo.Constraint(expr=m.x >= 4)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    res = opt.solve(m)
    assert res.solver.termination_condition == pyo.TerminationCondition.optimal
    assert m.x.value == pytest.approx(4.0, abs=1e-3)
    d = m.dual.get(m.c)
    assert d is not None, "discopt exposed duals but the plugin did not load them"
    assert d == pytest.approx(2.0, abs=1e-3)


def test_from_pyomo_matches_solver_plugin(opt):
    """`from_pyomo(m).solve()` must reach the same optimum as `SolverFactory('discopt')`
    on the same model (the issue #381 round-trip acceptance test)."""
    import discopt.modeling as dm

    # Reference: solve via the registered Pyomo solver plugin.
    m_ref = _tiny_minlp()
    res = opt.solve(m_ref)
    assert res.solver.termination_condition == pyo.TerminationCondition.optimal
    ref_obj = pyo.value(m_ref.obj)

    # Import the same model into a discopt Model and solve natively.
    m_imp = _tiny_minlp()
    dmodel = dm.from_pyomo(m_imp)
    assert isinstance(dmodel, dm.Model)
    r = dmodel.solve(time_limit=30, gap_tolerance=1e-4)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(ref_obj, rel=1e-4, abs=1e-6)
    assert r.objective == pytest.approx(1.0, abs=1e-3)


def test_from_pyomo_indexed_transportation():
    """A small indexed (Var/Constraint over sets) LP imports and solves correctly."""
    import discopt.modeling as dm

    supply = {0: 20.0, 1: 30.0}
    demand = {0: 10.0, 1: 25.0, 2: 15.0}
    cost = {(0, 0): 2.0, (0, 1): 3.0, (0, 2): 1.0, (1, 0): 5.0, (1, 1): 4.0, (1, 2): 8.0}

    m = pyo.ConcreteModel()
    m.P = pyo.RangeSet(0, 1)
    m.K = pyo.RangeSet(0, 2)
    m.ship = pyo.Var(m.P, m.K, domain=pyo.NonNegativeReals)
    m.obj = pyo.Objective(expr=sum(cost[i, j] * m.ship[i, j] for i in m.P for j in m.K))
    m.sup = pyo.Constraint(m.P, rule=lambda mm, i: sum(mm.ship[i, j] for j in mm.K) <= supply[i])
    m.dem = pyo.Constraint(m.K, rule=lambda mm, j: sum(mm.ship[i, j] for i in mm.P) >= demand[j])

    dmodel = dm.from_pyomo(m)
    r = dmodel.solve(time_limit=30, gap_tolerance=1e-4)
    assert r.status == "optimal"

    # Independent reference via the registered plugin.
    opt = pyo.SolverFactory("discopt")
    m2 = m.clone()
    opt.solve(m2)
    assert r.objective == pytest.approx(pyo.value(m2.obj), rel=1e-4, abs=1e-6)


def test_from_pyomo_no_variables_raises():
    """A variable-free Pyomo model has nothing to import -> ValueError."""
    import discopt.modeling as dm

    m = pyo.ConcreteModel()
    m.o = pyo.Objective(expr=1.0)
    with pytest.raises(ValueError, match="no variables"):
        dm.from_pyomo(m)


def test_duals_graceful_for_integer_model(opt):
    """Solving an integer model with a `dual` Suffix declared must not error,
    whether or not discopt exposes multipliers (it may surface relaxation duals).
    Any loaded value must be a finite number, never garbage."""
    import math

    m = pyo.ConcreteModel()
    m.y = pyo.Var(domain=pyo.Binary)
    m.o = pyo.Objective(expr=m.y, sense=pyo.minimize)
    m.c = pyo.Constraint(expr=m.y >= 0)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    res = opt.solve(m)
    assert res.solver.termination_condition in (
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.feasible,
    )
    d = m.dual.get(m.c)
    assert d is None or math.isfinite(d)  # absent or finite — never fabricated/NaN


# -- #1559: solve() kwargs are never silently dropped ---------------------------


def _spy_solve(monkeypatch):
    """Record the kwargs the plugin hands to ``Model.solve``."""
    import discopt.modeling as dm

    captured: dict = {}
    orig = dm.Model.solve

    def spy(self, *a, **k):
        captured.update(k)
        return orig(self, *a, **k)

    monkeypatch.setattr(dm.Model, "solve", spy)
    return captured


def _box(sense=None):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(1, 4))
    m.o = pyo.Objective(expr=m.x, sense=sense or pyo.minimize)
    return m


def test_unknown_kwarg_raises(opt):
    """An unrecognised keyword is a TypeError, as in Pyomo's own HiGHS plugins (#1559).

    Before the fix it was popped by nobody and the solve ran as if it were absent.
    """
    with pytest.raises(TypeError, match="bogus_option"):
        opt.solve(_box(), bogus_option=3)


def test_time_limit_alias_is_honoured(opt, monkeypatch):
    """`time_limit=` (discopt's own spelling) reaches Model.solve (#1559).

    Before the fix it was dropped and the solve ran at the 3600 s default.
    """
    captured = _spy_solve(monkeypatch)
    opt.solve(_box(), time_limit=11)
    assert captured.get("time_limit") == 11


def test_timelimit_still_honoured(opt, monkeypatch):
    captured = _spy_solve(monkeypatch)
    opt.solve(_box(), timelimit=13)
    assert captured.get("time_limit") == 13


def test_time_limit_and_timelimit_agreeing_is_accepted(opt, monkeypatch):
    captured = _spy_solve(monkeypatch)
    opt.solve(_box(), timelimit=5, time_limit=5)
    assert captured.get("time_limit") == 5


def test_time_limit_and_timelimit_conflict_raises(opt):
    with pytest.raises(ValueError, match="time limit"):
        opt.solve(_box(), timelimit=5, time_limit=9)


def test_timelimit_kwarg_conflicting_with_options_raises(opt):
    """A kwarg time limit that differs from one in `options` must not be dropped."""
    with pytest.raises(ValueError, match="time limit"):
        opt.solve(_box(), timelimit=5, options={"time_limit": 9})


def test_pyomo_standard_suffixes_kwarg_accepted(opt):
    """`suffixes=['dual']` is part of the legacy solve() contract and stays accepted."""
    res = opt.solve(_box(), suffixes=["dual"])
    assert res.solver.termination_condition == pyo.TerminationCondition.optimal


def test_unsupported_suffix_request_raises(opt):
    with pytest.raises(ValueError, match="slack"):
        opt.solve(_box(), suffixes=["slack"])


# -- #1560: incumbent and dual bound land in Pyomo's fields --------------------


@pytest.mark.parametrize("sense", [pyo.minimize, pyo.maximize])
def test_bound_fields_at_optimality(opt, sense):
    """Pyomo convention (HiGHS/appsi): minimize -> upper=incumbent, lower=dual bound;
    maximize -> lower=incumbent, upper=dual bound. At optimality lower <= upper.

    Before the fix the fields were swapped, visible here as lower > upper by ~1e-12.
    """
    res = opt.solve(_box(sense))
    assert res.solver.termination_condition == pyo.TerminationCondition.optimal
    lo, up = res.problem.lower_bound, res.problem.upper_bound
    expected = 1.0 if sense == pyo.minimize else 4.0
    assert lo == pytest.approx(expected, abs=1e-6)
    assert up == pytest.approx(expected, abs=1e-6)
    assert lo <= up


@pytest.mark.parametrize("sense", [pyo.minimize, pyo.maximize])
def test_bound_fields_match_solveresult(opt, monkeypatch, sense):
    """The incumbent and the dual bound go to distinct, sense-dependent fields.

    Uses a fabricated result with objective != bound so a swap cannot hide inside a
    solver tolerance.
    """
    import discopt.modeling as dm

    def fake_solve(self, **k):
        if sense == pyo.minimize:
            return dm.SolveResult(status="feasible", objective=3.0, bound=1.5, bound_valid=True)
        return dm.SolveResult(status="feasible", objective=1.5, bound=3.0, bound_valid=True)

    monkeypatch.setattr(dm.Model, "solve", fake_solve)
    res = opt.solve(_box(sense))
    assert res.problem.lower_bound == 1.5
    assert res.problem.upper_bound == 3.0


@pytest.mark.parametrize("sense", [pyo.minimize, pyo.maximize])
def test_timeout_without_incumbent_leaves_incumbent_unset(opt, monkeypatch, sense):
    """A time-limited exit with a valid dual bound but no incumbent (#1560).

    Before the fix a minimize reported upper_bound = the dual bound, i.e. claimed a
    feasible point that was never found, and left lower_bound at -inf.
    """
    import math

    import discopt.modeling as dm

    bound = -2.03 if sense == pyo.minimize else 7.5

    def fake_solve(self, **k):
        return dm.SolveResult(
            status="time_limit",
            objective=None,
            bound=bound,
            x=None,
            bound_valid=True,
            bound_source="bnb_tree",
        )

    monkeypatch.setattr(dm.Model, "solve", fake_solve)
    res = opt.solve(_box(sense))
    assert res.solver.termination_condition == pyo.TerminationCondition.maxTimeLimit
    if sense == pyo.minimize:
        assert res.problem.lower_bound == bound
        assert res.problem.upper_bound == math.inf
    else:
        assert res.problem.upper_bound == bound
        assert res.problem.lower_bound == -math.inf


def test_unvalidated_bound_is_not_reported_as_dual_bound(opt, monkeypatch):
    """A `bound` discopt does not assert valid (`bound_valid=False`, e.g. a local
    solve) must not be published as Pyomo's dual bound."""
    import math

    import discopt.modeling as dm

    def fake_solve(self, **k):
        return dm.SolveResult(status="time_limit", objective=None, bound=-2.03, x=None)

    monkeypatch.setattr(dm.Model, "solve", fake_solve)
    res = opt.solve(_box(pyo.minimize))
    assert res.problem.lower_bound == -math.inf
    assert res.problem.upper_bound == math.inf


# -- zero-variable models: optimal only when every constraint holds ------------


def _constant_model(con_rule):
    m = pyo.ConcreteModel()
    m.p = pyo.Param(mutable=True, initialize=1.0)
    m.c = pyo.Constraint(rule=con_rule)
    m.o = pyo.Objective(expr=m.p)
    return m


def test_constant_model_infeasible_constraint_is_infeasible(opt):
    """A zero-variable model whose constant constraint fails (1 >= 2) is infeasible.

    Before the fix the trivial path reported `optimal` without looking at the
    constraints, and (with the #1560 field fix) published 1.0 as a certified bound.
    """
    import math

    res = opt.solve(_constant_model(lambda m: m.p >= 2))
    assert res.solver.termination_condition == pyo.TerminationCondition.infeasible
    assert res.problem.lower_bound == -math.inf
    assert res.problem.upper_bound == math.inf


def test_constant_model_feasible_constraint_is_optimal(opt):
    res = opt.solve(_constant_model(lambda m: m.p >= 0))
    assert res.solver.termination_condition == pyo.TerminationCondition.optimal
    assert res.problem.lower_bound == 1.0
    assert res.problem.upper_bound == 1.0


def test_constant_model_exact_equality_is_optimal(opt):
    res = opt.solve(_constant_model(lambda m: m.p == 1.0))
    assert res.solver.termination_condition == pyo.TerminationCondition.optimal
    assert res.problem.lower_bound == 1.0
    assert res.problem.upper_bound == 1.0


def test_constant_model_violated_equality_is_infeasible(opt):
    res = opt.solve(_constant_model(lambda m: m.p == 1.5))
    assert res.solver.termination_condition == pyo.TerminationCondition.infeasible


def test_constant_model_fixed_var_infeasible(opt):
    """A model whose only variable is fixed also takes the zero-column path."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 10))
    m.x.fix(3.0)
    m.c = pyo.Constraint(expr=m.x <= 2)
    m.o = pyo.Objective(expr=m.x)
    res = opt.solve(m)
    assert res.solver.termination_condition == pyo.TerminationCondition.infeasible


def test_constant_model_unevaluable_constraint_is_error(opt):
    """A constant constraint whose body cannot be evaluated is an error, never optimal.

    End to end, Pyomo's NL writer rejects `log(-1)` first; the helper test below
    pins the trivial path's own refusal.
    """
    m = pyo.ConcreteModel()
    m.p = pyo.Param(mutable=True, initialize=-1.0)
    m.c = pyo.Constraint(expr=pyo.log(m.p) <= 5)
    m.o = pyo.Objective(expr=m.p)
    res = opt.solve(m)
    assert res.solver.termination_condition == pyo.TerminationCondition.error


@pytest.mark.parametrize("p0", [-1.0, float("nan")])
def test_constant_constraint_check_refuses_unevaluable_body(p0):
    """The zero-variable feasibility check raises rather than skipping a row it
    cannot evaluate (a domain error, or a non-finite value)."""
    from discopt.pyomo.solver import DiscoptSolver

    m = pyo.ConcreteModel()
    m.p = pyo.Param(mutable=True, initialize=p0)
    m.c = pyo.Constraint(expr=pyo.log(m.p) <= 5 if p0 < 0 else m.p <= 5)
    with pytest.raises(ValueError):
        DiscoptSolver._violated_constant_constraints(m)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
