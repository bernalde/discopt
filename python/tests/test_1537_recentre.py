"""#1537 C: exact recentring of large-offset variables (``DISCOPT_RECENTRE``).

The pass rebuilds a model whose boxes sit far from the origin in coordinates
``x = z + c`` with every scalar affine subtree constant-folded, solves that, and
maps the result back (``discopt/modeling/_recentre.py``). Default OFF (CLAUDE.md
§5); these tests pin its mechanics and its fixes with the flag ON, and that the
flag OFF changes nothing.
"""

from __future__ import annotations

import os

import discopt.modeling as dm
import numpy as np
import pytest
from _invariance import translate
from discopt.modeling._recentre import (
    RecentreUnsupported,
    plan_shifts,
    recentre,
)

CORPUS = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl")


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv("DISCOPT_RECENTRE", "1")


def _moved(r):
    return (r.solver_stats or {}).get("recentre/variables_moved")


# ── mechanics ─────────────────────────────────────────────────────────────────


def test_selectivity_and_shift_values():
    m = dm.Model("sel")
    a = m.continuous("a", lb=1e6, ub=1e6 + 3)  # moved: |lb| >= 100 * width
    b = m.continuous("b", lb=0, ub=5)  # well scaled: untouched
    c = m.continuous("c", lb=1e3, ub=1e3 + 50)  # 1e3 < 100 * 50: untouched
    i = m.integer("i", lb=-2_000_000, ub=-1_999_996)  # moved, integer shift
    m.minimize(a + b + c + i)
    plan = plan_shifts(m, 100.0)
    assert set(plan) == {id(a), id(i)}
    assert float(plan[id(a)]) == 1e6 and float(plan[id(i)]) == -2e6
    assert recentre(dm.Model("empty")) is None


def test_constant_folding_removes_the_cancelling_constant():
    """``(y - c)`` written by the user becomes exactly ``z``: the unfolded
    ``(z + c) - c`` certified a false infeasible on st_e36 (#1542)."""
    m = dm.Model("fold")
    y = m.continuous("y", lb=1e6, ub=1e6 + 3)
    i = m.integer("i", lb=-2e6, ub=-2e6 + 4)
    m.subject_to((y - 1e6) + 2 * (i + 2e6) <= 5)
    m.minimize((y - 1e6 - 0.3) ** 2 + (i + 2e6) ** 3)
    rc = recentre(m)
    z = {v.name: v for v in rc.model._variables}
    obj = rc.model._objective.expression
    assert obj.right.left is z["i"]  # (i + 2e6) ** 3  ->  i ** 3, constant gone
    assert repr(rc.model._objective.expression.left.left).startswith("Σ")
    for v in rc.model._variables:
        assert float(v.lb) == 0.0
    # exactness at sampled points: f_new(x - c) == f(x)
    from discopt._tape_nlp_evaluator import cached_tape_evaluator

    ev, ev_rc = cached_tape_evaluator(m), cached_tape_evaluator(rc.model)
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = np.array([1e6 + rng.uniform(0, 3), -2e6 + rng.integers(0, 5)])
        zpt = x - np.array([1e6, -2e6])
        assert ev_rc.evaluate_objective(zpt) == pytest.approx(ev.evaluate_objective(x), abs=1e-6)
        assert np.allclose(ev_rc.evaluate_constraints(zpt), ev.evaluate_constraints(x), atol=1e-6)


def test_parameters_are_not_folded_and_stay_live(flag_on):
    m = dm.Model("par")
    y = m.continuous("y", lb=1e6, ub=1e6 + 10)
    p = m.parameter("p", value=3.0)
    m.minimize((y - 1e6 - p) ** 2)
    r1 = m.solve(time_limit=20)
    p.value = np.asarray(7.0)
    r2 = m.solve(time_limit=20)
    assert _moved(r1) == 1.0 and _moved(r2) == 1.0
    assert float(np.asarray(r1.x["y"])) == pytest.approx(1e6 + 3, abs=1e-5)
    assert float(np.asarray(r2.x["y"])) == pytest.approx(1e6 + 7, abs=1e-5)


def test_unsupported_structure_refuses_and_solve_falls_back(flag_on):
    m = dm.Model("gdp")
    x = m.continuous("x", lb=1e6, ub=1e6 + 10)
    m.either_or([[x <= 1e6 + 1], [x >= 1e6 + 9]])
    m.minimize((x - 1e6 - 5) ** 2)
    with pytest.raises(RecentreUnsupported):
        recentre(m)
    r = m.solve(time_limit=20)  # unrecentred, still solved
    assert _moved(r) is None and r.status in ("optimal", "feasible")


def test_mapped_point_is_reverified_against_the_original(flag_on, monkeypatch):
    """A mapping defect cannot publish a certificate: corrupt ``to_outer`` and the
    point the user receives fails their model, so the certificate is withdrawn."""
    import discopt.modeling._recentre as rmod

    real = rmod.Recentring.to_outer

    def off_by_one(self, values):
        return {k: np.asarray(v) + 1.0 for k, v in real(self, values).items()}

    monkeypatch.setattr(rmod.Recentring, "to_outer", off_by_one)
    m = dm.Model("guard")
    y = m.continuous("y", lb=1e6, ub=1e6 + 3)
    m.subject_to(y <= 1e6 + 2.5)
    m.minimize(-(y - 1e6))  # optimum at the row: y = 1e6 + 2.5
    r = m.solve(time_limit=20)
    assert not r.gap_certified and r.status == "feasible"
    assert r.solver_stats.get("recentre/mapped_point_refused") == 1.0


def test_initial_solution_and_warm_start_are_mapped(flag_on):
    m = dm.Model("init")
    y = m.continuous("y", lb=1e6, ub=1e6 + 3)
    m.minimize((y - 1e6 - 1.7) ** 2)
    r = m.solve(time_limit=20, initial_solution={y: 1e6 + 1.0})
    assert r.gap_certified and float(np.asarray(r.x["y"])) == pytest.approx(1e6 + 1.7, abs=1e-5)
    r2 = m.solve(time_limit=20, warm_start=r)
    assert r2.gap_certified and float(np.asarray(r2.x["y"])) == pytest.approx(1e6 + 1.7, abs=1e-5)


def test_flag_off_is_untouched(monkeypatch):
    monkeypatch.delenv("DISCOPT_RECENTRE", raising=False)
    m = dm.Model("off")
    y = m.continuous("y", lb=1e6, ub=1e6 + 3)
    m.minimize((y - 1e6 - 1.7) ** 2)
    assert _moved(m.solve(time_limit=20)) is None


# ── the fixes (flag ON): each was a false certificate or a crash with it OFF ──


@pytest.mark.parametrize("c", [1e5, 1e6])
def test_1543_one_variable_integer_qp(flag_on, c):
    m = dm.Model("sq")
    y = m.integer("y", lb=c, ub=3 + c)
    u = y - c
    m.minimize(6 * u * u - 12 * u)
    r = m.solve(time_limit=20)
    assert r.gap_certified and r.objective == pytest.approx(-6.0, abs=1e-6)
    assert float(np.asarray(r.x["y"])) == pytest.approx(c + 1.0, abs=1e-6)


def test_1542_shifted_cubic(flag_on):
    rng = np.random.default_rng(1)
    A = rng.integers(-4, 5, size=(3, 5)).astype(float)
    b = rng.integers(2, 10, size=3).astype(float)
    cc = rng.integers(-5, 6, size=5).astype(float)
    s = np.array([798491.0, -1314226.0, -1100101.0, -1228561.0, 555147.0])
    m = dm.Model("cube")
    ys = [m.integer(f"i{k}", lb=s[k], ub=4 + s[k]) for k in range(3)] + [
        m.continuous(f"c{k}", lb=s[k], ub=5 + s[k]) for k in range(3, 5)
    ]
    x = [ys[j] - s[j] for j in range(5)]
    for r_ in range(3):
        m.subject_to(sum(A[r_, j] * x[j] for j in range(5)) <= b[r_])
    m.minimize(sum(cc[j] * x[j] for j in range(5)) + 0.2 * x[3] ** 3)
    r = m.solve(time_limit=20)
    assert r.gap_certified and r.objective == pytest.approx(-36.0, abs=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["st_miqp2", "st_miqp3", "st_miqp4", "alan"])
def test_corpus_shift_false_certificates_recover(flag_on, name):
    """The false-certificate / guard-refusal shift failures of the invariance
    corpus panel (#1543) come back certified at the base optimum. st_miqp2 is the
    graduation panel's ON-arm false certificate (262 vs 2): its x1 box
    [-1.3e6, 9.9987e9] slipped past a fixed "open" cutoff and stayed unfolded."""
    m = dm.from_nl(os.path.join(CORPUS, f"{name}.nl"))
    base = m.solve(time_limit=20)
    assert base.gap_certified
    r = translate(m, 1e6, seed=2).solve(time_limit=20)
    assert r.gap_certified and r.status == "optimal"
    assert r.objective == pytest.approx(base.objective, abs=1e-6, rel=1e-4)


@pytest.mark.slow
def test_corpus_nvs09_shift_no_longer_crashes(flag_on):
    """#1544: the shifted nvs09 raised RecursionError. Recentred it solves without
    a crash and without a false certificate; it does not re-certify within 20 s
    (the entry experiment's folded arm lost it too), so that is not asserted."""
    m = dm.from_nl(os.path.join(CORPUS, "nvs09.nl"))
    base = m.solve(time_limit=20)
    r = translate(m, 1e6, seed=2).solve(time_limit=20)
    if r.gap_certified:
        assert r.objective == pytest.approx(base.objective, abs=1e-6, rel=1e-4)


def test_one_sided_offsets_are_moved():
    m = dm.Model("one")
    a = m.continuous("a", lb=1e6, ub=1e15)  # effectively x >= 1e6
    b = m.integer("b", lb=-3e6, ub=1e15)  # ub effectively open (a bare lb gets the 1e6 default box)
    d = m.continuous("d", ub=-5e5)  # lb open: anchored at ub
    e = m.continuous("e", lb=50.0)  # |lb| < ratio: untouched
    m.minimize(a + b - d + e)
    plan = plan_shifts(m, 100.0)
    assert set(plan) == {id(a), id(b), id(d)}
    assert float(plan[id(d)]) == -5e5
