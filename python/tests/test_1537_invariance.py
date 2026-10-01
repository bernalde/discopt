"""#1537 workstream B: certificates must be invariant under a change of variables.

``x = y - c`` and a positive row rescaling leave every model mathematically
identical, so a certified answer before and after must agree (helpers in
``_invariance.py``). The September adversarial rounds never tested this; the first
4-model probe found a P0 on plain linear MILPs (#1536).

Three layers:

* ``test_harness_*`` -- the instrument's own self-test: the rebuilt model agrees
  with the original at sampled points, so a disagreement in the panels below is
  the solver's, never the harness's (CLAUDE.md §6).
* ``test_generated_panel`` -- seeded small models of each family, every PR
  (``correctness``, not ``slow``: the ``python-correctness`` lane).
* ``test_corpus_panel`` -- the in-repo MINLPLib ``.nl`` corpus, ``slow`` +
  ``correctness``: the dispatch-only ``python-correctness-slow`` lane, run before
  a release or when touching the certificate path.

Only FALSE certificates fail a test. A *lost* certificate (the transformed solve
is honest but uncertified) is a performance defect, measured and printed but not
asserted -- e.g. today every row-scaled MILP loses its certificate to the #1295
guard (#1537 workstream C), which is honest.
"""

from __future__ import annotations

import glob
import os

import discopt.modeling as dm
import numpy as np
import pytest
from _invariance import certified_answer_changed, rescale_rows, translate
from discopt._tape_nlp_evaluator import cached_tape_evaluator
from discopt.validation.feasibility import verify_point

CORPUS = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "data", "minlplib_nl", "*.nl")))


# ── generated families ────────────────────────────────────────────────────────


def _lin_rows(m, x, rng, n_rows=3):
    A = rng.integers(-4, 5, size=(n_rows, len(x))).astype(float)
    b = rng.integers(2, 10, size=n_rows).astype(float)
    for r in range(n_rows):
        m.subject_to(sum(A[r, j] * x[j] for j in range(len(x))) <= b[r])
    return rng.integers(-5, 6, size=len(x)).astype(float)


def _mixed(m):
    return [m.integer(f"i{k}", lb=0, ub=4) for k in range(3)] + [
        m.continuous(f"c{k}", lb=0, ub=5) for k in range(2)
    ]


def milp(seed):
    rng = np.random.default_rng(seed)
    m = dm.Model(f"milp{seed}")
    x = _mixed(m)
    c = _lin_rows(m, x, rng)
    m.minimize(sum(c[j] * x[j] for j in range(5)))
    return m


def bilinear(seed):
    rng = np.random.default_rng(seed)
    m = dm.Model(f"bilin{seed}")
    x = _mixed(m)
    c = _lin_rows(m, x, rng)
    m.subject_to(x[3] * x[4] <= 6)
    m.minimize(sum(c[j] * x[j] for j in range(5)) + 0.5 * x[0] * x[3] - 0.3 * x[1] * x[4])
    return m


def polynomial(seed):
    rng = np.random.default_rng(seed)
    m = dm.Model(f"poly{seed}")
    x = _mixed(m)
    c = _lin_rows(m, x, rng)
    m.minimize(sum(c[j] * x[j] for j in range(5)) + 0.2 * x[3] ** 3 - x[4] ** 2)
    return m


def convex_nlp(seed):
    rng = np.random.default_rng(seed)
    m = dm.Model(f"cvx{seed}")
    x = [m.continuous(f"x{k}", lb=-3, ub=3) for k in range(3)]
    a = rng.uniform(-1, 1, size=3)
    m.subject_to(x[0] + x[1] + x[2] >= 1)
    m.minimize(sum((x[k] - a[k]) ** 2 for k in range(3)) + dm.exp(0.3 * x[0]))
    return m


def maximize_milp(seed):
    m = milp(seed)
    m._objective = type(m._objective)(
        expression=-m._objective.expression, sense=dm.core.ObjectiveSense.MAXIMIZE
    )
    return m


FAMILIES = {
    "milp": milp,
    "maximize_milp": maximize_milp,
    "bilinear": bilinear,
    "polynomial": polynomial,
    "convex_nlp": convex_nlp,
}
TRANSFORMS = {
    "shift1e3": lambda m: translate(m, 1e3, seed=1),
    "shift1e6": lambda m: translate(m, 1e6, seed=2),
    "rows1e-3": lambda m: rescale_rows(m, 1e-3),
    "rows1e6": lambda m: rescale_rows(m, 1e6),
}


# ── the instrument's self-test ────────────────────────────────────────────────


def _box_samples(model, rng, k):
    lo = np.concatenate([np.ravel(v.lb) for v in model._variables])
    hi = np.concatenate([np.ravel(v.ub) for v in model._variables])
    lo, hi = np.maximum(lo, -10.0), np.minimum(hi, 10.0)
    hi = np.maximum(hi, lo)
    is_int = np.concatenate(
        [np.full(v.size, v.var_type is not dm.core.VarType.CONTINUOUS) for v in model._variables]
    )
    for _ in range(k):
        x = rng.uniform(lo, hi)
        x[is_int] = np.round(x[is_int])
        yield np.clip(x, lo, hi)


def _harness_models():
    for name, fam in FAMILIES.items():
        yield name, fam(0)
    for f in CORPUS[::6]:  # a spread of the corpus; the full set is the slow panel's job
        yield os.path.basename(f), dm.from_nl(f)


def _evaluator(model):
    ev = cached_tape_evaluator(model)
    assert ev is not None, f"{model.name}: no tape evaluator -- the self-test cannot evaluate it"
    return ev


@pytest.mark.parametrize(
    "name, model", list(_harness_models()), ids=lambda v: v if isinstance(v, str) else ""
)
def test_harness_preserves_the_model(name, model):
    """At sampled box points (feasible or not):  f_new(x + c) == f(x) and
    g_new(x + c) == g(x) under translation;  f unchanged and g_new == s * g under
    row scaling;  and ``verify_point``'s verdict is unchanged by translation."""
    rng = np.random.default_rng(0)
    shifted, scaled, s = translate(model, 1e3, seed=3), rescale_rows(model, 1e3), 1e3
    ev, ev_sh, ev_sc = _evaluator(model), _evaluator(shifted), _evaluator(scaled)
    compared = 0
    for x in _box_samples(model, rng, 8):
        f, g = ev.evaluate_objective(x), np.asarray(ev.evaluate_constraints(x))
        if not np.isfinite(f) or not np.all(np.isfinite(g)):
            continue  # outside the evaluator's domain (e.g. log of a negative)
        xs = x + shifted._invariance_shift
        ftol = 1e-7 * (1.0 + abs(f))
        gtol = 1e-7 * (1.0 + np.abs(g))
        assert ev_sh.evaluate_objective(xs) == pytest.approx(f, abs=ftol), name
        assert ev_sc.evaluate_objective(x) == pytest.approx(f, abs=ftol), name
        g_sh = np.asarray(ev_sh.evaluate_constraints(xs))
        g_sc = np.asarray(ev_sc.evaluate_constraints(x))
        assert np.all(np.abs(g_sh - g) <= gtol), (name, np.max(np.abs(g_sh - g)))
        assert np.all(np.abs(g_sc - s * g) <= s * gtol), (name, np.max(np.abs(g_sc - s * g)))
        assert verify_point(shifted, xs).ok == verify_point(model, x).ok, name
        compared += 1
    assert compared > 0, f"{name}: no sampled point in the evaluator's domain -- measured nothing"


def test_harness_refuses_rather_than_guesses():
    m = dm.Model("gdp")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x <= 1], [x >= 9]])
    m.minimize(x)
    with pytest.raises(ValueError, match="cannot rebuild"):
        translate(m, 1e3)
    with pytest.raises(ValueError, match="positive"):
        rescale_rows(milp(0), -1.0)


# ── panels ────────────────────────────────────────────────────────────────────


def _run_panel(cases, transforms, time_limit):
    compared, lost, false = 0, [], []
    for label, model in cases:
        base = model.solve(time_limit=time_limit)
        if not base.gap_certified:
            continue  # nothing certified to compare against
        for tname, tf in transforms.items():
            other = tf(model).solve(time_limit=time_limit)
            compared += 1
            why = certified_answer_changed(base, other)
            if why:
                false.append((label, tname, why))
            elif not other.gap_certified:
                lost.append((label, tname, other.status))
    return compared, lost, false


#: Known false certificates, each tied to the issue that tracks it. ``strict``: a
#: fix turns the xfail into a failure, so the marker cannot outlive the bug.
KNOWN_FALSE = {
    ("polynomial", "shift1e6"): "#1542: a shifted cubic certifies infeasible / super-optimal",
}


def _panel_params():
    for fam in FAMILIES:
        for tname in TRANSFORMS:
            reason = KNOWN_FALSE.get((fam, tname))
            marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
            yield pytest.param(fam, tname, marks=marks, id=f"{fam}-{tname}")


@pytest.mark.correctness
@pytest.mark.parametrize("family, transform", list(_panel_params()))
def test_generated_panel(family, transform):
    cases = [(f"{family}[{s}]", FAMILIES[family](s)) for s in range(4)]
    compared, lost, false = _run_panel(cases, {transform: TRANSFORMS[transform]}, time_limit=10)
    print(f"\n{family} x {transform}: compared={compared} false={len(false)} lost={len(lost)}")
    for row in lost:
        print("  lost:", row)
    assert compared >= 3, f"{family} x {transform}: only {compared} certified bases to compare"
    assert not false, false


@pytest.mark.slow
@pytest.mark.correctness
@pytest.mark.parametrize("path", CORPUS, ids=os.path.basename)
def test_corpus_panel(path):
    cases = [(os.path.basename(path), dm.from_nl(path))]
    compared, lost, false = _run_panel(cases, TRANSFORMS, time_limit=20)
    if compared == 0:
        pytest.skip("base solve not certified within 20 s: no certificate to compare against")
    for row in lost:
        print("  lost:", row)
    assert not false, false
