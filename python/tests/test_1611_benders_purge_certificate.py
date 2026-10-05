"""Regression test for #1611 D-29: single-cut Benders reported ``optimal`` with
a bound below the objective after cut purging.

The loop certified ``optimal`` against the master bound of the iteration it
stopped on, then re-solved the master after ``_purge_stale_cuts`` had dropped
cuts and reported that weaker bound: purge=20 gave ``optimal`` 410.2989 with
bound 409.4021 (relative gap 2.2e-3 > gap_tolerance 1e-4).
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.decomposition import BendersConfig, solve_benders
from discopt.decomposition._linear import relative_gap

pytestmark = pytest.mark.filterwarnings("ignore")

rng = np.random.default_rng(22)
n_w, n_c, n_s = 8, 15, 4
XY_w = rng.uniform(0, 100, (n_w, 2))
XY_c = rng.uniform(0, 100, (n_c, 2))
dist = np.sqrt(((XY_w[:, None] - XY_c[None]) ** 2).sum(-1))
base = rng.uniform(10, 40, n_c)
ang = np.arctan2(XY_c[:, 1] - 50, XY_c[:, 0] - 50)
d = base * (1 + 0.8 * np.cos(ang[None, :] - 2 * np.pi * np.arange(n_s)[:, None] / n_s))
cap = rng.uniform(150, 300, n_w)
f = rng.uniform(40, 80, n_w)
c = 13 * 0.5 * dist / 1000
x_ub = 2 * float(d.max())


def build_monolithic(name="warehouses"):
    m = dm.Model(name)
    y = m.binary("y", shape=(n_w,))
    x = m.continuous("x", shape=(n_s, n_w, n_c), lb=0, ub=x_ub)
    m.minimize(
        dm.sum(lambda i: float(f[i]) * y[i], over=range(n_w))
        + dm.sum(
            lambda s: dm.sum(
                lambda i: dm.sum(lambda j: float(c[i, j]) * x[s, i, j], over=range(n_c)),
                over=range(n_w),
            ),
            over=range(n_s),
        )
    )
    for s in range(n_s):
        for j in range(n_c):
            m.subject_to(dm.sum(lambda i: x[s, i, j], over=range(n_w)) >= float(d[s, j]))
        for i in range(n_w):
            m.subject_to(dm.sum(lambda j: x[s, i, j], over=range(n_c)) <= float(cap[i]) * y[i])
            for j in range(n_c):
                m.subject_to(x[s, i, j] <= float(d[s, j]) * y[i])
    return m, y, x


@pytest.mark.parametrize("purge", [20, 10**6])
def test_single_cut_benders_certificate_matches_reported_bound(purge):
    m, y, _ = build_monolithic("s")
    m.first_stage(y)
    cfg = BendersConfig(multicut=False, cut_purge_after=purge)
    r = solve_benders(m, config=cfg)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(410.2988766, rel=1e-6)
    assert r.bound is not None
    # The certificate is the reported pair: it must close the gap.
    assert relative_gap(r.objective, r.bound) <= cfg.gap_tolerance


def test_limit_hit_master_incumbent_is_never_kept_as_bound(monkeypatch):
    """A master that hits its limit with an incumbent and no dual bound returns
    its *primal* value; ``best_lb`` must not absorb it (review of #1642). Before
    the fix this certified ``optimal`` with bound 411.14 > the optimum 410.30."""
    import discopt.solvers.lp_backend as lpb
    from discopt.solvers import MILPResult, SolveStatus

    real_get = lpb.get_decomposition_master_solver
    calls = {"master": 0, "limited": 0}

    def get_decomposition_master_solver():
        real, engine = real_get()

        def milp(c, **kw):
            res = real(c, **kw)
            if len(c) > n_w and res.x is not None:  # with-eta masters only
                calls["master"] += 1
                if calls["master"] >= 3:
                    calls["limited"] += 1
                    return MILPResult(
                        status=SolveStatus.TIME_LIMIT,
                        x=res.x,
                        objective=float(res.objective) + 5.0,  # incumbent above the LB
                        bound=None,
                    )
            return res

        return milp, engine

    monkeypatch.setattr(lpb, "get_decomposition_master_solver", get_decomposition_master_solver)
    m, y, _ = build_monolithic("s")
    m.first_stage(y)
    r = solve_benders(m, config=BendersConfig(multicut=False))
    assert calls["limited"] > 0
    assert r.bound is None or r.bound <= 410.2988766 + 1e-6
    assert r.status != "optimal"
