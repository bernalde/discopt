"""#1586: root OBBT's #208 aux cascade must read every aux column through the varmap of
the relaxation build that produced it.

``obbt_tighten_root(..., cascade_aux=True)`` rebuilds the McCormick LP after DBBT
and again every sweep. The rebuild at a tightened box can lay the lifted columns
out differently (a term whose arguments become fixed is no longer lifted), but
the DBBT rebuild discarded its new varmap and the carried aux bounds were keyed
by raw column index. Reverse FBBT then read aux bounds belonging to other terms
and cut the optimum out of the box.

Witness (nvs22 with ``w == i4 + 14`` an explicit variable, which is exactly what
the #1537 affine-monomial lift introduces): the root-fixpoint OBBT call with
cutoff 7.59956 shrank a box containing the optimum (i1, i2, i3, i4) = (5, 1, 1, 2),
objective 6.05822, to the point (3, 1, 1, 1) with ``w = 18`` -- itself
inconsistent with ``w == i4 + 14`` -- and the solve certified 7.59956 as optimal
in 3 nodes.
"""

from __future__ import annotations

import discopt._relax.mccormick_lp as mc
import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.obbt import obbt_tighten_root

OPT = 6.05822  # minlplib.solu, nvs22
X_OPT = np.array([5.0, 1.0, 1.0, 2.0, 2121.640736, 10782.670341, 3.162278, 0.316228, 16.0])
BOX_LB = [3.0, 1.0, 1.0, 1.0, 530.4101838755299, 47.08695718814723, 2.06155281280883]
BOX_LB += [0.09514209428026271, 15.0]
BOX_UB = [9.0, 3.0, 2.0, 4.0, 4243.28147100424, 13587.753083967642, 5.255297392625561]
BOX_UB += [0.9701425001453321, 18.0]
CUTOFF = 7.5995600000000705


def _nvs22_with_w(w_integer: bool) -> dm.Model:
    """nvs22 (minus its redundant row e9) with ``14 + i4`` named as ``w``."""
    m = dm.Model("nvs22w")
    i1 = m.integer("i1", lb=1, ub=200)
    i2 = m.integer("i2", lb=1, ub=200)
    i3 = m.integer("i3", lb=1, ub=20)
    i4 = m.integer("i4", lb=1, ub=20)
    x5 = m.continuous("x5", lb=-1e7, ub=1e7)
    x6 = m.continuous("x6", lb=-1e7, ub=1e7)
    x7 = m.continuous("x7", lb=-1e7, ub=1e7)
    x8 = m.continuous("x8", lb=-1e7, ub=1e7)
    w = (m.integer if w_integer else m.continuous)("w", lb=15, ub=34)
    m.subject_to(w == i4 + 14)
    m.subject_to(-4243.28147100424 / (i3 * i4) + x5 == 0)
    m.subject_to(-dm.sqrt(0.25 * i4**2 + (0.5 * i1 + 0.5 * i3) ** 2) + x7 == 0)
    m.subject_to(
        -(59405.9405940594 + 2121.64073550212 * i4)
        * x7
        / (i3 * i4 * (0.0833333333333333 * i4**2 + (0.5 * i1 + 0.5 * i3) ** 2))
        + x6
        == 0
    )
    m.subject_to(-0.5 * i4 / x7 + x8 == 0)
    m.subject_to(-dm.sqrt(x5**2 + 2 * x5 * x6 * x8 + x6**2) >= -13600)
    m.subject_to(-504000 / (i1**2 * i2) >= -30000)
    m.subject_to(i2 - i3 >= 0)
    m.subject_to(
        0.0204744897959184 * dm.sqrt(1e13 * i2**3 * i1 * i1 * i2**3) * (1 - 0.0282346219657891 * i1)
        >= 6000
    )
    m.minimize(1.10471 * i3**2 * i4 + 0.04811 * i1 * i2 * w)
    return m


def _outside(lb, ub) -> list[int]:
    tol = 1e-6 * np.maximum(1.0, np.abs(X_OPT))
    return [k for k in range(len(X_OPT)) if not (lb[k] - tol[k] <= X_OPT[k] <= ub[k] + tol[k])]


@pytest.mark.parametrize("w_integer", [True, False])
def test_cascade_keeps_the_optimum_when_the_rebuild_changes_layout(monkeypatch, w_integer):
    """Fails before the fix: the cascade cut 8 of the 9 columns of the optimum."""
    assert not _outside(np.array(BOX_LB), np.array(BOX_UB)), "the input box must contain x*"
    n_cols: list[int] = []
    real_build = mc.build_milp_relaxation

    def counting_build(*a, **k):
        milp, varmap = real_build(*a, **k)
        n_cols.append(len(milp._bounds))
        return milp, varmap

    monkeypatch.setattr(mc, "build_milp_relaxation", counting_build)
    r = obbt_tighten_root(
        _nvs22_with_w(w_integer),
        np.array(BOX_LB),
        ub=np.array(BOX_UB),
        rounds=3,
        incumbent_cutoff=CUTOFF,
        cascade_aux=True,
    )
    # The witness only means something if a rebuild really changed the layout.
    assert len(set(n_cols)) >= 2, n_cols
    assert not r.infeasible
    assert _outside(r.lb, r.ub) == [], (r.lb.tolist(), r.ub.tolist())


def test_solve_certifies_the_true_optimum_with_an_integer_affine_variable(monkeypatch):
    """End to end. Fails before the fix: certified 7.59956 in 3 nodes.

    Recentring pinned OFF (#1537, default ON). The witness is the root cascade on
    the model as written. Recentring moves x5..x8 out of their [-1e7, 1e7] boxes
    under the one-sided rule, and the solve becomes a different search: measured
    on PR #1594 it still certifies 6.05822, but it takes 351 nodes and 33 s
    against 11 nodes and 3.4 s. Under xdist load that pushed it past the 60 s
    limit with no incumbent. This is recorded on #1537 as a performance cost of the
    one-sided rule on zero-containing boxes, not a correctness loss."""
    monkeypatch.setenv("DISCOPT_RECENTRE", "0")
    r = _nvs22_with_w(True).solve(time_limit=60)
    assert r.objective is not None
    assert r.objective == pytest.approx(OPT, abs=1e-4)
    if r.gap_certified:
        assert r.bound <= OPT + 1e-6
