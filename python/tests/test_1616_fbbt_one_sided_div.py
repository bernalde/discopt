"""#1616 D-09: backward FBBT through a product whose other factor touches zero.

``k * tau * (1 - X) == X`` with ``X in [0.9, 1]`` makes ``(1 - X) in [0, 0.1]``,
which touches 0, so backward FBBT skips the division and tightens nothing. Since
``X >= 0.9`` excludes 0 from the product, ``k * tau >= 9`` is a valid deduction,
and it would lift ``k``, ``T`` and ``tau`` through the Arrhenius row.

That rule was built behind ``DISCOPT_FBBT_ONE_SIDED_DIV`` and **retired** without
reaching main (``docs/dev/flag-retirement-audit.md``): sound on 213 panel checks,
but over the 29 MINLPLib instances where it fires it lost two certificates
(gancns, waternd1), loosened as many bounds as it tightened and cost +11 % wall.
These tests pin main's behaviour so the limitation stays visible: anyone
reintroducing the rule changes ``test_one_sided_product_is_not_divided`` and
owes the panel recorded in the audit row.
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import pytest
from discopt.tightening import fbbt_box


def _cstr():
    m = dm.Model("cstr")
    T = m.continuous("T", lb=300, ub=500)
    t = m.continuous("tau", lb=0, ub=60)
    k = m.continuous("k", lb=0, ub=10)
    X = m.continuous("X", lb=0.9, ub=1)
    m.subject_to(k == 1e7 * dm.exp(-8000 / T))
    m.subject_to(k * t * (1 - X) == X)
    m.minimize(2 * t + 2 * (T - 300))
    return m


@pytest.fixture(scope="module")
def box():
    return fbbt_box(_cstr())


def test_one_sided_product_is_not_divided(box):
    T_lb, tau_lb, k_lb = (float(v) for v in box.lb[:3])
    assert T_lb == pytest.approx(300.0)
    assert tau_lb == pytest.approx(0.0)
    assert k_lb < 1e-3


def test_known_feasible_point_survives(box):
    """A point satisfying both rows must stay inside the tightened box."""
    T = 480.0
    k = 1e7 * math.exp(-8000 / T)
    X = 0.95
    tau = X / (k * (1 - X))
    checks = 0
    for v, lo, hi in zip((T, tau, k, X), box.lb, box.ub, strict=True):
        assert float(lo) - 1e-9 <= v <= float(hi) + 1e-9
        checks += 1
    assert checks == 4
