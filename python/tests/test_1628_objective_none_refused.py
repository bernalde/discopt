"""#1628 -- ``minimize(None)`` must be refused, not solved as a NaN objective.

``Constant(None)`` was ``np.asarray(None, dtype=float64)``, i.e. ``nan``, with no
error. So ``m.minimize(None)`` -- typically a builder helper that forgot its
``return`` -- installed a NaN objective, and the solve reported
``status="feasible", objective=nan, bound=nan`` with no hint that the objective
had vanished. The same hole let ``x + None`` build ``(x + nan)`` and
``minimize("1.5")`` silently parse the string as ``1.5``.

discopt's established "no objective" semantics are: a model with no objective is
refused by ``validate()`` ("No objective set"), and a pure feasibility problem is
spelled ``minimize(0)`` (what the GAMS reader emits for an objectiveless CNS
solve). ``None`` therefore gets a loud ``TypeError`` pointing at that spelling,
rather than being mapped onto a feasibility problem the user may not have meant.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from discopt import Model

pytestmark = pytest.mark.smoke


def _model():
    m = Model("m1628")
    x = m.continuous("x", lb=-2, ub=2)
    return m, x


@pytest.mark.parametrize("sense", ["minimize", "maximize"])
def test_none_objective_is_refused(sense):
    m, _ = _model()
    with pytest.raises(TypeError, match=r"got None.*forgot its `return`.*minimize\(0\)"):
        getattr(m, sense)(None)
    # Nothing was installed: the model still reports the missing objective.
    assert m._objective is None
    with pytest.raises(ValueError, match="No objective set"):
        m.validate()


def test_forgotten_return_helper_is_refused():
    """The adversary's spelling: a builder that evaluates to ``None``."""
    m, x = _model()

    def build(v):
        v**2  # noqa: B018 -- the bug under test: no ``return``

    with pytest.raises(TypeError, match="got None"):
        m.minimize(build(x))


@pytest.mark.parametrize("sense", ["minimize", "maximize"])
@pytest.mark.parametrize("text", ["x", "1.5", b"1.5"])
def test_string_objective_is_refused(sense, text):
    m, _ = _model()
    with pytest.raises(TypeError, match="needs a numeric expression"):
        getattr(m, sense)(text)
    assert m._objective is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -math.inf, np.nan])
def test_non_finite_constant_objective_is_refused(value):
    m, _ = _model()
    with pytest.raises(ValueError, match="non-finite constant objective"):
        m.minimize(value)
    assert m._objective is None


def test_list_of_expressions_names_the_objective():
    m, x = _model()
    with pytest.raises(TypeError, match=r"list of expressions.*dm\.sum"):
        m.minimize([x, 2 * x])


@pytest.mark.parametrize("operand", [None, "1.5"])
def test_none_or_text_operand_is_refused_everywhere(operand):
    """The root cause is ``Constant``; arithmetic and constraints share it."""
    _, x = _model()
    with pytest.raises(TypeError, match="as a numeric constant"):
        x + operand
    with pytest.raises(TypeError, match="as a numeric constant"):
        x <= operand  # noqa: B015


def test_feasibility_spelling_still_solves_to_zero():
    """The documented replacement for ``None`` works and is certified."""
    m, x = _model()
    m.subject_to(x >= 1)
    m.minimize(0)
    r = m.solve(time_limit=10)
    assert r.status == "optimal"
    assert r.objective == 0.0
    assert not math.isnan(r.objective)


@pytest.mark.parametrize("arg", [3, 3.0, np.float64(3.0), np.array([3.0])])
def test_finite_numeric_constants_still_accepted(arg):
    m, _ = _model()
    m.minimize(arg)
    r = m.solve(time_limit=10)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(3.0)
