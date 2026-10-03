"""Reduced POUNCE Gate-1 flash feasibility and regime check (#1526).

This module rebuilds the ethane/n-butane phase-changing flash from
``pounce.examples.flash_mpcc`` on DiscOpt's expression tree.  POUNCE remains
the owner of the fixture and of the independent Michelsen/Rachford--Rice
oracle.  DiscOpt owns only the algebraic cubic-root encoding, the exact/local
comparison, and the version-2 comparison records.

The zero objective is intentional.  A zero gap proves only that a feasible
point was globally certified; the physical result is decided by the complete
point, explicit Boolean identities, and source residuals.

The exact GDP/SOS1 arms are warm-started from the POUNCE point (after
``accept_local_incumbent`` rechecks it).  With a zero objective the incumbent
meets the bound at the root, so a one-node certificate is a feasibility and
regime check *of POUNCE's point*, not an independent exact solve.
"""

from __future__ import annotations

import copy
import importlib.metadata
import json
import math
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence

import numpy as np

import discopt.modeling as dm
from discopt.modeling.core import Model, Variable
from discopt.mpec import Complementarity, complementarity, register_relations, solve_mpec
from discopt.mpec_report import accept_local_incumbent
from discopt.warm_start import validate_initial_solution

Method = Literal["gdp", "sos1", "scholtes"]
RootEncoding = Literal["disjunctive", "fixed"]

PRESSURE_PA = 1.0e6
GAS_CONSTANT = 8.314462618
SQRT_TWO = math.sqrt(2.0)
MOLE_FLOOR = 1.0e-12
TRIVIAL_LOG_K = 1.0e-4
DISCRIMINANT_GUARD = 1.0e-8
COMPARISON_TOLERANCE = 1.0e-6
CRITICAL_TEMPERATURE_K = np.array([305.3, 425.1])
CRITICAL_PRESSURE_PA = np.array([48.72, 37.96]) * 1.0e5
ACENTRIC_FACTOR = np.array([0.100, 0.200])
FEED_COMPOSITION = np.array([0.5, 0.5])

SMOKE_TEMPERATURES = (250.0, 268.0, 300.0, 324.0, 350.0)
FULL_TEMPERATURES = tuple(
    float(v)
    for v in np.unique(
        np.concatenate(
            [
                np.arange(230.0, 361.0, 10.0),
                np.arange(262.0, 273.0, 1.0),
                np.arange(322.0, 333.0, 1.0),
            ]
        )
    )
)

_NONTRIVIAL_NAMES = (
    "component_0_above",
    "component_0_below",
    "component_1_above",
    "component_1_below",
)


@dataclass(frozen=True)
class FlashPoint:
    """A physical flash point in the POUNCE variable convention."""

    temperature_k: float
    regime: str
    beta: float
    x: np.ndarray
    y: np.ndarray

    @property
    def sum_x(self) -> float:
        return float(np.sum(self.x))

    @property
    def sum_y(self) -> float:
        return float(np.sum(self.y))

    @property
    def packed(self) -> np.ndarray:
        return np.concatenate(([self.beta], self.x, self.y))


@dataclass
class FlashProblem:
    """One built DiscOpt flash plus the source objects needed to audit it."""

    temperature_k: float
    model: Model
    pairs: list[Complementarity]
    oracle: FlashPoint
    initial_solution: dict[Variable, Any]
    variables: dict[str, Variable]
    source_variables: tuple[Variable, ...]
    root_encoding: RootEncoding
    root_branches: tuple[str, str]
    root_indicators: tuple[tuple[Optional[Variable], Optional[Variable]], ...]
    nontrivial_branch: str
    nontrivial_indicators: tuple[Variable, ...]


@dataclass
class FlashSolve:
    """Internal solve result; ``record`` is the shared schema payload."""

    problem: FlashProblem
    result: Any
    record: dict[str, Any]
    warm_start_accepted: bool


def oracle_point(temperature_k: float) -> FlashPoint:
    """Evaluate POUNCE's independent flash oracle at one temperature."""
    from pounce.examples.flash_mpcc import GATE1_FLASH, flash

    t = float(temperature_k)
    ref = flash(t, GATE1_FLASH.pressure_pa, GATE1_FLASH.mixture)
    if not ref.converged or ref.trivial:
        raise RuntimeError(
            f"POUNCE oracle did not produce a nontrivial converged point at {t:g} K: "
            f"converged={ref.converged}, trivial={ref.trivial}, note={ref.note!r}"
        )
    return FlashPoint(t, ref.regime, float(ref.beta), np.asarray(ref.x), np.asarray(ref.y))


def pounce_seed(artifact: Mapping[str, Any], temperature_k: float) -> Optional[FlashPoint]:
    """Take one successful POUNCE leg point from a real version-1 artifact."""
    target = float(temperature_k)
    for leg in artifact.get("legs", []):
        for row in leg.get("records", []):
            if abs(float(row.get("temperature_k", math.inf)) - target) > 1.0e-10:
                continue
            packed = row.get("x")
            if not row.get("ok") or packed is None or len(packed) != 5:
                continue
            value = np.asarray(packed, dtype=float)
            return FlashPoint(
                target,
                str(row.get("regime") or "undetermined"),
                float(value[0]),
                value[1:3].copy(),
                value[3:5].copy(),
            )
    return None


def _sum(expressions: Sequence[Any]) -> Any:
    out = expressions[0]
    for expression in expressions[1:]:
        out = out + expression
    return out


def _component_constants(temperature_k: float) -> tuple[np.ndarray, np.ndarray]:
    kappa = 0.37464 + 1.54226 * ACENTRIC_FACTOR - 0.26992 * ACENTRIC_FACTOR**2
    alpha = (1.0 + kappa * (1.0 - np.sqrt(temperature_k / CRITICAL_TEMPERATURE_K))) ** 2
    ai = 0.45724 * GAS_CONSTANT**2 * CRITICAL_TEMPERATURE_K**2 * alpha / CRITICAL_PRESSURE_PA
    bi = 0.07780 * GAS_CONSTANT * CRITICAL_TEMPERATURE_K / CRITICAL_PRESSURE_PA
    return bi, np.sqrt(np.outer(ai, ai))


def _numeric_eos(
    composition: np.ndarray, temperature_k: float
) -> tuple[tuple[float, float, float], float, np.ndarray]:
    w = np.asarray(composition, dtype=float)
    bi, aij = _component_constants(temperature_k)
    amix = float(w @ aij @ w)
    bmix = float(w @ bi)
    aa = amix * PRESSURE_PA / (GAS_CONSTANT * temperature_k) ** 2
    bb = bmix * PRESSURE_PA / (GAS_CONSTANT * temperature_k)
    c2 = -(1.0 - bb)
    c1 = aa - 2.0 * bb - 3.0 * bb**2
    c0 = -(aa * bb - bb**2 - bb**3)
    depressed_p = c1 - c2**2 / 3.0
    depressed_q = 2.0 * c2**3 / 27.0 - c2 * c1 / 3.0 + c0
    discriminant = (depressed_q / 2.0) ** 2 + (depressed_p / 3.0) ** 3
    roots = np.roots([1.0, c2, c1, c0])
    real = np.sort(roots[np.abs(roots.imag) < 1.0e-9].real)
    return (float(c2), float(c1), float(c0)), float(discriminant), real


def _branch_for(composition: np.ndarray, temperature_k: float) -> tuple[str, float, np.ndarray]:
    _, discriminant, roots = _numeric_eos(composition, temperature_k)
    if abs(discriminant) <= DISCRIMINANT_GUARD:
        raise ValueError(
            f"Peng--Robinson discriminant {discriminant:.3e} at {temperature_k:g} K "
            "lies inside the explicit root-count boundary guard. The closed "
            "SELECT_ONE branches overlap at a repeated root; this benchmark makes "
            "no strict one-root/three-root identity claim there."
        )
    branch = "one_real_root" if discriminant > 0.0 else "three_real_roots"
    return branch, discriminant, roots


def _seed_nontrivial_branch(x: np.ndarray, y: np.ndarray) -> tuple[int, str]:
    wl, wv = x / np.sum(x), y / np.sum(y)
    log_k = np.log(wv) - np.log(wl)
    candidates = np.array([log_k[0], -log_k[0], log_k[1], -log_k[1]])
    index = int(np.argmax(candidates))
    if candidates[index] < TRIVIAL_LOG_K:
        raise ValueError("seed is the trivial stationary point and cannot identify the flash")
    return index, _NONTRIVIAL_NAMES[index]


def build_pounce_flash(
    temperature_k: float,
    *,
    oracle: Optional[FlashPoint] = None,
    seed: Optional[FlashPoint] = None,
    root_encoding: RootEncoding = "disjunctive",
) -> FlashProblem:
    """Build the Gate-1 source model with exact or fixed root identities.

    ``disjunctive`` is used by GDP and SOS1.  ``fixed`` retains the oracle seed's
    root-count and nontriviality branches as ordinary smooth rows for the local
    Scholtes comparison; this avoids asking an NLP solve to decide Boolean root
    identities while preserving the same active algebraic equations.
    """
    if root_encoding not in ("disjunctive", "fixed"):
        raise ValueError("root_encoding must be 'disjunctive' or 'fixed'")
    t = float(temperature_k)
    ref = oracle or oracle_point(t)
    start = seed or ref
    if abs(ref.temperature_k - t) > 1.0e-10 or abs(start.temperature_k - t) > 1.0e-10:
        raise ValueError("oracle and seed temperatures must match the model temperature")

    bi, aij = _component_constants(t)
    model = dm.Model(f"pounce_gate1_flash_{t:g}K_{root_encoding}")
    beta = model.continuous("beta", lb=0.0, ub=1.0)
    x = model.continuous("x", shape=2, lb=MOLE_FLOOR, ub=1.0)
    y = model.continuous("y", shape=2, lb=MOLE_FLOOR, ub=1.0)
    sx = model.continuous("sum_x", lb=2.0 * MOLE_FLOOR, ub=1.0)
    sy = model.continuous("sum_y", lb=2.0 * MOLE_FLOOR, ub=1.0)
    wl = model.continuous("w_liquid", shape=2, lb=MOLE_FLOOR, ub=1.0)
    wv = model.continuous("w_vapor", shape=2, lb=MOLE_FLOOR, ub=1.0)
    selected = model.continuous("compressibility", shape=2, lb=1.0e-4, ub=1.5)
    single = model.continuous("one_root_z", shape=2, lb=1.0e-4, ub=1.5)
    three = model.continuous("three_root_z", shape=(2, 3), lb=1.0e-4, ub=1.5)
    model.minimize(0.0 * beta)

    for i in range(2):
        model.subject_to(
            (1.0 - beta) * x[i] + beta * y[i] == FEED_COMPOSITION[i],
            name=f"balance_{i}",
        )
        model.subject_to(x[i] == sx * wl[i], name=f"normalize_liquid_{i}")
        model.subject_to(y[i] == sy * wv[i], name=f"normalize_vapor_{i}")
    model.subject_to(wl[0] + wl[1] == 1.0, name="simplex_liquid")
    model.subject_to(wv[0] + wv[1] == 1.0, name="simplex_vapor")

    lnphi: list[list[Any]] = []
    root_branches: list[str] = []
    root_indicators: list[tuple[Optional[Variable], Optional[Variable]]] = []
    root_initial: list[tuple[float, np.ndarray, str]] = []
    start_compositions = (start.x / start.sum_x, start.y / start.sum_y)
    for phase, w in enumerate((wl, wv)):
        amix = _sum([w[i] * aij[i, j] * w[j] for i in range(2) for j in range(2)])
        bmix = _sum([w[i] * bi[i] for i in range(2)])
        aa = amix * PRESSURE_PA / (GAS_CONSTANT * t) ** 2
        bb = bmix * PRESSURE_PA / (GAS_CONSTANT * t)
        c2 = -(1.0 - bb)
        c1 = aa - 2.0 * bb - 3.0 * bb**2
        c0 = -(aa * bb - bb**2 - bb**3)
        depressed_p = c1 - c2**2 / 3.0
        depressed_q = 2.0 * c2**3 / 27.0 - c2 * c1 / 3.0 + c0
        discriminant = (depressed_q / 2.0) ** 2 + (depressed_p / 3.0) ** 3
        cubic = single[phase] ** 3 + c2 * single[phase] ** 2 + c1 * single[phase] + c0
        r0, r1, r2 = (three[phase, k] for k in range(3))

        branch, d0, roots0 = _branch_for(start_compositions[phase], t)
        root_branches.append(branch)
        root_initial.append((d0, roots0, branch))

        def add_one(container: Any) -> None:
            container.subject_to(discriminant >= 0.0, name="cardano_positive")
            container.subject_to(cubic == 0.0, name="cubic")
            container.subject_to(selected[phase] == single[phase], name="select")

        def add_three(container: Any) -> None:
            container.subject_to(discriminant <= 0.0, name="cardano_nonpositive")
            container.subject_to(r0 + r1 + r2 == -c2, name="vieta_sum")
            container.subject_to(r0 * r1 + r0 * r2 + r1 * r2 == c1, name="vieta_pairs")
            container.subject_to(r0 * r1 * r2 == -c0, name="vieta_product")
            container.subject_to(r0 <= r1, name="order_0")
            container.subject_to(r1 <= r2, name="order_1")
            container.subject_to(selected[phase] == (r0 if phase == 0 else r2), name="select")

        if root_encoding == "disjunctive":
            one = model.make_disjunct(f"phase_{phase}_one_real_root")
            tri = model.make_disjunct(f"phase_{phase}_three_real_roots")
            add_one(one)
            add_three(tri)
            model.add_disjunction(
                [one, tri],
                name=f"phase_{phase}_root_count",
                semantics=dm.DisjunctionSemantics.SELECT_ONE,
            )
            root_indicators.append((one.indicator.variable, tri.indicator.variable))
        else:
            add_one(model) if branch == "one_real_root" else add_three(model)
            root_indicators.append((None, None))

        phase_lnphi = []
        for i in range(2):
            cross = _sum([aij[i, j] * w[j] for j in range(2)])
            phase_lnphi.append(
                (bi[i] / bmix) * (selected[phase] - 1.0)
                - dm.log(selected[phase] - bb)
                - aa
                / (2.0 * SQRT_TWO * bb)
                * (2.0 * cross / amix - bi[i] / bmix)
                * dm.log(
                    (selected[phase] + (1.0 + SQRT_TWO) * bb)
                    / (selected[phase] + (1.0 - SQRT_TWO) * bb)
                )
            )
        lnphi.append(phase_lnphi)

    for i in range(2):
        model.subject_to(
            dm.log(x[i]) + lnphi[0][i] == dm.log(y[i]) + lnphi[1][i],
            name=f"isofugacity_{i}",
        )

    nontrivial_index, nontrivial_branch = _seed_nontrivial_branch(start.x, start.y)
    nontrivial_indicators: list[Variable] = []
    nontrivial_rows = (
        dm.log(wv[0]) - dm.log(wl[0]) >= TRIVIAL_LOG_K,
        dm.log(wl[0]) - dm.log(wv[0]) >= TRIVIAL_LOG_K,
        dm.log(wv[1]) - dm.log(wl[1]) >= TRIVIAL_LOG_K,
        dm.log(wl[1]) - dm.log(wv[1]) >= TRIVIAL_LOG_K,
    )
    if root_encoding == "disjunctive":
        disjuncts = []
        for name, row in zip(_NONTRIVIAL_NAMES, nontrivial_rows):
            disjunct = model.make_disjunct(f"nontrivial_{name}")
            disjunct.subject_to(row)
            disjuncts.append(disjunct)
            nontrivial_indicators.append(disjunct.indicator.variable)
        model.add_disjunction(
            disjuncts,
            name="nontrivial_stationary_point",
            semantics=dm.DisjunctionSemantics.SELECT_ONE,
        )
    else:
        model.subject_to(nontrivial_rows[nontrivial_index], name="nontrivial_fixed_branch")

    pairs = [
        complementarity(beta, 1.0 - sy, name="vapor"),
        complementarity(1.0 - beta, 1.0 - sx, name="liquid"),
    ]

    selected0: list[float] = []
    single0: list[float] = []
    three0: list[np.ndarray] = []
    initial: dict[Variable, Any] = {
        beta: start.beta,
        x: start.x,
        y: start.y,
        sx: start.sum_x,
        sy: start.sum_y,
        wl: start.x / start.sum_x,
        wv: start.y / start.sum_y,
    }
    for phase, (_, roots, branch) in enumerate(root_initial):
        if branch == "three_real_roots":
            selected0.append(roots[0] if phase == 0 else roots[-1])
            single0.append(roots[-1])
            three0.append(roots)
        else:
            selected0.append(roots[0])
            single0.append(roots[0])
            three0.append(
                np.asarray(
                    [
                        max(1.0e-4, roots[0] * 0.05),
                        max(1.0e-4, roots[0] * 0.1),
                        roots[0],
                    ]
                )
            )
        one_indicator, three_indicator = root_indicators[phase]
        if one_indicator is not None and three_indicator is not None:
            initial[one_indicator] = float(branch == "one_real_root")
            initial[three_indicator] = float(branch == "three_real_roots")
    initial[selected] = np.asarray(selected0)
    initial[single] = np.asarray(single0)
    initial[three] = np.asarray(three0)
    for index, indicator in enumerate(nontrivial_indicators):
        initial[indicator] = float(index == nontrivial_index)

    variables = {
        "beta": beta,
        "x": x,
        "y": y,
        "sum_x": sx,
        "sum_y": sy,
        "w_liquid": wl,
        "w_vapor": wv,
        "compressibility": selected,
        "one_root_z": single,
        "three_root_z": three,
    }
    return FlashProblem(
        t,
        model,
        pairs,
        ref,
        initial,
        variables,
        tuple(model._variables),
        root_encoding,
        (root_branches[0], root_branches[1]),
        tuple(root_indicators),
        nontrivial_branch,
        tuple(nontrivial_indicators),
    )


def _solution_by_name(problem: FlashProblem, solution: Mapping[Any, Any]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in solution.items():
        name = key.name if isinstance(key, Variable) else str(key)
        out[name] = np.asarray(value, dtype=float)
    return out


def _numeric_lnphi(composition: np.ndarray, temperature_k: float, selected_z: float) -> np.ndarray:
    bi, aij = _component_constants(temperature_k)
    w = np.asarray(composition, dtype=float)
    amix = float(w @ aij @ w)
    bmix = float(w @ bi)
    aa = amix * PRESSURE_PA / (GAS_CONSTANT * temperature_k) ** 2
    bb = bmix * PRESSURE_PA / (GAS_CONSTANT * temperature_k)
    cross = aij @ w
    return np.asarray(
        (
            (bi / bmix) * (selected_z - 1.0)
            - np.log(selected_z - bb)
            - aa
            / (2.0 * SQRT_TWO * bb)
            * (2.0 * cross / amix - bi / bmix)
            * np.log((selected_z + (1.0 + SQRT_TWO) * bb) / (selected_z + (1.0 - SQRT_TWO) * bb))
        )
    )


def _active_root_branches(
    problem: FlashProblem, values: Mapping[str, np.ndarray]
) -> tuple[str, str]:
    if problem.root_encoding == "fixed":
        return problem.root_branches
    branches = []
    for one, three in problem.root_indicators:
        assert one is not None and three is not None
        one_value = float(np.asarray(values[one.name]))
        three_value = float(np.asarray(values[three.name]))
        branches.append("one_real_root" if one_value >= three_value else "three_real_roots")
    return branches[0], branches[1]


def _residual(
    value: float, definition: str, admitted_scale: Optional[float] = None
) -> dict[str, Any]:
    return {
        "value": float(max(0.0, value)),
        "definition": definition,
        "admitted_scale": None if admitted_scale is None else float(admitted_scale),
    }


def source_residuals(
    problem: FlashProblem,
    solution: Mapping[Any, Any],
    *,
    mpec_report: Any = None,
) -> dict[str, Any]:
    """Measure each source-row family separately at ``solution``."""
    values = _solution_by_name(problem, solution)
    beta = float(values["beta"])
    x, y = values["x"], values["y"]
    sx, sy = float(values["sum_x"]), float(values["sum_y"])
    wl, wv = values["w_liquid"], values["w_vapor"]
    selected = values["compressibility"]
    single, three = values["one_root_z"], values["three_root_z"]
    branches = _active_root_branches(problem, values)

    balance = max(
        float(np.max(np.abs((1.0 - beta) * x + beta * y - FEED_COMPOSITION))),
        float(np.max(np.abs(x - sx * wl))),
        float(np.max(np.abs(y - sy * wv))),
        abs(float(np.sum(wl)) - 1.0),
        abs(float(np.sum(wv)) - 1.0),
    )
    lnphi_l = _numeric_lnphi(wl, problem.temperature_k, float(selected[0]))
    lnphi_v = _numeric_lnphi(wv, problem.temperature_k, float(selected[1]))
    isofugacity = float(np.max(np.abs(np.log(x) + lnphi_l - np.log(y) - lnphi_v)))

    eos = 0.0
    root_selection = 0.0
    for phase, (w, branch) in enumerate(zip((wl, wv), branches)):
        coeff, discriminant, numeric_roots = _numeric_eos(w, problem.temperature_k)
        c2, c1, c0 = coeff

        def cubic(z: float) -> float:
            return z**3 + c2 * z**2 + c1 * z + c0

        desired = numeric_roots[0] if phase == 0 else numeric_roots[-1]
        if branch == "one_real_root":
            eos = max(eos, abs(cubic(float(single[phase]))))
            root_selection = max(
                root_selection,
                max(0.0, -discriminant),
                abs(float(selected[phase] - single[phase])),
            )
        else:
            r = np.asarray(three[phase], dtype=float)
            eos = max(
                eos,
                abs(float(r[0] + r[1] + r[2] + c2)),
                abs(float(r[0] * r[1] + r[0] * r[2] + r[1] * r[2] - c1)),
                abs(float(r[0] * r[1] * r[2] + c0)),
            )
            root_selection = max(
                root_selection,
                max(0.0, discriminant),
                max(0.0, float(r[0] - r[1])),
                max(0.0, float(r[1] - r[2])),
                abs(float(selected[phase] - (r[0] if phase == 0 else r[2]))),
            )
        root_selection = max(root_selection, abs(float(selected[phase]) - float(desired)))
        if problem.root_encoding == "disjunctive":
            one, tri = problem.root_indicators[phase]
            assert one is not None and tri is not None
            selectors = np.asarray([float(values[one.name]), float(values[tri.name])])
            root_selection = max(
                root_selection,
                abs(float(np.sum(selectors)) - 1.0),
                float(np.max(np.minimum(np.abs(selectors), np.abs(1.0 - selectors)))),
            )

    log_k = np.log(wv) - np.log(wl)
    candidates = np.asarray([log_k[0], -log_k[0], log_k[1], -log_k[1]], dtype=float)
    if problem.root_encoding == "disjunctive":
        selectors = np.asarray(
            [float(values[indicator.name]) for indicator in problem.nontrivial_indicators]
        )
        active_nontrivial = int(np.argmax(selectors))
        nontriviality = max(
            0.0,
            TRIVIAL_LOG_K - float(candidates[active_nontrivial]),
            abs(float(np.sum(selectors)) - 1.0),
            float(np.max(np.minimum(np.abs(selectors), np.abs(1.0 - selectors)))),
        )
    else:
        active_nontrivial = _NONTRIVIAL_NAMES.index(problem.nontrivial_branch)
        nontriviality = max(0.0, TRIVIAL_LOG_K - float(candidates[active_nontrivial]))
    bounds = 0.0
    for variable in problem.source_variables:
        if variable.name not in values:
            continue
        val = np.asarray(values[variable.name], dtype=float)
        bounds = max(
            bounds,
            float(np.max(np.maximum(np.maximum(variable.lb - val, val - variable.ub), 0.0))),
        )
    g = np.array([beta, 1.0 - beta])
    h = np.array([1.0 - sy, 1.0 - sx])
    sign = float(np.max(np.maximum(np.maximum(-g, -h), 0.0)))
    complementarity_value = float(np.max(np.abs(g * h)))
    admitted = None
    if mpec_report is not None:
        admitted = getattr(mpec_report.complementarity, "admitted_scale", None)

    return {
        "balance": _residual(
            balance,
            "max material-balance, x=sum_x*w_liquid, y=sum_y*w_vapor, and phase-simplex "
            "equality residual",
        ),
        "isofugacity": _residual(
            isofugacity,
            "max_i |ln(x_i)+ln(phi_i^L(w_L))-ln(y_i)-ln(phi_i^V(w_V))|",
        ),
        "eos": _residual(
            eos,
            "max active Peng--Robinson source-row residual: cubic in the one-root branch; "
            "Vieta sum, pair, and product identities in the three-root branch",
        ),
        "root_selection": _residual(
            root_selection,
            "max root-count sign, ordering, selected-root identity, numeric extreme-root error, "
            "and root-branch selector identity/integrality residual",
        ),
        "nontriviality": _residual(
            nontriviality,
            "max violation of the selected signed log(K_i) >= 1e-4 row and, for the "
            "disjunctive model, its selector identity/integrality residual",
        ),
        "bound": _residual(bounds, "max source-variable bound violation"),
        "sign": _residual(sign, "max complementarity-operand sign violation"),
        "complementarity": _residual(
            complementarity_value,
            "max(|beta*(1-sum_y)|, |(1-beta)*(1-sum_x)|)",
            admitted,
        ),
    }


def validate_oracle_equivalence(problem: FlashProblem, tolerance: float = 1.0e-7) -> dict[str, Any]:
    """Execute and assert every source-family check at the POUNCE point."""
    measured = source_residuals(problem, problem.initial_solution)
    required = {
        "balance",
        "isofugacity",
        "eos",
        "root_selection",
        "nontriviality",
        "bound",
        "sign",
        "complementarity",
    }
    if set(measured) != required:
        raise AssertionError(f"source-family execution drift: got {sorted(measured)}")
    failures = {name: row["value"] for name, row in measured.items() if row["value"] > tolerance}
    if failures:
        raise AssertionError(
            f"POUNCE point does not satisfy the DiscOpt algebraic source model at "
            f"{problem.temperature_k:g} K within {tolerance:g}: {failures}"
        )
    return measured


def accepted_exact_warm_start(problem: FlashProblem) -> Optional[dict[Variable, Any]]:
    """Gate a POUNCE point before it can seed an exact solve.

    The return is an incumbent *hint* only.  ``accept_local_incumbent`` returns
    an objective and has no surface on which to manufacture a dual bound.
    """
    register_relations(problem.model, problem.pairs)
    x_flat = validate_initial_solution(problem.model, problem.initial_solution)
    objective = accept_local_incumbent(problem.model, object(), x_flat=x_flat)
    return problem.initial_solution if objective is not None else None


def _active_branches(beta: float, sx: float, sy: float, tolerance: float) -> dict[str, list[str]]:
    pairs = ((beta, 1.0 - sy), (1.0 - beta, 1.0 - sx))
    out: dict[str, list[str]] = {}
    for name, (amount, slack) in zip(("vapor", "liquid"), pairs):
        active = []
        if abs(amount) <= tolerance:
            active.append("amount_zero")
        if abs(slack) <= tolerance:
            active.append("slack_zero")
        if not active:
            raise ValueError(
                f"{name} pair has no active branch within {tolerance:g}: "
                f"amount={amount:.3e}, slack={slack:.3e}"
            )
        out[name] = active
    return out


def _regime(active: Mapping[str, Sequence[str]]) -> str:
    vapor, liquid = set(active["vapor"]), set(active["liquid"])
    if {"amount_zero", "slack_zero"} <= vapor:
        return "bubble"
    if {"amount_zero", "slack_zero"} <= liquid:
        return "dew"
    if "amount_zero" in vapor:
        return "liquid"
    if "amount_zero" in liquid:
        return "vapor"
    if "slack_zero" in vapor and "slack_zero" in liquid:
        return "two_phase"
    return "undetermined"


def _nontrivial_branch(problem: FlashProblem, values: Mapping[str, np.ndarray]) -> str:
    if problem.root_encoding == "disjunctive":
        scores = [float(values[indicator.name]) for indicator in problem.nontrivial_indicators]
        return _NONTRIVIAL_NAMES[int(np.argmax(scores))]
    return problem.nontrivial_branch


def _point_record(
    problem: FlashProblem, solution: Mapping[Any, Any], tolerance: float
) -> dict[str, Any]:
    values = _solution_by_name(problem, solution)
    beta = float(values["beta"])
    sx, sy = float(values["sum_x"]), float(values["sum_y"])
    active = _active_branches(beta, sx, sy, tolerance)
    roots = _active_root_branches(problem, values)
    return {
        "beta": beta,
        "x": values["x"].tolist(),
        "y": values["y"].tolist(),
        "sum_x": sx,
        "sum_y": sy,
        "compressibility": {
            "liquid": float(values["compressibility"][0]),
            "vapor": float(values["compressibility"][1]),
        },
        "root_branches": {"liquid": roots[0], "vapor": roots[1]},
        "nontrivial_branch": _nontrivial_branch(problem, values),
        "regime": _regime(active),
        "active_branches": active,
    }


def _oracle_comparison(problem: FlashProblem, point: Mapping[str, Any]) -> dict[str, Any]:
    ref = problem.oracle
    liquid = ref.x / ref.sum_x
    vapor = ref.y / ref.sum_y
    liquid_branch, _, liquid_roots = _branch_for(liquid, ref.temperature_k)
    vapor_branch, _, vapor_roots = _branch_for(vapor, ref.temperature_k)
    ref_regime = ref.regime
    regime_match = point["regime"] == ref_regime
    if ref_regime == "liquid" and point["regime"] == "bubble":
        regime_match = True
    if ref_regime == "vapor" and point["regime"] == "dew":
        regime_match = True
    biactive = None
    if point["regime"] in ("bubble", "dew") or ref_regime in ("bubble", "dew"):
        pair = "vapor" if point["regime"] == "bubble" else "liquid"
        biactive = set(point["active_branches"][pair]) == {"amount_zero", "slack_zero"}
    return {
        "regime_match": regime_match,
        "biactive_branch_admissible": biactive,
        "beta_error": abs(float(point["beta"]) - ref.beta),
        "x_error": float(np.max(np.abs(np.asarray(point["x"]) - ref.x))),
        "y_error": float(np.max(np.abs(np.asarray(point["y"]) - ref.y))),
        "sum_x_error": abs(float(point["sum_x"]) - ref.sum_x),
        "sum_y_error": abs(float(point["sum_y"]) - ref.sum_y),
        "z_liquid_error": abs(float(point["compressibility"]["liquid"]) - float(liquid_roots[0])),
        "z_vapor_error": abs(float(point["compressibility"]["vapor"]) - float(vapor_roots[-1])),
        "root_branches_match": point["root_branches"]
        == {"liquid": liquid_branch, "vapor": vapor_branch},
    }


def _finite_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _comparison_record(problem: FlashProblem, method: Method, result: Any) -> dict[str, Any]:
    report = getattr(result, "mpec_report", None)
    has_point = bool(getattr(result, "x", None))
    certified = bool(
        method in ("gdp", "sos1")
        and getattr(result, "gap_certified", False)
        and has_point
        and _finite_or_none(getattr(result, "objective", None)) is not None
        and _finite_or_none(getattr(result, "bound", None)) is not None
        and _finite_or_none(getattr(result, "gap", None)) is not None
    )
    state = (
        "certified" if certified else ("local" if method == "scholtes" and has_point else "failed")
    )
    record = {
        "temperature_k": problem.temperature_k,
        "method": method,
        "state": state,
        "status": str(getattr(result, "status", None)) if result is not None else None,
        "gap_certified": certified,
        "objective": _finite_or_none(getattr(result, "objective", None)),
        "bound": _finite_or_none(getattr(result, "bound", None)) if certified else None,
        "gap": _finite_or_none(getattr(result, "gap", None)) if certified else None,
        "node_count": int(getattr(result, "node_count", 0)) if result is not None else None,
        "wall_s": _finite_or_none(getattr(result, "wall_time", None)),
        "point": None,
        "source": None,
        "lowered": None,
        "oracle": None,
        "error": getattr(result, "error", None) if result is not None else "solve did not run",
    }
    if state not in ("certified", "local"):
        record["objective"] = None
        return record

    tolerance = COMPARISON_TOLERANCE
    if report is not None and getattr(report, "continuation", None) is not None:
        admitted = report.continuation.admitted_residual_scale
        if admitted is not None:
            tolerance = max(tolerance, float(admitted))
    try:
        point = _point_record(problem, result.x, tolerance)
        source = source_residuals(problem, result.x, mpec_report=report)
        lowered = getattr(report, "lowered_row_residual", None) if report is not None else None
        if lowered is None:
            raise ValueError("lowered-row residual was not measured")
    except Exception as exc:  # an incomplete record must not masquerade as a result
        record.update(
            state="failed",
            gap_certified=False,
            objective=None,
            bound=None,
            gap=None,
            error=f"comparison measurement failed: {type(exc).__name__}: {exc}",
        )
        return record
    record["point"] = point
    record["source"] = source
    record["lowered"] = {
        "row": _residual(
            float(lowered.value),
            str(lowered.definition),
            getattr(lowered, "admitted_scale", None),
        )
    }
    record["oracle"] = _oracle_comparison(problem, point)
    return record


def comparison_record_failures(
    record: Mapping[str, Any],
    *,
    tolerance: float = COMPARISON_TOLERANCE,
) -> list[str]:
    """Return every Gate-1 acceptance-contract violation in one record."""
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")

    failures: list[str] = []

    def finite(label: str, value: Any) -> Optional[float]:
        number = _finite_or_none(value)
        if number is None:
            failures.append(f"{label} is not finite")
        return number

    def bounded(label: str, value: Any) -> None:
        number = finite(label, value)
        if number is not None and not 0.0 <= number <= tolerance:
            failures.append(f"{label}={number:.3e} exceeds {tolerance:.3e}")

    method = str(record.get("method"))
    if record.get("error") not in (None, ""):
        failures.append(f"record has error: {record['error']}")

    if method in ("gdp", "sos1"):
        if record.get("state") != "certified":
            failures.append("exact method state is not certified")
        if record.get("status") != "optimal":
            failures.append("exact method status is not optimal")
        if record.get("gap_certified") is not True:
            failures.append("exact method gap is not certified")
        finite("objective", record.get("objective"))
        finite("bound", record.get("bound"))
        gap = finite("gap", record.get("gap"))
        if gap is not None and not 0.0 <= gap <= tolerance:
            failures.append(f"gap={gap:.3e} exceeds {tolerance:.3e}")
    elif method == "scholtes":
        if record.get("state") != "local":
            failures.append("Scholtes state is not local")
        if record.get("status") != "local_optimal":
            failures.append("Scholtes status is not local_optimal")
        if record.get("gap_certified") is not False:
            failures.append("Scholtes result claims a certified gap")
        finite("objective", record.get("objective"))
        if record.get("bound") is not None:
            failures.append("Scholtes bound must be null")
        if record.get("gap") is not None:
            failures.append("Scholtes gap must be null")
    else:
        failures.append(f"unsupported method {method!r}")

    if not isinstance(record.get("point"), Mapping):
        failures.append("physical point is missing")

    source = record.get("source")
    source_names = (
        "balance",
        "isofugacity",
        "eos",
        "root_selection",
        "nontriviality",
        "bound",
        "sign",
        "complementarity",
    )
    if not isinstance(source, Mapping):
        failures.append("source residuals are missing")
    else:
        for name in source_names:
            row = source.get(name)
            if not isinstance(row, Mapping):
                failures.append(f"source.{name} is missing")
                continue
            bounded(f"source.{name}", row.get("value"))

    lowered = record.get("lowered")
    lowered_row = lowered.get("row") if isinstance(lowered, Mapping) else None
    if not isinstance(lowered_row, Mapping):
        failures.append("lowered.row is missing")
    else:
        bounded("lowered.row", lowered_row.get("value"))

    oracle = record.get("oracle")
    if not isinstance(oracle, Mapping):
        failures.append("oracle comparison is missing")
    else:
        if oracle.get("regime_match") is not True:
            failures.append("oracle regime does not match")
        if oracle.get("root_branches_match") is not True:
            failures.append("oracle root branches do not match")
        if oracle.get("biactive_branch_admissible") is False:
            failures.append("oracle biactive branch is inadmissible")
        for name in (
            "beta_error",
            "x_error",
            "y_error",
            "sum_x_error",
            "sum_y_error",
            "z_liquid_error",
            "z_vapor_error",
        ):
            bounded(f"oracle.{name}", oracle.get(name))
    return failures


def solve_flash_temperature(
    temperature_k: float,
    method: Method,
    *,
    seed: Optional[FlashPoint] = None,
    time_limit: float = 30.0,
) -> FlashSolve:
    """Run one method and return its schema-ready comparison record."""
    if method not in ("gdp", "sos1", "scholtes"):
        raise ValueError("method must be 'gdp', 'sos1', or 'scholtes'")
    encoding: RootEncoding = "fixed" if method == "scholtes" else "disjunctive"
    problem = build_pounce_flash(temperature_k, seed=seed, root_encoding=encoding)
    validate_oracle_equivalence(problem)
    warm_start_accepted = False
    result: Any
    try:
        if method == "scholtes":
            x0 = validate_initial_solution(problem.model, problem.initial_solution)
            result = solve_mpec(
                problem.model,
                problem.pairs,
                method="scholtes",
                x0=x0,
                t0=1.0e-2,
                sigma=0.1,
                t_min=1.0e-8,
                max_iter=7,
                nlp_options={"tol": 1.0e-9, "max_iter": 500, "print_level": 0},
            )
        else:
            warm = accepted_exact_warm_start(problem)
            warm_start_accepted = warm is not None
            result = solve_mpec(
                problem.model,
                problem.pairs,
                method=method,
                initial_solution=warm,
                deterministic=True,
                time_limit=float(time_limit),
                gap_tolerance=1.0e-6,
                abs_gap_tolerance=1.0e-8,
            )
    except Exception as exc:  # benchmark records failures; it never erases them

        class Failed:
            status = "error"
            objective = None
            bound = None
            gap = None
            gap_certified = False
            node_count = 0
            wall_time = None
            x = None
            mpec_report = None
            error = f"{type(exc).__name__}: {exc}"

        result = Failed()
    return FlashSolve(
        problem, result, _comparison_record(problem, method, result), warm_start_accepted
    )


def augment_pounce_artifact(
    pounce_artifact: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]],
    *,
    discopt_commit: str,
    discopt_version: Optional[str] = None,
) -> dict[str, Any]:
    """Add or replace DiscOpt comparison records in a real POUNCE artifact."""
    if pounce_artifact.get("schema") not in ("pounce-flash-results/1", "pounce-flash-results/2"):
        raise ValueError("input must be a real pounce-flash-results/1 or /2 artifact")
    if len(discopt_commit) < 7:
        raise ValueError("discopt_commit must identify the exact repository revision")
    rows = [copy.deepcopy(dict(row)) for row in records]
    methods = ("gdp", "sos1", "scholtes")
    temperatures = sorted({float(row["temperature_k"]) for row in rows})
    expected = {(temperature, method) for temperature in temperatures for method in methods}
    actual = {(float(row["temperature_k"]), str(row["method"])) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("comparison records must contain each GDP/SOS1/Scholtes cell exactly once")

    artifact = copy.deepcopy(dict(pounce_artifact))
    artifact["schema"] = "pounce-flash-results/2"
    repository = artifact["stamp"]["repositories"]["discopt"]
    repository.update(
        present=True,
        version=discopt_version or importlib.metadata.version("discopt"),
        commit=discopt_commit,
        comparison_run=True,
        reason="Gate-1 reduced comparison completed by jkitchin/discopt#1526.",
    )
    artifact["comparison"] = {
        "state": "complete",
        "discopt_issue": "jkitchin/discopt#1526",
        "temperatures_k": temperatures,
        "methods": list(methods),
        "records": rows,
        "reason": None,
    }
    return artifact


def validate_comparison_artifact(
    artifact: Mapping[str, Any], schema_path: Optional[str | Path] = None
) -> None:
    """Validate against POUNCE's packaged version-2 JSON Schema."""
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - benchmark environment concern
        raise RuntimeError(
            "artifact validation requires the optional 'jsonschema' package"
        ) from exc
    if schema_path is None:
        candidate = resources.files("pounce.examples").joinpath("flash_results_v2.schema.json")
        if not candidate.is_file():
            raise FileNotFoundError(
                "POUNCE does not ship flash_results_v2.schema.json; install a release "
                "containing jkitchin/pounce#972 or pass schema_path explicitly"
            )
        with candidate.open("r", encoding="utf-8") as handle:
            schema = json.load(handle)
    else:
        with Path(schema_path).open("r", encoding="utf-8") as handle:
            schema = json.load(handle)
    jsonschema.Draft7Validator(schema).validate(dict(artifact))


__all__ = [
    "COMPARISON_TOLERANCE",
    "DISCRIMINANT_GUARD",
    "FULL_TEMPERATURES",
    "FlashPoint",
    "FlashProblem",
    "FlashSolve",
    "SMOKE_TEMPERATURES",
    "accepted_exact_warm_start",
    "augment_pounce_artifact",
    "build_pounce_flash",
    "comparison_record_failures",
    "oracle_point",
    "pounce_seed",
    "solve_flash_temperature",
    "source_residuals",
    "validate_comparison_artifact",
    "validate_oracle_equivalence",
]
