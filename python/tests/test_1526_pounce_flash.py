"""Gate-1 POUNCE flash cross-validation (#1526)."""

from __future__ import annotations

import copy
import importlib.resources
import json
from pathlib import Path

import numpy as np
import pytest
from discopt.benchmarks.problems.pounce_flash import (
    FULL_TEMPERATURES,
    SMOKE_TEMPERATURES,
    accepted_exact_warm_start,
    augment_pounce_artifact,
    build_pounce_flash,
    comparison_record_failures,
    solve_flash_temperature,
    source_residuals,
    validate_comparison_artifact,
    validate_oracle_equivalence,
)


@pytest.mark.parametrize("temperature", SMOKE_TEMPERATURES)
def test_every_smoke_oracle_executes_every_source_family(temperature):
    problem = build_pounce_flash(temperature, root_encoding="disjunctive")
    measured = validate_oracle_equivalence(problem)
    assert len(measured) == 8, "a zero-check entry experiment is not evidence"
    assert set(measured) == {
        "balance",
        "isofugacity",
        "eos",
        "root_selection",
        "nontriviality",
        "bound",
        "sign",
        "complementarity",
    }
    assert [pair.name for pair in problem.pairs] == ["vapor", "liquid"]


def test_path_contract_is_the_pounce_fixture():
    assert SMOKE_TEMPERATURES == (250.0, 268.0, 300.0, 324.0, 350.0)
    assert len(FULL_TEMPERATURES) == 34
    assert FULL_TEMPERATURES[0] == 230.0
    assert FULL_TEMPERATURES[-1] == 360.0


def test_source_model_exposes_explicit_boolean_identities():
    problem = build_pounce_flash(300.0, root_encoding="disjunctive")
    assert problem.root_branches == ("three_real_roots", "three_real_roots")
    assert all(one is not None and three is not None for one, three in problem.root_indicators)
    assert len(problem.nontrivial_indicators) == 4
    assert problem.nontrivial_branch in {
        "component_0_above",
        "component_0_below",
        "component_1_above",
        "component_1_below",
    }


def _copied_initial_solution(problem):
    return {
        variable: np.asarray(value).copy() for variable, value in problem.initial_solution.items()
    }


def test_source_residuals_measure_active_vieta_rows_not_only_rootwise_cubics():
    problem = build_pounce_flash(300.0, root_encoding="disjunctive")
    solution = _copied_initial_solution(problem)
    roots = solution[problem.variables["three_root_z"]]
    roots[0, 2] = roots[0, 1]

    measured = source_residuals(problem, solution)

    assert measured["eos"]["value"] > 0.1


def test_source_residuals_measure_the_selected_nontriviality_row():
    problem = build_pounce_flash(300.0, root_encoding="disjunctive")
    solution = _copied_initial_solution(problem)
    for indicator in problem.nontrivial_indicators:
        solution[indicator] = np.asarray(0.0)
    solution[problem.nontrivial_indicators[1]] = np.asarray(1.0)

    measured = source_residuals(problem, solution)

    assert measured["nontriviality"]["value"] > 1.0


def test_source_residuals_measure_normalization_rows():
    problem = build_pounce_flash(300.0, root_encoding="disjunctive")
    solution = _copied_initial_solution(problem)
    solution[problem.variables["sum_x"]] = np.asarray(0.8)

    measured = source_residuals(problem, solution)

    assert measured["balance"]["value"] > 0.01


def test_source_residuals_measure_boolean_selector_identities():
    problem = build_pounce_flash(300.0, root_encoding="disjunctive")
    solution = _copied_initial_solution(problem)
    one, three = problem.root_indicators[0]
    assert one is not None and three is not None
    solution[one] = np.asarray(0.5)
    solution[three] = np.asarray(0.5)

    measured = source_residuals(problem, solution)

    assert measured["root_selection"]["value"] >= 0.5


def test_pounce_point_passes_the_only_exact_warm_start_gate():
    problem = build_pounce_flash(300.0, root_encoding="disjunctive")
    warm = accepted_exact_warm_start(problem)
    assert warm is problem.initial_solution
    # The gate returns only a primal objective to its caller. It does not and
    # cannot attach a dual bound to a local POUNCE point.
    assert not hasattr(warm, "bound")


@pytest.mark.slow
@pytest.mark.parametrize("method", ["gdp", "sos1", "scholtes"])
def test_entry_point_preserves_exact_vs_local_contract(method):
    solved = solve_flash_temperature(300.0, method, time_limit=30.0)
    record = solved.record
    assert record["point"]["regime"] == "two_phase", record
    assert record["oracle"]["regime_match"], record
    assert record["oracle"]["root_branches_match"], record
    assert record["source"]["balance"]["value"] < 1.0e-6
    assert record["source"]["isofugacity"]["value"] < 1.0e-6
    assert record["source"]["eos"]["value"] < 1.0e-6
    assert record["lowered"]["row"]["value"] < 1.0e-6
    if method == "scholtes":
        assert record["state"] == "local"
        assert record["gap_certified"] is False
        assert record["bound"] is None
        assert record["gap"] is None
    else:
        assert solved.warm_start_accepted
        assert record["state"] == "certified"
        assert record["gap_certified"] is True
        assert record["bound"] == pytest.approx(record["objective"])


def _failed_record(method: str) -> dict:
    return {
        "temperature_k": 300.0,
        "method": method,
        "state": "failed",
        "status": "error",
        "gap_certified": False,
        "objective": None,
        "bound": None,
        "gap": None,
        "node_count": 0,
        "wall_s": 0.0,
        "point": None,
        "source": None,
        "lowered": None,
        "oracle": None,
        "error": "deliberate contract fixture",
    }


def _successful_record(method: str) -> dict:
    path = (
        Path(__file__).parents[2]
        / "discopt_benchmarks"
        / "results"
        / "issue1526"
        / "pounce_gate1_smoke_v2.json"
    )
    with path.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    return copy.deepcopy(
        next(
            row
            for row in artifact["comparison"]["records"]
            if row["temperature_k"] == 300.0 and row["method"] == method
        )
    )


@pytest.mark.parametrize("method", ["gdp", "sos1", "scholtes"])
def test_gate_acceptance_contract_accepts_committed_successful_records(method):
    assert comparison_record_failures(_successful_record(method)) == []


def test_gate_acceptance_contract_rejects_success_lookalikes():
    mutations = [
        ("local-limit", "gdp", lambda row: row.update(status="local_limit")),
        ("regime", "gdp", lambda row: row["oracle"].update(regime_match=False)),
        ("roots", "gdp", lambda row: row["oracle"].update(root_branches_match=False)),
        ("beta", "gdp", lambda row: row["oracle"].update(beta_error=1.0e-3)),
        ("biactive", "gdp", lambda row: row["oracle"].update(biactive_branch_admissible=False)),
        ("source", "gdp", lambda row: row["source"]["balance"].update(value=1.0e-3)),
        ("lowered", "gdp", lambda row: row["lowered"]["row"].update(value=1.0e-3)),
    ]
    for label, method, mutate in mutations:
        record = _successful_record(method)
        mutate(record)
        assert comparison_record_failures(record), label


def _v1_artifact() -> dict:
    return {
        "schema": "pounce-flash-results/1",
        "issue": "gh#776 Gate 1",
        "stamp": {
            "repositories": {
                "pounce": {"commit": "1234567", "describe": "test"},
                "discopt": {
                    "present": False,
                    "commit": None,
                    "comparison_run": False,
                    "reason": "not run",
                },
            },
            "model_data_revision": "12345678",
            "environment": {},
            "model": {
                "case": "ethane_n_butane_10bar",
                "components": ["ethane", "n-butane"],
                "feed_composition": [0.5, 0.5],
                "pressure_pa": 1.0e6,
                "temperatures_k": [300.0],
            },
            "started_utc": "2026-09-28T00:00:00Z",
        },
        "config": {"base_options": {}, "supported_route": "test", "tau_schedule": []},
        "oracle": {"rows": [], "method": "test"},
        "hysteresis": {
            "disagreements": {},
            "iterations": {},
            "failures": {},
            "cold_legs_agree": True,
            "path_dependent": False,
        },
        "legs": [],
    }


def test_v2_augmentation_preserves_pounce_evidence_and_stamps_discopt():
    source = _v1_artifact()
    records = [_failed_record(method) for method in ("gdp", "sos1", "scholtes")]
    artifact = augment_pounce_artifact(
        source,
        records,
        discopt_commit="abcdef0123456789",
        discopt_version="test",
    )
    assert source["schema"] == "pounce-flash-results/1", "input must not be mutated"
    assert artifact["schema"] == "pounce-flash-results/2"
    assert artifact["stamp"]["repositories"]["pounce"] == source["stamp"]["repositories"]["pounce"]
    assert artifact["stamp"]["repositories"]["discopt"]["commit"] == "abcdef0123456789"
    assert artifact["comparison"]["state"] == "complete"


def test_tracked_v2_artifact_is_a_reproducible_input():
    path = (
        Path(__file__).parents[2]
        / "discopt_benchmarks"
        / "results"
        / "issue1526"
        / "pounce_gate1_smoke_v2.json"
    )
    with path.open("r", encoding="utf-8") as handle:
        source = json.load(handle)
    original = copy.deepcopy(source)
    records = [_failed_record(method) for method in ("gdp", "sos1", "scholtes")]

    artifact = augment_pounce_artifact(
        source,
        records,
        discopt_commit="abcdef0123456789",
        discopt_version="test",
    )

    assert source == original, "input must not be mutated"
    assert artifact["oracle"] == source["oracle"]
    assert artifact["legs"] == source["legs"]
    assert artifact["comparison"]["records"] == records


def test_v2_artifact_validates_when_companion_schema_is_installed():
    pytest.importorskip("jsonschema")
    schema = importlib.resources.files("pounce.examples").joinpath("flash_results_v2.schema.json")
    if not schema.is_file():
        pytest.skip("requires companion POUNCE schema PR #972")
    records = [_failed_record(method) for method in ("gdp", "sos1", "scholtes")]
    artifact = augment_pounce_artifact(
        _v1_artifact(),
        records,
        discopt_commit="abcdef0123456789",
        discopt_version="test",
    )
    validate_comparison_artifact(artifact)
