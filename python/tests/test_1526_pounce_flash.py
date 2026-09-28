"""Gate-1 POUNCE flash cross-validation (#1526)."""

from __future__ import annotations

import importlib.resources

import pytest
from discopt.benchmarks.problems.pounce_flash import (
    FULL_TEMPERATURES,
    SMOKE_TEMPERATURES,
    accepted_exact_warm_start,
    augment_pounce_artifact,
    build_pounce_flash,
    solve_flash_temperature,
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
