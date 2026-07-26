"""Locks for the shared reference-optima accessor (issue #862).

A quality metric is only as good as the reference it scores against. Before this,
every panel re-implemented the ``minlplib.solu`` parse or pasted a private
``OPT = {...}`` dict — and since the ``.solu`` snapshot is a local artifact absent
from CI, a script keyed to it measured nothing there without saying so. These tests
pin the fallback chain that keeps the panel scoring in CI, and the provenance
tagging that stops an unproven ``=best=`` value being treated as ground truth.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.reference_optima import (
    _parse_solu,
    oracle_table,
    reference_optimum,
    reference_oracle,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def vendored_only(monkeypatch, tmp_path):
    """Force the CI configuration: no .solu snapshot reachable.

    ``oracle_table`` is ``lru_cache``d, so pointing the env var elsewhere is not
    enough -- the cache has to be cleared on the way in AND on the way out, or this
    test resolves through a real snapshot on a machine that has one (and poisons the
    cache for every test after it).
    """
    monkeypatch.setenv("DISCOPT_MINLPLIB_SOLU", str(tmp_path / "absent.solu"))
    oracle_table.cache_clear()
    yield
    oracle_table.cache_clear()


class TestVendoredSources:
    def test_resolves_without_a_solu_snapshot(self, vendored_only):
        # The CI configuration: no .solu on disk, oracles still available.
        from utils.reference_optima import solu_path

        assert solu_path() is None, "fixture failed to neutralise the snapshot"
        assert reference_optimum("nvs17") == pytest.approx(-1100.4)
        assert reference_optimum("nvs19") == pytest.approx(-1098.4)
        assert reference_optimum("nvs24") == pytest.approx(-1033.2)

    def test_covers_the_lp_node_engine_scope_added_for_862(self, vendored_only):
        for name in ("nvs03", "nvs10", "nvs11", "nvs12", "nvs15", "st_miqp1", "st_testgr3"):
            assert reference_optimum(name) is not None, name

    def test_unknown_instance_is_none_so_a_sweep_can_continue(self):
        # A panel must run past unknown instances and report them as unscored;
        # raising here would make one missing oracle abort a corpus sweep.
        assert reference_optimum("not_a_real_instance") is None
        assert reference_oracle("not_a_real_instance") is None

    def test_carries_provenance(self):
        entry = reference_oracle("nvs17")
        assert entry is not None
        assert entry.source
        assert entry.proven is True

    def test_table_is_non_empty_and_numeric(self):
        table = oracle_table()
        assert len(table) > 20
        assert all(isinstance(o.value, float) for o in table.values())


class TestSoluParsing:
    def test_reads_opt_and_best_and_tags_proof_status(self, tmp_path):
        p = tmp_path / "minlplib.solu"
        p.write_text("=opt= tln4 8.3\n=best= tln6 15.3\n=bestdual= tln6 12.0\n")
        table = _parse_solu(p)
        assert table["tln4"].value == pytest.approx(8.3)
        assert table["tln4"].proven is True
        # =best= is a published incumbent, not a proof — gating soundness on it as
        # if it were ground truth would manufacture false violations.
        assert table["tln6"].value == pytest.approx(15.3)
        assert table["tln6"].proven is False
        assert "=bestdual=" not in str(table)

    def test_opt_wins_over_best_for_the_same_instance(self, tmp_path):
        p = tmp_path / "minlplib.solu"
        p.write_text("=best= tln5 11.0\n=opt= tln5 10.3\n")
        assert _parse_solu(p)["tln5"].value == pytest.approx(10.3)
        assert _parse_solu(p)["tln5"].proven is True

    def test_ignores_malformed_and_unknown_tag_lines(self, tmp_path):
        p = tmp_path / "minlplib.solu"
        p.write_text("=unbounded= foo\n=opt= bar notanumber\n\n=opt= baz 1.5\n")
        table = _parse_solu(p)
        assert list(table) == ["baz"]


class TestRegistryIntegrity:
    def test_every_added_entry_is_well_formed(self):
        """The registry is a correctness artifact: a typo'd optimum would silently
        turn a good incumbent into a reported regression (or hide a false primal)."""
        import tomllib

        path = (
            Path(__file__).resolve().parents[2] / "python" / "tests" / "data" / "known_optima.toml"
        )
        with path.open("rb") as fh:
            data = tomllib.load(fh)
        data.pop("schema", None)
        for name, entry in data.items():
            assert isinstance(entry.get("optimum"), (int, float)), f"{name}: optimum not numeric"
            assert entry.get("source"), f"{name}: missing source"
            assert entry.get("status"), f"{name}: missing status"

    def test_registry_agrees_with_the_legacy_in_repo_constants(self, vendored_only):
        """Cross-check against the tables the values were consolidated from, so a
        divergence shows up as a test failure rather than two disagreeing oracles."""
        legacy = {
            "nvs03": 16.0,
            "nvs04": 0.72,
            "nvs06": 1.77031250,
            "nvs07": 4.0,
            "nvs10": -310.80,
            "nvs11": -431.0,
            "nvs12": -481.20,
            "nvs15": 1.0,
            "nvs16": 0.70312500,
            "nvs17": -1100.4,
            "nvs19": -1098.4,
            "nvs24": -1033.2,
            "prob03": 10.0,
            "gear": 0.0,
        }
        for name, value in legacy.items():
            assert reference_optimum(name) == pytest.approx(value), name
