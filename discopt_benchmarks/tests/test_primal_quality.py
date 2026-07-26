"""Locks for the incumbent-quality metric (issue #862).

The #844 differential panel scored whether an incumbent existed, was sound, and
stayed in budget — never how good it was, so "a change could halve incumbent
quality and the panel would still pass". These tests pin the metric that closes
that hole, and in particular the two properties the panel depends on:

* an unscoreable instance is reported as *unscored*, never as a clean 0 gap;
* halving incumbent quality is detected as a regression.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.primal_quality import (
    is_false_primal,
    primal_gap,
    quality_regressions,
    relative_excess,
    summarize,
)

pytestmark = pytest.mark.unit


class TestPrimalGap:
    def test_exact_optimum_scores_zero(self):
        assert primal_gap(15.3, 15.3) == 0.0
        assert primal_gap(0.0, 0.0) == 0.0
        assert primal_gap(-1100.4, -1100.4) == 0.0

    def test_issue_862_headline_numbers(self):
        # tln4/5/6 from the issue: 9.3 vs 8.3, 32.2 vs 10.3, 65.3 vs 15.3.
        assert primal_gap(9.3, 8.3) == pytest.approx(1.0 / 9.3)
        assert primal_gap(32.2, 10.3) == pytest.approx(21.9 / 32.2)
        assert primal_gap(65.3, 15.3) == pytest.approx(50.0 / 65.3)

    def test_bounded_in_unit_interval(self):
        for obj, opt in [(1e9, 1.0), (1.0, 1e9), (-1e9, -1.0), (5.0, -5.0), (0.0, 7.0)]:
            g = primal_gap(obj, opt)
            assert 0.0 <= g <= 1.0

    def test_straddling_zero_saturates_rather_than_exploding(self):
        # A plain relative error would flip sign or divide by ~0 here.
        assert primal_gap(5.0, -5.0) == 1.0
        assert primal_gap(0.0, -2.0) == 1.0

    def test_numerically_exact_solution_at_a_zero_optimum_scores_zero(self):
        """Berthold's exact ``p == o`` saturates to 1.0 here — the maximum gap — on
        instances discopt solved to numerical exactness. The first #862 panel run hit
        this on gear (2.9e-07) and st_test1 (-1.6e-08) and reported the corpus mean as
        0.10 instead of ~0. At a zero optimum no relative measure is defined, so the
        comparison has to be absolute."""
        assert primal_gap(2.860864189286471e-07, 0.0) == 0.0
        assert primal_gap(-1.6495994490692313e-08, 0.0) == 0.0

    def test_a_genuinely_bad_incumbent_at_a_zero_optimum_still_saturates(self):
        # The tolerance must not become a blanket excuse for a zero optimum.
        assert primal_gap(3.0, 0.0) == 1.0
        assert primal_gap(-0.5, 0.0) == 1.0

    def test_absolute_tolerance_is_tunable_and_defaults_to_the_repo_value(self):
        assert primal_gap(1e-5, 0.0) == 1.0  # outside the default 1e-6
        assert primal_gap(1e-5, 0.0, atol=1e-4) == 0.0

    def test_tolerance_applies_away_from_zero_too(self):
        assert primal_gap(-1100.4000000001, -1100.4) == 0.0

    def test_missing_incumbent_or_oracle_is_none_not_zero(self):
        # The load-bearing distinction: "unscored" must never read as "perfect".
        assert primal_gap(None, 15.3) is None
        assert primal_gap(65.3, None) is None
        assert primal_gap(None, None) is None

    def test_non_finite_is_unscored(self):
        assert primal_gap(float("inf"), 1.0) is None
        assert primal_gap(1.0, float("nan")) is None

    def test_symmetric_in_its_arguments(self):
        assert primal_gap(65.3, 15.3) == pytest.approx(primal_gap(15.3, 65.3))


class TestRelativeExcess:
    def test_reproduces_the_percentages_quoted_in_the_issue(self):
        assert relative_excess(9.3, 8.3, "min") == pytest.approx(0.1205, abs=1e-4)  # +12%
        assert relative_excess(32.2, 10.3, "min") == pytest.approx(2.1262, abs=1e-4)  # +213%
        assert relative_excess(65.3, 15.3, "min") == pytest.approx(3.2680, abs=1e-4)  # +327%

    def test_positive_always_means_worse_for_either_sense(self):
        assert relative_excess(12.0, 10.0, "min") > 0
        assert relative_excess(8.0, 10.0, "max") > 0
        assert relative_excess(8.0, 10.0, "min") < 0
        assert relative_excess(12.0, 10.0, "max") < 0

    def test_undefined_at_zero_optimum(self):
        assert relative_excess(1.0, 0.0, "min") is None


class TestFalsePrimal:
    def test_incumbent_below_reference_optimum_is_a_soundness_failure(self):
        assert is_false_primal(-1200.0, -1100.4, "min")
        assert is_false_primal(20.0, 15.3, "max")

    def test_worse_than_optimal_is_never_a_soundness_failure(self):
        assert not is_false_primal(65.3, 15.3, "min")
        assert not is_false_primal(10.0, 15.3, "max")

    def test_tolerance_absorbs_float_noise(self):
        assert not is_false_primal(-1100.4000001, -1100.4, "min")

    def test_unscoreable_never_reports_a_violation(self):
        assert not is_false_primal(None, -1100.4, "min")
        assert not is_false_primal(-1200.0, None, "min")


class TestSenseIsRequired:
    """The review's blocking finding: ``sense`` used to default to ``"min"``, and the
    .nl reader does NOT normalize to MINIMIZE (syn05m, syn05hfsg, bchoco06/07/08 all
    load as MAXIMIZE). A wrong sense manufactures a correctness violation, so it must
    raise rather than guess."""

    def test_unrecognised_sense_raises(self):
        with pytest.raises(ValueError, match="sense must be one of"):
            is_false_primal(1.0, 2.0, "minimize")
        with pytest.raises(ValueError, match="sense must be one of"):
            relative_excess(1.0, 2.0, "MIN")

    def test_summarize_requires_a_sense_on_every_row(self):
        with pytest.raises(KeyError, match="objective sense is required"):
            summarize([{"name": "a", "objective": 9.3, "optimum": 8.3}])

    def test_a_sound_maximize_incumbent_is_not_a_false_primal(self):
        # Assuming "min" here would flag a perfectly sound incumbent as unsound.
        assert not is_false_primal(0.9 * 5685067.877, 5685067.877, "max")
        assert is_false_primal(0.9 * 5685067.877, 5685067.877, "min")

    def test_a_worse_maximize_incumbent_reports_as_worse_not_better(self):
        # Under an assumed "min" this reported -0.100 -- "better than optimal".
        assert relative_excess(9.0, 10.0, "max") == pytest.approx(0.1)
        assert relative_excess(9.0, 10.0, "min") == pytest.approx(-0.1)


class TestSummarize:
    def test_counts_unscored_separately_from_scored(self):
        s = summarize(
            [
                {"name": "a", "objective": 9.3, "optimum": 8.3, "sense": "min"},
                {"name": "b", "objective": None, "optimum": 15.3, "sense": "min"},  # no incumbent
                {"name": "c", "objective": 4.0, "optimum": None, "sense": "min"},  # no oracle
            ]
        )
        assert (s.scored, s.unscored, s.with_incumbent) == (1, 2, 2)

    def test_missing_incumbents_are_not_folded_in_as_gap_one(self):
        # Otherwise a change trading incumbents for quality would look neutral.
        with_none = summarize(
            [
                {"name": "a", "objective": 10.0, "optimum": 10.0, "sense": "min"},
                {"name": "b", "objective": None, "optimum": 1.0, "sense": "min"},
            ]
        )
        assert with_none.mean_gap == 0.0
        assert with_none.with_incumbent == 1

    def test_reports_the_worst_instance_by_name(self):
        s = summarize(
            [
                {"name": "tln4", "objective": 9.3, "optimum": 8.3, "sense": "min"},
                {"name": "tln6", "objective": 65.3, "optimum": 15.3, "sense": "min"},
            ]
        )
        assert s.worst_instance == "tln6"
        assert s.worst_gap == pytest.approx(50.0 / 65.3)

    def test_false_primals_are_counted_not_averaged_away(self):
        """primal_gap is symmetric, so a false primal scores as *better* than an
        honest incumbent the same distance away (min, opt 10: honest 12 -> 0.167,
        false 9 -> 0.100). Folding it into mean_gap would let a soundness failure
        read as a quality improvement, so it is counted and excluded."""
        s = summarize([{"name": "x", "objective": -1200.0, "optimum": -1100.4, "sense": "min"}])
        assert s.false_primals == 1
        assert s.scored == 0
        assert s.mean_gap is None  # not averaged in
        assert s.unscored == 1

    def test_a_false_primal_does_not_improve_a_corpus_mean(self):
        honest = summarize([{"name": "a", "objective": 12.0, "optimum": 10.0, "sense": "min"}])
        with_false = summarize(
            [
                {"name": "a", "objective": 12.0, "optimum": 10.0, "sense": "min"},
                {"name": "b", "objective": 9.0, "optimum": 10.0, "sense": "min"},
            ]
        )
        assert with_false.false_primals == 1
        assert with_false.mean_gap == pytest.approx(honest.mean_gap)

    def test_empty_corpus_yields_no_statistics_rather_than_zeros(self):
        s = summarize([])
        assert s.mean_gap is None and s.worst_gap is None and s.scored == 0

    def test_a_corpus_with_no_oracles_scores_nothing(self):
        """The precondition the panel's vacuity guard keys on. Without a
        ``scored > 0`` check, ``not regressions`` is trivially True over a corpus
        where nothing could be scored -- a --gate-quality run would pass having
        measured NOTHING, which is the exact failure this panel exists to prevent."""
        s = summarize(
            [
                {"name": f"i{i}", "objective": 1e6 * i, "optimum": None, "sense": "min"}
                for i in range(30)
            ]
        )
        assert s.scored == 0
        assert s.unscored == 30
        assert s.with_incumbent == 30


class TestQualityRegressions:
    def test_detects_the_failure_mode_issue_862_names(self):
        # "a change could halve incumbent quality and the panel would still pass"
        baseline = [{"name": "tln5", "objective": 11.0, "optimum": 10.3}]
        candidate = [{"name": "tln5", "objective": 32.2, "optimum": 10.3}]
        (reg,) = quality_regressions(baseline, candidate)
        assert reg["name"] == "tln5"
        assert reg["candidate_gap"] > reg["baseline_gap"]
        assert reg["delta"] == pytest.approx(reg["candidate_gap"] - reg["baseline_gap"])

    def test_improvement_is_not_a_regression(self):
        baseline = [{"name": "tln6", "objective": 65.3, "optimum": 15.3}]
        candidate = [{"name": "tln6", "objective": 16.0, "optimum": 15.3}]
        assert quality_regressions(baseline, candidate) == []

    def test_lost_incumbent_is_left_to_the_existing_gate(self):
        # Double-counting it here would obscure which gate actually fired.
        baseline = [{"name": "tln6", "objective": 65.3, "optimum": 15.3}]
        candidate = [{"name": "tln6", "objective": None, "optimum": 15.3}]
        assert quality_regressions(baseline, candidate) == []

    def test_gained_incumbent_is_not_scored_as_a_regression(self):
        baseline = [{"name": "tln6", "objective": None, "optimum": 15.3}]
        candidate = [{"name": "tln6", "objective": 65.3, "optimum": 15.3}]
        assert quality_regressions(baseline, candidate) == []

    def test_instances_absent_from_the_baseline_are_skipped(self):
        assert quality_regressions([], [{"name": "new", "objective": 1.0, "optimum": 0.5}]) == []

    def test_worst_regression_is_reported_first(self):
        baseline = [
            {"name": "a", "objective": 10.0, "optimum": 10.0},
            {"name": "b", "objective": 10.0, "optimum": 10.0},
        ]
        candidate = [
            {"name": "a", "objective": 11.0, "optimum": 10.0},
            {"name": "b", "objective": 100.0, "optimum": 10.0},
        ]
        regs = quality_regressions(baseline, candidate)
        assert [r["name"] for r in regs] == ["b", "a"]
