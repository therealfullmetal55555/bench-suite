"""Scores, and the aggregation the board prints.

The decision under test is in the module docstring of `core/score.py`: a score is a
distribution over tasks, not a float. So the interesting tests are the ones where a
number would have been wrong — a family mean mistaken for the headline, an error
averaged away, a task decided by an untrusted judge counted anyway.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from bench.core.score import (
    Score,
    TaskScore,
    family_mean,
    family_scores,
    stratified_mean,
    summarise,
)
from bench.core.stats import Interval


def make(
    task_id: str,
    *,
    family: str = "grounded-qa",
    outcome: str = "pass",
    score: float = 1.0,
    repeats: int = 1,
    passes: int = 1,
    judge_scored: bool = False,
    judge_trusted: bool = True,
) -> TaskScore:
    return TaskScore(
        task_id,
        family,
        outcome,
        score,
        repeats=repeats,
        passes=passes,
        judge_scored=judge_scored,
        judge_trusted=judge_trusted,
    )


# --- a score knows what it is -------------------------------------------------


def test_a_score_outside_zero_to_one_is_refused():
    """The wrong number that travels. A percentage written as `87` instead of `0.87`
    would rank first on every board and nobody would notice until the site looked
    odd."""
    with pytest.raises(ValueError, match="outside"):
        Score(87.0, 10, "aggregate")


def test_a_score_outside_its_own_interval_is_refused():
    """Means this the value was computed from a different set of tasks than the
    interval, which is the bug that produces a plausible-looking number with a
    bracket around it that does not contain it."""
    with pytest.raises(ValueError, match="outside its own interval"):
        Score(0.9, 10, "aggregate", Interval(0.1, 0.4))


def test_a_score_with_no_interval_says_so_in_its_string():
    """`n=` is always printed: a score without a task count invites the reader to
    assume one."""
    assert str(Score(0.75, 12, "grounded-qa")) == "0.750 (n=12)"
    assert str(Score(0.75, 12, "grounded-qa", Interval(0.6, 0.9))) == (
        "0.750 [60.0%, 90.0%] (n=12)"
    )


def test_a_negative_task_count_is_a_programming_error():
    with pytest.raises(ValueError, match="negative task count"):
        Score(0.5, -1, "aggregate")


def test_more_passes_than_repeats_is_refused():
    with pytest.raises(ValueError, match="repeats"):
        make("gqa-0001", repeats=3, passes=4)


def test_a_task_score_outside_the_range_is_refused():
    with pytest.raises(ValueError, match="outside"):
        make("gqa-0001", score=1.4)


# --- what a task score knows about itself ------------------------------------


def test_a_task_that_split_its_repeats_is_flaky():
    assert make("gqa-0001", repeats=5, passes=2).flaky
    assert make("gqa-0001", repeats=5, passes=3).flaky
    assert not make("gqa-0001", repeats=5, passes=5).flaky
    assert not make("gqa-0001", repeats=5, passes=0).flaky


def test_a_deterministic_failure_is_not_flaky():
    """Zero out of five is a failure, not a coin flip. The two get different
    treatment on the board: one is a score, the other is an asterisk."""
    failed = make("gqa-0001", outcome="fail", score=0.0, repeats=5, passes=0)
    assert failed.failed and not failed.flaky


def test_an_error_counts_as_a_failure():
    """Not a separate bucket. An entry that errored on 12% of tasks scores zero on
    them and pays for it, and the 12% is published beside the score."""
    errored = make("gqa-0001", outcome="error", score=0.0)
    assert errored.failed
    assert errored.counts_towards_quality


def test_a_skipped_task_does_not_count_towards_quality():
    """The one outcome that is not a judgement about the entrant: it is a judgement
    about the run, and averaging it in would punish an entry for being partial."""
    skipped = make("gqa-0001", outcome="skipped", score=0.0)
    assert not skipped.failed
    assert not skipped.counts_towards_quality


def test_a_task_decided_by_an_untrusted_judge_is_kept_but_not_counted():
    """Never deleted: "the judge was not good enough here" is a fact about the result
    that a reader needs. Never counted: a judge below its floor cannot decide a rank.
    """
    untrusted = make("gqa-0001", judge_scored=True, judge_trusted=False)
    assert not untrusted.counts_towards_quality
    assert untrusted.judge_scored
    grouped = family_scores([untrusted], trusted_only=False)
    assert grouped == {"grounded-qa": [1.0]}
    assert family_scores([untrusted]) == {}


def test_over_budget_is_a_failure_with_its_own_outcome():
    """An agent that answers correctly after four dollars has not answered."""
    spendthrift = make("gqa-0001", outcome="over-budget", score=0.0)
    assert spendthrift.failed


# --- aggregation -------------------------------------------------------------


def test_the_family_mean_averages_its_own_tasks():
    scores = [
        make("gqa-0001", score=1.0),
        make("gqa-0002", score=0.0),
        make("traj-0001", family="tool-trajectory", score=1.0),
    ]
    assert family_mean(scores, "grounded-qa").value == pytest.approx(0.5)
    assert family_mean(scores, "grounded-qa").n == 2
    assert family_mean(scores, "tool-trajectory").value == pytest.approx(1.0)


def test_a_family_with_no_scorable_tasks_is_zero_out_of_zero():
    """Not an error: a family can be entirely judge-decided by an untrusted judge,
    and the report has to say `0.000 (n=0)` rather than crash or invent a number."""
    scores = [make("gqa-0001", judge_scored=True, judge_trusted=False)]
    empty = family_mean(scores, "grounded-qa")
    assert empty.n == 0 and empty.value == 0.0


def test_the_aggregate_is_the_mean_of_family_means_not_of_tasks():
    """The whole point of stratifying. Eighty tasks in one family and twelve in
    another must not give the big family six times the vote."""
    scores = [make(f"gqa-{index:04d}", score=1.0) for index in range(80)]
    scores += [
        make(f"traj-{index:04d}", family="tool-trajectory", score=0.0) for index in range(12)
    ]
    aggregate = stratified_mean(scores, interval=False)
    assert aggregate.value == pytest.approx(0.5)  # not 0.87
    assert aggregate.n == 92
    assert aggregate.families == 2


def test_the_aggregate_carries_an_interval_and_its_own_family_count():
    scores = [make("gqa-0001", score=1.0), make("gqa-0002", score=0.0)]
    scores += [make("traj-0001", family="tool-trajectory", score=1.0)]
    aggregate = stratified_mean(scores)
    assert aggregate.interval is not None
    assert aggregate.value in aggregate.interval
    assert aggregate.families == 2


def test_the_aggregate_is_seeded_so_two_board_builds_agree():
    scores = [make(f"gqa-{index:04d}", score=float(index % 2)) for index in range(12)]
    assert stratified_mean(scores).interval == stratified_mean(scores).interval


def test_two_tasks_is_the_smallest_sample_that_gets_an_interval():
    """Below that the bootstrap has nothing to resample, and an interval that is a
    point is printed as a point rather than dressed up as uncertainty."""
    assert stratified_mean([make("gqa-0001", score=1.0)], interval=True).interval is None
    two = [make("gqa-0001", score=1.0), make("gqa-0002", score=0.0)]
    assert stratified_mean(two, interval=True).interval is not None


def test_an_empty_league_scores_zero_out_of_zero():
    empty = stratified_mean([], interval=False)
    assert empty.value == 0.0 and empty.n == 0 and empty.families == 0


def test_the_aggregate_ignores_skipped_and_untrusted_tasks():
    scores = [
        make("gqa-0001", score=1.0),
        make("gqa-0002", outcome="skipped", score=0.0),
        make("gqa-0003", judge_scored=True, judge_trusted=False),
    ]
    aggregate = stratified_mean(scores, interval=False)
    assert aggregate.value == pytest.approx(1.0)
    assert aggregate.n == 1


def test_family_means_clip_an_interval_that_escaped_the_bounds():
    """A percentile bootstrap of a mean of bounded values can land marginally outside
    the value's own range on tiny samples. Clipping beats refusing: refusing here
    would turn a two-task fixture into a crash."""
    scores = [make("gqa-0001", score=1.0), make("gqa-0002", score=1.0)]
    aggregate = stratified_mean(scores)
    assert aggregate.interval is not None
    assert aggregate.interval.low <= aggregate.value <= aggregate.interval.high
    assert aggregate.interval.low >= 0.0 and aggregate.interval.high <= 1.0


# --- the summary the report card prints --------------------------------------


def test_the_summary_counts_the_things_a_leaderboard_hides():
    scores = [
        make("gqa-0001", score=1.0),
        make("gqa-0002", outcome="error", score=0.0),
        make("gqa-0003", outcome="over-budget", score=0.0),
        make("gqa-0004", repeats=5, passes=3, score=1.0),
        make("gqa-0005", judge_scored=True, judge_trusted=False),
    ]
    summary = summarise(scores)
    assert summary["tasks"] == 5
    assert summary["error_share"] == pytest.approx(0.2)
    assert summary["over_budget_share"] == pytest.approx(0.2)
    assert summary["flaky_share"] == pytest.approx(0.2)
    assert summary["judge_share"] == pytest.approx(0.2)
    assert summary["untrusted_judge_share"] == pytest.approx(0.2)


def test_the_summary_of_nothing_is_empty_rather_than_zeroes():
    """A report card for a run that scored nothing should say nothing, not print six
    zeroes that read like a result."""
    assert summarise([]) == {}


def test_costs_stay_decimals_end_to_end():
    """The cost column is money. Float arithmetic across three hundred tasks produces
    cents that do not add up, and a benchmark that cannot add up is not worth the
    name."""
    score = TaskScore("gqa-0001", "grounded-qa", "pass", 1.0, cost_usd=Decimal("0.012345"))
    assert isinstance(score.cost_usd, Decimal)
    assert score.cost_usd == Decimal("0.012345")
