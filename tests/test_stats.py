"""The arithmetic the board defends its ordering with.

Every test here has a hand-computed answer or a property that would be wrong to
break. "The interval is roughly right" is not a test; it is a hope with a function
call in it.
"""

from __future__ import annotations

import pytest

from bench.core.score import TaskScore
from bench.core.stats import (
    DEFAULT_SEED,
    Interval,
    bootstrap_interval,
    mean,
    ngram_overlap,
    paired_bootstrap_interval,
    paired_differences,
    percentile,
)

# --- the basics -------------------------------------------------------------


def test_mean_refuses_the_empty_sample():
    """Zero would be the convenient answer and a lie: an empty set has no average."""
    with pytest.raises(ValueError, match="undefined"):
        mean([])


def test_percentile_interpolates_between_order_statistics():
    assert percentile([0.0, 1.0, 2.0, 3.0], 0.5) == pytest.approx(1.5)
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.25) == pytest.approx(1.75)
    assert percentile([5.0], 0.9) == 5.0
    assert percentile([0.0, 10.0], 0.0) == 0.0
    assert percentile([0.0, 10.0], 1.0) == 10.0


def test_percentile_rejects_a_fraction_outside_the_range():
    with pytest.raises(ValueError, match="fraction"):
        percentile([1.0, 2.0], 1.5)


def test_percentile_of_nothing_is_refused():
    with pytest.raises(ValueError, match="nothing"):
        percentile([], 0.5)


# --- intervals --------------------------------------------------------------


def test_an_inverted_interval_is_refused_at_construction():
    """The failure this prevents is unglamorous and expensive: a reversed interval
    silently contains nothing, so every comparison against it reads as a tie."""
    with pytest.raises(ValueError, match="inverted"):
        Interval(0.8, 0.2)


def test_the_confidence_level_has_to_be_a_probability():
    with pytest.raises(ValueError, match="confidence"):
        Interval(0.1, 0.2, 1.0)


def test_overlap_is_open_at_both_ends():
    """Two intervals that touch at a single point are two claims the data cannot
    separate. Treating that touch as a separation is exactly the mistake the board
    exists to avoid."""
    assert Interval(0.1, 0.5).overlaps(Interval(0.5, 0.9))
    assert Interval(0.1, 0.6).overlaps(Interval(0.5, 0.9))
    assert not Interval(0.1, 0.4).overlaps(Interval(0.5, 0.9))


def test_contains_uses_the_closed_interval():
    band = Interval(0.2, 0.4)
    assert 0.2 in band
    assert 0.4 in band
    assert 0.15 not in band


def test_an_interval_prints_as_a_signed_change_and_as_a_percent():
    assert str(Interval(-0.533, 0.0)) == "[-0.533, +0.000] (95%)"
    assert Interval(0.621, 0.963).as_percent() == "[62.1%, 96.3%]"


# --- bootstrap --------------------------------------------------------------


def test_the_bootstrap_is_seeded_so_two_reports_agree():
    values = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    first = bootstrap_interval(values)
    second = bootstrap_interval(values)
    assert first == second
    assert DEFAULT_SEED == 20261001


def test_a_different_seed_moves_the_edges_but_not_the_centre():
    """Worth pinning down, because it is why the seed is a constant and not a knob:
    the point estimate is a fact about the data, the interval is an estimate of the
    spread, and only the second one moves with the seed."""
    values = [
        0.13, 0.87, 0.42, 0.61, 0.29, 0.95, 0.08, 0.55, 0.73, 0.34,
        0.66, 0.21, 0.49, 0.81, 0.37, 0.58, 0.12, 0.90, 0.44, 0.68,  # fmt: skip
    ]
    a = bootstrap_interval(values, seed=1)
    b = bootstrap_interval(values, seed=2)
    assert (a.low, a.high) != (b.low, b.high)
    assert a.low == pytest.approx(b.low, abs=0.01)  # it moves, it does not wander
    assert mean(values) in a
    assert mean(values) in b


def test_a_coarse_sample_gives_a_stable_interval_whatever_the_seed():
    """Found by writing the test above with pass/fail values, and worth keeping.

    With ten binary tasks the bootstrap can only land on a handful of distinct
    resample means, so the 2.5th and 97.5th percentiles are the same whatever the
    seed does. The interval is coarse, not unstable — and the resolution of the
    interval is a property of the dataset, which is why the report prints the task
    count beside every score."""
    values = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    bands = {bootstrap_interval(values, seed=seed) for seed in range(5)}
    assert len(bands) == 1


def test_one_task_gets_a_degenerate_interval_and_not_an_invented_one():
    band = bootstrap_interval([1.0])
    assert band.low == band.high == 1.0


def test_bootstrapping_nothing_is_refused():
    with pytest.raises(ValueError, match="empty sample"):
        bootstrap_interval([])


def test_all_passes_gives_an_interval_pinned_at_one():
    band = bootstrap_interval([1.0] * 12)
    assert band.low == band.high == 1.0


def test_a_mixed_sample_brackets_its_own_mean():
    values = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    band = bootstrap_interval(values)
    assert band.low < mean(values) < band.high
    assert band.low >= 0.0 and band.high <= 1.0


# --- the paired test, which is what decides a rank --------------------------


def test_paired_differences_only_uses_tasks_both_entries_ran():
    """An entry that skipped the hard tasks has not tied with one that attempted
    them, so unshared tasks are dropped — and the ids come back so the caller can
    report how many were dropped and refuse a comparison that dropped too many."""
    left = [
        TaskScore("gqa-0001", "grounded-qa", "pass", 1.0),
        TaskScore("gqa-0002", "grounded-qa", "pass", 1.0),
        TaskScore("gqa-0003", "grounded-qa", "fail", 0.0),
    ]
    right = [
        TaskScore("gqa-0001", "grounded-qa", "fail", 0.0),
        TaskScore("gqa-0002", "grounded-qa", "pass", 1.0),
        TaskScore("gqa-9999", "grounded-qa", "pass", 1.0),
    ]
    differences, shared = paired_differences(left, right)
    assert shared == ["gqa-0001", "gqa-0002"]
    assert differences == [1.0, 0.0]


def test_a_clear_paired_win_excludes_zero():
    """Eighteen tasks won, two lost, out of twenty. Resampling those differences
    puts the 2.5th percentile at +0.5, so the entry above really is above."""
    differences = [1.0] * 18 + [-1.0] * 2
    band = paired_bootstrap_interval(differences)
    assert 0.0 not in band
    assert band.low > 0.4


def test_six_wins_out_of_eight_is_not_a_win():
    """The number that made this project worth writing. Six to two looks decisive
    on a leaderboard and says almost nothing: resampling eight tasks where the
    split is 75/25 puts the lower edge at -0.25, so the honest output is a tie, and
    the honest thing for the board to do is print one."""
    band = paired_bootstrap_interval([1.0] * 6 + [-1.0] * 2)
    assert 0.0 in band
    assert band.low < 0.0 < band.high


def test_a_mixed_paired_difference_includes_zero_and_is_a_tie():
    differences = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    band = paired_bootstrap_interval(differences)
    assert 0.0 in band


def test_an_identical_pair_is_tied_by_construction():
    differences = [0.0] * 20
    band = paired_bootstrap_interval(differences)
    assert band.low == band.high == 0.0
    assert 0.0 in band


# --- contamination, which is a statistic too ---------------------------------


def test_ngram_overlap_finds_a_near_verbatim_leak():
    task_text = "the v2 rollout stopped refunding orders because a permission changed"
    leaked = "note: the v2 rollout stopped refunding orders because a permission changed twice"
    assert ngram_overlap(task_text, leaked) > 0.9


def test_ngram_overlap_is_zero_for_unrelated_text():
    assert ngram_overlap("a" * 40, "b" * 40) == 0.0


def test_ngram_overlap_is_directional_and_that_is_the_point():
    """`ngram_overlap(a, b)` answers "how much of a is in b". When one text is the
    other plus a clause, the longer one measures 0.84 and the shorter 1.0 — which is
    why the duplicate finder takes the max of both directions rather than trusting
    whichever order its loops happen to produce. Found by a test that expected a
    near-duplicate to be caught and watched it not be."""
    base = "why did the v2 rollout stop refunding orders after the deploy"
    longer = base + " on tuesday afternoon"
    assert ngram_overlap(base, longer) == 1.0
    assert ngram_overlap(longer, base) < 0.9
    assert max(ngram_overlap(base, longer), ngram_overlap(longer, base)) == 1.0


def test_short_text_cannot_be_compared_and_says_zero():
    """Under `n` characters there are no n-grams to compare. Zero rather than a
    guess, and the caller reports the task as too short to check."""
    assert ngram_overlap("short", "short", n=8) == 0.0
