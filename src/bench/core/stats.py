"""Statistics for a leaderboard that has to defend its own ordering.

Three things happen here, and none of them is a t-test.

**The bootstrap is over tasks, not over repeats.** Repeats measure how flaky the
agent is; tasks measure how uncertain the score is. Resampling repeats would
produce a tight interval around a wrong number, which is the most confident way to
be wrong, so it is not offered.

**The stratified resample is the aggregate's interval.** Families are resampled
inside themselves and the family means are averaged, which is the same shape as the
aggregate itself. A family with eighty tasks therefore cannot outvote a family with
twelve — not in the headline, and not in the interval.

**The paired resample is what decides a rank.** Two entries are compared on the
tasks both of them ran, by resampling the *differences* — and a difference whose
interval includes zero is a tie, printed as a tie, sorted as a tie. Ranking by
overlapping intervals is how leaderboards end up claiming four distinct places for
four entries that are indistinguishable.

Seeds are fixed (`DEFAULT_SEED`), because a number that changes when you re-run the
report is not a number.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to the type checker
    from .score import TaskScore

DEFAULT_SEED = 20261001
"""Arbitrary, and never changed. Chosen on the day the board was first built and
left alone afterwards: changing a seed changes published numbers, so it is a
versioned constant, not a knob."""

DEFAULT_RESAMPLES = 10_000


@dataclass(frozen=True)
class Interval:
    low: float
    high: float
    level: float = 0.95

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(f"interval is inverted: [{self.low}, {self.high}]")
        if not 0.0 < self.level < 1.0:
            raise ValueError(f"confidence level must be in (0, 1), got {self.level}")

    def __contains__(self, value: float) -> bool:
        return self.low <= value <= self.high

    def overlaps(self, other: Interval) -> bool:
        """Whether two intervals share any point. Open at both ends on purpose: two
        intervals that touch at exactly one value are still two claims that cannot
        be told apart from the data."""
        return self.low <= other.high and other.low <= self.high

    def __str__(self) -> str:
        level = int(round(self.level * 100))
        return f"[{self.low:+.3f}, {self.high:+.3f}] ({level}%)"

    def as_percent(self) -> str:
        return f"[{self.low:.1%}, {self.high:.1%}]"


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean of nothing is not zero, it is undefined")
    return sum(values) / len(values)


def percentile(values: Sequence[float], fraction: float) -> float:
    """Linear interpolation between the two nearest order statistics.

    Not `statistics.quantiles(n=100)` and not the nearest-rank rule: with a few
    hundred tasks the difference between the conventions is larger than the effect
    being measured, and picking one silently is how two implementations of the same
    benchmark disagree. This one is stated here and covered by a test with a known
    answer.
    """
    if not values:
        raise ValueError("percentile of nothing")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_interval(
    values: Sequence[float],
    *,
    level: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Interval:
    """Percentile bootstrap over a single sample."""
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    if len(values) == 1:
        # One task has no spread to estimate, and inventing one would be a lie with
        # a decimal point. The interval is the point itself; callers that care about
        # that print a warning about sample size.
        return Interval(values[0], values[0], level)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(resamples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    tail = (1.0 - level) / 2.0
    return Interval(percentile(means, tail), percentile(means, 1.0 - tail), level)


def stratified_bootstrap_interval(
    by_family: Mapping[str, Sequence[float]],
    *,
    level: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Interval:
    """The aggregate's interval: resample inside each family, then average the
    family means, which is exactly how the headline number is computed."""
    groups = {name: list(values) for name, values in by_family.items() if values}
    if not groups:
        raise ValueError("cannot bootstrap an empty league")
    if len(groups) == 1:
        only = next(iter(groups.values()))
        return bootstrap_interval(only, level=level, resamples=resamples, seed=seed)
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        family_means = []
        for values in groups.values():
            n = len(values)
            family_means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
        draws.append(sum(family_means) / len(family_means))
    tail = (1.0 - level) / 2.0
    return Interval(percentile(draws, tail), percentile(draws, 1.0 - tail), level)


def paired_bootstrap_interval(
    differences: Sequence[float],
    *,
    level: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Interval:
    """The interval on the *mean difference* between two entries on shared tasks.

    A one-sided zero here is what a rank claim means, and there is no other way to
    earn it in this codebase.
    """
    return bootstrap_interval(differences, level=level, resamples=resamples, seed=seed)


def paired_differences(
    left: Iterable[TaskScore], right: Iterable[TaskScore]
) -> tuple[list[float], list[str]]:
    """Per-task differences on the tasks both runs scored, plus the ids involved.

    Tasks present in one run and not the other are dropped, not treated as zero: an
    entry that skipped the hard tasks has not tied with one that attempted them.
    The count of dropped tasks is reported by the caller, and the board refuses a
    comparison that drops more than a small share of either side.
    """
    left_by_id = {item.task_id: item for item in left}
    right_by_id = {item.task_id: item for item in right}
    shared = sorted(set(left_by_id) & set(right_by_id))
    return [left_by_id[key].score - right_by_id[key].score for key in shared], shared


def ngram_overlap(left: str, right: str, *, n: int = 8) -> float:
    """Share of `left`'s character n-grams that occur in `right`.

    Used for the contamination column: a task whose text or expected answer shows up
    in an entrant's declared training data is worth flagging even if nothing can be
    proved. Character n-grams rather than words because the interesting leaks are
    near-verbatim, and because whitespace normalisation is a third of the work of a
    word-based version for none of the benefit.

    `n=8` is a judgement: shorter flags ordinary English, longer misses a sentence
    with one word changed.
    """
    if len(left) < n:
        return 0.0
    grams = {left[index : index + n] for index in range(len(left) - n + 1)}
    if not grams:
        return 0.0
    seen = {right[index : index + n] for index in range(len(right) - n + 1)}
    return len(grams & seen) / len(grams)
