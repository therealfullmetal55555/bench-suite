"""Scores, and the one decision this project is built around.

**A score is a distribution over tasks, not a float.** Every number in this codebase
that claims to be a result is a `Score`: a value, the number of tasks it rests on,
a label, and — where one has been computed — an interval. There is no method that
returns a bare `float` for a family, an entry or the board, because the first thing
that happens to a float is that it gets sorted, and sorting is the act of claiming a
rank you cannot support.

Three consequences, all of them visible in the code below:

* `Score.__post_init__` refuses a value outside [0, 1] and an interval that does not
  contain its own value. Both are the kind of wrong number that travels.
* Aggregation is explicit about *which* mean it is. `family_mean` averages tasks;
  `stratified_mean` averages family means, which is what the board prints; the two
  are different numbers and the report shows both, because the gap between them is
  the most useful thing a reader can learn about a benchmark's task mix.
* An entry that errored on 12% of tasks scores 0 on them — an error *is* a failure —
  and the 12% is published as its own column. Hiding errors in a separate "did not
  run" bucket is how a benchmark ranks a fragile agent above a working one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from .stats import Interval, mean, stratified_bootstrap_interval

__all__ = [
    "Score",
    "TaskScore",
    "family_mean",
    "family_scores",
    "stratified_mean",
    "summarise",
]


@dataclass(frozen=True)
class Score:
    """A number that knows how thin the ice under it is."""

    value: float
    n: int
    label: str
    interval: Interval | None = None
    families: int = 0
    """How many families went into it. Zero for a task-level or family-level score,
    set for an aggregate — a headline built from two families out of six should say
    so on its own face."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"{self.label}: score {self.value} is outside [0, 1]")
        if self.n < 0:
            raise ValueError(f"{self.label}: negative task count")
        if self.interval is not None and self.value not in self.interval:
            raise ValueError(
                f"{self.label}: value {self.value:.3f} is outside its own interval "
                f"{self.interval} — one of them was computed from the wrong tasks"
            )

    def __str__(self) -> str:
        if self.interval is None:
            return f"{self.value:.3f} (n={self.n})"
        return f"{self.value:.3f} {self.interval.as_percent()} (n={self.n})"

    @property
    def tied_with(self) -> None:  # pragma: no cover - documentation only
        """Read the module docstring: ties are decided in `board.ranking`, not by a
        property on a score, because a tie is a relation between two entries and
        neither one owns it."""
        return None


@dataclass(frozen=True)
class TaskScore:
    """One task's outcome for one entry, after all repeats.

    `score` carries partial credit (a schema task with 9 of 10 fields right scores
    0.9 while still failing as a whole). `outcome` says what happened, which is what
    the report groups by. The two are not redundant: an error and an honest wrong
    answer both score 0, and only one of them means the entry is fragile.
    """

    task_id: str
    family: str
    outcome: str
    score: float
    repeats: int = 1
    passes: int = 0
    split: str = "public"
    error: str | None = None
    cost_usd: Decimal | None = None
    latency_ms: int = 0
    judge_scored: bool = False
    judge_trusted: bool = True
    """Whether the judge that decided this task cleared its calibration floor. A
    task decided by an untrusted judge is kept, marked, and excluded from the
    aggregate — never deleted, because "the judge was not good enough here" is a
    fact about the result that a reader needs."""

    notes: list[str] = field(default_factory=list)
    """Scorer explanations, one per expectation that failed. The failure list on the
    report card is built from these, and they are written to be read by a person who
    has thirty seconds."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"{self.task_id}: score {self.score} is outside [0, 1]")
        if self.passes > self.repeats:
            raise ValueError(f"{self.task_id}: {self.passes} passes out of {self.repeats} repeats")

    @property
    def failed(self) -> bool:
        return self.outcome in {"fail", "error", "over-budget"}

    @property
    def flaky(self) -> bool:
        """Split the repeats and the task is flaky, whatever the majority says.

        The share of these is a published column with an asterisk on the entry: a
        score that is the average of a coin flip is a different object from a score
        that is the same five times running.
        """
        return 0 < self.passes < self.repeats

    @property
    def counts_towards_quality(self) -> bool:
        """Everything counts except tasks the harness deliberately did not run, and
        tasks decided by a judge that failed its calibration."""
        return self.outcome != "skipped" and self.judge_trusted


def family_scores(scores: list[TaskScore], *, trusted_only: bool = True) -> dict[str, list[float]]:
    """Scores grouped by family, in the shape the bootstrap wants."""
    grouped: dict[str, list[float]] = {}
    for item in scores:
        if trusted_only and not item.counts_towards_quality:
            continue
        grouped.setdefault(item.family, []).append(item.score)
    return grouped


def family_mean(scores: list[TaskScore], family: str) -> Score:
    """The plain mean over that family's tasks. Shown per family on the report card,
    because a family is where a regression becomes a hypothesis about the world."""
    values = [
        item.score for item in scores if item.family == family and item.counts_towards_quality
    ]
    if not values:
        return Score(0.0, 0, family)
    return Score(mean(values), len(values), family)


def stratified_mean(
    scores: list[TaskScore],
    *,
    label: str = "aggregate",
    interval: bool = True,
    seed: int | None = None,
) -> Score:
    """The headline: the mean of family means, with a stratified interval.

    Equal weight per family rather than per task, so a family that happens to have
    more tasks — or a generator that emitted more of them — cannot decide the board.
    """
    grouped = family_scores(scores)
    if not grouped:
        return Score(0.0, 0, label)
    family_means = [mean(values) for values in grouped.values()]
    value = mean(family_means)
    total = sum(len(values) for values in grouped.values())
    if not interval or total < 2:
        return Score(value, total, label, families=len(grouped))
    kwargs = {} if seed is None else {"seed": seed}
    band = stratified_bootstrap_interval(grouped, **kwargs)
    # The percentile bootstrap of a mean of bounded values can land marginally
    # outside the average's own value on tiny samples; clip rather than refuse,
    # because refusing here would turn a small dataset into a crash.
    low, high = min(band.low, value), max(band.high, value)
    return Score(value, total, label, Interval(low, high, band.level), families=len(grouped))


def summarise(scores: list[TaskScore]) -> dict[str, float]:
    """The counts the report card prints beside the score. All of them are shares of
    tasks, so a reader can multiply by 300 and get a task count."""
    if not scores:
        return {}
    n = len(scores)
    errored = sum(1 for item in scores if item.outcome == "error")
    over_budget = sum(1 for item in scores if item.outcome == "over-budget")
    flaky = sum(1 for item in scores if item.flaky)
    judged = sum(1 for item in scores if item.judge_scored)
    untrusted = sum(1 for item in scores if not item.judge_trusted)
    return {
        "tasks": float(n),
        "error_share": errored / n,
        "over_budget_share": over_budget / n,
        "flaky_share": flaky / n,
        "judge_share": judged / n,
        "untrusted_judge_share": untrusted / n,
    }
