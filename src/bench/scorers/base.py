"""Scoring: an expectation in, a verdict and a sentence out.

The shape of the whole thing, in one paragraph. A task has expectations. A target
produced `repeats` traces. Each expectation is checked against each trace, and each
check returns an `ExpectationResult` — whether it passed, how much it weighs, and one
sentence a person can read. A repeat passes when every expectation passes. A task's
`score` is the mean partial credit across repeats, its `passes` is how many repeats
passed, and `TaskScore.flaky` falls out of those two numbers without anybody computing
it.

Three decisions worth defending.

**Weights, not a threshold.** An expectation carries `weight` (default 1). Partial credit
is the share of weights that passed, which means a schema task with nine fields right
scores 0.9 while still failing as a whole. The alternative — all-or-nothing per field —
makes a 200-field extraction task a coin flip, and the alternative to *that* — a
threshold like "80% passes" — invents a number nobody can defend.

**A judged expectation that cannot be judged is excluded, not failed.** No judge
configured, or a κ below the floor: the expectation is dropped from the weighted sum and
the task is marked `judge_trusted=False`, which the aggregate then skips. Scoring an
unjudged task zero would punish an entry for the harness's own gap, and the report says
which expectation went unchecked.

**Every result carries a sentence.** `trajectory: missing required tool search_docs`
beats `expectation 2 failed` by a wide margin, and the sentence is what ends up in the
failure list on the report card — the part of a benchmark people actually read.

The dispatcher is a dict from expectation type to function. Adding a family means adding
entries to it and nothing else: no subclassing, no registry decorators, and the
`EXPECTATION_TYPES` table in `scorers/__init__.py` already tells the task linter which
families may use which type.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..core.corpus import Corpus
from ..core.score import TaskScore
from ..core.task import Expectation, Task
from ..core.trace import Trace

Check = Callable[["ScoringContext", Expectation], "ExpectationResult"]
"""`(context, expectation) -> result`. The context carries everything a check may need —
the task, the trace, the corpus, the judge — so a check function is a pure function of
things it was handed and can be tested with two lines of setup."""


@dataclass(frozen=True)
class ExpectationResult:
    """One expectation, one trace, one verdict."""

    type: str
    passed: bool
    note: str
    weight: float = 1.0
    scored: bool = True
    """`False` means "this check could not be made", which is different from "it failed".
    An unscored expectation leaves the weighted sum rather than contributing a zero, and
    the caller marks the task as not fully scored."""

    detail: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        mark = "✓" if self.passed else ("?" if not self.scored else "✗")
        return f"{mark} {self.type}: {self.note}"


class Judge(Protocol):
    """What a scorer needs from a judge, and nothing more.

    A protocol rather than the concrete client so that the scorer tests can hand it a
    dictionary of verdicts: a test that reaches for an HTTP server to check a weighted
    sum is a test that fails for the wrong reason.
    """

    def verdict(
        self,
        *,
        rubric: str,
        prompt: str,
        output: str,
        task_id: str,
        scale: int | None = None,
    ) -> Any: ...

    """One verdict. `scale` asks for a graded reply on a 1..n rubric; a judge that does
    not grade is free to ignore it, which is why it defaults to `None` rather than
    being a separate method."""

    @property
    def trusted(self) -> bool: ...


@dataclass
class ScoringContext:
    """Everything a check may look at for one trace of one task."""

    task: Task
    trace: Trace
    corpus: Corpus | None = None
    judge: Judge | None = None
    repeat: int = 0

    @property
    def answer(self) -> str:
        """The answer text, or the empty string.

        `Trace.output` is `None` for a target that only made tool calls, and every check
        would otherwise need the same three-line guard. An empty string is the right
        stand-in: `contains` fails against it, `not_contains` passes, and neither raises.
        """
        return self.trace.output or ""


def evaluate(
    context: ScoringContext, expectation: Expectation, registry: dict[str, Check]
) -> ExpectationResult:
    """Run one expectation, turning a scorer's crash into a failed check.

    A check that raises is a bug in this repository, and it must not take down a
    three-hundred-task run: the task scores zero with a note that says the scorer broke,
    which is a bad number with a loud explanation rather than a traceback at task 87.
    """
    check = registry.get(expectation.type)
    if check is None:
        return ExpectationResult(
            expectation.type,
            passed=False,
            note=f"no scorer is registered for {expectation.type!r}",
            scored=False,
        )
    try:
        return check(context, expectation)
    except Exception as exc:  # noqa: BLE001 - a broken scorer is a failed check
        return ExpectationResult(
            expectation.type,
            passed=False,
            note=f"the scorer for {expectation.type!r} raised {type(exc).__name__}: {exc}",
        )


def weight_of(expectation: Expectation) -> float:
    """The expectation's weight, defaulting to 1.

    Read from `params` rather than declared as a field on the model, because the
    expectation model is deliberately open (`extra="allow"`) and a scorer-specific field
    on it would be the first of many. A weight that is not a positive number is a
    configuration mistake, so it is refused rather than coerced.
    """
    raw = expectation.params.get("weight", 1.0)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"weight {raw!r} is not a number") from None
    if value <= 0:
        raise ValueError(f"weight must be positive, got {value}")
    return value


def score_traces(
    task: Task,
    traces: Sequence[Trace],
    *,
    registry: dict[str, Check],
    corpora: dict[str, Corpus] | None = None,
    judge: Judge | None = None,
) -> TaskScore:
    """The whole scoring of one task: every expectation against every repeat.

    Returns a `TaskScore` whose `notes` are the failures, deduplicated across repeats
    with a count — `2/3 repeats: missing required tool search_docs` — because a failure
    that happened in every repeat and one that happened once are different news and the
    failure list is the only place that distinction survives.
    """
    if not traces:
        return TaskScore(
            task_id=task.id,
            family=task.family,
            outcome="error",
            score=0.0,
            repeats=0,
            split=task.split,
            error="no traces: the task never ran",
        )

    corpus = _corpus_for(task, corpora)
    per_repeat: list[float] = []
    passed_repeats = 0
    failures: Counter[str] = Counter()
    unscored: Counter[str] = Counter()
    errored = 0
    scored_anything = False
    any_judged = False
    any_unjudged = False

    for index, trace in enumerate(traces):
        context = ScoringContext(task=task, trace=trace, corpus=corpus, judge=judge, repeat=index)
        results = [evaluate(context, item, registry) for item in task.expectations]
        for result, expectation in zip(results, task.expectations, strict=True):
            result = _with_weight(result, expectation)
            if not result.scored:
                if expectation.type == "judge" or not _is_deterministic(expectation.type):
                    any_judged, any_unjudged = True, True
                # Kept, not dropped: an expectation that could not be checked is the
                # reason a task's score is missing a term, and a failure list that only
                # shows failed checks makes an incomplete harness look like a bad agent.
                unscored[result.note] += 1
                continue
            scored_anything = True
            if _is_judged(expectation.type):
                any_judged = True
                if judge is None or not getattr(judge, "trusted", False):
                    any_unjudged = True
            if not result.passed:
                failures[result.note] += 1

        earned = 0.0
        possible = 0.0
        for result, expectation in zip(results, task.expectations, strict=True):
            result = _with_weight(result, expectation)
            if not result.scored:
                continue
            if _is_judged(expectation.type) and (
                judge is None or not getattr(judge, "trusted", False)
            ):
                # Unscored on purpose: a judge that is not on file cannot decide a score,
                # and the task is marked so the aggregate skips it.
                continue
            possible += result.weight
            earned += result.weight if result.passed else 0.0

        if trace.failed:
            errored += 1
        if possible == 0.0:
            # Every expectation on this repeat was unscored. Contributing 0 to the mean
            # here would turn "we could not check" into "it failed".
            continue
        repeat_score = earned / possible
        per_repeat.append(repeat_score)
        if repeat_score >= 1.0:
            passed_repeats += 1

    if not per_repeat:
        # Nothing produced a verdict on any repeat. That is an *error*, not a failure:
        # the task was not measured, and a zero here would be a number the harness
        # cannot justify. The reason the measurement did not happen goes in `notes` —
        # "nothing could be scored" on its own is the least debuggable message in this
        # repository, and it is exactly the message a reader would have to open a
        # debugger to act on.
        reasons = []
        if errored and errored == len(traces):
            reasons.append(f"every repeat errored: {traces[0].error}")
        reasons.extend(
            f"{count}/{len(traces)} repeats: {note}" for note, count in failures.most_common()
        )
        reasons.extend(
            f"{count}/{len(traces)} repeats not scored: {note}"
            for note, count in unscored.most_common()
        )
        reasons.append("no expectation produced a verdict, so nothing was scored")
        if any_unjudged:
            reasons.append(
                "judge-scored expectations were not counted: "
                + (
                    "no judge is configured"
                    if judge is None
                    else "the judge is below its calibration floor"
                )
            )
        return TaskScore(
            task_id=task.id,
            family=task.family,
            outcome="error",
            score=0.0,
            repeats=len(traces),
            passes=0,
            split=task.split,
            error=traces[0].error if errored == len(traces) else reasons[0],
            judge_scored=any_judged,
            judge_trusted=not any_unjudged,
            notes=reasons,
        )

    score = sum(per_repeat) / len(per_repeat)
    if errored == len(traces):
        outcome = "error"
    elif passed_repeats == len(per_repeat):
        outcome = "pass"
    else:
        outcome = "fail"

    notes = [f"{count}/{len(traces)} repeats: {note}" for note, count in failures.most_common()]
    if errored:
        # A repeat that errored produced no answer, so it scores nothing and the task
        # fails — an entry that cannot answer one time in three has not answered. The
        # count goes above the failure list because a reader deciding whether to trust an
        # entry needs the flakiness before the arithmetic, and because "the target
        # returned a 429" and "the target said the wrong thing" are different verdicts on
        # the same row.
        if errored == len(traces):
            notes.insert(0, f"every repeat errored: {trace_error(traces)}")
        else:
            notes.insert(0, f"{errored}/{len(traces)} repeats errored: {trace_error(traces)}")
    notes.extend(
        f"{count}/{len(traces)} repeats not scored: {note}"
        for note, count in unscored.most_common()
        if not note.startswith("not scored:")  # the judge notes are added below, once
    )
    if not scored_anything:
        notes.append("no expectation produced a verdict")
    if any_unjudged:
        notes.append(
            "judge-scored expectations were not counted: "
            + (
                "no judge is configured"
                if judge is None
                else "the judge is below its calibration floor"
            )
        )

    return TaskScore(
        task_id=task.id,
        family=task.family,
        outcome=outcome,
        score=score,
        repeats=len(traces),
        passes=passed_repeats,
        split=task.split,
        error=traces[0].error if outcome == "error" else None,
        latency_ms=max((trace.latency_ms for trace in traces), default=0),
        judge_scored=any_judged,
        judge_trusted=not any_unjudged,
        notes=notes,
    )


def trace_error(traces: Sequence[Trace]) -> str:
    """The first error among the traces, for the one-line note."""
    for trace in traces:
        if trace.failed:
            return str(trace.error)
    return "unknown error"


def _with_weight(result: ExpectationResult, expectation: Expectation) -> ExpectationResult:
    """Fold the expectation's declared weight into the result, once.

    Kept out of every check function so that no scorer has to remember that weights
    exist — and so that a check cannot quietly decide its own weight, which is how a
    benchmark ends up with one family outweighing another by accident.
    """
    if result.weight != 1.0:
        return result
    try:
        weight = weight_of(expectation)
    except ValueError as exc:
        return ExpectationResult(result.type, False, str(exc), scored=False)
    if weight == 1.0:
        return result
    return ExpectationResult(
        result.type, result.passed, result.note, weight, result.scored, result.detail
    )


def _corpus_for(task: Task, corpora: dict[str, Corpus] | None) -> Corpus | None:
    if not task.corpus or not corpora:
        return None
    return corpora.get(task.corpus)


def _is_deterministic(expectation_type: str) -> bool:
    from . import EXPECTATION_TYPES

    known = EXPECTATION_TYPES.get(expectation_type)
    return bool(known and known.deterministic)


def _is_judged(expectation_type: str) -> bool:
    return not _is_deterministic(expectation_type)


# --- the scorer a runner gets -------------------------------------------------


def build_scorer(
    *,
    root: Path | None = None,
    corpus_paths: dict[str, Path] | None = None,
    judge: Judge | None = None,
) -> Callable[[Task, Sequence[Trace]], TaskScore]:
    """The `(task, traces) -> TaskScore` callable `Runner` takes.

    Corpora are loaded once, lazily, and cached by name: a 300-task set with 12 corpus
    directories should read each of them once, and a corpus that fails to load should
    fail when a task using it is scored rather than when the scorer is built — otherwise
    one bad directory stops a run that mostly does not need it.
    """
    registry = default_registry()
    cache: dict[str, Corpus] = {}
    failures: dict[str, str] = {}

    def scorer(task: Task, traces: Sequence[Trace]) -> TaskScore:
        if task.corpus and task.corpus not in cache:
            location = (corpus_paths or {}).get(task.corpus)
            if location is None and root is not None:
                location = root / task.corpus
            if location is not None and task.corpus not in failures:
                try:
                    cache[task.corpus] = Corpus.load(location)
                except Exception as exc:  # noqa: BLE001 - reported as a note, not a crash
                    failures[task.corpus] = f"{type(exc).__name__}: {exc}"
        return score_traces(task, traces, registry=registry, corpora=cache, judge=judge)

    return scorer


def default_registry() -> dict[str, Check]:
    """Every check this package implements, keyed by the expectation type.

    Imported here rather than at module level because the family modules import this one
    for `ScoringContext`: a cycle that shows up as an import error for a new user and
    never for anybody who has been running the tests all along.
    """
    from .citations import CHECKS as citation_checks
    from .deterministic import CHECKS as deterministic_checks
    from .extraction import CHECKS as extraction_checks
    from .judged import CHECKS as judged_checks
    from .safety import CHECKS as safety_checks
    from .trajectory import CHECKS as trajectory_checks

    registry: dict[str, Check] = {}
    # Six modules, not one per family: `state` lives with the trajectory checks because
    # it reads the same thing they do, and `refusal` gets its own because the family that
    # requires a refusal and the family that spots one are not the same reader.
    for group in (
        deterministic_checks,
        trajectory_checks,
        citation_checks,
        extraction_checks,
        safety_checks,
        judged_checks,
    ):
        for name, check in group.items():
            if name in registry:  # pragma: no cover - a name collision is a coding error
                raise RuntimeError(f"two scorers are registered for {name!r}")
            registry[name] = check
    return registry
