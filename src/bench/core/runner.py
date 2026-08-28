"""The runner: tasks in, traces out, with repeats and budgets.

Four decisions, all of them consequences of the same rule from `targets/base.py` — a
failure is a value.

**A task runs `repeats` times, and the spread is kept.** Repeats are not averaging;
they are the measurement of flakiness, which the board publishes as its own column. A
task that split its repeats is a task the entry cannot be trusted on, and that is
hidden the moment somebody writes `passes / repeats` and moves on.

**The budget stops a task, not the run.** A per-task budget exceeded produces an
`over-budget` result, and the next task starts fresh. A per-*run* budget stops
everything and marks the remainder `skipped` — and the run file records that it was
partial, because a partial run that reads like a full one is the single most dangerous
artefact this project can produce.

**Concurrency is a semaphore, not a thread pool.** The work is I/O-bound and the
targets are async; a pool would add threads that exist only to call `run_until_complete`.

**Order is preserved in the result.** Tasks finish in whatever order the network
decides, and the run file lists them in the order the task set defines them, because a
diff of two run files that shuffles is a diff nobody reads.

The runner knows nothing about scoring. It produces `TaskRun`s and hands them to a
`scorer` callable if one is given; `scorers/` (next step) provides the real ones and the
tests provide a fake. That seam is deliberate: the runner's contract is "did the target
survive this task, and what did it cost", and scoring is a different question with its
own file.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal

from ..targets.base import Target
from .budget import BudgetMeter
from .score import TaskScore
from .task import Task
from .trace import Trace

RunOutcome = Literal["completed", "error", "over-budget", "skipped"]
"""What happened to one task in one run.

`completed` is not a pass — whether the answers were right is the scorer's business.
`error` means every repeat failed to produce a trace at all. Keeping the two apart is
what lets the report say "this entry is broken" and "this entry is wrong" as different
sentences.
"""

Scorer = Callable[[Task, Sequence[Trace]], TaskScore]
"""`(task, traces) -> TaskScore`. Called once per task, after its repeats are in."""


@dataclass
class TaskRun:
    """One task's execution: the traces, the outcome, and what it cost."""

    task_id: str
    family: str
    split: str
    outcome: RunOutcome
    traces: list[Trace] = field(default_factory=list)
    repeats: int = 1
    error: str | None = None
    """Set when the failure was the harness's, not the target's: a missing cassette
    entry, a budget stop, a task that was never attempted."""

    spent_usd: Decimal | None = None
    wall_seconds: float = 0.0

    @property
    def failed_repeats(self) -> int:
        return sum(1 for trace in self.traces if trace.failed)

    @property
    def latencies_ms(self) -> list[int]:
        return [trace.latency_ms for trace in self.traces]

    def cost(self) -> Decimal | None:
        """Sum of what the traces reported, or `None` if nothing was priced.

        `None` rather than zero when every trace is unpriced: the difference between
        "free" and "not measured" decides whether an entry belongs in a cost
        comparison at all.
        """
        priced = [trace.cost_usd for trace in self.traces if trace.cost_usd is not None]
        if not priced:
            return None
        return sum(priced, Decimal("0")).quantize(Decimal("0.000001"))

    def summarise(self) -> str:
        if self.outcome == "skipped":
            return f"skipped ({self.error or 'run budget exhausted'})"
        if self.outcome == "error":
            return f"error — {self.error or 'every repeat failed'}"
        if self.outcome == "over-budget":
            return f"over budget — {self.error}"
        return f"{self.repeats} repeat(s) in {self.wall_seconds:.2f}s"


@dataclass
class RunResult:
    """Everything one run produced, in dataset order."""

    label: str
    tasks: list[TaskRun]
    started_at: str
    wall_seconds: float
    repeats: int
    concurrency: int
    partial: bool = False
    partial_reason: str | None = None
    scores: list[TaskScore] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def completed(self) -> list[TaskRun]:
        return [item for item in self.tasks if item.outcome == "completed"]

    @property
    def errored(self) -> list[TaskRun]:
        return [item for item in self.tasks if item.outcome == "error"]

    @property
    def skipped(self) -> list[TaskRun]:
        return [item for item in self.tasks if item.outcome == "skipped"]

    @property
    def over_budget(self) -> list[TaskRun]:
        return [item for item in self.tasks if item.outcome == "over-budget"]

    def cost(self) -> Decimal | None:
        priced = [item.cost() for item in self.tasks if item.cost() is not None]
        if not priced:
            return None
        return sum((value for value in priced if value is not None), Decimal("0"))

    def flaky_task_ids(self) -> list[str]:
        """Tasks whose repeats did not agree. Read from the scores when scoring ran,
        and from the traces' error pattern when it did not — the second is a weaker
        signal (a task whose repeats all errored is not flaky, it is broken) and the
        report says which one it used."""
        if self.scores:
            return [score.task_id for score in self.scores if score.flaky]
        return [item.task_id for item in self.tasks if 0 < item.failed_repeats < len(item.traces)]

    def summary(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "tasks": len(self.tasks),
            "completed": len(self.completed),
            "errored": len(self.errored),
            "over_budget": len(self.over_budget),
            "skipped": len(self.skipped),
            "repeats": self.repeats,
            "wall_seconds": round(self.wall_seconds, 3),
            "partial": self.partial,
            "cost_usd": str(self.cost()) if self.cost() is not None else None,
        }


class Runner:
    """Runs a task set against one target.

    Constructed with everything it needs and nothing it might mutate: the target, the
    tasks, the repeat count, the concurrency, the budgets. `run()` is the only method
    that does anything.
    """

    def __init__(
        self,
        target: Target,
        *,
        repeats: int = 1,
        concurrency: int = 8,
        scorer: Scorer | None = None,
        run_budget_usd: Decimal | None = None,
        run_budget_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        on_task_done: Callable[[TaskRun], None] | None = None,
    ) -> None:
        if repeats < 1:
            raise ValueError("repeats must be at least 1")
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self.target = target
        self.repeats = repeats
        self.concurrency = concurrency
        self.scorer = scorer
        self.run_budget_usd = run_budget_usd
        self.run_budget_seconds = run_budget_seconds
        self.clock = clock
        self.on_task_done = on_task_done
        self._meter = BudgetMeter(max_usd=run_budget_usd, max_seconds=run_budget_seconds)
        self._stopped: str | None = None

    # --- one task ----------------------------------------------------------

    async def run_task(self, task: Task) -> TaskRun:
        """All the repeats of one task, inside the task's own budget."""
        started = self.clock()
        note: str | None = None
        if self._stopped is not None:
            return TaskRun(
                task.id,
                task.family,
                task.split,
                "skipped",
                error=f"run budget exhausted: {self._stopped}",
                repeats=self.repeats,
            )

        meter = BudgetMeter(
            max_usd=Decimal(str(task.budget.max_usd)) if task.budget.max_usd else None,
            max_seconds=task.budget.max_seconds,
            clock=self.clock,
        )
        meter.start()
        traces: list[Trace] = []
        for repeat in range(self.repeats):
            trace = await self.target.run(task, repeat=repeat)
            traces.append(trace)
            if trace.cost_usd is not None:
                meter.spend(usd=trace.cost_usd)
            meter.spend(seconds=0.0)
            self._meter.spend(usd=trace.cost_usd or Decimal("0"))
            stop = meter.over_budget()
            if stop is not None:
                note = stop
                # The remaining repeats are not attempted and are not silently dropped:
                # the count in the run file is the count that ran.
                break

        wall = self.clock() - started
        self._meter.spend(seconds=wall)
        outcome: RunOutcome = "completed"
        error: str | None = note
        if note is not None:
            outcome = "over-budget"
        elif traces and all(trace.failed for trace in traces):
            outcome = "error"
            error = traces[0].error or "every repeat failed"

        task_run = TaskRun(
            task_id=task.id,
            family=task.family,
            split=task.split,
            outcome=outcome,
            traces=traces,
            repeats=len(traces),
            error=error,
            spent_usd=meter.spent_usd if not task.budget.is_empty else None,
            wall_seconds=wall,
        )
        self._check_run_budget()
        if self.on_task_done is not None:
            self.on_task_done(task_run)
        return task_run

    def _check_run_budget(self) -> None:
        stop = self._meter.over_budget()
        if stop is not None and self._stopped is None:
            self._stopped = stop

    # --- the whole run -----------------------------------------------------

    async def run(self, tasks: Iterable[Task], *, label: str | None = None) -> RunResult:
        """Run everything, bounded by `concurrency`, preserving the input order.

        The semaphore is created here rather than in `__init__` because an
        `asyncio.Semaphore` binds to the running loop, and a Runner constructed outside
        `asyncio.run` and used inside it is a mistake people make once.
        """
        ordered = list(tasks)
        started_at = _stamp()
        started = self.clock()
        semaphore = asyncio.Semaphore(self.concurrency)

        async def guarded(task: Task) -> TaskRun:
            async with semaphore:
                return await self.run_task(task)

        runs = await asyncio.gather(*(guarded(task) for task in ordered))
        target_label = getattr(self.target, "label", None)
        result = RunResult(
            label=label or (target_label if isinstance(target_label, str) else "target"),
            tasks=list(runs),
            started_at=started_at,
            wall_seconds=self.clock() - started,
            repeats=self.repeats,
            concurrency=self.concurrency,
            partial=any(item.outcome == "skipped" for item in runs),
            partial_reason=self._stopped,
        )
        if result.partial:
            result.notes.append(
                f"{len(result.skipped)} task(s) were never attempted: {self._stopped}. "
                f"A partial run cannot be compared with a full one."
            )
        if self.repeats == 1:
            result.notes.append("repeats=1: this run cannot measure flakiness, only the score.")
        if self.scorer is not None:
            result.scores = [
                self.scorer(task, run.traces) for task, run in zip(ordered, runs, strict=True)
            ]
        return result

    async def aclose(self) -> None:
        await self.target.aclose()


async def run_target(
    target: Target,
    tasks: Sequence[Task],
    *,
    repeats: int = 1,
    concurrency: int = 8,
    scorer: Scorer | None = None,
    label: str | None = None,
    **kwargs: Any,
) -> RunResult:
    """The one-shot form, which is what the CLI calls.

    Closes the target afterwards, including on the way out of an exception: a leaked
    HTTP client shows up as a warning about an unclosed session and a hang at exit, and
    the person who sees that is not the person who wrote this function.
    """
    runner = Runner(target, repeats=repeats, concurrency=concurrency, scorer=scorer, **kwargs)
    try:
        return await runner.run(tasks, label=label)
    finally:
        await runner.aclose()


def _stamp() -> str:
    """ISO-8601 UTC, second resolution, `T` separator.

    Not `datetime.now()`: the run files are sorted and diffed by name, and a local
    timestamp from a laptop in Tallinn and a runner in UTC produce a directory nobody
    can order.
    """
    import datetime as _datetime

    return (
        _datetime.datetime.now(_datetime.UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
