"""The runner: repeats, budgets, concurrency, and what a partial run says about itself.

Most of these tests are about the two ways a run can be partially true: a task that
spent more than it was allowed, and a run that stopped before the end. Both produce
results, both are labelled, and neither is allowed to look like a clean run — because a
partial run that reads like a full one is the most dangerous artefact this project can
produce.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path

import pytest
from conftest import task, write_set

from bench.core.runner import Runner, run_target
from bench.core.score import TaskScore
from bench.core.task import Task, load_tasks
from bench.core.trace import Trace
from bench.targets import ScriptedTarget, TargetConfig


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class StubTarget:
    """A target whose every answer is decided by the test, including its cost."""

    label = "stub"

    def __init__(self, traces: list[Trace] | None = None) -> None:
        self.traces = traces or [Trace(output="ok", latency_ms=5)]
        self.calls: list[tuple[str, int]] = []
        self.closed = False
        self._index = 0

    async def run(self, task_obj: Task, *, repeat: int) -> Trace:
        self.calls.append((task_obj.id, repeat))
        trace = self.traces[self._index % len(self.traces)]
        self._index += 1
        return trace.model_copy(deep=True)

    async def aclose(self) -> None:
        self.closed = True


def scripted(**overrides) -> ScriptedTarget:
    config = {"kind": "scripted", "callable": "fake_agents:echo"}
    config.update(overrides)
    return ScriptedTarget("scripted", TargetConfig.model_validate(config))


def dataset(tmp_path: Path, cases: int = 3) -> list[Task]:
    entries = [
        task(
            id=f"gqa-{index:04d}",
            input={"question": f"prompt {index}"},
            expectations=[{"type": "contains", "value": f"prompt {index}"}],
        )
        for index in range(cases)
    ]
    return list(load_tasks(write_set(tmp_path, entries)))


def score_everything(current: Task, traces: list[Trace]) -> TaskScore:
    """A stand-in scorer, enough to exercise the seam without `scorers/` (which does not
    exist yet): an answer counts unless it is the fixture's wrong one or the trace
    errored. Partial credit falls out of the same count, which is what makes the flaky
    fixture produce two passes out of four."""
    passed = sum(1 for trace in traces if not trace.failed and trace.output != "wrong")
    return TaskScore(
        task_id=current.id,
        family=current.family,
        outcome="pass" if passed == len(traces) else "fail",
        score=passed / len(traces),
        repeats=len(traces),
        passes=passed,
    )


# --- repeats -----------------------------------------------------------------


def test_a_task_runs_the_configured_number_of_times(tmp_path: Path):
    target = scripted()
    runner = Runner(target, repeats=3)
    runs = asyncio.run(runner.run(dataset(tmp_path, cases=2)))
    assert [item.repeats for item in runs.tasks] == [3, 3]
    assert len(target.__dict__)  # the target was constructed and used
    assert len(runs.tasks[0].traces) == 3


def test_repeats_are_numbered_so_a_fixture_can_be_deliberately_flaky(tmp_path: Path):
    """The fixture entrant that passes on even repeats is the reason the number is
    passed through at all."""
    target = scripted(callable="fake_agents:flaky")
    runs = asyncio.run(Runner(target, repeats=4).run(dataset(tmp_path, cases=1)))
    assert [trace.output for trace in runs.tasks[0].traces] == [
        "correct",
        "wrong",
        "correct",
        "wrong",
    ]


def test_one_repeat_says_it_cannot_measure_flakiness(tmp_path: Path):
    """Said in the notes rather than left for the reader to notice, because the number
    that is missing is the one people assume is there."""
    runs = asyncio.run(Runner(scripted(), repeats=1).run(dataset(tmp_path, cases=1)))
    assert any("cannot measure flakiness" in note for note in runs.notes)


def test_the_result_keeps_the_order_the_tasks_were_given_in(tmp_path: Path):
    """Tasks finish in whatever order the network decides; a run file that shuffles is a
    diff nobody reads."""
    tasks = dataset(tmp_path, cases=5)
    runs = asyncio.run(Runner(scripted(), repeats=2, concurrency=5).run(tasks))
    assert [item.task_id for item in runs.tasks] == [item.id for item in tasks]


def test_repeats_must_be_at_least_one():
    with pytest.raises(ValueError, match="at least 1"):
        Runner(scripted(), repeats=0)


def test_concurrency_must_be_at_least_one():
    with pytest.raises(ValueError, match="at least 1"):
        Runner(scripted(), concurrency=0)


def test_concurrency_is_respected(tmp_path: Path):
    """Measured rather than assumed: a semaphore that is constructed but never awaited
    looks identical in a passing test suite."""
    live = {"now": 0, "peak": 0}

    class CountingTarget(StubTarget):
        async def run(self, task_obj: Task, *, repeat: int) -> Trace:
            live["now"] += 1
            live["peak"] = max(live["peak"], live["now"])
            await asyncio.sleep(0.01)
            live["now"] -= 1
            return await super().run(task_obj, repeat=repeat)

    asyncio.run(Runner(CountingTarget(), repeats=1, concurrency=2).run(dataset(tmp_path, cases=6)))
    assert live["peak"] <= 2


# --- failure is a value -------------------------------------------------------


def test_a_target_that_raises_produces_an_error_task_not_an_exception(tmp_path: Path):
    runs = asyncio.run(
        Runner(scripted(callable="fake_agents:explode"), repeats=2).run(dataset(tmp_path, cases=2))
    )
    assert [item.outcome for item in runs.tasks] == ["error", "error"]
    assert "retrieval index" in (runs.tasks[0].error or "")
    assert len(runs.errored) == 2


def test_a_task_that_fails_some_repeats_is_completed_not_errored(tmp_path: Path):
    """The distinction the whole report rests on: a flaky answer is a judgement about
    the entrant, a broken endpoint is a judgement about the run."""
    target = StubTarget(
        [Trace(output="ok", latency_ms=1), Trace(error="ConnectError: refused", latency_ms=1)]
    )
    runs = asyncio.run(Runner(target, repeats=2).run(dataset(tmp_path, cases=1)))
    assert runs.tasks[0].outcome == "completed"
    assert runs.tasks[0].failed_repeats == 1


def test_a_task_where_every_repeat_errored_is_an_error(tmp_path: Path):
    target = StubTarget([Trace(error="ConnectError: refused", latency_ms=1)])
    runs = asyncio.run(Runner(target, repeats=3).run(dataset(tmp_path, cases=1)))
    assert runs.tasks[0].outcome == "error"
    assert runs.tasks[0].error == "ConnectError: refused"


def test_flakiness_is_read_from_the_scores_when_scoring_ran(tmp_path: Path):
    target = scripted(callable="fake_agents:flaky")
    runs = asyncio.run(
        Runner(target, repeats=4, scorer=score_everything).run(dataset(tmp_path, cases=1))
    )
    assert runs.flaky_task_ids() == ["gqa-0000"]
    assert runs.scores[0].flaky


def test_flakiness_falls_back_to_the_traces_when_scoring_did_not_run(tmp_path: Path):
    """Runs without a scorer still get a flakiness list, and the report says which of
    the two signals it used: the fallback reads "some repeats bad", which a task whose
    repeats *all* errored would also satisfy — and that task is broken, not flaky."""
    target = StubTarget([Trace(output="ok", latency_ms=1), Trace(error="timeout", latency_ms=1)])
    runs = asyncio.run(Runner(target, repeats=2).run(dataset(tmp_path, cases=1)))
    assert runs.flaky_task_ids() == ["gqa-0000"]


def test_a_task_whose_repeats_all_errored_is_not_flaky(tmp_path: Path):
    target = StubTarget([Trace(error="timeout", latency_ms=1)])
    runs = asyncio.run(Runner(target, repeats=3).run(dataset(tmp_path, cases=1)))
    assert runs.flaky_task_ids() == []


# --- budgets ------------------------------------------------------------------


def test_a_task_over_its_cost_budget_stops_and_says_so(tmp_path: Path):
    target = StubTarget([Trace(output="ok", cost_usd=Decimal("0.02"), latency_ms=1)])
    tasks = dataset(tmp_path, cases=1)
    tasks[0].budget.max_usd = 0.05
    runs = asyncio.run(Runner(target, repeats=5).run(tasks))
    item = runs.tasks[0]
    assert item.outcome == "over-budget"
    # three repeats at two cents is over five cents; the other two were not attempted
    assert item.repeats == 3
    assert "over the $0.0500 budget" in (item.error or "")


def test_a_task_over_its_time_budget_is_stopped_by_wall_time(tmp_path: Path):
    """Wall time, not the target's own accounting: a retry loop the target does not
    report still blows the budget."""
    clock = FakeClock()

    class SlowTarget(StubTarget):
        async def run(self, task_obj: Task, *, repeat: int) -> Trace:
            clock.advance(0.6)
            return Trace(output="ok", latency_ms=600)

    tasks = dataset(tmp_path, cases=1)
    tasks[0].budget.max_seconds = 1.0
    runs = asyncio.run(Runner(SlowTarget(), repeats=5, clock=clock).run(tasks))
    item = runs.tasks[0]
    assert item.outcome == "over-budget"
    assert item.repeats == 2  # 0.6 then 1.2
    assert "1.0s budget" in (item.error or "")


def test_a_task_inside_its_budget_is_untouched(tmp_path: Path):
    target = StubTarget([Trace(output="ok", cost_usd=Decimal("0.001"), latency_ms=1)])
    tasks = dataset(tmp_path, cases=1)
    tasks[0].budget.max_usd = 0.05
    runs = asyncio.run(Runner(target, repeats=3).run(tasks))
    assert runs.tasks[0].outcome == "completed"
    assert runs.tasks[0].repeats == 3


def test_a_run_over_its_budget_marks_the_rest_skipped(tmp_path: Path):
    """A per-run budget stops everything, and the tasks that never ran are *skipped* —
    not zeroes, not failures. Averaging a skipped task in would punish an entry for
    being partial."""
    target = StubTarget([Trace(output="ok", cost_usd=Decimal("0.03"), latency_ms=1)])
    tasks = dataset(tmp_path, cases=4)
    runs = asyncio.run(
        Runner(target, repeats=1, concurrency=1, run_budget_usd=Decimal("0.05")).run(tasks)
    )
    assert runs.partial
    assert runs.skipped
    assert all(item.outcome == "skipped" for item in runs.skipped)
    assert "run budget exhausted" in (runs.skipped[0].error or "")


def test_a_partial_run_says_so_in_its_notes_and_its_summary(tmp_path: Path):
    target = StubTarget([Trace(output="ok", cost_usd=Decimal("0.03"), latency_ms=1)])
    tasks = dataset(tmp_path, cases=4)
    runs = asyncio.run(
        Runner(target, repeats=1, concurrency=1, run_budget_usd=Decimal("0.05")).run(tasks)
    )
    assert runs.summary()["partial"] is True
    assert runs.partial_reason and "budget" in runs.partial_reason
    assert any("cannot be compared with a full one" in note for note in runs.notes)


def test_a_skipped_run_still_reports_its_finished_tasks(tmp_path: Path):
    """The results that did happen are still results: throwing them away because the run
    stopped early is how a 90%-complete run becomes worthless."""
    target = StubTarget([Trace(output="ok", cost_usd=Decimal("0.03"), latency_ms=1)])
    tasks = dataset(tmp_path, cases=4)
    runs = asyncio.run(
        Runner(target, repeats=1, concurrency=1, run_budget_usd=Decimal("0.05")).run(tasks)
    )
    assert len(runs.completed) >= 1
    assert runs.summary()["tasks"] == 4


# --- cost and latency ---------------------------------------------------------


def test_the_run_cost_is_a_decimal_or_none():
    """`None` when nothing was priced. Zero would claim the run was free."""
    target = StubTarget([Trace(output="ok", latency_ms=1)])
    runs = asyncio.run(Runner(target, repeats=1).run([Task.model_validate(task())]))
    assert runs.cost() is None


def test_costs_are_summed_across_tasks_and_repeats(tmp_path: Path):
    target = StubTarget([Trace(output="ok", cost_usd=Decimal("0.010000"), latency_ms=1)])
    runs = asyncio.run(Runner(target, repeats=2).run(dataset(tmp_path, cases=3)))
    assert runs.cost() == Decimal("0.060000")


def test_latencies_are_kept_per_repeat_not_averaged(tmp_path: Path):
    """p95 of a run is over the repeats that happened; an average would hide the tail,
    which is the part that hurts."""
    target = StubTarget([Trace(output="ok", latency_ms=10), Trace(output="ok", latency_ms=800)])
    runs = asyncio.run(Runner(target, repeats=2).run(dataset(tmp_path, cases=1)))
    assert sorted(runs.tasks[0].latencies_ms) == [10, 800]


def test_the_summary_counts_every_outcome_separately(tmp_path: Path):
    target = StubTarget([Trace(error="boom", latency_ms=1)])
    runs = asyncio.run(Runner(target, repeats=1).run(dataset(tmp_path, cases=2)))
    summary = runs.summary()
    assert summary["tasks"] == 2
    assert summary["errored"] == 2
    assert summary["completed"] == 0
    assert summary["skipped"] == 0
    assert summary["cost_usd"] is None


# --- the seam -----------------------------------------------------------------


def test_the_scorer_is_called_once_per_task_with_its_traces(tmp_path: Path):
    seen: list[tuple[str, int]] = []

    def scorer(current: Task, traces: list[Trace]) -> TaskScore:
        seen.append((current.id, len(traces)))
        return TaskScore(
            current.id, current.family, "pass", 1.0, repeats=len(traces), passes=len(traces)
        )

    runs = asyncio.run(Runner(scripted(), repeats=2, scorer=scorer).run(dataset(tmp_path, cases=3)))
    assert seen == [("gqa-0000", 2), ("gqa-0001", 2), ("gqa-0002", 2)]
    assert len(runs.scores) == 3


def test_the_runner_does_not_need_a_scorer(tmp_path: Path):
    """The layering: the runner's contract is what happened and what it cost. Whether
    the answers were right is a different file's business."""
    runs = asyncio.run(Runner(scripted(), repeats=1).run(dataset(tmp_path, cases=2)))
    assert runs.scores == []
    assert all(item.outcome == "completed" for item in runs.tasks)


def test_a_callback_sees_each_task_as_it_finishes(tmp_path: Path):
    """What a progress line is built from. Deliberately not a logging call inside the
    runner: the CLI owns stdout, and a library that prints is a library that cannot be
    quiet."""
    done: list[str] = []
    asyncio.run(
        Runner(scripted(), repeats=1, on_task_done=lambda item: done.append(item.task_id)).run(
            dataset(tmp_path, cases=3)
        )
    )
    assert sorted(done) == ["gqa-0000", "gqa-0001", "gqa-0002"]


# --- the one-shot form --------------------------------------------------------


def test_run_target_closes_the_target(tmp_path: Path):
    """A leaked HTTP client shows up as a warning about an unclosed session and a hang
    at exit, and the person who sees it is not the person who wrote the runner."""
    target = StubTarget()
    asyncio.run(run_target(target, dataset(tmp_path, cases=2), repeats=1))
    assert target.closed


def test_run_target_closes_the_target_even_when_something_raises(tmp_path: Path):
    class Exploding(StubTarget):
        async def run(self, task_obj: Task, *, repeat: int) -> Trace:
            raise KeyboardInterrupt

    target = Exploding()
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(run_target(target, dataset(tmp_path, cases=1), repeats=1))
    assert target.closed


def test_run_target_labels_the_run_after_the_target_by_default(tmp_path: Path):
    target = scripted()
    runs = asyncio.run(run_target(target, dataset(tmp_path, cases=1), repeats=1))
    assert runs.label == "scripted"
    assert runs.started_at.endswith("Z")
