"""Run files: written atomically, never overwritten, resolvable four ways.

Every test in this file is a defect that costs somebody a day. The one that matters
most is the first: a benchmark whose run file can be silently replaced reports "nothing
changed" when the thing being compared against was overwritten by the comparison —
which is the most dangerous possible wrong answer, and exactly what the naive version of
this code does.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from decimal import Decimal
from pathlib import Path

import pytest
from conftest import task, write_set

from bench.core.runner import RunResult, TaskRun, run_target
from bench.core.score import TaskScore
from bench.core.store import (
    StoreError,
    _copy_number,
    list_runs,
    load_run,
    resolve_run,
    run_id,
    save_run,
    scores_from_document,
    to_document,
    traces_from_document,
)
from bench.core.task import Task, load_tasks
from bench.core.trace import Trace
from bench.targets import ScriptedTarget, TargetConfig


def sample_run(tmp_path: Path, *, cases: int = 2) -> RunResult:
    entries = [
        task(
            id=f"gqa-{index:04d}",
            input={"question": f"prompt {index}"},
            expectations=[{"type": "contains", "value": f"prompt {index}"}],
        )
        for index in range(cases)
    ]
    tasks = list(load_tasks(write_set(tmp_path, entries)))
    target = ScriptedTarget(
        "scripted",
        TargetConfig.model_validate({"kind": "scripted", "callable": "fake_agents:echo"}),
    )
    return asyncio.run(run_target(target, tasks, repeats=2))


# --- never overwritten --------------------------------------------------------


def test_a_second_run_in_the_same_second_does_not_replace_the_first(tmp_path: Path):
    """The defect this rule exists for. Two runs started in the same second share a run
    id; without the guard the baseline is replaced by the run being compared against it,
    every comparison reads "nothing changed", and the gate passes."""
    run = sample_run(tmp_path)
    first = save_run(tmp_path / "runs", result=run)
    second = save_run(tmp_path / "runs", result=run)
    assert first != second
    assert first.exists() and second.exists()
    assert second.name.endswith("-2.run.json")
    assert load_run(first) == load_run(second)


def test_the_suffix_keeps_counting(tmp_path: Path):
    """Compared as a set, not in sorted order — because a name sort puts `-2` *before*
    the file it was copied from (`-` is 0x2d, `.` is 0x2e). That is the trap `list_runs`
    avoids by sorting on write time, and writing this test with `sorted()` is how the
    trap gets rediscovered."""
    run = sample_run(tmp_path)
    paths = [save_run(tmp_path / "runs", result=run) for _ in range(4)]
    suffixes = {path.name.removeprefix(run_id(run)) for path in paths}
    assert suffixes == {".run.json", "-2.run.json", "-3.run.json", "-4.run.json"}


def test_a_suffixed_file_is_still_found_by_latest(tmp_path: Path):
    """The number goes before the extension, not before `.json`. Get that wrong and the
    duplicate does not match `*.run.json`, `list_runs` skips it, and `latest` resolves to
    a different run — the same silent-substitution bug the suffix exists to prevent,
    arriving through the guard itself. Found by writing the test above and reading the
    failure: the file was called `…-8df1b7.run-2.json`."""
    run = sample_run(tmp_path)
    directory = tmp_path / "runs"
    for _ in range(3):
        save_run(directory, result=run)
    assert len(list_runs(directory)) == 3
    assert resolve_run(directory, "latest") == list_runs(directory)[0]


def test_the_run_id_carries_the_label_and_the_tasks(tmp_path: Path):
    """The timestamp alone collides inside one second, and the label alone collides
    across days. The hash is of the question — the label and the task ids — so the same
    entry re-run on the same tasks in the same second stays distinguishable."""
    run = sample_run(tmp_path)
    identifier = run_id(run)
    assert identifier.endswith(run_id(run)[-6:])
    assert len(identifier.split("-")[-1]) == 6

    other = sample_run(tmp_path)
    other.label = "a different entrant"
    assert run_id(other) != run_id(run)


# --- atomic -------------------------------------------------------------------


def test_a_write_leaves_no_temporary_file_behind(tmp_path: Path):
    save_run(tmp_path / "runs", result=sample_run(tmp_path))
    assert not list((tmp_path / "runs").glob("*.tmp"))


def test_a_payload_and_a_result_together_are_refused(tmp_path: Path):
    with pytest.raises(StoreError, match="either a result or a payload"):
        save_run(tmp_path / "runs", result=sample_run(tmp_path), payload={"run_id": "x"})


def test_a_document_without_a_run_id_is_refused(tmp_path: Path):
    with pytest.raises(StoreError, match="run_id"):
        save_run(tmp_path / "runs", payload={"label": "no id"})


# --- reading back -------------------------------------------------------------


def test_a_run_round_trips_through_the_file(tmp_path: Path):
    run = sample_run(tmp_path)
    path = save_run(tmp_path / "runs", result=run)
    document = load_run(path)
    assert document["label"] == "scripted"
    assert document["summary"]["tasks"] == 2
    assert document["repeats"] == 2
    assert [entry["task_id"] for entry in document["tasks"]] == ["gqa-0000", "gqa-0001"]
    assert document["tasks"][0]["traces"][0]["output"] == "prompt 0"


def test_traces_can_be_read_back_out_by_task_id(tmp_path: Path):
    run = sample_run(tmp_path)
    document = load_run(save_run(tmp_path / "runs", result=run))
    traces = traces_from_document(document, "gqa-0001")
    assert len(traces) == 2
    assert all(isinstance(trace, Trace) for trace in traces)
    with pytest.raises(StoreError, match="not in this run"):
        traces_from_document(document, "gqa-9999")


def test_a_run_that_is_not_a_run_file_says_so(tmp_path: Path):
    file = tmp_path / "notes.json"
    file.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    with pytest.raises(StoreError, match="does not look like a run file"):
        load_run(file)


def test_broken_json_is_reported_with_the_reason(tmp_path: Path):
    file = tmp_path / "half.json"
    file.write_text('{"run_id": "x"', encoding="utf-8")
    with pytest.raises(StoreError, match="not valid JSON"):
        load_run(file)


def test_a_missing_run_file_says_which_one(tmp_path: Path):
    with pytest.raises(StoreError, match="no run file at"):
        load_run(tmp_path / "nope.run.json")


# --- traces above the limit ---------------------------------------------------


def test_a_large_run_drops_the_traces_and_says_so(tmp_path: Path):
    """A 300-task run with four repeats is a 40 MB file: fine for jq, hostile to git.
    The summary is what the board reads; the traces are what a person reads when one
    task moved — and saying they were dropped beats leaving the reader to notice."""
    run = sample_run(tmp_path)
    document = to_document(run, with_traces=False)
    assert "traces" not in document["tasks"][0]
    assert "truncated_traces" in document
    assert "traces omitted" in document["truncated_traces"]


def test_the_limit_is_a_number_and_not_a_boolean(tmp_path: Path):
    run = sample_run(tmp_path)
    assert "truncated_traces" not in to_document(run, with_traces=True, traces_above=5000)
    assert "truncated_traces" in to_document(run, with_traces=True, traces_above=1)


# --- listing and resolving ----------------------------------------------------


def make_three(tmp_path: Path) -> Path:
    directory = tmp_path / "runs"
    for index in range(3):
        run = sample_run(tmp_path)
        run.started_at = f"2026-10-01T09:1{index}:00Z"
        path = save_run(directory, result=run)
        os.utime(path, ns=(time.time_ns() + index * 1_000_000,) * 2)
    return directory


def test_runs_are_listed_newest_first_by_write_time(tmp_path: Path):
    """Write time, not name. A `-2` suffix sorts *before* the id it was copied from,
    because `-` is 0x2d and `.` is 0x2e — so a name sort hands `latest` the older file,
    which is exactly backwards."""
    directory = make_three(tmp_path)
    runs = list_runs(directory)
    assert len(runs) == 3
    assert runs[0].stat().st_mtime_ns >= runs[-1].stat().st_mtime_ns


def test_latest_resolves_to_the_newest_file(tmp_path: Path):
    directory = make_three(tmp_path)
    assert resolve_run(directory, "latest") == list_runs(directory)[0]


def test_a_run_id_resolves_with_or_without_the_suffix(tmp_path: Path):
    directory = make_three(tmp_path)
    newest = list_runs(directory)[0]
    identifier = newest.name[: -len(".run.json")]
    assert resolve_run(directory, identifier) == newest
    assert resolve_run(directory, newest.name) == newest


def test_a_path_resolves_when_it_exists(tmp_path: Path):
    directory = make_three(tmp_path)
    some = list_runs(directory)[-1]
    assert resolve_run(directory, str(some)) == some


def test_an_unresolvable_reference_lists_what_is_there(tmp_path: Path):
    """A `latest` that fails with "no such file" and no listing costs somebody a minute
    every time they mistype a run id."""
    directory = make_three(tmp_path)
    with pytest.raises(StoreError) as excinfo:
        resolve_run(directory, "20260101T000000Z-deadbe")
    assert "run ids in" in (excinfo.value.hint or "")


def test_latest_on_an_empty_directory_explains_itself(tmp_path: Path):
    with pytest.raises(StoreError) as excinfo:
        resolve_run(tmp_path / "empty", "latest")
    assert "run something first" in (excinfo.value.hint or "")


def test_baseline_pointing_nowhere_is_an_error_and_not_a_silent_success(tmp_path: Path):
    with pytest.raises(StoreError, match="no baseline is configured"):
        resolve_run(tmp_path, "baseline")


# --- regenerating scores from a file ------------------------------------------


def test_scores_can_be_rebuilt_from_a_run_file_alone(tmp_path: Path):
    """So a report can be regenerated months later without re-running anything, which is
    the whole reason the run file stores the explanation strings rather than a reference
    to an expectation."""

    def scorer(current: Task, traces: list[Trace]) -> TaskScore:
        passed = sum(1 for trace in traces if not trace.failed)
        return TaskScore(
            current.id,
            current.family,
            "pass" if passed else "fail",
            passed / len(traces),
            repeats=len(traces),
            passes=passed,
            notes=["checked the citation block"],
        )

    entries = [
        task(
            id="gqa-0001",
            input={"question": "prompt 1"},
            expectations=[{"type": "contains", "value": "prompt 1"}],
        )
    ]
    tasks = list(load_tasks(write_set(tmp_path, entries)))
    target = ScriptedTarget(
        "scripted",
        TargetConfig.model_validate({"kind": "scripted", "callable": "fake_agents:echo"}),
    )
    run = asyncio.run(run_target(target, tasks, repeats=2, scorer=scorer))
    document = load_run(save_run(tmp_path / "runs", result=run))

    scores = scores_from_document(document)
    assert len(scores) == 1
    assert scores[0].task_id == "gqa-0001"
    assert scores[0].score == 1.0
    assert scores[0].notes == ["checked the citation block"]


def test_a_run_file_without_scores_yields_no_scores(tmp_path: Path):
    """A run that had no scorer is not a run of zero scores; it is a run of none."""
    run = sample_run(tmp_path)
    assert scores_from_document(to_document(run)) == []


# --- what the document holds --------------------------------------------------


def test_the_document_records_a_partial_run_as_partial(tmp_path: Path):
    """A partial run that reads like a full one is the most dangerous artefact this
    project can produce, so the flag and the reason are both in the file."""
    run = sample_run(tmp_path)
    run.partial = True
    run.partial_reason = "cost $0.0600 over the $0.0500 budget"
    run.notes.append("4 task(s) were never attempted")
    document = to_document(run)
    assert document["partial"] is True
    assert document["partial_reason"] == "cost $0.0600 over the $0.0500 budget"
    assert document["notes"]


def test_costs_stay_strings_so_a_reader_with_awk_does_not_lose_cents(tmp_path: Path):
    run = sample_run(tmp_path)
    document = to_document(run)
    assert document["tasks"][0]["cost_usd"] is None  # nothing was priced
    item = TaskRun("gqa-0001", "grounded-qa", "public", "completed", [Trace(output="x")])
    assert to_document(_wrap(item))["tasks"][0]["cost_usd"] is None

    priced = TaskRun(
        "gqa-0002",
        "grounded-qa",
        "public",
        "completed",
        [Trace(output="x", cost_usd=Decimal("0.000123"))],
    )
    assert to_document(_wrap(priced))["tasks"][0]["cost_usd"] == "0.000123"


def _wrap(*items: TaskRun) -> RunResult:
    return RunResult(
        label="t",
        tasks=list(items),
        started_at="2026-10-01T09:14:22Z",
        wall_seconds=0.5,
        repeats=1,
        concurrency=1,
    )


def test_a_copy_written_in_the_same_tick_still_counts_as_the_latest(tmp_path: Path) -> None:
    """Two runs a few milliseconds apart share an `st_mtime_ns`, and then the tie-break
    decides which one `latest` means.

    It used to be the name, and the name sorts backwards: `…-abc123-2.run.json` comes
    *before* `…-abc123.run.json` because `-` is below `.`, so the duplicate — the newer
    run — lost to the file it was copied from. Found by a cassette test that passed alone
    and failed in the suite, which is the signature of a timing assumption rather than of
    a broken feature.
    """
    first = tmp_path / "20260930T090000Z-abc123.run.json"
    second = tmp_path / "20260930T090000Z-abc123-2.run.json"
    third = tmp_path / "20260930T090000Z-abc123-10.run.json"
    for path in (first, second, third):
        path.write_text("{}", encoding="utf-8")
    stamp = first.stat().st_mtime_ns
    for path in (second, third):
        os.utime(path, ns=(stamp, stamp))

    assert list_runs(tmp_path)[0] == third  # 10 is a later copy than 2, not a string
    assert list_runs(tmp_path)[1] == second
    assert list_runs(tmp_path)[2] == first


def test_a_name_that_is_not_a_copy_number_is_treated_as_the_original(tmp_path: Path) -> None:
    path = tmp_path / "20260930T090000Z-0a1b2c.run.json"
    path.write_text("{}", encoding="utf-8")
    assert list_runs(tmp_path) == [path]
    assert _copy_number(path.name) == 0
