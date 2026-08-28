"""Run files on disk: written atomically, never overwritten, always readable back.

Three rules, and each of them is a defect that costs somebody a day.

**A run file is never overwritten.** Two runs in the same second get `-2`, `-3`. The
alternative — silently replacing the baseline with the thing being compared against it —
reads as "nothing changed", which is the most dangerous possible wrong answer, and it is
the one a benchmark is most likely to give because the alternative is exactly what the
naive code does.

**A write is atomic.** A killed process must not leave a half-written JSON file that
parses as a shorter run.

**`latest` means write time, not name.** Directory order is filesystem order; name order
puts `-2` before the file it was copied from, because `-` sorts before `.` in the ids
this project generates. Write time is what "latest" means and is what `list_runs` sorts
by.

The traces are dropped above `traces_above` tasks unless `with_traces` is asked for: a
300-task run with four repeats is a 40 MB file, which is fine for `jq` and hostile to
git. The summary is what the board reads; the traces are what a person reads when one
task moved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..errors import BenchError
from .score import TaskScore
from .trace import Trace

RUN_SUFFIX = ".run.json"


class StoreError(BenchError):
    """A run file that cannot be written or read. `BenchError` with exit code 2: this
    is the harness failing, not the entry."""


def save_run(
    path: str | Path,
    *,
    result: Any = None,
    payload: dict[str, Any] | None = None,
    with_traces: bool = True,
    traces_above: int = 5000,
) -> Path:
    """Write a run file and return where it landed.

    `result` is a `RunResult`; `payload` is a dict for the tests and for hand-written
    files. Exactly one of them.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    if result is not None and payload is None:
        document = to_document(result, with_traces=with_traces, traces_above=traces_above)
    elif payload is not None and result is None:
        document = payload
    else:
        raise StoreError("save_run takes either a result or a payload, not both or neither")

    run_id = document.get("run_id")
    if not run_id:
        raise StoreError("a run file needs a run_id")
    target = _free_name(directory / f"{run_id}{RUN_SUFFIX}")

    # atomic: a killed process must not leave a file that parses as a shorter run
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=False), encoding="utf-8")
    temporary.replace(target)
    return target


def _free_name(target: Path) -> Path:
    """`…-2.run.json`, with the number *before* the extension and not before `.json`.

    Built from the name rather than from `Path.stem`, because `stem` would give
    `…-8df1b7.run-2.json` — a file that does not match `*.run.json`, so `list_runs`
    would skip it and `latest` would quietly resolve to a different run. That is the
    same class of bug as the overwrite this guard exists to prevent, arriving through
    the guard itself, which is why the suffix is inserted explicitly.
    """
    if not target.exists():
        return target
    stem = target.name[: -len(RUN_SUFFIX)] if target.name.endswith(RUN_SUFFIX) else target.name
    for index in range(2, 1000):
        candidate = target.with_name(f"{stem}-{index}{RUN_SUFFIX}")
        if not candidate.exists():
            return candidate
    raise StoreError(f"{target.parent} is full of {target.name} — 999 copies already")


def load_run(path: str | Path) -> dict[str, Any]:
    file = Path(path)
    if not file.exists():
        raise StoreError(f"no run file at {file}")
    try:
        document = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StoreError(f"{file} is not valid JSON: {exc}") from exc
    if not isinstance(document, dict) or "run_id" not in document:
        raise StoreError(f"{file} does not look like a run file (no run_id)")
    return document


def list_runs(directory: str | Path) -> list[Path]:
    """Newest first, by write time, with the copy number as a tie-break.

    The tie-break is the *sequence number*, not the name, and that distinction cost a
    flaky test to learn. Filesystem timestamps are coarser than they look — two runs
    written a few milliseconds apart routinely share an `st_mtime_ns` — and when they do,
    the name decides, and `…-e83ccf-2.run.json` sorts *before* `…-e83ccf.run.json`
    because `-` (0x2d) is below `.` (0x2e). The duplicate, which is the newer run, would
    lose to the file it was copied from, and `latest` would point at the older one.

    So: same write time, higher copy number wins. `_copy_number` is what reads the
    suffix, and it is also what `_free_name` writes.
    """
    path = Path(directory)
    if not path.exists():
        return []
    files = [item for item in path.glob(f"*{RUN_SUFFIX}") if item.is_file()]
    return sorted(
        files,
        key=lambda item: (item.stat().st_mtime_ns, _copy_number(item.name), item.name),
        reverse=True,
    )


def _copy_number(name: str) -> int:
    """The `-2` in `…-e83ccf-2.run.json`, or 0 for a file with no suffix.

    Parsed rather than compared as a string, because the string comparison is exactly what
    gets this wrong. A name that does not parse is 0 — it was not written by `_free_name`,
    so it has no claim to being a later copy of anything.
    """
    stem = name[: -len(RUN_SUFFIX)] if name.endswith(RUN_SUFFIX) else name
    _, separator, tail = stem.rpartition("-")
    return int(tail) if separator and tail.isdigit() else 0


def resolve_run(
    directory: str | Path, reference: str, *, baseline: str | Path | None = None
) -> Path:
    """Turn `latest`, `baseline`, a run id or a path into a file.

    Four spellings because all four get typed in practice, and the error message when
    none of them match lists what *is* there — a `passmark compare latest` that fails
    with "no such file" and no listing is a minute of somebody's life per occurrence.
    """
    if reference == "baseline":
        if baseline is None:
            raise StoreError(
                "no baseline is configured",
                hint="record one with `bench run --baseline`, or pass a run id",
            )
        return Path(baseline)
    if reference == "latest":
        runs = list_runs(directory)
        if not runs:
            raise StoreError(
                f"no runs in {directory}",
                hint="run something first: `bench run --tasks tasks --target target.yaml`",
            )
        return runs[0]
    candidate = Path(reference)
    if candidate.exists():
        return candidate
    for run in list_runs(directory):
        if run.name == reference or run.name == f"{reference}{RUN_SUFFIX}":
            return run
    available = ", ".join(item.stem.replace(RUN_SUFFIX, "") for item in list_runs(directory)[:5])
    raise StoreError(
        f"no run matches {reference!r}",
        hint=f"run ids in {directory}: {available or 'none'}",
    )


# --- the document -------------------------------------------------------------


def to_document(
    result: Any, *, with_traces: bool = True, traces_above: int = 5000
) -> dict[str, Any]:
    """A `RunResult` as the dict that gets written.

    The shape is stable and versioned, because a result bundle from today is read by a
    verifier in six months: `run_id`, `label`, `summary`, `tasks`, and `traces` only when
    they were kept. `truncated_traces` says so plainly rather than leaving the reader to
    notice that the array is missing.
    """
    keep_traces = with_traces and len(result.tasks) <= traces_above
    document: dict[str, Any] = {
        "format": 1,
        "run_id": run_id(result),
        "label": result.label,
        "started_at": result.started_at,
        "wall_seconds": round(result.wall_seconds, 3),
        "repeats": result.repeats,
        "concurrency": result.concurrency,
        "partial": result.partial,
        "partial_reason": result.partial_reason,
        "summary": result.summary(),
        "notes": list(result.notes),
        "tasks": [],
    }
    if not keep_traces:
        document["truncated_traces"] = (
            f"traces omitted: {len(result.tasks)} tasks is above the {traces_above} limit; "
            f"re-run with --with-traces"
        )

    scores = {score.task_id: score for score in result.scores}
    for item in result.tasks:
        entry: dict[str, Any] = {
            "task_id": item.task_id,
            "family": item.family,
            "split": item.split,
            "outcome": item.outcome,
            "repeats": item.repeats,
            "failed_repeats": item.failed_repeats,
            "latency_ms": item.latencies_ms,
            "cost_usd": str(item.cost()) if item.cost() is not None else None,
            "wall_seconds": round(item.wall_seconds, 3),
            "error": item.error,
        }
        if item.task_id in scores:
            score = scores[item.task_id]
            entry["score"] = round(score.score, 6)
            entry["score_outcome"] = score.outcome
            entry["passes"] = score.passes
            entry["judge_scored"] = score.judge_scored
            entry["judge_trusted"] = score.judge_trusted
            entry["notes"] = list(score.notes)
        if keep_traces and item.traces:
            entry["traces"] = [json.loads(trace.model_dump_json()) for trace in item.traces]
        document["tasks"].append(entry)
    return document


def run_id(result: Any) -> str:
    """`20261001T091422Z-<6 hex>`, where the hash covers the label and the task ids.

    The timestamp alone collides when two runs are started in the same second (which
    happens in tests and in `--limit 5` loops), and the label alone collides across days.
    The hash is of the *question*, not the answer — so re-running the same entry on the
    same tasks twice in one second is the thing that stays distinguishable.
    """
    import hashlib

    blob = json.dumps(
        {"label": result.label, "tasks": [item.task_id for item in result.tasks]},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:6]
    stamp = result.started_at.replace("-", "").replace(":", "").replace("T", "T").rstrip("Z")
    return f"{stamp}Z-{digest}"


def traces_from_document(document: dict[str, Any], task_id: str) -> list[Trace]:
    """Pull one task's traces back out of a run file, for `bench show <run> <task>`."""
    for entry in document.get("tasks", []):
        if entry.get("task_id") == task_id:
            return [Trace.model_validate(payload) for payload in entry.get("traces", [])]
    raise StoreError(f"{task_id} is not in this run")


def scores_from_document(document: dict[str, Any]) -> list[TaskScore]:
    """Rebuild `TaskScore`s so a report can be regenerated from a run file alone.

    Only the fields a report reads are restored; anything the scorer knew and did not
    write is gone, which is why the run file stores the explanation strings rather than
    a reference to the expectation.
    """
    scores: list[TaskScore] = []
    for entry in document.get("tasks", []):
        if "score" not in entry:
            continue
        scores.append(
            TaskScore(
                task_id=entry["task_id"],
                family=entry["family"],
                outcome=entry.get("score_outcome", entry["outcome"]),
                score=float(entry["score"]),
                repeats=int(entry.get("repeats", 1)),
                passes=int(entry.get("passes", 0)),
                split=entry.get("split", "public"),
                judge_scored=bool(entry.get("judge_scored")),
                judge_trusted=bool(entry.get("judge_trusted", True)),
                notes=list(entry.get("notes", [])),
            )
        )
    return scores
