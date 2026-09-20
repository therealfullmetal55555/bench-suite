"""The command line.

Four verbs so far: `schema`, `task`, `run` and `show`. They are the ones whose code
exists — a command that prints "not implemented" is a promise the README then has to
keep, and a benchmark repository above all others should not be promising numbers it
cannot compute. `score`, `report`, `board build`, `submit`, `verify` and `export`
arrive with the scorers and the board.

Exit codes are the interface (SPEC §8):

* `0` — the command did what it says
* `2` — the run could not finish: a target that will not build, a task set that does
  not lint, a report asked for a run file that is not there
* `3` — the result exists and may not be published. Not fired yet: it belongs to
  `board build` and `verify`, which are the commands that decide publication.

Nothing exits `1` yet either. `1` means "this entry is worse than the one it is
compared against", and that sentence needs the scorers before it can be true.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from . import __version__
from .core.runner import run_target
from .core.store import load_run, resolve_run, save_run, traces_from_document
from .core.task import Task, find_near_duplicates, load_tasks
from .core.trace import write_schema
from .errors import EXIT_OK, EXIT_RUN_FAILED, BenchError

if TYPE_CHECKING:  # the score models are imported for annotations only
    from collections.abc import Sequence

    from .core.score import TaskScore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench",
        description=(
            "A public benchmark for retrieval and tool-using agents. "
            "The leaderboard is built from results/, not from a database."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"bench {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    schema_parser = subparsers.add_parser("schema", help="the published trace contract")
    schema_sub = schema_parser.add_subparsers(dest="action", metavar="action")
    write = schema_sub.add_parser("write", help="regenerate schema/trace.schema.json")
    write.add_argument("--out", default=None, help="write somewhere else than schema/")
    print_parser = schema_sub.add_parser("print", help="print the schema to stdout")
    print_parser.add_argument("--compact", action="store_true", help="one line, for piping")

    task_parser = subparsers.add_parser("task", help="the task set")
    task_sub = task_parser.add_subparsers(dest="action", metavar="action")
    validate = task_sub.add_parser("validate", help="load and lint every task")
    validate.add_argument("path", nargs="?", default="tasks", help="tasks directory or file")
    lint_parser = task_sub.add_parser(
        "lint", help="validate, and check for near-duplicates across the splits"
    )
    lint_parser.add_argument("path", nargs="?", default="tasks", help="tasks directory or file")
    lint_parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="n-gram overlap above which two tasks are called near-duplicates",
    )

    generate = task_sub.add_parser(
        "generate", help="build the task set from the generator data and write one file per family"
    )
    generate.add_argument("--out", default="tasks", help="where the task files are written")
    generate.add_argument(
        "--added",
        default=None,
        help="the date stamped on every generated task (default: the generator's pinned date)",
    )
    generate.add_argument(
        "--public", type=int, default=None, help="cap the public tasks per family"
    )
    generate.add_argument(
        "--private", type=int, default=None, help="cap the private tasks per family"
    )
    generate.add_argument("--quiet", action="store_true")

    run_parser = subparsers.add_parser("run", help="run tasks against one target")
    run_parser.add_argument("--tasks", default="tasks", help="tasks directory or file")
    run_parser.add_argument("--target", required=True, help="a target.yaml")
    run_parser.add_argument("--split", choices=("public", "private"), default=None)
    run_parser.add_argument("--family", action="append", default=None)
    run_parser.add_argument("--limit", type=int, default=None, help="run the first N tasks")
    run_parser.add_argument("--repeats", type=int, default=None, help="override the config")
    run_parser.add_argument("--concurrency", type=int, default=8)
    run_parser.add_argument("--runs-dir", default="runs", help="where run files are written")
    run_parser.add_argument("--from-cassette", default=None, help="replay a recorded cassette")
    run_parser.add_argument("--record", default=None, help="record the live run into a cassette")
    run_parser.add_argument("--max-usd", type=float, default=None, help="stop the run past this")
    run_parser.add_argument("--max-seconds", type=float, default=None)
    run_parser.add_argument(
        "--judge",
        default=None,
        help="a judge.yaml; without it, judge-scored expectations are reported and excluded",
    )
    run_parser.add_argument("--no-write", action="store_true", help="do not write a run file")
    run_parser.add_argument("--quiet", action="store_true")

    judge_parser = subparsers.add_parser(
        "judge", help="the judged expectations, and whether to trust them"
    )
    judge_sub = judge_parser.add_subparsers(dest="action")
    calibrate = judge_sub.add_parser(
        "calibrate", help="compare the judge against hand-labelled answers"
    )
    calibrate.add_argument("--labels", required=True, help="the JSONL of labelled answers")
    calibrate.add_argument("--judge", default="judge.yaml", help="the judge to calibrate")
    calibrate.add_argument(
        "--no-write",
        action="store_true",
        help="print the report without writing the calibration file",
    )
    status = judge_sub.add_parser("status", help="what the calibration on file says")
    status.add_argument("--judge", default="judge.yaml")

    show_parser = subparsers.add_parser("show", help="print a run file")
    show_parser.add_argument("reference", help="a run id, a path, or `latest`")
    show_parser.add_argument("--runs-dir", default="runs")
    show_parser.add_argument("--task", default=None, help="print one task's traces as JSON")
    show_parser.add_argument("--json", action="store_true", help="the whole document")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return EXIT_OK

    try:
        if args.command == "schema":
            return _schema(args, parser)
        if args.command == "task":
            return _task(args, parser)
        if args.command == "run":
            return _run(args, parser)
        if args.command == "judge":
            return _judge(args, parser)
        if args.command == "show":
            return _show(args, parser)
    except BenchError as exc:
        print(f"bench: {exc}", file=sys.stderr)
        if exc.hint:
            print(f"       {exc.hint}", file=sys.stderr)
        return exc.exit_code
    parser.print_help()
    return EXIT_OK


def _schema(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.action == "write":
        path = write_schema(Path(args.out) if args.out else None)
        print(f"wrote {path}")
        return EXIT_OK
    if args.action == "print":
        from .core.trace import schema

        document = schema()
        if args.compact:
            print(json.dumps(document, separators=(",", ":")))
        else:
            print(json.dumps(document, indent=2))
        return EXIT_OK
    parser.parse_args(["schema", "--help"])
    return EXIT_OK


def _task(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.action == "generate":
        from .generators import DEFAULT_ADDED, build, cap, write

        families, _ = build(args.added or DEFAULT_ADDED)
        families = cap(families, public=args.public, private=args.private)
        written = write(Path(args.out), families=families)
        if not args.quiet:
            print(f"wrote {written.total} task(s) to {args.out}")
            print(f"  {written.describe()}")
        # Loading what was just written is the cheapest possible check that the generator
        # agrees with the loader, and it is exactly the check that a hand-edited file skips.
        task_set = load_tasks(args.out)
        print(f"  {len(task_set)} tasks load: {task_set.digest}")
        return EXIT_OK

    if args.action in {"validate", "lint"}:
        task_set = load_tasks(args.path)
        counts = {split: len(task_set.split(split)) for split in ("public", "private")}
        print(
            f"{len(task_set)} tasks: {counts['public']} public, {counts['private']} private, "
            f"{task_set.digest}"
        )
        print(f"  {task_set.describe()}")
        if args.action == "lint":
            pairs = find_near_duplicates(task_set, threshold=args.threshold)
            if pairs:
                # Not a failure. A near-duplicate pair across the splits is a finding
                # for a person to judge — sometimes two tasks really are that similar,
                # and the person who wrote them is the one who knows.
                print(f"  {len(pairs)} near-duplicate pair(s) across the splits:")
                for private_id, public_id, overlap in pairs[:20]:
                    print(f"    {private_id} ≈ {public_id}  ({overlap:.0%})")
                if len(pairs) > 20:
                    print(f"    … and {len(pairs) - 20} more")
            else:
                print("  no near-duplicates across the splits")
        return EXIT_OK
    parser.parse_args(["task", "--help"])
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - the entry point is `bench`
    raise SystemExit(main())


__all__ = ["build_parser", "main"]


# --- run ----------------------------------------------------------------------


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Run tasks against one target and write a run file.

    The one place in this project that spends money, so three things are said out loud
    before anything is called: which target, how many tasks, and how many calls that is
    (`tasks × repeats`). A benchmark that starts spending without saying how much is a
    benchmark people are afraid to run.
    """
    import asyncio
    from decimal import Decimal

    from .judge import JudgeClient, JudgeConfig
    from .scorers.base import build_scorer
    from .targets import build_target, load_target

    task_set = load_tasks(args.tasks)
    selected = _select(task_set, args)
    if not selected:
        raise BenchError(
            "no tasks selected",
            hint="check --split and --family against `bench task validate`",
        )

    judge = None
    if args.judge:
        judge_config = JudgeConfig.from_file(args.judge)
        judge = JudgeClient(judge_config)
        if not args.quiet:
            # Said before the first call, not after the run: whether the judged tasks
            # count is the difference between a score and a partial score, and finding
            # that out from a footnote in a JSON file is too late.
            print(f"judge {judge_config.model}: {judge.trust_note()}")

    target_file = load_target(args.target)
    config = target_file.target
    repeats = args.repeats if args.repeats is not None else _config_repeats(config)
    target = build_target(
        target_file.name,
        config,
        cassette=args.from_cassette,
        record_to=args.record,
    )

    calls = len(selected) * repeats
    if not args.quiet:
        print(
            f"bench run  {target_file.name}"
            f"{f' {target_file.version}' if target_file.version else ''}"
            f"  →  {target.label}"
        )
        print(
            f"  {len(selected)} task(s) × {repeats} repeat(s) = {calls} call(s)"
            f"   concurrency {args.concurrency}"
        )
        if args.from_cassette:
            print(f"  replaying {args.from_cassette} (nothing is being spent)")
        if args.limit:
            print(f"  --limit {args.limit}: this run is not the whole split")

    # The label carries the version, because it is what ends up in the run id: two
    # builds of one agent with the same label produce the same hash, and a run id that
    # cannot tell v1 from v2 is a run id that makes `latest` a coin flip.
    label = f"{target_file.name} {target_file.version}" if target_file.version else target_file.name
    # The tasks directory is where a task's `corpus:` lives, so it is the root the scorer
    # loads corpora from. `--tasks` may name a single file, in which case the corpora are
    # beside it — the same convention `task validate` uses, for the same reason.
    tasks_root = Path(args.tasks) if Path(args.tasks).is_dir() else Path(args.tasks).parent
    scorer = build_scorer(root=tasks_root, judge=judge)
    result = asyncio.run(
        run_target(
            target,
            selected,
            repeats=repeats,
            concurrency=args.concurrency,
            scorer=scorer,
            label=label,
            run_budget_usd=Decimal(str(args.max_usd)) if args.max_usd else None,
            run_budget_seconds=args.max_seconds,
            on_task_done=None if args.quiet else _progress,
        )
    )
    if not args.quiet:
        print()

    if not args.no_write:
        path = save_run(args.runs_dir, result=result)
        if not args.quiet:
            print(f"  written to {path}")
    if not args.quiet:
        _print_run_summary(result)

    # Exit codes, and the reason there are two of them for bad news. A run where
    # nothing completed at all is not a score of zero, it is a run that failed, and a
    # gate that treats `2` as `0` is a gate nobody notices is broken. A partial run is
    # reported as `2` as well: its numbers are real but they are not the numbers anyone
    # asked for.
    if not result.completed:
        print("bench: no task produced a trace — the target is not answering", file=sys.stderr)
        return EXIT_RUN_FAILED
    if result.partial:
        print(f"bench: partial run — {result.partial_reason}", file=sys.stderr)
        return EXIT_RUN_FAILED
    return EXIT_OK


def _config_repeats(config: object) -> int:
    """Where `repeats` comes from when it is not on the command line.

    The target file may carry one; otherwise 1, and the run says in its notes that a
    single repeat cannot measure flakiness. Deliberately not 3: silently tripling
    somebody's bill is a worse default than a number that admits what it is.
    """
    value = getattr(config, "repeats", None)
    return int(value) if isinstance(value, int) and value >= 1 else 1


def _select(task_set, args: argparse.Namespace) -> list[Task]:
    """`--split`, `--family` and `--limit`, applied in that order.

    `--limit` takes the first N *after* the other two, so `--family grounded-qa --limit 5`
    is the first five grounded tasks rather than five tasks of which some are grounded.
    Ordering is by task id, so the same selection is the same set on two machines.
    """
    tasks = list(task_set.tasks)
    if args.split:
        tasks = [item for item in tasks if item.split == args.split and item.retired is None]
    else:
        tasks = [item for item in tasks if item.retired is None]
    if args.family:
        wanted = set(args.family)
        tasks = [item for item in tasks if item.family in wanted]
    tasks.sort(key=lambda item: item.id)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    return tasks


def _progress(item: object) -> None:
    """One line per task, ticked off as it lands.

    A callback rather than a print inside the runner: the CLI owns stdout, and a library
    that prints is a library that cannot be quiet. `\\r` rather than a newline so that
    three hundred tasks do not produce three hundred lines in a CI log.
    """
    outcome = getattr(item, "outcome", "?")
    task_id = getattr(item, "task_id", "?")
    mark = {"completed": "·", "error": "✗", "over-budget": "$", "skipped": "-"}.get(outcome, "?")
    print(f"\r  {mark} {task_id:<24}", end="", flush=True)


def _print_run_summary(result: object) -> None:
    """The terminal report, kept to what a person needs to decide whether to look
    further: what ran, what it cost, and what it scored.

    The score line prints the mean over the tasks that *could* be scored and names how
    many could not. A mean that swallowed the unscorable tasks as zeroes would read as an
    agent's failure when it is the harness's gap, and that is the one mistake this project
    does not get to make twice.
    """
    summary = result.summary()  # type: ignore[attr-defined]
    print(
        f"  {summary['completed']}/{summary['tasks']} completed"
        f"   {summary['errored']} errored"
        f"   {summary['over_budget']} over budget"
        f"   {summary['skipped']} skipped"
    )
    cost = summary["cost_usd"]
    spent = "not measured" if cost is None else f"${cost}"
    print(f"  wall {summary['wall_seconds']}s   cost {spent}")
    scores = getattr(result, "scores", [])
    if scores:
        print("  " + _score_line(scores))
    for note in result.notes:  # type: ignore[attr-defined]
        print(f"  · {note}")


def _score_line(scores: Sequence[TaskScore]) -> str:
    """`score 0.714 over 7 task(s)  5 passed  2 failed  1 not scorable  3 judged`.

    The judged count is on the line because the share of a score that rested on a model is
    the number a reader needs in order to decide how much to believe the result, and a
    score line without it is an advertisement.

    `outcome == "error"` tasks are counted separately rather than averaged in as zeroes:
    a task nothing could be scored on is a hole in the harness, and folding it into the
    mean would report it as a failure of the agent.
    """
    counted = [score for score in scores if score.outcome != "error"]
    errored = [score for score in scores if score.outcome == "error"]
    if not counted:
        return f"score — nothing could be scored ({len(errored)} task(s) unusable)"
    mean = sum(score.score for score in counted) / len(counted)
    passed = sum(1 for score in counted if score.outcome == "pass")
    judged = sum(1 for score in scores if score.judge_scored)
    untrusted = [score for score in scores if not score.judge_trusted]
    parts = [
        f"score {mean:.3f} over {len(counted)} task(s)",
        f"{passed} passed",
        f"{len(counted) - passed} failed",
    ]
    if errored:
        parts.append(f"{len(errored)} not scorable")
    if judged:
        parts.append(f"{judged} judged")
    if untrusted:
        parts.append(f"{len(untrusted)} with an untrusted judge — excluded")
    return "  ".join(parts)


# --- judge ----------------------------------------------------------------------

#: `bench judge calibrate` refuses to write a number nobody should trust: below this many
#: labels, agreement and kappa are anecdotes. The floor is published here rather than
#: buried in the calibration module because it is a claim about the board.
MIN_LABELS = 20


def _judge(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Calibrate the judge, or say what the calibration on file already claims.

    Separate from `run` on purpose. Calibrating is a research act — somebody sat down and
    labelled answers by hand — and running a benchmark is a routine one. A single verb
    that did both would make the routine path quietly re-measure its own ruler.
    """
    from .judge import JudgeClient, JudgeConfig, calibrate, read_labels, write_calibration

    if args.action == "status":
        config = JudgeConfig.from_file(args.judge)
        judge = JudgeClient(config)
        print(f"judge {config.model}  (κ floor {config.kappa_floor:.2f})")
        print(f"  {judge.trust_note()}")
        if judge.calibration:
            print(
                f"  labels {judge.calibration.get('labels')}   "
                f"agreement {judge.calibration.get('agreement')}   "
                f"κ {judge.calibration.get('kappa')}"
            )
        return EXIT_OK

    if args.action == "calibrate":
        config = JudgeConfig.from_file(args.judge)
        labels = read_labels(args.labels)
        if len(labels) < MIN_LABELS:
            print(
                f"bench: {len(labels)} label(s) is below the {MIN_LABELS} this harness needs "
                f"before it will call a judge calibrated",
                file=sys.stderr,
            )
            return EXIT_RUN_FAILED
        judge = JudgeClient(config)
        report = calibrate(judge, labels, labels_file=str(args.labels))
        print(f"judge {report.model}  on {report.labels} labelled answer(s)")
        print(f"  {report.summary()}")
        if report.wrong_ids:
            shown = ", ".join(report.wrong_ids[:10])
            more = f" … and {len(report.wrong_ids) - 10} more" if len(report.wrong_ids) > 10 else ""
            print(f"  wrong: {shown}{more}")
        spent = "not measured" if report.cost_usd is None else f"${report.cost_usd}"
        print(f"  calls {report.calls}   cache hits {report.cache_hits}   cost {spent}")
        if args.no_write:
            print("  not written (--no-write): this judge stays untrusted")
            return EXIT_OK
        path = write_calibration(report, config)
        print(f"  written to {path}")
        if not report.usable:
            # Exit 2, not 0: a calibration that says "do not trust this judge" is a
            # failure of the thing that was just measured, and a CI job that runs
            # `bench judge calibrate` should go red on it.
            print(f"bench: the judge is not usable — {report.summary()}", file=sys.stderr)
            return EXIT_RUN_FAILED
        return EXIT_OK

    parser.parse_args(["judge", "--help"])
    return EXIT_OK


# --- show ---------------------------------------------------------------------


def _show(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    path = resolve_run(args.runs_dir, args.reference)
    document = load_run(path)

    if args.json:
        print(json.dumps(document, indent=2))
        return EXIT_OK

    if args.task:
        traces = traces_from_document(document, args.task)
        print(json.dumps([json.loads(trace.model_dump_json()) for trace in traces], indent=2))
        return EXIT_OK

    summary = document.get("summary", {})
    print(f"{document['run_id']}   {document['label']}")
    print(
        f"  started {document['started_at']}   wall {document['wall_seconds']}s"
        f"   {document['repeats']} repeat(s)   concurrency {document['concurrency']}"
    )
    line = (
        f"  {summary.get('completed', 0)}/{summary.get('tasks', 0)} completed"
        f"   {summary.get('errored', 0)} errored"
        f"   {summary.get('over_budget', 0)} over budget"
    )
    if document.get("partial"):
        line += f"   PARTIAL — {document.get('partial_reason')}"
    print(line)
    if "truncated_traces" in document:
        print(f"  {document['truncated_traces']}")

    scored = [entry for entry in document.get("tasks", []) if "score" in entry]
    if scored:
        passed = sum(1 for entry in scored if entry.get("score_outcome") == "pass")
        mean = sum(entry["score"] for entry in scored) / len(scored)
        print(f"  score {mean:.3f} over {len(scored)} task(s)   {passed} passed")

    failures = [entry for entry in document.get("tasks", []) if entry.get("outcome") != "completed"]
    if failures:
        print("\n  not completed")
        for entry in failures[:20]:
            print(f"    {entry['task_id']:<24} {entry.get('error') or entry['outcome']}")
        if len(failures) > 20:
            print(f"    … and {len(failures) - 20} more")

    # The failure list: every task that did not pass, with the scorer's own sentence for
    # each failed expectation. This is the part of a leaderboard people actually read,
    # and it is why every check writes a note instead of a code.
    not_passed = [
        entry
        for entry in document.get("tasks", [])
        if entry.get("score_outcome") in {"fail", "error"} and entry.get("outcome") == "completed"
    ]
    for entry in not_passed[:20]:
        print(f"\n  {entry['task_id']}  {entry.get('score_outcome')}  score {entry['score']:.3f}")
        for note in entry.get("notes") or ["no notes: the scorer said nothing"]:
            print(f"    - {note}")
    if len(not_passed) > 20:
        print(f"\n  … and {len(not_passed) - 20} more task(s) that did not pass")
    for note in document.get("notes", []):
        print(f"  · {note}")
    return EXIT_OK


__all__ = ["build_parser", "main"]
