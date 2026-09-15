"""The command line as it stands: two verbs, and the exit codes that go with them.

Everything here is offline and touches no model. That is the point of wiring these
two verbs first — `task validate` is what somebody runs while writing a task, and it
has to work before anything else in the repository does.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import task, write_set

from bench.cli import main
from bench.errors import EXIT_OK, EXIT_RUN_FAILED


def test_version_prints_and_exits_zero(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "bench 0.1.0" in capsys.readouterr().out


def test_no_arguments_shows_help_rather_than_an_error(capsys: pytest.CaptureFixture[str]):
    """`bench` on its own is a person asking what this is. An argparse usage error is
    a bad answer to that question."""
    assert main([]) == EXIT_OK
    out = capsys.readouterr().out
    assert "usage: bench" in out
    assert "schema" in out and "task" in out


def test_an_unknown_command_is_an_argparse_error():
    """SystemExit(2), which is also `EXIT_RUN_FAILED` — the run could not finish.
    Pinned here so that a later refactor of the exit codes has to think about it."""
    with pytest.raises(SystemExit) as excinfo:
        main(["invent"])
    assert excinfo.value.code == EXIT_RUN_FAILED


# --- schema ------------------------------------------------------------------


def test_schema_print_emits_a_schema(capsys: pytest.CaptureFixture[str]):
    assert main(["schema", "print"]) == EXIT_OK
    document = json.loads(capsys.readouterr().out)
    assert document["title"] == "bench trace"
    assert "properties" in document


def test_schema_print_compact_is_one_line(capsys: pytest.CaptureFixture[str]):
    """For piping into `jq` in a pipeline, which is how a schema gets diffed against a
    checked-in copy in somebody else's repository."""
    assert main(["schema", "print", "--compact"]) == EXIT_OK
    out = capsys.readouterr().out.strip()
    assert "\n" not in out
    assert json.loads(out)["type"] == "object"


def test_schema_write_goes_where_it_is_told(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    target = tmp_path / "nested" / "trace.schema.json"
    assert main(["schema", "write", "--out", str(target)]) == EXIT_OK
    assert target.exists()
    assert str(target) in capsys.readouterr().out


def test_schema_write_regenerates_the_published_file(tmp_path: Path):
    """The verb the drift test in test_trace.py tells you to run."""
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    main(["schema", "write", "--out", str(first)])
    main(["schema", "write", "--out", str(second)])
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


# --- task validate -----------------------------------------------------------


def test_task_validate_reports_the_splits_and_the_digest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    root = tasks_in(
        tmp_path,
        task(id="gqa-0001"),
        task(id="traj-0001", family="tool-trajectory", split="private", tags=["tools"]),
    )
    assert main(["task", "validate", str(root)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "2 tasks: 1 public, 1 private" in out
    assert "grounded-qa 1p/0v" in out


def test_task_validate_fails_loudly_on_a_broken_set(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """Exit 2 and the reason on stderr, because this runs in a pre-commit hook and a
    silent pass is the failure mode that matters."""
    root = write_set(tmp_path, [task(expectations=[])])
    assert main(["task", "validate", str(root)]) == EXIT_RUN_FAILED
    err = capsys.readouterr().err
    assert err.startswith("bench: ")
    assert "expectations" in err


def test_task_validate_on_a_missing_directory_points_at_the_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    assert main(["task", "validate"]) == EXIT_RUN_FAILED
    assert "tasks" in capsys.readouterr().err


def test_task_lint_reports_near_duplicates_without_failing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """A finding for a person, not a gate: sometimes two tasks really are that
    similar, and the person who wrote them is the one who knows."""
    question = "why did the v2 rollout stop refunding orders after the deploy"
    root = write_set(
        tmp_path,
        [
            task(id="gqa-0001", input={"question": question}),
            task(
                id="gqa-0002",
                split="private",
                tags=["retrieval"],
                input={"question": question + " on tuesday"},
            ),
        ],
    )
    assert main(["task", "lint", str(root)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "near-duplicate pair(s)" in out
    assert "gqa-0002 ≈ gqa-0001" in out


def test_task_lint_says_so_when_the_splits_are_clean(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    root = write_set(
        tmp_path,
        [
            task(id="gqa-0001", input={"question": "a question about refund windows here"}),
            task(
                id="gqa-0002",
                split="private",
                tags=["retrieval"],
                input={"question": "how many tokens does the summariser use per call"},
            ),
        ],
    )
    assert main(["task", "lint", str(root)]) == EXIT_OK
    assert "no near-duplicates across the splits" in capsys.readouterr().out


def test_task_with_no_action_shows_its_own_help(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as excinfo:
        main(["task"])
    assert excinfo.value.code == EXIT_OK
    assert "validate" in capsys.readouterr().out


def tasks_in(tmp_path: Path, *entries: dict) -> Path:
    """Tasks in their own directory, with the target file beside it.

    Not the same directory: `load_tasks` reads every YAML under the path it is given, so
    a `target.yaml` sitting among the tasks gets parsed as one and fails on a missing
    `id`. That is the loader being honest, and it is also the layout everybody uses by
    the time they have more than two tasks.
    """
    directory = tmp_path / "tasks"
    write_set(directory, list(entries), name="set.yaml")
    return directory


# --- run ---------------------------------------------------------------------


def target_file(tmp_path: Path, *, callable: str = "fake_agents:echo", version: str = "v1") -> Path:
    path = tmp_path / "target.yaml"
    path.write_text(
        f"name: fixture\nversion: {version}\n"
        f"target:\n  kind: scripted\n  callable: {callable}\n  label: fixture\n",
        encoding="utf-8",
    )
    return path


def test_run_writes_a_run_file_and_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    root = tasks_in(tmp_path, *[task(id=f"gqa-{index:04d}") for index in range(3)])
    code = main(
        [
            "run",
            "--tasks",
            str(root),
            "--target",
            str(target_file(tmp_path)),
            "--repeats",
            "2",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--quiet",
        ]
    )
    assert code == EXIT_OK
    assert list((tmp_path / "runs").glob("*.run.json"))


def test_run_says_what_it_is_about_to_spend(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """The one command in this project that spends money. Three things are said out
    loud first: which target, how many tasks, and how many calls that is."""
    root = tasks_in(tmp_path, *[task(id=f"gqa-{index:04d}") for index in range(3)])
    main(
        [
            "run",
            "--tasks",
            str(root),
            "--target",
            str(target_file(tmp_path)),
            "--repeats",
            "2",
            "--runs-dir",
            str(tmp_path / "runs"),
        ]
    )
    out = capsys.readouterr().out
    assert "3 task(s) × 2 repeat(s) = 6 call(s)" in out
    assert "fixture" in out


def test_run_with_a_limit_says_the_run_is_not_the_whole_split(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """A limited run must never be mistaken for a full one: the bundle records it, and
    the terminal says it before the number is read."""
    root = tasks_in(tmp_path, *[task(id=f"gqa-{index:04d}") for index in range(4)])
    main(
        [
            "run",
            "--tasks",
            str(root),
            "--target",
            str(target_file(tmp_path)),
            "--limit",
            "2",
            "--runs-dir",
            str(tmp_path / "runs"),
        ]
    )
    out = capsys.readouterr().out
    assert "--limit 2: this run is not the whole split" in out


def test_a_run_where_nothing_completed_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """Not a score of zero — a run that failed. A gate that treats 2 as 0 is a gate
    nobody notices is broken."""
    root = tasks_in(tmp_path, task(id="gqa-0001"))
    code = main(
        [
            "run",
            "--tasks",
            str(root),
            "--target",
            str(target_file(tmp_path, callable="fake_agents:explode")),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--quiet",
        ]
    )
    assert code == EXIT_RUN_FAILED
    assert "the target is not answering" in capsys.readouterr().err


def test_a_partial_run_exits_two_and_says_why(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """The numbers are real but they are not the numbers anyone asked for."""
    root = tasks_in(
        tmp_path,
        *[
            task(id=f"gqa-{index:04d}", expectations=[{"type": "contains", "value": "x"}])
            for index in range(4)
        ],
    )
    path = tmp_path / "target.yaml"
    path.write_text(
        "name: fixture\ntarget:\n  kind: scripted\n  callable: fake_agents:costly\n",
        encoding="utf-8",
    )
    code = main(
        [
            "run",
            "--tasks",
            str(root),
            "--target",
            str(path),
            "--concurrency",
            "1",
            "--max-usd",
            "0.05",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--quiet",
        ]
    )
    assert code == EXIT_RUN_FAILED
    assert "partial run" in capsys.readouterr().err


def test_run_can_select_a_family_and_a_split(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    root = tasks_in(
        tmp_path,
        task(id="gqa-0001"),
        task(id="traj-0001", family="tool-trajectory", split="private", tags=["tools"]),
    )
    main(
        [
            "run",
            "--tasks",
            str(root),
            "--target",
            str(target_file(tmp_path)),
            "--family",
            "tool-trajectory",
            "--split",
            "private",
            "--runs-dir",
            str(tmp_path / "runs"),
        ]
    )
    assert "1 task(s) × 1 repeat(s) = 1 call(s)" in capsys.readouterr().out


def test_run_with_a_target_that_does_not_build_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    root = tasks_in(tmp_path, task(id="gqa-0001"))
    code = main(
        [
            "run",
            "--tasks",
            str(root),
            "--target",
            str(target_file(tmp_path, callable="nope.missing:answer")),
            "--quiet",
        ]
    )
    assert code == EXIT_RUN_FAILED
    assert "cannot import" in capsys.readouterr().err


def test_run_with_no_scorer_says_the_cost_is_not_measured(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """`not measured` rather than `$0.0000`: the difference between free and unpriced is
    the difference between two entries being comparable and not."""
    root = tasks_in(tmp_path, task(id="gqa-0001"))
    main(
        [
            "run",
            "--tasks",
            str(root),
            "--target",
            str(target_file(tmp_path)),
            "--runs-dir",
            str(tmp_path / "runs"),
        ]
    )
    assert "cost not measured" in capsys.readouterr().out


def test_run_prices_the_calls_when_the_target_config_has_rates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    root = tasks_in(tmp_path, task(id="gqa-0001"))
    path = tmp_path / "target.yaml"
    path.write_text(
        "name: fixture\n"
        "target:\n"
        "  kind: scripted\n"
        "  callable: fake_agents:as_dict\n"
        "  price_prompt_per_mtok: 3.0\n"
        "  price_completion_per_mtok: 15.0\n",
        encoding="utf-8",
    )
    main(["run", "--tasks", str(root), "--target", str(path), "--runs-dir", str(tmp_path / "runs")])
    assert "cost $0." in capsys.readouterr().out


def test_two_builds_of_one_target_get_different_run_ids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """The version goes into the run id. Without it, `latest` cannot tell v1 from v2 —
    which is the same silent-substitution bug the store's no-overwrite rule exists for."""
    root = tasks_in(tmp_path, task(id="gqa-0001"))
    runs = tmp_path / "runs"
    for version in ("v1", "v2"):
        path = tmp_path / f"target-{version}.yaml"
        path.write_text(
            f"name: fixture\nversion: {version}\n"
            f"target:\n  kind: scripted\n  callable: fake_agents:echo\n",
            encoding="utf-8",
        )
        main(
            ["run", "--tasks", str(root), "--target", str(path), "--runs-dir", str(runs), "--quiet"]
        )
    ids = {item.name.removeprefix("20").split("Z-")[1] for item in runs.glob("*.run.json")}
    assert len(ids) == 2


def test_run_records_a_cassette_then_replays_it_offline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    root = tasks_in(tmp_path, task(id="gqa-0001"))
    cassette = tmp_path / "main.json"
    runs = tmp_path / "runs"
    code = main(
        [
            "run",
            "--tasks",
            str(root),
            "--target",
            str(target_file(tmp_path)),
            "--record",
            str(cassette),
            "--runs-dir",
            str(runs),
        ]
    )
    assert code == EXIT_OK
    assert cassette.exists()

    code = main(
        [
            "run",
            "--tasks",
            str(root),
            "--target",
            str(target_file(tmp_path)),
            "--from-cassette",
            str(cassette),
            "--runs-dir",
            str(runs),
            "--quiet",
        ]
    )
    assert code == EXIT_OK
    assert len(list(runs.glob("*.run.json"))) == 2
    # the replay wrote a run file whose trace came from the cassette. Asking the store for
    # `latest` rather than sorting names: a name sort puts a `-2` duplicate before the
    # file it was copied from, which is the trap `list_runs` exists to avoid and not
    # something a test should rediscover.
    from bench.core.store import resolve_run

    document = json.loads(resolve_run(runs, "latest").read_text(encoding="utf-8"))
    assert document["tasks"][0]["traces"][0]["raw"].get("cassette") == cassette.name


# --- show --------------------------------------------------------------------


def write_run(tmp_path: Path, **overrides) -> Path:
    root = tasks_in(tmp_path, task(id="gqa-0001"))
    path = target_file(tmp_path)
    main(
        [
            "run",
            "--tasks",
            str(root),
            "--target",
            str(path),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--quiet",
        ]
    )
    return tmp_path / "runs"


def test_show_prints_the_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    runs = write_run(tmp_path)
    assert main(["show", "latest", "--runs-dir", str(runs)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "fixture v1" in out
    assert "1/1 completed" in out


def test_show_json_is_the_whole_document(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    runs = write_run(tmp_path)
    assert main(["show", "latest", "--runs-dir", str(runs), "--json"]) == EXIT_OK
    document = json.loads(capsys.readouterr().out)
    assert document["format"] == 1
    assert document["tasks"][0]["task_id"] == "gqa-0001"


def test_show_one_task_dumps_its_traces(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    runs = write_run(tmp_path)
    assert main(["show", "latest", "--runs-dir", str(runs), "--task", "gqa-0001"]) == EXIT_OK
    traces = json.loads(capsys.readouterr().out)
    assert len(traces) == 1
    assert traces[0]["latency_ms"] >= 0


def test_show_a_task_that_is_not_in_the_run_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    runs = write_run(tmp_path)
    assert (
        main(["show", "latest", "--runs-dir", str(runs), "--task", "gqa-9999"]) == EXIT_RUN_FAILED
    )
    assert "not in this run" in capsys.readouterr().err


def test_show_with_no_runs_explains_itself(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["show", "latest", "--runs-dir", str(tmp_path / "empty")]) == EXIT_RUN_FAILED
    assert "run something first" in capsys.readouterr().err


def test_show_lists_the_tasks_that_did_not_complete(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    root = tasks_in(tmp_path, task(id="gqa-0001"))
    main(
        [
            "run",
            "--tasks",
            str(root),
            "--target",
            str(target_file(tmp_path, callable="fake_agents:explode")),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--quiet",
        ]
    )
    main(["show", "latest", "--runs-dir", str(tmp_path / "runs")])
    out = capsys.readouterr().out
    assert "not completed" in out
    assert "retrieval index" in out


# --- scoring, wired into `run` and `show` -------------------------------------


def test_run_prints_a_score_line_because_that_is_what_the_harness_is_for(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """`fixture_agents:echo` answers with the task's own question, and the fixture task
    expects the word `permission` — which the question does not contain, so the score is a
    real number and a real failure rather than a hand-written expectation about output."""
    root = tasks_in(tmp_path, task(id="gqa-0001"))
    code = main(
        [
            "run",
            "--tasks",
            str(root),
            "--target",
            str(target_file(tmp_path)),
            "--runs-dir",
            str(tmp_path / "runs"),
        ]
    )
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "score 0.000 over 1 task(s)" in out
    assert "0 passed" in out and "1 failed" in out


def test_a_run_with_nothing_scorable_says_so_instead_of_printing_a_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """A judge-scored task with no judge is the case this exists for: the mean over an
    empty set has no value, and printing `0.000` would report the harness's own gap as
    the agent's failure."""
    root = tasks_in(
        tmp_path,
        task(
            id="gqa-0001",
            expectations=[{"type": "judge", "rubric": "names the cause"}],
        ),
    )
    main(
        [
            "run",
            "--tasks",
            str(root),
            "--target",
            str(target_file(tmp_path)),
            "--runs-dir",
            str(tmp_path / "runs"),
        ]
    )
    out = capsys.readouterr().out
    assert "nothing could be scored" in out


def test_run_with_a_judge_says_whether_to_trust_it_before_it_spends(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """The trust note is printed before the run, not after: whether the judged tasks count
    is the difference between a score and a partial score, and learning that from a
    footnote in a JSON file is too late to do anything about it."""
    root = tasks_in(tmp_path, task(id="gqa-0001"))
    judge_file = tmp_path / "judge.yaml"
    judge_file.write_text("model: fixture-judge\ncache_dir: null\n", encoding="utf-8")
    main(
        [
            "run",
            "--tasks",
            str(root),
            "--target",
            str(target_file(tmp_path)),
            "--judge",
            str(judge_file),
            "--runs-dir",
            str(tmp_path / "runs"),
        ]
    )
    out = capsys.readouterr().out
    assert "judge fixture-judge" in out
    assert "no calibration on file" in out


def test_show_prints_the_score_and_the_scorers_sentences(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """The failure list is the part of a leaderboard people actually read, so the CLI
    prints the notes rather than a code: `did not cite …` is actionable, `expected 2` is
    not."""
    root = tasks_in(tmp_path, task(id="gqa-0001"))
    runs = tmp_path / "runs"
    main(
        [
            "run",
            "--tasks",
            str(root),
            "--target",
            str(target_file(tmp_path)),
            "--runs-dir",
            str(runs),
            "--quiet",
        ]
    )
    assert main(["show", "latest", "--runs-dir", str(runs)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "score 0.000 over 1 task(s)   0 passed" in out
    assert "gqa-0001  fail  score 0.000" in out
    assert "the answer does not contain 'permission'" in out


# --- judge calibrate ----------------------------------------------------------


def labels_in(tmp_path: Path, count: int, name: str = "labels.jsonl") -> Path:
    path = tmp_path / name
    rows = [
        {
            "case_id": f"case-{index}",
            "rubric": "names the cause",
            "prompt": "why did the rollout stop?",
            "output": f"answer-{index}",
            "human": index % 2 == 0,
        }
        for index in range(count)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_calibrating_on_too_few_labels_refuses_and_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """Twenty is the floor, and it is a claim about the board rather than a taste: below
    it, the κ is an anecdote and a CI job that publishes it is publishing noise."""
    judge_file = tmp_path / "judge.yaml"
    judge_file.write_text("model: fixture-judge\ncache_dir: null\n", encoding="utf-8")
    code = main(
        [
            "judge",
            "calibrate",
            "--judge",
            str(judge_file),
            "--labels",
            str(labels_in(tmp_path, 3)),
        ]
    )
    assert code == EXIT_RUN_FAILED
    assert "is below the 20 this harness needs" in capsys.readouterr().err


def test_a_calibration_reports_kappa_and_writes_the_file_it_will_be_trusted_from(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    """The client is replaced, not the calibration: what this test is about is the CLI —
    the report it prints, the exit code, and the file it writes where the judge looks."""
    from bench.judge import JudgeClient, JudgeVerdict

    cache = tmp_path / "judge-cache"

    class Canned:
        def __init__(self, config: object) -> None:
            self.config = config
            self.calls = 0
            self.cache_hits = 0
            self.spent_usd = 0

        trusted = True

        def verdict(
            self,
            *,
            rubric: str,
            prompt: str,
            output: str,
            task_id: str = "",
            scale: int | None = None,
        ) -> JudgeVerdict:
            self.calls += 1
            # Right on every other label and wrong on two of the twenty-four: an
            # agreement of about 0.92 and a κ a real judge could plausibly produce, which
            # is the point — a fixture that agrees perfectly would not exercise the
            # report's arithmetic at all.
            index = int(task_id.split("-")[1])
            agreed = index % 2 == 0
            return JudgeVerdict(passed=not agreed if index in {1, 3} else agreed, reason="canned")

    monkeypatch.setattr("bench.judge.JudgeClient", Canned)
    assert JudgeClient is not Canned  # the module-level name is what the CLI imports
    judge_file = tmp_path / "judge.yaml"
    judge_file.write_text(
        f"model: fixture-judge\ncache_dir: {cache}\nkappa_floor: 0.10\n", encoding="utf-8"
    )
    code = main(
        [
            "judge",
            "calibrate",
            "--judge",
            str(judge_file),
            "--labels",
            str(labels_in(tmp_path, 24)),
        ]
    )
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "agreement" in out and "κ" in out
    assert (cache / "judge-calibration.json").exists()


def test_judge_status_reads_the_calibration_that_is_on_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    cache = tmp_path / "judge-cache"
    cache.mkdir()
    (cache / "judge-calibration.json").write_text(
        json.dumps(
            {"format": 1, "model": "fixture-judge", "kappa": 0.83, "labels": 40, "agreement": 0.88}
        ),
        encoding="utf-8",
    )
    judge_file = tmp_path / "judge.yaml"
    judge_file.write_text(f"model: fixture-judge\ncache_dir: {cache}\n", encoding="utf-8")
    assert main(["judge", "status", "--judge", str(judge_file)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "fixture-judge" in out
    assert "judge-scored tasks count" in out
    assert "κ 0.83" in out
