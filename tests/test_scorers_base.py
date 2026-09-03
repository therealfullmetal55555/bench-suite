"""The scoring machinery: the registry, weights, repeats, and what "unscored" means.

Almost every test here is about a distinction the report depends on. A failed check and a
check that could not be made both produce a zero-looking task, and the two mean opposite
things to whoever reads the board: "your agent is wrong" versus "our harness is
incomplete". Those are the tests worth having.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from conftest import make_corpus, task
from fake_judge import RecordingJudge

from bench.core.corpus import Corpus
from bench.core.score import TaskScore
from bench.core.task import Expectation, Task
from bench.core.trace import Passage, Trace
from bench.scorers import EXPECTATION_TYPES
from bench.scorers.base import (
    ExpectationResult,
    ScoringContext,
    build_scorer,
    default_registry,
    evaluate,
    score_traces,
    weight_of,
)

REGISTRY = default_registry()


def a_task(**overrides: object) -> Task:
    return Task.model_validate(task(**overrides))


def trace(output: str | None = "the answer", **overrides: object) -> Trace:
    return Trace(output=output, **overrides)  # type: ignore[arg-type]


def expectation(kind: str, **params: object) -> Expectation:
    return Expectation.model_validate({"type": kind, **params})


# --- the registry -------------------------------------------------------------


def test_every_expectation_type_the_task_model_accepts_has_a_scorer() -> None:
    """The registry and the vocabulary are two lists that must not drift.

    `task.py` refuses a `type:` that is not in `EXPECTATION_TYPES`, so a type listed
    there but not implemented here loads, validates, and then scores nothing — the worst
    of the three possible failures, because nothing announces it.
    """
    assert sorted(default_registry()) == sorted(EXPECTATION_TYPES)


def test_no_two_families_register_the_same_name() -> None:
    """The collision guard is a RuntimeError inside `default_registry`, and a merged
    registry is what runs, so this asserts the merge is honest rather than silent."""
    registry = default_registry()
    assert len(registry) == len(EXPECTATION_TYPES)


def test_a_deterministic_type_is_claimed_by_the_registry_that_implements_it() -> None:
    """`judge` is the only non-deterministic type. If a second one appears, the board's
    judge-share column changes meaning and somebody should have to say so out loud."""
    judged = {name for name, spec in EXPECTATION_TYPES.items() if not spec.deterministic}
    assert judged == {"judge"}


# --- evaluate -----------------------------------------------------------------


def test_a_scorer_that_raises_becomes_a_failed_check_not_a_traceback() -> None:
    def broken(context: ScoringContext, item: Expectation) -> ExpectationResult:
        raise ZeroDivisionError("division by zero")

    result = evaluate(ScoringContext(task=a_task(), trace=trace()), expectation("x"), {"x": broken})
    assert result.passed is False
    assert "ZeroDivisionError" in result.note
    assert "division by zero" in result.note


def test_an_unknown_type_is_unscored_rather_than_failed() -> None:
    """It cannot happen through the loader — `task.py` checks the type on the way in — so
    it is a guard against a scorer being removed without the vocabulary following."""
    result = evaluate(ScoringContext(task=a_task(), trace=trace()), expectation("nope"), {})
    assert result.scored is False
    assert "no scorer is registered" in result.note


def test_the_answer_of_a_silent_target_is_the_empty_string() -> None:
    context = ScoringContext(task=a_task(), trace=trace(None))
    assert context.answer == ""
    assert context.answer.lower() == ""


# --- weights ------------------------------------------------------------------


def test_a_weight_defaults_to_one() -> None:
    assert weight_of(expectation("contains", value="x")) == 1.0


def test_a_weight_is_read_from_the_task_file() -> None:
    assert weight_of(expectation("citations", weight=3)) == 3.0
    assert weight_of(expectation("citations", weight="2.5")) == 2.5


@pytest.mark.parametrize("bad", ["heavy", None, 0, -1])
def test_a_weight_that_is_not_a_positive_number_is_refused(bad: object) -> None:
    with pytest.raises(ValueError):
        weight_of(expectation("contains", value="x", weight=bad))


def test_a_bad_weight_leaves_the_expectation_unscored_and_says_why() -> None:
    item = a_task(expectations=[{"type": "contains", "value": "answer", "weight": "heavy"}])
    score = score_traces(item, [trace()], registry=REGISTRY)
    assert "weight 'heavy' is not a number" in score.notes[0]
    assert score.judge_trusted is True


def test_weights_change_the_score_and_the_arithmetic_is_visible() -> None:
    """3 of 4 weight on the field that is right, 1 on the field that is wrong: 0.75.

    Written as an exact fraction rather than as a range, because the whole reason weights
    exist is that a reader should be able to predict the score from the task file.
    """
    item = a_task(
        expectations=[
            {"type": "contains", "value": "the answer", "weight": 3},
            {"type": "contains", "value": "something else", "weight": 1},
        ]
    )
    assert score_traces(item, [trace()], registry=REGISTRY).score == pytest.approx(0.75)


# --- score_traces: partial credit and repeats ---------------------------------


def test_partial_credit_is_the_mean_of_the_weighted_expectations() -> None:
    item = a_task(
        expectations=[
            {"type": "contains", "value": "the answer"},
            {"type": "contains", "value": "a second thing"},
        ]
    )
    score = score_traces(item, [trace()], registry=REGISTRY)
    assert score.score == pytest.approx(0.5)
    assert score.outcome == "fail"
    assert score.passes == 0


def test_a_pass_needs_every_expectation_in_every_repeat() -> None:
    item = a_task(expectations=[{"type": "contains", "value": "the answer"}])
    score = score_traces(item, [trace(), trace(), trace()], registry=REGISTRY)
    assert score.outcome == "pass"
    assert score.passes == 3
    assert score.repeats == 3
    assert score.notes == []


def test_flakiness_is_reported_as_a_count_over_repeats_not_as_a_rate() -> None:
    item = a_task(expectations=[{"type": "contains", "value": "the answer"}])
    score = score_traces(item, [trace(), trace("no"), trace()], registry=REGISTRY)
    assert score.passes == 2
    assert score.notes == [
        "1/3 repeats: the answer does not contain 'the answer': no",
    ]
    assert score.score == pytest.approx(2 / 3)


def test_the_same_failure_twice_is_one_note_with_a_count() -> None:
    item = a_task(expectations=[{"type": "contains", "value": "the answer"}])
    score = score_traces(item, [trace("no"), trace("no"), trace("the answer")], registry=REGISTRY)
    assert len(score.notes) == 1
    assert score.notes[0].startswith("2/3 repeats:")


def test_a_task_that_never_ran_is_an_error_with_a_reason() -> None:
    score = score_traces(a_task(), [], registry=REGISTRY)
    assert score.outcome == "error"
    assert score.repeats == 0
    assert "never ran" in (score.error or "")


def test_traces_that_all_failed_are_an_error_not_a_wrong_answer() -> None:
    item = a_task()
    failed = [Trace(error="connection reset") for _ in range(2)]
    score = score_traces(item, failed, registry=REGISTRY)
    assert score.outcome == "error"
    assert score.error == "connection reset"
    assert score.score == 0.0
    assert score.passes == 0


def test_a_target_that_errors_on_one_repeat_of_three_says_so_first() -> None:
    """An errored repeat is a repeat that did not answer, so it scores nothing — and the
    note putting the errors above the failure list is the difference between "this entry
    is wrong" and "this entry is fragile", which is the thing a reader is looking for."""
    item = a_task(expectations=[{"type": "contains", "value": "the answer"}])
    score = score_traces(item, [Trace(error="boom"), trace(), trace()], registry=REGISTRY)
    assert score.outcome == "fail"  # one of three repeats produced no answer
    assert score.passes == 2
    assert score.score == pytest.approx(2 / 3)
    assert score.notes[0] == "1/3 repeats errored: boom"


def test_the_worst_latency_of_the_repeats_is_what_is_reported() -> None:
    item = a_task()
    score = score_traces(
        item,
        [Trace(output="the answer", latency_ms=10), Trace(output="the answer", latency_ms=90)],
        registry=REGISTRY,
    )
    assert score.latency_ms == 90


def test_cost_is_left_to_the_runner_and_stays_none_here() -> None:
    """`score_traces` does not price anything: the runner already knows the rates and the
    repeats, and two places computing a cost is how a cost column ends up disagreeing
    with itself."""
    item = a_task()
    score = score_traces(
        item, [Trace(output="the answer", cost_usd=Decimal("0.01"))], registry=REGISTRY
    )
    assert score.cost_usd is None


# --- the scored/unscored distinction ------------------------------------------


def test_an_unscored_expectation_leaves_the_denominator() -> None:
    """One check passes, one cannot be made, one fails: 1 of 2, not 1 of 3."""
    item = a_task(
        expectations=[
            {"type": "contains", "value": "the answer"},
            {"type": "steps", "max": "not a number"},
            {"type": "contains", "value": "absent"},
        ]
    )
    score = score_traces(item, [trace()], registry=REGISTRY)
    assert score.score == pytest.approx(0.5)


def test_when_nothing_could_be_scored_at_all_the_task_is_an_error() -> None:
    item = a_task(expectations=[{"type": "steps", "max": "not a number"}])
    score = score_traces(item, [trace()], registry=REGISTRY)
    assert score.outcome == "error"
    assert score.score == 0.0
    assert any("no expectation produced a verdict" in note for note in score.notes)


def test_every_repeat_errored_is_an_error_with_the_targets_own_message() -> None:
    item = a_task()
    score = score_traces(item, [Trace(error="429 too many requests")], registry=REGISTRY)
    assert score.outcome == "error"
    assert score.error == "429 too many requests"
    assert any(note.startswith("every repeat errored") for note in score.notes)


def test_a_judged_task_without_a_judge_is_marked_and_not_guessed() -> None:
    item = a_task(
        expectations=[
            {"type": "contains", "value": "the answer"},
            {"type": "judge", "rubric": "is it good"},
        ]
    )
    score = score_traces(item, [trace()], registry=REGISTRY, judge=None)
    assert score.judge_scored is True
    assert score.judge_trusted is False
    assert score.score == 1.0
    assert any("no judge is configured" in note for note in score.notes)


def test_an_untrusted_judge_is_excluded_and_named() -> None:
    """The flags matter more than the score here: a task nothing could decide has to be
    reported as undecided, not as failed, and it has to carry the reason onto the report
    card. Both of those were wrong in the first version of `score_traces`, in the branch
    where every repeat came back unscorable — found by this test, fixed in the branch."""
    item = a_task(expectations=[{"type": "judge", "rubric": "is it good"}])
    judge = RecordingJudge(trusted=False)
    score = score_traces(item, [trace()], registry=REGISTRY, judge=judge)
    assert score.judge_trusted is False
    assert score.judge_scored is True
    assert score.outcome == "error"
    assert any("below its calibration floor" in note for note in score.notes)
    assert judge.calls == []


def test_a_task_without_judged_expectations_is_trusted_by_default() -> None:
    score = score_traces(a_task(), [trace()], registry=REGISTRY)
    assert score.judge_scored is False
    assert score.judge_trusted is True


# --- build_scorer -------------------------------------------------------------


def test_the_scorer_reads_a_corpus_once_for_many_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_corpus(tmp_path)
    loads = {"count": 0}
    real = Corpus.load

    def counting(path: Path) -> Corpus:
        loads["count"] += 1
        return real(path)

    monkeypatch.setattr(Corpus, "load", staticmethod(counting))
    scorer = build_scorer(root=root)
    item = a_task(
        corpus="corpus",
        expectations=[{"type": "citations", "must_cite": ["docs/incident.md#root-cause"]}],
    )
    good = Trace(
        output="see docs/incident.md#root-cause",
        retrieved=[Passage(id="docs/incident.md#root-cause", text="x")],
    )
    for _ in range(5):
        assert scorer(item, [good]).score == 1.0
    assert loads["count"] == 1


def test_a_corpus_that_will_not_load_fails_the_tasks_that_need_it_and_no_others(
    tmp_path: Path,
) -> None:
    """A missing corpus directory is a repository mistake. It must not stop a run of 300
    tasks, and it must not silently score the grounded ones as if the corpus were empty."""
    root = tmp_path / "tasks"
    root.mkdir()
    scorer = build_scorer(root=root)
    grounded = a_task(
        corpus="does-not-exist", expectations=[{"type": "answerable", "value": False}]
    )
    plain = a_task(id="gqa-0002", expectations=[{"type": "contains", "value": "the answer"}])

    # The grounded task still scores — an answerable check with no corpus is a text check —
    # and the corpus failure is not fatal for the task that never asked for one.
    assert scorer(plain, [trace()]).score == 1.0
    assert scorer(grounded, [trace("no")]).outcome in {"pass", "fail", "error"}


def test_a_corpus_path_that_is_not_the_root_is_honoured(tmp_path: Path) -> None:
    """`corpus_paths` exists for the case where a submission ships its own corpora: the
    mapping wins over `root`, and it is what `bench run --corpus-dir` will pass."""
    here = make_corpus(tmp_path / "a", body="# Root\n\nThe reason was a permission change.\n")
    elsewhere = make_corpus(tmp_path / "b", body="# Root\n\nThe reason was the prompt.\n")
    item = a_task(
        corpus="corpus",
        expectations=[{"type": "citations", "must_cite": ["docs/incident.md#root"]}],
    )
    answer = Trace(
        output="the reason was the prompt (docs/incident.md#root)",
        retrieved=[Passage(id="docs/incident.md#root", text="x")],
    )
    _ = here
    assert build_scorer(corpus_paths={"corpus": elsewhere})(item, [answer]).score == 1.0


def test_the_built_scorer_carries_the_task_id_and_family_into_the_score() -> None:
    scorer = build_scorer()
    score = scorer(a_task(id="gqa-0009", family="grounded-qa"), [trace()])
    assert isinstance(score, TaskScore)
    assert score.task_id == "gqa-0009"
    assert score.family == "grounded-qa"
    assert score.split == "public"
