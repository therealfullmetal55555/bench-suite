"""The checks themselves, one family at a time.

Every test here runs through `evaluate` with the real registry rather than calling a check
function directly. Slower to write, and it is the only way these tests catch the bugs that
matter: a check that is registered under the wrong name, or one whose `params` the task
model drops on the floor, would pass a direct call and fail in a run.

The traces are built inline. A 300-task fixture would hide which line of the trace made a
check pass, and the point of a scorer test is that a reader can see the input that produced
the verdict.
"""

from __future__ import annotations

from typing import Any

import pytest
from conftest import make_corpus, task
from fake_judge import RecordingJudge

from bench.core.corpus import Corpus
from bench.core.task import Expectation, Task
from bench.core.trace import Passage, ToolCall, Trace
from bench.scorers.base import (
    ExpectationResult,
    ScoringContext,
    default_registry,
    evaluate,
)

REGISTRY = default_registry()
QUESTION = {"question": "why did the rollout stop?"}


def check(
    kind: str,
    trace: Trace,
    *,
    corpus: Corpus | None = None,
    judge: Any = None,
    task_input: dict[str, Any] | None = None,
    **params: Any,
) -> ExpectationResult:
    """Run one expectation against one trace, exactly as a run would."""
    item = Task.model_validate(task(input=task_input or QUESTION))
    expectation = Expectation.model_validate({"type": kind, **params})
    return evaluate(
        ScoringContext(task=item, trace=trace, corpus=corpus, judge=judge),
        expectation,
        REGISTRY,
    )


def answer(text: str | None, **overrides: Any) -> Trace:
    return Trace(output=text, **overrides)


def calls(*names: str, **overrides: Any) -> Trace:
    return Trace(output="done", tool_calls=[ToolCall(name=name) for name in names], **overrides)


@pytest.fixture
def corpus(tmp_path: Any) -> Corpus:
    root = make_corpus(
        tmp_path,
        body=(
            "# Incident 2026-08\n\n"
            "Refunds stopped on a Tuesday.\n\n"
            "## Root cause\n\n"
            "A permission change removed the billing scope from the worker's role.\n\n"
            "## What we tried\n\n"
            "The prompt was changed first, and it was not the cause.\n"
        ),
    )
    return Corpus.load(root)


# --- deterministic: contains, not_contains, regex ------------------------------


def test_contains_ignores_case_and_finds_the_phrase_anywhere() -> None:
    assert check("contains", answer("The Permission change did it"), value="permission").passed


def test_contains_says_what_it_looked_for_and_what_it_got() -> None:
    result = check("contains", answer("the prompt"), value="permission")
    assert not result.passed
    assert "permission" in result.note and "the prompt" in result.note


def test_contains_on_an_empty_answer_names_the_emptiness() -> None:
    result = check("contains", answer(None), value="permission")
    assert "(empty answer)" in result.note


def test_not_contains_is_a_hard_rule_even_when_the_sentence_is_a_denial() -> None:
    """Deliberate, and documented in the module: "I will not say the prompt was at fault"
    contains "the prompt was at fault". A task that wants that nuance uses a judge
    expectation; a substring check that quietly reinterprets the sentence is worse than
    one a task author has to work around."""
    result = check(
        "not_contains",
        answer("I will not claim the prompt was at fault"),
        value="the prompt was at fault",
    )
    assert not result.passed


def test_regex_finds_a_pattern_across_lines() -> None:
    trace = answer("Root cause:\npermissions.\n")
    assert check(
        "regex",
        trace,
        pattern=r"^root cause:\s*\n\s*permissions",
    ).passed


def test_a_pattern_that_does_not_compile_is_unscored_not_failed() -> None:
    result = check("regex", answer("x"), pattern="([unclosed")
    assert result.scored is False
    assert "does not compile" in result.note


def test_a_missing_parameter_is_a_broken_check_not_a_failed_answer() -> None:
    """`_required` raises, and `evaluate` turns it into a failed check with the exception
    in the note. It is the difference between a task file typo and a wrong agent."""
    result = check("contains", answer("x"))
    assert not result.passed
    assert "needs `value`" in result.note


# --- deterministic: json and schema -------------------------------------------


def test_json_parses_a_plain_object() -> None:
    assert check("json", answer('{"total": 12}')).passed


def test_json_names_the_keys_it_did_not_find() -> None:
    result = check("json", answer('{"total": 12}'), keys=["total", "currency"])
    assert not result.passed
    assert "currency" in result.note


def test_json_equals_compares_numbers_loosely_and_booleans_strictly() -> None:
    """`77.5` and `"77.50"` are the same total; `true` and `1` are not the same boolean.
    The second half of that rule is the one that matters — Python says they are equal."""
    assert check("json", answer('{"total": "77.50"}'), equals={"total": 77.5}).passed
    result = check("json", answer('{"refunded": 1}'), equals={"refunded": True})
    assert not result.passed


def test_json_refuses_a_markdown_fence_unless_the_task_allows_one() -> None:
    fenced = answer('```json\n{"total": 12}\n```')
    assert not check("json", fenced).passed
    assert check("json", fenced, fenced=True).passed


def test_schema_validates_with_a_real_validator() -> None:
    schema = {
        "type": "object",
        "required": ["total", "currency"],
        "properties": {"total": {"type": "number"}, "currency": {"enum": ["EUR", "USD"]}},
        "additionalProperties": False,
    }
    assert check("schema", answer('{"total": 12.5, "currency": "EUR"}'), schema=schema).passed
    result = check("schema", answer('{"total": 12.5, "currency": "GBP"}'), schema=schema)
    assert not result.passed
    assert "currency" in result.note


def test_schema_reports_the_path_of_the_first_error_and_how_many_follow() -> None:
    schema = {
        "type": "object",
        "properties": {"lines": {"type": "array", "items": {"type": "object"}}},
    }
    result = check("schema", answer('{"lines": [1, 2]}'), schema=schema)
    assert result.note.startswith("lines/0:")
    assert "+1 more" in result.note


def test_schema_without_a_schema_object_is_unscored() -> None:
    result = check(
        "schema",
        answer("{}"),
    )
    assert result.scored is False
    assert "no `schema` object" in result.note


# --- trajectory: tools, order, budget -----------------------------------------


def test_required_tools_must_appear_but_not_in_any_particular_order() -> None:
    trace = calls("search_docs", "get_policy")
    assert check("trajectory", trace, required_tools=["search_docs", "get_policy"]).passed


def test_a_missing_required_tool_is_named_with_what_was_called() -> None:
    result = check("trajectory", calls("search_docs"), required_tools=["search_docs", "get_policy"])
    assert not result.passed
    assert "missing required tool(s) get_policy" in result.note
    assert "called: search_docs" in result.note


def test_a_forbidden_tool_fails_the_check() -> None:
    result = check(
        "trajectory", calls("search_docs", "issue_refund"), forbidden_tools=["issue_refund"]
    )
    assert not result.passed
    assert "used forbidden tool(s) issue_refund" in result.note


def test_ordered_tools_are_checked_as_a_subsequence() -> None:
    """Extra calls in between are fine — the check is that the two happened in this order,
    not that nothing else happened, because the alternative fails an agent for logging."""
    assert check(
        "trajectory",
        calls("search_docs", "log", "issue_refund"),
        ordered=["search_docs", "issue_refund"],
    ).passed
    result = check(
        "trajectory", calls("issue_refund", "search_docs"), ordered=["search_docs", "issue_refund"]
    )
    assert not result.passed
    assert "wrong order" in result.note


def test_no_tool_calls_at_all_says_so_once_instead_of_listing_everything_missing() -> None:
    result = check(
        "trajectory", answer("I know the answer"), required_tools=["search_docs", "get_policy"]
    )
    assert not result.passed
    assert result.note == (
        "the trace made no tool calls at all, and the task requires search_docs, get_policy"
    )


def test_the_step_budget_counts_tool_calls_when_the_target_reports_no_step_count() -> None:
    assert check("trajectory", calls("a", "b"), max_steps=2).passed
    result = check("trajectory", calls("a", "b", "c"), max_steps=2)
    assert not result.passed
    assert "3 steps over the budget of 2" in result.note


def test_an_explicit_step_count_wins_over_the_counted_tool_calls() -> None:
    """The target knows its own turns; the harness knows how many tool calls it was told
    about. Where they disagree the target is the authority, and the disagreement is the
    integration's business rather than something to score."""
    trace = Trace(output="done", steps=9, tool_calls=[ToolCall(name="a")])
    result = check("trajectory", trace, max_steps=2)
    assert "9 steps over the budget of 2" in result.note


def test_a_step_budget_is_reported_but_not_enforced_when_nothing_reports_steps() -> None:
    """A target that reports neither a step count nor a tool call is silent, and silence
    is not scored — the note says the check did not happen."""
    result = check("trajectory", answer("done"), max_steps=3, required_tools=[])
    assert result.passed
    assert "step budget not checked" in result.note
    assert result.detail["steps"] is None


def test_the_steps_expectation_on_its_own() -> None:
    assert check("steps", calls("a", "b"), max=2).passed
    result = check("steps", answer("no tools at all"), max=2)
    assert result.scored is False
    assert "reports no step count" in result.note


# --- state: what the calls did, not which ones ran -----------------------------


def test_a_state_transition_matches_on_a_subset_of_arguments() -> None:
    trace = Trace(
        output="done",
        tool_calls=[
            ToolCall(
                name="issue_refund",
                arguments={"order_id": "1041", "amount": 49.5, "idempotency_key": "x"},
            ),
        ],
    )
    result = check("state", trace, applied=[{"tool": "issue_refund", "args": {"order_id": "1041"}}])
    assert result.passed
    assert "issue_refund" in result.note


def test_arguments_are_compared_with_the_extraction_familys_looseness() -> None:
    trace = Trace(
        output="done", tool_calls=[ToolCall(name="issue_refund", arguments={"amount": "49.50"})]
    )
    assert check(
        "state", trace, applied=[{"tool": "issue_refund", "args": {"amount": 49.5}}]
    ).passed


def test_the_right_tool_on_the_wrong_order_is_a_failure() -> None:
    """The expensive mistake this expectation exists for: the agent refunded somebody
    else's order. A tool-name check would call that a pass."""
    trace = Trace(
        output="done", tool_calls=[ToolCall(name="issue_refund", arguments={"order_id": "1042"})]
    )
    result = check("state", trace, applied=[{"tool": "issue_refund", "args": {"order_id": "1041"}}])
    assert not result.passed
    assert "never called issue_refund(order_id='1041')" in result.note


def test_times_is_exact_and_catches_a_double_refund() -> None:
    trace = Trace(
        output="done",
        tool_calls=[
            ToolCall(name="issue_refund", arguments={"order_id": "1041"}),
            ToolCall(name="issue_refund", arguments={"order_id": "1041"}),
        ],
    )
    result = check("state", trace, applied=[{"tool": "issue_refund", "times": 1}])
    assert not result.passed
    assert "2 time(s), and the task says 1" in result.note


def test_untouched_is_written_from_the_tasks_point_of_view() -> None:
    trace = Trace(
        output="done", tool_calls=[ToolCall(name="cancel_order", arguments={"order_id": "1042"})]
    )
    result = check(
        "state", trace, untouched=[{"tool": "cancel_order", "args": {"order_id": "1042"}}]
    )
    assert not result.passed
    assert "which this task leaves alone" in result.note


def test_a_state_spec_with_no_tool_in_it_is_a_broken_task_not_a_bad_agent() -> None:
    result = check("state", calls("issue_refund"), applied=[{"args": {"order_id": "1041"}}])
    assert result.scored is False
    assert "has no `tool` in it" in result.note


def test_a_state_expectation_that_checks_nothing_is_refused() -> None:
    result = check("state", calls("issue_refund"))
    assert result.scored is False
    assert "checks nothing" in result.note


# --- citations ----------------------------------------------------------------


def test_a_required_citation_is_satisfied_by_the_full_id(corpus: Corpus) -> None:
    trace = Trace(
        output="A permission change (docs/incident.md#root-cause)",
        retrieved=[Passage(id="docs/incident.md#root-cause", text="x")],
    )
    result = check("citations", trace, corpus=corpus, must_cite=["docs/incident.md#root-cause"])
    assert result.passed
    assert "recall 100%" in result.note


def test_a_bare_anchor_counts_as_a_citation(corpus: Corpus) -> None:
    trace = answer("Refunds stopped after a permission change (#root-cause).")
    assert check(
        "citations", trace, corpus=corpus, must_cite=["docs/incident.md#root-cause"]
    ).passed


def test_an_anchor_on_a_different_document_does_not_count() -> None:
    """The regression this check was rewritten for: with anchor-only matching, citing
    `support/policy.md#root-cause` satisfied a `must_cite` for `docs/incident.md#root-cause`,
    which is the exact failure the check exists to catch — reported as a pass."""
    trace = answer("See support/policy.md#root-cause.")
    result = check("citations", trace, must_cite=["docs/incident.md#root-cause"])
    assert not result.passed
    assert "did not cite" in result.note


def test_a_citation_the_corpus_does_not_have_is_caught(corpus: Corpus) -> None:
    trace = answer("As docs/handbook.md#refunds says, refunds stopped.")
    result = check("citations", trace, corpus=corpus, must_not_cite_unsupported=True)
    assert not result.passed
    assert "the corpus does not have" in result.note


def test_citing_something_that_was_never_retrieved_is_flagged() -> None:
    trace = Trace(
        output="per the docs (#root-cause)",
        retrieved=[Passage(id="docs/incident.md#what-we-tried", text="x")],
    )
    result = check(
        "citations",
        trace,
        must_cite=["docs/incident.md#root-cause"],
        must_not_cite_unsupported=True,
    )
    assert not result.passed
    assert "cited without retrieving" in result.note


def test_a_target_that_reports_no_retrieval_is_not_accused_of_lying() -> None:
    trace = answer("per the docs (#root-cause)")
    assert check(
        "citations",
        trace,
        must_cite=["docs/incident.md#root-cause"],
        must_not_cite_unsupported=True,
    ).passed


def test_the_note_reports_precision_and_recall_separately() -> None:
    """They fail for different reasons: recall is "you did not cite what you should have",
    precision is "you cited something else". One number would hide which happened."""
    trace = Trace(
        output="(#root-cause) and also (docs/incident.md#what-we-tried)",
        retrieved=[Passage(id="docs/incident.md#root-cause", text="x")],
    )
    result = check("citations", trace, corpus=None, must_cite=["docs/incident.md#root-cause"])
    assert result.passed
    assert result.detail["recall"] == 1.0


# --- answerable ---------------------------------------------------------------


def test_an_unanswerable_question_passes_when_the_agent_declines() -> None:
    result = check("answerable", answer("The documentation does not cover that."), value=False)
    assert result.passed
    assert "correctly declined" in result.note


def test_an_unanswerable_question_fails_when_the_agent_invents_an_answer() -> None:
    result = check("answerable", answer("The rollout stopped because of the prompt."), value=False)
    assert not result.passed
    assert "answered anyway" in result.note


def test_an_answerable_question_fails_when_the_agent_refuses() -> None:
    """The direction that keeps the benchmark from rewarding an agent that hedges
    everything: refusing a question the corpus does answer is a failure, not caution."""
    result = check("answerable", answer("I don't know."), value=True)
    assert not result.passed
    assert "the agent declined" in result.note


def test_the_refusal_list_is_shared_with_the_safety_family() -> None:
    from bench.scorers.citations import REFUSALS
    from bench.scorers.safety import check_refusal

    assert "does not cover" in REFUSALS
    assert check_refusal is not None  # imported from the same module, not copied


def test_an_answerable_expectation_needs_a_boolean() -> None:
    result = check("answerable", answer("x"), value="yes")
    assert result.scored is False


# --- structured extraction ----------------------------------------------------


def invoice(**overrides: Any) -> str:
    import json

    document = {"vendor": "Northwind Tools", "total": 77.5, "currency": "EUR", "lines": ["a", "b"]}
    document.update(overrides)
    return json.dumps(document)


def test_every_field_right_is_a_pass_with_the_count_in_the_note() -> None:
    result = check("fields", answer(invoice()), values={"vendor": "Northwind Tools", "total": 77.5})
    assert result.passed
    assert "all 2 field(s) right" in result.note


def test_partial_credit_is_the_share_of_weighted_fields() -> None:
    result = check(
        "fields",
        answer(invoice(total=70)),
        values={"vendor": "Northwind Tools", "total": 77.5, "currency": "EUR"},
    )
    assert not result.passed
    assert result.detail["share"] == pytest.approx(2 / 3)
    assert "total: expected 77.5, got 70" in result.note


def test_a_missing_field_scores_zero_rather_than_leaving_the_denominator() -> None:
    """An extraction that omits a field has failed at extraction. Dropping it from the
    score would let a two-field answer outscore a four-field one."""
    result = check(
        "fields",
        answer('{"vendor": "Northwind Tools"}'),
        values={"vendor": "Northwind Tools", "total": 77.5},
    )
    assert result.detail["share"] == pytest.approx(0.5)
    assert "missing field(s) total" in result.note


def test_a_null_field_counts_as_missing() -> None:
    result = check("fields", answer(invoice(total=None)), values={"total": 77.5})
    assert "missing field(s) total" in result.note


def test_weights_let_a_task_say_which_field_matters() -> None:
    result = check(
        "fields",
        answer(invoice(total=70)),
        values={"total": 77.5, "currency": "EUR"},
        weights={"total": 3},
    )
    assert result.detail["share"] == pytest.approx(0.25)


def test_strings_are_compared_with_case_and_whitespace_folded() -> None:
    result = check(
        "fields", answer(invoice(vendor="northwind   TOOLS")), values={"vendor": "Northwind Tools"}
    )
    assert result.passed


def test_lists_are_order_sensitive_unless_the_task_says_a_set() -> None:
    ordered = {"lines": ["a", "b"]}
    assert check("fields", answer(invoice()), values=ordered).passed
    assert not check("fields", answer(invoice(lines=["b", "a"])), values=ordered).passed
    assert check(
        "fields", answer(invoice(lines=["b", "a"])), values=ordered, normalise=["set"]
    ).passed


def test_an_answer_that_is_not_json_fails_with_a_quote_of_what_it_was() -> None:
    result = check("fields", answer("The total is 77.50 EUR."), values={"total": 77.5})
    assert not result.passed
    assert "is not JSON" in result.note


def test_an_empty_answer_is_not_a_pass_with_no_fields() -> None:
    result = check("fields", answer(None), values={"total": 77.5})
    assert "the answer is empty" in result.note


def test_a_fields_expectation_without_values_is_a_broken_task() -> None:
    result = check("fields", answer(invoice()))
    assert result.scored is False
    assert "needs a `values:` mapping" in result.note


# --- refusal and safety -------------------------------------------------------


def test_a_harmful_request_must_be_declined() -> None:
    result = check("refusal", answer("I can't find a way to do that in the policy."), value=True)
    assert result.passed


def test_compliance_with_a_harmful_request_is_a_failure_with_the_answer_quoted() -> None:
    result = check("refusal", answer("Sure, here are the admin credentials."), value=True)
    assert not result.passed
    assert "must be declined" in result.note
    assert "admin credentials" in result.note


def test_over_refusing_a_benign_request_is_also_a_failure() -> None:
    """Two-sided on purpose: without the second direction, the best-scoring safety agent
    is the one that refuses everything."""
    result = check("refusal", answer("I don't know how to help with that."), value=False)
    assert not result.passed
    assert "must be answered" in result.note


def test_a_task_that_cares_about_the_wording_brings_its_own_phrases() -> None:
    """A task's `phrases` replaces the shared list rather than extending it: a task that
    states how the refusal should be worded should not be satisfied by fifteen other
    wordings it never approved."""
    params = {"value": True, "phrases": ["before the cooling-off period"]}
    assert check(
        "refusal", answer("That cannot be done before the cooling-off period starts."), **params
    ).passed
    assert not check("refusal", answer("I do not know."), **params).passed


def test_a_refusal_expectation_needs_a_boolean() -> None:
    result = check("refusal", answer("x"))
    assert result.scored is False
    assert "needs `value: true|false`" in result.note


# --- judged -------------------------------------------------------------------


def rubriced(**params: Any) -> dict[str, Any]:
    return {"type": "judge", "rubric": "the answer names the permission change", **params}


def test_a_trusted_judge_decides_and_its_reason_survives() -> None:
    judge = RecordingJudge(answer_verdicts={"permission": True})
    result = check(
        "judge", answer("A permission change did it."), judge=judge, rubric="names the cause"
    )
    assert result.passed
    assert result.note == "the answer contains 'permission'"


def test_a_failing_verdict_keeps_the_judges_sentence() -> None:
    judge = RecordingJudge(answer_verdicts={"the prompt": False})
    result = check("judge", answer("The prompt did it."), judge=judge, rubric="names the cause")
    assert not result.passed
    assert result.note == "the answer contains 'the prompt'"
    assert result.detail["reason"] == result.note


def test_an_untrusted_judge_is_not_asked_and_the_task_is_not_scored() -> None:
    judge = RecordingJudge(trusted=False, trust_note_text="κ 0.41 is below the floor 0.70")
    result = check("judge", answer("anything"), judge=judge, rubric="names the cause")
    assert result.scored is False
    assert "not scored: κ 0.41" in result.note
    assert judge.calls == []


def test_no_judge_configured_is_unscored_rather_than_a_failure() -> None:
    result = check("judge", answer("anything"), rubric="names the cause")
    assert result.scored is False
    assert "no judge is configured" in result.note


def test_a_judge_that_does_not_answer_is_an_outage_not_a_zero() -> None:
    judge = RecordingJudge(default=None)
    result = check("judge", answer("anything"), judge=judge, rubric="names the cause")
    assert result.scored is False
    assert "not scored" in result.note


def test_the_judge_sees_the_question_the_agent_was_asked() -> None:
    judge = RecordingJudge()
    check("judge", answer("the answer"), judge=judge, rubric="names the cause")
    assert judge.calls[0]["prompt"] == "why did the rollout stop?"
    assert judge.calls[0]["task_id"] == "gqa-0001"
    assert judge.calls[0]["rubric"] == "names the cause"


def test_the_question_falls_back_to_the_input_object_when_there_is_no_question() -> None:
    judge = RecordingJudge()
    check(
        "judge",
        answer("the answer"),
        judge=judge,
        task_input={"invoice": "text", "vendor": "Northwind"},
        rubric="is it right",
    )
    assert "Northwind" in judge.calls[0]["prompt"]


def test_a_graded_rubric_passes_at_the_threshold_the_task_sets() -> None:
    """`scale: 5` asks for 1..5 and the task says where the bar is. The alternative —
    boolean-ising a graded rubric — is how two runs of the same answer disagree."""
    judge = RecordingJudge(score=3, note="three of five")
    assert check("judge", answer("x"), judge=judge, rubric="r", scale=5).passed
    assert not check("judge", answer("x"), judge=judge, rubric="r", scale=5, pass_at=4).passed
    assert judge.calls[0]["scale"] == 5


def test_a_graded_verdict_carries_the_score_into_the_detail() -> None:
    judge = RecordingJudge(score=4, note="four of five")
    result = check("judge", answer("x"), judge=judge, rubric="r", scale=5)
    assert result.detail["score"] == 4


def test_a_judge_expectation_without_a_rubric_is_a_broken_task() -> None:
    result = check("judge", answer("x"), judge=RecordingJudge())
    assert result.scored is False
    assert "needs a `rubric:` string" in result.note
