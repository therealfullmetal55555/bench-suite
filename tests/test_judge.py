"""The judge: the reply parser, the cache, and the calibration it will be trusted on.

Three groups, and the split is deliberate. `parse_verdict` and `render` are pure functions
and are tested as such — a hundred cases without a socket. `JudgeClient` is tested against
`httpx.MockTransport`, because the things that go wrong there are HTTP-shaped: a 500, a
timeout, a reply with prose in it. And calibration is tested against a dictionary of canned
verdicts, because what it computes is arithmetic and the arithmetic is the thing the whole
judged share of the board rests on.

The one test a reader should read first is `test_a_judge_that_errors_is_not_a_disagreement`:
an outage that scores as a failure is the failure mode this package is shaped to avoid.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from bench.errors import BenchError
from bench.judge import (
    JudgeClient,
    JudgeConfig,
    JudgeError,
    JudgeVerdict,
    calibrate,
    calibration_path,
    cohens_kappa,
    parse_verdict,
    read_labels,
    render,
    scaled_prompt,
    write_calibration,
)


def config(tmp_path: Path, **overrides: Any) -> JudgeConfig:
    base: dict[str, Any] = {
        "model": "test-judge",
        "base_url": "https://judge.test/v1",
        "api_key_env": "BENCH_TEST_JUDGE_KEY",
        "cache_dir": str(tmp_path / "cache"),
        "price_prompt_per_mtok": 1.0,
        "price_completion_per_mtok": 2.0,
    }
    base.update(overrides)
    return JudgeConfig(**base)


def reply(content: str, usage: dict[str, int] | None = None) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": usage or {"prompt_tokens": 100, "completion_tokens": 20},
        },
    )


def client(tmp_path: Path, handler: Any, **overrides: Any) -> JudgeClient:
    judge = JudgeClient(config(tmp_path, **overrides))
    judge.replace_transport(httpx.MockTransport(handler))
    return judge


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BENCH_TEST_JUDGE_KEY", "not-a-real-key")


# --- parsing the reply ---------------------------------------------------------


def test_a_plain_verdict_parses() -> None:
    assert parse_verdict('{"pass": true, "reason": "names the cause"}') == (
        True,
        "names the cause",
        None,
    )


def test_a_verdict_wrapped_in_prose_and_a_fence_still_parses() -> None:
    """The prompt asks for bare JSON and models answer in prose anyway. A benchmark that
    fails on a stray backtick is a benchmark nobody runs twice."""
    text = (
        "Sure! Here is my verdict:\n```json\n"
        '{"pass": false, "reason": "no cause"}\n```\nHope that helps.'
    )
    assert parse_verdict(text) == (False, "no cause", None)


def test_the_first_verdict_wins_when_the_model_keeps_talking() -> None:
    text = '{"pass": true, "reason": "first"} and then {"pass": false, "reason": "second"}'
    assert parse_verdict(text) == (True, "first", None)


def test_a_string_that_looks_like_a_boolean_is_not_one() -> None:
    """`"pass": "true"` is the model hedging, and a parser that coerces it is a parser
    that invents verdicts."""
    assert parse_verdict('{"pass": "true", "reason": "sure"}') is None


def test_a_verdict_without_a_reason_is_still_a_verdict() -> None:
    assert parse_verdict('{"pass": true}') == (True, "", None)


def test_a_graded_reply_parses_into_a_score() -> None:
    assert parse_verdict('{"score": 4, "reason": "mostly right"}') == (True, "mostly right", 4)


def test_a_reply_with_no_json_object_at_all_is_not_a_verdict() -> None:
    assert parse_verdict("I think it is probably fine.") is None
    assert parse_verdict('{"verdict": "pass"}') is None


def test_an_unbalanced_brace_does_not_swallow_the_next_object() -> None:
    assert parse_verdict('{ oops {"pass": true, "reason": "second object"}') == (
        True,
        "second object",
        None,
    )


def test_the_user_message_carries_the_rubric_the_question_and_the_answer() -> None:
    message = render(rubric="names the cause", prompt="why?", output="a permission change")
    assert "names the cause" in message and "why?" in message and "a permission change" in message


def test_an_empty_answer_is_labelled_rather_than_left_blank() -> None:
    assert "(empty)" in render(rubric="r", prompt="p", output="")


def test_swapping_the_rubric_reverses_the_requirements_without_changing_them() -> None:
    one = "Names the cause. Does not blame the prompt. One sentence."
    two = "One sentence. Does not blame the prompt. Names the cause."
    assert render(rubric=one, prompt="p", output="a", swap=True) == render(
        rubric=two, prompt="p", output="a"
    )


def test_a_graded_rubric_asks_for_a_number_on_the_scale() -> None:
    assert '{"score": 1..5' in scaled_prompt(5)
    assert "1 to 5" in scaled_prompt(5)
    assert scaled_prompt(None).count('"pass"') == 1


# --- the client ----------------------------------------------------------------


def test_a_call_returns_the_verdict_and_prices_it(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return reply('{"pass": true, "reason": "yes"}')

    judge = client(tmp_path, handler)
    verdict = judge.verdict(rubric="r", prompt="q", output="a")
    assert verdict.passed is True and verdict.scored
    assert verdict.cost_usd is not None and str(verdict.cost_usd) == "0.000140"
    assert judge.spent_usd == verdict.cost_usd
    assert calls[0]["model"] == "test-judge"
    assert calls[0]["temperature"] == 0.0


def test_the_same_question_is_not_paid_for_twice(tmp_path: Path) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return reply('{"pass": true, "reason": "yes"}')

    judge = client(tmp_path, handler)
    first = judge.verdict(rubric="r", prompt="q", output="a")
    second = judge.verdict(rubric="r", prompt="q", output="a")
    assert calls["n"] == 1
    assert second.cached is True and second.reason == first.reason
    assert judge.cache_hits == 1 and judge.cache_misses == 1
    assert str(judge.spent_usd) == "0.000140"  # once


def test_a_different_answer_is_a_different_question_even_with_the_same_rubric(
    tmp_path: Path,
) -> None:
    canned = {
        "a good answer": '{"pass": true, "reason": "yes"}',
        "a bad answer": '{"pass": false, "reason": "no"}',
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        text = body["messages"][1]["content"]
        return reply(next(value for key, value in canned.items() if key in text))

    judge = client(tmp_path, handler)
    assert judge.verdict(rubric="r", prompt="q", output="a good answer").passed is True
    assert judge.verdict(rubric="r", prompt="q", output="a bad answer").passed is False


def test_the_cache_survives_a_new_client_on_the_same_directory(tmp_path: Path) -> None:
    """That is the whole point: a re-scored submission costs no judge calls, and the
    calibration report says how many were served from cache so a reader can tell."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return reply('{"pass": true, "reason": "yes"}')

    first = client(tmp_path, handler)
    first.verdict(rubric="r", prompt="q", output="a")
    second = client(tmp_path, handler)
    verdict = second.verdict(rubric="r", prompt="q", output="a")
    assert calls["n"] == 1
    assert verdict.cached is True and verdict.passed is True


def test_a_graded_verdict_keeps_its_score(tmp_path: Path) -> None:
    judge = client(tmp_path, lambda request: reply('{"score": 4, "reason": "good"}'))
    verdict = judge.verdict(rubric="r", prompt="q", output="a", scale=5)
    assert verdict.score == 4 and verdict.passed is True
    assert judge.verdict(rubric="r", prompt="q", output="a", scale=5).score == 4


def test_an_http_error_is_an_outage_not_a_failed_verdict(tmp_path: Path) -> None:
    judge = client(tmp_path, lambda request: httpx.Response(500, text="upstream is sad"))
    verdict = judge.verdict(rubric="r", prompt="q", output="a")
    assert verdict.passed is None and verdict.scored is False
    assert verdict.error == "HTTP 500" and "upstream is sad" in verdict.reason


def test_a_transport_error_is_an_outage_too(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    verdict = client(tmp_path, handler).verdict(rubric="r", prompt="q", output="a")
    assert verdict.scored is False and verdict.error == "ConnectError: no route to host"


def test_a_reply_with_prose_instead_of_a_verdict_is_not_a_verdict(tmp_path: Path) -> None:
    judge = client(tmp_path, lambda request: reply("I would say it mostly passes."))
    verdict = judge.verdict(rubric="r", prompt="q", output="a")
    assert verdict.scored is False
    assert verdict.error == "no verdict in the reply"


def test_a_response_with_no_choices_in_it_is_reported_as_unparseable(tmp_path: Path) -> None:
    judge = client(tmp_path, lambda request: httpx.Response(200, json={"error": "nope"}))
    verdict = judge.verdict(rubric="r", prompt="q", output="a")
    assert verdict.scored is False and verdict.error == "unparseable response"


def test_an_outage_is_not_written_to_the_cache(tmp_path: Path) -> None:
    """Otherwise one bad minute is remembered for ever, and every later re-scoring
    silently inherits it."""
    answers = {"first": True, "second": False}

    def handler(request: httpx.Request) -> httpx.Response:
        if answers.pop("first", False):
            return httpx.Response(503, text="try later")
        return reply('{"pass": false, "reason": "no"}')

    judge = client(tmp_path, handler)
    assert judge.verdict(rubric="r", prompt="q", output="a").scored is False
    retry = judge.verdict(rubric="r", prompt="q", output="a")
    assert retry.scored is True and retry.passed is False


def test_a_missing_key_is_a_configuration_error_with_a_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BENCH_TEST_JUDGE_KEY")
    judge = client(tmp_path, lambda request: reply("{}"))
    with pytest.raises(JudgeError) as caught:
        judge.verdict(rubric="r", prompt="q", output="a")
    assert "BENCH_TEST_JUDGE_KEY is not set" in str(caught.value)


def test_a_judge_with_no_cache_directory_does_not_write_one(tmp_path: Path) -> None:
    judge = client(tmp_path, lambda request: reply('{"pass": true, "reason": "y"}'), cache_dir=None)
    assert judge.verdict(rubric="r", prompt="q", output="a").passed is True
    assert not (tmp_path / "cache").exists()


def test_cost_is_not_measured_when_no_prices_are_configured(tmp_path: Path) -> None:
    """`None`, never zero: a cost column that prints $0.0000 for a judge nobody priced is
    claiming the judging was free."""
    judge = client(
        tmp_path,
        lambda request: reply('{"pass": true, "reason": "y"}'),
        price_prompt_per_mtok=None,
        price_completion_per_mtok=None,
    )
    assert judge.verdict(rubric="r", prompt="q", output="a").cost_usd is None


def test_the_config_file_is_read_and_unknown_keys_are_kept(tmp_path: Path) -> None:
    path = tmp_path / "judge.yaml"
    path.write_text(
        "model: gpt-whatever\nbase_url: https://example.test/v1\n"
        "temperature: 0.2\nheaders:\n  X-Org: team\n",
        encoding="utf-8",
    )
    config_object = JudgeConfig.from_file(path)
    assert (config_object.model, config_object.temperature) == ("gpt-whatever", 0.2)
    assert config_object.extra["headers"] == {"X-Org": "team"}


def test_a_config_file_without_a_model_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "judge.yaml"
    path.write_text("temperature: 0\n", encoding="utf-8")
    with pytest.raises(JudgeError):
        JudgeConfig.from_file(path)


# --- kappa ---------------------------------------------------------------------


def test_kappa_is_one_when_two_raters_agree_and_both_vary() -> None:
    assert cohens_kappa(
        [(True, True), (False, False), (True, True), (False, False)]
    ) == pytest.approx(1.0)


def test_kappa_is_negative_when_they_agree_less_than_chance() -> None:
    pairs = [(True, False), (True, False), (False, True), (False, True)]
    assert cohens_kappa(pairs) == pytest.approx(-1.0)


def test_a_rater_that_never_varies_gets_a_kappa_of_zero_not_one() -> None:
    """The 0/0 case, and the reason it is not reported as perfect: on a set that is all
    passes, a judge that always says pass agrees 100% of the time and has measured
    nothing. Writing 1.0 there is how an uncalibrated judge gets trusted."""
    assert cohens_kappa([(True, True)] * 6) == 0.0


def test_kappa_of_chance_level_agreement_is_about_zero() -> None:
    pairs = [(True, True), (True, False), (False, True), (False, False)] * 3
    assert abs(cohens_kappa(pairs)) < 0.05


# --- labels --------------------------------------------------------------------


def labels_file(tmp_path: Path, rows: list[dict[str, Any]], name: str = "labels.jsonl") -> Path:
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def label(case_id: str = "gqa-0001", human: bool = True, **overrides: Any) -> dict[str, Any]:
    row = {
        "case_id": case_id,
        "rubric": CANONICAL,
        "prompt": "why did the rollout stop?",
        "output": "a permission change",
        "human": human,
    }
    row.update(overrides)
    return row


def test_labels_are_read_in_order(tmp_path: Path) -> None:
    path = labels_file(tmp_path, [label("a"), label("b", human=False)])
    rows = read_labels(path)
    assert [row["case_id"] for row in rows] == ["a", "b"]


def test_a_label_line_that_cannot_be_compared_is_refused_with_its_line_number(
    tmp_path: Path,
) -> None:
    """Skipping it silently is how a calibration ends up reporting agreement over four of
    the fifty rows somebody thought they had written."""
    path = tmp_path / "labels.jsonl"
    path.write_text(
        json.dumps(label("a")) + "\n" + '{"case_id": "b", "rubric": "r"}\n' + "not json\n",
        encoding="utf-8",
    )
    with pytest.raises(BenchError) as caught:
        read_labels(path)
    assert "line 2" in str(caught.value) and "line 3" in str(caught.value)


def test_a_human_verdict_that_is_not_boolean_is_refused(tmp_path: Path) -> None:
    path = labels_file(tmp_path, [label("a", human="yes")])
    with pytest.raises(BenchError):
        read_labels(path)


def test_a_labels_file_with_nothing_in_it_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text("# all commented out\n", encoding="utf-8")
    with pytest.raises(BenchError):
        read_labels(path)


# --- calibration ----------------------------------------------------------------


CANONICAL = "Names the cause. Does not blame the prompt."
"""Two sentences on purpose: `_swap_halves` has nothing to reorder in one, and a
single-sentence rubric would make the flip probe silently do nothing."""


class CannedJudge:
    """A judge whose verdicts are a function of the answer, so a calibration's arithmetic
    can be checked against hand-counted numbers.

    `swapped` is keyed by answer as well and is consulted when the rubric it is handed is
    not `CANONICAL` — that is, on the reordered pass — which is how a test can say "this
    answer flips when the requirements are read in the other order".
    """

    def __init__(
        self,
        verdicts: dict[str, bool],
        swapped: dict[str, bool] | None = None,
        labels: int = 0,
    ) -> None:
        self.verdicts = verdicts
        self.swapped = swapped or {}
        self.config = JudgeConfig(model="canned", cache_dir=None, kappa_floor=0.70)
        self.calls = 0
        self.cache_hits = 0
        self.spent_usd = 0

    trusted = True

    def verdict(
        self, *, rubric: str, prompt: str, output: str, task_id: str = "", scale: Any = None
    ) -> JudgeVerdict:
        self.calls += 1
        table = self.swapped if rubric != CANONICAL else self.verdicts
        if output not in table:
            return JudgeVerdict(None, "the judge is unreachable", error="unreachable")
        return JudgeVerdict(passed=table[output], reason="canned")


def rows(*pairs: tuple[bool, bool]) -> list[dict[str, Any]]:
    return [
        label(f"case-{index}", human=human, output=f"answer-{index}")
        for index, (human, _judged) in enumerate(pairs)
    ]


def test_agreement_kappa_and_the_two_kinds_of_error_are_counted(tmp_path: Path) -> None:
    pairs = [(True, True), (True, False), (False, True), (False, False)]
    judge = CannedJudge(
        {"answer-0": True, "answer-1": False, "answer-2": True, "answer-3": False},
        swapped={"answer-0": True, "answer-1": False, "answer-2": True, "answer-3": False},
    )
    report = calibrate(judge, rows(*pairs))
    assert report.labels == 4
    assert report.agreement == pytest.approx(0.5)
    assert report.false_pass == 1 and report.false_fail == 1
    assert report.unanswered == 0
    assert report.flips == 0


def test_a_judge_that_errors_on_every_label_is_reported_as_unanswered_not_wrong(
    tmp_path: Path,
) -> None:
    """An outage is not a disagreement. A calibration that counted 40 timeouts as 40
    disagreements would conclude the judge is bad at its job, which is not what happened."""
    judge = CannedJudge({})
    report = calibrate(judge, rows(*[(True, True)] * 4))
    assert report.unanswered == 4
    assert report.kappa == 0.0
    assert "none of the 4 labels could be judged" in report.summary()


def test_a_verdict_that_flips_under_a_reordered_rubric_is_counted(tmp_path: Path) -> None:
    judge = CannedJudge({"answer-0": True}, swapped={"answer-0": False})
    report = calibrate(judge, rows((True, True)))
    assert report.flips == 1
    assert report.flip_rate == pytest.approx(1.0)
    assert report.usable is False


def test_twenty_labels_are_the_minimum_that_means_anything(tmp_path: Path) -> None:
    answers = {f"answer-{index}": True for index in range(10)}
    judge = CannedJudge(answers, swapped=answers)
    report = calibrate(judge, rows(*[(True, True)] * 10))
    assert report.labels == 10
    assert report.usable is False
    assert "below the 20 needed" in report.summary()


def test_a_calibration_is_written_where_the_judge_looks_for_it(tmp_path: Path) -> None:
    """The failure this prevents: a calibration written somewhere and read from nowhere,
    which reads on the board as "the judge was never calibrated"."""
    config_object = config(tmp_path)
    answers = {f"answer-{index}": index % 2 == 0 for index in range(24)}
    judge = CannedJudge(answers, swapped=answers)
    report = calibrate(
        judge, rows(*[(index % 2 == 0, True) for index in range(24)]), labels_file="labels.jsonl"
    )
    path = write_calibration(report, config_object)
    assert (
        path
        == calibration_path(config_object)
        == Path(config_object.cache_dir) / "judge-calibration.json"
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["model"] == "canned" and "kappa" in document


def test_a_client_with_a_calibration_at_the_floor_is_trusted(tmp_path: Path) -> None:
    judge = JudgeClient(config(tmp_path))
    path = calibration_path(judge.config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"format": 1, "model": "test-judge", "kappa": 0.81, "labels": 40}),
        encoding="utf-8",
    )
    assert JudgeClient(config(tmp_path)).trusted is True
    assert "judge-scored tasks count" in JudgeClient(config(tmp_path)).trust_note()


def test_a_client_whose_calibration_is_below_the_floor_is_not_trusted(tmp_path: Path) -> None:
    judge = JudgeClient(config(tmp_path))
    path = calibration_path(judge.config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"format": 1, "model": "test-judge", "kappa": 0.62, "labels": 40}),
        encoding="utf-8",
    )
    fresh = JudgeClient(config(tmp_path))
    assert fresh.trusted is False
    assert "below the floor" in fresh.trust_note()


def test_no_calibration_means_not_trusted_and_the_note_says_what_to_run(tmp_path: Path) -> None:
    judge = JudgeClient(config(tmp_path))
    assert judge.trusted is False
    assert "bench judge calibrate" in judge.trust_note()


def test_a_calibration_for_another_model_is_ignored(tmp_path: Path) -> None:
    """A κ that describes a different model is not evidence about this one, and the
    tempting shortcut — trusting any file that happens to be there — is how a board ends
    up with a judge that was never audited."""
    path = calibration_path(config(tmp_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"format": 1, "model": "some-other-judge", "kappa": 0.95}), encoding="utf-8"
    )
    judge = JudgeClient(config(tmp_path))
    assert judge.trusted is False
    assert judge.calibration is None
