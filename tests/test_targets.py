"""The four target kinds, and the boundary that turns anything into a trace.

The theme is the rule from `targets/base.py`: a failure is a value. So most of what is
tested here is what happens when the thing being benchmarked misbehaves — a timeout, a
500, a body that is not JSON, a function that raises, a cassette that does not cover the
task — and every one of those has to come back as a `Trace` with something useful in
`error`, not as an exception that ends the run.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from conftest import task

from bench.core.task import Task
from bench.core.trace import Trace
from bench.errors import TargetError
from bench.targets import (
    CassetteTarget,
    HttpTarget,
    OpenAITarget,
    ScriptedTarget,
    TargetConfig,
    build_target,
    coerce_trace,
    load_target,
    messages_from_input,
)


def make_task(**overrides) -> Task:
    return Task.model_validate(task(**overrides))


def config(**overrides) -> TargetConfig:
    base: dict = {"kind": "scripted", "callable": "fake_agents:echo"}
    base.update(overrides)
    return TargetConfig.model_validate(base)


def run(target, task_obj: Task | None = None, repeat: int = 0) -> Trace:
    return asyncio.run(target.run(task_obj or make_task(), repeat=repeat))


# --- the boundary -------------------------------------------------------------


def test_a_string_is_the_answer():
    """The most common integration of all. It does not deserve to be treated as a
    mistake, and it does not need a wrapper class either."""
    trace = coerce_trace("the permission change", latency_ms=12)
    assert trace.output == "the permission change"
    assert trace.latency_ms == 12
    assert not trace.failed


def test_a_mapping_is_a_trace():
    trace = coerce_trace(
        {"output": "yes", "tokens": {"prompt": 5, "completion": 1}, "steps": 3}, latency_ms=4
    )
    assert trace.output == "yes"
    assert trace.tokens.prompt == 5
    assert trace.steps == 3


def test_a_mapping_that_is_not_a_trace_says_what_a_trace_is():
    with pytest.raises(TargetError) as excinfo:
        coerce_trace({"answr": "typo"}, latency_ms=1)
    assert "not a trace" in str(excinfo.value)
    assert "output" in (excinfo.value.hint or "")


def test_none_is_refused_by_name():
    """`return None` from an early exit in an integrator's function is the single most
    common way to get here, and the message has to say which value was useless."""
    with pytest.raises(TargetError, match="returned None"):
        coerce_trace(None, latency_ms=1)


def test_something_with_no_answer_in_it_names_its_type():
    with pytest.raises(TargetError) as excinfo:
        coerce_trace(42, latency_ms=1)
    assert "int" in str(excinfo.value)


def test_an_object_with_an_output_attribute_is_read():
    class Response:
        output = "from an object"
        latency_ms = 30

    trace = coerce_trace(Response(), latency_ms=1)
    assert trace.output == "from an object"
    assert trace.latency_ms == 30


def test_pricing_is_applied_when_the_config_knows_the_rates():
    trace = coerce_trace(
        {"output": "x", "tokens": {"prompt": 1_000_000, "completion": 1_000_000}},
        latency_ms=1,
        config=config(price_prompt_per_mtok=3.0, price_completion_per_mtok=15.0),
    )
    assert trace.cost_usd == pytest.approx(18.0)


def test_no_rates_means_no_cost_and_not_zero():
    trace = coerce_trace({"output": "x", "tokens": {"prompt": 10}}, latency_ms=1, config=config())
    assert trace.cost_usd is None


# --- scripted -----------------------------------------------------------------


def test_a_scripted_target_echoes_the_question():
    trace = run(ScriptedTarget("s", config()))
    assert trace.output == "why did the rollout stop?"


def test_a_scripted_target_takes_a_dict_back():
    trace = run(ScriptedTarget("s", config(callable="fake_agents:as_dict")))
    assert trace.tool_calls[0].name == "lookup_order"
    assert trace.tokens.cached == 100
    assert trace.steps == 2


def test_a_scripted_target_awaits_a_coroutine():
    trace = run(ScriptedTarget("s", config(callable="fake_agents:async_answer")))
    assert trace.output == "answered asynchronously"


def test_a_scripted_target_reads_an_object_with_usage():
    trace = run(ScriptedTarget("s", config(callable="fake_agents:as_object")))
    assert trace.output == "from an object"
    assert trace.tokens.prompt == 30
    assert trace.tool_calls[0].arguments == {"order_id": "ORD-2"}


def test_a_callable_without_the_repeat_keyword_still_works():
    """Forcing the keyword onto an integrator's function means every entrant writes a
    wrapper, so the signature is inspected instead."""
    trace = run(ScriptedTarget("s", config(callable="fake_agents:no_repeat_argument")))
    assert trace.output == "no repeat keyword here"


def test_a_raising_target_becomes_a_traced_failure():
    trace = run(ScriptedTarget("s", config(callable="fake_agents:explode")))
    assert trace.failed
    assert "RuntimeError" in (trace.error or "")
    assert "retrieval index" in (trace.error or "")


def test_a_target_returning_nonsense_becomes_a_traced_failure_not_a_crash():
    """The other 299 tasks still have something to say about this entrant."""
    trace = run(ScriptedTarget("s", config(callable="fake_agents:wrong")))
    assert trace.failed
    assert "int" in (trace.error or "")


def test_a_file_path_where_an_import_path_belongs_says_what_to_write():
    with pytest.raises(TargetError) as excinfo:
        ScriptedTarget("s", config(callable="agents/run.py:answer"))
    assert "import path, not a file path" in str(excinfo.value)
    assert "agents.run:answer" in str(excinfo.value)


def test_a_missing_module_and_a_missing_attribute_are_different_messages():
    """Both are startup errors, and both say which of the two is wrong: the difference
    between "you misconfigured the harness" and "your agent is broken"."""
    with pytest.raises(TargetError, match="cannot import"):
        ScriptedTarget("s", config(callable="nope.missing:answer"))
    with pytest.raises(TargetError, match="no attribute"):
        ScriptedTarget("s", config(callable="fake_agents:nonexistent"))


def test_a_malformed_callable_is_refused():
    with pytest.raises(TargetError, match="module.path:function"):
        ScriptedTarget("s", config(callable="no_colon"))


def test_a_module_next_to_the_config_is_importable(tmp_path: Path):
    """The lesson passmark paid for: `import_paths` resolved against the config file, so
    the harness does not only work from a checkout."""
    (tmp_path / "side_agent.py").write_text(
        "def answer(case_input, *, repeat=0):\n    return 'from next to the config'\n",
        encoding="utf-8",
    )
    (tmp_path / "target.yaml").write_text(
        "name: side\n"
        "target:\n"
        "  kind: scripted\n"
        "  callable: side_agent:answer\n"
        f"  import_paths: ['{tmp_path}']\n",
        encoding="utf-8",
    )
    loaded = load_target(tmp_path / "target.yaml")
    trace = run(build_target("side", loaded.target))
    assert trace.output == "from next to the config"


def test_a_target_file_that_does_not_validate_names_the_field(tmp_path: Path):
    file = tmp_path / "bad.yaml"
    file.write_text("name: x\ntarget:\n  kind: telepathy\n", encoding="utf-8")
    with pytest.raises(TargetError) as excinfo:
        load_target(file)
    assert "kind" in str(excinfo.value)


def test_a_missing_target_file_says_so(tmp_path: Path):
    with pytest.raises(TargetError, match="no such target file"):
        load_target(tmp_path / "nope.yaml")


# --- messages from task input -------------------------------------------------


def test_a_question_becomes_one_user_message():
    messages = messages_from_input({"question": "why?"})
    assert len(messages) == 1 and messages[0].role == "user"
    assert messages[0].content == "why?"


def test_passages_are_rendered_as_a_labelled_block():
    """Rendered rather than sent as JSON: a model reads a labelled block better than an
    array of objects, and the format is fixed so two entrants' prompts differ only in
    the entrant's own doing."""
    messages = messages_from_input(
        {"question": "why?", "passages": [{"id": "a", "text": "the reason"}, "a bare string"]}
    )
    assert "Passages:" in messages[0].content
    assert "[1] the reason" in messages[0].content
    assert messages[0].content.endswith("Question: why?")


def test_messages_are_used_when_the_task_has_them():
    messages = messages_from_input(
        {"messages": [{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}]}
    )
    assert [message.role for message in messages] == ["system", "user"]


def test_a_task_with_nothing_to_send_is_refused():
    with pytest.raises(TargetError, match="no `question`"):
        messages_from_input({"corpus": "docs"})


# --- http ---------------------------------------------------------------------


def http_config(**overrides) -> TargetConfig:
    base = {"kind": "http", "endpoint": "http://agent.invalid/run"}
    base.update(overrides)
    return TargetConfig.model_validate(base)


def with_transport(target, handler) -> None:
    target.replace_transport(httpx.MockTransport(handler))


def test_an_http_target_reads_a_trace_back():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["task_id"] == "gqa-0001"
        assert body["family"] == "grounded-qa"
        return httpx.Response(
            200, json={"output": "the permission change", "tokens": {"prompt": 9}}
        )

    target = HttpTarget("http", http_config())
    with_transport(target, handler)
    trace = run(target)
    assert trace.output == "the permission change"
    assert trace.tokens.prompt == 9
    assert trace.raw["status"] == 200


def test_an_answer_field_is_accepted_as_an_output():
    """`{"answer": …}` shows up in every second integration, and asking an entrant to
    change their service's response to be benchmarked is how a benchmark ends up with
    four entrants."""
    target = HttpTarget("http", http_config())
    with_transport(target, lambda request: httpx.Response(200, json={"answer": "yes"}))
    assert run(target).output == "yes"


def test_a_server_error_becomes_a_traced_failure_with_the_body_quoted():
    target = HttpTarget("http", http_config())
    with_transport(target, lambda request: httpx.Response(500, text="upstream died\nat 09:14"))
    trace = run(target)
    assert trace.failed
    assert "HTTP 500" in (trace.error or "")
    # the newline is flattened and the body kept: it is the useful part
    assert "upstream died at 09:14" in (trace.error or "")


def test_a_client_error_is_not_retried_but_is_reported():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, text="bad input: `question` missing")

    target = HttpTarget("http", http_config(retries=2))
    with_transport(target, handler)
    trace = run(target)
    assert trace.failed
    assert "bad input" in (trace.error or "")
    assert len(calls) == 1


def test_a_server_error_is_retried_when_the_config_asks_for_it():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(503, text="try later")
        return httpx.Response(200, json={"output": "third time lucky"})

    target = HttpTarget("http", http_config(retries=3))
    with_transport(target, handler)
    assert run(target).output == "third time lucky"
    assert len(calls) == 3


def test_a_transport_failure_is_a_traced_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    target = HttpTarget("http", http_config())
    with_transport(target, handler)
    trace = run(target)
    assert trace.failed
    assert "ConnectError" in (trace.error or "")


def test_a_body_that_is_not_json_is_reported_with_what_it_was():
    target = HttpTarget("http", http_config())
    with_transport(target, lambda request: httpx.Response(200, text="<html>502</html>"))
    trace = run(target)
    assert trace.failed
    assert "did not return JSON" in (trace.error or "")
    assert "html" in (trace.error or "")


def test_an_empty_body_is_reported_as_such():
    target = HttpTarget("http", http_config())
    with_transport(target, lambda request: httpx.Response(200, content=b""))
    assert "empty body" in (run(target).error or "")


def test_a_json_array_is_not_a_trace():
    target = HttpTarget("http", http_config())
    with_transport(target, lambda request: httpx.Response(200, json=[1, 2, 3]))
    assert "list" in (run(target).error or "")


def test_a_body_with_no_answer_in_it_is_a_failure_and_not_a_zero():
    """A 200 carrying the right shape and no answer would otherwise score zero and read
    in the failure list as "answered wrong", when the truth is "answered nothing"."""
    target = HttpTarget("http", http_config())
    with_transport(target, lambda request: httpx.Response(200, json={"nonsense": 1}))
    trace = run(target)
    assert trace.failed
    assert "no answer" in (trace.error or "")


def test_an_empty_string_answer_is_a_value():
    """A model that replied with nothing has told you something, and it is scored."""
    trace = coerce_trace({"output": ""}, latency_ms=1)
    assert trace.output == ""
    assert not trace.failed


def test_tool_calls_without_an_output_are_a_trace():
    """An agent that called tools and produced no final prose is mid-flight, not
    silent — the trajectory scorers read exactly that."""
    trace = coerce_trace({"tool_calls": [{"name": "lookup_order"}]}, latency_ms=1)
    assert trace.tool_calls[0].name == "lookup_order"
    assert trace.output is None


def test_unknown_keys_are_kept_in_raw_rather_than_refused():
    """A real service returns `request_id`, `status`, a debug object. A body with the
    right answer and one extra key is not a failed task — and the extra is kept, so the
    run file still holds what arrived."""
    target = HttpTarget("http", http_config())
    with_transport(
        target,
        lambda request: httpx.Response(
            200, json={"output": "yes", "request_id": "req_1", "cached": True}
        ),
    )
    trace = run(target)
    assert trace.output == "yes"
    assert trace.raw["request_id"] == "req_1"
    assert trace.raw["cached"] is True


def test_strict_mode_raises_instead_of_tracing():
    """For the case where the response shape is the harness's own contract and a
    mismatch means the harness is wrong, not the entrant."""
    target = HttpTarget("http", http_config(strict=True))
    with_transport(target, lambda request: httpx.Response(200, json={"nonsense": 1}))
    with pytest.raises(TargetError):
        run(target)


def test_an_endpoint_that_is_not_a_url_stops_the_run():
    """A configuration error, so it raises: 300 identical failures in the report reads
    as "the entrant is bad" when the truth is "the entrant is misspelled"."""
    with pytest.raises(TargetError, match="absolute URL"):
        HttpTarget("http", http_config(endpoint="agent.example.com/run"))


def test_an_http_target_without_an_endpoint_says_so():
    with pytest.raises(TargetError, match="needs an `endpoint`"):
        HttpTarget("http", TargetConfig(kind="http"))


# --- openai -------------------------------------------------------------------


def openai_config(**overrides) -> TargetConfig:
    base = {
        "kind": "openai",
        "model": "example-mini",
        "base_url": "http://model.invalid/v1",
        "api_key_env": "BENCH_TEST_KEY",
        "price_prompt_per_mtok": 0.15,
        "price_completion_per_mtok": 0.60,
    }
    base.update(overrides)
    return TargetConfig.model_validate(base)


def completion(**overrides) -> dict:
    payload = {
        "id": "cmpl-1",
        "model": "example-mini",
        "choices": [
            {
                "message": {"role": "assistant", "content": "the permission change"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }
    payload.update(overrides)
    return payload


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BENCH_TEST_KEY", "test-key")


def test_a_completion_becomes_a_trace_with_tokens_and_cost():
    target = OpenAITarget("gpt", openai_config())
    with_transport(target, lambda request: httpx.Response(200, json=completion()))
    trace = run(target)
    assert trace.output == "the permission change"
    assert trace.tokens.total == 120
    assert trace.raw["priced_from"] == "provider"
    assert trace.cost_usd is not None and trace.cost_usd > 0
    # the prompt that was sent is kept, so a report can show what the entrant asked
    assert [message.role for message in trace.messages] == ["user", "assistant"]


def test_cached_tokens_are_read_from_the_details_block():
    payload = completion(
        usage={
            "prompt_tokens": 1000,
            "completion_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 900},
        }
    )
    target = OpenAITarget("gpt", openai_config(price_cached_per_mtok=0.03))
    with_transport(target, lambda request: httpx.Response(200, json=payload))
    trace = run(target)
    assert trace.tokens.cached == 900
    # 100 at 0.15 + 900 at 0.03 per million
    assert float(trace.cost_usd or 0) == pytest.approx(0.000042, abs=1e-9)


def test_tokens_are_estimated_when_the_provider_forgets_them():
    """A cost column that is silently zero for the first ten tasks and real for the rest
    is worse than no cost column, so the harness counts instead and says it did."""
    payload = completion()
    payload.pop("usage")
    target = OpenAITarget("gpt", openai_config())
    with_transport(target, lambda request: httpx.Response(200, json=payload))
    trace = run(target)
    assert trace.raw["priced_from"] == "estimated"
    assert trace.tokens.total > 0


def test_tool_calls_come_back_with_parsed_arguments():
    payload = completion(
        choices=[
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "lookup_order",
                                "arguments": '{"order_id": "ORD-1"}',
                            },
                        }
                    ],
                }
            }
        ]
    )
    target = OpenAITarget("gpt", openai_config())
    with_transport(target, lambda request: httpx.Response(200, json=payload))
    trace = run(target)
    assert trace.tool_calls[0].name == "lookup_order"
    assert trace.tool_calls[0].arguments == {"order_id": "ORD-1"}


def test_arguments_that_are_not_json_are_kept_rather_than_dropped():
    payload = completion(
        choices=[
            {
                "message": {
                    "content": None,
                    "tool_calls": [{"function": {"name": "lookup_order", "arguments": "{oops"}}],
                }
            }
        ]
    )
    target = OpenAITarget("gpt", openai_config())
    with_transport(target, lambda request: httpx.Response(200, json=payload))
    trace = run(target)
    assert trace.tool_calls[0].arguments == {"_raw": "{oops"}


def test_a_response_with_no_choices_says_what_it_got():
    target = OpenAITarget("gpt", openai_config())
    with_transport(
        target, lambda request: httpx.Response(200, json={"error": {"message": "quota exceeded"}})
    )
    trace = run(target)
    assert trace.failed
    assert "quota exceeded" in (trace.error or "")


def test_a_missing_key_is_a_configuration_error_and_not_300_failed_tasks(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("BENCH_TEST_KEY", raising=False)
    target = OpenAITarget("gpt", openai_config())
    with_transport(target, lambda request: httpx.Response(200, json=completion()))
    with pytest.raises(TargetError) as excinfo:
        run(target)
    assert "BENCH_TEST_KEY is not set" in str(excinfo.value)
    assert "export" in (excinfo.value.hint or "")


def test_a_model_is_required():
    with pytest.raises(TargetError, match="needs a `model`"):
        OpenAITarget("gpt", TargetConfig(kind="openai"))


def test_the_request_carries_temperature_and_max_tokens_when_configured():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=completion())

    target = OpenAITarget("gpt", openai_config(temperature=0.0, max_tokens=64))
    with_transport(target, handler)
    run(target)
    assert seen["temperature"] == 0.0
    assert seen["max_tokens"] == 64
    assert seen["model"] == "example-mini"


def test_a_rate_limited_response_is_retried():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        return httpx.Response(200, json=completion())

    target = OpenAITarget("gpt", openai_config(retries=1))
    with_transport(target, handler)
    assert run(target).output == "the permission change"
    assert len(calls) == 2


# --- cassette -----------------------------------------------------------------


def test_a_recording_then_a_replay_returns_the_same_trace(tmp_path: Path):
    cassette = tmp_path / "main.json"
    live = ScriptedTarget("s", config(callable="fake_agents:as_dict"))
    recorder = CassetteTarget("s", config(record_to=str(cassette)), inner=live)
    recorded = run(recorder)
    asyncio.run(recorder.aclose())

    assert cassette.exists()
    replayed = run(CassetteTarget("s", config(cassette=str(cassette))))
    assert replayed.output == recorded.output
    assert replayed.tool_calls[0].name == recorded.tool_calls[0].name
    assert replayed.tokens.prompt == recorded.tokens.prompt


def test_replaying_a_task_the_cassette_does_not_cover_is_a_named_failure(tmp_path: Path):
    """Half live and half recorded is the worst of both: expensive and unreproducible.
    So a miss fails the task, and the message names the key and the cassette."""
    cassette = tmp_path / "main.json"
    recorder = CassetteTarget(
        "s", config(record_to=str(cassette)), inner=ScriptedTarget("s", config())
    )
    run(recorder, make_task(id="gqa-0001"))
    asyncio.run(recorder.aclose())

    replay = CassetteTarget("s", config(cassette=str(cassette)))
    trace = run(replay, make_task(id="gqa-0002"))
    assert trace.failed
    assert "gqa-0002/0" in (trace.error or "")
    assert trace.raw["cassette_miss"] == "gqa-0002/0"


def test_the_cassette_digest_changes_when_an_entry_changes(tmp_path: Path):
    """Without it, "the entrant improved" and "somebody edited the cassette" look the
    same on the board."""
    cassette = tmp_path / "main.json"
    recorder = CassetteTarget(
        "s", config(record_to=str(cassette)), inner=ScriptedTarget("s", config())
    )
    run(recorder)
    asyncio.run(recorder.aclose())
    first = CassetteTarget("s", config(cassette=str(cassette))).digest()

    payload = json.loads(cassette.read_text(encoding="utf-8"))
    payload["entries"]["gqa-0001/0"]["output"] = "somebody edited this"
    cassette.write_text(json.dumps(payload), encoding="utf-8")
    second = CassetteTarget("s", config(cassette=str(cassette))).digest()
    assert first != second


def test_recording_resumes_and_keeps_what_it_already_had(tmp_path: Path):
    """A 300-task recording has to survive a typo on task four."""
    cassette = tmp_path / "main.json"
    first = CassetteTarget(
        "s", config(record_to=str(cassette)), inner=ScriptedTarget("s", config())
    )
    run(first, make_task(id="gqa-0001"))
    asyncio.run(first.aclose())

    second = CassetteTarget(
        "s",
        config(cassette=str(cassette), record_to=str(cassette)),
        inner=ScriptedTarget("s", config()),
    )
    run(second, make_task(id="gqa-0002"))
    asyncio.run(second.aclose())

    payload = json.loads(cassette.read_text(encoding="utf-8"))
    assert set(payload["entries"]) == {"gqa-0001/0", "gqa-0002/0"}


def test_a_cassette_from_another_format_version_is_refused(tmp_path: Path):
    """A file that looks like a cassette and cannot be read is worse than a missing
    one, because it fails in somebody else's repository months later."""
    cassette = tmp_path / "old.json"
    cassette.write_text(json.dumps({"version": 0, "entries": {}}), encoding="utf-8")
    with pytest.raises(TargetError, match="format"):
        CassetteTarget("s", config(cassette=str(cassette)))


def test_a_file_that_is_not_a_cassette_says_what_one_is(tmp_path: Path):
    file = tmp_path / "not-a-cassette.json"
    file.write_text(json.dumps({"results": []}), encoding="utf-8")
    with pytest.raises(TargetError) as excinfo:
        CassetteTarget("s", config(cassette=str(file)))
    assert "is not a cassette" in str(excinfo.value)


def test_a_missing_cassette_is_refused_at_startup(tmp_path: Path):
    with pytest.raises(TargetError, match="cassette not found"):
        CassetteTarget("s", config(cassette=str(tmp_path / "nope.json")))


def test_a_cassette_target_with_neither_mode_is_refused():
    with pytest.raises(TargetError, match="needs `cassette`"):
        CassetteTarget("s", config())


def test_a_write_is_atomic_and_leaves_no_temporary_file(tmp_path: Path):
    cassette = tmp_path / "main.json"
    recorder = CassetteTarget(
        "s", config(record_to=str(cassette)), inner=ScriptedTarget("s", config())
    )
    run(recorder)
    asyncio.run(recorder.aclose())
    assert not list(tmp_path.glob("*.tmp"))


# --- build_target -------------------------------------------------------------


def test_build_target_picks_the_right_class():
    assert isinstance(build_target("s", config()), ScriptedTarget)
    assert isinstance(build_target("h", http_config()), HttpTarget)
    assert isinstance(build_target("o", openai_config()), OpenAITarget)


def test_build_target_wraps_a_live_target_in_a_cassette(tmp_path: Path):
    """Replay what is recorded, call through for the rest — one object, two modes,
    which is why the recorder and the replayer cannot drift apart."""
    cassette = tmp_path / "main.json"
    cassette.write_text(json.dumps({"version": 1, "entries": {}}), encoding="utf-8")
    target = build_target("s", config(cassette=str(cassette)), record_to=str(cassette))
    assert isinstance(target, CassetteTarget)
    run(target)
    asyncio.run(target.aclose())
    assert "gqa-0001/0" in json.loads(cassette.read_text(encoding="utf-8"))["entries"]


def test_the_label_prefers_the_config_over_the_key():
    assert build_target("key", config(label="retrieval-v3")).label == "retrieval-v3"
    assert build_target("key", config()).label == "key"


def test_pricing_uses_the_rates_from_the_config():
    target = ScriptedTarget(
        "s",
        config(
            callable="fake_agents:as_dict",
            price_prompt_per_mtok=3.0,
            price_completion_per_mtok=15.0,
        ),
    )
    trace = run(target)
    # 120 prompt tokens (100 of them cached, and with no cache rate configured they are
    # billed at the full prompt rate) at $3/MTok, plus 18 completion at $15/MTok
    assert trace.cost_usd is not None
    assert float(trace.cost_usd) == pytest.approx(0.00063, abs=1e-6)
