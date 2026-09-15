"""The trace contract, and the published schema.

Two things are being tested here, and only one of them is about code.

The first is the contract itself: what a target has to produce, and what happens when
it produces nothing, or something broken. The second is that the *published* JSON
Schema in `schema/trace.schema.json` still describes the models in `core/trace.py`.
A schema file that is published and stale is worse than no schema file at all: an
integration written against it fails months later in somebody else's repository.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import jsonschema
import pytest

from bench.core import trace as trace_module
from bench.core.trace import Message, Passage, TokenUsage, ToolCall, Trace, write_schema

SCHEMA_FILE = Path(__file__).resolve().parents[1] / "schema" / "trace.schema.json"


# --- the contract ------------------------------------------------------------


def test_a_trace_of_a_simple_answer_round_trips():
    trace = Trace(
        messages=[Message(role="user", content="hello"), Message(role="assistant", content="hi")],
        output="hi",
        tokens=TokenUsage(prompt=10, completion=2),
        latency_ms=120,
    )
    restored = Trace.model_validate(json.loads(trace.model_dump_json()))
    assert restored == trace
    assert restored.tool_names() == []
    assert not restored.failed


def test_a_tool_call_keeps_its_arguments_and_result():
    trace = Trace(
        tool_calls=[
            ToolCall(name="lookup_order", arguments={"order_id": "ORD-1"}, result="shipped")
        ]
    )
    assert trace.tool_names() == ["lookup_order"]
    assert trace.tool_calls[0].arguments["order_id"] == "ORD-1"


def test_a_negative_latency_on_a_tool_call_is_refused():
    """A negative duration means the clock or the integration is wrong, and it would
    quietly improve an entry's p95."""
    with pytest.raises(ValueError, match="negative"):
        ToolCall(name="lookup_order", latency_ms=-5)


def test_a_failure_is_a_value_not_an_exception():
    """The decision the whole harness rests on: a hundred tasks still produce a
    hundred results, and the report can say "37 tasks failed because the endpoint
    refused connections" instead of stopping at the first one."""
    trace = Trace(error="ConnectError: connection refused", latency_ms=3)
    assert trace.failed
    assert trace.output is None
    assert trace.tool_names() == []


def test_cost_is_none_when_nothing_priced_it():
    """Not zero. `$0.0000` in a cost column is a claim that a run was free."""
    assert Trace().cost_usd is None
    assert Trace(cost_usd=0.0).cost_usd == 0.0


def test_an_unknown_field_is_refused():
    """`extra="forbid"` everywhere in the contract: a typo in an integration should
    be an error at the boundary, not a field that is silently dropped and a score
    that is quietly computed from nothing."""
    with pytest.raises(ValueError):
        Trace.model_validate({"output": "hi", "tokns": {"prompt": 1}})


def test_a_passage_keeps_its_text_so_a_score_survives_the_corpus_moving_on():
    trace = Trace(retrieved=[Passage(id="docs/a.md#root-cause", text="the reason", score=0.7)])
    assert trace.retrieved[0].id == "docs/a.md#root-cause"
    assert trace.retrieved[0].text == "the reason"


def test_token_usage_totals_prompt_and_completion():
    usage = TokenUsage(prompt=100, completion=20, cached=30)
    assert usage.total == 120
    assert usage.cached == 30  # cached is a subset of prompt, not an addition


def test_the_role_vocabulary_is_the_openai_one():
    """So a trace recorded from a real agent round-trips without a mapping table in
    the middle."""
    for role in ("system", "user", "assistant", "tool"):
        assert Message(role=role, content="x").role == role
    with pytest.raises(ValueError):
        Message(role="moderator", content="x")


def test_a_tool_message_can_name_the_tool_it_answers():
    message = Message(role="tool", content="ORD-1 shipped", name="lookup_order")
    assert message.name == "lookup_order"


# --- the published schema ----------------------------------------------------


def test_the_published_schema_is_current():
    """The drift check. Regenerate with `bench schema write`; the failure message says
    which of the two files to change."""
    published = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    generated = json.loads(json.dumps(trace_module.schema()))
    assert published == generated, "schema/trace.schema.json is stale — run `bench schema write`"


def test_the_schema_accepts_a_full_trace_and_rejects_a_typo():
    """The point of publishing a schema: somebody writing an integration in Go gets
    the same answer the pydantic models would give, without installing Python."""
    schema_doc = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    full = Trace(
        messages=[Message(role="assistant", content="the permission change")],
        output="the permission change",
        tool_calls=[ToolCall(name="lookup_order", arguments={"order_id": "ORD-1"})],
        retrieved=[Passage(id="docs/a.md", text="text")],
        tokens=TokenUsage(prompt=10, completion=4, cached=2),
        latency_ms=88,
        cost_usd=0.0001,
        steps=2,
        raw={"model": "example"},
    ).model_dump(mode="json")
    jsonschema.validate(full, schema_doc)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"tokns": {}}, schema_doc)


def test_the_schema_declares_that_it_is_generated():
    """So that nobody edits it by hand and expects the edit to survive."""
    document = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    assert "do not edit by hand" in document["description"]
    assert document["additionalProperties"] is False


def test_writing_the_schema_is_idempotent(tmp_path: Path):
    first = write_schema(tmp_path / "trace.schema.json").read_text(encoding="utf-8")
    second = write_schema(tmp_path / "trace.schema.json").read_text(encoding="utf-8")
    assert first == second
    assert first.endswith("\n")


# --- layering ----------------------------------------------------------------


def test_core_does_not_import_the_scorers():
    """The layering the package docstrings claim, checked rather than asserted in
    prose. `scorers` imports the models from `core`; if `core` imported `scorers` back
    there would be a cycle, and cycles in this directory would appear as an import
    error for a new user and not for anybody who has been running the tests all
    along."""
    core = Path(trace_module.__file__).parent
    offenders: list[str] = []
    for file in sorted(core.glob("*.py")):
        tree = ast.parse(file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("bench.scorers"):
                offenders.append(f"{file.name}:{node.lineno}")
    assert offenders == []


def test_the_core_package_exports_the_names_the_layering_promises():
    import bench.core as core_package

    assert core_package.__doc__ and "runner" in core_package.__doc__
