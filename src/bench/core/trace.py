"""The trace contract, and the schema file that publishes it.

A benchmark is only useful if the thing it measures is portable. So the shape a
target has to produce is written down twice on purpose: once as pydantic models that
the harness uses, and once as a JSON Schema at `schema/trace.schema.json` that
somebody writing an integration in Go can read without installing anything. A test
regenerates the second from the first and fails when they drift, because a schema
file that is published and stale is worse than no schema file.

The other half of this file is the deliberate decision from `passmark`: **a failure
is a value.** A target that times out produces a `Trace` with `error` set, so a
hundred tasks still produce a hundred results and the report can say "37 tasks
failed because the endpoint refused connections" instead of stopping at the first
one. The one exception is a misconfigured target — an endpoint that will not parse
as a URL — which raises and stops the run, because a hundred identical errors is a
hundred pieces of evidence that nobody configured anything.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schema" / "trace.schema.json"
"""Where the published schema lives, relative to the installed package.

Deliberately *not* inside the package: it is a document for other people, and it
belongs where a person browsing the repository will find it.
"""

Role = Literal["system", "user", "assistant", "tool"]
Outcome = Literal["pass", "fail", "error", "over-budget", "skipped"]


class Message(BaseModel, extra="forbid"):
    role: Role
    content: str = ""
    name: str | None = None
    """For `role: tool`, which tool this is the result of. Matches the OpenAI wire
    format so that a trace recorded from a real agent round-trips without a mapping
    table in the middle."""


class ToolCall(BaseModel, extra="forbid"):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str | None = None
    error: str | None = None
    latency_ms: int | None = None

    @field_validator("latency_ms")
    @classmethod
    def _non_negative(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("latency_ms cannot be negative")
        return value


class Passage(BaseModel, extra="forbid"):
    """One retrieved chunk, kept with the trace rather than looked up later.

    The reason is that a score has to be explainable a year from now: "the answer
    cited passage 3, and passage 3 says the rollout was stopped by a permission
    change" is a claim that survives the corpus being edited. A retrieval id alone
    does not.
    """

    id: str
    text: str
    score: float | None = None
    """The retriever's own relevance score, when it has one. Reported, never used
    for grading: every retriever's scale is its own."""


class TokenUsage(BaseModel, extra="forbid"):
    prompt: int = 0
    completion: int = 0
    cached: int = 0
    """Tokens served from a provider-side prompt cache. Tracked separately because
    it is the difference between an expensive agent and a cheap one with a long
    system prompt, and the cost column is wrong without it."""

    @property
    def total(self) -> int:
        return self.prompt + self.completion


class Trace(BaseModel, extra="forbid"):
    """What a target produces per task. Everything but `messages` is optional,
    because a target that cannot report cost should be scored on what it can
    report rather than refused."""

    messages: list[Message] = Field(default_factory=list)
    output: str | None = None
    """The final answer, when there is one. Pulled out separately because scorers
    spend most of their time here and reaching into the last assistant message to
    find it is the kind of thing that breaks when a target adds a summary turn."""

    tool_calls: list[ToolCall] = Field(default_factory=list)
    retrieved: list[Passage] = Field(default_factory=list)
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: int = 0
    cost_usd: Decimal | None = None
    """`None` means "not measured", which is not the same as `0`. A leaderboard that
    prints `$0.0000` for an agent nobody priced is claiming it is free."""

    steps: int | None = None
    """Turns or tool rounds, whichever the target counts. Compared against a task's
    `max_steps` when the target reports it; a target that does not is not penalised
    for silence."""

    error: str | None = None
    """Set when the target failed. A trace with an error is a result, not an
    exception — see the module docstring."""

    raw: dict[str, Any] = Field(default_factory=dict)
    """Provider response ids, retry counts, whatever the integration wants to keep
    for debugging. Never read by the harness."""

    @property
    def failed(self) -> bool:
        return self.error is not None

    def tool_names(self) -> list[str]:
        return [call.name for call in self.tool_calls]


def schema() -> dict[str, Any]:
    """The published JSON Schema, generated from the models above."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://github.com/therealfullmetal55555/bench-suite/schema/trace.schema.json",
        "title": "bench trace",
        "description": (
            "What a target must produce for one task. Generated from "
            "src/bench/core/trace.py — do not edit by hand, run `bench schema write`."
        ),
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "messages": {"type": "array", "items": Message.model_json_schema()},
            "output": {"type": ["string", "null"]},
            "tool_calls": {"type": "array", "items": ToolCall.model_json_schema()},
            "retrieved": {"type": "array", "items": Passage.model_json_schema()},
            "tokens": TokenUsage.model_json_schema(),
            "latency_ms": {"type": "integer", "minimum": 0},
            "cost_usd": {"type": ["string", "number", "null"]},
            "steps": {"type": ["integer", "null"], "minimum": 0},
            "error": {"type": ["string", "null"]},
            "raw": {"type": "object"},
        },
    }


def write_schema(path: Path | None = None) -> Path:
    """Write the schema where it is published. Called by the CLI and by the test
    that checks the published copy is current."""
    target = path or SCHEMA_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(schema(), indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return target
