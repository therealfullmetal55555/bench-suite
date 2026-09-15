"""Fixture agents, loaded by import path.

Each one is a shape the boundary has to handle, and the names say which: `echo`
returns a string, `as_dict` a mapping, `as_object` somebody else's response object,
`wrong` something with no answer in it at all. `flaky` exists so the runner's flakiness
accounting can be tested without a network.

This file has to stay importable and side-effect free, because it is imported by the
harness the same way an entrant's module is.
"""

from __future__ import annotations

from typing import Any


def echo(case_input: dict[str, Any], *, repeat: int = 0) -> str:
    """The answer is the question. The most common integration there is."""
    return str(case_input.get("question", case_input.get("prompt", "")))


def as_dict(case_input: dict[str, Any], *, repeat: int = 0) -> dict[str, Any]:
    return {
        "output": "the permission change stopped it",
        "tool_calls": [{"name": "lookup_order", "arguments": {"order_id": "ORD-1"}}],
        "tokens": {"prompt": 120, "completion": 18, "cached": 100},
        "latency_ms": 7,
        "steps": 2,
    }


async def async_answer(case_input: dict[str, Any], *, repeat: int = 0) -> str:
    return "answered asynchronously"


class Answer:
    """Somebody else's response object: an attribute called `.output`, a `.usage` with
    the OpenAI field names, and tool calls in the wire shape."""

    def __init__(self) -> None:
        self.output = "from an object"
        self.latency_ms = 42
        self.usage = type("Usage", (), {"prompt_tokens": 30, "completion_tokens": 4})()
        self.tool_calls = [
            type(
                "Call",
                (),
                {
                    "function": type(
                        "Function",
                        (),
                        {"name": "lookup_order", "arguments": '{"order_id": "ORD-2"}'},
                    )()
                },
            )()
        ]


def as_object(case_input: dict[str, Any], *, repeat: int = 0) -> Answer:
    return Answer()


def no_repeat_argument(case_input: dict[str, Any]) -> str:
    """A callable that does not take `repeat` — most integrations do not, and forcing
    the keyword on them would mean every entrant writes a wrapper."""
    return "no repeat keyword here"


def explode(case_input: dict[str, Any], *, repeat: int = 0) -> str:
    raise RuntimeError("the retrieval index is not mounted")


def wrong(case_input: dict[str, Any], *, repeat: int = 0) -> int:
    """Not a trace, not a dict, not a string, no `.output` — the mistake a person makes
    while writing an integration, and the one the error message has to explain."""
    return 42


def flaky(case_input: dict[str, Any], *, repeat: int = 0) -> str:
    """Passes on even repeats. Exists so the runner's flakiness accounting can be tested
    without reaching for the network, and so the board's asterisk has a fixture."""
    return "correct" if repeat % 2 == 0 else "wrong"


def slow(case_input: dict[str, Any], *, repeat: int = 0) -> str:
    """Reports a cost, for the budget tests. Does not actually sleep: a test that sleeps
    is a test that fails on a busy CI runner."""
    return "expensive"


def costly(case_input: dict[str, Any], *, repeat: int = 0) -> dict[str, Any]:
    """Two cents a call, for the tests that need a run to blow a budget without waiting
    for one. A test that sleeps is a test that fails on a busy CI runner."""
    return {"output": "expensive answer", "cost_usd": "0.02"}
