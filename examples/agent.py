"""The example entrant: a small support agent in two builds.

Written the way an integration actually looks — one function, the task's input in, an
answer out — and deliberately *not* a good agent. It is a fixture for the harness: it
retrieves from the corpus in `examples/tasks/corpus/` with the retriever that ships in
`bench.core.corpus`, answers from the passage it found, and records the tool call it
made, so the deterministic scorers have something real to score.

Two versions, selected by `AGENT_VERSION`, because the first thing anybody does with a
benchmark is measure a change:

* `v1` — cites what it retrieved, refuses when the corpus has nothing.
* `v2` — a plausible edit that makes the evals worse: it cites without quoting, gives up
  the tool call, and answers confidently when the corpus is silent. Every one of those is
  a real regression shape from the incident write-ups this corpus is built out of.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from bench.core.corpus import Corpus
from bench.core.trace import Passage, ToolCall, Trace

CORPUS = Corpus.load(Path(__file__).parent / "tasks" / "corpus")
VERSION = os.environ.get("AGENT_VERSION", "v1")


def answer(case_input: dict[str, Any], *, repeat: int = 0) -> Trace:
    question = str(case_input.get("question", ""))
    tool_calls: list[ToolCall] = []
    retrieved: list[Passage] = []

    if VERSION == "v1":
        hits = CORPUS.search(question, limit=2)
        if not hits:
            # The honest answer, and the one `answerable: false` tasks are written to
            # check. v2 loses this behaviour, which is what the benchmark is for.
            return Trace(
                output="The documentation does not cover that.",
                latency_ms=12,
                tokens=_tokens(question, 12),
            )
        tool_calls.append(
            ToolCall(name="search_docs", arguments={"query": question}, result=hits[0].id)
        )
        retrieved = hits
        top = hits[0]
        body = _first_sentence(top.text)
        output = f"{body} (source: {top.id})"
    elif VERSION == "v2":
        hits = CORPUS.search(question, limit=1)
        retrieved = hits
        if hits:
            body = _first_sentence(hits[0].text)
            output = f"{body} — see the docs."  # no citation, and no tool call
        else:
            output = "I think the rollout was stopped because of the prompt."
    else:  # pragma: no cover - a version that does not exist is a configuration error
        raise ValueError(f"unknown AGENT_VERSION {VERSION!r}")

    return Trace(
        output=output,
        tool_calls=tool_calls,
        retrieved=retrieved,
        latency_ms=40 if VERSION == "v2" else 25,
        tokens=_tokens(question, len(output)),
        steps=len(tool_calls),
        raw={"agent_version": VERSION},
    )


def _first_sentence(text: str) -> str:
    body = text.split("\n\n", 1)[-1].strip()
    for stop in (". ", ".\n"):
        if stop in body:
            return body.split(stop, 1)[0].strip() + "."
    return body


def _tokens(question: str, answer_length: int) -> dict[str, int]:
    """Rough but not silly: ~4 characters per token on English prose, and the corpus
    block that goes in the prompt counted at the same rate."""
    return {
        "prompt": (len(question) + 900) // 4,
        "completion": answer_length // 4,
        "cached": 600 // 4,
    }
