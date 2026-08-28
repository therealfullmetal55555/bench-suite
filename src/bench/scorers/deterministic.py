"""The five checks that need nothing but the text.

`contains`, `not_contains`, `regex`, `json`, `schema`. They are the ones a reviewer can
disagree with by reading them, which is the point: a rule you can read is a spec you can
argue with, and a dataset that is mostly judge-scored has no spec at all.

Two things here are more careful than they look.

**`not_contains` is checked against the whole answer, not the last line.** The classic
false pass is an agent that says "I will not tell you the refund was processed" — the
forbidden phrase is present, the sentence is a refusal, and a naive substring check
fails it. This one still fails it, deliberately: the benchmark's rule is "this phrase
must not appear", and the task that wants the nuance writes a judge expectation instead.
Being strict and legible beats being clever and unexplainable, and the task author is
the one who decides which they wanted.

**`json` and `schema` never repair the answer.** An agent that wraps its JSON in a
markdown fence has failed the JSON contract, and the note says so with the first line of
what it actually produced. Extracting the fenced block would raise the score of an agent
that is not following the instruction it was given — and the tasks that *do* want a fence
tolerated set `fenced: true` explicitly, in the task file, where a reviewer sees it.
"""

from __future__ import annotations

import json
import re
from typing import Any

import jsonschema

from ..core.task import Expectation
from .base import ExpectationResult, ScoringContext

_FENCE = re.compile(r"^\s*```(?:json)?\s*(?P<body>.*?)\s*```\s*$", re.DOTALL)


def check_contains(context: ScoringContext, expectation: Expectation) -> ExpectationResult:
    value = _required(expectation, "value")
    needle = str(value)
    if str(needle).lower() in context.answer.lower():
        return ExpectationResult("contains", True, f"the answer contains {needle!r}")
    return ExpectationResult(
        "contains",
        False,
        f"the answer does not contain {needle!r}: {_quote(context.answer)}",
    )


def check_not_contains(context: ScoringContext, expectation: Expectation) -> ExpectationResult:
    value = _required(expectation, "value")
    needle = str(value)
    if needle.lower() not in context.answer.lower():
        return ExpectationResult("not_contains", True, f"{needle!r} is absent")
    return ExpectationResult(
        "not_contains",
        False,
        f"the answer says {needle!r}: {_quote(context.answer)}",
    )


def check_regex(context: ScoringContext, expectation: Expectation) -> ExpectationResult:
    pattern = str(_required(expectation, "pattern"))
    try:
        compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    except re.error as exc:
        return ExpectationResult(
            "regex", False, f"the pattern {pattern!r} does not compile: {exc}", scored=False
        )
    if compiled.search(context.answer):
        return ExpectationResult("regex", True, f"/{pattern}/ matches")
    return ExpectationResult(
        "regex", False, f"/{pattern}/ does not match: {_quote(context.answer)}"
    )


def check_json(context: ScoringContext, expectation: Expectation) -> ExpectationResult:
    """The answer parses as JSON, and optionally has these keys at these values.

    `equals` compares loosely on purpose: `77.5` and `"77.50"` are the same total, and a
    benchmark that fails an extraction over a trailing zero is measuring the formatter.
    Anything where the difference matters gets a `fields` expectation, which is exact.
    """
    text = _maybe_unfenced(context.answer, expectation)
    try:
        document = json.loads(text)
    except ValueError as exc:
        return ExpectationResult(
            "json", False, f"the answer is not JSON ({exc}): {_quote(context.answer)}"
        )

    wanted = expectation.params.get("equals")
    if isinstance(wanted, dict):
        if not isinstance(document, dict):
            return ExpectationResult(
                "json", False, f"the answer is a {type(document).__name__}, not an object"
            )
        wrong = [
            key for key, value in wanted.items() if not _loosely_equal(document.get(key), value)
        ]
        if wrong:
            return ExpectationResult(
                "json",
                False,
                "wrong value for " + ", ".join(sorted(wrong)) + f": {_quote(json.dumps(document))}",
            )
        return ExpectationResult(
            "json", True, f"JSON with the right values for {', '.join(sorted(wanted))}"
        )

    keys = expectation.params.get("keys")
    if isinstance(keys, list) and document and isinstance(document, dict):
        missing = [str(key) for key in keys if key not in document]
        if missing:
            return ExpectationResult("json", False, f"missing key(s): {', '.join(missing)}")
    return ExpectationResult("json", True, "the answer parses as JSON")


def check_schema(context: ScoringContext, expectation: Expectation) -> ExpectationResult:
    """JSON Schema validation against the task's own schema.

    `jsonschema` rather than a hand-rolled subset: the schemas in these tasks are written
    by hand and a validator that quietly ignores `patternProperties` would pass an answer
    the author believed they had excluded.
    """
    schema = expectation.params.get("schema")
    if not isinstance(schema, dict):
        return ExpectationResult(
            "schema", False, "the expectation has no `schema` object", scored=False
        )
    text = _maybe_unfenced(context.answer, expectation)
    try:
        document = json.loads(text)
    except ValueError as exc:
        return ExpectationResult("schema", False, f"the answer is not JSON: {exc}")

    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(document), key=lambda error: list(error.path))
    if not errors:
        return ExpectationResult("schema", True, "the answer validates against the schema")
    first = errors[0]
    where = "/".join(str(part) for part in first.path) or "(root)"
    more = f" (+{len(errors) - 1} more)" if len(errors) > 1 else ""
    return ExpectationResult("schema", False, f"{where}: {first.message}{more}")


# --- helpers ------------------------------------------------------------------


def _required(expectation: Expectation, key: str) -> Any:
    if key not in expectation.params:
        raise ValueError(f"a {expectation.type} expectation needs `{key}`")
    return expectation.params[key]


def _maybe_unfenced(text: str, expectation: Expectation) -> str:
    """Strip a markdown fence, but only when the task asked for that.

    `fenced: true` in the task file, next to the expectation, so a reviewer reading the
    task sees that leniency was granted on purpose rather than discovering it in a
    scorer.
    """
    if not expectation.params.get("fenced"):
        return text.strip()
    match = _FENCE.match(text)
    return match.group("body").strip() if match else text.strip()


def _loosely_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        # Before the equality shortcut, not after it: `True == 1` in Python, so a check
        # that compares first and asks about booleans second accepts `1` where the task
        # said `true`. Found by the test that asserts the exact opposite, after the
        # comment saying exactly this had already been written.
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    if left == right:
        return True
    if isinstance(left, int | float) and isinstance(right, str):
        try:
            return float(right) == float(left)
        except ValueError:
            return False
    if isinstance(right, int | float) and isinstance(left, str):
        return _loosely_equal(right, left)
    if isinstance(left, str) and isinstance(right, str):
        return left.strip().casefold() == right.strip().casefold()
    return False


def _quote(text: str, limit: int = 140) -> str:
    flat = " ".join(text.split())
    if not flat:
        return "(empty answer)"
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


CHECKS = {
    "contains": check_contains,
    "not_contains": check_not_contains,
    "regex": check_regex,
    "json": check_json,
    "schema": check_schema,
}
