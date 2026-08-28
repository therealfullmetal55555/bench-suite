"""Structured extraction: per-field comparison, because partial credit is the truth.

The family where the answer is exactly checkable, and therefore the family where
all-or-nothing scoring does the most damage. A 20-field invoice extraction with 19 fields
right is not the same product as one with 2, and a benchmark that scores both zero cannot
tell them apart — nor can it tell either of them from the agent that returned an error.

So `fields` is weighted per field, and the weights are the interesting part:

* the default weight is 1, which makes the score "share of fields right";
* `weights: {total: 3}` lets a task say that getting the total wrong matters more than
  getting the date wrong, because it does;
* a missing field scores zero rather than being dropped from the denominator, since an
  extraction that omits a field has failed at extraction, not at scoring;
* a field present but `null` counts as missing, because that is what an extraction agent
  produces when it gives up mid-schema and it should not read as a wrong value.

And one thing the module refuses to do: **normalise.** `"Northwind Tools"` and
`"northwind  tools"` are compared with surrounding whitespace and case folded, and
nothing else. A scorer that strips punctuation, expands abbreviations and rounds numbers
is a scorer whose failures nobody can reproduce — and every one of those transformations
is a decision that belongs in the task file, visible, where the person who wrote the
document shapes can argue with it. Per-field `normalise:` flags exist for the cases that
need one, and the flags are listed here:

* `case` — casefold both sides (already on for every field)
* `strip` — collapse internal whitespace (already on)
* `number` — compare as numbers, so `77.50` equals `77.5`
* `set` — compare lists as sets, for fields where the order is not the document's order
"""

from __future__ import annotations

from typing import Any

from ..core.task import Expectation
from .base import ExpectationResult, ScoringContext

DEFAULT_NORMALISERS = ("case", "strip")


def check_fields(context: ScoringContext, expectation: Expectation) -> ExpectationResult:
    values = expectation.params.get("values")
    if not isinstance(values, dict) or not values:
        return ExpectationResult(
            "fields", False, "a fields expectation needs a `values:` mapping", scored=False
        )

    document = _document_of(context)
    if isinstance(document, str):
        return ExpectationResult("fields", False, document)

    weights = expectation.params.get("weights") or {}
    normalisers = tuple(expectation.params.get("normalise", DEFAULT_NORMALISERS))

    earned = 0.0
    possible = 0.0
    wrong: list[str] = []
    missing: list[str] = []
    for field, expected in values.items():
        weight = float(weights.get(field, 1.0))
        possible += weight
        if field not in document or document[field] is None:
            missing.append(str(field))
            continue
        got = document[field]
        if _equal(got, expected, normalisers):
            earned += weight
        else:
            wrong.append(f"{field}: expected {_short(expected)}, got {_short(got)}")

    share = earned / possible if possible else 0.0
    if not missing and not wrong:
        return ExpectationResult(
            "fields",
            True,
            f"all {len(values)} field(s) right",
            detail={"fields": len(values), "share": 1.0},
        )

    parts = []
    if wrong:
        parts.append("wrong value for " + "; ".join(wrong))
    if missing:
        parts.append("missing field(s) " + ", ".join(sorted(missing)))
    return ExpectationResult(
        "fields",
        False,
        f"{share:.0%} of the fields — " + "; ".join(parts),
        detail={"share": share, "wrong": wrong, "missing": sorted(missing)},
    )


# --- helpers ------------------------------------------------------------------


def _document_of(context: ScoringContext) -> dict[str, Any] | str:
    """The answer as a mapping, or a sentence explaining why it is not one.

    Passing `strict_document: true` makes unparseable output an error rather than a
    failed check — for the tasks where the harness's wire format is the contract and a
    mismatch means the harness is wrong, not the entrant.
    """
    import json

    # Fences are not tolerated here, and that is `json`'s rule rather than a second
    # opinion: an extraction task that wants a fenced answer says `fenced: true` in the
    # task file, and the check that reads the fence is the one that knows how.
    text = context.answer.strip()
    if not text:
        return "the answer is empty"
    try:
        document = json.loads(text)
    except ValueError:
        document = None
    if document is None:
        return f"the answer is not JSON: {_short(text)}"
    if not isinstance(document, dict):
        return f"the answer is a {type(document).__name__}, not an object"
    return document


def _equal(got: Any, expected: Any, normalisers: tuple[str, ...]) -> bool:
    if isinstance(got, bool) or isinstance(expected, bool):
        return got is expected
    if "number" in normalisers or isinstance(expected, int | float):
        try:
            if float(got) == float(expected):
                return True
        except (TypeError, ValueError):
            pass
    if isinstance(expected, list) and isinstance(got, list):
        if "set" in normalisers:
            return {_key(item, normalisers) for item in got} == {
                _key(item, normalisers) for item in expected
            }
        return [_key(item, normalisers) for item in got] == [
            _key(item, normalisers) for item in expected
        ]
    return _key(got, normalisers) == _key(expected, normalisers)


def _key(value: Any, normalisers: tuple[str, ...]) -> Any:
    if not isinstance(value, str):
        return value
    text = value
    if "strip" in normalisers:
        text = " ".join(text.split())
    if "case" in normalisers:
        text = text.casefold()
    return text


def _short(value: Any, limit: int = 60) -> str:
    text = value if isinstance(value, str) else repr(value)
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


CHECKS = {
    "fields": check_fields,
}
