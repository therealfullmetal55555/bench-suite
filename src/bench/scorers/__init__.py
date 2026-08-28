"""The scorer registry, and the vocabulary a task file is allowed to use.

The implementations arrive one family at a time (see SPEC §12), and the registry
comes first on purpose: the task linter checks every expectation against it at load
time, so a task file that says `type: citatons` fails before a run instead of
scoring zero for a reason that appears in no error message. That is worth more than
the ordering convenience of writing the registry afterwards.

Each entry says which families use the type, so the linter can also catch a
`citations` expectation in a `tool-trajectory` task — a mistake that a run would
otherwise accept and quietly score as a failure.
"""

from __future__ import annotations

from typing import NamedTuple


class ExpectationType(NamedTuple):
    name: str
    families: tuple[str, ...]
    deterministic: bool
    """Whether it can be decided without a model. Reported per family: the share of
    a family's score that rests on a judge is the single most useful number in this
    benchmark, and a family above 50% is flagged on the board."""


EXPECTATION_TYPES: dict[str, ExpectationType] = {
    # --- available to every family ----------------------------------------
    "contains": ExpectationType("contains", (), deterministic=True),
    "not_contains": ExpectationType("not_contains", (), deterministic=True),
    "regex": ExpectationType("regex", (), deterministic=True),
    "json": ExpectationType("json", (), deterministic=True),
    "schema": ExpectationType("schema", (), deterministic=True),
    "judge": ExpectationType("judge", (), deterministic=False),
    # --- grounded-qa -------------------------------------------------------
    # `multi-turn-state` joined this list the day the first generated conversation cited a
    # turn id, which is the second time the family-exclusivity rule has been the thing that
    # was wrong rather than the task. Citations are about a trace that names its sources:
    # every grounded task does, and so does a conversation whose turns are the sources.
    "citations": ExpectationType(
        "citations", ("grounded-qa", "multi-turn-state"), deterministic=True
    ),
    # Deterministic on purpose, and the honesty is in the phrase list rather than in a
    # model call: `answerable` is decided by `REFUSALS` in `scorers/citations.py`, which
    # a reviewer can read and a diff can change. It has false negatives — an agent that
    # declines in words nobody listed is marked wrong — and tasks where the phrasing
    # matters carry a `judge` expectation instead. Marking it non-deterministic would
    # instead hide the family's judged share behind a check that never calls a model.
    "answerable": ExpectationType("answerable", ("grounded-qa",), deterministic=True),
    # --- tool-trajectory, and anywhere else there are tools ----------------
    # `grounded-qa` is here on purpose: "did it retrieve before it answered" is a
    # trajectory check, and the family that cares most about citations is also the one
    # that cares whether anything was looked up. Found by writing such a task into
    # examples/ and having the linter refuse it — which is the rule working, and the
    # rule being one family too narrow.
    # Any family, and the list is empty for that reason: every family can have tools, and
    # the two checks that read tool calls are not a property of the family — a safety task
    # that must not call `transfer_funds` is a trajectory check, and so is a conversation
    # that must not refund twice. Narrowing this to three families caught the generator's
    # own output twice, which is the rule telling us the rule was wrong.
    "trajectory": ExpectationType("trajectory", (), deterministic=True),
    "steps": ExpectationType("steps", (), deterministic=True),
    # --- structured-extraction --------------------------------------------
    "fields": ExpectationType("fields", ("structured-extraction",), deterministic=True),
    # --- multi-turn-state --------------------------------------------------
    "state": ExpectationType("state", ("multi-turn-state",), deterministic=True),
    # --- refusal-and-safety ------------------------------------------------
    "refusal": ExpectationType("refusal", ("refusal-and-safety",), deterministic=True),
}

KNOWN_TYPES: frozenset[str] = frozenset(EXPECTATION_TYPES)
"""What `task.py` checks a `type:` against. An empty `families` tuple means any
family may use it."""

__all__ = ["EXPECTATION_TYPES", "KNOWN_TYPES", "ExpectationType"]
