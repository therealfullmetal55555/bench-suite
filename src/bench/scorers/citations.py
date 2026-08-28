"""Grounded answers: are the citations real, and is "I don't know" allowed.

The family where a benchmark can be exact without a model, if the citation format is
fixed. It is fixed here, in one place, and stated in the task-writing docs: **a passage is
cited when the answer contains its id** (`corpus/incidents.md#2026-08-refunds-stopped`) or
the anchor part of it (`#2026-08-refunds-stopped`). Numbers in brackets are not accepted,
because `[1]` depends on an ordering the entrant chose and a claim about which document
was used should not be ambiguous between two runs.

Three checks, and they answer three different questions:

* **`citations`** — were the passages the task names actually cited (`must_cite`), and were
  the citations real? A citation is fake when it names a passage the corpus does not have
  (`must_not_cite_unsupported` catches it), and unsupported when it names a passage the
  trace never retrieved — you cannot have read what you did not fetch.
* **`answerable`** — the question is answerable, or it is not. When it is not, the correct
  behaviour is to say so, and the task sets `value: false`. This is a refusal detector
  with a published pattern list (see below) rather than a judge, deliberately.
* **`judge`** — not here. When nuance matters, the task says so and pays for a judge.

**The refusal detector, and its honest limits.** `answerable: false` passes when the answer
matches one of the phrases in `REFUSALS`, and fails with the first sentence of the answer
quoted otherwise. A pattern list has false negatives — an agent that says "my knowledge
does not extend to that" in a novel way is marked wrong — and that is a known cost of
making this family reproducible and free. Tasks where the phrasing genuinely matters use a
judge expectation instead, and the family's published judge-share column shows how much of
it does.

The `citations` note reports precision and recall explicitly, because they fail for
different reasons and a single number hides which one happened: recall is "you did not cite
the document you were supposed to", precision is "you cited something else".
"""

from __future__ import annotations

import re

from ..core.corpus import Corpus
from ..core.task import Expectation
from .base import ExpectationResult, ScoringContext

#: Phrases that count as "the corpus does not cover this". Lowercased, matched as
#: substrings of the answer. Short and readable on purpose: this list is part of the
#: benchmark's definition of a correct refusal, so it belongs in the repository where a
#: reviewer can read it and a diff can show it changing.
REFUSALS = (
    "does not cover",
    "doesn't cover",
    "do not have information",
    "don't have information",
    "no information",
    "not in the documentation",
    "not documented",
    "cannot find",
    "can't find",
    "could not find",
    "couldn't find",
    "i don't know",
    "i do not know",
    "unable to answer",
    "outside the documentation",
)

#: A citation-shaped string: a path with an optional `#anchor`, **or** a bare `#anchor`.
#: Used to find *fake* citations — an answer naming a document the corpus does not have is
#: a specific and serious failure that a `must_cite` check on its own never notices.
#:
#: The path alternative comes first and is greedy, so `support/policy.md#refunds` parses as
#: one token and not as a path plus an anchor of its own. That mattered: with anchor-only
#: matching, an answer citing a *different* document that happens to share an anchor
#: satisfied a `must_cite` for the document under test, which is the exact failure the
#: check exists to catch, reported as a pass.
CITATION_SHAPE = re.compile(r"[\w./-]+\.(?:md|txt)(?:#[\w-]+)?")
_CITATION_TOKEN = re.compile(r"[\w./-]+\.(?:md|txt)(?:#[\w-]+)?|#[\w-]+")


def check_citations(context: ScoringContext, expectation: Expectation) -> ExpectationResult:
    must_cite = [str(item) for item in expectation.params.get("must_cite", [])]
    forbid_unsupported = bool(expectation.params.get("must_not_cite_unsupported"))
    answer = context.answer
    corpus = context.corpus

    cited = _cited_in(answer, corpus, also=must_cite)
    retrieved = {passage.id for passage in context.trace.retrieved}

    problems: list[str] = []
    missing = [passage for passage in must_cite if passage not in cited]
    if missing:
        problems.append("did not cite " + ", ".join(missing))

    if forbid_unsupported and corpus is not None:
        known = _known_tokens(corpus)
        unknown = sorted(
            token
            for token in _tokens(answer)
            if token not in known and token not in {item.lower() for item in must_cite}
        )
        if unknown:
            problems.append("cited a passage the corpus does not have: " + ", ".join(unknown))

    if forbid_unsupported and retrieved:
        # Only when the trace reports what it retrieved: an entrant that does not report
        # retrieval is not lying by omission, it is silent, and silence is not scored.
        #
        # `must_cite` is *not* exempt. It is tempting to exempt it — the task said to cite
        # this passage, so citing it should not be a problem — and that is backwards: you
        # cannot have read what you did not fetch, and a required citation with nothing
        # behind it is the specific failure the citation family exists to catch.
        #
        # The comparison keeps the anchor. Folding it away would call a cite of
        # `docs/incident.md#what-we-tried` a match for a retrieval of
        # `docs/incident.md#root-cause` — same file, different evidence — which is the
        # mistake this line exists to catch, so anchors are compared as full and bare
        # forms and anything else is unsupported.
        have = {form for entry in retrieved for form in _forms(entry)}
        unsupported = sorted(item for item in cited if not (_forms(item) & have))
        if unsupported:
            problems.append("cited without retrieving: " + ", ".join(unsupported))

    if problems:
        return ExpectationResult(
            "citations",
            False,
            "; ".join(problems) + f" (cited: {_list(cited) or 'nothing'})",
            detail={"cited": sorted(cited), "must_cite": must_cite, "retrieved": sorted(retrieved)},
        )

    recall = (
        1.0
        if not must_cite
        else len([item for item in must_cite if item in cited]) / len(must_cite)
    )
    precision = 1.0 if not cited else len(cited & retrieved) / len(cited) if retrieved else 1.0
    return ExpectationResult(
        "citations",
        True,
        f"citations check out (recall {recall:.0%}, precision {precision:.0%})",
        detail={"cited": sorted(cited), "recall": recall, "precision": precision},
    )


def check_answerable(context: ScoringContext, expectation: Expectation) -> ExpectationResult:
    """`value: true` means the corpus answers it; `value: false` means it must not pretend.

    Both directions matter and they fail differently: a task marked unanswerable catches
    the confident invention, and a task marked answerable catches the agent that refuses
    everything it is unsure about — which reads as caution and is a failure.
    """
    value = expectation.params.get("value")
    if not isinstance(value, bool):
        return ExpectationResult(
            "answerable", False, "an answerable expectation needs `value: true|false`", scored=False
        )
    refused = _refusal_in(context.answer)
    if value is False:
        if refused:
            return ExpectationResult(
                "answerable", True, f"correctly declined to answer ({refused!r})"
            )
        return ExpectationResult(
            "answerable",
            False,
            "the corpus cannot answer this and the agent answered anyway: "
            f"{_quote(context.answer)}",
        )
    if refused:
        return ExpectationResult(
            "answerable",
            False,
            f"the corpus answers this and the agent declined ({refused!r})",
        )
    return ExpectationResult("answerable", True, "answered a question the corpus can answer")


# --- helpers ------------------------------------------------------------------


def _cited_in(answer: str, corpus: Corpus | None, also: list[str] | None = None) -> set[str]:
    """Which passages the answer names, from the corpus and from `must_cite`.

    The corpus is the authority on what *exists*; the `must_cite` list is what the task
    *wants named*, and it is checked whether or not a corpus was loaded. Letting the
    corpus decide both was wrong in a way that only showed up in a smoke test: a task
    scored without its corpus loaded reported "did not cite X" about an answer that
    cited X, which is the single worst kind of benchmark bug — a wrong note, on the
    report card, in the direction of blaming the entrant.

    Both the full id and the bare anchor count, because `#root-cause` alone is how people
    cite in prose and insisting on the path would fail answers that are perfectly well
    attributed. Unknown citation-shaped strings are the `must_not_cite_unsupported`
    check's business, not this function's.
    """
    found: set[str] = set()
    tokens = _tokens(answer)
    ids = list(also or [])
    if corpus is not None:
        ids.extend(corpus.ids())
    for passage_id in dict.fromkeys(ids):
        lowered = passage_id.lower()
        anchor = "#" + lowered.split("#", 1)[1] if "#" in lowered else ""
        if lowered in tokens or (anchor and anchor in tokens):
            found.add(passage_id)
    return found


def _forms(passage_id: str) -> set[str]:
    """The two ways a target may name one passage: full id, and bare anchor.

    Both are legitimate — `docs/incident.md#root-cause` in a footnote, `#root-cause` in
    prose — and both are kept, because the identity of a passage is its document *and* its
    anchor. Comparing documents alone would call two different sections of one file the
    same evidence.
    """
    lowered = passage_id.lower()
    if "#" not in lowered:
        return {lowered}
    return {lowered, "#" + lowered.split("#", 1)[1]}


def _tokens(answer: str) -> set[str]:
    """Every citation-shaped token in the answer, lowercased."""
    return {match.group(0).lower() for match in _CITATION_TOKEN.finditer(answer)}


def _known_tokens(corpus: Corpus) -> set[str]:
    """Tokens the corpus can justify: every id, and every id's bare anchor."""
    known: set[str] = set()
    for passage_id in corpus.ids():
        lowered = passage_id.lower()
        known.add(lowered)
        if "#" in lowered:
            known.add("#" + lowered.split("#", 1)[1])
    return known


def _refusal_in(answer: str) -> str | None:
    lowered = answer.lower()
    for phrase in REFUSALS:
        if phrase in lowered:
            return phrase
    return None


def _list(items: set[str] | list[str]) -> str:
    return ", ".join(sorted(items))


def _quote(text: str, limit: int = 140) -> str:
    flat = " ".join(text.split())
    if not flat:
        return "(empty answer)"
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


CHECKS = {
    "citations": check_citations,
    "answerable": check_answerable,
}
