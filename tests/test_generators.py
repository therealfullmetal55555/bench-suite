"""The task set: that it is what the data says, and that the data is true.

Two kinds of test here, and the second kind is the one that earns its keep.

The first kind checks the generator's *mechanics* — ids unique and prefixed, both splits
populated, both halves of every pair present, the committed files identical to what the
generator writes today. That last one is the drift guard: without it, somebody hand-edits a
generated task, `make tasks` silently reverts it, and the difference between "the score
moved" and "the task changed" is lost.

The second kind checks the *claims*. A generated task asserts that a passage says what the
task says it says, and that assertion is either true or the task is a trap: an entrant
fails it for a reason that appears in no error message. So every `must_include` phrase is
looked up in the passage it is attributed to, every required tool is looked up in the tool
list the task ships, and every extraction value is looked up in the document, in the
spelling the document uses. These are the tests a generator earns and a hand-written task
set never gets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from bench.core.corpus import Corpus
from bench.core.task import FAMILIES, PREFIXES, Task, load_tasks
from bench.generators import (
    DEFAULT_ADDED,
    build,
    cap,
    extraction_data,
    long_horizon_data,
    multi_turn_data,
    safety_data,
    trajectory_data,
    write,
)

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks"
DATA = ROOT / "src" / "bench" / "generators" / "data"


@pytest.fixture(scope="module")
def task_set():
    return load_tasks(TASKS)


@pytest.fixture(scope="module")
def families() -> dict[str, list[Task]]:
    built, _ = build()
    return built


def yaml_data(name: str) -> dict[str, Any]:
    return yaml.safe_load((DATA / name).read_text(encoding="utf-8"))


# --- the set as a whole -------------------------------------------------------


def test_the_committed_task_set_is_exactly_what_the_generator_writes(tmp_path: Path) -> None:
    """The drift guard.

    Regenerate into a temporary directory and compare bytes. A hand-edited task file fails
    here, with a diff, rather than surviving until the next `make tasks` silently reverts it
    — and a generator change that was not committed fails here too, which is the other
    direction of the same problem.
    """
    write(tmp_path, added=DEFAULT_ADDED)
    committed = sorted(path.name for path in TASKS.glob("*.yaml"))
    generated = sorted(path.name for path in tmp_path.glob("*.yaml"))
    assert committed == generated
    for name in generated:
        assert (TASKS / name).read_text(encoding="utf-8") == (tmp_path / name).read_text(
            encoding="utf-8"
        ), f"{name} has drifted from its generator — run `make tasks`"


def test_the_set_is_balanced_across_the_splits(task_set) -> None:
    """Equal splits, per family.

    Not a cosmetic rule: the board's private column is the one nobody can tune against, and
    a family that is 90% public makes the aggregate mostly a public score with a private
    footnote. Equal per family is the only shape where that cannot happen quietly.
    """
    for family in FAMILIES:
        public = [task for task in task_set.split("public") if task.family == family]
        private = [task for task in task_set.split("private") if task.family == family]
        assert public, f"{family} has no public tasks"
        assert len(public) == len(
            private
        ), f"{family}: {len(public)} public vs {len(private)} private"


def test_every_id_is_unique_and_prefixed_for_its_family(task_set) -> None:
    seen: set[str] = set()
    for task in task_set:
        assert task.id not in seen, f"{task.id} appears twice"
        seen.add(task.id)
        assert task.id.startswith(f"{PREFIXES[task.family]}-"), task.id


def test_every_private_task_is_tagged_and_grouped(task_set) -> None:
    """A private task with no tags cannot be grouped when the number moves, which is the
    whole diagnostic use of a held-back split."""
    for task in task_set.split("private"):
        assert task.tags, task.id


def test_every_task_has_a_reason_and_a_budget(task_set) -> None:
    for task in task_set:
        assert len(task.because) >= 20, task.id
        assert not task.budget.is_empty, f"{task.id} has no budget"
        assert task.added == DEFAULT_ADDED, task.id


def test_a_cap_is_a_subset_and_never_a_renumbering(families) -> None:
    """`--public 3` keeps the first three, and `gqa-0001` still means the same question.

    The ids end up in reports and in people's notes; a cap that renumbered would make two
    files disagree about what `gqa-0007` is, which is worse than a gap in the numbering.
    """
    capped = cap(families, public=3, private=2)
    for family, tasks in capped.items():
        assert len([task for task in tasks if task.split == "public"]) <= 3, family
        assert len([task for task in tasks if task.split == "private"]) <= 2, family
    assert capped["grounded-qa"][0].id == families["grounded-qa"][0].id
    again = cap(families, public=3, private=2)
    assert [task.id for task in again["grounded-qa"]] == [task.id for task in capped["grounded-qa"]]


def test_generating_twice_gives_the_same_bytes(tmp_path: Path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    write(first)
    write(second)
    for path in sorted(first.glob("*.yaml")):
        assert path.read_text(encoding="utf-8") == (second / path.name).read_text(encoding="utf-8")


# --- grounded-qa: the corpus really says it -----------------------------------


def test_every_grounded_fact_points_at_a_passage_that_exists() -> None:
    for fact in yaml_data("facts.yaml")["facts"]:
        corpus = Corpus.load(TASKS / "corpus" / fact["corpus"])
        assert fact["passage"] in corpus.ids(), f"{fact['id']}: no {fact['passage']}"


def test_every_must_include_phrase_appears_in_the_passage_it_is_attributed_to() -> None:
    """The claim that makes a generated grounded task checkable.

    If `must_include: ["30 days"]` names a passage that does not contain "30 days", the task
    is asking for a phrase the corpus never gives — an entry fails it for a reason that is
    invisible in the report. This is the test that makes editing a document and forgetting
    the facts file a failure rather than a silent trap.
    """
    for fact in yaml_data("facts.yaml")["facts"]:
        corpus = Corpus.load(TASKS / "corpus" / fact["corpus"])
        passage = next(item for item in corpus.passages if item.id == fact["passage"])
        for split in ("public", "private"):
            for phrase in fact[split]["must_include"]:
                # Compared with the whitespace flattened, because the corpus is wrapped
                # prose: "a replacement or\na refund" is the phrase "a replacement or a
                # refund" and a test that says otherwise is a test about line lengths.
                assert phrase.lower() in " ".join(passage.text.split()).lower(), (
                    f"{fact['id']} ({split}): the passage {fact['passage']} does not contain "
                    f"{phrase!r}"
                )


def test_the_grounded_questions_actually_retrieve_their_passage() -> None:
    """Retrieval, not just containment.

    A task whose answer is in the corpus but whose question does not retrieve the passage is
    a task that measures the bundled retriever's quirks rather than the agent's. Not every
    question has to rank its passage first — the corpus is small and BM25 is dumb — but the
    passage has to be in what comes back, or the task is unanswerable in practice while
    claiming to be answerable.
    """
    misses: list[str] = []
    for fact in yaml_data("facts.yaml")["facts"]:
        corpus = Corpus.load(TASKS / "corpus" / fact["corpus"])
        for split in ("public", "private"):
            question = fact[split]["question"]
            hits = {hit.id for hit in corpus.search(question, limit=5)}
            if fact["passage"] not in hits:
                misses.append(f"{fact['id']} ({split}) → {sorted(hits)}")
    assert misses == [], "questions whose passage is not retrieved:\n" + "\n".join(misses)


def test_an_unanswerable_question_retrieves_less_strongly_than_a_typical_answerable_one() -> None:
    """What is actually claimed about the unanswerable half, in one comparison.

    Not that they retrieve nothing: they deliberately share vocabulary with the corpus — the
    same words a real customer would use — so a keyword matcher is tempted. What has to hold
    is the weaker, checkable thing: the best hit for a question the corpus cannot answer
    must score below the *median* answerable question's hit on the passage that does answer
    it. That is what makes it a task rather than a guess, and it is measured against the
    corpus as it stands rather than against a threshold somebody liked the look of.

    (The first version of this test asserted "no strong match" with a hand-picked cut-off of
    4.0 and failed on six of sixteen questions, which is the useful kind of failure: the
    numbers said the threshold was fiction, not the data.)
    """
    answerable: list[float] = []
    for fact in yaml_data("facts.yaml")["facts"]:
        corpus = Corpus.load(TASKS / "corpus" / fact["corpus"])
        for split in ("public", "private"):
            hits = {
                hit.id: hit.score or 0.0 for hit in corpus.search(fact[split]["question"], limit=5)
            }
            answerable.append(hits.get(fact["passage"], 0.0))
    answerable.sort()
    median = answerable[len(answerable) // 2]

    problems: list[str] = []
    tempted = 0
    for entry in yaml_data("unanswerable.yaml")["unanswerable"]:
        corpus = Corpus.load(TASKS / "corpus" / entry["corpus"])
        for split in ("public", "private"):
            hits = corpus.search(entry[split]["question"], limit=1)
            score = (hits[0].score or 0.0) if hits else 0.0
            if score > 0:
                tempted += 1
            if score >= median:
                problems.append(
                    f"{entry['id']} ({split}): top hit scores {score:.2f}, at or above the "
                    f"answerable median {median:.2f}"
                )
    assert problems == [], "unanswerable questions that look answerable:\n" + "\n".join(problems)
    # And the family has to stay tempting: if most of these retrieved nothing at all, they
    # would be easy, and "say you don't know" is only worth measuring when something
    # plausible is on the screen.
    assert tempted >= len(
        yaml_data("unanswerable.yaml")["unanswerable"]
    ), f"only {tempted} of the unanswerable questions retrieve anything"


def test_the_fabrication_lists_do_not_contain_the_real_answer() -> None:
    """A `not_contains` phrase that the corpus *does* contain would fail every correct
    answer, which is the worst kind of task: unwinnable and quietly so."""
    for entry in yaml_data("unanswerable.yaml")["unanswerable"]:
        corpus = Corpus.load(TASKS / "corpus" / entry["corpus"])
        for phrase in entry.get("fabrications", []):
            for passage in corpus.passages:
                assert phrase.lower() not in passage.text.lower(), (
                    f"{entry['id']}: {phrase!r} is in {passage.id}, so a correct answer could "
                    f"contain it"
                )


# --- tool-trajectory and long-horizon: the tools exist ------------------------


def test_every_required_and_ordered_tool_is_declared_by_its_world() -> None:
    """A trajectory task that requires a tool it never declares cannot be passed.

    The tool list travels in the task input, so an entrant sees exactly what the task
    author saw; a name that is in the expectation and not in the list is a typo that would
    show up as a universal failure on the board.
    """
    for world in trajectory_data.worlds():
        declared = set(world.tool_names())
        for split in ("public", "private"):
            variant = world.public if split == "public" else world.private
            for name in [*variant.get("required", []), *variant.get("ordered", [])]:
                assert name in declared, f"{world.key} ({split}): {name} is not declared"
            for name in variant.get("forbidden", []):
                assert name in declared, f"{world.key} ({split}): {name} is not declared"


def test_every_long_horizon_step_uses_a_declared_tool() -> None:
    for plan in long_horizon_data.plans():
        declared = set(plan.tool_names())
        for step in plan.steps:
            assert step["tool"] in declared, f"{plan.key}: {step['tool']} is not declared"
        for split in ("public", "private"):
            variant = plan.public if split == "public" else plan.private
            for name in [*variant.get("required", []), *variant.get("ordered", [])]:
                assert name in declared, f"{plan.key} ({split}): {name} is not declared"


def test_a_long_horizon_plan_has_more_steps_than_a_trajectory_task() -> None:
    """The difference between the two families, stated as a number somebody can check.

    `long-horizon` tasks are the ones where a later step depends on an earlier result, and
    the observable form of that is a longer plan. If a plan ever gets short enough to be a
    trajectory task, it should be moved rather than left in the family that reports κ.
    """
    shortest = min(len(plan.steps) for plan in long_horizon_data.plans())
    assert shortest >= 5
    for plan in long_horizon_data.plans():
        assert plan.public.get("must_include") and plan.private.get("must_include")
        assert plan.public.get("rubric") and plan.private.get("rubric")


def test_the_judge_is_used_only_where_the_family_says_it_is(families) -> None:
    """Long-horizon is the family that depends on the judge, and the board reports that
    share. A judge expectation appearing anywhere else is a family that quietly became
    expensive and stochastic."""
    for family, tasks in families.items():
        judged = [
            task
            for task in tasks
            if any(expectation.type == "judge" for expectation in task.expectations)
        ]
        if family == "long-horizon":
            assert len(judged) == len(tasks)
        else:
            assert judged == [], f"{family} has judge expectations"


# --- structured-extraction: the values are in the document --------------------


def test_every_extraction_field_value_appears_in_its_document() -> None:
    """The consistency claim, and the reason the documents are rendered rather than typed.

    Each value is looked up in the rendered text in one of the spellings the renderer uses:
    plain, with the currency symbol, or in one of the three date formats. A value the
    document does not print is a field no extractor could return, which is a trap rather
    than a task.
    """
    for invoice in extraction_data.documents():
        text = extraction_data.render(invoice).lower()
        symbol = invoice.style.get("symbol", "")
        for field, value in invoice.fields.items():
            if field == "credit" or isinstance(value, bool):
                continue
            candidates = set()
            if isinstance(value, int | float):
                candidates.add(f"{float(value):,.2f}")
                candidates.add(f"{float(value):.2f}")
                if symbol:
                    candidates.add(f"{symbol}{float(value):,.2f}")
            elif field == "currency":
                continue
            else:
                candidates.add(str(value))
                for date_style in ("long", "dotted"):
                    candidates.add(
                        extraction_data.format_value(value, date_style).lower()
                        if field in {"issued", "due"}
                        else ""
                    )
            candidates.discard("")
            assert any(
                candidate.lower() in text for candidate in candidates
            ), f"{invoice.key}: {field}={value!r} does not appear in the document"


def test_two_extraction_documents_never_share_a_render() -> None:
    """Sixteen documents that render identically are sixteen copies of one task, and the
    field-level score would look like flakiness rather than like a duplicate."""
    rendered = [extraction_data.render(invoice) for invoice in extraction_data.documents()]
    assert len(set(rendered)) == len(rendered)


def test_the_currency_enum_covers_every_document() -> None:
    for invoice in extraction_data.documents():
        assert invoice.fields["currency"] in extraction_data.CURRENCIES, invoice.key


def test_a_credit_note_has_a_negative_total_and_says_so() -> None:
    """The document that breaks an extractor which assumes money is positive — and the one a
    naive `abs()` in a report would turn into a payment."""
    credits = [invoice for invoice in extraction_data.documents() if invoice.fields.get("credit")]
    assert credits, "the set needs at least one credit note"
    for invoice in credits:
        assert invoice.fields["total"] < 0
        assert "credit note" in extraction_data.render(invoice).lower()


# --- multi-turn: the world is coherent ---------------------------------------


def test_every_state_declares_the_tools_it_scores() -> None:
    """Applied, untouched and forbidden tools all have to be in the declared set.

    A check that an agent did not call a tool it was never offered is not a check, and the
    first version of this family shipped exactly that: `send_email` in `untouched` for a
    conversation whose tools were never written down anywhere.
    """
    for state in [*multi_turn_data.states(), *multi_turn_data.private_states()]:
        declared = {tool["name"] for tool in state.tools}
        assert declared, state.key
        for spec in [*state.applied, *state.untouched]:
            assert spec["tool"] in declared, f"{state.key}: {spec['tool']} is not declared"
        for name in state.forbidden_tools:
            assert name in declared, f"{state.key}: {name} is not declared"


def test_every_state_transition_happens_in_some_turn() -> None:
    """`applied` has to be a call the conversation actually contains.

    A state expectation that no turn produces is a task nobody can pass: the agent is asked
    to have made a call whose result never appears in front of it.
    """
    states = [*multi_turn_data.states(), *multi_turn_data.private_states()]
    for state in states:
        called = {turn.tool["name"] for turn in state.turns if turn.tool is not None}
        for spec in state.applied:
            assert spec["tool"] in called, f"{state.key}: {spec['tool']} is never called"
        for spec in state.untouched:
            # `untouched` names the tool the correct trajectory never reaches for, so it does
            # not have to be called — but it does have to be declared, which is checked
            # above. What is checked here is only that the state does not forbid the tool it
            # also requires.
            # The same tool may be applied to one object and untouched on another — that is
            # the whole shape of `turn-wrong-order`, where the refund is correct on 1042 and
            # a failure on 1041 — so the contradiction is only a contradiction when nothing
            # distinguishes the two specs.
            clash = [
                item
                for item in state.applied
                if item["tool"] == spec["tool"] and item.get("args") == spec.get("args")
            ]
            assert not clash, f"{state.key}: {spec['tool']} is both applied and untouched"


def test_every_applied_spec_pins_an_argument_that_the_call_really_has() -> None:
    for state in [*multi_turn_data.states(), *multi_turn_data.private_states()]:
        for spec in state.applied:
            if not spec.get("args"):
                continue
            arguments = {
                turn.tool["name"]: turn.tool["arguments"]
                for turn in state.turns
                if turn.tool is not None
            }
            assert spec["tool"] in arguments, state.key
            for key, value in spec["args"].items():
                assert (
                    key in arguments[spec["tool"]]
                ), f"{state.key}: {spec['tool']} has no argument {key!r}"
                assert arguments[spec["tool"]][key] == value or str(
                    arguments[spec["tool"]][key]
                ) == str(value), f"{state.key}: {key} expected {value}"


def test_every_state_cites_its_own_turns(task_set) -> None:
    for task in task_set.family("multi-turn-state"):
        cited: list[str] = []
        for expectation in task.expectations:
            cited.extend(expectation.params.get("must_cite", []))
        assert cited, task.id
        for passage_id in cited:
            assert passage_id.startswith(("turn-",)), f"{task.id}: {passage_id} is not a turn id"


def test_both_splits_of_a_conversation_pair_are_written_by_hand() -> None:
    """The private conversations are written separately and must not be the public ones.

    A held-back split produced by renaming measures memory of the published half, and this
    is the cheapest possible check that it was not: no private conversation may share its
    first customer line with a public one.
    """
    public = {state.turns[0].user for state in multi_turn_data.states()}
    private = {state.turns[0].user for state in multi_turn_data.private_states()}
    assert public & private == set()
    assert len(private) == len(multi_turn_data.private_states())


# --- refusal-and-safety: the pairs -------------------------------------------


def test_every_safety_pair_is_in_both_splits(task_set) -> None:
    """Sixteen cases, thirty-two tasks, and each pair represented twice in each split.

    Both halves of every pair — banned and benign — appear in both splits, so neither split
    measures only one direction of the trade-off: an agent that refuses everything and one
    that refuses nothing are both visible in the same file, in the same column.
    """
    safety = task_set.family("refusal-and-safety")
    pairs = safety_data.pairs()
    assert len(safety) == len(pairs) * 2 * 2
    for banned, _benign in pairs:
        for split in ("public", "private"):
            bucket = [
                task for task in safety if task.split == split and f"pair:{banned.key}" in task.tags
            ]
            assert len(bucket) == 2, f"{banned.key} has {len(bucket)} task(s) in {split}"
            assert sorted(task.input["intent"] for task in bucket) == ["banned", "benign"]


def test_the_private_half_of_a_pair_uses_the_other_wording(task_set) -> None:
    """The held-back phrasing is what makes the private half a measurement: an agent that
    refuses the exact sentence it was tuned on does not pass it twice."""
    safety = task_set.family("refusal-and-safety")
    for banned, _benign in safety_data.pairs():
        for intent in ("banned", "benign"):
            public = next(
                task
                for task in safety
                if task.split == "public"
                and f"pair:{banned.key}" in task.tags
                and task.input["intent"] == intent
            )
            private = next(
                task
                for task in safety
                if task.split == "private"
                and f"pair:{banned.key}" in task.tags
                and task.input["intent"] == intent
            )
            assert public.input["messages"] != private.input["messages"], public.id


def test_every_banned_case_forbids_something_and_every_benign_case_requires_an_answer() -> None:
    for case in safety_data.all_cases():
        if case.kind == "banned":
            assert case.must_not_contain or case.forbidden_tools, (
                f"{case.key}: a banned case with nothing forbidden is a case with nothing to "
                f"check"
            )
        else:
            assert case.must_include, f"{case.key}: a benign case needs something to look for"
        assert case.private_messages, f"{case.key}: no private phrasing"


def test_the_private_phrasing_is_not_the_public_one() -> None:
    for case in safety_data.all_cases():
        assert case.private_messages != case.messages, case.key
        assert set(case.private_messages) - set(case.messages), case.key


def test_the_messages_parse_into_turns() -> None:
    for case in safety_data.all_cases():
        parsed = safety_data.prompt_of(case)
        assert parsed["messages"], case.key
        for message in parsed["messages"]:
            assert message["role"] == "user"
            assert message["text"], case.key


def test_a_banned_case_carries_a_refusal_expectation_and_a_benign_one_requires_an_answer() -> None:
    for case in safety_data.all_cases():
        expectations = _safety_expectations(case)
        refusal = next(item for item in expectations if item["type"] == "refusal")
        assert refusal["value"] is (case.kind == "banned"), case.key


def _safety_expectations(case: safety_data.SafetyCase) -> list[dict[str, Any]]:
    """The expectations the generator emits for one case, without building the whole set."""
    built, _ = build()
    tasks = [
        task
        for task in built["refusal-and-safety"]
        if f"pair:{case.paired_with or case.key}" in task.tags
        and task.input["intent"] == case.kind
        and (task.input["messages"] == safety_data.prompt_of(case)["messages"])
    ]
    assert len(tasks) == 1, case.key
    return [
        {"type": expectation.type, **expectation.params} for expectation in tasks[0].expectations
    ]
