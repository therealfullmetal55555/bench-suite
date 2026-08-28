"""The generators: the task set is built from data, and the data is checked.

Five modules of ground truth — facts over the corpus, unanswerable questions, tool worlds,
multi-turn conversations, safety pairs, long plans — and one writer that turns them into the
YAML in `tasks/`. Run it with `bench task generate` (or `make tasks`). The ground truth lives
in `generators/data/`, corpora and all, so a checkout, a wheel and a tarball generate the same
set.

Three decisions here, and they are the ones that keep a 150-task split from becoming 150
files nobody can audit:

* **One file per family.** `load_tasks` reads every YAML under the path it is given, so the
  choice is one file per task or one file per family. Four thousand files is a diff review
  nobody does; six files is a diff review that takes ten minutes, and the tasks in a family
  are the ones a reviewer is comparing anyway.
* **The output is deterministic**, and a test regenerates `tasks/` into a temporary
  directory and compares bytes. A task set that drifts from its generator is a task set
  where somebody hand-edited a file and the next `make tasks` silently reverts it — which is
  a confusing hour, on a good day.
* **The generator validates before it writes.** Every task it emits goes through
  `Task.model_validate`, so a bad phrase list or a missing `because` fails at generation
  time with a file and a line, rather than at load time in somebody's run.

Pairing, and why it is in the ids and not just the tags: each safety case and each long plan
emits a public and a private task with the same `key`, and the private one is generated from
the *same entry* — so the held-back half cannot describe a world the public half does not,
which is the usual way a private split stops measuring anything.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..core.task import FAMILIES, PREFIXES, Task
from ..errors import TaskError
from . import extraction_data, long_horizon_data, multi_turn_data, safety_data, trajectory_data

#: Family order for the generated files, and therefore for the ids. Stable order matters
#: more than a nice one: the ids end up in reports, and a report that renumbers when
#: somebody edits an unrelated file is a report nobody can compare with last week's.
FAMILY_ORDER = (
    "grounded-qa",
    "tool-trajectory",
    "structured-extraction",
    "multi-turn-state",
    "refusal-and-safety",
    "long-horizon",
)

DEFAULT_ADDED = "2026-09-30"


@dataclass
class Counts:
    """What was written, per family and split. Printed by the CLI and asserted by tests."""

    generated: dict[str, dict[str, int]] = field(default_factory=dict)

    def add(self, family: str, split: str) -> None:
        bucket = self.generated.setdefault(family, {"public": 0, "private": 0})
        bucket[split] += 1

    @property
    def total(self) -> int:
        return sum(sum(bucket.values()) for bucket in self.generated.values())

    def describe(self) -> str:
        parts = []
        for family in FAMILY_ORDER:
            bucket = self.generated.get(family)
            if bucket:
                parts.append(f"{family} {bucket['public']}p/{bucket['private']}v")
        return " · ".join(parts)


class Counter:
    """Per-family numbering, shared by both splits and independent of any cap.

    `gqa-0001` is the first public grounded task and `gqa-00NN` the first private one: the
    numbering follows the order the entries are written in, which is the order they appear
    in the data file. A cap changes how many are emitted, never what an existing id means —
    regenerating a smaller set is a smaller set, not a renumbered one.
    """

    def __init__(self) -> None:
        self._next: dict[str, int] = {}

    def take(self, family: str) -> str:
        index = self._next.get(family, 1)
        self._next[family] = index + 1
        return f"{PREFIXES[family]}-{index:04d}"


def _task(
    *,
    task_id: str,
    family: str,
    split: str,
    input_document: dict[str, Any],
    expectations: list[dict[str, Any]],
    because: str,
    tags: list[str],
    added: str,
    corpus: str | None = None,
    limits: dict[str, Any] | None = None,
    budget: dict[str, Any] | None = None,
) -> Task:
    """Build one task, through the real model.

    `model_validate` rather than a dict the writer hopes is right: the generator is the
    first place a malformed task can be caught, and catching it here names the file and the
    field instead of failing a run six weeks later.
    """
    document: dict[str, Any] = {
        "id": task_id,
        "family": family,
        "split": split,
        "input": input_document,
        "expectations": expectations,
        "because": " ".join(because.split()),
        "tags": sorted(set(tags)),
        "added": added,
    }
    if corpus:
        document["corpus"] = corpus
    if limits:
        document["limits"] = limits
    if budget:
        document["budget"] = budget
    try:
        return Task.model_validate(document)
    except Exception as exc:  # noqa: BLE001 - re-raised as a generation error
        raise TaskError(f"{task_id} could not be built: {exc}") from exc


# --- the six families ---------------------------------------------------------


def grounded_tasks(
    counter: Counter, added: str, facts: dict[str, Any], unanswerable: dict[str, Any]
) -> list[tuple[str, Task]]:
    """Answerable questions from `facts.yaml`, unanswerable ones from `unanswerable.yaml`.

    The two halves are generated from different sources on purpose: an answerable task names
    a passage it must cite, and an unanswerable task must not cite anything at all, so a
    single template that emitted both would produce tasks whose expectations contradict
    their own question.
    """
    produced: list[tuple[str, Task]] = []
    for fact in facts["facts"]:
        for split in ("public", "private"):
            variant = fact[split]
            produced.append(
                (
                    split,
                    _task(
                        task_id=counter.take("grounded-qa"),
                        family="grounded-qa",
                        split=split,
                        corpus=f"corpus/{fact['corpus']}",
                        input_document={
                            "question": variant["question"],
                            "corpus": f"corpus/{fact['corpus']}",
                            "style": variant.get("style", "direct"),
                        },
                        expectations=[
                            {"type": "contains", "value": phrase}
                            for phrase in variant["must_include"]
                        ]
                        + [
                            {
                                "type": "citations",
                                "must_cite": [f"{fact['corpus']}/{fact['passage']}"],
                                "must_not_cite_unsupported": True,
                            },
                            {"type": "answerable", "value": True},
                        ],
                        because=fact["because"],
                        tags=[*fact.get("tags", []), "grounded"],
                        added=added,
                        limits={"max_steps": 6},
                        budget={"max_usd": 0.05, "max_seconds": 60},
                    ),
                )
            )

    for entry in unanswerable["unanswerable"]:
        for split in ("public", "private"):
            variant = entry[split]
            produced.append(
                (
                    split,
                    _task(
                        task_id=counter.take("grounded-qa"),
                        family="grounded-qa",
                        split=split,
                        corpus=f"corpus/{entry['corpus']}",
                        input_document={
                            "question": variant["question"],
                            "corpus": f"corpus/{entry['corpus']}",
                            "style": "unanswerable",
                        },
                        expectations=[
                            {"type": "answerable", "value": False},
                            # The fabrication list is what makes this a measurement rather
                            # than a shrug: the task names the confident wrong answer, so a
                            # model that guesses one of them fails even though it also said
                            # it was not sure.
                            *[
                                {"type": "not_contains", "value": phrase}
                                for phrase in entry.get("fabrications", [])
                            ],
                        ],
                        because=entry["because"],
                        tags=[*entry.get("tags", []), "grounded", "unanswerable"],
                        added=added,
                        limits={"max_steps": 6},
                        budget={"max_usd": 0.03, "max_seconds": 45},
                    ),
                )
            )
    return produced


def trajectory_tasks(counter: Counter, added: str, worlds: list[Any]) -> list[tuple[str, Task]]:
    produced: list[tuple[str, Task]] = []
    for world in worlds:
        for split in ("public", "private"):
            variant = world.public if split == "public" else world.private
            trajectory: dict[str, Any] = {
                "type": "trajectory",
                "required_tools": variant.get("required", []),
                "ordered": variant.get("ordered", []),
                "max_steps": variant["max_steps"],
            }
            if variant.get("forbidden"):
                trajectory["forbidden_tools"] = variant["forbidden"]
            produced.append(
                (
                    split,
                    _task(
                        task_id=counter.take("tool-trajectory"),
                        family="tool-trajectory",
                        split=split,
                        input_document={
                            "question": variant["question"],
                            "tools": world.tools,
                            "style": "tools",
                        },
                        expectations=[
                            trajectory,
                            {"type": "contains", "value": variant["must_include"]},
                        ],
                        because=world.because,
                        tags=[*world.tags, "trajectory"],
                        added=added,
                        limits={"max_steps": variant["max_steps"]},
                        budget={"max_usd": 0.08, "max_seconds": 90},
                    ),
                )
            )
    return produced


def extraction_tasks(counter: Counter, added: str, documents: list[Any]) -> list[tuple[str, Task]]:
    """Every third document also gets a `schema` expectation.

    Not every one: the two checks are different questions — "are these values right" and "is
    this a document of the right shape" — and running both on all sixteen doubles the task's
    weight on the field-level arithmetic without adding a question. The split is by position
    so it is stable across regenerations, and it is stated here rather than discovered by
    reading the YAML.
    """
    produced: list[tuple[str, Task]] = []
    for index, invoice in enumerate(documents):
        text = extraction_data.render(invoice)
        required = invoice.schema_fields()
        properties: dict[str, Any] = {
            "invoice_id": {"type": "string"},
            "vendor": {"type": "string"},
            "issued": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
            "due": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
            "net": {"type": "number"},
            "tax": {"type": "number"},
            "total": {"type": "number"},
            "po_number": {"type": "string"},
            "currency": {"type": "string", "enum": list(extraction_data.CURRENCIES)},
        }
        schema = {
            "type": "object",
            "required": required,
            "properties": properties,
            "additionalProperties": False,
        }
        expectations: list[dict[str, Any]] = [
            {"type": "fields", "values": invoice.fields, "weights": {"total": 2}},
        ]
        if index % 3 == 0:
            expectations.append({"type": "schema", "schema": schema, "fenced": True})
        produced.append(
            (
                "private" if index % 2 else "public",
                _task(
                    task_id=counter.take("structured-extraction"),
                    family="structured-extraction",
                    split="private" if index % 2 else "public",
                    input_document={
                        "document": text,
                        "schema": schema,
                        "style": "extraction",
                    },
                    expectations=expectations,
                    because=(
                        "An invoice in the shape one company sends and not the shape the "
                        "extractor was written against: the same values, dressed differently, "
                        "which is where field-level extraction actually fails."
                    ),
                    tags=["extraction", "documents"],
                    added=added,
                    limits={"max_steps": 3},
                    budget={"max_usd": 0.06, "max_seconds": 60},
                ),
            )
        )
    return produced


def multi_turn_tasks(counter: Counter, added: str, states: list[Any]) -> list[tuple[str, Task]]:
    produced: list[tuple[str, Task]] = []
    for state in states:
        turns = [
            {
                "user": turn.user,
                "tool_call": turn.tool,
                "tool_result": turn.tool["result"] if turn.tool else None,
            }
            for turn in state.turns
        ]
        expectations: list[dict[str, Any]] = [
            {
                "type": "state",
                "applied": state.applied,
                "untouched": state.untouched,
            },
            {
                "type": "citations",
                "must_cite": state.passage_ids(),
                "must_not_cite_unsupported": True,
            },
        ]
        if state.forbidden_tools:
            expectations.append({"type": "trajectory", "forbidden_tools": state.forbidden_tools})
        if state.final_must_include:
            expectations.extend(
                {"type": "contains", "value": phrase} for phrase in state.final_must_include
            )
        if state.final_must_not_contain:
            expectations.extend(
                {"type": "not_contains", "value": phrase} for phrase in state.final_must_not_contain
            )
        produced.append(
            (
                state.split,
                _task(
                    task_id=counter.take("multi-turn-state"),
                    family="multi-turn-state",
                    split=state.split,
                    input_document={
                        "conversation": turns,
                        "tools": state.tools,
                        "goal": state.title,
                        "style": "conversation",
                    },
                    expectations=expectations,
                    because=state.because,
                    tags=[*state.tags, "conversation"],
                    added=added,
                    limits={"max_steps": 10},
                    budget={"max_usd": 0.25, "max_seconds": 180},
                ),
            )
        )
    return produced


def safety_tasks(counter: Counter, added: str, cases: list[Any]) -> list[tuple[str, Task]]:
    produced: list[tuple[str, Task]] = []
    for case in cases:
        if case.kind == "banned":
            expectations: list[dict[str, Any]] = [
                {"type": "refusal", "value": True},
                *[{"type": "not_contains", "value": phrase} for phrase in case.must_not_contain],
            ]
            if case.forbidden_tools:
                expectations.append({"type": "trajectory", "forbidden_tools": case.forbidden_tools})
        else:
            expectations = [
                {"type": "refusal", "value": False},
                *[{"type": "contains", "value": phrase} for phrase in case.must_include],
            ]
        # The pair tag is what lets the report show "this entry refuses 90% of banned
        # requests and 40% of the benign twins" instead of one aggregate number that hides
        # exactly the trade-off the family exists to measure. Both halves carry it, keyed on
        # the banned case, so a report can group them without guessing which is which.
        tags = [*case.tags, f"pair:{case.paired_with or case.key}"]
        # Both halves of every pair go into *both* splits. The alternative — banned cases
        # public, benign ones held back — would make the private split a different
        # measurement (over-refusal) rather than the same one on unseen wording, and a
        # single number over both would hide the trade-off the family exists to show.
        for split, messages in (
            ("public", case.messages),
            ("private", case.private_messages or case.messages),
        ):
            produced.append(
                (
                    split,
                    _task(
                        task_id=counter.take("refusal-and-safety"),
                        family="refusal-and-safety",
                        split=split,
                        input_document={
                            **safety_data.prompt_of(case, messages),
                            "style": "safety",
                            "intent": case.kind,
                        },
                        expectations=expectations,
                        because=case.because,
                        tags=tags,
                        added=added,
                        limits={"max_steps": 4},
                        budget={"max_usd": 0.04, "max_seconds": 45},
                    ),
                )
            )
    return produced


def long_horizon_tasks(counter: Counter, added: str, plans: list[Any]) -> list[tuple[str, Task]]:
    produced: list[tuple[str, Task]] = []
    for plan in plans:
        for split in ("public", "private"):
            variant = plan.public if split == "public" else plan.private
            trajectory: dict[str, Any] = {
                "type": "trajectory",
                "required_tools": variant.get("required", []),
                "ordered": variant.get("ordered", []),
                "max_steps": variant["max_steps"],
            }
            if variant.get("forbidden"):
                trajectory["forbidden_tools"] = variant["forbidden"]
            produced.append(
                (
                    split,
                    _task(
                        task_id=counter.take("long-horizon"),
                        family="long-horizon",
                        split=split,
                        input_document={
                            "question": variant["question"],
                            "tools": plan.tools,
                            "environment": plan.steps,
                            "style": "plan",
                        },
                        expectations=[
                            trajectory,
                            {"type": "contains", "value": variant["must_include"]},
                            {"type": "judge", "rubric": variant["rubric"]},
                        ],
                        because=plan.because,
                        tags=[*plan.tags, "plan"],
                        added=added,
                        limits={"max_steps": variant["max_steps"]},
                        budget={"max_usd": 0.20, "max_seconds": 240},
                    ),
                )
            )
    return produced


# --- writing ------------------------------------------------------------------


def build(added: str = DEFAULT_ADDED) -> tuple[dict[str, list[Task]], Counts]:
    """Every task, keyed by family, plus the counts. Pure: no filesystem, no clock.

    `added` is a parameter rather than `date.today()` so that regenerating the repository's
    tasks on a different day is a no-op. A test compares the checked-in files against this
    function's output, and a clock in here would make that test fail every morning — which
    is the kind of test that gets deleted rather than fixed.
    """
    # The ground truth travels with the package rather than with the repository: a
    # generator that only works from a checkout is a generator nobody can run against their
    # own corpus, and `data/` next to this file is where `pip install` puts it too.
    data = Path(__file__).resolve().parent / "data"
    facts = _yaml(data / "facts.yaml")
    unanswerable = _yaml(data / "unanswerable.yaml")

    counter = Counter()
    counts = Counts()
    families: dict[str, list[Task]] = {}
    sources: list[tuple[str, list[tuple[str, Task]]]] = [
        ("grounded-qa", grounded_tasks(counter, added, facts, unanswerable)),
        ("tool-trajectory", trajectory_tasks(counter, added, trajectory_data.worlds())),
        ("structured-extraction", extraction_tasks(counter, added, extraction_data.documents())),
        (
            "multi-turn-state",
            multi_turn_tasks(
                counter,
                added,
                [*multi_turn_data.states(), *multi_turn_data.private_states()],
            ),
        ),
        ("refusal-and-safety", safety_tasks(counter, added, safety_data.all_cases())),
        ("long-horizon", long_horizon_tasks(counter, added, long_horizon_data.plans())),
    ]
    for family, produced in sources:
        tasks = []
        for split, item in produced:
            counts.add(family, split)
            tasks.append(item)
        families[family] = tasks
    return families, counts


def cap(
    families: dict[str, list[Task]], *, public: int | None, private: int | None
) -> dict[str, list[Task]]:
    """Keep at most N per split per family, in the order the data defines.

    A subset, never a renumbering: `gqa-0007` means the same question whether the set was
    generated with a cap or without one, because ids end up in reports and in people's
    notes. A capped set therefore has gaps in its numbering, which is the honest way for a
    subset to look.
    """
    if public is None and private is None:
        return families
    capped: dict[str, list[Task]] = {}
    for family, tasks in families.items():
        keep: list[Task] = []
        seen = {"public": 0, "private": 0}
        limits = {"public": public, "private": private}
        for task in tasks:
            limit = limits[task.split]
            if limit is None or seen[task.split] < limit:
                seen[task.split] += 1
                keep.append(task)
        capped[family] = keep
    return capped


def write(
    root: Path, *, added: str = DEFAULT_ADDED, families: dict[str, list[Task]] | None = None
) -> Counts:
    """Write one YAML file per family, and nothing else.

    Atomic per file, because the alternative is a half-written task set that still loads:
    `load_tasks` reads every YAML it finds, so a generator killed mid-write leaves a valid
    file with half the tasks in it and no error until somebody counts.
    """
    if families is None:
        families, _ = build(added)
    counts = Counts()
    root.mkdir(parents=True, exist_ok=True)
    # The documents come with the tasks. `--out` pointed at a fresh directory has to produce
    # something `bench task validate` can read, and the four corpora are inputs in exactly the
    # way the facts file is: they live in the package and are copied out, not referenced, so a
    # generated set can be moved, tarred or committed without its surroundings.
    documents = Path(__file__).resolve().parent / "data" / "corpus"
    for source in sorted(documents.glob("*")):
        if source.is_dir():
            shutil.copytree(source, root / "corpus" / source.name, dirs_exist_ok=True)
    for family in FAMILY_ORDER:
        tasks = families.get(family) or []
        if not tasks:
            continue
        path = root / f"{family}.yaml"
        temporary = path.with_suffix(".yaml.tmp")
        temporary.write_text(_dump(tasks), encoding="utf-8")
        temporary.replace(path)
        for task in tasks:
            counts.add(family, task.split)
    return counts


def _yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise TaskError(
            f"the generator needs {path}",
            hint="the task set is built from the corpus facts; see corpus/facts.yaml",
        )
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TaskError(f"{path} should be a mapping")
    return document


def _dump(tasks: list[Task]) -> str:
    """YAML with `because` as a block scalar and everything else inline.

    The default dumper quotes a multi-line string and escapes the newlines, which turns the
    one field a reviewer reads into the one field a reviewer skips. Registering a
    representer for strings containing a newline is four lines and makes the diff of a
    generated task readable, which is the whole reason it is checked in.
    """
    payload = {"tasks": [_as_document(task) for task in tasks]}
    return yaml.safe_dump(
        payload, sort_keys=False, allow_unicode=True, width=100, default_flow_style=False
    )


class _Block(str):
    """A string that is dumped as `|` rather than as a quoted one-liner."""


def _represent_block(dumper: yaml.SafeDumper, data: _Block) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(data), style="|")


yaml.add_representer(_Block, _represent_block, Dumper=yaml.SafeDumper)


def _as_document(task: Task) -> dict[str, Any]:
    """The task as a plain dict, with `because` marked for block style.

    Field order is the order a person reads it in — id, family, split, input, expectations,
    because, the rest — rather than alphabetical, because this file is a document and the
    diff of a document should read like one.
    """
    document: dict[str, Any] = {
        "id": task.id,
        "family": task.family,
        "split": task.split,
    }
    if task.corpus:
        document["corpus"] = task.corpus
    document["input"] = task.input
    document["expectations"] = [
        {"type": expectation.type, **expectation.params} for expectation in task.expectations
    ]
    if task.tags:
        document["tags"] = list(task.tags)
    limits = task.limits.model_dump(exclude_none=True)
    if limits:
        document["limits"] = limits
    if not task.budget.is_empty:
        document["budget"] = task.budget.model_dump(exclude_none=True)
    document["added"] = task.added
    document["because"] = _Block(" ".join(task.because.split()))
    return document


def family_names() -> tuple[str, ...]:
    """The families the generator knows about, in the order it writes them."""
    return FAMILY_ORDER


__all__ = [
    "DEFAULT_ADDED",
    "FAMILY_ORDER",
    "Counts",
    "Counter",
    "build",
    "cap",
    "family_names",
    "write",
    "FAMILIES",
]
