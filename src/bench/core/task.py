"""Tasks, the splits, and the linter that keeps them honest.

A benchmark is its tasks. Everything else in this repository is plumbing to run them
and arithmetic to average them, so this file is where the project either has a spine
or does not.

Two rules are enforced at load time rather than recommended in a document:

**Every task carries a `because`.** Why this task exists, in a sentence, ideally
naming the failure it came from. A task without one is a puzzle, and puzzles measure
puzzle-solving. The linter refuses the file rather than warning, because a warning
in a generator script is a warning nobody reads.

**A task's definition is hashed.** The fingerprint goes into every result. Between
two runs, a task whose id is the same and whose fingerprint differs is a different
question, and both the board and the verifier refuse to compare across that — the
alternative is a benchmark that silently improves because somebody made the hard
tasks easier. The same hash is what catches the same task appearing in both splits.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..errors import TaskError
from .stats import ngram_overlap

SPLITS = ("public", "private")

FAMILIES: dict[str, str] = {
    "grounded-qa": "answering from a corpus, with citations that have to check out",
    "tool-trajectory": "choosing and ordering tools inside a step budget",
    "structured-extraction": "turning messy documents into a schema",
    "multi-turn-state": "holding state across turns and correcting itself",
    "refusal-and-safety": "not doing the harmful thing, and not over-refusing",
    "long-horizon": "plans whose steps only pay off five turns later",
}
"""The six families, and what each one is for. Written here rather than in the docs
alone because the loader, the board and the generators all need to agree, and three
copies of a list is how a family quietly stops being scored."""

ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")

PREFIXES: dict[str, str] = {
    "grounded-qa": "gqa",
    "tool-trajectory": "traj",
    "structured-extraction": "ext",
    "multi-turn-state": "turn",
    "refusal-and-safety": "safe",
    "long-horizon": "long",
}
"""Task ids start with the family's prefix (`gqa-0042`). It makes a task id readable
in a failure list without a lookup, and it makes a mistyped family visible at a
glance in a diff."""


class Budget(BaseModel, extra="forbid"):
    max_usd: float | None = None
    max_seconds: float | None = None

    @field_validator("max_usd", "max_seconds")
    @classmethod
    def _positive(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("a budget of zero is a task that can never be run")
        return value

    @property
    def is_empty(self) -> bool:
        return self.max_usd is None and self.max_seconds is None


class Limits(BaseModel, extra="forbid"):
    max_steps: int | None = None
    """Turns or tool rounds. `None` means the family does not score steps."""

    @field_validator("max_steps")
    @classmethod
    def _at_least_one(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("max_steps must be at least 1")
        return value


class Expectation(BaseModel):
    """One checkable claim about a trace.

    `extra="allow"` is deliberate. Expectations are family-specific — a citation
    rule has fields a schema rule has never heard of — and the core has no business
    policing the vocabulary of six families whose scorers live elsewhere. What the
    core *does* police is that the `type` is one the scorers package actually
    implements, checked at load time against `scorers.KNOWN_TYPES`, so a typo fails
    before a run rather than scoring a task zero for a reason nobody can see.
    """

    model_config = ConfigDict(extra="allow")

    type: str

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.model_extra) if self.model_extra else {}

    def describe(self) -> str:
        """One line for the failure list. `citations(must_cite=1)`, not a repr."""
        if not self.params:
            return self.type
        rendered = ", ".join(f"{key}={value!r}" for key, value in sorted(self.params.items()))
        return f"{self.type}({rendered})"


class Task(BaseModel, extra="forbid"):
    id: str
    family: str
    split: Literal["public", "private"]
    input: dict[str, Any]
    expectations: list[Expectation]
    because: str
    limits: Limits = Field(default_factory=Limits)
    budget: Budget = Field(default_factory=Budget)
    corpus: str | None = None
    """A corpus directory, relative to the tasks directory. Grounded tasks have one;
    the others usually do not, and a task whose corpus is missing is a load error
    rather than a zero."""

    tags: list[str] = Field(default_factory=list)
    added: str | None = None
    """ISO date. Kept because "this task is three releases old and everybody passes
    it" is the retirement criterion, and that needs a date to start from."""

    retired: str | None = None
    """Set when the task is kept for the record but no longer scored. Retired tasks
    stay in the repository: the history of what stopped measuring anything is part
    of the benchmark."""

    @field_validator("because")
    @classmethod
    def _because_has_a_reason(cls, value: str) -> str:
        if len(value.strip()) < 20:
            raise ValueError(
                "a `because` has to say something: what failure is this task standing in for?"
            )
        return value.strip()

    @field_validator("id")
    @classmethod
    def _id_shape(cls, value: str) -> str:
        if not ID_PATTERN.match(value):
            raise ValueError(f"task id {value!r} should be lowercase-with-dashes, like 'gqa-0042'")
        return value

    @field_validator("expectations")
    @classmethod
    def _has_expectations(cls, value: list[Expectation]) -> list[Expectation]:
        if not value:
            raise ValueError(
                "a task with no expectations is a task whose answer is whatever the "
                "judge felt like; give it at least one check"
            )
        return value

    @property
    def fingerprint(self) -> str:
        """A hash of the *definition*, not of the id.

        `exclude` is set so that moving a task from public to private, adding a tag
        or retiring it does not read as "the question changed" — those are
        bookkeeping, and the fingerprint is about what the agent is asked and how it
        is scored. `added` too: a date is a note in the margin.
        """
        payload = self.model_dump(mode="json", exclude={"split", "tags", "added", "retired", "id"})
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    @property
    def expected_prefix(self) -> str:
        return PREFIXES[self.family]


class TaskSet:
    """Every task the repository knows about, in both splits.

    Loaded as a set rather than as two sets because the checks that matter — no id
    twice, no definition in both splits, every family present in both — only make
    sense with the whole picture. `public` and `private` are views onto it.
    """

    def __init__(self, tasks: list[Task], *, root: Path | None = None) -> None:
        self.tasks = tasks
        self.root = root

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self) -> Iterator[Task]:
        return iter(self.tasks)

    def split(self, name: str) -> list[Task]:
        if name not in SPLITS:
            raise TaskError(f"unknown split {name!r}; the splits are {', '.join(SPLITS)}")
        return [task for task in self.tasks if task.split == name and not task.retired]

    def family(self, name: str) -> list[Task]:
        return [task for task in self.tasks if task.family == name and not task.retired]

    def by_id(self, task_id: str) -> Task:
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise TaskError(f"no task with id {task_id!r}")

    def families(self) -> list[str]:
        return sorted({task.family for task in self.tasks})

    @property
    def digest(self) -> str:
        """One hash over every task's id and fingerprint, per split.

        This is what makes "the same benchmark" checkable: a result bundle records
        it, and the verifier re-runs only if the digest of the split it used is
        unchanged.
        """
        payload = {
            split: sorted((task.id, task.fingerprint) for task in self.split(split))
            for split in SPLITS
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def describe(self) -> str:
        parts = []
        for name in self.families():
            public = len([task for task in self.family(name) if task.split == "public"])
            private = len([task for task in self.family(name) if task.split == "private"])
            parts.append(f"{name} {public}p/{private}v")
        return " · ".join(parts)


def load_tasks(path: str | Path) -> TaskSet:
    """Load a tasks directory or a single file.

    A file may hold one task or a `tasks:` list of them; both shapes turn up in
    practice — a hand-written regression case is one task, a generated family is
    three hundred — and refusing the second shape means the generators write a file
    per task and the repository ends up with four thousand files.
    """
    root = Path(path)
    files: list[Path]
    if root.is_dir():
        files = sorted(
            item for item in root.rglob("*") if item.suffix in {".yaml", ".yml"} and item.is_file()
        )
        if not files:
            raise TaskError(f"no task files under {root}", hint="expected *.yaml with tasks")
    elif root.is_file():
        files = [root]
    else:
        raise TaskError(f"no such file or directory: {root}")

    tasks: list[Task] = []
    for file in files:
        tasks.extend(_load_file(file))
    if not tasks:
        raise TaskError(f"{root} contains no tasks")
    task_set = TaskSet(tasks, root=root if root.is_dir() else root.parent)
    lint(task_set)
    return task_set


def _load_file(file: Path) -> list[Task]:
    try:
        raw = yaml.safe_load(file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise TaskError(f"{file} is not valid YAML: {exc}") from exc
    if raw is None:
        raise TaskError(f"{file} is empty")
    if not isinstance(raw, dict):
        raise TaskError(f"{file} should be a mapping, got {type(raw).__name__}")

    entries = raw.get("tasks")
    if entries is None:
        entries = [raw]
    if not isinstance(entries, list):
        raise TaskError(f"{file}: `tasks` should be a list")

    loaded: list[Task] = []
    for index, entry in enumerate(entries):
        try:
            loaded.append(Task.model_validate(entry))
        except ValidationError as exc:
            lines = [
                f"  - {'/'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors()
            ]
            where = f"{file}#{index}" if len(entries) > 1 else str(file)
            raise TaskError(f"{where} does not validate:\n" + "\n".join(lines)) from exc
    return loaded


def lint(task_set: TaskSet) -> None:
    """Everything that has to be true before a single model call is worth making."""
    problems: list[str] = []

    seen: dict[str, tuple[str, str]] = {}
    for task in task_set.tasks:
        if task.id in seen:
            split, family = seen[task.id]
            problems.append(
                f"{task.id}: defined twice ({family}/{split} and {task.family}/{task.split})"
            )
        seen[task.id] = (task.split, task.family)

        if task.family not in FAMILIES:
            problems.append(
                f"{task.id}: unknown family {task.family!r}; known: {', '.join(FAMILIES)}"
            )
        elif not task.id.startswith(f"{task.expected_prefix}-"):
            problems.append(
                f"{task.id}: a {task.family} task should be named {task.expected_prefix}-…"
            )
        if task.corpus and task_set.root is not None:
            corpus = (task_set.root / task.corpus).resolve()
            if not corpus.exists():
                problems.append(f"{task.id}: corpus {task.corpus!r} not found at {corpus}")
        if task.split == "private" and not task.tags:
            # The private split is the one nobody develops against, so a task there
            # with no tags is a task nobody can group when the number moves.
            problems.append(f"{task.id}: a private task needs at least one tag")

    problems.extend(_split_leaks(task_set))
    problems.extend(_unknown_expectation_types(task_set))

    if problems:
        raise TaskError(
            f"{len(problems)} problem(s) in the task set:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )


#: Input keys that describe the *environment* rather than the question. Two tasks that
#: ship the same tool list or run against the same world are not the same task: the
#: tool-trajectory pairs are exactly that shape — one world, two questions, different correct
#: trajectories — and comparing the whole input scored them at 98% and called a deliberate
#: design a leak. The check is about the ask, so it compares the ask.
ENVIRONMENT_KEYS = frozenset({"tools", "environment", "schema", "style"})


def _the_ask(task: Task) -> dict[str, Any]:
    """The task's input, minus the parts that describe the world it happens in."""
    return {key: value for key, value in task.input.items() if key not in ENVIRONMENT_KEYS}


def _split_leaks(task_set: TaskSet) -> list[str]:
    """Catches the two ways a private task stops being private.

    The same id in both splits is the obvious one. The other is the same *task*
    under two ids, which happens whenever a generator is re-run with a different
    seed and nobody diffs the output — higher priority here than anywhere else,
    because a leaked private task inflates exactly the number the board is built to
    report.
    """
    problems: list[str] = []
    public = {task.id: task for task in task_set.split("public")}
    private = {task.id: task for task in task_set.split("private")}
    for task_id in set(public) & set(private):
        problems.append(f"{task_id}: in both splits")

    for task in private.values():
        for candidate in public.values():
            if task.fingerprint == candidate.fingerprint:
                problems.append(
                    f"{task.id}: identical to public task {candidate.id} — the private "
                    f"split is only worth having if it is not"
                )
    return problems


def _unknown_expectation_types(task_set: TaskSet) -> list[str]:
    """Every `type:` has to be one the scorers package implements, in a family that
    uses it.

    The import is local because the dependency points this way: `scorers` imports the
    trace models from `core`, so `core` importing `scorers` at module level would be
    a cycle. Doing it at call time keeps the layering one-directional, and the cost —
    an import on the first lint — is paid once.
    """
    from ..scorers import EXPECTATION_TYPES

    problems: list[str] = []
    for task in task_set.tasks:
        for expectation in task.expectations:
            known = EXPECTATION_TYPES.get(expectation.type)
            if known is None:
                problems.append(
                    f"{task.id}: unknown expectation type {expectation.type!r}; "
                    f"known types: {', '.join(sorted(EXPECTATION_TYPES))}"
                )
            elif known.families and task.family not in known.families:
                problems.append(
                    f"{task.id}: {expectation.type!r} is not scored in family "
                    f"{task.family!r}; it belongs to {' or '.join(known.families)}"
                )
    return problems


def find_near_duplicates(
    task_set: TaskSet, *, threshold: float = 0.85, n: int = 8
) -> list[tuple[str, str, float]]:
    """Near-verbatim pairs across splits, worst first.

    The exact-fingerprint check above catches a re-run generator; this catches a task
    that was edited after being leaked — one word changed, same question. Run by
    `bench task lint --duplicates`, kept out of the load path because it is
    quadratic and a three-hundred-task set should not pay for it on every run.
    """
    pairs: list[tuple[str, str, float]] = []
    private = task_set.split("private")
    public = task_set.split("public")
    for left in private:
        blob_left = json.dumps(_the_ask(left), sort_keys=True)
        for right in public:
            blob_right = json.dumps(_the_ask(right), sort_keys=True)
            # Both directions, because `ngram_overlap` answers "how much of A is in
            # B" and that is not symmetric: when one task is the other plus a clause,
            # the longer text scores 0.84 against the shorter and 1.0 the other way
            # round. Measuring one way means a leaked task is caught or missed
            # depending on which split the longer copy happens to be in, which is
            # exactly the kind of coin flip this check exists to remove.
            overlap = max(
                ngram_overlap(blob_left, blob_right, n=n),
                ngram_overlap(blob_right, blob_left, n=n),
            )
            if overlap >= threshold:
                pairs.append((left.id, right.id, overlap))
    return sorted(pairs, key=lambda item: -item[2])
