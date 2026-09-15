"""The task model and the split discipline.

The theme is that the rules are enforced rather than documented. A benchmark whose
rules live in a contributor guide is a benchmark whose rules are honoured by whoever
read the guide, and the person who did not read it is exactly the person whose task
would have broken the split.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from conftest import make_corpus, task, write_set

from bench.core.task import FAMILIES, Task, TaskSet, find_near_duplicates, lint, load_tasks
from bench.errors import TaskError


def load_one(tmp_path: Path, **overrides) -> Task:
    root = write_set(tmp_path, [task(**overrides)])
    return load_tasks(root).tasks[0]


# --- loading ----------------------------------------------------------------


def test_a_valid_task_loads(tmp_path: Path):
    loaded = load_one(tmp_path)
    assert loaded.id == "gqa-0001"
    assert loaded.family == "grounded-qa"
    assert loaded.budget.is_empty
    assert loaded.retired is None


def test_a_single_task_file_works_without_a_tasks_key(tmp_path: Path):
    """Both shapes exist in practice: a hand-written regression case is one task, a
    generated family is three hundred."""
    path = tmp_path / "one.yaml"
    path.write_text(yaml.safe_dump(task(id="gqa-0007"), sort_keys=False), encoding="utf-8")
    loaded = load_tasks(path)
    assert [item.id for item in loaded] == ["gqa-0007"]


def test_empty_yaml_is_an_error_not_an_empty_benchmark(tmp_path: Path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(TaskError, match="empty"):
        load_tasks(path)


def test_a_missing_directory_says_so(tmp_path: Path):
    with pytest.raises(TaskError, match="no such file"):
        load_tasks(tmp_path / "nope")


def test_a_directory_with_no_yaml_is_an_error(tmp_path: Path):
    (tmp_path / "readme.md").write_text("not a task", encoding="utf-8")
    with pytest.raises(TaskError, match="no task files"):
        load_tasks(tmp_path)


def test_a_validation_error_names_the_field(tmp_path: Path):
    """The whole value of validating on load is the message. A traceback pointing at
    a dict index is a bug report about the loader."""
    root = write_set(tmp_path, [task(expectations="contains the word permission")])
    with pytest.raises(TaskError) as excinfo:
        load_tasks(root)
    assert "expectations" in str(excinfo.value)


# --- the rules that make a task worth having --------------------------------


def test_a_task_without_a_because_is_refused(tmp_path: Path):
    with pytest.raises(TaskError) as excinfo:
        load_one(tmp_path, because="a bug")
    assert "because" in str(excinfo.value)


def test_a_task_with_no_expectations_is_refused(tmp_path: Path):
    with pytest.raises(TaskError) as excinfo:
        load_one(tmp_path, expectations=[])
    assert "expectations" in str(excinfo.value)


def test_an_id_has_to_look_like_an_id(tmp_path: Path):
    with pytest.raises(TaskError) as excinfo:
        load_one(tmp_path, id="GQA 1")
    assert "lowercase-with-dashes" in str(excinfo.value)


def test_the_id_prefix_has_to_match_the_family(tmp_path: Path):
    with pytest.raises(TaskError) as excinfo:
        load_one(tmp_path, id="gqa-0001", family="tool-trajectory")
    assert "gqa-" in str(excinfo.value)


def test_an_unknown_family_is_refused(tmp_path: Path):
    with pytest.raises(TaskError) as excinfo:
        load_one(tmp_path, id="qa-0001", family="vibes")
    assert "unknown family" in str(excinfo.value)


def test_an_unknown_expectation_type_is_refused_before_any_run(tmp_path: Path):
    """The check the registry exists for. Without it a typo scores the task zero and
    the failure list says `citatons` — which is at least a clue, but arrives after
    300 model calls."""
    with pytest.raises(TaskError) as excinfo:
        load_one(tmp_path, expectations=[{"type": "citatons", "must_cite": ["a"]}])
    assert "unknown expectation type" in str(excinfo.value)


def test_the_registry_says_which_families_use_each_type():
    """The table in `scorers/__init__.py` is the one copy; this test is what keeps the
    linter, the loader and the generators from disagreeing about a family.

    The two entries that are `()` — the widest setting — are pinned here with the reason,
    because both of them started narrower and the generator's own output broke the rule
    twice: a grounded task may check its trajectory, a safety task must be able to forbid a
    tool call, and a conversation cites its turns. A rule that keeps firing on the tasks it
    is meant to describe is the wrong rule.
    """
    from bench.scorers import EXPECTATION_TYPES

    assert EXPECTATION_TYPES["trajectory"].families == ()
    assert EXPECTATION_TYPES["steps"].families == ()
    assert EXPECTATION_TYPES["contains"].families == ()
    assert EXPECTATION_TYPES["citations"].families == ("grounded-qa", "multi-turn-state")
    assert EXPECTATION_TYPES["fields"].families == ("structured-extraction",)
    assert EXPECTATION_TYPES["refusal"].families == ("refusal-and-safety",)
    assert not EXPECTATION_TYPES["judge"].deterministic


def test_an_expectation_from_another_family_is_refused(tmp_path: Path):
    """`citations` decided by a scorer that has never seen this family's tasks is a
    silent zero, so the linter refuses it while there is still a filename to print."""
    with pytest.raises(TaskError) as excinfo:
        load_one(
            tmp_path,
            id="traj-0001",
            family="tool-trajectory",
            expectations=[{"type": "citations", "must_cite": ["passage"]}],
        )
    assert "belongs to grounded-qa or multi-turn-state" in str(excinfo.value)


def test_a_generic_expectation_is_allowed_in_any_family(tmp_path: Path):
    loaded = load_one(
        tmp_path,
        family="tool-trajectory",
        id="traj-0001",
        expectations=[{"type": "contains", "value": "x"}],
    )
    assert loaded.family == "tool-trajectory"


def test_a_private_task_needs_a_tag(tmp_path: Path):
    """Nobody develops against the private split, so a task there with no tags is a
    task nobody can group when the number moves."""
    with pytest.raises(TaskError) as excinfo:
        load_one(tmp_path, split="private", tags=[])
    assert "at least one tag" in str(excinfo.value)


def test_the_same_id_twice_is_refused(tmp_path: Path):
    root = write_set(tmp_path, [task(), task(input={"question": "another"})])
    with pytest.raises(TaskError) as excinfo:
        load_tasks(root)
    assert "defined twice" in str(excinfo.value)


def test_the_same_task_in_both_splits_is_refused(tmp_path: Path):
    """The exact-fingerprint leak. Two ids, one question — which is what a generator
    re-run with a different seed produces when nobody diffs the output."""
    root = write_set(
        tmp_path,
        [
            task(id="gqa-0001", split="public"),
            task(id="gqa-0002", split="private", tags=["retrieval"]),
        ],
    )
    with pytest.raises(TaskError) as excinfo:
        load_tasks(root)
    assert "identical to public task" in str(excinfo.value)


def test_a_missing_corpus_is_a_load_error(tmp_path: Path):
    with pytest.raises(TaskError) as excinfo:
        load_one(tmp_path, corpus="corpus/not-there")
    assert "corpus" in str(excinfo.value)


def test_a_corpus_that_exists_is_accepted(tmp_path: Path):
    make_corpus(tmp_path)
    loaded = load_one(tmp_path, corpus="corpus")
    assert loaded.corpus == "corpus"


# --- fingerprints -----------------------------------------------------------


def test_the_fingerprint_covers_what_the_agent_is_asked(tmp_path: Path):
    first = load_one(tmp_path, input={"question": "why?"})
    second = load_one(tmp_path, input={"question": "why? "})
    assert first.fingerprint != second.fingerprint


def test_bookkeeping_does_not_change_the_fingerprint(tmp_path: Path):
    """Moved to the private split, gained a tag, given a date: same question, same
    fingerprint. If these counted, the board would refuse every comparison across a
    tidy-up commit."""
    first = load_one(tmp_path, split="public", tags=[], added=None)
    second = load_one(tmp_path, split="private", tags=["retrieval"], added="2026-09-30")
    assert first.fingerprint == second.fingerprint


def test_changing_the_scoring_changes_the_fingerprint(tmp_path: Path):
    first = load_one(tmp_path, expectations=[{"type": "contains", "value": "permission"}])
    second = load_one(tmp_path, expectations=[{"type": "contains", "value": "policy"}])
    assert first.fingerprint != second.fingerprint


def test_a_retired_task_keeps_its_fingerprint(tmp_path: Path):
    first = load_one(tmp_path)
    second = load_one(tmp_path, retired="2026-09-30")
    assert first.fingerprint == second.fingerprint
    assert second.retired == "2026-09-30"


# --- the task set as a whole -------------------------------------------------


def test_the_digest_moves_when_a_task_changes_and_not_when_it_is_retired(tmp_path: Path):
    root = write_set(tmp_path, [task()])
    before = load_tasks(root).digest
    again = load_tasks(root).digest
    assert before == again
    root = write_set(
        tmp_path,
        [
            task(),
            task(id="traj-0001", family="tool-trajectory", split="private", tags=["tools"]),
        ],
    )
    assert load_tasks(root).digest != before


def test_retired_tasks_are_not_in_the_split(tmp_path: Path):
    """Retirement is a policy, and the point of it is that a task everybody passes
    stops being scored without vanishing from the record."""
    root = write_set(
        tmp_path,
        [
            task(id="gqa-0001"),
            task(id="gqa-0002", retired="2026-09-30"),
        ],
    )
    loaded = load_tasks(root)
    assert [item.id for item in loaded.split("public")] == ["gqa-0001"]
    assert len(loaded.tasks) == 2
    assert "gqa-0002" in {item.id for item in loaded.tasks}


def test_split_and_family_views_agree_with_the_whole_set(tmp_path: Path):
    root = write_set(
        tmp_path,
        [
            task(id="gqa-0001"),
            task(id="gqa-0002"),
            task(id="traj-0001", family="tool-trajectory", split="private", tags=["tools"]),
        ],
    )
    loaded = load_tasks(root)
    assert len(loaded.split("public")) == 2
    assert len(loaded.split("private")) == 1
    assert len(loaded.family("grounded-qa")) == 2
    assert loaded.families() == ["grounded-qa", "tool-trajectory"]
    assert "tool-trajectory 0p/1v" in loaded.describe()


def test_an_unknown_split_is_refused(tmp_path: Path):
    loaded = load_tasks(write_set(tmp_path, [task()]))
    with pytest.raises(TaskError, match="unknown split"):
        loaded.split("holdout")


def test_by_id_names_the_missing_task(tmp_path: Path):
    loaded = load_tasks(write_set(tmp_path, [task()]))
    assert loaded.by_id("gqa-0001").id == "gqa-0001"
    with pytest.raises(TaskError, match="gqa-9999"):
        loaded.by_id("gqa-9999")


def test_the_linter_can_be_called_on_a_hand_built_set(tmp_path: Path):
    """`lint` is a function, not a side effect of loading, so the generators can
    check their own output before writing it to disk."""
    loaded = load_tasks(write_set(tmp_path, [task()]))
    lint(loaded)  # does not raise
    broken = TaskSet([loaded.tasks[0]], root=tmp_path)
    assert len(broken) == 1


# --- near duplicates, which is the check a re-run generator needs ------------


def test_near_duplicates_across_splits_are_found(tmp_path: Path):
    question = "why did the v2 rollout stop refunding orders after the deploy"
    root = write_set(
        tmp_path,
        [
            task(id="gqa-0001", split="public", input={"question": question}),
            task(
                id="gqa-0002",
                split="private",
                tags=["retrieval"],
                input={"question": question + " on tuesday"},
            ),
        ],
    )
    loaded = load_tasks(root)
    pairs = find_near_duplicates(loaded)
    assert pairs
    assert pairs[0][0] == "gqa-0002" and pairs[0][1] == "gqa-0001"
    assert pairs[0][2] > 0.9


def test_unrelated_tasks_are_not_near_duplicates(tmp_path: Path):
    root = write_set(
        tmp_path,
        [
            task(id="gqa-0001", input={"question": "what broke the refund path" * 3}),
            task(
                id="gqa-0002",
                split="private",
                tags=["retrieval"],
                input={"question": "how do I rotate an api token safely" * 3},
            ),
        ],
    )
    assert find_near_duplicates(load_tasks(root)) == []


# --- the family table --------------------------------------------------------


def test_every_family_has_a_description_and_a_prefix():
    """The table is written once; this is the test that keeps the three users of it —
    the loader, the linter, the generators — from disagreeing about a family."""
    from bench.core.task import PREFIXES

    assert set(FAMILIES) == set(PREFIXES)
    for name, description in FAMILIES.items():
        assert len(description) > 20, name
        prefix = PREFIXES[name]
        assert prefix.isalpha() and prefix.islower(), prefix
        assert not prefix.endswith("-"), "the separator is added by the id format"


def test_a_task_with_no_limits_and_no_budget_is_valid():
    """Most tasks have neither, and requiring them would push authors to invent
    numbers — which is worse than no number, because an invented one gets compared
    against."""
    from bench.core.task import Budget, Limits

    assert Budget().is_empty
    assert Limits().max_steps is None


def test_a_budget_of_zero_is_refused():
    from bench.core.task import Budget

    with pytest.raises(ValueError, match="never be run"):
        Budget(max_usd=0.0)


def test_max_steps_has_to_allow_one_step():
    from bench.core.task import Limits

    with pytest.raises(ValueError, match="at least 1"):
        Limits(max_steps=0)


def test_an_expectation_describes_itself_for_the_failure_list():
    """The failure list on a report card is read by somebody with thirty seconds, so
    it prints `contains(value='permission')` and not a pydantic dump."""
    from bench.core.task import Expectation

    described = Expectation.model_validate({"type": "contains", "value": "permission"}).describe()
    assert described == "contains(value='permission')"
    assert Expectation.model_validate({"type": "answerable"}).describe() == "answerable"
