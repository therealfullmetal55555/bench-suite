"""Fixtures for the core tests.

Two factories, `task()` and `write_set()`, so that a test asserting "a private task
needs a tag" writes one line about tags rather than eleven lines of YAML. Every test
file in this repository was written against the same helpers for that reason: a test
that is mostly setup is a test nobody reads before changing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import yaml


def task(**overrides: Any) -> dict[str, Any]:
    """A valid task, with anything overridable.

    Valid on purpose: every negative test then reads as the one thing that is wrong
    with it, which is the difference between a test that documents a rule and a test
    that documents a fixture.
    """
    base: dict[str, Any] = {
        "id": "gqa-0001",
        "family": "grounded-qa",
        "split": "public",
        "input": {"question": "why did the rollout stop?"},
        "expectations": [{"type": "contains", "value": "permission"}],
        "because": "A real outage: the answer blamed the prompt when it was a policy.",
        "tags": ["retrieval"],
    }
    base.update(overrides)
    return base


def write_set(tmp_path: Path, tasks: list[dict[str, Any]], name: str = "tasks.yaml") -> Path:
    """Write a `tasks:` file and return the directory, ready for `load_tasks`."""
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"tasks": tasks}, sort_keys=False), encoding="utf-8")
    return path.parent


def make_corpus(tmp_path: Path, body: str | None = None, name: str = "docs/incident.md") -> Path:
    """Write a small markdown corpus and return the directory."""
    default = (
        "# Incident 2026-08\n\n"
        "A short note about the rollout.\n\n"
        "## Root cause\n\n"
        "The v2 rollout stopped refunding because a permission change removed the\n"
        "billing scope from the worker's role. The prompt was not involved.\n\n"
        "## Timeline\n\n"
        "Deployed at 09:14, rolled back at 09:40.\n"
    )
    file = tmp_path / "corpus" / name
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(default if body is None else body, encoding="utf-8")
    return tmp_path / "corpus"


@pytest.fixture
def corpus_root(tmp_path: Path) -> Path:
    return make_corpus(tmp_path)


@pytest.fixture(autouse=True)
def _importable_fixture_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    """The `fake_agents` module has to be importable by path, because that is how a
    scripted target loads an entrant's code. One line here beats `sys.path` surgery in
    every test file, and the sessions where it is missing produce an error message about
    a module nobody wrote."""
    tests_dir = str(Path(__file__).parent)
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
