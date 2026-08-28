"""bench-suite — a public benchmark for retrieval and tool-using agents.

Read `SPEC.md` first, then `README.md`. The one design decision this project is
built around is in `core/score.py`: a score is a distribution over tasks, not a
float, and nothing in this package returns a bare number that claims to be a result.
"""

from __future__ import annotations

from .core.budget import BudgetMeter, price_tokens
from .core.corpus import Corpus
from .core.score import Score, TaskScore, family_mean, stratified_mean, summarise
from .core.stats import Interval, bootstrap_interval, paired_bootstrap_interval
from .core.task import FAMILIES, SPLITS, Budget, Expectation, Limits, Task, TaskSet, load_tasks
from .core.trace import Message, Passage, TokenUsage, ToolCall, Trace, schema, write_schema
from .errors import (
    EXIT_OK,
    EXIT_RUN_FAILED,
    EXIT_UNPUBLISHABLE,
    EXIT_WORSE,
    BenchError,
    BudgetExceeded,
    JudgeUncalibrated,
    TargetError,
    TaskError,
    UnpublishableResult,
)

__version__ = "0.1.0"

__all__ = [
    "EXIT_OK",
    "EXIT_RUN_FAILED",
    "EXIT_UNPUBLISHABLE",
    "EXIT_WORSE",
    "FAMILIES",
    "SPLITS",
    "BenchError",
    "Budget",
    "BudgetExceeded",
    "BudgetMeter",
    "Corpus",
    "Expectation",
    "Interval",
    "JudgeUncalibrated",
    "Limits",
    "Message",
    "Passage",
    "Score",
    "TargetError",
    "Task",
    "TaskError",
    "TaskScore",
    "TaskSet",
    "TokenUsage",
    "ToolCall",
    "Trace",
    "UnpublishableResult",
    "__version__",
    "bootstrap_interval",
    "family_mean",
    "load_tasks",
    "paired_bootstrap_interval",
    "price_tokens",
    "schema",
    "stratified_mean",
    "summarise",
    "write_schema",
]
