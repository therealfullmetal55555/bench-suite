"""Exceptions, and the exit codes, in one file.

The exit codes are the interface for anything that runs this in CI, so they are
defined once and never invented at a call site. `1` and `3` are deliberately apart:
"this entry is worse than the one below it" and "this result cannot be published"
are different messages and collapsing them teaches people to ignore both.
"""

from __future__ import annotations

EXIT_OK = 0
"""Everything ran, and the claim the command makes is true."""

EXIT_WORSE = 1
"""An entry lost ground against the comparison it was given. This is the code a
gate is written against."""

EXIT_RUN_FAILED = 2
"""The run itself could not finish: a target that does not answer, a task file that
does not parse, a budget spent on nothing. A distinct code because the fix is
different — this is my problem, not the entrant's."""

EXIT_UNPUBLISHABLE = 3
"""The run finished, and the result may not go on the board: judge below the
calibration floor, private split leaked into the public one, contaminated task,
or a bundle whose summary disagrees with its own per-task results."""


class BenchError(Exception):
    """Anything this project raises on purpose. `cli.py` turns it into a message
    and an exit code; nothing else catches it."""

    exit_code = EXIT_RUN_FAILED

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


class TaskError(BenchError):
    """A task file is wrong: a missing `because`, an unknown expectation type, two
    tasks with the same id in one split, a corpus that is not there."""


class TargetError(BenchError):
    """The thing being benchmarked is misconfigured. Deliberately *not* the same as
    a task that fails: a misconfigured target is the entrant's configuration
    problem and should stop the run, while a failed task is a data point."""


class BudgetExceeded(BenchError):
    """A task or a run spent more than it was allowed to.

    Not an error in the sense of a crash: it is an outcome, and `runner.py` turns it
    into a `over-budget` task result. It is raised only where the caller asked for a
    hard stop."""


class UnpublishableResult(BenchError):
    """Raised by the board and the verifier when a result may not be published. The
    message has to say *why*, because the reason is the useful part."""

    exit_code = EXIT_UNPUBLISHABLE


class JudgeUncalibrated(UnpublishableResult):
    """The judge scored tasks without a calibration on file, or with a κ below the
    floor. On the board this is a column, not a scandal — but the family is
    excluded from the aggregate and named as excluded."""
