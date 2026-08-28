"""The in-process target: import a callable and call it.

This is the integration for a team whose agent already lives in Python, and for the
fixture entrants the tests use. The import happens once, at construction, so a typo in
the import path is a startup error rather than three hundred identical per-task errors
— the difference between "you misconfigured the harness" and "your agent is broken",
and confusing those two is how a benchmark loses a team's trust on day one.

The same three lessons `passmark` paid for apply here, and are implemented rather than
remembered:

* the config's `import_paths` go on `sys.path` before the import, because the config
  file is what says where its code lives — without this the harness works from a
  checkout and fails under CI, where nobody stands in the repository root;
* a file path where an import path belongs gets a message saying what to write;
* Python caches imports per module name, which is stated out loud because it costs
  somebody an afternoon otherwise.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..core.task import Task
from ..core.trace import Trace
from ..errors import TargetError
from .base import TargetConfig, coerce_trace, describe_error


class ScriptedTarget:
    """Call `module.path:function(input, repeat=n)`.

    The callable may be sync or async and may return a `Trace`, a string, a mapping, or
    an object with `.output` — `coerce_trace` sorts out which. It is also told how many
    repeats it is on, so a fixture entrant can be deliberately flaky without any of the
    harness knowing that is what it is.
    """

    def __init__(self, label: str, config: TargetConfig) -> None:
        self.label = label
        self.config = config
        for root in reversed(config.import_paths):
            resolved = str(Path(root).resolve())
            if resolved not in sys.path:
                sys.path.insert(0, resolved)
        if config.env:
            # before the import: a module that reads its configuration at import time
            # sees the right value
            os.environ.update(config.env)

        module_path, _, attribute = (config.callable or "").partition(":")
        if not module_path or not attribute:
            raise TargetError(
                f"callable must look like 'module.path:function', got {config.callable!r}"
            )
        if module_path.endswith(".py") or "/" in module_path or "\\" in module_path:
            dotted = module_path[: -len(".py")] if module_path.endswith(".py") else module_path
            hint = dotted.strip("./").replace("/", ".").replace("\\", ".")
            raise TargetError(
                f"callable takes an import path, not a file path: try {hint}:{attribute}. "
                f"The module has to be importable from the directory the config lives in"
            )
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise TargetError(f"cannot import {module_path}: {exc}") from exc
        try:
            self._function: Callable[..., Any] = getattr(module, attribute)
        except AttributeError as exc:
            raise TargetError(f"{module_path} has no attribute {attribute!r}") from exc
        if not callable(self._function):
            raise TargetError(f"{config.callable} is not callable")

        self._wants_repeat = _accepts_repeat(self._function)

    async def run(self, task: Task, *, repeat: int) -> Trace:
        started = time.perf_counter()
        try:
            if self._wants_repeat:
                result = self._function(task.input, repeat=repeat)
            else:
                result = self._function(task.input)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:  # noqa: BLE001 - a failure is a value, by design
            return Trace(error=describe_error(exc), latency_ms=_ms(started))
        try:
            return coerce_trace(result, latency_ms=_ms(started), config=self.config)
        except TargetError as exc:
            # A target that returns something unusable is a failure of *this task*, not
            # of the run: the other 299 tasks still have something to say.
            return Trace(error=str(exc), latency_ms=_ms(started))

    async def aclose(self) -> None:
        """An in-process target has nothing to close, and the method exists so the
        runner does not have to know that."""
        return


def _accepts_repeat(function: Callable[..., Any]) -> bool:
    """Whether the callable takes `repeat`.

    Checked rather than always passed, because an integrator's function is their
    function and forcing a keyword onto it means every entrant writes a wrapper. An
    unreadable signature is treated as "no", which fails towards the simpler call.
    """
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):  # pragma: no cover - builtins and C functions
        return False
    if "repeat" in signature.parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1000))


async def really_async(value: Any) -> Any:  # pragma: no cover - convenience for entrants
    """Helper for a target that wants to be a coroutine without writing `async def`."""
    if inspect.isawaitable(value):
        return await value
    await asyncio.sleep(0)
    return value
