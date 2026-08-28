"""The deployed target: POST the task, read a trace back.

This is where the interesting failures live — timeouts, 502s, a JSON body that changed
shape last Tuesday — so this file's job is to turn all of them into values that end up
in the report instead of exceptions that end the run.

Two decisions:

**A retry is off by default.** An agent that fails 5% of the time because of a timeout
fails 5% of the time, and a benchmark that retries is measuring a different product
from the one in production. `retries: 2` exists because sometimes the honest answer is
"the endpoint is flaky and I want to measure the model", and the report prints the
setting so the reader can tell which question was asked.

**The response is read as a trace, or as whatever it is.** `{"output": "..."}` and
`{"answer": "...", "tool_calls": [...]}` both arrive in the world, and an entrant should
not have to wrap their service in a shim to be measurable. What is *not* tolerated is
silence: a 200 with a body that contains no answer is a traced failure with the body's
first 200 characters quoted, because "it returned nothing useful" without the nothing
is the least debuggable sentence in software.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx

from ..core.task import Task
from ..core.trace import Trace
from ..errors import TargetError
from .base import TargetConfig, coerce_trace, describe_error

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
"""Worth another attempt when retries are on. 4xx other than these is a request the
service understood and refused, and retrying it three times just costs three times as
much to get the same answer."""


class HttpTarget:
    def __init__(self, label: str, config: TargetConfig) -> None:
        self.label = label
        self.config = config
        if not config.endpoint:
            raise TargetError(
                "an http target needs an `endpoint`",
                hint="endpoint: https://agent.example.com/run",
            )
        if not config.endpoint.startswith(("http://", "https://")):
            raise TargetError(
                f"endpoint should be an absolute URL, got {config.endpoint!r}",
                hint="a configuration error stops the run on purpose: the alternative is "
                "300 identical failures in the report",
            )
        self._client: httpx.AsyncClient | None = None
        self._timeout = config.request_timeout_s or config.timeout_s

    async def run(self, task: Task, *, repeat: int) -> Trace:
        started = time.perf_counter()
        payload = {
            "task_id": task.id,
            "family": task.family,
            "input": task.input,
            "repeat": repeat,
        }
        headers = {"Content-Type": "application/json", **self.config.headers}

        attempt = 0
        last_error: str | None = None
        while True:
            try:
                response = await self._client_or_new().post(
                    self.config.endpoint or "", json=payload, headers=headers
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = describe_error(exc)
            else:
                if response.status_code in RETRYABLE_STATUS:
                    last_error = f"HTTP {response.status_code}: {_quote(response.text)}"
                elif response.status_code >= 400:
                    # Not retried: the service understood and refused. Reported with the
                    # body, because a 400 from an agent endpoint usually explains itself.
                    return Trace(
                        error=f"HTTP {response.status_code}: {_quote(response.text)}",
                        latency_ms=_ms(started),
                        raw={"status": response.status_code},
                    )
                else:
                    return self._parse(response, started)
            if attempt >= self.config.retries:
                return Trace(
                    error=last_error, latency_ms=_ms(started), raw={"attempts": attempt + 1}
                )
            attempt += 1
            await asyncio.sleep(min(2.0**attempt * 0.1, 2.0))

    def _parse(self, response: httpx.Response, started: float) -> Trace:
        if not response.content:
            return Trace(error="the endpoint returned an empty body", latency_ms=_ms(started))
        try:
            body: Any = response.json()
        except ValueError as exc:
            return Trace(
                error=f"the endpoint did not return JSON: {describe_error(exc)}; "
                f"body starts with {_quote(response.text)}",
                latency_ms=_ms(started),
            )
        if not isinstance(body, dict):
            return Trace(
                error=f"the endpoint returned a {type(body).__name__}, not an object",
                latency_ms=_ms(started),
            )

        # The two field names that show up in every second integration. Mapped rather
        # than required, because asking an entrant to change their service's response to
        # be benchmarked is how a benchmark ends up with four entrants.
        payload = dict(body)
        if "output" not in payload and isinstance(payload.get("answer"), str):
            payload["output"] = payload.pop("answer")

        # Anything else the service sent — `request_id`, `status`, `cached`, a whole
        # debug object — goes into `raw` instead of failing validation. A 200 whose body
        # carries the right answer and one extra key is not a failed task, and treating
        # it as one would make the benchmark refuse the integrations it most wants: the
        # real ones. The keys are kept, so the run file still holds what arrived.
        known = set(Trace.model_fields)
        extra = {key: payload.pop(key) for key in list(payload) if key not in known}
        if extra:
            payload["raw"] = {**payload.get("raw", {}), **extra}

        try:
            trace = coerce_trace(payload, latency_ms=_ms(started), config=self.config)
        except TargetError as exc:
            # `strict` decides which side of the boundary this lands on: a harness
            # problem (raise, stop the run) or an entrant's problem (a failed task, and
            # the report says so).
            if self.config.strict:
                raise
            return Trace(error=str(exc), latency_ms=_ms(started))
        trace.raw.setdefault("status", response.status_code)
        for key in ("model", "id", "trace_id"):
            if key in body:
                trace.raw.setdefault(key, body[key])
        return trace

    def _client_or_new(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    def replace_transport(self, transport: Any) -> None:
        """Point this target at a canned transport. The seam tests use, and the one
        `--offline` uses: replacing the *transport* rather than the client keeps the
        retry, parsing and pricing code that runs under test identical to the code that
        runs live."""
        import httpx

        self._client = httpx.AsyncClient(transport=transport, timeout=self._timeout)

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()


def _quote(text: str, limit: int = 200) -> str:
    """The first couple of hundred characters of a body, on one line.

    A body quoted into a failure list has to survive being read in a table cell, so
    newlines become spaces and the tail is elided rather than wrapped.
    """
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "…"


def _ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1000))


def json_body(value: Any) -> str:  # pragma: no cover - used by the CLI's dry run
    """Render a payload the way `--dry-run` prints it: readable, and stable enough to
    diff between two runs."""
    return json.dumps(value, indent=2, sort_keys=True)
