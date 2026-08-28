"""The prompt-only target: an OpenAI-compatible chat endpoint.

For benchmarking a prompt without a whole agent around it, and for the fixture
entrants in the examples. The wire format is the one everybody implements, so the same
target drives a hosted model, a local server, and a test stand-in that answers with a
fixed string.

Three things this file is careful about, all of them costing a leaderboard its
credibility when they go wrong:

**Tokens are recorded even when the provider forgets.** A response with no `usage`
block is priced from a count the harness makes itself, and the trace says which of the
two happened (`raw.priced_from`). An entry whose cost column is silently zero for the
first ten tasks and real for the rest is worse than one with no cost column.

**Retries are off by default**, same reasoning as the HTTP target, and the setting is
recorded.

**A malformed response is a failed task, not a crash.** A model that returns prose where
JSON was asked for has told you something about the model, and that belongs in the
failure list, not in a stack trace.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx

from ..core.task import Task
from ..core.trace import Message, TokenUsage, ToolCall, Trace
from ..errors import TargetError
from .base import TargetConfig, api_key, describe_error, messages_from_input

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class OpenAITarget:
    def __init__(self, label: str, config: TargetConfig) -> None:
        self.label = label
        self.config = config
        if not config.model:
            raise TargetError("an openai target needs a `model`")
        self._base_url = (config.base_url or config.endpoint or "https://api.openai.com/v1").rstrip(
            "/"
        )
        if not self._base_url.startswith(("http://", "https://")):
            raise TargetError(
                f"base_url should be an absolute URL, got {self._base_url!r}",
                hint="the harness will not guess a scheme: a wrong URL is 300 failed tasks",
            )
        self._key: str | None = None
        self._client: httpx.AsyncClient | None = None
        self._timeout = config.request_timeout_s or config.timeout_s

    # --- the call ----------------------------------------------------------

    async def run(self, task: Task, *, repeat: int) -> Trace:  # noqa: ARG002
        # `repeat` is deliberately not sent anywhere: for a prompt-only target the
        # repeats *are* the measurement. Three identical requests that come back
        # differently is the model's own variance, and seeding or warming around it
        # would hide the number the flakiness column exists to report.
        started = time.perf_counter()
        try:
            messages = messages_from_input(task.input)
        except TargetError as exc:
            return Trace(error=str(exc), latency_ms=_ms(started))

        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [self._wire(message) for message in messages],
        }
        if self.config.temperature is not None:
            body["temperature"] = self.config.temperature
        if self.config.max_tokens is not None:
            body["max_tokens"] = self.config.max_tokens

        attempt = 0
        last_error: str | None = None
        while True:
            try:
                response = await self._client_or_new().post(
                    f"{self._base_url}/chat/completions",
                    json=body,
                    headers={
                        "Authorization": f"Bearer {self._key_or_raise()}",
                        "Content-Type": "application/json",
                        **self.config.headers,
                    },
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = describe_error(exc)
            else:
                if response.status_code in RETRYABLE_STATUS:
                    last_error = f"HTTP {response.status_code}: {_flat(response.text)}"
                elif response.status_code >= 400:
                    return Trace(
                        error=f"HTTP {response.status_code}: {_flat(response.text)}",
                        latency_ms=_ms(started),
                        raw={"status": response.status_code, "attempts": attempt + 1},
                    )
                else:
                    return self._parse(response, started, messages)
            if attempt >= self.config.retries:
                return Trace(
                    error=last_error, latency_ms=_ms(started), raw={"attempts": attempt + 1}
                )
            attempt += 1
            await asyncio.sleep(min(2.0**attempt * 0.1, 2.0))

    # --- the response ------------------------------------------------------

    def _parse(self, response: httpx.Response, started: float, sent: list[Message]) -> Trace:
        try:
            payload = response.json()
        except ValueError as exc:
            return Trace(
                error=f"the endpoint did not return JSON: {describe_error(exc)}",
                latency_ms=_ms(started),
            )
        if not isinstance(payload, dict):
            return Trace(error="the response is not a JSON object", latency_ms=_ms(started))
        try:
            choice = payload["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError):
            # A shape error is described rather than raised: the provider's error body
            # is usually in `payload`, and it is the useful part.
            summary = _flat(json.dumps(payload))[:200] if payload else "an empty object"
            return Trace(
                error=f"the response has no choices[0].message — got {summary}",
                latency_ms=_ms(started),
                raw={"status": response.status_code},
            )

        content = message.get("content")
        if isinstance(content, list):
            # Multimodal-shaped content: keep the text parts, note that there were
            # others. Dropping them silently would make a vision model look like it
            # answered from nothing.
            parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            content = "".join(parts)
        output = content if isinstance(content, str) else None

        tool_calls: list[ToolCall] = []
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    arguments = {"_raw": arguments}
            tool_calls.append(
                ToolCall(name=str(function.get("name", "")), arguments=arguments or {})
            )

        tokens, priced_from = _usage(payload, sent, output, tool_calls)
        trace = Trace(
            messages=[*sent, Message(role="assistant", content=output or "")],
            output=output,
            tool_calls=tool_calls,
            tokens=tokens,
            latency_ms=_ms(started),
            raw={
                "provider": "openai",
                "priced_from": priced_from,
                "model": payload.get("model", self.config.model),
            },
        )
        if payload.get("id"):
            trace.raw["id"] = payload["id"]
        if isinstance(choice.get("finish_reason"), str):
            trace.raw["finish_reason"] = choice["finish_reason"]
        trace.cost_usd = self.config.price(tokens)
        return trace

    # --- plumbing ----------------------------------------------------------

    def _wire(self, message: Message) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.name:
            payload["name"] = message.name
        return payload

    def _key_or_raise(self) -> str:
        if self._key is None:
            self._key = api_key(self.config)
        return self._key

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


def _usage(
    payload: dict[str, Any], sent: list[Message], output: str | None, tool_calls: list[ToolCall]
) -> tuple[TokenUsage, str]:
    """Tokens from the provider when it reports them, from the harness when it does not.

    The fallback is `len(text) // 4`, which is the estimate every tokeniser blog post
    uses and is right to within about a fifth on English prose. It is labelled
    `estimated` in the trace so that a cost column built on it can be read with the
    right amount of suspicion — and so a reviewer can tell an entry whose prices came
    from the provider apart from one where they did not.
    """
    usage = payload.get("usage")
    if isinstance(usage, dict) and usage:
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens") if isinstance(details, dict) else None
        return (
            TokenUsage(
                prompt=int(usage.get("prompt_tokens", 0) or 0),
                completion=int(usage.get("completion_tokens", 0) or 0),
                cached=int(cached or 0),
            ),
            "provider",
        )
    prompt_chars = sum(len(message.content) for message in sent)
    completion_chars = len(output or "") + sum(len(str(call.arguments)) for call in tool_calls)
    return TokenUsage(prompt=prompt_chars // 4, completion=completion_chars // 4), "estimated"


def _flat(text: str, limit: int = 200) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1000))


def replay_client(responses: dict[str, dict[str, Any]]) -> httpx.MockTransport:
    """A transport that answers from a dict of canned payloads, keyed by the last user
    message. Used by the tests and by `--offline` in the examples."""
    import httpx as _httpx

    def handler(request: _httpx.Request) -> _httpx.Response:
        body = json.loads(request.content)
        question = next(
            (
                message.get("content", "")
                for message in reversed(body.get("messages", []))
                if message.get("role") == "user"
            ),
            "",
        )
        for needle, payload in responses.items():
            if needle.lower() in str(question).lower():
                return _httpx.Response(200, json=payload)
        return _httpx.Response(
            404, json={"error": {"message": f"no canned answer for {question!r}"}}
        )

    return _httpx.MockTransport(handler)
