"""What a target is, and the config that describes one.

A target is anything that turns a task into a trace. Four kinds ship — `scripted`
(a Python callable), `http` (a deployed endpoint), `openai` (a chat-completions
model, for benchmarking a prompt without a whole agent around it) and `cassette`
(recorded traces, replayed). They share one contract:

    async def run(task: Task, *, repeat: int) -> Trace
    async def aclose() -> None

and one rule, which is the same rule `passmark` is built on: **a failure is a value,
not an exception.** A target that times out returns a `Trace` with `error` set, so a
hundred tasks still produce a hundred results and the report can say "37 tasks failed
because the endpoint refused connections" rather than stopping at the first. The one
exception is a target that is *misconfigured* — an `endpoint` that is not a URL, a
`callable` that does not import — which raises `TargetError` and stops the run,
because a hundred identical configuration errors are not a hundred pieces of evidence.

The other decision in this file is `coerce_trace`. Anybody integrating an agent has
produced a string, a dict, a dataclass, or somebody else's response object, and
refusing all four means writing a mapping layer for every entrant. So the boundary
accepts what people actually have and reports precisely what it could not use —
including the case where the answer is a plain string, which is the most common
integration of all and does not deserve to be treated as a mistake.
"""

from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..core.budget import price_tokens
from ..core.task import Task
from ..core.trace import Message, Passage, TokenUsage, ToolCall, Trace
from ..errors import TargetError

Kind = Literal["scripted", "http", "openai", "cassette"]


class TargetConfig(BaseModel, extra="forbid"):
    """One `targets:` entry from a config file.

    `extra="forbid"` because almost every field here changes a *number* on the board:
    a typo in `price_prompt_per_mtok` that silently falls back to no price turns an
    entry's cost column into "not measured", and that is a wrong number rather than a
    missing one.
    """

    kind: Kind = "http"
    label: str | None = None
    """What to call this entrant in reports. `retrieval-v3`, not `target-1`."""

    # --- scripted ----------------------------------------------------------
    callable: str | None = None
    """`module.path:function`, imported from the config file's directory (`import_paths`)."""

    import_paths: list[str] = Field(default_factory=lambda: [".", ".."])
    """Directories the module is importable from, relative to the config file.

    Both defaults earn their place: `.` is the config next to the agent, `..` is the
    ordinary repository with the config one level down. The same defaults and the same
    reasoning as `passmark`'s, deliberately — two harnesses in one portfolio should not
    disagree about where a module lives.
    """

    env: dict[str, str] = Field(default_factory=dict)
    """Environment for the imported module. How two builds of one in-process agent get
    compared without two config files and a `source`d shell script. Applied to the whole
    process when the target is built — which is true of mutating the environment in
    general, and is why the runner builds one target per process."""

    # --- http / openai -----------------------------------------------------
    endpoint: str | None = None
    base_url: str | None = None
    model: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_s: float = 60.0
    retries: int = 0
    """Default 0. A retry hides a failure mode the benchmark exists to measure: an
    agent that fails 5% of the time because of a timeout fails 5% of the time. Turning
    retries on changes what is being measured, so the report prints the setting."""

    temperature: float | None = None
    max_tokens: int | None = None
    system_prompt: str | None = None
    request_timeout_s: float | None = None

    # --- cassette ----------------------------------------------------------
    cassette: str | None = None
    record_to: str | None = None
    """Where a recording run writes. Recording and replaying are the same object with
    different modes, because a recorder that is not the replayer drifts from it."""

    # --- prices (USD per million tokens) -----------------------------------
    price_prompt_per_mtok: float | None = None
    price_completion_per_mtok: float | None = None
    price_cached_per_mtok: float | None = None
    """Left `None` when unknown, and then `cost_usd` stays `None` and the report says
    "not measured". Substituting zero would claim the run was free."""

    strict: bool = False
    """Whether an unparseable response is an error or a traced failure. Off by default
    because the interesting benchmark is of agents, and an agent whose JSON is bad has
    told you something about the agent."""

    @property
    def name(self) -> str:
        return self.label or self.kind

    def price(self, tokens: TokenUsage) -> Decimal | None:
        return price_tokens(
            prompt_tokens=tokens.prompt,
            completion_tokens=tokens.completion,
            cached_tokens=tokens.cached,
            prompt_per_mtok=self.price_prompt_per_mtok,
            completion_per_mtok=self.price_completion_per_mtok,
            cached_per_mtok=self.price_cached_per_mtok,
        )


@runtime_checkable
class Target(Protocol):
    """The whole contract. Everything else about a target is its own business."""

    label: str

    async def run(self, task: Task, *, repeat: int) -> Trace: ...

    async def aclose(self) -> None: ...


class TargetFile(BaseModel):
    """A `target.yaml`: one entrant, named.

    A file rather than a flag because a submission is a configuration — model, prompt,
    retrieval, tools, harness version — and a bundle has to record the whole of it.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str | None = None
    target: TargetConfig
    notes: str | None = None


def load_target(path: str | Path) -> TargetFile:
    """Read a target file, resolving its relative paths against its own directory."""
    file = Path(path)
    if not file.exists():
        raise TargetError(f"no such target file: {file}")
    try:
        raw = yaml.safe_load(file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise TargetError(f"{file} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise TargetError(f"{file} should be a mapping with `name:` and `target:`")
    try:
        loaded = TargetFile.model_validate(raw)
    except ValidationError as exc:
        lines = [
            f"  - {'/'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        ]
        raise TargetError(f"{file} does not validate:\n" + "\n".join(lines)) from exc

    base = file.parent
    config = loaded.target
    config.import_paths = [_resolve(value, base) for value in config.import_paths]
    for field in ("cassette", "record_to"):
        value = getattr(config, field)
        if value:
            setattr(config, field, str(_resolve(value, base)))
    return loaded


def _resolve(value: str, base: Path) -> str:
    candidate = Path(value)
    return str(candidate if candidate.is_absolute() else (base / candidate).resolve())


def api_key(config: TargetConfig) -> str:
    """The key, or a refusal that names the variable.

    A missing key is a configuration error and stops the run. Sending the request
    without one produces a 401 per task, which reads in the report as "the entrant
    failed 300 tasks" — the most misleading possible outcome of a typo.
    """
    value = os.environ.get(config.api_key_env)
    if not value:
        raise TargetError(
            f"{config.api_key_env} is not set",
            hint=f"export {config.api_key_env}=… , or point api_key_env at the variable "
            f"that holds it",
        )
    return value


# --- the boundary -------------------------------------------------------------


def coerce_trace(result: Any, *, latency_ms: int, config: TargetConfig | None = None) -> Trace:
    """Turn whatever the target returned into a `Trace`.

    Deliberately generous, and deliberately precise about failure. The shapes people
    actually return, in the order they are tried:

    * a `Trace` — used as is
    * a string — the answer, and nothing else to say about it
    * a mapping — the trace fields, any subset
    * anything with `output` / `text` / `content` — somebody else's response object

    None of them is a mistake. What *is* a mistake is something with no answer in it at
    all — an `int`, a `None`, a list — and the message says what it got, because
    "expected a trace" leaves the integrator guessing which of their four types the
    boundary wanted.
    """
    if isinstance(result, Trace):
        trace = result
    elif isinstance(result, str):
        trace = Trace(output=result, latency_ms=latency_ms)
    elif isinstance(result, dict):
        payload = dict(result)
        payload.setdefault("latency_ms", latency_ms)
        try:
            trace = Trace.model_validate(payload)
        except ValidationError as exc:
            raise TargetError(
                f"the target returned a mapping that is not a trace: {exc.errors()[0]['msg']}",
                hint="a trace is `{'output': '…', 'tool_calls': […], 'tokens': {…}}`",
            ) from exc
    elif result is None:
        raise TargetError(
            "the target returned None",
            hint="return a string answer, or a dict of trace fields, or a Trace",
        )
    else:
        trace = _coerce_object(result, latency_ms=latency_ms)

    # Silence is the one thing not tolerated. A 200 whose body carries the right shape
    # and no answer in it would otherwise sail through as a trace with `output=None`,
    # score zero, and read in the failure list as "the entrant answered wrong" — when
    # the truth is "the entrant answered nothing". An empty *string* is a value: a model
    # that replied with nothing has told you something, and it is scored as such.
    if trace.output is None and not trace.tool_calls and trace.error is None:
        raise TargetError(
            "the response has no answer in it",
            hint="a trace needs `output` or `tool_calls`; a body with neither is a "
            "failure for this task, not a scored zero",
        )

    if trace.cost_usd is None and config is not None:
        trace.cost_usd = config.price(trace.tokens)
    if not trace.latency_ms:
        trace.latency_ms = latency_ms
    return trace


def _coerce_object(result: Any, *, latency_ms: int) -> Trace:
    """The duck-typed path: somebody else's response object.

    `output` first, then `text`, then `content` — the three names the ecosystems
    actually use — and a `messages`-style list is read only if it holds mappings with a
    `role`, which is what a wire-format response looks like once it has been parsed.
    """
    output: str | None = None
    for attribute in ("output", "text", "content"):
        value = getattr(result, attribute, None)
        if isinstance(value, str):
            output = value
            break

    tool_calls: list[ToolCall] = []
    raw_calls = getattr(result, "tool_calls", None) or []
    for call in raw_calls:
        if isinstance(call, ToolCall):
            tool_calls.append(call)
            continue
        name = getattr(call, "name", None) or getattr(getattr(call, "function", None), "name", None)
        arguments = getattr(call, "arguments", None)
        if arguments is None:
            arguments = getattr(getattr(call, "function", None), "arguments", None)
        if isinstance(arguments, str):
            try:
                import json

                arguments = json.loads(arguments)
            except ValueError:
                arguments = {"_raw": arguments}
        if name:
            tool_calls.append(ToolCall(name=str(name), arguments=arguments or {}))

    if output is None and not tool_calls:
        raise TargetError(
            f"the target returned {type(result).__name__}, which has no answer in it",
            hint="return a string, a dict of trace fields, or an object with `.output`",
        )

    tokens = TokenUsage()
    usage = getattr(result, "usage", None)
    if usage is not None:
        tokens = TokenUsage(
            prompt=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion=int(getattr(usage, "completion_tokens", 0) or 0),
            cached=int(getattr(usage, "cached_tokens", 0) or 0),
        )

    return Trace(
        output=output,
        tool_calls=tool_calls,
        tokens=tokens,
        latency_ms=int(getattr(result, "latency_ms", 0) or latency_ms),
        raw={"python_type": type(result).__name__},
    )


def describe_error(exc: BaseException) -> str:
    """`ConnectError: connection refused`, not a forty-line traceback.

    The error string ends up in a failure list beside two hundred others, so it says
    what broke in the words of the thing that broke.
    """
    text = str(exc).strip() or exc.__class__.__name__
    if len(text) > 300:
        text = text[:297] + "…"
    return f"{type(exc).__name__}: {text}"


def messages_from_input(task_input: dict[str, Any]) -> list[Message]:
    """Build a chat request from a task's input.

    Reads `messages` when the task has them, a bare `question` or `prompt` otherwise.
    `retrieved` is passed through as context *when the task carries it* — which is how
    a grounded-qa task tests a prompt rather than the entrant's retriever.
    """
    if isinstance(task_input.get("messages"), list):
        return [Message.model_validate(entry) for entry in task_input["messages"]]
    question = task_input.get("question") or task_input.get("prompt")
    if not question:
        raise TargetError(
            "the task has no `question`, `prompt` or `messages` to send",
            hint="every family's tasks carry one of the three; check the task's input block",
        )
    content = str(question)
    passages = task_input.get("passages")
    if isinstance(passages, list) and passages:
        # Rendered rather than sent as JSON: a model reads a labelled block better than
        # an array of objects, and the format is fixed so that two entrants' prompts
        # differ only in the entrant's own doing.
        blocks = []
        for index, passage in enumerate(passages, start=1):
            if isinstance(passage, dict):
                blocks.append(f"[{index}] {passage.get('text', '')}".strip())
            else:
                blocks.append(f"[{index}] {passage}".strip())
        content = "Passages:\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"
    return [Message(role="user", content=content)]


def passages_from_task(task: Task) -> list[Passage]:
    """Passages the *task* supplies, as opposed to ones an entrant retrieved."""
    raw = task.input.get("passages")
    if not isinstance(raw, list):
        return []
    out = []
    for entry in raw:
        if isinstance(entry, dict) and "id" in entry and "text" in entry:
            out.append(Passage.model_validate(entry))
    return out
