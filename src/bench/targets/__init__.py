"""Targets: the four ways to point the harness at an agent.

`build_target` is the only name the rest of the package needs. It reads a
`TargetConfig` and returns something with `run(task, repeat=…)`, raising `TargetError`
for a configuration that cannot work — an import path that does not import, an endpoint
that is not a URL, a cassette that is not there. Those are *startup* errors on purpose:
the alternative is three hundred identical failures in the report, which reads as "the
entrant is bad" when the truth is "the entrant is misspelled".
"""

from __future__ import annotations

from ..errors import TargetError
from .base import (
    Target,
    TargetConfig,
    TargetFile,
    api_key,
    coerce_trace,
    describe_error,
    load_target,
    messages_from_input,
)
from .cassette import FORMAT_VERSION, CassetteTarget
from .http import HttpTarget
from .openai import OpenAITarget
from .scripted import ScriptedTarget

__all__ = [
    "FORMAT_VERSION",
    "CassetteTarget",
    "HttpTarget",
    "OpenAITarget",
    "ScriptedTarget",
    "Target",
    "TargetConfig",
    "TargetFile",
    "api_key",
    "build_target",
    "coerce_trace",
    "describe_error",
    "load_target",
    "messages_from_input",
]


def build_target(
    name: str,
    config: TargetConfig,
    *,
    cassette: str | None = None,
    record_to: str | None = None,
    transport: object | None = None,
) -> Target:
    """Construct the target a config describes.

    `cassette` and `record_to` override the config, which is how `--from-cassette`
    works in CI without a second config file: the same `target.yaml` drives a live run
    and an offline one.

    A cassette around a live target is the recording mode — replay what is recorded,
    call through for the rest — and it wraps *any* kind, which is why the wrapping
    happens after the inner target is built rather than in a branch of the same `if`.
    """
    label = config.name if config.label else name
    if cassette:
        config = config.model_copy(update={"cassette": cassette})
    if record_to:
        config = config.model_copy(update={"record_to": record_to})

    if config.kind == "cassette":
        return CassetteTarget(label, config)

    inner: Target
    if config.kind == "scripted":
        inner = ScriptedTarget(label, config)
    elif config.kind == "http":
        inner = HttpTarget(label, config)
    elif config.kind == "openai":
        inner = OpenAITarget(label, config)
    else:  # pragma: no cover - the Literal already restricts this
        raise TargetError(f"unknown target kind {config.kind!r}")

    # Tests and `--offline` replace the transport rather than the client, so the retry,
    # parsing and pricing code under test stays the code that runs live. The isinstance
    # is the reason `build_target` is a function and not a classmethod on the protocol: a
    # test seam is allowed to know which class it is testing.
    if transport is not None and isinstance(inner, HttpTarget | OpenAITarget):
        inner.replace_transport(transport)

    if config.cassette or config.record_to:
        # One object, two modes, which is why the recorder and the replayer cannot
        # drift apart: a separate recorder is a recorder that writes a file the replayer
        # can no longer read, and nobody notices until an old bundle fails to verify.
        return CassetteTarget(label, config, inner=inner)
    return inner
