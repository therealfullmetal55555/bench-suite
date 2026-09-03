"""The judge: one model call, one verdict, and a cache keyed by the arguments.

A judge is a function of `(rubric, question, answer)` and it is priced per call, so it is
cached on exactly those three plus the model name and temperature. The key is a hash of
the *arguments*, not of the task id: the same rubric applied to the same answer has the
same verdict whoever asks, and a cache keyed by task id would go stale the moment a task
was edited. That is also what keeps a re-run free — a submission re-scored against a fixed
harness costs no judge calls at all, and the calibration report says how many were served
from cache so that a reader can tell.

**The verdict format is fixed by the harness and parsed leniently.** `{"pass": true,
"reason": "..."}`, asked for in the system message; parsed by finding the first JSON object
in the reply, because models wrap answers in prose and fences no matter how plainly they
are asked not to. A reply with no parseable verdict is an *error*, not a fail: the judge
failing to answer says nothing about the entrant, and turning it into a zero would let an
outage become a regression.

**The temperature defaults to 0 and is recorded.** A judge at temperature 1 turns the
board's own noise into a result; the calibration's κ is measured at the setting the board
uses, and the setting is in the file.

Two things this file deliberately does not do: it does not let an entrant supply the model
or the prompt (SPEC §6), and it does not run the judge twice and pick the kinder answer.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from ..core.budget import price_tokens
from ..core.trace import TokenUsage
from ..errors import BenchError

SYSTEM_PROMPT = """You are grading one answer against one rubric.

Return a single JSON object and nothing else:

  {"pass": true, "reason": "one sentence"}

Rules:
- `pass` is true only when the rubric is fully satisfied.
- Judge the answer as written. Do not reward it for what it meant, and do not penalise it
  for being short if the rubric does not ask for length.
- The reason is one sentence, and it must name the part of the rubric that decided it.
"""


class JudgeError(BenchError):
    """The judge could not be asked, or did not answer with a verdict."""


@dataclass(frozen=True)
class JudgeVerdict:
    passed: bool | None
    """`None` when the judge did not answer. Deliberately not `False`: an unparseable
    reply is an outage, and an outage that scores zero is a regression nobody caused."""

    reason: str
    cached: bool = False
    cost_usd: Decimal | None = None
    latency_ms: int = 0
    raw: str = ""
    error: str | None = None
    score: int | None = None
    """A graded rubric's value, when the task asked for one. `passed` is still the thing
    the scorer reads; the score rides along in the result's detail so a graded task
    contributes its grade to the report instead of a thresholded coin flip."""

    @property
    def scored(self) -> bool:
        return self.passed is not None and self.error is None


@dataclass
class JudgeConfig:
    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "BENCH_JUDGE_KEY"
    temperature: float = 0.0
    timeout_s: float = 60.0
    cache_dir: str | None = "judge-cache"
    kappa_floor: float = 0.70
    """Below this the judge's verdicts are reported and excluded from the aggregate. 0.70
    is "substantial agreement" on the usual reading of κ, and it is a deliberate choice to
    be strict: a judge that agrees with a human 60% of the time is worse than no judge,
    because it launders a coin flip into a number."""

    price_prompt_per_mtok: float | None = None
    price_completion_per_mtok: float | None = None

    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> JudgeConfig:
        import yaml

        file = Path(path)
        if not file.exists():
            raise JudgeError(f"no judge file at {file}")
        raw = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict) or "model" not in raw:
            raise JudgeError(f"{file} needs at least a `model:`")
        known = {key: value for key, value in raw.items() if key in cls.__dataclass_fields__}
        extra = {key: value for key, value in raw.items() if key not in cls.__dataclass_fields__}
        config = cls(**known)
        config.extra = extra
        return config


class JudgeClient:
    def __init__(self, config: JudgeConfig) -> None:
        self.config = config
        self._sync_client: httpx.Client | None = None
        self.calls = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.spent_usd = Decimal("0")
        self._cache: dict[str, JudgeVerdict] = {}
        self._cache_dir = Path(config.cache_dir) if config.cache_dir else None
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._calibration: dict[str, Any] | None = _read_calibration(config)

    # --- the public call ----------------------------------------------------

    def verdict(
        self,
        *,
        rubric: str,
        prompt: str,
        output: str,
        task_id: str = "",
        scale: int | None = None,
    ) -> JudgeVerdict:
        """Judge one answer. Synchronous, because the scorers are.

        The runner scrapes repeats concurrently and then scores them in one pass; making
        the scorer async to overlap judge calls would put the concurrency in two places
        and make every scorer a coroutine for one family's benefit.

        `task_id` is carried for error messages only — it is deliberately absent from the
        cache key, because a verdict belongs to the answer that was judged and a cache
        that goes stale when a task is renamed is a cache that silently re-scores.
        """
        key = self._key(rubric=rubric, prompt=prompt, output=output, scale=scale)
        if key in self._cache:
            # Marked as cached and priced at zero, like the on-disk hits: this call cost
            # nothing, and the verdict that did cost something already added its price to
            # `spent_usd`. Returning the stored object as-is made the same question look
            # expensive the second time in the same process, which is the number the
            # calibration report prints.
            self.cache_hits += 1
            return replace(self._cache[key], cached=True, cost_usd=Decimal("0"))
        cached = self._load_cached(key)
        if cached is not None:
            self.cache_hits += 1
            self._cache[key] = cached
            return cached

        self.cache_misses += 1
        verdict = self._ask(
            rubric=rubric, prompt=prompt, output=output, scale=scale, task_id=task_id
        )
        if verdict.error is None:
            self._store(key, verdict)
        return verdict

    def close(self) -> None:
        if self._sync_client is not None and not self._sync_client.is_closed:
            self._sync_client.close()

    # --- calibration -------------------------------------------------------

    @property
    def calibration(self) -> dict[str, Any] | None:
        return self._calibration

    @property
    def trusted(self) -> bool:
        """Whether this judge's verdicts may decide a score.

        False when no calibration is on file, and false when κ is below the floor. Not a
        warning: a task judged by an uncalibrated judge is excluded from the aggregate by
        `score.py`, and the report names it as excluded.
        """
        if not self._calibration:
            return False
        kappa = self._calibration.get("kappa")
        return isinstance(kappa, int | float) and kappa >= self.config.kappa_floor

    def trust_note(self) -> str:
        if not self._calibration:
            return (
                "no calibration on file for this judge — run `bench judge calibrate`; "
                "judge-scored tasks are reported and excluded"
            )
        kappa = self._calibration.get("kappa")
        floor = self.config.kappa_floor
        if isinstance(kappa, int | float) and kappa >= floor:
            return f"κ {kappa:.2f} ≥ floor {floor:.2f}: judge-scored tasks count"
        return (
            f"κ {kappa:.2f} is below the floor {floor:.2f}: judge-scored tasks are excluded"
            if isinstance(kappa, int | float)
            else "the calibration file has no κ in it"
        )

    # --- the call ----------------------------------------------------------

    def _ask(
        self,
        *,
        rubric: str,
        prompt: str,
        output: str,
        scale: int | None = None,
        task_id: str = "",
    ) -> JudgeVerdict:
        started = time.perf_counter()
        body = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "system", "content": scaled_prompt(scale)},
                {"role": "user", "content": render(rubric=rubric, prompt=prompt, output=output)},
            ],
        }
        try:
            response = self._client_or_new().post(
                f"{self.config.base_url.rstrip('/')}/chat/completions",
                json=body,
                headers={
                    "Authorization": f"Bearer {self._key_or_raise()}",
                    "Content-Type": "application/json",
                    **self.config.extra.get("headers", {}),
                },
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            return JudgeVerdict(
                None,
                f"the judge is unreachable for {_which(task_id)}: {type(exc).__name__}: {exc}",
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=_ms(started),
            )
        if response.status_code >= 400:
            return JudgeVerdict(
                None,
                f"the judge returned HTTP {response.status_code} for {_which(task_id)}: "
                f"{_quote(response.text)}",
                error=f"HTTP {response.status_code}",
                latency_ms=_ms(started),
            )
        self.calls += 1
        return self._parse(response, started)

    def _parse(self, response: httpx.Response, started: float) -> JudgeVerdict:
        try:
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            return JudgeVerdict(
                None,
                f"the judge's reply has no message in it: {_quote(response.text)}",
                error="unparseable response",
                latency_ms=_ms(started),
            )

        usage = payload.get("usage") or {}
        tokens = TokenUsage(
            prompt=int(usage.get("prompt_tokens", 0) or 0),
            completion=int(usage.get("completion_tokens", 0) or 0),
        )
        cost = price_tokens(
            prompt_tokens=tokens.prompt,
            completion_tokens=tokens.completion,
            prompt_per_mtok=self.config.price_prompt_per_mtok,
            completion_per_mtok=self.config.price_completion_per_mtok,
        )
        if cost is not None:
            self.spent_usd += cost

        verdict = parse_verdict(content or "")
        if verdict is None:
            return JudgeVerdict(
                None,
                f"the judge did not answer with a verdict: {_quote(content or '')}",
                error="no verdict in the reply",
                cost_usd=cost,
                latency_ms=_ms(started),
                raw=content or "",
            )
        passed, reason, score = verdict
        return JudgeVerdict(
            passed=passed,
            reason=reason,
            score=score,
            cost_usd=cost,
            latency_ms=_ms(started),
            raw=content or "",
        )

    # --- the cache ---------------------------------------------------------

    def _key(self, *, rubric: str, prompt: str, output: str, scale: int | None = None) -> str:
        blob = json.dumps(
            {
                "model": self.config.model,
                "temperature": self.config.temperature,
                "version": 1,
                "scale": scale,
                "rubric": " ".join(rubric.split()),
                "prompt": " ".join(prompt.split()),
                "output": (output or "").strip(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    def _load_cached(self, key: str) -> JudgeVerdict | None:
        if self._cache_dir is None:
            return None
        file = self._cache_dir / f"{key}.json"
        if not file.exists():
            return None
        try:
            payload = json.loads(file.read_text(encoding="utf-8"))
            passed = payload["pass"]
        except (ValueError, KeyError):
            return None
        if not isinstance(passed, bool):
            return None
        score = payload.get("score")
        return JudgeVerdict(
            passed=passed,
            reason=str(payload.get("reason", "")),
            score=score if isinstance(score, int) else None,
            cached=True,
            cost_usd=Decimal("0"),
        )

    def _store(self, key: str, verdict: JudgeVerdict) -> None:
        self._cache[key] = verdict
        if self._cache_dir is None:
            return
        file = self._cache_dir / f"{key}.json"
        temporary = file.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "pass": verdict.passed,
                    "reason": verdict.reason,
                    "score": verdict.score,
                    "model": self.config.model,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(file)

    # --- plumbing ----------------------------------------------------------

    def _key_or_raise(self) -> str:
        value = os.environ.get(self.config.api_key_env)
        if not value:
            raise JudgeError(
                f"{self.config.api_key_env} is not set",
                hint=f"export {self.config.api_key_env}=… , or point api_key_env at the "
                f"variable that holds it",
            )
        return value

    def _client_or_new(self) -> httpx.Client:
        if self._sync_client is None or self._sync_client.is_closed:
            self._sync_client = httpx.Client(timeout=self.config.timeout_s)
        return self._sync_client

    def replace_transport(self, transport: Any) -> None:
        """The test seam, same as the targets': replace the transport, not the client.

        A transport rather than a URL, so the tests exercise the real request building,
        the real headers and the real parsing, and stub only the socket.
        """
        self._sync_client = httpx.Client(transport=transport, timeout=self.config.timeout_s)


# --- the pieces that are pure functions, and therefore tested as such ---------


def render(*, rubric: str, prompt: str, output: str, swap: bool = False) -> str:
    """The user message. `swap` reorders the rubric's halves.

    A rubric is a list of requirements, and a model that passes or fails an answer
    depending on which requirement it read first is a model whose verdict is an artefact
    of phrasing. `bench judge check` runs every labelled answer both ways and reports how
    often the verdict flips; a judge that flips is reported as untrustworthy for the same
    reason a low κ is.
    """
    if swap:
        rubric = _swap_halves(rubric)
    return (
        f"RUBRIC\n{rubric.strip()}\n\n"
        f"QUESTION\n{prompt.strip()}\n\n"
        f"ANSWER\n{(output or '').strip() or '(empty)'}"
    )


def _swap_halves(rubric: str) -> str:
    """Reverse the rubric's sentences, coarsely and deterministically.

    Sentence-level rather than a real parse: the swap is a probe, and a probe has to be
    reproducible from the file alone. Three sentences or fewer are reversed as a block,
    which is enough to change the reading order without changing the meaning.
    """
    # Trailing full stops come off before the rejoin and one goes back on at the end,
    # otherwise the sentence that was last keeps its stop and the join adds another:
    # "One sentence.. Does not blame the prompt." — which is a different string from the
    # rubric it claims to be the same as, and the `swap=True` probe compares strings.
    parts = [
        part.strip().rstrip(".") for part in rubric.replace("\n", " ").split(". ") if part.strip()
    ]
    if len(parts) < 2:
        # One sentence has no order to change. Returning it unchanged is the honest
        # answer: a probe that rewrote a single-sentence rubric would be measuring the
        # rewrite rather than the reading order.
        return rubric
    swapped = ". ".join(reversed(parts))
    return swapped + "." if rubric.strip().endswith(".") else swapped


def scaled_prompt(scale: int | None) -> str:
    """The system message, for a boolean judge or a graded one.

    A graded rubric asked for as a boolean is the failure this exists to prevent: the
    judge reads "3 of 5" and answers `pass` according to which sentence it weighted most,
    and two runs of the same answer disagree. Asking for the number and thresholding it
    in the scorer keeps that decision in the task file, where `pass_at` is visible.
    """
    if scale is None:
        return SYSTEM_PROMPT
    return (
        SYSTEM_PROMPT.replace(
            '{"pass": true, "reason": "one sentence"}',
            f'{{"score": 1..{scale}, "reason": "one sentence"}}',
        )
        + f"\n- `score` is an integer from 1 to {scale}. Score the rubric as written; do\n"
        "  not round to the ends of the scale to be safe.\n"
    )


def parse_verdict(text: str) -> tuple[bool, str, int | None] | None:
    """Find the first JSON object with a boolean `pass` or an integer `score` in it.

    Lenient on purpose: models wrap answers in prose and fences however plainly they are
    asked not to, and a benchmark that fails on a stray backtick is a benchmark nobody
    runs twice. Strict about the two things that matter: the verdict has to be a boolean
    or an integer, and a `true`/`false` string is not a boolean no matter how much it
    looks like one to a human reading the raw reply.
    """
    for candidate in _json_candidates(text):
        try:
            payload = json.loads(candidate)
        except ValueError:
            continue
        if not isinstance(payload, dict):
            continue
        reason = payload.get("reason")
        reason_text = str(reason) if isinstance(reason, str) else ""
        score = payload.get("score")
        graded = score if isinstance(score, int) and not isinstance(score, bool) else None
        if isinstance(payload.get("pass"), bool):
            return payload["pass"], reason_text, graded
        if graded is not None:
            # A graded reply carries no boolean; the scorer thresholds it. Reported here
            # as `passed = True` so that `scored` is true and the score reaches the
            # caller, which is where `pass_at` decides — never here, where the scale is
            # unknown.
            return True, reason_text, graded
    return None


def _json_candidates(text: str) -> list[str]:
    """Every balanced `{...}` in the text, in the order it starts.

    A stack rather than a depth counter, so an object nested inside an unbalanced one is
    still found. The first version only emitted objects that closed at depth zero, which
    turned a reply starting with a stray `{` into "no verdict" — a false outage, charged to
    the entrant as an unscored task. Found by the test that feeds it exactly that.
    """
    starts: list[int] = []
    found: list[tuple[int, str]] = []
    for index, character in enumerate(text):
        if character == "{":
            starts.append(index)
        elif character == "}" and starts:
            start = starts.pop()
            found.append((start, text[start : index + 1]))
    return [candidate for _, candidate in sorted(found)]


def _read_calibration(config: JudgeConfig) -> dict[str, Any] | None:
    from .calibration import calibration_path

    path = calibration_path(config)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("model") not in {None, config.model}:
        # A calibration for a different judge is not a calibration for this one. Saying so
        # by returning nothing is better than trusting a κ that describes another model.
        return None
    return payload


def _which(task_id: str) -> str:
    """`task gqa-0007`, or `an unnamed task`.

    Carried into the outage messages and nowhere else: a judge that times out on one task
    in three hundred produces a note somebody has to act on, and "the judge is unreachable"
    without the task id means opening the run file to find out which one.
    """
    return f"task {task_id}" if task_id else "an unnamed task"


def _quote(text: str, limit: int = 160) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1000))
