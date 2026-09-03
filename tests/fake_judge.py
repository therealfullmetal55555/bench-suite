"""A judge that answers from a dictionary.

The scorers depend on the `Judge` protocol and nothing else, so the whole judged path —
weights, unscored expectations, the `judge_trusted` flag — is testable without a socket.
That is the point of the protocol, and a test that reached for HTTP to check a weighted
sum would be testing httpx.

`RecordingJudge` is deliberately dumber than a real judge: it answers per answer-substring
or per rubric-substring, it can be told it is untrusted, and it records its calls so a
"the judge was asked about the right question" assertion has something to read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeVerdict:
    """The shape `check_judge` reads.

    Not `JudgeVerdict` on purpose: the scorer must not depend on the concrete client's
    class, and a test that constructed one would hide the day it started to.
    """

    passed: bool | None = True
    reason: str = "looks right"
    error: str | None = None
    score: int | None = None
    cached: bool = False

    @property
    def scored(self) -> bool:
        return self.passed is not None and self.error is None


@dataclass
class RecordingJudge:
    answer_verdicts: dict[str, bool] = field(default_factory=dict)
    """Answer substring → verdict. Checked first, longest match wins, because most tests
    care about what the agent said rather than about the rubric."""

    verdicts: dict[str, bool] = field(default_factory=dict)
    """Rubric substring → verdict."""

    trusted: bool = True
    default: bool | None = True
    """`None` makes the judge fail to answer, which is how the "an outage is not a fail"
    path is exercised."""

    score: int | None = None
    note: str = ""
    trust_note_text: str = "κ 0.81 ≥ floor 0.70: judge-scored tasks count"
    calls: list[dict[str, Any]] = field(default_factory=list)

    def verdict(
        self,
        *,
        rubric: str,
        prompt: str,
        output: str,
        task_id: str = "",
        scale: int | None = None,
    ) -> FakeVerdict:
        self.calls.append(
            {
                "rubric": rubric,
                "prompt": prompt,
                "output": output,
                "task_id": task_id,
                "scale": scale,
            }
        )
        for needle, verdict in sorted(self.answer_verdicts.items(), key=lambda kv: -len(kv[0])):
            if needle.lower() in output.lower():
                return FakeVerdict(
                    passed=verdict, reason=f"the answer contains {needle!r}", score=self.score
                )
        for needle, verdict in sorted(self.verdicts.items(), key=lambda kv: -len(kv[0])):
            if needle.lower() in rubric.lower():
                return FakeVerdict(
                    passed=verdict, reason=f"the rubric mentions {needle!r}", score=self.score
                )
        if self.default is None:
            return FakeVerdict(
                passed=None,
                reason=self.note or "the judge is unreachable",
                error=self.note or "the judge is unreachable",
            )
        return FakeVerdict(
            passed=self.default, reason=self.note or "no strong opinion", score=self.score
        )

    def trust_note(self) -> str:
        return self.trust_note_text
