"""The judged expectation: the one place a model is allowed to decide a score.

Everything else in this package is a rule. This is the exception, and it exists because
some of the things worth measuring are not rules: whether a summary is accurate, whether a
tone is right, whether a refusal explained itself. Those tasks are marked on the board and
their share is reported per family, so a reader can see how much of a score rests on a
model agreeing with itself.

What the check does is small on purpose. It hands the judge the task's rubric, the user's
question and the agent's answer, and turns the verdict into an `ExpectationResult`. The
interesting parts are the three conditions:

* **An untrusted judge does not score.** When no calibration is on file, or κ is below the
  floor, or too few labels were compared, the expectation is returned *unscored* and the
  task is marked `judge_trusted=False`, which keeps it out of the aggregate. The
  alternative — scoring it zero — would make a missing calibration look like a
  regression, and the first thing that happens to a benchmark that does that is that
  nobody trusts its numbers.
* **A judge that does not answer is not a fail.** An outage, a timeout, a reply with no
  verdict in it: all unscored, all carrying the reason, all surfaced in the run's notes.
  Same reasoning as above, one level down.
* **A rubric is a string, not a template.** The task writes the rubric; the check does not
  wrap it in "the answer must..." boilerplate. Inserting the answer into a sentence the
  task author never saw is how a benchmark's passing criterion drifts away from the task
  file that is supposed to define it.

The optional `scale` is the concession to reality that keeps this honest: some rubrics
are graded ("3 of 5: mentions the cause but not the date"), and a boolean judge reading a
rubric that says so gives inconsistent answers. `scale: 5` asks for
`{"score": 4, "reason": "..."}` and passes at or above `pass_at` (default: more than
half). The score is kept in the result's detail, so a task with a graded rubric still
contributes the graded value to the report instead of a coin flip.
"""

from __future__ import annotations

from typing import Any

from ..core.task import Expectation
from .base import ExpectationResult, ScoringContext


def check_judge(context: ScoringContext, expectation: Expectation) -> ExpectationResult:
    rubric = expectation.params.get("rubric")
    if not isinstance(rubric, str) or not rubric.strip():
        return ExpectationResult(
            "judge", False, "a judge expectation needs a `rubric:` string", scored=False
        )
    if context.judge is None:
        return ExpectationResult(
            "judge",
            False,
            "no judge is configured for this run",
            scored=False,
            detail={"rubric": rubric, "reason": "no judge"},
        )
    if not getattr(context.judge, "trusted", False):
        note = (
            context.judge.trust_note()
            if hasattr(context.judge, "trust_note")
            else "the judge is not calibrated"
        )
        return ExpectationResult(
            "judge", False, f"not scored: {note}", scored=False, detail={"rubric": rubric}
        )

    prompt = _prompt_of(context)
    scale = expectation.params.get("scale")
    verdict = _ask(context, rubric=rubric, prompt=prompt, scale=scale)
    failed_reason = getattr(verdict, "error", None)
    if failed_reason or getattr(verdict, "passed", None) is None:
        return ExpectationResult(
            "judge",
            False,
            f"not scored: {getattr(verdict, 'reason', failed_reason)}",
            scored=False,
            detail={"rubric": rubric, "error": failed_reason},
        )

    score = getattr(verdict, "score", None)
    passed = bool(verdict.passed)
    if isinstance(score, int) and isinstance(scale, int):
        pass_at = float(expectation.params.get("pass_at", (scale + 1) / 2))
        passed = float(score) >= pass_at

    detail: dict[str, Any] = {"rubric": rubric, "reason": verdict.reason}
    if score is not None:
        detail["score"] = score
    if getattr(verdict, "cached", False):
        detail["cached"] = True
    note = verdict.reason or ("passes the rubric" if passed else "fails the rubric")
    return ExpectationResult("judge", passed, note, detail=detail)


def _ask(context: ScoringContext, *, rubric: str, prompt: str, scale: Any) -> Any:
    """Call the judge. `scale=None` is a boolean verdict, and it is always passed.

    One call site rather than two branches: the protocol says `scale` defaults to `None`,
    so a judge that does not grade simply ignores it, and a scorer that special-cased the
    boolean path would be a scorer with two ways to be wrong.
    """
    judge = context.judge
    assert judge is not None  # the caller checked; this is for the type checker
    return judge.verdict(
        rubric=rubric,
        prompt=prompt,
        output=context.answer,
        task_id=context.task.id,
        scale=scale,
    )


def _prompt_of(context: ScoringContext) -> str:
    """The question the agent was asked, from the task's input.

    `question` first because that is what every task in the suite calls it, then `input`
    for the ones that carry a single prose field, then the whole input as JSON. The judge
    seeing the question matters: a rubric-only comparison cannot tell "the answer is
    wrong" from "the answer is right about a different question".
    """
    import json

    for key in ("question", "input", "prompt"):
        value = context.task.input.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return json.dumps(context.task.input, sort_keys=True, default=str)


CHECKS = {
    "judge": check_judge,
}
