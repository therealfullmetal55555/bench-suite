"""Tool use: which tools, in what order, how many steps, with what arguments.

The family that needs no judge at all, which is why it is the first one written. Every
claim here is checkable by reading the trace: the tools are named, the calls are ordered,
the arguments are the arguments. A scoring rule a reviewer cannot argue with is worth
more than one that agrees with them.

Four things an author can ask for, and the reasoning behind each default:

* `required_tools` — must appear, at least once each. Order among them is not checked
  unless asked for: an agent that looks up the order and then the policy has done the
  same work as one that took them the other way round, and failing it for that is how a
  benchmark starts measuring its author's taste.
* `forbidden_tools` — must not appear at all. This is where the expensive mistakes live
  (`issue_refund` before checking the window), so it is a hard failure with no partial
  credit.
* `ordered` — a list that must appear in this relative order, for the tasks where order
  *is* the thing being tested: retrieve, then answer; never the other way round.
* `max_steps` — the trace's own `steps` when it reports one, otherwise the number of tool
  calls. A target that reports neither is not penalised for the harness's silence: the
  check is skipped and says so, because scoring an unreported step count as zero would
  punish an honest integration.

`steps` as its own expectation exists because some tasks only care about the budget.
`max_steps` inside `trajectory` exists because most care about both, and splitting every
such task into two expectations doubles the task file for no reader's benefit.

`state` is here too, further down, because it reads the same thing the rest of the file
reads — the calls — and adds the part a tool name cannot carry: the arguments, and how
many times.
"""

from __future__ import annotations

from typing import Any

from ..core.task import Expectation
from .base import ExpectationResult, ScoringContext

#: Tools that end or spend something. Kept here as documentation rather than as a rule:
#: `forbidden_tools` is explicit per task, and this list exists so that the task linter
#: (next) can warn when a task in a support family has none of them forbidden.
IRREVERSIBLE = frozenset({"issue_refund", "cancel_order", "send_email", "delete_account"})


def check_trajectory(context: ScoringContext, expectation: Expectation) -> ExpectationResult:
    calls = context.trace.tool_names()
    required = [str(name) for name in expectation.params.get("required_tools", [])]
    forbidden = [str(name) for name in expectation.params.get("forbidden_tools", [])]
    ordered = [str(name) for name in expectation.params.get("ordered", [])]
    max_steps = expectation.params.get("max_steps")

    if not calls and (required or ordered):
        return ExpectationResult(
            "trajectory",
            False,
            "the trace made no tool calls at all, and the task requires "
            f"{', '.join(required or ordered)}",
        )

    problems: list[str] = []
    missing = [name for name in required if name not in calls]
    if missing:
        problems.append(f"missing required tool(s) {', '.join(missing)}")
    used_forbidden = [name for name in forbidden if name in calls]
    if used_forbidden:
        problems.append(f"used forbidden tool(s) {', '.join(used_forbidden)}")

    if ordered:
        position = 0
        for name in calls:
            if position < len(ordered) and name == ordered[position]:
                position += 1
        if position < len(ordered):
            problems.append(
                f"wrong order: expected {' → '.join(ordered)}, got {' → '.join(calls) or 'nothing'}"
            )

    if isinstance(max_steps, int):
        steps = _steps(context)
        if steps is None:
            # No step count and no way to infer one: reported, not failed. See the
            # module docstring — the harness does not punish silence.
            unchecked = "step budget not checked (the target reports no step count)"
            note = (
                f"{'; '.join(problems)}; {unchecked}"
                if problems
                else f"tools are right; {unchecked}"
            )
            return ExpectationResult("trajectory", not problems, note, detail={"steps": None})
        if steps > max_steps:
            problems.append(f"{steps} steps over the budget of {max_steps}")

    steps = _steps(context)
    if problems:
        return ExpectationResult(
            "trajectory",
            False,
            "; ".join(problems) + f" (called: {', '.join(calls) or 'nothing'})",
            detail={"tool_calls": calls, "steps": steps},
        )
    summary = ", ".join(calls) if calls else "no tool calls"
    return ExpectationResult(
        "trajectory", True, f"trajectory is right: {summary}", detail={"tool_calls": calls}
    )


def check_steps(context: ScoringContext, expectation: Expectation) -> ExpectationResult:
    """The step budget on its own, for tasks whose only requirement is thrift."""
    limit = expectation.params.get("max")
    if not isinstance(limit, int):
        return ExpectationResult(
            "steps", False, "a steps expectation needs `max: <int>`", scored=False
        )
    steps = _steps(context)
    if steps is None:
        return ExpectationResult(
            "steps",
            False,
            "the target reports no step count, and no tool calls to infer one from",
            scored=False,
        )
    if steps <= limit:
        return ExpectationResult("steps", True, f"{steps} steps, within the budget of {limit}")
    return ExpectationResult("steps", False, f"{steps} steps over the budget of {limit}")


def _steps(context: ScoringContext) -> int | None:
    """The trace's step count, or one inferred from its tool calls.

    Inferred rather than assumed absent, because every target that reports tool calls has
    told us something about steps — one call is one step — and refusing to count it would
    make the `steps` expectation unusable for the most common integrations. A trace with
    an explicit `steps: 0` and three tool calls is contradictory; the explicit number
    wins, and the contradiction is the target's business.
    """
    if context.trace.steps is not None:
        return context.trace.steps
    if context.trace.tool_calls:
        return len(context.trace.tool_calls)
    return None


# `CHECKS` is at the bottom of this file: `check_state` is added by the multi-turn
# section below, and a registry built before its check exists is a NameError at import.


# --- multi-turn state ----------------------------------------------------------
#
# `state` is here, next to the tool-call checks, because it reads the same thing they do:
# the calls, with their arguments. What it adds is what the argument-side of a trajectory
# check needs and a tool name cannot express — *this* refund, for *this* order, once.
#
# A multi-turn task's world is not changed by which tools ran; it is changed by what
# they carried. An agent that calls `issue_refund(order_id=1042)` on a task about order
# 1041 has done something worse than calling nothing, and a check that only counted tool
# names would score it as a pass. That failure — the right verb on the wrong object — is
# the one that costs money in production, so it gets its own expectation type rather than
# being folded into `trajectory` as a boolean nobody reads.


def check_state(context: ScoringContext, expectation: Expectation) -> ExpectationResult:
    """What the agent did to the world, as opposed to which tools it reached for.

    Two lists, and the second one matters as much as the first:

    * `applied: [{tool, args, times}]` — a transition that must have happened. `args` is a
      *subset* to match against, so a task pins the order id without having to predict the
      idempotency key the integration happens to send. `times` is exact when given, which
      is how "refund once" is expressed; without it, at-least-once.
    * `untouched: [{tool, args}]` — a transition that must not have happened. Written from
      the task's own point of view ("another customer's order"), because that is the
      mistake being guarded against and the task author should not have to enumerate the
      calls that are fine.
    """
    applied = expectation.params.get("applied", [])
    untouched = expectation.params.get("untouched", [])
    if not isinstance(applied, list) or not isinstance(untouched, list):
        return ExpectationResult(
            "state", False, "`applied` and `untouched` have to be lists", scored=False
        )
    if not applied and not untouched:
        return ExpectationResult(
            "state",
            False,
            "a state expectation needs `applied:` or `untouched:` — with neither it checks nothing",
            scored=False,
        )

    for spec in [*applied, *untouched]:
        if not isinstance(spec, dict) or "tool" not in spec:
            # A spec with no tool in it is a broken task, and an entrant must not lose a
            # point for it. Unscored rather than failed: the difference between "your
            # agent did the wrong thing" and "my task file is malformed" is the whole
            # reason `scored` exists on the result.
            return ExpectationResult(
                "state", False, f"the spec {spec!r} has no `tool` in it", scored=False
            )

    calls = context.trace.tool_calls
    problems: list[str] = []
    for spec in applied:
        matches = _matching(calls, spec)
        wanted = spec.get("times")
        if not matches:
            problems.append(f"never called {_spec_text(spec)}")
        elif isinstance(wanted, int) and len(matches) != wanted:
            problems.append(
                f"called {_spec_text(spec)} {len(matches)} time(s), and the task says {wanted}"
            )
    for spec in untouched:
        if _matching(calls, spec):
            problems.append(f"called {_spec_text(spec)}, which this task leaves alone")

    if problems:
        return ExpectationResult(
            "state",
            False,
            "; ".join(problems) + f" (called: {_calls_text(calls) or 'nothing'})",
            detail={"tool_calls": [call.name for call in calls]},
        )
    return ExpectationResult(
        "state",
        True,
        f"the world is where the task expects it: {_calls_text(calls) or 'no tool calls'}",
        detail={"tool_calls": [call.name for call in calls]},
    )


def _matching(calls: list[Any], spec: dict[str, Any]) -> list[Any]:
    """Calls of this tool whose arguments contain the spec's pairs."""
    wanted = spec.get("args") or {}
    if not isinstance(wanted, dict):
        return []
    return [
        call
        for call in calls
        if call.name == str(spec["tool"]) and _args_contain(call.arguments, wanted)
    ]


def _args_contain(arguments: dict[str, Any], wanted: dict[str, Any]) -> bool:
    """Subset match with the extraction family's comparison, imported rather than copied.

    `amount: 49.5` matching `"49.50"` is the same decision here as it is there, and two
    implementations of "these are the same value" is how a benchmark ends up disagreeing
    with itself about a rounding.
    """
    from .extraction import _equal

    return all(
        key in arguments and _equal(arguments[key], value, ("case", "strip", "number"))
        for key, value in wanted.items()
    )


def _spec_text(spec: dict[str, Any]) -> str:
    args = spec.get("args") or {}
    rendered = ", ".join(f"{key}={value!r}" for key, value in sorted(args.items()))
    return f"{spec.get('tool')}({rendered})"


def _calls_text(calls: list[Any]) -> str:
    return ", ".join(f"{call.name}({', '.join(sorted(call.arguments))})" for call in calls)


CHECKS = {
    "trajectory": check_trajectory,
    "steps": check_steps,
    "state": check_state,
}
