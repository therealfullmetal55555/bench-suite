"""Six long plans, twelve tasks: the family where a step taken early is only paid for later.

The difference between this family and `tool-trajectory` is not the number of steps, it is
that the *content* of a later step depends on a result the agent got from an earlier one.
That is what "long horizon" means when it is a real thing rather than a bigger step budget,
and it is why the steps here carry their observations: step four's argument is a value that
only exists because step two returned it.

So a task in this family is one question with a sequence of tool calls behind it, and the
private half of each pair changes the world so that the plan must branch — a check that
fails, a verification that disagrees — because a held-back split that only renames things
tests nothing. The judged expectation is at the end of each: the artifact is the thing being
graded ("write the postmortem"), and the trajectory checks are what keep it honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Plan:
    key: str
    title: str
    tools: list[dict[str, str]]
    steps: list[dict[str, Any]]
    """Each step: `{"tool", "arguments", "result"}`. `result` is prose the agent reads, and
    later arguments quote values out of it — the chain is the task."""

    because: str
    tags: list[str] = field(default_factory=list)

    public: dict[str, Any] = field(default_factory=dict)
    private: dict[str, Any] = field(default_factory=dict)

    def tool_names(self) -> list[str]:
        return [tool["name"] for tool in self.tools]


def plans() -> list[Plan]:
    return [
        Plan(
            key="long-postmortem",
            title="Write the postmortem for an incident, after reading it",
            tools=[
                {"name": "list_incidents", "description": "list incidents in a time range"},
                {"name": "read_incident", "description": "read one incident's timeline"},
                {"name": "query_logs", "description": "search logs by service and pattern"},
                {"name": "read_deploys", "description": "list deploys with their timestamps"},
                {"name": "write_document", "description": "write the artifact and stop"},
            ],
            tags=["incidents", "long-horizon"],
            because=(
                "The August postmortem is the document this portfolio keeps re-learning from: "
                "the timeline said prompt, the permission change said otherwise, and the "
                "postmortem that only read the timeline was wrong for two days."
            ),
            steps=[
                {
                    "tool": "list_incidents",
                    "arguments": {"from": "2026-08-01"},
                    "result": "INC-441 refunds failing, 2026-08-12, service support-worker",
                },
                {
                    "tool": "read_incident",
                    "arguments": {"id": "INC-441"},
                    "result": "timeline: 09:14 deploy 4471, 09:31 refund errors, 11:02 prompt "
                    "retuned,"
                    "14:40 rollback",
                },
                {
                    "tool": "query_logs",
                    "arguments": {"service": "support-worker", "pattern": "refund"},
                    "result": "403 billing:refund scope missing, first at 09:31, 1 842 occurrences",
                },
                {
                    "tool": "read_deploys",
                    "arguments": {"service": "support-worker"},
                    "result": "deploy 4471 changed the worker role bindings at 09:14",
                },
                {
                    "tool": "write_document",
                    "arguments": {"kind": "postmortem"},
                    "result": "written",
                },
            ],
            public={
                "question": "Write the postmortem for INC-441.",
                "required": ["read_incident", "query_logs", "write_document"],
                "ordered": ["read_incident", "query_logs"],
                "max_steps": 8,
                "must_include": ["permission"],
                "rubric": "Names the missing billing:refund scope from deploy 4471 as the cause. "
                "Does"
                "not name the prompt change as the cause.",
            },
            private={
                # The same incident for a different reader, which is the honest way to hold
                # a plan back: the customer-facing note has to explain a two-day detour
                # without naming an internal theory as a cause, and the trajectory is longer
                # because the deploy history is what makes that possible.
                "question": "Customers were told refunds were being retuned. Write the note that "
                "goes on"
                "the status page for INC-441, and address the two days in which we said that.",
                "required": ["read_incident", "query_logs", "read_deploys", "write_document"],
                "ordered": ["read_incident", "read_deploys", "write_document"],
                "max_steps": 9,
                "must_include": ["deploy 4471"],
                "rubric": "Attributes the outage to the missing billing:refund scope from deploy "
                "4471,"
                "and says the prompt retune was a correction rather than the cause.",
            },
        ),
        Plan(
            key="long-tenant-migration",
            title="Move a tenant to a new plan, verifying at both ends",
            tools=[
                {"name": "get_subscription", "description": "read plan, quota and status"},
                {"name": "list_memberships", "description": "list accepted members"},
                {"name": "change_plan", "description": "change the plan; bills the customer"},
                {"name": "get_usage", "description": "read metered usage for the period"},
                {"name": "write_document", "description": "write the change record"},
            ],
            tags=["tenancy", "billing", "long-horizon"],
            because=(
                "A plan change that skips the arithmetic charges for seats nobody has, and a "
                "change that skips the verification cannot prove it did not. Both checks are "
                "separate tool calls, three steps apart."
            ),
            steps=[
                {
                    "tool": "get_subscription",
                    "arguments": {"org": "12"},
                    "result": "plan growth, quota 500k events, status active",
                },
                {
                    "tool": "list_memberships",
                    "arguments": {"org": "12"},
                    "result": "9 accepted memberships, 2 pending invitations",
                },
                {
                    "tool": "get_usage",
                    "arguments": {"org": "12"},
                    "result": "usage 71% of quota for the period, climbing 6% a week",
                },
                {
                    "tool": "change_plan",
                    "arguments": {"org": "12", "plan": "scale"},
                    "result": "changed; effective immediately",
                },
                {
                    "tool": "write_document",
                    "arguments": {"kind": "change-record"},
                    "result": "written",
                },
            ],
            public={
                "question": "Organisation 12 is at 71% of its quota and climbing. Move them to the "
                "scale"
                "plan and record it.",
                "required": [
                    "get_subscription",
                    "list_memberships",
                    "get_usage",
                    "change_plan",
                    "write_document",
                ],
                "ordered": ["get_subscription", "change_plan"],
                "max_steps": 9,
                "must_include": ["scale"],
                "rubric": "States the seat count and the usage figure that justified the change, "
                "and the"
                "date it takes effect.",
            },
            private={
                "question": "Move organisation 12 to the scale plan and record it. Their last "
                "invoice is"
                "unpaid.",
                "required": ["get_subscription", "write_document"],
                "ordered": ["get_subscription"],
                "forbidden": ["change_plan"],
                "max_steps": 9,
                "must_include": ["payment"],
                "rubric": "Refuses to change the plan while the invoice is unpaid, and names the "
                "dunning"
                "step that has to happen first.",
            },
        ),
        Plan(
            key="long-cost-investigation",
            title="Find why the bill tripled, and say what to change",
            tools=[
                {"name": "query_metrics", "description": "read a metric series over a range"},
                {"name": "query_logs", "description": "search logs by service and pattern"},
                {"name": "read_config", "description": "read a deployed configuration"},
                {
                    "name": "read_pricing",
                    "description": "read the provider's per-million-token rates",
                },
                {"name": "write_document", "description": "write the finding"},
            ],
            tags=["observability", "cost", "long-horizon"],
            because=(
                "The cost of a run is tokens times rate, and an investigation that reads only "
                "the tokens concludes 'traffic' half the time. Reading the config and the "
                "pricing is what separates the two, and neither is the first thing an agent "
                "reaches for."
            ),
            steps=[
                {
                    "tool": "query_metrics",
                    "arguments": {"metric": "tokens"},
                    "result": "tokens flat week over week, 40M/day",
                },
                {
                    "tool": "query_metrics",
                    "arguments": {"metric": "cost"},
                    "result": "cost up 3.1x since Tuesday",
                },
                {
                    "tool": "query_logs",
                    "arguments": {"service": "support-worker", "pattern": "cache"},
                    "result": "cache hit ratio fell from 0.82 to 0.04 on Tuesday",
                },
                {
                    "tool": "read_config",
                    "arguments": {"service": "support-worker"},
                    "result": "prompt prefix changed: a timestamp was added at the top of the "
                    "system"
                    "message",
                },
                {
                    "tool": "read_pricing",
                    "arguments": {"provider": "primary"},
                    "result": "cached prompt tokens 0.10/M, uncached 0.40/M",
                },
                {"tool": "write_document", "arguments": {"kind": "finding"}, "result": "written"},
            ],
            public={
                "question": "The support worker's bill tripled on Tuesday and the token count did "
                "not"
                "move. Find out why.",
                "required": ["query_metrics", "read_config", "read_pricing", "write_document"],
                "ordered": ["query_metrics", "read_config"],
                "max_steps": 10,
                "must_include": ["cache"],
                "rubric": "Identifies the timestamp at the top of the system message as the cause "
                "of the"
                "cache miss, and cites the two rates.",
            },
            private={
                "question": "The support worker's bill tripled. Traffic is up 8%. Find out why and "
                "say"
                "what to change.",
                "required": ["query_metrics", "query_logs", "read_config", "write_document"],
                "ordered": ["query_metrics", "read_config"],
                "max_steps": 10,
                "must_include": ["prefix"],
                "rubric": "Rejects traffic as the cause with the numbers, names the prompt prefix, "
                "and"
                "says what to change rather than to raise the budget.",
            },
        ),
        Plan(
            key="long-dunning-recovery",
            title="Recover a failed payment without losing the customer",
            tools=[
                {"name": "list_invoices", "description": "list invoices and their status"},
                {"name": "get_subscription", "description": "read plan, quota and status"},
                {"name": "list_memberships", "description": "list accepted members"},
                {"name": "retry_payment", "description": "retry the last failed payment"},
                {"name": "write_document", "description": "write the customer-facing note"},
            ],
            tags=["billing", "long-horizon"],
            because=(
                "The retry schedule is day 1, 3 and 7, and a retry outside it is a charge the "
                "provider will refuse. The workspace stays read-only through all of it, which "
                "is a decision the customer-facing note has to explain."
            ),
            steps=[
                {
                    "tool": "list_invoices",
                    "arguments": {"org": "12"},
                    "result": "INV-8801 failed 2026-09-02, three attempts, past_due",
                },
                {
                    "tool": "get_subscription",
                    "arguments": {"org": "12"},
                    "result": "status past_due, workspace read-only since 2026-09-09",
                },
                {
                    "tool": "list_memberships",
                    "arguments": {"org": "12"},
                    "result": "9 accepted memberships, nobody locked out of reading",
                },
                {
                    "tool": "retry_payment",
                    "arguments": {"invoice": "INV-8801"},
                    "result": "card declined again, code 51 insufficient funds",
                },
                {
                    "tool": "write_document",
                    "arguments": {"kind": "customer-note"},
                    "result": "written",
                },
            ],
            public={
                "question": "Organisation 12's payment failed three times. Write the note that "
                "goes to"
                "their admin.",
                "required": ["list_invoices", "get_subscription", "write_document"],
                "ordered": ["list_invoices", "get_subscription"],
                "max_steps": 8,
                "must_include": ["read-only"],
                "rubric": "Says the workspace stays readable, names the retry days, and does not "
                "promise"
                "a specific date for access to be restored.",
            },
            private={
                "question": "Organisation 12's card was declined with code 51. Their admin is "
                "asking"
                "whether they will lose access to their data.",
                "required": ["list_invoices", "write_document"],
                "ordered": ["list_invoices"],
                "max_steps": 8,
                "must_include": ["402"],
                "rubric": "States that reads keep working and writes return 402, and does not "
                "threaten"
                "deletion or a deadline that is not in the corpus.",
            },
        ),
        Plan(
            key="long-briefing-cache",
            title="Rebuild a cache that has been serving the wrong briefings",
            tools=[
                {"name": "read_config", "description": "read the workflow's filters"},
                {"name": "read_cache", "description": "read a cache entry's key and age"},
                {"name": "invalidate_cache", "description": "delete a cache entry"},
                {"name": "read_mail", "description": "read the mailbox with a filter"},
                {"name": "write_document", "description": "write the incident note"},
            ],
            tags=["automation", "incidents", "long-horizon"],
            because=(
                "The cache key was the account and not the filter, and six days of wrong "
                "briefings followed. The fix is the key, so the trajectory has to read the "
                "config before it invalidates anything — otherwise the next run renews the "
                "same wrong entry."
            ),
            steps=[
                {
                    "tool": "read_cache",
                    "arguments": {"account": "ada"},
                    "result": "entry key: account=ada, written 2026-04-02, hits 41",
                },
                {
                    "tool": "read_config",
                    "arguments": {"workflow": "briefing"},
                    "result": "filter set: from:(team) newer_than:1d | label:inbox -label:filed",
                },
                {
                    "tool": "read_mail",
                    "arguments": {"filter": "from:(team) newer_than:1d"},
                    "result": "record count 0; the filter changed on 2026-04-02 to exclude filed "
                    "mail",
                },
                {
                    "tool": "invalidate_cache",
                    "arguments": {"account": "ada"},
                    "result": "entry deleted",
                },
                {
                    "tool": "write_document",
                    "arguments": {"kind": "incident-note"},
                    "result": "written",
                },
            ],
            public={
                "question": "The morning briefing has been listing filed mail for a week. Fix it "
                "and write"
                "the note.",
                "required": ["read_config", "invalidate_cache", "write_document"],
                "ordered": ["read_config", "invalidate_cache"],
                "max_steps": 9,
                "must_include": ["key"],
                "rubric": "Says the cache key was missing the filter set, and that the fix is the "
                "key"
                "rather than a rebuild or a shorter TTL.",
            },
            private={
                "question": "The briefing listed the same unread invoice for six mornings and "
                "missed new"
                "mail. Why, and what is the change?",
                "required": ["read_config", "read_cache", "write_document"],
                "ordered": ["read_config", "read_cache"],
                "max_steps": 9,
                "must_include": ["filter"],
                "rubric": "Connects the missing filter hash to the six days, and states that "
                "'changed'"
                "means the fields, not the timestamp.",
            },
        ),
        Plan(
            key="long-leaderboard-audit",
            title="Audit a benchmark result before publishing it",
            tools=[
                {"name": "list_runs", "description": "list run files with their dates"},
                {"name": "read_run", "description": "read a run's per-task scores"},
                {"name": "compare_runs", "description": "compare two runs task by task"},
                {"name": "rerun_task", "description": "re-run one task and write a fresh run"},
                {"name": "write_document", "description": "write the audit note"},
            ],
            tags=["benchmark", "long-horizon"],
            because=(
                "Reading the rules: a submission that improved is a submission somebody has to "
                "check, and the check is a re-run of the task that moved. This is the family's "
                "one task about this benchmark itself, and it is the one an entrant will read "
                "most carefully."
            ),
            steps=[
                {
                    "tool": "list_runs",
                    "arguments": {"target": "agent-v3"},
                    "result": "20260928T101500Z (public, 150 tasks), 20260929T090000Z (public, 150 "
                    "tasks)",
                },
                {
                    "tool": "read_run",
                    "arguments": {"run": "20260929T090000Z"},
                    "result": "aggregate 0.71, up from 0.58; 14 tasks moved, all in grounded-qa",
                },
                {
                    "tool": "compare_runs",
                    "arguments": {"left": "20260928T101500Z", "right": "20260929T090000Z"},
                    "result": "14 tasks moved up, 0 moved down, all 14 share the tag retrieval",
                },
                {
                    "tool": "rerun_task",
                    "arguments": {"task": "gqa-0042", "repeat": 3},
                    "result": "3 repeats: 2 passes, 1 fail; the task is flaky at this build",
                },
                {
                    "tool": "write_document",
                    "arguments": {"kind": "audit-note"},
                    "result": "written",
                },
            ],
            public={
                "question": "Audit the v3 submission before it goes on the board.",
                "required": ["list_runs", "read_run", "compare_runs", "write_document"],
                "ordered": ["list_runs", "compare_runs"],
                "max_steps": 9,
                "must_include": ["14"],
                "rubric": "Names the tasks that moved, the direction, and what the audit still "
                "needs"
                "before publication.",
            },
            private={
                "question": "The v3 submission gained 13 points and every task that moved is a "
                "retrieval"
                "task. Publish or not?",
                "required": ["compare_runs", "rerun_task", "write_document"],
                "ordered": ["compare_runs", "rerun_task"],
                "max_steps": 9,
                "must_include": ["flaky"],
                "rubric": "Holds the submission back, states that the re-run showed flakiness, and "
                "says"
                "what would make it publishable.",
            },
        ),
    ]
