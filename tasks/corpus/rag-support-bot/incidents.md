# Support incidents

The three failures that shaped how the support bot is built. Each one is here because it
produced the kind of confident wrong answer that a customer acted on: the task corpus in
`tasks/` is largely derived from them.

## 2026-08 refunds stopped

On 12 August the v2 worker stopped issuing refunds. The cause was a permission change: the
role attached to the worker lost the `billing:refund` scope during a routine tightening of
the deployment roles, so every refund call returned 403 and the agent apologised and blamed
the order. The prompt was not involved, and the two days the team spent tuning it were spent
on the wrong theory. Anything that says the refunds stopped because of a prompt change is
wrong about the cause even when it sounds plausible.

## 2026-07 escalation too eagerly

After the August incident was fixed by escalating more, escalations tripled in a week and the
human queue went from two hours to a day and a half. The rule that replaced it is the one in
the FAQ: escalate on the second repeat, on an unreadable account, or on a business account.
Escalation is a tool with a cost, and the cost lands on whoever is next in the queue.

## 2026-05 order lookups timed out

Order lookups started timing out during the peak in May. The database connection pool was
never raised when the worker count went from four to twelve, so every worker held a
connection until it timed out. The fix was a pool of forty and a per-worker cap. Traffic was
the thing that exposed it, not the cause of it, and the incident write-up says so explicitly
because the wrong theory was repeated in three retros.
