# Support incidents

Notes kept by hand after each incident, with the parts that mattered.

## 2026-08 refunds stopped

The v2 rollout stopped refunding orders after a permission change removed the billing
scope from the worker's role. The prompt was not involved, and rolling the permission
back fixed it within the hour.

## 2026-07 escalated too eagerly

The escalation rule matched the word "manager" anywhere in the message, so "my manager
asked me to check" was escalated as a complaint. Narrowed to explicit requests for a
human.

## 2026-05 order lookups timed out

The order lookup tool called the carrier API once per item, which for a ten-item order
took eleven seconds and hit the client's timeout. Batched, and the p95 fell to 900ms.
