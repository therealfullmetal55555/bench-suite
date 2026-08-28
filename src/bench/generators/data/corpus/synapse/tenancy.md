# Tenancy

How the multi-tenant workbench keeps one customer's rows away from another's. The rules are
short and the implementation is boring on purpose: a tenancy bug is not a bug, it is an
incident with a lawyer attached.

## Row level security

Every tenant-owned table has row level security enabled *and* forced. Forced matters: without
it the table owner bypasses its own policies, which means the one connection that runs
migrations is also the one connection that can read every tenant. The policy is
`tenant_id = current_setting('app.tenant')::uuid`, and the setting is applied with
`SET LOCAL` inside the transaction so it cannot leak into the next request on a pooled
connection.

## Missing rows are 404

An object belonging to another tenant returns 404, never 403. A 403 confirms that the object
exists, which is a fact about somebody else's account, and an attacker who can enumerate ids
learns which ones are real. The API has exactly one answer for "not yours" and "not there".

## Seats are counted, never stored

A seat is a query, not a column. Storing a seat count means two places can disagree about how
many people are in an organisation, and the place that is wrong is always the one that bills.
The count is `select count(*) from memberships where org_id = … and accepted_at is not null`,
evaluated when the invoice is drafted and never cached.

## Invitations

An invitation is a row with an unguessable token, an expiry of seven days, and a role. It is
accepted once: the accept path sets `accepted_at` in a conditional update and returns 404 if
the row is already accepted, so a forwarded email cannot add a second member. Roles are
`owner`, `admin`, `member`, and `billing`, and only `owner` and `admin` may invite.
