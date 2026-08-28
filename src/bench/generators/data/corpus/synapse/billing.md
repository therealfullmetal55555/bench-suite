# Billing

Subscriptions, metered usage, and the parts of the money path that are allowed to be wrong
and the parts that are not. The general rule: it is acceptable to under-charge by a rounding
error and unacceptable to double-charge, and every guard in this file exists to keep it that
way.

## Idempotency

Every write to the billing provider carries an idempotency key derived from the invoice id
and the period. A retry after a timeout reuses the key, so the provider returns the original
charge instead of making a second one. The key is stored with the invoice because the retry
that needs it happens in a different process, hours later, after the first one was killed.

## Metered usage

Usage is reported hourly as a delta, not as a running total, and the sum for a period must
reconcile with the provider's own count before the invoice is drafted. A mismatch blocks the
invoice and pages the on-call engineer. Dropping a usage batch is survivable; invoicing a
customer for a number the provider cannot reproduce is not.

## Over-quota

When an organisation is over its quota the API returns 402 with the current usage and the
limit in the body, and the workspace stays readable. Read-only over-quota is a decision, not
an oversight: locking a customer out of their own data because of a failed payment makes the
non-payment harder to fix, and the recovery path from a read-only workspace is one click.

## Dunning

A failed payment retries on day 1, 3, and 7. After the third failure the subscription is
marked `past_due` and the workspace stays read-only until payment succeeds or the customer
cancels. The trial is 14 days and does not require a card; a card taken at signup is charged
on day 15 whether or not the workspace was opened, which is why the signup form does not ask
for one.
