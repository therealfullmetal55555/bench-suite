# Assistant incidents

Two failures, both of them the kind that make a person stop trusting an automation: not
crashing, but being confidently wrong at 06:30 in a way that costs them a meeting.

## 2026-06 duplicate briefings

The workflow was rescheduled by an overlapping trigger and sent the briefing twice, ninety
seconds apart. The fix was a lock keyed on the date, taken before the fetch, so the second
run exits without reading anything. Cleaning up afterwards — deduplicating the items instead
of the run — would have hidden the second copy rather than prevented it, and the send path
is idempotent only because the lock is not.

## 2026-04 stale cache

The mailbox cache was keyed on the account id and never invalidated when a filter changed, so
for six days the briefing listed mail that had already been filed and missed the mail that
had not. The cache key now includes a hash of the filter set, and a cache entry older than
the last filter change is discarded rather than refreshed lazily — the lazy version is what
produced six days of wrong briefings and no error.
