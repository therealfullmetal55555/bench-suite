# Executive assistant workflows

The nightly and morning automations that read a calendar, a mailbox and a task list, and
produce one briefing. Small workflow, sharp edges: everything here exists because a version
without it sent something embarrassing.

## The morning briefing

The briefing runs at 06:30 local time and contains: today's meetings with their preparation
notes, the mail that arrived since the last briefing and needs a reply, and the tasks due
today or overdue. It is sent as one message, not three, because three messages at 06:30 is
noise and one is a briefing. On a day with nothing in it the briefing is not sent at all —
an empty digest teaches the reader to stop opening them.

## Deduplication

Every item in the briefing carries a stable id: a calendar event's id, a mail's Message-ID, a
task's id. An item that appeared in yesterday's briefing and is unchanged is dropped. The
alternative is a briefing that repeats the same unread invoice for eleven mornings, which is
how the whole workflow gets muted. "Changed" means the item's fields differ from the copy
stored yesterday, not that its timestamp moved.

## Conflicts

Two meetings that overlap are reported together, with the shorter one named as the one to
move — the shorter one is cheaper to move and usually the one the assistant booked. The
briefing never proposes a specific new slot: it names the conflict and the two events, and
the human picks, because the calendar's free/busy view has never known what a person actually
intends.

## Approvals

Anything that sends mail on the user's behalf, changes a calendar entry, or moves money is
drafted and queued for approval rather than executed. Approval is a single-tap response over
the same channel the draft was sent to, and a draft expires after 24 hours so that a
half-read queue does not fire off a week-old reply.
