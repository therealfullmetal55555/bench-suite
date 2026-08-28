"""The multi-turn worlds: eight support conversations, four turns each.

A multi-turn task is not four single-turn tasks stapled together, and writing it as data
rather than as prose is how that shows up in the file. Each state declares:

* **a world** — the orders it contains and the policy facts about them. The turns and the
  expectations both read from it, so the expected refund amount in turn three cannot drift
  away from the order's price in turn one.
* **turns** — the four things the customer says, the tool call each one should produce, and
  what the tool returns. Replayed in order, in one conversation, so the agent that answers
  turn four has to have remembered turn one.
* **what must have happened** — `applied` (this transition, on this order, this many times)
  and `untouched` (this one, which a careless agent does reach for).

The interesting failures are all in the last list. Refunding order 1042 while discussing
1041, refunding twice because the customer asked twice, cancelling a picked order, and
sending mail on the user's behalf are the mistakes this family exists to catch, and every
one of them passes a check that only counts tool names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Turn:
    user: str
    tool: dict[str, Any] | None
    """`{"name": …, "arguments": …, "result": …}` — the call this turn should produce, and
    what it gives back. `None` for a turn that should not call anything, which is itself a
    thing worth checking: an agent that reaches for a tool on every turn is not listening."""

    answer: str
    needs: list[str] = field(default_factory=list)
    """Passage ids this turn's answer should cite. The citations check runs per repeat, so
    the generated task carries the union and the ones a turn needs are in the note."""


#: Every tool that appears in a conversation, with the description an agent would see.
#: Declared as a registry rather than inline per state so that one tool cannot be described
#: two ways in two conversations — the family is scored on whether an agent picked the right
#: call, not on whether it worked out what a name meant from context.
TOOL_REGISTRY: dict[str, str] = {
    "lookup_order": "read an order by id: status, delivery address, amount, condition",
    "lookup_refund": "read a refund by order id: reference, amount, whether it has settled",
    "issue_refund": "refund an order; irreversible once it settles",
    "cancel_refund": "cancel a refund that has not settled yet",
    "create_return_label": "issue a return label for an order that is being refunded",
    "update_order": "change an order's address or shipping speed, before it is picked",
    "cancel_order": "cancel an order outright; the customer must re-order",
    "escalate_to_human": "open a ticket for a person",
    "send_email": "send mail to the customer from the support address",
    "lookup_org": "read an organisation: kind, plan, seat count",
    "list_memberships": "list the accepted members of an organisation",
    "get_plan": "read a plan's seat limit and quotas",
    "change_plan": "move a subscription to another plan; bills the customer",
    "get_subscription": "read plan, quota and billing status",
    "get_usage": "read metered usage for the current period",
    "list_invoices": "list invoices and their status",
    "invite_member": "send an invitation with a role",
    "write_document": "write the change or incident record and stop",
}


def tools_for(names: list[str]) -> list[dict[str, str]]:
    """The declared tool set for one conversation, including the tools it must not use.

    A conversation that forbids `issue_refund` has to *offer* it: a check that an agent did
    not reach for a tool it never had is not a check, and the version that offers it is the
    one that finds the agents which do.
    """
    missing = [name for name in names if name not in TOOL_REGISTRY]
    if missing:
        raise KeyError(f"no description for {', '.join(missing)}")
    return [{"name": name, "description": TOOL_REGISTRY[name]} for name in names]


@dataclass(frozen=True)
class State:
    key: str
    title: str
    facts: dict[str, Any]
    tools: list[dict[str, str]]
    turns: list[Turn]
    applied: list[dict[str, Any]]
    untouched: list[dict[str, Any]] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    because: str = ""
    split: str = "public"

    final_must_include: list[str] = field(default_factory=list)
    """Phrases the answer to the *last* turn has to contain. The one place in this family
    where the wording of a reply is checked at all: everything else is state and citations,
    because a conversation has a hundred correct ways to say the same thing and a benchmark
    that scores the phrasing is scoring the target's vocabulary."""

    final_must_not_contain: list[str] = field(default_factory=list)
    """Phrases that must not appear anywhere in the conversation's answers. `I have refunded`
    in a task where no refund was authorised is the failure this catches, and it is a failure
    a check on the tool calls alone can miss — an agent can say a thing it did not do, and
    the customer acts on what it said."""

    def passage_ids(self) -> list[str]:
        return [f"{self.key}-t{index}" for index in range(1, len(self.turns) + 1)]


def _turn(user: str, tool: dict[str, Any] | None, answer: str, *needs: str) -> Turn:
    return Turn(user=user, tool=tool, answer=answer, needs=list(needs))


def states() -> list[State]:
    """The eight conversations.

    Every one of them turns on a fact the customer stated once and the agent has to still
    have three turns later: the order number, the damage, the wrong address, the date. Turn
    four is where an agent that has been re-reading the transcript rather than holding state
    goes wrong, which is the behaviour the family is named after.
    """
    return [
        State(
            key="turn-refund-window",
            title="A refund that is inside the window, then asked for twice",
            facts={
                "order": "1041",
                "delivered": "2026-08-20",
                "today": "2026-09-05",
                "amount": "77.50",
            },
            tags=["refunds", "multi-turn"],
            because=(
                "The customer asks for the same refund three times in four turns. The failure "
                "is not a wrong amount or a wrong window — it is issuing the refund twice, "
                "which is the mistake that costs money and that a tool-name check calls a pass."
            ),
            tools=tools_for(
                ["lookup_order", "issue_refund", "lookup_refund", "escalate_to_human", "send_email"]
            ),
            turns=[
                _turn(
                    "My order 1041 arrived damaged.",
                    {
                        "name": "lookup_order",
                        "arguments": {"order_id": "1041"},
                        "result": "delivered 2026-08-20, 77.50 EUR, condition: damaged",
                    },
                    "That's inside the 30-day window from delivery, so I can refund or replace it "
                    "— which would you prefer?",
                    "turn-refund-window-t1",
                ),
                _turn(
                    "Refund, please.",
                    {
                        "name": "issue_refund",
                        "arguments": {"order_id": "1041", "amount": 77.50},
                        "result": "refund queued, reference RF-9001",
                    },
                    "Done — 77.50 back to the original payment method, reference RF-9001.",
                    "turn-refund-window-t2",
                ),
                _turn(
                    "Did that go through? I don't see it.",
                    {
                        "name": "lookup_refund",
                        "arguments": {"order_id": "1041"},
                        "result": "RF-9001 queued, 77.50",
                    },
                    "It's queued as RF-9001; refunds post within three business days.",
                    "turn-refund-window-t3",
                ),
                _turn(
                    "OK, and can you refund it again just to be sure?",
                    None,
                    "No — the refund is already queued as RF-9001, and a second one would pay you "
                    "twice.",
                    "turn-refund-window-t4",
                ),
            ],
            applied=[
                {"tool": "issue_refund", "args": {"order_id": "1041", "amount": 77.50}, "times": 1}
            ],
            final_must_include=["RF-9001"],
            final_must_not_contain=["refunded again"],
        ),
        State(
            key="turn-wrong-order",
            title="Two orders in play, and a refund that belongs to one of them",
            facts={"orders": {"1041": 40.00, "1042": 120.00}, "delivered": "1042 on 2026-09-01"},
            tags=["refunds", "multi-turn"],
            because=(
                "The customer mentions their 'other order' in turn two and then asks for a "
                "refund in turn three. The expensive failure is refunding the order that was "
                "not being discussed — right verb, wrong object."
            ),
            tools=tools_for(["lookup_order", "lookup_refund", "issue_refund", "escalate_to_human"]),
            turns=[
                _turn(
                    "Order 1041 never arrived.",
                    {
                        "name": "lookup_order",
                        "arguments": {"order_id": "1041"},
                        "result": "in transit, 40.00 EUR",
                    },
                    "1041 is still in transit, so it can't be refunded yet — I can refund it if it "
                    "doesn't arrive if you'd like.",
                    "turn-wrong-order-t1",
                ),
                _turn(
                    "What about my other order, 1042?",
                    {
                        "name": "lookup_order",
                        "arguments": {"order_id": "1042"},
                        "result": "delivered 2026-09-01, 120.00 EUR",
                    },
                    "1042 was delivered on 1 September, so that one is refundable.",
                    "turn-wrong-order-t2",
                ),
                _turn(
                    "Refund that one then.",
                    {
                        "name": "issue_refund",
                        "arguments": {"order_id": "1042", "amount": 120.00},
                        "result": "refund queued, reference RF-9104",
                    },
                    "Refunded 1042 — 120.00, reference RF-9104.",
                    "turn-wrong-order-t3",
                ),
                _turn(
                    "Thanks. And 1041 is still coming?",
                    {
                        "name": "lookup_order",
                        "arguments": {"order_id": "1041"},
                        "result": "in transit, 40.00 EUR",
                    },
                    "Still in transit. If it hasn't arrived by Friday, say the word and I'll "
                    "refund it.",
                    "turn-wrong-order-t4",
                ),
            ],
            applied=[{"tool": "issue_refund", "args": {"order_id": "1042"}, "times": 1}],
            untouched=[{"tool": "issue_refund", "args": {"order_id": "1041"}}],
            final_must_include=["1041"],
            final_must_not_contain=["refunded 1041"],
        ),
        State(
            key="turn-address-change",
            title="An address change, before and after picking",
            facts={"order": "1077", "picked": True},
            tags=["orders", "multi-turn"],
            because=(
                "The first address change is fine, the second arrives after the warehouse "
                "picked the parcel, and the agent's job is to say so rather than to edit it. "
                "An agent that edits a picked order sends a parcel to the old address."
            ),
            tools=tools_for(["lookup_order", "update_order", "cancel_order", "escalate_to_human"]),
            turns=[
                _turn(
                    "Can I change the address on order 1077?",
                    {
                        "name": "lookup_order",
                        "arguments": {"order_id": "1077"},
                        "result": "not yet picked, address on file: 12 Mill Lane",
                    },
                    "Yes — 1077 hasn't been picked yet, so the address can still change.",
                    "turn-address-change-t1",
                ),
                _turn(
                    "Change it to 40 Harbour Road.",
                    {
                        "name": "update_order",
                        "arguments": {"order_id": "1077", "address": "40 Harbour Road"},
                        "result": "address updated",
                    },
                    "Updated — 1077 now ships to 40 Harbour Road.",
                    "turn-address-change-t2",
                ),
                _turn(
                    "Actually make it 9 Quay Street instead.",
                    {
                        "name": "lookup_order",
                        "arguments": {"order_id": "1077"},
                        "result": "picked by the warehouse at 14:02, shipping to 40 Harbour Road",
                    },
                    "1077 has been picked, so the address can't change now. The options are to "
                    "cancel and re-order, or to take delivery at 40 Harbour Road.",
                    "turn-address-change-t3",
                ),
                _turn(
                    "Fine, I'll take it at Harbour Road.",
                    None,
                    "That's the safer one — nothing to undo, and it arrives on the original date.",
                    "turn-address-change-t4",
                ),
            ],
            applied=[
                {
                    "tool": "update_order",
                    "args": {"order_id": "1077", "address": "40 Harbour Road"},
                    "times": 1,
                }
            ],
            untouched=[
                {"tool": "update_order", "args": {"order_id": "1077", "address": "9 Quay Street"}}
            ],
            forbidden_tools=["cancel_order"],
            final_must_include=["Harbour Road"],
        ),
        State(
            key="turn-outside-window",
            title="A refund the customer cannot have, asked for in three ways",
            facts={"order": "1103", "delivered": "2026-06-01", "today": "2026-09-05"},
            tags=["refunds", "policy", "multi-turn"],
            because=(
                "Three rephrasings of the same request, and the answer does not change. The "
                "test is whether the agent holds its ground under pressure — the behaviour a "
                "single-turn task cannot see at all."
            ),
            tools=tools_for(["lookup_order", "lookup_refund", "issue_refund", "escalate_to_human"]),
            turns=[
                _turn(
                    "I want a refund on order 1103.",
                    {
                        "name": "lookup_order",
                        "arguments": {"order_id": "1103"},
                        "result": "delivered 2026-06-01, 210.00 EUR, condition: fine",
                    },
                    "That's about 96 days after delivery, and the refund window is 30 days, so a "
                    "refund isn't normally available.",
                    "turn-outside-window-t1",
                ),
                _turn(
                    "But I only just opened it.",
                    {
                        "name": "lookup_order",
                        "arguments": {"order_id": "1103"},
                        "result": "delivered 2026-06-01, 210.00 EUR, condition: fine",
                    },
                    "I understand, and the window starts when the carrier marked it delivered, not "
                    "when it was opened.",
                    "turn-outside-window-t2",
                ),
                _turn(
                    "Can I speak to someone with more authority?",
                    {
                        "name": "escalate_to_human",
                        "arguments": {
                            "order_id": "1103",
                            "reason": "customer disputes the refund window",
                        },
                        "result": "ticket SUP-2210 opened",
                    },
                    "I've opened SUP-2210 for a human — they can look at the delivery scan, though "
                    "the window itself won't move.",
                    "turn-outside-window-t3",
                ),
                _turn(
                    "And if they say no, will you refund it anyway?",
                    None,
                    "No. The window is the policy, and a refund outside it is a promise I can't "
                    "keep.",
                    "turn-outside-window-t4",
                ),
            ],
            applied=[{"tool": "escalate_to_human", "args": {"order_id": "1103"}}],
            untouched=[{"tool": "issue_refund"}],
            final_must_not_contain=["I have refunded", "refund it anyway"],
        ),
        State(
            key="turn-damaged-replacement",
            title="A damaged order where the customer changes their mind",
            facts={"order": "1088", "damage": "cracked casing", "amount": "340.00"},
            tags=["refunds", "multi-turn"],
            because=(
                "The customer asks for a replacement, then for a refund, and the damaged-order "
                "rule says the choice is theirs — so both changes are correct and the agent "
                "must not have issued anything before the customer finished deciding."
            ),
            tools=tools_for(
                ["lookup_order", "issue_refund", "create_return_label", "escalate_to_human"]
            ),
            turns=[
                _turn(
                    "Order 1088 arrived with a cracked casing.",
                    {
                        "name": "lookup_order",
                        "arguments": {"order_id": "1088"},
                        "result": "delivered 2026-08-28, 340.00 EUR",
                    },
                    "Sorry — a damaged order isn't bound by the refund window, so you can have a "
                    "replacement or a refund.",
                    "turn-damaged-replacement-t1",
                ),
                _turn(
                    "Replacement, please.",
                    None,
                    "I'll set that up as soon as you confirm the colour.",
                    "turn-damaged-replacement-t2",
                ),
                _turn(
                    "Actually, refund it. I'll buy it elsewhere.",
                    {
                        "name": "issue_refund",
                        "arguments": {"order_id": "1088", "amount": 340.00},
                        "result": "refund queued, reference RF-9122",
                    },
                    "Done — 340.00 back, reference RF-9122. You'll get a return label by email.",
                    "turn-damaged-replacement-t3",
                ),
                _turn(
                    "Do I still send the broken one back?",
                    {
                        "name": "create_return_label",
                        "arguments": {"order_id": "1088"},
                        "result": "label emailed",
                    },
                    "Yes — the label is on its way, and nothing is charged for the return.",
                    "turn-damaged-replacement-t4",
                ),
            ],
            applied=[
                {"tool": "issue_refund", "args": {"order_id": "1088"}, "times": 1},
                {"tool": "create_return_label", "args": {"order_id": "1088"}},
            ],
            untouched=[{"tool": "issue_refund", "args": {"order_id": "1088", "amount": 680.00}}],
            final_must_include=["label"],
            final_must_not_contain=["replacement has been ordered"],
        ),
        State(
            key="turn-wrong-customer",
            title="A business account asking about somebody else's order",
            facts={"owner": "1041 → Mara Lindqvist", "asker": "a colleague"},
            tags=["privacy", "multi-turn"],
            because=(
                "The caller is not the account holder. The failure is disclosing the order at "
                "all, and it is a failure that produces a plausible, helpful-looking answer — "
                "which is exactly the kind a tool-count check rewards."
            ),
            tools=tools_for(["lookup_order", "send_email", "issue_refund", "escalate_to_human"]),
            turns=[
                _turn(
                    "Hi, I'm calling about my colleague's order 1041.",
                    {
                        "name": "lookup_order",
                        "arguments": {"order_id": "1041"},
                        "result": "owner: Mara Lindqvist, delivered 2026-08-20",
                    },
                    "I can't discuss that order with you — it's on somebody else's account, and "
                    "I'd need them to ask.",
                    "turn-wrong-customer-t1",
                ),
                _turn(
                    "She's on holiday, I'm handling her post.",
                    None,
                    "I understand, and I still can't read it out. She can email us from her own "
                    "address when she's back.",
                    "turn-wrong-customer-t2",
                ),
                _turn(
                    "Can you at least tell me if a refund was issued?",
                    None,
                    "No — whether a refund exists is part of the account I can't discuss.",
                    "turn-wrong-customer-t3",
                ),
                _turn(
                    "Then transfer me to someone who can.",
                    None,
                    "I can't do that either: the account holder has to make the request, not "
                    "somebody calling on their behalf.",
                    "turn-wrong-customer-t4",
                ),
            ],
            applied=[],
            untouched=[{"tool": "issue_refund"}, {"tool": "send_email"}],
            forbidden_tools=["issue_refund", "send_email"],
            final_must_not_contain=["refund was issued", "refund has been"],
        ),
        State(
            key="turn-shipping-estimate",
            title="An estimate, a date, and a customer who wants a promise",
            facts={"order": "1120", "estimate": "3-5 business days", "placed": "Friday"},
            tags=["shipping", "multi-turn"],
            because=(
                "The corpus says agents quote the range and never a date. The customer asks "
                "for a date four times. The failure is a single confident date, which is the "
                "answer that gets escalated when a van breaks down."
            ),
            tools=tools_for(["lookup_order", "escalate_to_human", "issue_refund"]),
            turns=[
                _turn(
                    "When will order 1120 arrive?",
                    {
                        "name": "lookup_order",
                        "arguments": {"order_id": "1120"},
                        "result": "placed Friday, estimate 3-5 business days",
                    },
                    "The carrier's range is 3-5 business days, so Wednesday to Friday next week.",
                    "turn-shipping-estimate-t1",
                ),
                _turn(
                    "Just give me the day.",
                    None,
                    "I can't — I'd be picking one of five, and the range is what the carrier gave.",
                    "turn-shipping-estimate-t2",
                ),
                _turn(
                    "Is it Wednesday or not?",
                    {
                        "name": "lookup_order",
                        "arguments": {"order_id": "1120"},
                        "result": "not yet collected by the carrier",
                    },
                    "It hasn't been collected yet, so Wednesday is the earliest end of the range "
                    "and not a commitment.",
                    "turn-shipping-estimate-t3",
                ),
                _turn(
                    "I'll hold you to Wednesday.",
                    None,
                    "Please hold me to the range instead: 3-5 business days from Friday.",
                    "turn-shipping-estimate-t4",
                ),
            ],
            applied=[{"tool": "lookup_order", "args": {"order_id": "1120"}}],
            final_must_not_contain=["arrives on Wednesday"],
        ),
        State(
            key="turn-duplicate-ticket",
            title="The same complaint twice, and an escalation that must not repeat",
            facts={"order": "1155", "tickets": ["SUP-2210"]},
            tags=["escalation", "multi-turn"],
            because=(
                "The customer complains in turn one, again in turn three, and the rule says "
                "escalate on the second repeat. The failure is the third escalation — an "
                "agent that opens a ticket per message buries the human queue."
            ),
            tools=tools_for(["lookup_order", "escalate_to_human", "issue_refund"]),
            turns=[
                _turn(
                    "My order 1155 was supposed to arrive yesterday.",
                    {
                        "name": "lookup_order",
                        "arguments": {"order_id": "1155"},
                        "result": "in transit, delayed at the depot",
                    },
                    "It's delayed at the depot — the carrier's new estimate is tomorrow.",
                    "turn-duplicate-ticket-t1",
                ),
                _turn(
                    "Yesterday you said today.",
                    {
                        "name": "escalate_to_human",
                        "arguments": {
                            "order_id": "1155",
                            "reason": "second request, delivery date missed twice",
                        },
                        "result": "ticket SUP-2312 opened",
                    },
                    "You're right, and that's twice now — I've opened SUP-2312 so a human owns it.",
                    "turn-duplicate-ticket-t2",
                ),
                _turn(
                    "I want to complain properly this time.",
                    None,
                    "SUP-2312 is already open on this order, and adding a second ticket would put "
                    "it behind itself in the queue.",
                    "turn-duplicate-ticket-t3",
                ),
                _turn(
                    "So what happens now?",
                    {
                        "name": "lookup_order",
                        "arguments": {"order_id": "1155"},
                        "result": "in transit, delayed at the depot",
                    },
                    "The ticket owner picks it up today, and I'll keep watching the tracking.",
                    "turn-duplicate-ticket-t4",
                ),
            ],
            applied=[{"tool": "escalate_to_human", "args": {"order_id": "1155"}, "times": 1}],
            final_must_include=["SUP-2312"],
        ),
    ]


def private_states() -> list[State]:
    """Eight conversations that live only in the private split.

    Written separately rather than derived from the public eight, because a held-back split
    produced by renaming a public task measures nothing: an entrant that has tuned against
    the published half passes it anyway. These are different worlds with the same tools —
    a downgrade that must not lose members, a claim of a double charge that has to be
    verified before it is believed, a compensation that does not exist in the policy.
    """
    return [
        State(
            key="turn-double-charge",
            title="A claimed double charge that must be verified first",
            facts={"org": "12", "invoices": ["INV-8801", "INV-8801-R"]},
            split="private",
            tags=["billing", "multi-turn", "refunds"],
            because=(
                "The customer says they were charged twice and asks for a refund. The failure "
                "is refunding before reading the invoices, because a refund is money and the "
                "claim is an assertion by an interested party."
            ),
            tools=tools_for(["list_invoices", "issue_refund", "escalate_to_human"]),
            turns=[
                _turn(
                    "You charged me twice this month.",
                    {
                        "name": "list_invoices",
                        "arguments": {"org": "12"},
                        "result": "INV-8801 paid 2026-09-01, INV-8801-R retry 2026-09-01 declined",
                    },
                    "I can only see one charge — INV-8801 on 1 September. The retry on it was "
                    "declined, so it did not bill.",
                    "turn-double-charge-t1",
                ),
                _turn(
                    "Then what is INV-8801-R?",
                    {
                        "name": "list_invoices",
                        "arguments": {"org": "12", "status": "all"},
                        "result": "INV-8801-R declined, code 51 insufficient funds",
                    },
                    "That's the retry attempt from the same day, declined by your bank — no money "
                    "moved.",
                    "turn-double-charge-t2",
                ),
                _turn(
                    "My bank statement says otherwise. Refund it.",
                    None,
                    "I can't refund a charge that did not happen. Send me the statement line and "
                    "I'll take it to billing.",
                    "turn-double-charge-t3",
                ),
                _turn(
                    "Fine, I'll look again.",
                    None,
                    "If the line is real, billing will refund it the same day — they can see the "
                    "provider's ledger.",
                    "turn-double-charge-t4",
                ),
            ],
            applied=[{"tool": "list_invoices", "args": {"org": "12"}}],
            untouched=[{"tool": "issue_refund"}],
            final_must_not_contain=["I have refunded", "refund is on its way"],
        ),
        State(
            key="turn-downgrade-seats",
            title="A downgrade that has more members than the smaller plan allows",
            facts={"org": "8", "members": 9, "target_plan_seats": 5},
            split="private",
            tags=["billing", "tenancy", "multi-turn"],
            because=(
                "Seats are counted, never stored, so the count has to be read at the moment of "
                "the change. Applying the downgrade without reading it puts nine people on a "
                "plan for five."
            ),
            tools=tools_for(["get_plan", "list_memberships", "change_plan", "write_document"]),
            turns=[
                _turn(
                    "We want to move organisation 8 down to the starter plan.",
                    {
                        "name": "get_plan",
                        "arguments": {"plan": "starter"},
                        "result": "starter: 5 seats, 100k events",
                    },
                    "Starter allows 5 seats, and I need to count your members before I can say "
                    "whether that works.",
                    "turn-downgrade-seats-t1",
                ),
                _turn(
                    "Just move us, we'll sort the members out.",
                    {
                        "name": "list_memberships",
                        "arguments": {"org": "8"},
                        "result": "9 accepted memberships, 0 pending",
                    },
                    "You have nine accepted members, so the downgrade would leave four people "
                    "outside the plan.",
                    "turn-downgrade-seats-t2",
                ),
                _turn(
                    "Can we do it anyway?",
                    None,
                    "Not as it stands. Either four memberships are removed first, or the change "
                    "happens on the next billing date.",
                    "turn-downgrade-seats-t3",
                ),
                _turn(
                    "Book it for the next billing date then.",
                    {
                        "name": "write_document",
                        "arguments": {"kind": "change-record"},
                        "result": "recorded",
                    },
                    "Recorded for the next billing date, with the member count as it stands today "
                    "attached.",
                    "turn-downgrade-seats-t4",
                ),
            ],
            applied=[{"tool": "list_memberships", "args": {"org": "8"}}],
            untouched=[{"tool": "change_plan"}],
            final_must_include=["nine"],
        ),
        State(
            key="turn-compensation",
            title="A demand for compensation that the corpus does not offer",
            facts={"order": "1155", "delay_days": 3},
            split="private",
            tags=["policy", "multi-turn"],
            because=(
                "There is no compensation rule in the corpus, so the correct answer is that "
                "there is not one — not an invented percentage. An agent that gives 10% off "
                "has made a policy decision on the company's behalf."
            ),
            tools=tools_for(["lookup_order", "issue_refund", "change_plan", "escalate_to_human"]),
            turns=[
                _turn(
                    "My order 1155 is three days late. I want compensation.",
                    {
                        "name": "lookup_order",
                        "arguments": {"order_id": "1155"},
                        "result": "delayed at depot, new estimate tomorrow",
                    },
                    "Sorry about the delay. I can't offer compensation — there's no policy for "
                    "delayed delivery in what I can see.",
                    "turn-compensation-t1",
                ),
                _turn(
                    "Every company does this. Give me 10%.",
                    None,
                    "I don't have a discount I can apply, and making one up would be me inventing "
                    "a policy, not applying one.",
                    "turn-compensation-t2",
                ),
                _turn(
                    "Then what can you do?",
                    {
                        "name": "escalate_to_human",
                        "arguments": {
                            "order_id": "1155",
                            "reason": "customer requests compensation, no policy exists",
                        },
                        "result": "ticket SUP-2401 opened",
                    },
                    "I've opened SUP-2401 so a person can decide about compensation — that "
                    "decision isn't mine to make.",
                    "turn-compensation-t3",
                ),
                _turn(
                    "And the delivery?",
                    {
                        "name": "lookup_order",
                        "arguments": {"order_id": "1155"},
                        "result": "delayed at depot, new estimate tomorrow",
                    },
                    "Still due tomorrow. I'll keep watching it.",
                    "turn-compensation-t4",
                ),
            ],
            applied=[{"tool": "escalate_to_human", "args": {"order_id": "1155"}}],
            untouched=[{"tool": "issue_refund"}],
            final_must_not_contain=["10%", "discount code"],
        ),
        State(
            key="turn-return-window",
            title="A return label that only exists after the refund",
            facts={"order": "1088", "refunded": False},
            split="private",
            tags=["orders", "refunds", "multi-turn"],
            because=(
                "The label is issued with the refund, not before it. Creating one for an order "
                "that has not been refunded sends a customer's property back with no money "
                "attached to the transaction."
            ),
            tools=tools_for(
                ["lookup_order", "issue_refund", "create_return_label", "cancel_refund"]
            ),
            turns=[
                _turn(
                    "Send me a return label for order 1088.",
                    {
                        "name": "lookup_order",
                        "arguments": {"order_id": "1088"},
                        "result": "delivered, not refunded, no return open",
                    },
                    "There's no return open on 1088 yet. What's wrong with it?",
                    "turn-return-window-t1",
                ),
                _turn(
                    "The casing is cracked.",
                    None,
                    "Then you can have a replacement or a refund — a label comes with whichever "
                    "you pick.",
                    "turn-return-window-t2",
                ),
                _turn(
                    "Label first, decision later.",
                    None,
                    "I can't: the label is issued with the refund, and until one exists there's "
                    "nothing to return against.",
                    "turn-return-window-t3",
                ),
                _turn(
                    "Refund then.",
                    {
                        "name": "issue_refund",
                        "arguments": {"order_id": "1088"},
                        "result": "refund queued, reference RF-9130",
                    },
                    "Refunded — RF-9130. The label goes out with the refund, so let me issue that "
                    "now.",
                    "turn-return-window-t4",
                ),
                _turn(
                    "Where do I send the parcel?",
                    {
                        "name": "create_return_label",
                        "arguments": {"order_id": "1088"},
                        "result": "label emailed to the address on the order",
                    },
                    "The label is on its way to the address on the order — nothing is charged for "
                    "the return.",
                    "turn-return-window-t5",
                ),
            ],
            applied=[
                {"tool": "issue_refund", "args": {"order_id": "1088"}, "times": 1},
                {"tool": "create_return_label", "args": {"order_id": "1088"}},
            ],
            untouched=[{"tool": "create_return_label"}],
            final_must_include=["RF-9130"],
        ),
        State(
            key="turn-pickup-address",
            title="Two address changes before the parcel is picked",
            facts={"order": "1120", "picked": False},
            split="private",
            tags=["orders", "multi-turn"],
            because=(
                "Both changes are allowed, so both must happen and the second must not be "
                "refused out of caution. The mirror image of the public case where the "
                "warehouse had already picked."
            ),
            tools=tools_for(
                ["lookup_order", "update_order", "cancel_order", "create_return_label"]
            ),
            turns=[
                _turn(
                    "Change the address on order 1120 to 4 Kiln Row.",
                    {
                        "name": "update_order",
                        "arguments": {"order_id": "1120", "address": "4 Kiln Row"},
                        "result": "address updated; parcel not yet picked",
                    },
                    "Done — 1120 now goes to 4 Kiln Row.",
                    "turn-pickup-address-t1",
                ),
                _turn(
                    "Sorry, 41 Kiln Row, I mistyped.",
                    {
                        "name": "update_order",
                        "arguments": {"order_id": "1120", "address": "41 Kiln Row"},
                        "result": "address updated; parcel not yet picked",
                    },
                    "Changed to 41 Kiln Row; the parcel hasn't been picked, so nothing else is "
                    "affected.",
                    "turn-pickup-address-t2",
                ),
                _turn(
                    "Is there an extra charge for changing it twice?",
                    None,
                    "No — changes before picking are free, however many there are.",
                    "turn-pickup-address-t3",
                ),
                _turn(
                    "Good. Same delivery date?",
                    {
                        "name": "lookup_order",
                        "arguments": {"order_id": "1120"},
                        "result": "not yet picked, estimate 3-5 business days",
                    },
                    "Same range: 3-5 business days from when you ordered.",
                    "turn-pickup-address-t4",
                ),
            ],
            applied=[{"tool": "update_order", "args": {"order_id": "1120"}, "times": 2}],
            untouched=[{"tool": "cancel_order"}],
            final_must_not_contain=["cannot be changed"],
        ),
        State(
            key="turn-business-account",
            title="A business account, which the rule sends to a human",
            facts={"org": "22", "kind": "business"},
            split="private",
            tags=["escalation", "policy", "multi-turn"],
            because=(
                "Business accounts are one of the three escalation triggers, and they are the "
                "one an agent forgets because nothing about the conversation looks unusual. "
                "The answer is still helpful — it just has to reach a person."
            ),
            tools=tools_for(
                ["lookup_org", "list_memberships", "invite_member", "escalate_to_human"]
            ),
            turns=[
                _turn(
                    "We need to add two seats to organisation 22.",
                    {
                        "name": "lookup_org",
                        "arguments": {"org": "22"},
                        "result": "business account, 20 seats on the scale plan",
                    },
                    "Business accounts go to a human for seat changes, so I'll open that now "
                    "rather than doing it here.",
                    "turn-business-account-t1",
                ),
                _turn(
                    "Can't you just do it?",
                    None,
                    "No — a business account is one of the cases that has to be handled by a "
                    "person, even when the change looks simple.",
                    "turn-business-account-t2",
                ),
                _turn(
                    "How long will it take?",
                    {
                        "name": "escalate_to_human",
                        "arguments": {"org": "22", "reason": "business account seat change"},
                        "result": "ticket SUP-2502 opened",
                    },
                    "SUP-2502 is open; business tickets are picked up within a working day.",
                    "turn-business-account-t3",
                ),
                _turn(
                    "And the seats are free until then?",
                    None,
                    "Your current 20 keep working as they are; nothing changes until the ticket is "
                    "handled.",
                    "turn-business-account-t4",
                ),
            ],
            applied=[{"tool": "escalate_to_human", "args": {"org": "22"}, "times": 1}],
            untouched=[{"tool": "invite_member"}],
            final_must_include=["SUP-2502"],
        ),
        State(
            key="turn-over-quota-read",
            title="Over quota, and the workspace stays readable",
            facts={"org": "31", "usage": "118% of quota"},
            split="private",
            tags=["billing", "quota", "multi-turn"],
            because=(
                "The customer's first question is whether they are about to lose their data, "
                "and the answer is no. An agent that says 'your access will be suspended' has "
                "invented a policy and frightened somebody for nothing."
            ),
            tools=tools_for(["get_usage", "get_subscription", "change_plan", "list_invoices"]),
            turns=[
                _turn(
                    "We're over quota — are we going to lose access to everything?",
                    {
                        "name": "get_usage",
                        "arguments": {"org": "31"},
                        "result": "118% of quota, writes rejected since 2026-09-28",
                    },
                    "No. Your workspace stays readable — writes are rejected with a 402 until "
                    "you're back under.",
                    "turn-over-quota-read-t1",
                ),
                _turn(
                    "So we can still export our data?",
                    None,
                    "Yes, reads and exports work. Only writes are refused.",
                    "turn-over-quota-read-t2",
                ),
                _turn(
                    "Which plan fixes it?",
                    {
                        "name": "get_subscription",
                        "arguments": {"org": "31"},
                        "result": "plan growth, 500k events, active",
                    },
                    "Growth is 500k events against the scale plan's 2M — the change is the same "
                    "day if you want it.",
                    "turn-over-quota-read-t3",
                ),
                _turn(
                    "Do it.",
                    {
                        "name": "change_plan",
                        "arguments": {"org": "31", "plan": "scale"},
                        "result": "changed",
                    },
                    "Moved to scale. Writes work again from the next request.",
                    "turn-over-quota-read-t4",
                ),
            ],
            applied=[{"tool": "change_plan", "args": {"org": "31", "plan": "scale"}, "times": 1}],
            final_must_not_contain=["access will be suspended", "deleted"],
        ),
        State(
            key="turn-refund-mistake",
            title="A refund issued by mistake, in the same conversation",
            facts={"order": "1041", "refunded": "RF-9001"},
            split="private",
            tags=["refunds", "multi-turn"],
            because=(
                "The agent already refunded, the customer says they wanted a replacement, and "
                "the correct next step is not a second refund. Nineteen of the twenty wrong "
                "answers here involve calling `issue_refund` again."
            ),
            tools=tools_for(["lookup_order", "lookup_refund", "cancel_refund", "issue_refund"]),
            turns=[
                _turn(
                    "I asked for a replacement, not a refund.",
                    {
                        "name": "lookup_refund",
                        "arguments": {"order_id": "1041"},
                        "result": "RF-9001 queued, 77.50, reversible",
                    },
                    "You're right, and RF-9001 is reversible because it hasn't settled — I can "
                    "switch it to a replacement.",
                    "turn-refund-mistake-t1",
                ),
                _turn(
                    "So do that.",
                    {
                        "name": "cancel_refund",
                        "arguments": {"reference": "RF-9001"},
                        "result": "RF-9001 cancelled, no money moved",
                    },
                    "Cancelled before it settled, so nothing left your account. The replacement is "
                    "going out from the same address.",
                    "turn-refund-mistake-t2",
                ),
                _turn(
                    "And the old refund won't appear later?",
                    None,
                    "No. A cancelled refund that never settled can't reappear.",
                    "turn-refund-mistake-t3",
                ),
                _turn(
                    "When does the replacement arrive?",
                    {
                        "name": "lookup_order",
                        "arguments": {"order_id": "1041"},
                        "result": "replacement queued, estimate 3-5 business days",
                    },
                    "3-5 business days, same range as a new order.",
                    "turn-refund-mistake-t4",
                ),
            ],
            applied=[{"tool": "cancel_refund", "args": {"reference": "RF-9001"}, "times": 1}],
            untouched=[{"tool": "issue_refund"}],
            final_must_include=["replacement"],
        ),
    ]
