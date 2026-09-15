"""Budgets, and the clock that has to be injected for these tests to mean anything.

A budget test that sleeps is a test that fails on a loaded CI runner and passes on a
quiet laptop, which is the same class of defect as a benchmark that measures the
wrong thing. The clock is a parameter here, so "over the wall-clock budget" is
asserted by moving a number rather than by waiting for one.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from bench.core.budget import BudgetMeter, price_tokens


class FakeClock:
    """Monotonic time that only moves when a test says so."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# --- the meter ---------------------------------------------------------------


def test_a_task_inside_its_budget_is_not_over_it():
    clock = FakeClock()
    meter = BudgetMeter(max_usd=Decimal("0.10"), max_seconds=30.0, clock=clock)
    meter.start()
    meter.spend(usd=Decimal("0.05"), seconds=1.0)
    clock.advance(2.0)
    assert meter.over_budget() is None
    assert not meter.exhausted


def test_cost_is_checked_before_time_because_it_keeps_growing():
    """Both of them blown, and the report has to pick one sentence. Cost wins: it is
    the one that carries on after the task stops."""
    clock = FakeClock()
    meter = BudgetMeter(max_usd=Decimal("0.01"), max_seconds=1.0, clock=clock)
    meter.start()
    meter.spend(usd=Decimal("0.05"))
    clock.advance(60.0)
    reason = meter.over_budget()
    assert reason is not None
    assert reason.startswith("cost $0.0500 over the $0.0100 budget")


def test_wall_time_counts_even_when_the_target_reports_nothing():
    """A target that spends thirty seconds in a retry loop it does not report still
    blows the budget. This is the whole reason `elapsed` adds recorded time and wall
    time rather than trusting the target's own accounting."""
    clock = FakeClock()
    meter = BudgetMeter(max_seconds=10.0, clock=clock)
    meter.start()
    clock.advance(12.0)
    assert meter.over_budget() == "wall 12.0s over the 10.0s budget"


def test_recorded_seconds_and_wall_time_add_up():
    clock = FakeClock()
    meter = BudgetMeter(clock=clock)
    meter.start()
    clock.advance(3.0)
    meter.spend(seconds=4.0)
    assert meter.elapsed() == pytest.approx(7.0)


def test_time_does_not_run_before_start():
    """Asking a meter that has not started how long it has been going is a
    programming error, and the useful answer is zero rather than the age of the
    process."""
    meter = BudgetMeter(clock=FakeClock())
    assert meter.elapsed() == 0.0


def test_a_budget_of_zero_is_refused():
    with pytest.raises(ValueError, match="never be run"):
        BudgetMeter(max_usd=Decimal("0"))
    with pytest.raises(ValueError, match="never be run"):
        BudgetMeter(max_seconds=0.0)


def test_no_budget_means_no_limit():
    meter = BudgetMeter(clock=FakeClock())
    meter.start()
    meter.spend(usd=Decimal("999.0"))
    assert meter.over_budget() is None
    assert meter.remaining_usd() is None
    assert meter.summary() == "no budget"


def test_the_meter_can_say_how_much_is_left():
    """Used by the runner to tell a target how much more it may spend, and by the
    report to say how close an entry came to its ceiling."""
    meter = BudgetMeter(max_usd=Decimal("0.10"))
    meter.spend(usd=Decimal("0.03"))
    assert meter.remaining_usd() == Decimal("0.07")
    meter.spend(usd=Decimal("0.20"))
    assert meter.remaining_usd() == Decimal("0")  # never negative


def test_the_summary_reads_like_a_budget_line():
    meter = BudgetMeter(max_usd=Decimal("0.10"), max_seconds=30.0)
    meter.start()
    meter.spend(usd=Decimal("0.025"))
    assert meter.summary().startswith("$0.0250/$0.1000 0.0s/30.0s")


def test_spending_past_the_line_is_reported_and_not_clamped():
    """The overspend is the interesting number: an entry that blew a five-cent
    budget by two dollars should not read as "5/5 cents"."""
    meter = BudgetMeter(max_usd=Decimal("0.05"))
    meter.spend(usd=Decimal("2.05"))
    assert meter.spent_usd == Decimal("2.05")
    assert meter.over_budget() == "cost $2.0500 over the $0.0500 budget"


# --- pricing -----------------------------------------------------------------


def test_pricing_is_none_when_no_rate_is_configured():
    """`None` is not zero. A leaderboard that prints `$0.0000` for an agent nobody
    priced is claiming it is free, and the difference between free and unmeasured is
    the difference between two entries being comparable and not."""
    assert (
        price_tokens(
            prompt_tokens=1000,
            completion_tokens=500,
            prompt_per_mtok=None,
            completion_per_mtok=None,
        )
        is None
    )
    assert (
        price_tokens(
            prompt_tokens=1000,
            completion_tokens=500,
            prompt_per_mtok=0.15,
            completion_per_mtok=None,
        )
        is None
    )


def test_pricing_arithmetic_on_a_known_call():
    # 1M prompt tokens at $3/MTok and 1M completion at $15/MTok is $18.
    cost = price_tokens(
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        prompt_per_mtok=3.0,
        completion_per_mtok=15.0,
    )
    assert cost == Decimal("18.000000")


def test_cached_tokens_are_billed_at_the_cache_rate():
    """The difference between an expensive agent and a cheap one with a long system
    prompt, and the cost column is wrong without it."""
    cost = price_tokens(
        prompt_tokens=1_000_000,
        completion_tokens=0,
        cached_tokens=900_000,
        prompt_per_mtok=3.0,
        completion_per_mtok=15.0,
        cached_per_mtok=0.3,
    )
    # 100k at $3 + 900k at $0.30
    assert cost == Decimal("0.570000")


def test_without_a_cache_rate_the_full_prompt_price_is_charged():
    """Conservative on purpose: granting an unpriced discount flatters an entry, and
    the platform that would have to justify it is not the one being measured."""
    cost = price_tokens(
        prompt_tokens=1_000_000,
        completion_tokens=0,
        cached_tokens=900_000,
        prompt_per_mtok=3.0,
        completion_per_mtok=15.0,
    )
    assert cost == Decimal("3.000000")


def test_cached_more_than_prompt_is_clamped_and_not_billed_as_a_discount():
    """Providers do report a cache hit larger than the prompt — usually after a retry
    that lost its usage record. Billing the phantom tokens at the *cache* rate lets an
    entry come in under its true cost, so the cached count is clamped to the prompt
    count first. Found by this test expecting the clamped answer and getting the
    unclamped one."""
    cost = price_tokens(
        prompt_tokens=100,
        completion_tokens=0,
        cached_tokens=500,
        prompt_per_mtok=3.0,
        completion_per_mtok=15.0,
        cached_per_mtok=0.3,
    )
    # all 100 prompt tokens at the cache rate, and the extra 400 nowhere
    assert cost == Decimal("0.000030")
    assert cost >= 0


def test_a_negative_cache_count_cannot_make_a_call_free():
    cost = price_tokens(
        prompt_tokens=1000,
        completion_tokens=0,
        cached_tokens=-50,
        prompt_per_mtok=3.0,
        completion_per_mtok=15.0,
    )
    assert cost == Decimal("0.003000")


def test_cost_is_a_decimal_because_it_is_money():
    """`0.1 + 0.2 != 0.3` in a cost column is how a report ends up arguing with
    itself across a thousand calls."""
    total = Decimal("0")
    for _ in range(1000):
        total += price_tokens(
            prompt_tokens=100,
            completion_tokens=100,
            prompt_per_mtok=0.15,
            completion_per_mtok=0.6,
        ) or Decimal("0")
    assert total == Decimal("0.075000")
