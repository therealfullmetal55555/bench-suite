"""Budgets, and the rule that spending too much is a way to lose.

Half of real agent work is "and it has to finish". A benchmark that lets an agent
take six minutes and four dollars to answer a question ranks it above one that
answers in two seconds for a tenth of a cent and calls that better, which is not a
statement anybody's product needs.

So a budget is enforced while the task runs, not compared afterwards. Two shapes:

* **Per task** (`budget.max_usd`, `budget.max_seconds` on the task) — the task is
  stopped, the partial trace is kept, and the outcome is `over-budget`.
* **Per run** (`--max-usd`, `--max-seconds` on the command line) — the run stops
  cleanly, every task not yet started is `skipped`, and the bundle records that it
  was a partial run. A partial run can never be mistaken for a full one, because the
  result carries the reason.

The clock is injected. A budget test that sleeps is a test that fails on a busy CI
machine and passes on a quiet one, which is the same class of bug as a benchmark
that measures the wrong thing.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal


@dataclass
class BudgetMeter:
    """Tracks what one task or one run has spent so far.

    Costs arrive as `Decimal` because they are money and because `0.1 + 0.2 != 0.3`
    in a cost column is how a report ends up arguing with itself. Seconds arrive as
    floats because they are durations.
    """

    max_usd: Decimal | None = None
    max_seconds: float | None = None
    spent_usd: Decimal = Decimal("0")
    spent_seconds: float = 0.0
    clock: Callable[[], float] = time.monotonic

    _started: float = field(default=0.0, repr=False)
    _running: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_usd is not None and self.max_usd <= 0:
            raise ValueError("a budget of zero is a task that can never be run")
        if self.max_seconds is not None and self.max_seconds <= 0:
            raise ValueError("a budget of zero seconds is a task that can never be run")

    # --- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        self._started = self.clock()
        self._running = True

    def spend(self, *, usd: Decimal | None = None, seconds: float | None = None) -> None:
        """Record one model call or tool call. Passing both is the normal case: a
        call has a cost and a duration, and a budget that only watches one of them
        is a budget that can be beaten by the other."""
        if usd is not None:
            self.spent_usd += usd
        if seconds is not None:
            self.spent_seconds += seconds

    def elapsed(self) -> float:
        """Wall time since `start()`, plus whatever was recorded explicitly.

        Both, because they measure different things: recorded seconds are the calls
        the target told us about, wall time is what actually happened. A target that
        spends 30 seconds in a retry loop it does not report still blows the budget,
        which is the behaviour anybody running a benchmark wants.
        """
        wall = (self.clock() - self._started) if self._running else 0.0
        return self.spent_seconds + wall

    # --- questions ----------------------------------------------------------

    def over_budget(self) -> str | None:
        """Why, in the words the report will print. `None` means fine.

        Cost is checked first: it is the one that keeps growing after the fact, and
        "you are four dollars in" is more informative than "you are ninety seconds
        in" when both are true.
        """
        if self.max_usd is not None and self.spent_usd > self.max_usd:
            return f"cost ${self.spent_usd:.4f} over the ${self.max_usd:.4f} budget"
        if self.max_seconds is not None and self.elapsed() > self.max_seconds:
            return f"wall {self.elapsed():.1f}s over the {self.max_seconds:.1f}s budget"
        return None

    @property
    def exhausted(self) -> bool:
        return self.over_budget() is not None

    def remaining_usd(self) -> Decimal | None:
        if self.max_usd is None:
            return None
        return max(Decimal("0"), self.max_usd - self.spent_usd)

    def summary(self) -> str:
        parts = []
        if self.max_usd is not None:
            parts.append(f"${self.spent_usd:.4f}/${self.max_usd:.4f}")
        if self.max_seconds is not None:
            parts.append(f"{self.elapsed():.1f}s/{self.max_seconds:.1f}s")
        return " ".join(parts) or "no budget"


def price_tokens(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    prompt_per_mtok: float | None,
    completion_per_mtok: float | None,
    cached_per_mtok: float | None = None,
) -> Decimal | None:
    """Price a call from per-million-token rates.

    `None` when no rate is configured, and the caller must keep it `None` rather
    than substituting zero: `$0.0000` in a cost column is a claim that the run was
    free, and the difference between "free" and "not measured" is the difference
    between two leaderboard entries being comparable and not.

    Cached tokens fall back to the prompt rate when no cache rate is given, which is
    the conservative direction — it charges the entrant the full price rather than
    silently granting a discount. And the cached count is clamped to the prompt count
    before it is used: providers do report a cache hit larger than the prompt, usually
    after a retry that lost its usage record, and billing those phantom tokens at the
    *discounted* rate is a quiet way for an entry to come in under its true cost.
    """
    if prompt_per_mtok is None or completion_per_mtok is None:
        return None
    million = Decimal("1000000")
    cached = min(max(0, cached_tokens), prompt_tokens)
    billed_prompt = prompt_tokens - cached
    prompt_cost = Decimal(str(prompt_per_mtok)) * Decimal(billed_prompt) / million
    cache_rate = prompt_per_mtok if cached_per_mtok is None else cached_per_mtok
    cache_cost = Decimal(str(cache_rate)) * Decimal(cached) / million
    completion_cost = Decimal(str(completion_per_mtok)) * Decimal(completion_tokens) / million
    return (prompt_cost + cache_cost + completion_cost).quantize(Decimal("0.000001"))
