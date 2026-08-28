# Runbooks

What to do when a page fires, written the night it fired. Each runbook is one alert, one
cause, and the command that tells them apart — because the difference between a 3am page
that takes ten minutes and one that takes two hours is whether the runbook names the next
thing to look at.

## Token cost spike

The `TokenCostSpike` alert fires when the hourly cost of a service is more than triple its
seven-day median. The usual cause is a prompt change that turned a cached prefix into a fresh
one: the tokens are the same and the bill is not. Check the cached-token ratio in the
`tokens` panel first; if the ratio fell off a cliff, the change is a prompt, not traffic.
Raising the budget is not a fix, it is a decision to keep paying.

## Trace loss

`TraceLoss` fires when the collector accepts spans and the backend stores fewer than 95% of
them. The cause is almost always the exporter queue filling while the collector's memory
limiter drops batches. Look at `otelcol_exporter_queue_size` and
`otelcol_processor_dropped_spans`; if the second is rising, the fix is a bigger queue or
fewer attributes, not a bigger backend.

## Collector backpressure

When the memory limiter refuses to allocate, the collector stops accepting data and the
sending SDKs start retrying, which doubles the traffic the collector sees. The limit is
`limit_mib`, the number that matters is `spike_limit_mib`, and a collector that has hit
either one is a collector whose limits were sized for a smaller cluster. Sample aggressively
at the SDK before raising anything, because unsampled debug logs are what fills the queue in
the first place.

## Noisy alert

An alert that pages more than twice a week without a customer-visible symptom is deleted, not
silenced. Silencing turns a known-bad signal into a rumour: it stops being read, and the one
time it was right nobody looks. The alert rule goes in the same pull request as the deletion
of its runbook.
