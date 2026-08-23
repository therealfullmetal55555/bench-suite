# Alert rules

The rules that page a human, with the thresholds and the reason each threshold is where it
is. A threshold without a reason is a number somebody picked at the end of a long day, and
the first thing an on-call engineer does with those is turn them off.

## Latency

`LatencyP99` pages when the p99 of any HTTP route is above 2 seconds for ten minutes. Ten
minutes rather than one because a single slow minute is a garbage collection pause, and one
minute of paging per GC pause is how an on-call rotation burns out. The alert carries the
slowest route in its labels; without it the first action at 3am is a dashboard link.

## Error rate

`ErrorRate` pages when the five-minute error ratio for a service is above 2% and the request
rate is above 1 per second. Both conditions are needed: 100% of three requests is a deploy
with a broken health check, not an outage, and paging on it teaches people to ignore pager
traffic. The error ratio counts 5xx only; 4xx is the customer's business, not the platform's.

## Absence

`ScrapeAbsence` pages when a target that has reported for a week stops reporting for ten
minutes. Absence is the alert everyone forgets to write and the one that matters, because a
service that has stopped being scraped also stops firing every other alert it owns — a
silent collector looks exactly like a healthy system from inside the dashboard.

## Saturation

`Saturation` warns at 80% of a resource for thirty minutes and pages at 95% for five. The
warning exists so the page is not the first thing anyone hears about it: a queue at 80% for
half an hour is a team that can add a consumer without being woken up to do it.
