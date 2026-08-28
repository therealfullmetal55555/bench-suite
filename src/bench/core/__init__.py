"""The harness: tasks, corpora, traces, budgets, scores, and the runner.

Layering, which the imports follow and a test enforces: `stats` and `trace` depend on
nothing here; `corpus`, `task`, `budget` and `score` depend on those; `runner` and
`store` depend on all of it. `scorers` sit beside `core`, not inside it, because a
scorer is a rule about answers and the core should not know what one is.
"""
