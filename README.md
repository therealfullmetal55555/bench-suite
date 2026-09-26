# bench-suite

A public benchmark for retrieval-augmented and tool-using agents. Six task families,
a leaderboard built from files in `results/`, and every number published with the
interval it rests on, the cost it took, and the list of tasks the entrant failed.

**Status: the harness runs, it scores, and there is a task set.** Tasks, corpora, the trace
contract, budgets, targets, the runner, run files and the statistics are written and tested,
and so are all thirteen expectation types across the six families — including the judge, with
a calibration that has to clear a κ floor before a single judged verdict is allowed to decide
a score. `bench run` drives a real entrant, scores it, writes a run file, and `bench show`
prints the score next to the sentences that produced it. `bench task generate` builds the
task set — **172 tasks, 86 public and 86 private**, every phrase checked against the document
it is attributed to. The board and the submission tooling are next (SPEC §12). The status
table near the bottom says exactly what runs today, because a README that describes a
finished benchmark while the harness cannot score a task is the first thing a reviewer
notices.

---

## The problem this solves

Leaderboards are usually one number. 71.4%, no interval, no cost, no latency, one run
per entrant. Two entries three places apart are usually indistinguishable
statistically, and the board does not say so, so people read the rank as the truth.
Then the tasks leak into training data, the number goes up, and the benchmark quietly
stops measuring anything.

This one is built around four refusals:

- **No rank without an interval.** Differences are tested on shared tasks. Overlapping
  intervals print as `tied` and sort as tied.
- **No score without a price.** Dollars and p95 latency sit beside quality, and the
  default view is the Pareto frontier rather than a list.
- **No judgement without an audit.** Each family that leans on an LLM judge publishes
  Cohen's κ against human labels. Below the floor, the family is *excluded from the
  aggregate and named as excluded* — not scored badly.
- **No number without its failures.** Every report card lists the task ids the entry
  failed, with a transcript link. A score without the failures is an advertisement.

---

## The one design decision that matters

**A score is a distribution over tasks, not a float.**

Nothing in this package returns a bare number that claims to be a result. `Score` is a
value, a task count, a label and an interval; `TaskScore` is one task's outcome with
its repeat count, its flakiness and whether an untrusted judge decided it. The reason
is what happens to a float the moment it exists: it gets sorted, and sorting is the act
of claiming a rank you cannot support. So the arithmetic is done in a type that
carries its own sample size, and `Score.__post_init__` refuses a value outside [0, 1]
or an interval that does not contain its own value — the two wrong numbers that
travel.

Two consequences worth knowing before reading the code:

- Aggregation is explicit about **which** mean it is. A family mean averages tasks; the
  headline is a **stratified** mean of family means, so a family with eighty tasks
  cannot outvote a family with twelve. Both are printed, because the gap between them
  is the most useful thing a reader can learn about a benchmark's task mix.
- **An error is a failure.** An entry that errored on 12% of tasks scores zero on them
  and pays for it; the 12% is published as its own column. Hiding errors in a "did not
  run" bucket is how a benchmark ranks a fragile agent above a working one.

---

## How a task is scored

A task carries expectations, each one a claim about the trace. Every check is a function of
what it was handed — the answer, the tool calls, the corpus, and for one of them a judge —
and every one of them returns a sentence as well as a verdict, because the sentence is what
ends up in the failure list a person reads.

| Type | Reads | Fails with |
| --- | --- | --- |
| `contains` · `not_contains` | the answer text | `the answer does not contain 'permission': I think the rollout …` |
| `regex` | the answer text | `/^root cause:/ does not match …` |
| `json` · `schema` | the answer as a document | `lines/0: 1 is not of type 'object' (+1 more)` |
| `citations` | the answer, the corpus, what was retrieved | `did not cite faq.md#refunds; cited without retrieving: policy.md#refunds` |
| `answerable` | the answer, against a published refusal list | `the corpus cannot answer this and the agent answered anyway: …` |
| `trajectory` · `steps` | the tool calls | `missing required tool(s) search_docs (called: get_policy)` |
| `state` | the tool calls **and their arguments** | `called issue_refund(order_id='1041') 2 time(s), and the task says 1` |
| `fields` | the answer as a mapping, per field, weighted | `67% of the fields — wrong value for total: expected 77.5, got 70` |
| `refusal` | the answer, two-sided | `this request must be declined and the answer did not: Sure, here are …` |
| `judge` | a model, and only if it cleared its calibration | `not scored: κ 0.41 is below the floor 0.70` |

Four decisions in that table are worth naming, because they are the ones a reviewer would
otherwise have to reconstruct from the code:

- **`answerable` is a phrase list, not a judge.** It has false negatives — an agent that
  declines in words nobody listed is marked wrong — and that is the price of a check that
  is free, reproducible and readable in a diff. The list is in `scorers/citations.py`,
  where it can be argued with.
- **`state` is separate from `trajectory`** because the right verb on the wrong object is
  the expensive mistake: refunding order 1042 on a task about 1041 is worse than doing
  nothing, and a tool-name check calls it a pass.
- **`fields` gives partial credit and never normalises beyond case and whitespace.** A
  19-of-20 extraction is not the same product as a 2-of-20 one, and a scorer that strips
  punctuation is a scorer whose failures nobody can reproduce.
- **An unscored expectation leaves the denominator.** A task nothing could decide is
  reported `not scorable` and excluded from the mean — never averaged in as a zero, because
  a harness gap that reads as an agent's failure is the one thing this project does not get
  to do.

Judge verdicts are cached on the *question* — rubric, prompt, answer, model, temperature —
rather than on a task id, so a re-scored submission costs no judge calls, and editing a
task cannot serve a stale verdict. Until a calibration on file says the judge agrees with a
person at κ ≥ 0.70 over at least 20 labels, judged expectations are reported and excluded:

```bash
bench judge calibrate --judge judge.yaml --labels labels.jsonl   # exit 2 if unusable
bench judge status --judge judge.yaml                            # what is on file now
```

Both files have a worked example: `examples/judge.yaml` and `examples/labels.jsonl`, the
latter being twenty-four hand-labelled answers over the example corpus — deliberately half
of them borderline, because twenty-four obvious cases would give any judge a high κ and
measure nothing. Running the calibration is the one thing in this repository that needs a
live model and an API key, which is why it is the one thing that has not been run yet.

---

## The task set

172 tasks in six families, generated from data rather than typed out, with both splits written
from scratch — the private half is not a rename of the public half, and the tests say so.

| Family | Tasks | Public / private | A pass means | Scored by |
| --- | --- | --- | --- | --- |
| `grounded-qa` | 80 | 40 / 40 | the answer, its citation, and whether the question was answerable at all | contains · citations · answerable |
| `tool-trajectory` | 16 | 8 / 8 | the right calls, in the right order, and none of the forbidden ones | trajectory · steps |
| `structured-extraction` | 16 | 8 / 8 | every field of an invoice, against a schema | json · schema · fields |
| `multi-turn-state` | 16 | 8 / 8 | the world was read before it was changed | state · trajectory · citations |
| `refusal-and-safety` | 32 | 16 / 16 | refused when it must be, helped when it must be | refusal · trajectory |
| `long-horizon` | 12 | 6 / 6 | a plan where a later step depends on an earlier result | steps · trajectory · judge |

The data lives in `src/bench/generators/data/`: 32 grounded facts with the passage each one
rests on, 8 unanswerable questions with the fabricated answers a guess would produce, 16
invoices (three of them adversarial — a credit note with a negative total, one with no
invoice number at all, one where the VAT is only in the prose), 8 tool worlds, 8 public and 8
private conversations, 8 safety pairs, 6 long plans. `make tasks` runs the generator and
writes `tasks/` — one YAML file per family plus the four corpora, so a generated set can be
moved or committed on its own.

Two things make a generated task set trustworthy rather than merely larger, and both are
tests:

* **Every claim is looked up.** Each `must_include` phrase is checked against the passage it
  is attributed to, in the spelling and word wrapping the document uses; every required tool
  is checked against the tool list the task ships; every `applied` call is checked against the
  transcript it is supposed to appear in; every extraction value is checked against the
  rendered document. A task whose claim is false fails in CI instead of failing an entrant
  who cannot see why.
* **The committed set is byte-compared to what the generator writes today.** Somebody who
  hand-edits a task and forgets the generator finds out on the next `make check` — a diff,
  not a silently reverted file.

The generator also says what it does *not* do, because a portfolio piece that claims to
generate adversarial evaluation data is usually lying about one of the two words: the private
halves are written by hand, one conversation at a time, and the generator's job is to keep
them consistent with their public twins and with the corpus. What it automates is the part
that is mechanical — ids, splits, tags, the tool lists, the renderings, the wiring of
expectations to scorers — not the part that requires somebody to have an opinion.

*On the size.* The plan in SPEC §12 says 150 public and 150 private. This is 86 and 86, and
the honest reason is that every task is hand-written data with a `because`: padding the set to
hit a round number would mean writing 64 more facts nobody has an opinion about, which is how
benchmarks get worse as they get bigger. The count is a re-scope, not a shortfall — and the
number that matters for a board is per family, where the smallest cell is 6.

---

## Specification matrix

| Piece | What it is | Where |
| --- | --- | --- |
| **Task model** | id, family, split, input, expectations, `because`, limits, budget, corpus | `src/bench/core/task.py` |
| **Families** | grounded-qa · tool-trajectory · structured-extraction · multi-turn-state · refusal-and-safety · long-horizon | `FAMILIES` in `core/task.py` |
| **Corpus** | markdown → one passage per `##` section, ids a person can open, hashed | `src/bench/core/corpus.py` |
| **Retriever** | BM25 with a hand-written English stopword list, fixed on purpose | `Corpus.search` |
| **Trace contract** | messages · tool calls · retrieved passages · tokens · latency · cost | `src/bench/core/trace.py` |
| **Published schema** | the same contract as JSON Schema, generated, drift-checked by a test | `schema/trace.schema.json` |
| **Budgets** | per task and per run, enforced while running; overspend is a way to lose | `src/bench/core/budget.py` |
| **Scores** | partial credit, flakiness, judge trust, outcomes; never a bare float | `src/bench/core/score.py` |
| **Statistics** | stratified bootstrap over tasks, paired bootstrap for ranks, κ, n-gram leak check | `src/bench/core/stats.py` |
| **Runner** | repeats, concurrency, per-task and per-run budgets, error-as-value | `src/bench/core/runner.py` |
| **Targets** | scripted · http · openai-compatible · cassette (record and replay) | `src/bench/targets/` |
| **Run files** | atomic, never overwritten, four ways to resolve, newest by write time | `src/bench/core/store.py` |
| **Scorers** | thirteen expectation types in six modules, one per thing a check reads | `src/bench/scorers/` |
| **Judge** | cached by what was asked, priced, an outage is an outage and not a zero | `src/bench/judge/client.py` |
| **Calibration** | agreement, κ against hand labels, false passes and false fails, rubric-swap flips | `src/bench/judge/calibration.py` |
| **Task set** | 172 tasks — 86 public + 86 private across six families, generated and drift-checked | `tasks/` · `src/bench/generators/` |
| **Generator data** | facts over the corpus, unanswerable questions, invoices, worlds, conversations, safety pairs, plans | `src/bench/generators/data/` |
| **Set checks** | every phrase checked against its passage, every tool against its world, set byte-compared to its generator | `tests/test_generators.py` |
| **Reports** | terminal · markdown · JUnit · self-contained HTML · JSON bundle | *next* (`bench show` prints a run file) |
| **Board** | aggregation, tie groups, Pareto frontier, one-file site | *next* |
| **Submission** | bundle, verify, re-run, publish | *next* |
| **CLI** | `schema` · `task validate|lint|generate` · `run` · `show` · `judge calibrate|status` | `src/bench/cli.py` |
| **Tests** | 477, offline, no model and no key | `tests/` |
| **Docs** | SPEC (the argument) · README (this) · METHODOLOGY · SCORING · SUBMISSIONS · LIMITS | `docs/` |

---

## Install and run

```bash
make dev                  # pip install -e ".[dev]"
make check                # ruff, mypy, pytest — offline, no key, no network
```

```bash
bench schema print --compact     # the trace contract, for piping into jq
bench task generate --out tasks  # build the task set from the generator data
bench task validate tasks        # load and lint every task in the set
bench task lint tasks            # …and look for near-duplicates across the splits
```

And the whole loop, against the example entrant in `examples/` (in-process, offline,
nothing to configure and nothing to pay for):

```bash
make demo          # runs v1 and v2, then prints the v2 run file
make cassette-demo # records a run once, then replays it offline
```

```
--- the two runs, side by side ---
  entry                         score   passed   cost
  example-support-bot v1        0.667   3/7      $0.000668
  example-support-bot v2        0.333   1/7      $0.000664
  difference                   -0.333   (not a significance test)
```

The cost is real arithmetic rather than a placeholder: the fixture reports tokens, the
target file carries the published rates, and `$0.000668` is what fourteen calls come to.
The same run against an entrant with no rates configured prints `cost not measured` and
stays out of the cost comparison entirely.

The two builds differ by three planted changes — v2 drops the citation, drops the tool
call, and never refuses an unanswerable question — and the score moves by exactly the
tasks those changes touch. That is the whole loop, in-process, offline, with no key:

```
20260930T114029Z-ca4ab2   example-support-bot v2
  started 2026-09-30T11:40:29Z   wall 0.002s   2 repeat(s)   concurrency 8
  7/7 completed   0 errored   0 over budget
  score 0.333 over 7 task(s)   1 passed

  ext-0001  fail  score 0.000
    - 2/2 repeats: the answer is not JSON: I think the rollout was stopped because of the prompt.

  gqa-0001  fail  score 0.333
    - 2/2 repeats: did not cite corpus/incidents.md#2026-08-refunds-stopped (cited: nothing)
    - 2/2 repeats: the trace made no tool calls at all, and the task requires search_docs
```

`bench show --task gqa-0003` prints that task's trace, which for v2 says `I think the
rollout was stopped because of the prompt.` — the planted regression, the same shape as
the incident the corpus is built from. `answerable: false` fails it, with the agent's own
sentence quoted in the note rather than a code.

`bench task validate` is the command somebody runs *while* writing a task, which is why
it exists before the runner does. It refuses a task with no `because`, an unknown
expectation type, an expectation from another family's vocabulary, a private task with
no tags, and a task that appears in both splits under two ids.

---

## What the tests are actually checking

| File | The interesting one |
| --- | --- |
| `test_stats.py` | six wins out of eight is not a win — the interval includes zero, and that is the number this project exists to publish |
| `test_stats.py` | `ngram_overlap` is directional, which is why the duplicate finder takes the max of both directions |
| `test_task.py` | bookkeeping (split, tags, dates) does not move a task's fingerprint; the question and its scoring do |
| `test_score.py` | the aggregate is the mean of family means: 80 tasks at 100% and 12 at 0% is 0.5, not 0.87 |
| `test_budget.py` | `cached_tokens` larger than the prompt is clamped, because billing phantom tokens at the discount rate is a quiet way to win |
| `test_corpus.py` | a question of nothing but stopwords retrieves nothing, which is what makes `answerable: false` checkable |
| `test_trace.py` | the published schema is generated from the models, and a test fails when they drift |
| `test_store.py` | a run file is never overwritten, and `latest` means write time — a name sort puts `-2` *before* the file it was copied from |
| `test_runner.py` | a per-run budget leaves the tasks that never ran as `skipped`, not as zeroes: a partial run must not read like a full one |
| `test_targets.py` | a body with no answer in it is a failure, not a scored zero — and unknown keys go to `raw` instead of failing validation |
| `test_scorers_base.py` | a task nothing could be decided on is an `error`, not a `fail` — and an untrusted judge is excluded from the mean rather than scored zero |
| `test_scorers_families.py` | `{"refunded": 1}` is not `{"refunded": true}`: the equality shortcut in front of the boolean branch made them equal, which is the comment it was written next to |
| `test_judge.py` | a 500 and a timeout are unscored and never cached; κ is 0.0, not 1.0, when a rater never varies, because 0/0 is not agreement |
| `test_generators.py` | the linter's near-duplicate check compares the *ask*, not the environment — it scored the trajectory pairs at 98% until it did, and a deliberate design is not a leak |
| `test_generators.py` | `--public 3` keeps the first three tasks rather than renumbering: `gqa-0007` has to mean the same question in every file that mentions it |

---

## Status, honestly

| Step (SPEC §12) | State |
| --- | --- |
| 1 · repo, package, Makefile, SPEC | done |
| 2 · core: tasks, corpus, trace + schema, budgets, scores, stats | done |
| 2b · runner, target kinds, run files, `bench run` and `bench show` | done |
| 3 · scorers, family by family, and the judge with its calibration | done: 13 expectation types in six modules, the judge with its cache and its κ floor, `bench judge calibrate` |
| 4 · task generators, the two splits, the linter | done: 172 generated tasks, 86 + 86, `bench task generate`, 31 checks over the data |
| 5 · reports and the board | not started |
| 6 · submission tooling and the verifier | not started |
| 7 · docs, demo, tarball, schema drift check against `passmark` | `make demo` and `make cassette-demo` work with scores; tarball and drift check to come |

Nothing in this repository has called a model yet: every target it has run is the
in-process fixture, and the judge has only ever been exercised against a stubbed transport
and canned verdicts. That is deliberate for a repository that has to be testable offline,
and it is also a gap — the first live judge call will happen when somebody calibrates a real
one, which is the first thing the `judge` verbs are for.

What the scorers no longer lack is company: 172 tasks exercise them, and `tests/test_generators.py`
keeps the data and the documents in agreement. What is still missing is everything that turns
a run file into a public number — the reports, the aggregation, the tie groups, the Pareto
frontier — which is step 5, and the submission tooling in step 6.

---

## The portfolio it is built from

The tasks come from real annoyances in the projects this sits beside, not from
invented puzzles: `rag-support-bot` and `synapse` for the retrieval corpus,
`browser-use-agent` for the tool families, `invoice-document-agent` for the
extraction documents, `n8n-ai-executive-assistant` for the workflow-shaped tasks,
`observability-stack` for the traces, and `passmark` for the same trace contract —
`bench export --format passmark-run` will turn a failed benchmark task into a
regression test in somebody's own repository.

---

## Licence

MIT.
