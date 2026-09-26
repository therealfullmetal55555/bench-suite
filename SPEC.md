# bench-suite — specification

A public benchmark suite for retrieval-augmented and tool-using agents, with a
leaderboard whose numbers come with error bars, a cost column, and a list of the
tasks the entrant failed.

Version 0.1.0 · 2026-09-30 · MIT

---

## 1. Why this exists, and why it is not passmark

`passmark` answers *did my change make my agent worse?* — the same agent, my cases,
before and after. This answers a different question: *which of these agents is
better at this kind of work, and what does better cost?* Different question,
different incentives, different failure modes, and it deserves its own repository
rather than a flag on the first one.

**What is wrong with the benchmarks we have.** They are usually one number. A
score of 71.4% with no interval, no cost, no latency, and a single run. Two
entrants three places apart on the board are usually indistinguishable in the
statistical sense, and the board does not say so, so people read the rank as the
truth. Then the tasks leak into training data, the number goes up, and the
benchmark quietly stops measuring anything.

**What this one does about it:**

- **Every entry carries a confidence interval.** Results are re-run; the reported
  number is the mean with a stratified bootstrap interval over tasks. If your
  interval overlaps the entry above you, the leaderboard prints "tied" instead of
  sorting you.
- **Cost and latency are columns, not an afterthought.** A leaderboard that
  ignores the price of an answer ranks a model that costs a hundred times more as
  simply "better". Here a Pareto view is the default: quality against dollars per
  task, with the frontier drawn.
- **The judge is audited on the board.** Judge-scored families carry Cohen's κ
  against human labels; a family where the judge is below the floor is shown as
  unscored rather than scored badly, and cannot decide a rank.
- **Stability is measured.** Each task runs `repeats` times. An agent that passes
  the same task 3 times out of 5 gets a variance penalty and a `flaky` flag,
  because a build you cannot trust is not the same product as a build that is
  merely worse.
- **Public split and private split.** Tasks are published so people can develop
  against them; the score that counts is computed on a held-back split the same
  size, re-run by the maintainers' CI. Contamination checks (n-gram overlap
  against the entrant's declared training data, where it is declared) are reported
  as a column, not as a sanction.
- **Task retirement is policy, written down.** Each family has a retirement
  criterion (everyone above 95%, or the task's answer is in the wild) and the
  history of retired tasks is kept in the repository. A benchmark that only ever
  grows measures memorisation.

**What it is not.** Not a model comparison in the abstract — the entrant is a
*configuration*: model, prompt, retrieval setup, tools, and the harness version.
Not a leaderboard of vendors: anyone can submit, including me, and my own
portfolio entries are marked as such. Not a place to submit without running:
entries are re-run from a fresh clone by CI, and a submission that does not
reproduce is not published.

---

## 2. The task families

Six, chosen because between them they cover what agents are actually asked to do
and because each one has a *different* right way to score it. A family is a
directory with tasks, a scorer, and a rubric where needed.

| Family | What it exercises | How it is scored |
|---|---|---|
| `grounded-qa` | answering from a retrieval corpus, with citations | every claim must trace to a retrieved passage; citation precision/recall plus a judged answerable/unanswerable check |
| `tool-trajectory` | choosing and ordering tools | required/forbidden tools, step budget, argument correctness — deterministic, no judge |
| `structured-extraction` | turning messy documents into a schema | exact JSON schema validation and per-field comparison, with weights per field |
| `multi-turn-state` | holding state across turns, correcting itself | final state plus the conversation's constraints — deterministic plus a judged coherence check |
| `refusal-and-safety` | not doing the harmful or impossible thing, and not over-refusing | two-sided: a refusal on a benign request is a fail, and so is compliance with a harmful one; prompt-injection subfamily |
| `long-horizon` | plan steps that only pay off after five or six turns | judged outcome plus trajectory checks; the family where the judge is most load-bearing, so it is reported with κ |

The tasks come from real annoyances rather than invented puzzles. `grounded-qa`
uses a corpus assembled from the documentation and issue history of the projects
in this portfolio — `rag-support-bot`, `synapse`, `n8n-ai-executive-assistant`.
`structured-extraction` uses invoice-shaped
documents, borrowed in spirit from `invoice-document-agent`. `tool-trajectory`'s
tools are the ones `browser-use-agent` and `observability-stack` actually expose:
query a log, list a run's spans, click a thing, stop. `refusal-and-safety` is
built from the incident write-ups that will live in the `incident-postmortem`
project, with the personal details removed.

A task looks like this:

```yaml
id: gqa-0042
family: grounded-qa
split: public
input:
  question: "Why did the v2 rollout of the support bot stop refunding orders?"
  corpus: corpus/rag-support-bot        # a directory of passages, hashed
expectations:
  - type: citations
    must_cite: [passage/incident-2026-08.md#root-cause]
    must_not_cite_unsupported: true
  - type: answerable
    value: true
limits:
  max_steps: 8
budget: {max_usd: 0.05, max_seconds: 60}
because: |
  The failure was a permission change, not a prompt change, and an agent that
  blames the prompt is wrong about the cause even when its answer looks right.
```

Every task has a `because`. A benchmark task without a reason to exist is a
puzzle, and puzzles measure puzzle-solving.

---

## 3. The harness

```
src/bench/
  core/       tasks, corpus, the trace contract, budgets, run files
  targets/    the entrant: openai-compatible, http, scripted, cassette
  runners/    one per family, because the families differ in how they drive a model
  scorers/    deterministic, citation, schema, judge
  judge/      the client, the cache, and the calibration against human labels
  report/     terminal, markdown, JUnit, self-contained HTML, JSON
  board/      leaderboard: aggregation, ties, Pareto frontier, site generation
  submit/     the submission bundle: run, sign, verify, re-run
```

Python 3.12, `httpx` + `pydantic` + `pyyaml` + `jsonschema`. No framework, no
database, no service. `bench run --task gqa-0042 --target ./target.yaml` works from
a checkout; `bench board build` regenerates the site from `results/*.json`.

**The trace contract.** The same shape passmark consumes — messages, tool calls,
retrieved passages, tokens, latency, cost — published as a JSON Schema at
`schema/trace.schema.json`. That is deliberate: a task that fails here can be
dropped into a team's own `passmark` dataset with the expectations intact, and
`bench export --format passmark-run` produces a file passmark can read. The CI
checks that export against passmark's loader, so the two projects cannot drift
apart silently.

**Cassettes for everything but the model.** Retrieval, tool calls and any other
side effect is recorded once and replayed, so a re-run costs model tokens and
nothing else, and the only thing that varies between two entrants is the model
under test. `bench run --offline` replays the model too, for developing tasks.

**Budgets are enforced, not reported.** `budget.max_usd` and `max_seconds` are per
task. A run that exceeds either is stopped and the task is scored `over-budget`,
which is a legitimate way to lose: an agent that answers correctly after 400
seconds or four dollars has not answered.

---

## 4. Scoring, and how it is aggregated

Per family, a score in [0, 1] and the number of tasks it rests on. The aggregate is
a **stratified mean** — the mean of family means, not the mean of all tasks — so a
family with eighty tasks cannot outvote a family with twelve. Both are printed.

Uncertainty comes from a paired bootstrap over tasks within each family, resampled
seed-fixed, reported as a 95% interval. Entries are compared pairwise: a difference
is only claimed when the interval on the *paired* difference excludes zero. Ranks
are printed with `tied` groups, and the board's sort key is quality, with cost and
latency as tie-breakers in that order.

Three things are printed that leaderboards usually hide:

- **The failures.** Every entry's report card lists the task ids it failed, with a
  link to the transcript. A score without the failures is an advertisement.
- **The cost of the score.** Dollars per task, tokens per task, p95 latency, and
  the share of tasks that hit a budget.
- **The flakiness.** Share of tasks whose outcome differed across repeats, with the
  ids. An entry with a flaky share above 10% is published with an asterisk, because
  its score is a coin-flip average.

The judge's own audit is a row in the same table: agreement, κ, and how many tasks
it decided.

---

## 5. The leaderboard

Not a website with a database: a directory of results and a generator.

```
results/
  2026-10-01T09-14-22Z__openai-gpt-5.1__retrieval-v3.json     # the bundle
  2026-10-01T09-14-22Z__openai-gpt-5.1__retrieval-v3.card.md   # the report card
leaderboard.json      # generated: entries, intervals, ties, frontier, retired tasks
site/index.html       # generated: one self-contained file, no CDN, no scripts
```

`bench board build` reads `results/`, validates every bundle against the harness
version it claims, recomputes the aggregate from the per-task results rather than
trusting the bundle's summary, and writes the JSON and the site. The site is one
HTML file with inline CSS: it opens from a file system, it renders in ten years,
and nobody has to run npm to look at a number.

**Submission.** `bench submit` runs the public split, writes the bundle (per-task
results, transcripts, cassettes digests, harness git sha, image digest, cost), and
prints a PR body. The maintainers' CI re-runs the private split on the frozen
harness in a fresh container, and only then does the entry appear. `bench verify
<bundle>` re-runs the public split from the bundle alone — the entrant can check
their own submission the way the board will, and a reader can check an entry years
later without asking anybody.

---

## 6. Anti-gaming, in the order the problems occur

1. **Overfitting to the public split.** A private split of the same size, drawn
   from the same generators, re-run by the maintainers. The gap between public and
   private is printed as a column — an entrant with 20 points of gap is not banned,
   they are visible.
2. **Contamination.** Task text and expected answers are checked with n-gram
   overlap against any training data an entrant declares, and against the contents
   of the public web cache for the corpus where one exists. Reported as a column.
3. **Judge shopping.** The judge model and prompt are fixed by the harness, the
   calibration is published, and an entrant cannot supply their own judge.
4. **Run shopping.** Everything is `repeats`-times and published with the flaky
   share; re-running until it looks good raises the flaky column, which is on the
   board.
5. **Harness edits.** An entrant submits a bundle, not a diff; the maintainers run
   the harness at the version the bundle declares, from their own image.

None of this makes gaming impossible. It makes gaming *legible*, which is the most
a benchmark can honestly offer.

---

## 7. Costs, and who pays

A full public-split run of the six families is 300 tasks × 3 repeats = 900 model
calls, plus roughly 200 judge calls on the judged families. At the prices of a
mid-tier model that is a few dollars; at the top of the current range it is a few
hundred. Three decisions follow from that:

- **Cheap by default.** `bench run` defaults to the public split with
  `repeats: 1` and says loudly that a single repeat cannot measure stability.
- **The judge is cached by argument hash** — rubric, model, temperature, prompt and
  output — so re-running the same submission to check a harness change costs no
  judge calls.
- **`--limit N` and `--family`** for iterating, and the bundle records that it was
  limited, so a partial run can never be mistaken for a full one.

CI runs offline against recorded model output (`--offline`), so the project's own
pipeline costs nothing and can be run by anybody who forks it.

---

## 8. What ships in 0.1.0

| Piece | Detail |
|---|---|
| Tasks | 172 across six families: 86 public, 86 private, generated from checked-in data so the split is auditable. The 0.1.0 figure of 300 was a guess made before the data existed; every task here carries a `because`, and the padding that would close the gap is padding. The per-family cells (6 to 40) are the numbers the board publishes, and they grow family by family |
| Harness | `run`, `score`, `report`, `board build`, `submit`, `verify`, `task validate`, `task lint`, `export` |
| Scorers | citation precision/recall, JSON schema with per-field weights, trajectory, refusal two-sided, judge with calibration |
| Judge | OpenAI-compatible client, argument-hash cache, κ calibration against 200 human labels per judged family |
| Reports | terminal, markdown, JUnit, self-contained HTML, JSON bundle, report card |
| Board | `leaderboard.json` + one-file `site/index.html` with the Pareto view, tie groups, retired tasks |
| CLI | exit codes as the interface: 0 clean, 1 a task worsened, 2 the run failed, 3 a result that cannot be published |
| Ops | Dockerfile (non-root, `--offline` capable), CI (lint, types, tests, offline board build, schema drift check against passmark), package tarball |
| Docs | README (Specification Matrix), SPEC (this), METHODOLOGY, SCORING, SUBMISSIONS, LIMITS, DEMO with real numbers |
| Tests | every scorer, the aggregator's tie and bootstrap maths, the board generator, the submission verifier, the task linter, and a fixture league of four fake entrants that produces a known-correct board |

---

## 9. Methodology, written before the first entry

A leaderboard is a claim about the world, and the claim has to be falsifiable. So
the methodology document is part of 0.1.0, not of some later documentation pass,
and it commits to:

- The exact resampling procedure, seeds, and the fact that the interval is over
  tasks, not over repeats. Repeats estimate flakiness; tasks estimate uncertainty.
- That a difference smaller than the interval is called a tie, in the data and in
  the site's sort order.
- That judge-scored families below the κ floor are excluded from the aggregate and
  listed separately, and how much of the total that was.
- The four retirement criteria and the procedure for retiring a task (with the
  history kept in git).
- What the numbers do not mean: they are about these tasks, in this harness
  version, at these prices, in this week.

---

## 10. The portfolio, as the corpus and as the entrants

- `rag-support-bot` — a retrieval corpus and a baseline entrant; its `/chat`
  endpoint is the first external target the harness ever runs against.
- `browser-use-agent` — the tools behind `tool-trajectory`, and the reason step
  budgets are enforced rather than reported.
- `invoice-document-agent` — the document shapes for `structured-extraction`.
- `n8n-ai-executive-assistant` — the workflow-shaped tasks, where the correct answer
  is usually "ask a question first".
- `observability-stack` — the harness emits its own traces in the same OTLP shape,
  so a benchmark run shows up in the same dashboards as production traffic, with
  `bench.family` and `bench.entry` as span attributes.
- `passmark` — the same trace schema, and `bench export --format passmark-run`, so a
  failing benchmark task becomes a regression test in someone's repository.
- `incident-postmortem` — the source of the `refusal-and-safety` tasks: real
  incidents, personal details removed, and an explicit note about what was redacted.

---

## 11. Deliberately out of scope for 0.1.0

- **Hosted submissions and accounts.** `bench submit` produces a bundle and a PR
  body. A server is a company, not a chunk of a repository.
- **Vision, audio, and computer-use benchmarks.** Different inputs, different
  scoring, and pretending otherwise would mean scoring screenshots with a template
  matcher and calling it a metric.
- **Training-data decontamination at scale.** We check what an entrant declares and
  what a corpus cache shows. Anything stronger requires the training set, which
  nobody hands over.
- **A single "overall" number.** The stratified mean is aggregate enough. A single
  headline figure is what made the benchmarks we have stop being believed.
- **Automatic task generation from the web.** Tasks are written and reviewed by
  people, and the generator scripts are only for expanding a family along
  parameters a person chose. In practice that means the generator writes ids,
  splits, tags, tool lists and renderings, and a person writes the questions: the
  private halves of every family are written by hand, and the generator's job is to
  keep them consistent with their public twins and with the corpus.

---

## 12. Delivery plan

Each step ends with something runnable and tested, not with a folder of stubs.

1. `SPEC.md` (this), `README.md` skeleton, package layout, `Makefile`, `Dockerfile`.
2. Core: the task model, the corpus, the trace contract and its schema, budgets, run
   files, the target kinds, and the runner. Tests first for the schema and budgets.
3. Scorers, one family at a time, each with its own test file; the judge client and
   the calibration against hand labels.
4. Task generators and the two splits, with the linter that enforces `because` and
   the family schema. Done: 172 tasks from `src/bench/generators/data/`, `bench task
   generate`, and a test file that checks every generated claim against the document
   it is attributed to.
5. Reporting: terminal, markdown, JUnit, HTML; then the board aggregator with the
   tie logic and the Pareto frontier, tested against a fixture league with a
   known-correct outcome.
6. Submission tooling and the verifier, then the offline CI board build.
7. Docs with real numbers, the demo walkthrough, the package tarball, and the
   schema-drift check against `passmark`.
