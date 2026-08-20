.DEFAULT_GOAL := help
SHELL := /bin/bash
PY ?= python3
TASKS ?= tasks

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# --- install ----------------------------------------------------------------

.PHONY: install
install: ## Install the package in editable mode
	$(PY) -m pip install -e .

.PHONY: dev
dev: ## Install everything the checks need
	$(PY) -m pip install -e ".[dev]"

# --- checks -----------------------------------------------------------------

.PHONY: test
test: ## Run the test suite (offline: no model, no network, no key)
	$(PY) -m pytest tests

.PHONY: cover
cover: ## Tests with a coverage report
	$(PY) -m pytest tests --cov=bench --cov-report=term-missing

.PHONY: lint
lint: ## ruff
	$(PY) -m ruff check src tests

.PHONY: fmt
fmt: ## black
	$(PY) -m black -l 100 src tests

.PHONY: types
types: ## mypy
	$(PY) -m mypy src/bench

.PHONY: check
check: lint types test ## Everything CI runs, in the order it runs it

# --- the task set -----------------------------------------------------------

.PHONY: tasks
tasks: ## Regenerate tasks/ from the generator data
	$(PY) -m bench task generate --out $(TASKS)

.PHONY: validate
validate: ## Load and lint every task
	$(PY) -m bench task validate $(TASKS)

.PHONY: lint-tasks
lint-tasks: ## Validate, and look for near-duplicates across the splits
	$(PY) -m bench task lint $(TASKS)

.PHONY: schema
schema: ## Regenerate the published trace schema
	$(PY) -m bench schema write

.PHONY: schema-check
schema-check: schema ## Fail if the published schema was stale
	@git diff --quiet -- schema/trace.schema.json \
		&& echo "schema is current" \
		|| (echo "schema/trace.schema.json was stale and has been regenerated — commit it"; exit 1)

# --- the demo ---------------------------------------------------------------

# Two runs side by side, from `examples/two_runs.py`: the store's own API, in a file,
# because a make recipe that has to escape an f-string is a recipe nobody dares to edit.


.PHONY: demo
demo: ## Run the example entrant twice — v1 and v2 — and print both runs
	@$(PY) -m bench run --tasks examples/tasks --target examples/target.yaml \
		--repeats 2 --runs-dir examples/runs --quiet
	@$(PY) -m bench run --tasks examples/tasks --target examples/target-v2.yaml \
		--repeats 2 --runs-dir examples/runs --quiet
	@echo
	@echo "--- the two runs, side by side ---"
	@$(PY) examples/two_runs.py
	@echo
	@echo "--- the second run (v2), as the harness recorded it ---"
	@$(PY) -m bench show latest --runs-dir examples/runs
	@echo
	@echo "the traces of one task:"
	@$(PY) -m bench show latest --runs-dir examples/runs --task gqa-0003 | head -12

.PHONY: cassette-demo
cassette-demo: ## Record the example run once, then replay it offline
	@$(PY) -m bench run --tasks examples/tasks --target examples/target.yaml \
		--record examples/cassette.json --runs-dir examples/runs --quiet
	@$(PY) -m bench run --tasks examples/tasks --target examples/target.yaml \
		--from-cassette examples/cassette.json --runs-dir examples/runs --quiet
	@echo "recorded and replayed: examples/cassette.json"
	@$(PY) -m bench show latest --runs-dir examples/runs | head -4

# --- packaging --------------------------------------------------------------

.PHONY: package
package: ## Build dist/bench-suite-<version>.tar.gz
	@bash scripts/package.sh

.PHONY: clean
clean: ## Remove caches and build output
	@rm -rf build dist *.egg-info src/*.egg-info \
		.pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	@find . -name __pycache__ -type d -prune -exec rm -rf {} +
