"""Print the last two runs in `examples/runs` side by side.

There is no `bench compare` yet — that is the reporting step of SPEC §12 — and a demo that
printed an invented comparison would be worse than one that reads the two files it just
wrote. So this is the store's own API, called from a file rather than from a `-c` one-liner
squeezed into a Make variable, because a make recipe that has to escape an f-string is a
recipe nobody will dare to edit.

It is an example, not a verb: the real comparison, with a paired bootstrap on the
difference, arrives with the board.
"""

from __future__ import annotations

import sys
from pathlib import Path

from bench.core.store import list_runs, load_run


def main(runs_dir: str = "examples/runs") -> int:
    paths = list_runs(Path(runs_dir))[:2]
    if len(paths) < 2:
        print(f"need two runs in {runs_dir}; run `make demo`", file=sys.stderr)
        return 1
    documents = [load_run(path) for path in paths]
    print(f"  {'entry':<28} {'score':>6}   passed   cost")
    for document in reversed(documents):  # oldest first, so the demo reads forwards
        tasks = document["tasks"]
        score = sum(entry["score"] for entry in tasks) / len(tasks)
        passed = sum(1 for entry in tasks if entry["score_outcome"] == "pass")
        cost = document["summary"]["cost_usd"]
        print(
            f"  {document['label']:<28} {score:>6.3f}   {passed}/{len(tasks)}"
            f"      ${cost if cost is not None else 'not measured'}"
        )
    oldest, newest = documents[1], documents[0]
    before = sum(entry["score"] for entry in oldest["tasks"]) / len(oldest["tasks"])
    after = sum(entry["score"] for entry in newest["tasks"]) / len(newest["tasks"])
    print(f"  {'difference':<28} {after - before:>+6.3f}   (not a significance test)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
