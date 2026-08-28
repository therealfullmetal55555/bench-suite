"""Recording and replaying: the thing that makes a benchmark reproducible and CI cheap.

A cassette is a JSON file of traces, keyed by `task_id/repeat`. Two uses:

* **Record once, replay forever.** Run against the live model once, then re-run the
  scoring, the report and the board offline as often as you like. A submission bundle
  contains its cassettes, so a maintainer re-running a private split pays for retrieval
  and tool calls and nothing else.
* **The only way to compare two entrants fairly when one of them is a model.** The
  tasks' side effects — retrieval, tool calls — are identical for both, so the
  difference the board reports is the difference under test.

The recorder is the replayer. A separate recorder is a recorder that drifts: the second
one to be edited writes a file the first can no longer read, and nobody notices until a
bundle from three months ago fails to verify.

**A missing key is not an error at record time and is one at replay time.** Replaying a
cassette that does not cover the task means the result would be half live and half
recorded, which is the worst of both: expensive and unreproducible. So replay fails the
task with a message naming the task, the repeat, and the cassette's digest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..core.task import Task
from ..core.trace import Trace
from ..errors import TargetError
from .base import Target, TargetConfig

FORMAT_VERSION = 1


class CassetteTarget:
    """Replay traces from a cassette, and record into one when `record_to` is set.

    Both at once, if configured: a recording run replays what the cassette already has
    and calls through for the rest, so an interrupted run can be resumed without paying
    for the tasks it already paid for. That is not an optimisation — it is what makes a
    300-task recording survivable when the fourth task has a typo in it.
    """

    def __init__(self, label: str, config: TargetConfig, *, inner: Target | None = None) -> None:
        self.label = label
        self.config = config
        self._inner = inner
        self.cassette = Path(config.cassette) if config.cassette else None
        self.record_to = Path(config.record_to) if config.record_to else None
        if config.cassette and not Path(config.cassette).exists():
            raise TargetError(f"cassette not found: {config.cassette}")
        if not config.cassette and not config.record_to:
            raise TargetError(
                "a cassette target needs `cassette` (to replay) or `record_to` (to record)",
                hint="record with `bench record --target live.yaml --out cassettes/main.json`",
            )
        self._entries: dict[str, dict[str, Any]] = {}
        self._recorded: dict[str, dict[str, Any]] = {}
        self._meta: dict[str, Any] = {}
        if self.cassette is not None:
            self._load(self.cassette)

    # --- replay ------------------------------------------------------------

    async def run(self, task: Task, *, repeat: int) -> Trace:
        key = _key(task.id, repeat)
        entry = self._entries.get(key)
        if entry is not None:
            trace = Trace.model_validate(entry)
            trace.raw.setdefault("cassette", self.cassette.name if self.cassette else "memory")
            return trace

        if self._inner is not None:
            # Recording mode: what the cassette does not have, the live target answers.
            trace = await self._inner.run(task, repeat=repeat)
            self._recorded[key] = json.loads(trace.model_dump_json())
            return trace

        digest = self.digest()
        return Trace(
            error=(
                f"the cassette has no trace for {key}" + (f" (cassette {digest})" if digest else "")
            ),
            raw={"cassette_miss": key},
        )

    # --- the file ----------------------------------------------------------

    def _load(self, path: Path) -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TargetError(f"{path} is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict) or "entries" not in payload:
            raise TargetError(
                f"{path} is not a cassette",
                hint='a cassette is `{"version": 1, "entries": {"task-id/0": {…trace…}}}`',
            )
        version = payload.get("version")
        if version != FORMAT_VERSION:
            raise TargetError(
                f"{path} is cassette format {version!r}, this harness writes {FORMAT_VERSION}",
                hint="re-record it: a cassette from another version is not safely readable",
            )
        self._entries = payload["entries"]
        self._meta = payload.get("meta", {}) if isinstance(payload.get("meta"), dict) else {}

    def save(self, path: Path | None = None) -> Path | None:
        """Write the recorded entries, merged over whatever the cassette already had.

        Merged, not replaced: a resumed recording must not throw away the first half.
        """
        target = path or self.record_to
        if target is None:
            return None
        merged = {**self._entries, **self._recorded}
        payload = {
            "version": FORMAT_VERSION,
            "meta": {**self._meta, "label": self.label, "entries": len(merged)},
            "entries": merged,
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        # atomic: a half-written cassette from a killed recording is a file that looks
        # like a cassette and fails verification in somebody else's repository
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(target)
        self._entries = merged
        self._recorded = {}
        return target

    def digest(self) -> str:
        """A hash of the entries, recorded in every result that replayed them.

        Without it, "the entrant improved" and "somebody edited the cassette" look the
        same on the board.
        """
        if not self._entries:
            return ""
        blob = json.dumps(self._entries, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    @property
    def recorded(self) -> int:
        return len(self._recorded)

    @property
    def covered(self) -> int:
        return len(self._entries)

    async def aclose(self) -> None:
        if self._inner is not None:
            await self._inner.aclose()
        self.save()


def _key(task_id: str, repeat: int) -> str:
    """`gqa-0042/0`. Readable in a diff, which matters because a cassette is reviewed:
    the interesting question about a recorded run is which tasks it covers."""
    return f"{task_id}/{repeat}"
