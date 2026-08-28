"""Calibration: does this judge agree with a person, and by how much on purpose.

An LLM judge is a measuring instrument, and a measuring instrument that nobody has checked
is a story with a decimal point. So the harness refuses to let judge verdicts decide a
score until they have been compared against human labels — and the comparison is a file in
the repository, not a claim in a README.

The file is a JSONL of hand-labelled answers:

    {"case_id": "gqa-0001", "rubric": "...", "prompt": "...", "output": "...", "human": true}

and the report that comes out of it carries four numbers, each of which is a different
kind of bad news:

* **agreement** — the crude rate. Easy to read, and easy to be fooled by: a judge that
  says "pass" to everything agrees 85% of the time on a set that is 85% passes.
* **Cohen's κ** — agreement corrected for what chance would produce, which is the number
  the floor applies to. 0.70 is "substantial" on the usual reading and is deliberately
  strict: below it, the judge is laundered coin-flip, which is worse than no judge.
* **false passes and false fails, separately** — because they cost different things. A
  false pass lets a broken entry through a gate; a false fail blocks a working one. A
  benchmark that only reported agreement would let a judge with a 50/50 split of errors
  look identical to one that is wrong in one direction.
* **flips under a reordered rubric** — the verdict is not allowed to depend on which
  requirement the judge read first. `bench judge check` runs each label both ways; the
  flip rate is in the file and above 10% the judge is reported as untrustworthy whatever
  its κ.

The calibration is written next to the judge file (or into the cache directory), and
`_read_calibration` refuses one whose `model` is not the judge's: a κ that describes a
different model is not evidence about this one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import BenchError

CALIBRATION_NAME = "judge-calibration.json"
FLIP_FLOOR = 0.10
"""Above this share of verdicts flipping under a reordered rubric, the judge is reported
as untrustworthy. One in ten is already generous; the probe exists to catch a judge whose
verdict is an artefact of phrasing, and those are not subtle."""


def cohens_kappa(pairs: list[tuple[bool, bool]]) -> float:
    """Cohen's κ for two raters on a binary scale.

    Written out rather than imported: it is nine lines, the formula is the thing being
    relied on, and a reviewer should be able to check it without following a dependency.
    Returns 0.0 when one rater never varies — the observed agreement is 1.0 and the
    expected agreement is also 1.0, so κ is 0/0, and reporting 1.0 there would let a judge
    that passes everything look perfect on a set that is all passes.
    """
    if not pairs:
        raise ValueError("kappa of nothing is undefined")
    n = len(pairs)
    observed = sum(1 for left, right in pairs if left == right) / n
    left_true = sum(1 for left, _ in pairs if left) / n
    right_true = sum(1 for _, right in pairs if right) / n
    expected = left_true * right_true + (1 - left_true) * (1 - right_true)
    if expected >= 1.0:
        return 0.0
    return (observed - expected) / (1 - expected)


@dataclass
class CalibrationReport:
    model: str
    labels: int
    agreement: float
    kappa: float
    false_pass: int
    false_fail: int
    unanswered: int
    """Labels the judge did not answer at all. Not counted as failures — an outage is not
    a disagreement — but reported, because a calibration with 40 unanswered labels out of
    50 has not measured anything."""

    flips: int = 0
    calls: int = 0
    cache_hits: int = 0
    cost_usd: str | None = None
    temperature: float = 0.0
    labels_file: str = ""
    floor: float = 0.70
    wrong_ids: list[str] = field(default_factory=list)

    @property
    def flip_rate(self) -> float:
        return self.flips / self.labels if self.labels else 0.0

    @property
    def usable(self) -> bool:
        return (
            self.labels >= 20
            and self.kappa >= self.floor
            and self.flip_rate <= FLIP_FLOOR
            and self.unanswered == 0
        )

    def summary(self) -> str:
        if self.labels == 0:
            return "no labels: nothing was calibrated"
        if self.unanswered == self.labels:
            return (
                f"none of the {self.labels} labels could be judged — the judge is not "
                f"answering, so this is not a disagreement about the answers"
            )
        verdict = "usable" if self.usable else "not usable"
        reason = ""
        if not self.usable:
            if self.labels < 20:
                reason = f"; {self.labels} labels is below the 20 needed to mean anything"
            elif self.kappa < self.floor:
                reason = f"; κ {self.kappa:.2f} is below the floor {self.floor:.2f}"
            elif self.flip_rate > FLIP_FLOOR:
                reason = f"; {self.flip_rate:.0%} of verdicts flipped under a reordered rubric"
            else:
                reason = f"; {self.unanswered} labels were not judged"
        return (
            f"agreement {self.agreement:.2f}, κ {self.kappa:.2f} ({verdict}{reason}); "
            f"{self.false_pass} false pass(es), {self.false_fail} false fail(s)"
        )

    def as_document(self) -> dict[str, Any]:
        return {
            "format": 1,
            "model": self.model,
            "temperature": self.temperature,
            "labels": self.labels,
            "agreement": round(self.agreement, 4),
            "kappa": round(self.kappa, 6),
            "kappa_floor": self.floor,
            "false_pass": self.false_pass,
            "false_fail": self.false_fail,
            "unanswered": self.unanswered,
            "flips": self.flips,
            "flip_rate": round(self.flip_rate, 4),
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "cost_usd": self.cost_usd,
            "labels_file": self.labels_file,
            "usable": self.usable,
            "wrong_ids": self.wrong_ids[:50],
        }


def read_labels(path: str | Path) -> list[dict[str, Any]]:
    """Read the JSONL, refusing the lines that would make a κ meaningless.

    A label without a `human` verdict, or with a rubric but no answer, is a row that
    cannot be compared — and silently skipping it is how a calibration ends up reporting
    agreement over four of the fifty rows somebody thought they had written.
    """
    file = Path(path)
    if not file.exists():
        raise BenchError(f"no labels file at {file}")
    labels: list[dict[str, Any]] = []
    problems: list[str] = []
    for number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            problems.append(f"line {number}: not JSON ({exc})")
            continue
        missing = [key for key in ("case_id", "rubric", "output", "human") if key not in row]
        if missing:
            problems.append(f"line {number}: missing {', '.join(missing)}")
            continue
        if not isinstance(row["human"], bool):
            problems.append(f"line {number}: `human` has to be true or false")
            continue
        labels.append(row)
    if problems:
        raise BenchError(
            f"{file} has {len(problems)} unusable line(s):\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )
    if not labels:
        raise BenchError(f"{file} has no labels in it")
    return labels


def calibrate(
    judge: Any,
    labels: list[dict[str, Any]],
    *,
    labels_file: str = "",
    check_flips: bool = True,
) -> CalibrationReport:
    """Compare the judge against the labels, and against itself reordered.

    The client is passed in rather than constructed here, so this is testable with a
    dictionary of canned verdicts — and so that the caller decides which model and key the
    calibration is about, which is the difference between a calibration that describes the
    board and one that describes somebody's laptop.
    """
    pairs: list[tuple[bool, bool]] = []
    false_pass = false_fail = unanswered = flips = 0
    wrong_ids: list[str] = []
    for row in labels:
        verdict = judge.verdict(
            rubric=str(row["rubric"]),
            prompt=str(row.get("prompt", "")),
            output=str(row["output"]),
            task_id=str(row["case_id"]),
        )
        if not getattr(verdict, "scored", False):
            unanswered += 1
            continue
        human = bool(row["human"])
        judged = bool(verdict.passed)
        pairs.append((human, judged))
        if judged == human:
            pass
        elif judged and not human:
            false_pass += 1
            wrong_ids.append(str(row["case_id"]))
        else:
            false_fail += 1
            wrong_ids.append(str(row["case_id"]))
        if check_flips and isinstance(row.get("prompt"), str):
            flipped = judge.verdict(
                rubric=_swapped(str(row["rubric"])),
                prompt=str(row["prompt"]),
                output=str(row["output"]),
                task_id=str(row["case_id"]),
            )
            if getattr(flipped, "scored", False) and bool(flipped.passed) != judged:
                flips += 1

    agreement = sum(1 for human, judged in pairs if human == judged) / len(pairs) if pairs else 0.0
    kappa = cohens_kappa(pairs) if pairs else 0.0
    config = getattr(judge, "config", None)
    return CalibrationReport(
        model=getattr(config, "model", "unknown"),
        labels=len(labels),
        agreement=agreement,
        kappa=kappa,
        false_pass=false_pass,
        false_fail=false_fail,
        unanswered=unanswered,
        flips=flips,
        calls=getattr(judge, "calls", 0),
        cache_hits=getattr(judge, "cache_hits", 0),
        cost_usd=str(judge.spent_usd) if hasattr(judge, "spent_usd") else None,
        temperature=float(getattr(config, "temperature", 0.0)),
        labels_file=labels_file,
        floor=float(getattr(config, "kappa_floor", 0.70)),
        wrong_ids=wrong_ids,
    )


def _swapped(rubric: str) -> str:
    from .client import _swap_halves

    return _swap_halves(rubric)


def calibration_path(config: Any) -> Path:
    """Where a judge's calibration lives: beside its cache, or beside the judge file.

    One rule, so that `trusted` and `bench judge calibrate` cannot disagree about which
    file they mean — the failure mode being a calibration that is written somewhere and
    read from nowhere, which reads on the board as "the judge was never calibrated".
    """
    cache = getattr(config, "cache_dir", None)
    if cache:
        return Path(cache) / CALIBRATION_NAME
    return Path(CALIBRATION_NAME)


def write_calibration(report: CalibrationReport, config: Any) -> Path:
    path = calibration_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(report.as_document(), indent=2, sort_keys=False), encoding="utf-8"
    )
    temporary.replace(path)
    return path
