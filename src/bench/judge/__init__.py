"""The judge, kept in its own package so the dependency is visible.

`bench.scorers` is pure Python: strings, regexes, arithmetic. One expectation type is not,
and putting the HTTP client here rather than inside the scorers package is what makes that
legible — `grep -r httpx src/bench/scorers` returning nothing is the strongest possible
statement that five of the six families can be re-scored offline, on a plane, in 2049.

What lives here:

* `JudgeConfig` / `JudgeClient` — the call, its cache, and its price.
* `CalibrationReport` / `calibrate` — the comparison against human labels, and the κ that
  decides whether this judge is allowed to decide anything.
* `cohens_kappa` — nine lines, written out rather than imported, because it is the number
  the whole judged share of the board rests on and a reviewer should be able to read it.

Nothing else. The scorer-side glue (`check_judge`) is in `bench.scorers.judged`, and it
talks to this package through the `Judge` protocol, which means the scorer tests never
import this file at all.
"""

from __future__ import annotations

from .calibration import (
    CALIBRATION_NAME,
    FLIP_FLOOR,
    CalibrationReport,
    calibrate,
    calibration_path,
    cohens_kappa,
    read_labels,
    write_calibration,
)
from .client import (
    SYSTEM_PROMPT,
    JudgeClient,
    JudgeConfig,
    JudgeError,
    JudgeVerdict,
    parse_verdict,
    render,
    scaled_prompt,
)

__all__ = [
    "CALIBRATION_NAME",
    "FLIP_FLOOR",
    "SYSTEM_PROMPT",
    "CalibrationReport",
    "JudgeClient",
    "JudgeConfig",
    "JudgeError",
    "JudgeVerdict",
    "calibrate",
    "calibration_path",
    "cohens_kappa",
    "parse_verdict",
    "read_labels",
    "render",
    "scaled_prompt",
    "write_calibration",
]
