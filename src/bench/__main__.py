"""`python -m bench`. Kept as a one-liner: the console script is the documented way
in, and this exists because the first thing anybody tries is the module form."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
