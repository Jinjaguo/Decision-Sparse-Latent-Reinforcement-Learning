#!/usr/bin/env python
"""EXP2 R5 entry point; refuses to run without a passing formal R4 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero-twin-run", type=Path, required=True)
    args = parser.parse_args()
    metrics = json.loads((args.zero_twin_run / "metrics.json").read_text(encoding="utf-8"))
    if not metrics.get("gate", {}).get("passed", False):
        raise RuntimeError("R5 is forbidden because the supplied R4 zero-twin gate did not pass")
    raise NotImplementedError("R5 implementation is added only after the formal R4 gate legally passes")


if __name__ == "__main__":
    raise SystemExit(main())
