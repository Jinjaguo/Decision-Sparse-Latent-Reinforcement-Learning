"""Protocol-name entry point for the EXP7 independent cohort audit."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.exp7.audit_demo_availability import main

if __name__ == "__main__":
    raise SystemExit(main())
