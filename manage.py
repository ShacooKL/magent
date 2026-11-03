"""Workspace entrypoint; legacy runtime is an explicit local dependency."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "python-agent" / "src"))
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    from magents.cli import main
    main()
