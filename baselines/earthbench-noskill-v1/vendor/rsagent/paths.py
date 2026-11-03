"""Checkout locations shared by CLI defaults; independent of module nesting."""
import json
from pathlib import Path

REPOSITORY_ROOT = next(
    (parent for parent in Path(__file__).resolve().parents
     if (parent / "pyproject.toml").is_file()), Path.cwd()
)
_snapshot = REPOSITORY_ROOT / "snapshot.json"
WORKSPACE_ROOT = (Path(json.loads(_snapshot.read_text())["workspace_root"])
                  if _snapshot.is_file() else REPOSITORY_ROOT.parent)
VAR_ROOT = REPOSITORY_ROOT / "var"
