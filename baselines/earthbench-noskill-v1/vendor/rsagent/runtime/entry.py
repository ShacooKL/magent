"""Direct subprocess entry for benchmark workers, including installed packages."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rsagent.runtime.agent import main

if __name__ == "__main__":
    main()
