import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_ROOT = Path(os.environ.get("GEORISK_ASSETS_ROOT", PROJECT_ROOT.parent)).expanduser().resolve()
