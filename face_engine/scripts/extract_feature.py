from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, (ROOT / "src").as_posix())

from face_engine.extract_feature_cli import main  # noqa: E402


if __name__ == "__main__":
    main()
