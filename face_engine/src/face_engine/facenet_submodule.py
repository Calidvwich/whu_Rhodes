from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBMODULE_ROOT = REPO_ROOT / "facenet_pytorch"


def ensure_facenet_submodule_on_path() -> None:
    if not SUBMODULE_ROOT.exists():
        raise RuntimeError(
            "facenet_pytorch submodule is missing. Run: "
            "git submodule update --init --recursive"
        )
    repo_root_path = REPO_ROOT.as_posix()
    if repo_root_path not in sys.path:
        sys.path.insert(0, repo_root_path)


def import_facenet_symbols() -> tuple[object, object]:
    ensure_facenet_submodule_on_path()
    module = importlib.import_module("facenet_pytorch")
    module_file = Path(getattr(module, "__file__", "")).resolve()
    try:
        module_file.relative_to(SUBMODULE_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(
            "facenet_pytorch was not loaded from the local git submodule. "
            "Please initialize the submodule and avoid relying on a separately installed package."
        ) from exc
    return module.InceptionResnetV1, module.MTCNN


InceptionResnetV1, MTCNN = import_facenet_symbols()

__all__ = [
    "InceptionResnetV1",
    "MTCNN",
    "REPO_ROOT",
    "SUBMODULE_ROOT",
    "ensure_facenet_submodule_on_path",
]
