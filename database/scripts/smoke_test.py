from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, ROOT.as_posix())

from src.service import FaceRecService, to_dict


def _unit_vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=(512,)).astype(np.float32)
    return vector / np.linalg.norm(vector)


def _run(db_path: Path) -> None:
    service = FaceRecService(db_path)
    service.init_db()

    enrolled = service.enroll("smoke_alice", _unit_vector(1), image_path="smoke/alice.jpg")

    hit = service.recognize(_unit_vector(1), device_info="smoke-test", input_image_url="smoke/hit.jpg")
    stranger = service.recognize(-_unit_vector(1), device_info="smoke-test", input_image_url="smoke/stranger.jpg")

    logs_by_user = service.get_logs_by_user(enrolled.user_id)
    logs_by_time = service.get_logs_by_time_range("1970-01-01 00:00:00", "2999-12-31 23:59:59")
    logs_by_device = service.get_logs_by_device_info("smoke-test")
    deactivated_count = service.deactivate_features_by_user(enrolled.user_id)
    after_deactivate = service.recognize(_unit_vector(1), device_info="smoke-test-after-deactivate")

    checks = [
        ("enroll returns ids", enrolled.user_id > 0 and enrolled.feature_id > 0),
        ("same vector matches", hit.matched and hit.matched_user_id == enrolled.user_id),
        ("opposite vector rejects", not stranger.matched),
        ("logs by user available", len(logs_by_user) >= 1),
        ("logs by time range available", len(logs_by_time) >= 2),
        ("logs by device available", len(logs_by_device) >= 2),
        ("deactivate user features", deactivated_count >= 1),
        ("deactivated feature no longer matches", not after_deactivate.matched),
    ]

    for name, ok in checks:
        print(f"{'OK' if ok else 'FAIL'}: {name}")

    if not all(ok for _, ok in checks):
        raise SystemExit(1)

    print("summary:", {"enrolled": to_dict(enrolled), "hit": to_dict(hit), "stranger": to_dict(stranger)})
    print(f"OK: smoke test database at {db_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run database integration smoke test.")
    parser.add_argument("--db-path", type=Path, help="SQLite db path. Defaults to a temporary database.")
    args = parser.parse_args()

    if args.db_path is not None:
        _run(args.db_path)
        return

    with tempfile.TemporaryDirectory(prefix="whu_rhodes_db_smoke_") as tmp:
        _run(Path(tmp) / "app.sqlite3")


if __name__ == "__main__":
    main()
