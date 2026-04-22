from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, ROOT.as_posix())

from src.db.connection import DbConfig, connect  # noqa: E402
from src.db.dao import (  # noqa: E402
    get_config_by_key,
    get_user_by_username,
    insert_face_feature,
    insert_recognition_log,
    insert_user,
    iter_all_active_features,
)
from src.feature.matcher import match_best  # noqa: E402
from src.feature.vector_codec import VECTOR_DIM, l2_normalize  # noqa: E402


def _load_vector_json(path: Path) -> np.ndarray:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "vector" in obj:
        obj = obj["vector"]
    arr = np.asarray(obj, dtype=np.float32)
    if arr.shape != (VECTOR_DIM,):
        raise SystemExit(f"vector shape must be ({VECTOR_DIM},), got {arr.shape} from {path}")
    return arr


def cmd_enroll(args: argparse.Namespace) -> None:
    db_path = ROOT / "data" / "app.sqlite3"
    vec = l2_normalize(_load_vector_json(Path(args.vector_json)))

    conn = connect(DbConfig(path=db_path))
    try:
        existing = get_user_by_username(conn, args.username)
        user_id = existing.user_id if existing else insert_user(
            conn, username=args.username, password_hash=args.password_hash, role=args.role
        )
        feature_id = insert_face_feature(conn, user_id=user_id, feature_vector=vec, image_path=args.image_path)
        print(
            "enroll_result:",
            {"user_id": user_id, "username": args.username, "feature_id": feature_id, "image_path": args.image_path},
        )
    finally:
        conn.close()


def cmd_recognize(args: argparse.Namespace) -> None:
    db_path = ROOT / "data" / "app.sqlite3"
    vec = l2_normalize(_load_vector_json(Path(args.vector_json)))

    conn = connect(DbConfig(path=db_path))
    try:
        threshold = float(get_config_by_key(conn, "threshold") or "0.80")
        lib = list(iter_all_active_features(conn))
        result = match_best(vec, lib, threshold=threshold)

        log_id = insert_recognition_log(
            conn,
            user_id=result.matched_user_id,
            input_image_url=args.input_image_url,
            similarity=result.max_similarity,
            result=1 if result.matched else 0,
            device_info=args.device_info,
        )
        print(
            "recognize_result:",
            {
                "threshold": threshold,
                "matched_user_id": result.matched_user_id,
                "max_similarity": round(result.max_similarity, 6),
                "matched": result.matched,
                "library_size": len(lib),
                "log_id": log_id,
            },
        )
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Enroll/recognize using 512-d feature vector JSON file.")
    sub = p.add_subparsers(dest="cmd", required=True)

    enroll = sub.add_parser("enroll", help="insert a feature vector into face_feature")
    enroll.add_argument("--username", required=True)
    enroll.add_argument("--password-hash", default="dummy_hash")
    enroll.add_argument("--role", default="user", choices=["user", "admin"])
    enroll.add_argument("--vector-json", required=True, help="path to JSON containing 512 floats")
    enroll.add_argument("--image-path", default=None)
    enroll.set_defaults(func=cmd_enroll)

    rec = sub.add_parser("recognize", help="match input vector against library and write recognition_log")
    rec.add_argument("--vector-json", required=True)
    rec.add_argument("--device-info", default="cli")
    rec.add_argument("--input-image-url", default=None)
    rec.set_defaults(func=cmd_recognize)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

