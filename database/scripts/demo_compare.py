from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, ROOT.as_posix())

import numpy as np

from src.db.connection import DbConfig, connect
from src.db.dao import (
    get_user_by_username,
    get_config_by_key,
    insert_face_feature,
    insert_recognition_log,
    insert_user,
    iter_all_active_features,
)
from src.feature.matcher import match_best
from src.feature.vector_codec import VECTOR_DIM, l2_normalize


def main() -> None:
    db_path = ROOT / "data" / "app.sqlite3"

    conn = connect(DbConfig(path=db_path))
    try:
        # 1) create (or reuse) a user
        existing = get_user_by_username(conn, "test_user")
        user_id = existing.user_id if existing else insert_user(
            conn, username="test_user", password_hash="dummy_hash", role="user"
        )

        # 2) insert a few random features for that user
        rng = np.random.default_rng(42)
        base = l2_normalize(rng.normal(size=(VECTOR_DIM,)).astype(np.float32))
        for _ in range(5):
            noise = rng.normal(scale=0.02, size=(VECTOR_DIM,)).astype(np.float32)
            insert_face_feature(conn, user_id=user_id, feature_vector=l2_normalize(base + noise))

        # 3) build an input vector close to that user
        input_vec = l2_normalize(base + rng.normal(scale=0.01, size=(VECTOR_DIM,)).astype(np.float32))

        threshold = float(get_config_by_key(conn, "threshold") or "0.65")
        lib = list(iter_all_active_features(conn))
        result = match_best(input_vec, lib, threshold=threshold)

        insert_recognition_log(
            conn,
            user_id=result.matched_user_id,
            input_image_url=None,
            similarity=result.max_similarity,
            result=1 if result.matched else 0,
            device_info="demo",
        )

        print(
            "match_result:",
            {
                "threshold": threshold,
                "matched_user_id": result.matched_user_id,
                "max_similarity": round(result.max_similarity, 4),
                "matched": result.matched,
                "library_size": len(lib),
            },
        )
        print("OK: wrote recognition_log")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

