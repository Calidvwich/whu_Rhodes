from __future__ import annotations

import sqlite3
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    db_path = root / "data" / "app.sqlite3"
    if not db_path.exists():
        raise SystemExit(f"db not found: {db_path}")

    conn = sqlite3.connect(db_path.as_posix())
    conn.row_factory = sqlite3.Row
    try:
        tables = [
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        ]
        print("db:", db_path)
        print("tables:", tables)

        for t in tables:
            n = conn.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()["n"]
            print(f"count[{t}]={n}")

        row = conn.execute(
            """
            SELECT log_id, user_id, similarity, result, recognize_time, device_info
            FROM recognition_log
            ORDER BY log_id DESC
            LIMIT 1
            """
        ).fetchone()
        print("latest_recognition_log:", dict(row) if row else None)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

