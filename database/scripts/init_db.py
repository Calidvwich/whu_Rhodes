from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, ROOT.as_posix())

from src.db.connection import DbConfig, connect
from src.db.dao import init_schema


def main() -> None:
    db_path = ROOT / "data" / "app.sqlite3"
    schema_path = ROOT / "src" / "db" / "schema.sql"

    schema_sql = schema_path.read_text(encoding="utf-8")
    conn = connect(DbConfig(path=db_path))
    try:
        init_schema(conn, schema_sql)
        print(f"OK: initialized database at {db_path}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

