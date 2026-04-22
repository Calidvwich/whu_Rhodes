from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DbConfig:
    path: Path


def connect(cfg: DbConfig) -> sqlite3.Connection:
    cfg.path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(cfg.path.as_posix())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

