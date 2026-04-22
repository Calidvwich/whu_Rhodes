from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np

from ..feature.vector_codec import decode_feature_vector, encode_feature_vector


@dataclass(frozen=True)
class UserRow:
    user_id: int
    username: str
    password: str
    phone: str | None
    email: str | None
    status: int
    role: str


def init_schema(conn: sqlite3.Connection, schema_sql: str) -> None:
    conn.executescript(schema_sql)
    conn.commit()


def insert_user(
    conn: sqlite3.Connection,
    *,
    username: str,
    password_hash: str,
    phone: str | None = None,
    email: str | None = None,
    role: str = "user",
) -> int:
    cur = conn.execute(
        """
        INSERT INTO user (username, password, phone, email, role)
        VALUES (?, ?, ?, ?, ?)
        """,
        (username, password_hash, phone, email, role),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_user_by_username(conn: sqlite3.Connection, username: str) -> UserRow | None:
    row = conn.execute(
        """
        SELECT user_id, username, password, phone, email, status, role
        FROM user
        WHERE username = ?
        """,
        (username,),
    ).fetchone()
    if row is None:
        return None
    return UserRow(
        user_id=int(row["user_id"]),
        username=str(row["username"]),
        password=str(row["password"]),
        phone=row["phone"],
        email=row["email"],
        status=int(row["status"]),
        role=str(row["role"]),
    )


def update_user_status(conn: sqlite3.Connection, user_id: int, status: int) -> None:
    conn.execute("UPDATE user SET status = ? WHERE user_id = ?", (int(status), int(user_id)))
    conn.commit()


def insert_face_feature(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    feature_vector: np.ndarray,
    image_path: str | None = None,
    is_active: int = 1,
) -> int:
    blob = encode_feature_vector(feature_vector)
    cur = conn.execute(
        """
        INSERT INTO face_feature (user_id, feature_vector, is_active, image_path)
        VALUES (?, ?, ?, ?)
        """,
        (int(user_id), sqlite3.Binary(blob), int(is_active), image_path),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_face_features_by_user_id(conn: sqlite3.Connection, user_id: int) -> list[np.ndarray]:
    rows = conn.execute(
        """
        SELECT feature_vector
        FROM face_feature
        WHERE user_id = ? AND is_active = 1
        ORDER BY feature_id ASC
        """,
        (int(user_id),),
    ).fetchall()
    return [decode_feature_vector(row["feature_vector"]) for row in rows]


def iter_all_active_features(conn: sqlite3.Connection) -> Iterable[tuple[int, np.ndarray]]:
    rows = conn.execute(
        """
        SELECT user_id, feature_vector
        FROM face_feature
        WHERE is_active = 1
        """,
    ).fetchall()
    for row in rows:
        yield int(row["user_id"]), decode_feature_vector(row["feature_vector"])


def insert_recognition_log(
    conn: sqlite3.Connection,
    *,
    user_id: int | None,
    input_image_url: str | None,
    similarity: float | None,
    result: int,
    device_info: str | None,
    recognize_time: datetime | None = None,
) -> int:
    recognize_time_str = (recognize_time or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    cur = conn.execute(
        """
        INSERT INTO recognition_log
          (user_id, input_image_url, similarity, result, recognize_time, device_info)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (user_id, input_image_url, similarity, int(result), recognize_time_str, device_info),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_config_by_key(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute(
        "SELECT config_value FROM system_config WHERE config_key = ?",
        (key,),
    ).fetchone()
    return None if row is None else str(row["config_value"])


def update_config(conn: sqlite3.Connection, key: str, value: str, description: str | None = None) -> None:
    conn.execute(
        """
        INSERT INTO system_config (config_key, config_value, description, update_time)
        VALUES (?, ?, ?, datetime('now'))
        ON CONFLICT(config_key) DO UPDATE SET
          config_value = excluded.config_value,
          description = COALESCE(excluded.description, system_config.description),
          update_time = datetime('now')
        """,
        (key, value, description),
    )
    conn.commit()

