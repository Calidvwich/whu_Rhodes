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


@dataclass(frozen=True)
class RecognitionLogRow:
    log_id: int
    user_id: int | None
    input_image_url: str | None
    similarity: float | None
    result: int
    recognize_time: str
    device_info: str | None


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


def get_user_by_id(conn: sqlite3.Connection, user_id: int) -> UserRow | None:
    row = conn.execute(
        """
        SELECT user_id, username, password, phone, email, status, role
        FROM user
        WHERE user_id = ?
        """,
        (int(user_id),),
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


def get_all_users(conn: sqlite3.Connection) -> list[UserRow]:
    rows = conn.execute(
        """
        SELECT user_id, username, password, phone, email, status, role
        FROM user
        ORDER BY user_id ASC
        """
    ).fetchall()
    return [
        UserRow(
            user_id=int(row["user_id"]),
            username=str(row["username"]),
            password=str(row["password"]),
            phone=row["phone"],
            email=row["email"],
            status=int(row["status"]),
            role=str(row["role"]),
        )
        for row in rows
    ]


def update_user_status(conn: sqlite3.Connection, user_id: int, status: int) -> None:
    conn.execute("UPDATE user SET status = ? WHERE user_id = ?", (int(status), int(user_id)))
    conn.commit()


def update_user_info(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    username: str,
    password_hash: str | None = None,
    photo: str | None = None,
) -> None:
    conn.execute(
        """
        UPDATE user
        SET username = ?,
            password = COALESCE(?, password),
            photo = COALESCE(?, photo)
        WHERE user_id = ?
        """,
        (username, password_hash, photo, int(user_id)),
    )
    conn.commit()


def delete_user_by_username(conn: sqlite3.Connection, username: str) -> None:
    conn.execute("DELETE FROM user WHERE username = ?", (username,))
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


def deactivate_face_feature(conn: sqlite3.Connection, feature_id: int) -> int:
    cur = conn.execute(
        """
        UPDATE face_feature
        SET is_active = 0
        WHERE feature_id = ?
        """,
        (int(feature_id),),
    )
    conn.commit()
    return int(cur.rowcount)


def deactivate_face_features_by_user_id(conn: sqlite3.Connection, user_id: int) -> int:
    cur = conn.execute(
        """
        UPDATE face_feature
        SET is_active = 0
        WHERE user_id = ? AND is_active = 1
        """,
        (int(user_id),),
    )
    conn.commit()
    return int(cur.rowcount)


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


def _recognition_log_from_row(row: sqlite3.Row) -> RecognitionLogRow:
    similarity = row["similarity"]
    return RecognitionLogRow(
        log_id=int(row["log_id"]),
        user_id=None if row["user_id"] is None else int(row["user_id"]),
        input_image_url=row["input_image_url"],
        similarity=None if similarity is None else float(similarity),
        result=int(row["result"]),
        recognize_time=str(row["recognize_time"]),
        device_info=row["device_info"],
    )


def get_recognition_logs_by_user_id(
    conn: sqlite3.Connection,
    user_id: int,
    *,
    limit: int = 100,
) -> list[RecognitionLogRow]:
    rows = conn.execute(
        """
        SELECT log_id, user_id, input_image_url, similarity, result, recognize_time, device_info
        FROM recognition_log
        WHERE user_id = ?
        ORDER BY recognize_time DESC, log_id DESC
        LIMIT ?
        """,
        (int(user_id), int(limit)),
    ).fetchall()
    return [_recognition_log_from_row(row) for row in rows]


def get_recognition_logs_by_time_range(
    conn: sqlite3.Connection,
    start_time: str,
    end_time: str,
    *,
    limit: int = 100,
) -> list[RecognitionLogRow]:
    rows = conn.execute(
        """
        SELECT log_id, user_id, input_image_url, similarity, result, recognize_time, device_info
        FROM recognition_log
        WHERE recognize_time >= ? AND recognize_time <= ?
        ORDER BY recognize_time DESC, log_id DESC
        LIMIT ?
        """,
        (start_time, end_time, int(limit)),
    ).fetchall()
    return [_recognition_log_from_row(row) for row in rows]


def get_recognition_logs_by_device_info(
    conn: sqlite3.Connection,
    device_info: str,
    *,
    limit: int = 100,
) -> list[RecognitionLogRow]:
    rows = conn.execute(
        """
        SELECT log_id, user_id, input_image_url, similarity, result, recognize_time, device_info
        FROM recognition_log
        WHERE device_info = ?
        ORDER BY recognize_time DESC, log_id DESC
        LIMIT ?
        """,
        (device_info, int(limit)),
    ).fetchall()
    return [_recognition_log_from_row(row) for row in rows]


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

