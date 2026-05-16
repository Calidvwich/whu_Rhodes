from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .db.connection import DbConfig, connect
from .db.dao import (
    RecognitionLogRow,
    deactivate_face_feature,
    deactivate_face_features_by_user_id,
    get_config_by_key,
    get_recognition_logs_by_device_info,
    get_recognition_logs_by_time_range,
    get_recognition_logs_by_user_id,
    get_user_by_id,
    get_user_by_username,
    init_schema,
    insert_face_feature,
    insert_recognition_log,
    insert_user,
    iter_all_active_features,
)
from .feature.matcher import match_best
from .feature.vector_codec import VECTOR_DIM, l2_normalize


@dataclass(frozen=True)
class EnrollResult:
    user_id: int
    feature_id: int
    username: str


@dataclass(frozen=True)
class RecognizeResult:
    matched: bool
    matched_user_id: int | None
    matched_username: str | None
    max_similarity: float
    threshold: float
    log_id: int


def default_db_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "app.sqlite3"


def default_schema_path() -> Path:
    return Path(__file__).resolve().parent / "db" / "schema.sql"


def to_dict(obj: object) -> dict:
    return asdict(obj)


class FaceRecService:
    """Small service layer used by the UI/backend integration code."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_db_path()

    def init_db(self, schema_path: str | Path | None = None) -> None:
        schema_file = Path(schema_path) if schema_path is not None else default_schema_path()
        schema_sql = schema_file.read_text(encoding="utf-8")
        with self._connect() as conn:
            init_schema(conn, schema_sql)

    def enroll(
        self,
        username: str,
        feature_vector: Iterable[float] | np.ndarray,
        *,
        password_hash: str = "dummy_hash",
        phone: str | None = None,
        email: str | None = None,
        role: str = "user",
        image_path: str | None = None,
    ) -> EnrollResult:
        vector = self._prepare_vector(feature_vector)
        with self._connect() as conn:
            user = get_user_by_username(conn, username)
            user_id = user.user_id if user is not None else insert_user(
                conn,
                username=username,
                password_hash=password_hash,
                phone=phone,
                email=email,
                role=role,
            )
            feature_id = insert_face_feature(conn, user_id=user_id, feature_vector=vector, image_path=image_path)
        return EnrollResult(user_id=user_id, feature_id=feature_id, username=username)

    def recognize(
        self,
        feature_vector: Iterable[float] | np.ndarray,
        *,
        device_info: str | None = None,
        input_image_url: str | None = None,
    ) -> RecognizeResult:
        vector = self._prepare_vector(feature_vector)
        with self._connect() as conn:
            threshold = float(get_config_by_key(conn, "threshold") or "0.65")
            result = match_best(vector, iter_all_active_features(conn), threshold=threshold)

            matched_username: str | None = None
            if result.matched and result.matched_user_id is not None:
                user = get_user_by_id(conn, result.matched_user_id)
                matched_username = user.username if user is not None else None

            log_id = insert_recognition_log(
                conn,
                user_id=result.matched_user_id,
                input_image_url=input_image_url,
                similarity=result.max_similarity,
                result=1 if result.matched else 0,
                device_info=device_info,
            )

        return RecognizeResult(
            matched=result.matched,
            matched_user_id=result.matched_user_id,
            matched_username=matched_username,
            max_similarity=result.max_similarity,
            threshold=threshold,
            log_id=log_id,
        )

    def deactivate_feature(self, feature_id: int) -> int:
        with self._connect() as conn:
            return deactivate_face_feature(conn, feature_id)

    def deactivate_features_by_user(self, user_id: int) -> int:
        with self._connect() as conn:
            return deactivate_face_features_by_user_id(conn, user_id)

    def get_logs_by_user(self, user_id: int, *, limit: int = 100) -> list[RecognitionLogRow]:
        with self._connect() as conn:
            return get_recognition_logs_by_user_id(conn, user_id, limit=limit)

    def get_logs_by_time_range(self, start_time: str, end_time: str, *, limit: int = 100) -> list[RecognitionLogRow]:
        with self._connect() as conn:
            return get_recognition_logs_by_time_range(conn, start_time, end_time, limit=limit)

    def get_logs_by_device_info(self, device_info: str, *, limit: int = 100) -> list[RecognitionLogRow]:
        with self._connect() as conn:
            return get_recognition_logs_by_device_info(conn, device_info, limit=limit)

    @contextmanager
    def _connect(self) -> Iterator:
        conn = connect(DbConfig(path=self.db_path))
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _prepare_vector(feature_vector: Iterable[float] | np.ndarray) -> np.ndarray:
        vector = np.asarray(feature_vector, dtype=np.float32)
        if vector.shape != (VECTOR_DIM,):
            raise ValueError(f"feature_vector must be shape ({VECTOR_DIM},), got {vector.shape}")
        return l2_normalize(vector)
