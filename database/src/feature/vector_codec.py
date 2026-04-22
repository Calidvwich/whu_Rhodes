from __future__ import annotations

import numpy as np

VECTOR_DIM = 512
VECTOR_DTYPE = np.float32


def encode_feature_vector(vec: np.ndarray) -> bytes:
    arr = np.asarray(vec, dtype=VECTOR_DTYPE)
    if arr.shape != (VECTOR_DIM,):
        raise ValueError(f"feature_vector must be shape ({VECTOR_DIM},), got {arr.shape}")
    return arr.tobytes(order="C")


def decode_feature_vector(blob: bytes) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=VECTOR_DTYPE)
    if arr.shape != (VECTOR_DIM,):
        raise ValueError(f"feature_vector blob has wrong length: got {arr.shape}")
    return arr


def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < eps:
        return v
    return v / n

