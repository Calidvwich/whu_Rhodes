from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .vector_codec import l2_normalize


@dataclass(frozen=True)
class MatchResult:
    matched_user_id: int | None
    max_similarity: float
    matched: bool


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))


def match_best(
    input_vector: np.ndarray,
    library: Iterable[tuple[int, np.ndarray]],
    threshold: float,
) -> MatchResult:
    best_user_id: int | None = None
    best_sim = -1.0

    for user_id, vec in library:
        sim = cosine_similarity(input_vector, vec)
        if sim > best_sim:
            best_sim = sim
            best_user_id = int(user_id)

    matched = best_user_id is not None and best_sim >= float(threshold)
    return MatchResult(matched_user_id=best_user_id if matched else None, max_similarity=best_sim, matched=matched)

