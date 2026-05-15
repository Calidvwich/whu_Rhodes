from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, ROOT.as_posix())

from src.feature.matcher import cosine_similarity
from src.feature.vector_codec import VECTOR_DIM, l2_normalize


def _base_vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return l2_normalize(rng.normal(size=(VECTOR_DIM,)).astype(np.float32))


def _near_vector(base: np.ndarray, seed: int, scale: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=scale, size=(VECTOR_DIM,)).astype(np.float32)
    return l2_normalize(base + noise)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print threshold sanity samples for cosine matching.")
    parser.add_argument("--threshold", type=float, default=0.65, help="Cosine threshold to evaluate.")
    args = parser.parse_args()

    threshold = float(args.threshold)
    base = _base_vector(2026)
    samples = [
        ("same_identity_like", _near_vector(base, 1, 0.02), "same-person enrollment noise"),
        ("borderline_like", _near_vector(base, 2, 0.05), "larger same-person/image noise"),
        ("different_identity_like", _base_vector(3001), "independent random face-like vector"),
        ("opposite_vector", -base, "known rejection control sample"),
    ]

    print(f"threshold: {threshold:.4f}")
    print("label,similarity,decision,note")
    for label, vector, note in samples:
        similarity = cosine_similarity(base, vector)
        decision = "match" if similarity >= threshold else "reject"
        print(f"{label},{similarity:.4f},{decision},{note}")

    print("note: this is a deterministic synthetic sanity report; final threshold evidence should use real face vectors.")


if __name__ == "__main__":
    main()
