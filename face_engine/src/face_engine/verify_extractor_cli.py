from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

from .facenet_extractor import extract_from_aligned_face, save_vector_json
from .facenet_submodule import MTCNN, REPO_ROOT

DEFAULT_SAMPLES = [
    {
        "person_id": "obama",
        "name": "Barack Obama",
        "image_source": "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg",
        "filename": "obama_1.jpg",
    },
    {
        "person_id": "obama",
        "name": "Barack Obama",
        "image_source": "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama2.jpg",
        "filename": "obama_2.jpg",
    },
    {
        "person_id": "biden",
        "name": "Joe Biden",
        "image_source": "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/biden.jpg",
        "filename": "biden_1.jpg",
    },
    {
        "person_id": "biden",
        "name": "Joe Biden",
        "image_source": "https://upload.wikimedia.org/wikipedia/commons/6/68/Joe_Biden_presidential_portrait.jpg",
        "filename": "biden_2.jpg",
    },
]


@dataclass
class Sample:
    person_id: str
    name: str
    image_path: str
    image_source: str


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b))


def _download_default_samples(samples_dir: Path) -> list[Sample]:
    samples_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "whu-rhodes-face-verify/1.0"})

    records: list[Sample] = []
    for spec in DEFAULT_SAMPLES:
        out_path = samples_dir / spec["filename"]
        resp = session.get(spec["image_source"], timeout=30)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        records.append(
            Sample(
                person_id=spec["person_id"],
                name=spec["name"],
                image_path=out_path.as_posix(),
                image_source=spec["image_source"],
            )
        )

    manifest_path = samples_dir / "download_manifest.json"
    manifest_path.write_text(
        json.dumps({"samples": [asdict(r) for r in records]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return records


def _load_samples_from_manifest(path: Path) -> list[Sample]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and isinstance(obj.get("samples"), list):
        items = obj["samples"]
    elif isinstance(obj, list):
        items = obj
    else:
        raise SystemExit(f"manifest format error: {path}")

    out: list[Sample] = []
    for item in items:
        person_id = str(item["person_id"])
        name = str(item.get("name", person_id))
        image_source = str(item.get("image_source", item.get("image_path", "")))
        image_raw = item.get("image_path") or item.get("image")
        if not image_raw:
            raise SystemExit(f"manifest item missing image path: {item}")
        image_path = Path(image_raw)
        if not image_path.is_absolute():
            image_path = (path.parent / image_path).resolve()
        out.append(
            Sample(
                person_id=person_id,
                name=name,
                image_path=image_path.as_posix(),
                image_source=image_source,
            )
        )
    return out


def _compute_pairwise(samples: list[Sample], vectors: dict[str, np.ndarray]) -> tuple[list[float], list[float]]:
    same: list[float] = []
    diff: list[float] = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            a = samples[i]
            b = samples[j]
            sim = cosine_similarity(vectors[a.image_path], vectors[b.image_path])
            if a.person_id == b.person_id:
                same.append(sim)
            else:
                diff.append(sim)
    return same, diff


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _align_face_to_160(mtcnn: MTCNN, src_path: Path, aligned_path: Path) -> None:
    img = Image.open(src_path).convert("RGB")
    face = mtcnn(img)
    if face is None:
        raise RuntimeError(f"no face detected in image: {src_path}")
    if face.ndim == 4:
        face = face[0]
    arr = face.permute(1, 2, 0).detach().cpu().numpy()
    if arr.max() <= 1.5:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    aligned_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(aligned_path)


def cmd_verify(args: argparse.Namespace) -> None:
    if not args.download_samples and not args.manifest:
        raise SystemExit("please provide at least one source: --download-samples or --manifest")

    samples: list[Sample] = []
    if args.download_samples:
        samples.extend(_download_default_samples(Path(args.samples_dir)))
    if args.manifest:
        samples.extend(_load_samples_from_manifest(Path(args.manifest)))

    if len(samples) < 3:
        raise SystemExit(f"need >=3 samples, got {len(samples)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vectors: dict[str, np.ndarray] = {}
    resolved_device = _resolve_device(args.device)
    mtcnn = None
    if args.auto_align:
        mtcnn = MTCNN(
            image_size=160,
            margin=0,
            post_process=False,
            select_largest=True,
            device=resolved_device,
        )

    vector_items: list[dict] = []
    for sample in samples:
        source_image_path = Path(sample.image_path)
        image_path = source_image_path
        if mtcnn is not None:
            aligned_path = out_dir / "aligned_faces" / f"{source_image_path.stem}_aligned.png"
            _align_face_to_160(mtcnn, source_image_path, aligned_path)
            image_path = aligned_path

        vector = extract_from_aligned_face(
            image_path,
            l2_normalize_output=not args.no_l2,
            device=resolved_device,
            model_cache=args.model_cache,
        )
        out_json = out_dir / f"{image_path.stem}.json"
        save_vector_json(vector, out_json)
        vectors[sample.image_path] = vector
        vector_items.append(
            {
                "person_id": sample.person_id,
                "name": sample.name,
                "image_path": sample.image_path,
                "aligned_image_path": image_path.as_posix(),
                "image_source": sample.image_source,
                "vector_json": out_json.as_posix(),
                "l2_normalized": not args.no_l2,
            }
        )

    same, diff = _compute_pairwise(samples, vectors)
    if not same:
        raise SystemExit("at least one same-person pair is required")
    if not diff:
        raise SystemExit("at least one different-person pair is required")

    same_sorted = sorted(same, reverse=True)
    diff_sorted = sorted(diff, reverse=True)
    sim_same_1 = same_sorted[0]
    sim_same_2 = same_sorted[1] if len(same_sorted) > 1 else None
    sim_diff = diff_sorted[0]

    passed = sim_same_1 > sim_diff and (sim_same_2 is None or sim_same_2 > sim_diff)
    report = {
        "num_samples": len(samples),
        "num_same_pairs": len(same),
        "num_diff_pairs": len(diff),
        "sim_same_1": sim_same_1,
        "sim_same_2": sim_same_2,
        "sim_diff": sim_diff,
        "verdict": "PASS" if passed else "FAIL",
        "l2_normalized": not args.no_l2,
        "auto_align": args.auto_align,
        "items": vector_items,
    }

    report_path = out_dir / "verify_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("verify_result:", report)
    print("verify_report_path:", report_path.as_posix())

    if not passed:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Verify FaceNet extractor by same/diff person similarity.")
    p.add_argument("--download-samples", action="store_true", help="download public sample images")
    p.add_argument("--samples-dir", default=(REPO_ROOT / "face_engine" / "examples" / "verify_samples").as_posix())
    p.add_argument("--manifest", default=None, help="local sample manifest json")
    p.add_argument("--out-dir", default=(REPO_ROOT / "face_engine" / "examples" / "verify_vectors").as_posix())
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--model-cache", default=None, help="optional path used as TORCH_HOME cache")
    p.add_argument("--no-l2", action="store_true", help="disable L2 normalization on vectors")
    p.add_argument("--auto-align", action="store_true", help="align faces with MTCNN before extracting")
    p.add_argument("--no-auto-align", action="store_true", help="skip auto align and use image as-is")
    p.set_defaults(func=cmd_verify)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.auto_align and args.no_auto_align:
        raise SystemExit("do not use both --auto-align and --no-auto-align")
    args.auto_align = True if not args.no_auto_align else False
    args.func(args)
