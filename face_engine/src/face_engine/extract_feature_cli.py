from __future__ import annotations

import argparse
import json
from pathlib import Path

from .facenet_extractor import extract_from_aligned_face, save_vector_json

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _default_output_json(out_dir: Path, image_path: Path) -> Path:
    return out_dir / f"{image_path.stem}.json"


def cmd_single(args: argparse.Namespace) -> None:
    image_path = Path(args.image)
    out_json = Path(args.out_json)
    vec = extract_from_aligned_face(
        image_path,
        l2_normalize_output=not args.no_l2,
        device=args.device,
        model_cache=args.model_cache,
    )
    save_vector_json(vec, out_json)
    print(
        "extract_single_result:",
        {
            "image": image_path.as_posix(),
            "out_json": out_json.as_posix(),
            "device": args.device,
            "l2_normalized": not args.no_l2,
        },
    )


def _iter_images_in_dir(input_dir: Path) -> list[Path]:
    images: list[Path] = []
    for p in sorted(input_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            images.append(p)
    return images


def _load_manifest(path: Path) -> list[dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("samples"), list):
        return obj["samples"]
    raise SystemExit(f"manifest must be a list or {{\"samples\": [...]}}: {path}")


def cmd_batch(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    failures: list[dict[str, str]] = []

    if args.manifest:
        manifest_path = Path(args.manifest)
        records = _load_manifest(manifest_path)
        base_dir = manifest_path.parent
        items: list[tuple[Path, Path]] = []
        for rec in records:
            image_raw = rec.get("image_path") or rec.get("image") or rec.get("path")
            if not image_raw:
                raise SystemExit(f"manifest item missing image path: {rec}")
            image_path = Path(image_raw)
            if not image_path.is_absolute():
                image_path = (base_dir / image_path).resolve()

            out_raw = rec.get("out_json")
            out_json = Path(out_raw) if out_raw else _default_output_json(out_dir, image_path)
            if not out_json.is_absolute():
                out_json = out_dir / out_json
            items.append((image_path, out_json))
    else:
        if not args.input_dir:
            raise SystemExit("batch mode requires either --manifest or --input-dir")
        input_dir = Path(args.input_dir)
        images = _iter_images_in_dir(input_dir)
        items = [(p, _default_output_json(out_dir, p)) for p in images]

    for image_path, out_json in items:
        try:
            vec = extract_from_aligned_face(
                image_path,
                l2_normalize_output=not args.no_l2,
                device=args.device,
                model_cache=args.model_cache,
            )
            save_vector_json(vec, out_json)
            processed += 1
        except Exception as exc:
            failures.append({"image": image_path.as_posix(), "error": str(exc)})

    print(
        "extract_batch_result:",
        {
            "processed": processed,
            "failed": len(failures),
            "out_dir": out_dir.as_posix(),
            "l2_normalized": not args.no_l2,
        },
    )
    if failures:
        print("extract_batch_failures:", failures)
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract 512-d FaceNet features from aligned face images.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--model-cache", default=None, help="optional path used as TORCH_HOME cache")
    p.add_argument("--no-l2", action="store_true", help="disable L2 normalization on output vectors")
    sub = p.add_subparsers(dest="cmd", required=True)

    single = sub.add_parser("single", help="extract vector from one image")
    single.add_argument("--image", required=True, help="path to aligned face image")
    single.add_argument("--out-json", required=True, help="where to write JSON vector file")
    single.set_defaults(func=cmd_single)

    batch = sub.add_parser("batch", help="extract vectors in batch")
    batch.add_argument("--input-dir", default=None, help="directory of images (recursive)")
    batch.add_argument("--manifest", default=None, help="optional manifest json for explicit inputs")
    batch.add_argument("--out-dir", required=True, help="directory for output JSON files")
    batch.set_defaults(func=cmd_batch)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
