from __future__ import annotations

import argparse
import json
from pathlib import Path

from .face_preprocessor import preprocess_face, save_preprocessed_face

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _default_output_image(out_dir: Path, image_path: Path) -> Path:
    return out_dir / f"{image_path.stem}_aligned.png"


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


def cmd_single(args: argparse.Namespace) -> None:
    image_path = Path(args.image)
    out_image = Path(args.out_image)
    try:
        result = preprocess_face(
            image_path,
            device=args.device,
            output_size=args.size,
            min_confidence=args.min_confidence,
        )
    except Exception as exc:
        raise SystemExit(f"preprocess failed for {image_path}: {exc}") from exc
    save_preprocessed_face(result.image, out_image)
    print(
        "preprocess_single_result:",
        {
            "image": image_path.as_posix(),
            "out_image": out_image.as_posix(),
            "size": args.size,
            "device": args.device,
            "selected_face_probability": round(result.probability, 6),
            "selected_face_box": [round(float(v), 3) for v in result.box.tolist()],
        },
    )


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

            out_raw = rec.get("out_image") or rec.get("out_path")
            out_image = Path(out_raw) if out_raw else _default_output_image(out_dir, image_path)
            if not out_image.is_absolute():
                out_image = out_dir / out_image
            items.append((image_path, out_image))
    else:
        if not args.input_dir:
            raise SystemExit("batch mode requires either --manifest or --input-dir")
        input_dir = Path(args.input_dir)
        images = _iter_images_in_dir(input_dir)
        items = [(p, _default_output_image(out_dir, p)) for p in images]

    for image_path, out_image in items:
        try:
            result = preprocess_face(
                image_path,
                device=args.device,
                output_size=args.size,
                min_confidence=args.min_confidence,
            )
            save_preprocessed_face(result.image, out_image)
            processed += 1
        except Exception as exc:
            failures.append({"image": image_path.as_posix(), "error": str(exc)})

    print(
        "preprocess_batch_result:",
        {
            "processed": processed,
            "failed": len(failures),
            "out_dir": out_dir.as_posix(),
            "size": args.size,
        },
    )
    if failures:
        print("preprocess_batch_failures:", failures)
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preprocess raw face images into aligned 160x160 RGB faces.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--size", default=160, type=int, help="aligned output size, default 160")
    p.add_argument(
        "--min-confidence",
        default=0.0,
        type=float,
        help="discard detections below this probability before choosing the largest face",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    single = sub.add_parser("single", help="preprocess one raw image")
    single.add_argument("--image", required=True, help="path to raw image")
    single.add_argument("--out-image", required=True, help="where to write aligned RGB face image")
    single.set_defaults(func=cmd_single)

    batch = sub.add_parser("batch", help="preprocess raw images in batch")
    batch.add_argument("--input-dir", default=None, help="directory of images (recursive)")
    batch.add_argument("--manifest", default=None, help="optional manifest json for explicit inputs")
    batch.add_argument("--out-dir", required=True, help="directory for aligned outputs")
    batch.set_defaults(func=cmd_batch)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
