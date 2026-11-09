"""Run RF-DETR Small on one image and print the person detections.

Usage: uv run python scripts/smoke_detect.py [IMAGE_PATH_OR_URL] [--out annotated.jpg]
"""

from __future__ import annotations

import argparse
import io
import time
import urllib.request
from pathlib import Path

import supervision as sv
from PIL import Image

from detr_tracking.detection.detector import detect_persons, load_detector

DEFAULT_IMAGE = "http://images.cocodataset.org/val2017/000000000785.jpg"


def load_image(source: str) -> Image.Image:
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=30) as response:
            return Image.open(io.BytesIO(response.read())).convert("RGB")
    return Image.open(source).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default=DEFAULT_IMAGE)
    parser.add_argument("--out", default="outputs/smoke_detect.jpg")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    image = load_image(args.image)
    model = load_detector()
    detect_persons(model, image, args.threshold)  # warm-up

    start = time.perf_counter()
    detections = detect_persons(model, image, args.threshold)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"{len(detections)} person(s) in {elapsed_ms:.0f} ms")
    for box, score in zip(detections.xyxy, detections.confidence, strict=True):
        print(f"  score={score:.2f} box={box.round(1).tolist()}")

    annotated = sv.BoxAnnotator().annotate(image.copy(), detections)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    annotated.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
