"""Check that boxes leaving the image survive RF-DETR's real training pipeline.

Builds RF-DETR's train dataset (annotation loading, random crop, resize, flip, normalisation),
draws the same samples (same random seed) with RF-DETR's default clamping and with
`enable_amodal_boxes()`, and reports how many target boxes extend past the image.

Output: outputs/checks/amodal_pipeline.jpg (left: default, right: amodal patch).

Usage: uv run python scripts/check_amodal_pipeline.py [--samples 200]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from detr_tracking.data.amodal import enable_amodal_boxes  # noqa: E402
from detr_tracking.data.rfdetr_dataset import build_dataset  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def sample(dataset, index: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (image HxWx3 uint8 BGR, boxes xyxy in pixels) for one augmented sample."""
    random.seed(seed)
    torch.manual_seed(seed)
    image, target = dataset[index]
    img = image.permute(1, 2, 0).numpy() * STD + MEAN
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)[:, :, ::-1].copy()
    h, w = img.shape[:2]
    cx, cy, bw, bh = target["boxes"].numpy().T  # normalised cxcywh after RF-DETR's Normalize
    x1, y1, x2, y2 = (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h
    boxes = np.stack([x1, y1, x2, y2], 1)
    return img, boxes


def outside(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    tol = 1.0
    return (
        (boxes[:, 0] < -tol)
        | (boxes[:, 1] < -tol)
        | (boxes[:, 2] > w + tol)
        | (boxes[:, 3] > h + tol)
    )


def render(img: np.ndarray, boxes: np.ndarray, title: str, pad: int = 160) -> np.ndarray:
    h, w = img.shape[:2]
    canvas = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(50, 50, 50))
    cv2.rectangle(canvas, (pad, pad), (pad + w, pad + h), (255, 255, 255), 2)
    out = outside(boxes, w, h)
    for box, is_out in zip(boxes.astype(int) + pad, out, strict=True):
        color = (0, 220, 255) if is_out else (255, 140, 0)
        cv2.rectangle(canvas, tuple(box[:2]), tuple(box[2:]), color, 3 if is_out else 1)
    label = f"{title}: {int(out.sum())}/{len(boxes)} boxes past the border"
    cv2.putText(canvas, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--samples", type=int, default=200, help="samples for the statistics")
    parser.add_argument("--out", type=Path, default=Path("outputs/checks/amodal_pipeline.jpg"))
    args = parser.parse_args()

    ann_dir = args.data_root / "mot17" / "annotations"
    dataset_dir = args.data_root / "mot17" / "rfdetr"
    build_dataset(ann_dir, args.data_root / "raw" / "MOT17" / "train", dataset_dir)

    from rfdetr import RFDETRSmall
    from rfdetr._namespace import _namespace_from_configs
    from rfdetr.datasets import build_dataset as rf_build_dataset

    model = RFDETRSmall()
    train_config = model.get_train_config(
        dataset_dir=str(dataset_dir), augmentation_backend="torchvision", class_names=["person"]
    )
    ns = _namespace_from_configs(model.model_config, train_config)
    dataset = rf_build_dataset("train", ns, model.model_config.resolution)

    stats_idx = random.Random(0).sample(range(len(dataset)), args.samples)

    samples: dict[str, list] = {}
    for name in ("default", "amodal"):
        if name == "amodal":
            enable_amodal_boxes()
        samples[name] = [sample(dataset, i, seed=k) for k, i in enumerate(stats_idx)]
        total = sum(len(b) for _, b in samples[name])
        n_out = sum(int(outside(b, im.shape[1], im.shape[0]).sum()) for im, b in samples[name])
        print(f"{name:8s}: {n_out}/{total} target boxes past the border ({n_out / total:.1%})")

    # show the 3 samples (same seed in both columns) with the most boxes kept past the border
    counts = [int(outside(b, im.shape[1], im.shape[0]).sum()) for im, b in samples["amodal"]]
    shown = sorted(range(len(counts)), key=lambda k: -counts[k])[:3]
    results = {name: [render(*samples[name][k], name) for k in shown] for name in samples}

    rows = []
    for left, right in zip(results["default"], results["amodal"], strict=True):
        hh = max(left.shape[0], right.shape[0])
        left = cv2.copyMakeBorder(left, 0, hh - left.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        right = cv2.copyMakeBorder(right, 0, hh - right.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        rows.append(np.hstack([left, right]))
    width = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 10, 0, width - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
