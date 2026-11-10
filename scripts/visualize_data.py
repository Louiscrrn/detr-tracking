"""Qualitative views of the data: a dataset grid and augmented training samples.

- outputs/checks/dataset_grid.png: 4 frames per MOT17 sequence, labelled with its split
  with pedestrians (green) and ignore regions (grey).
- outputs/checks/augmentations.png: per row, one source frame with its GT, then several
  samples from RF-DETR's real training pipeline (resize, random crop, flip) with the amodal
  patch; the image border is white, boxes kept past it are yellow.

Usage: uv run python scripts/visualize_data.py
"""

from __future__ import annotations

import argparse
import json
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

GREEN, GREY, YELLOW = (60, 200, 60), (160, 160, 160), (0, 220, 255)
BLUE, WHITE = (255, 140, 0), (255, 255, 255)
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def put_label(img: np.ndarray, text: str, scale: float = 0.6) -> None:
    cv2.rectangle(img, (0, 0), (img.shape[1], int(30 * scale / 0.6)), (0, 0, 0), -1)
    cv2.putText(img, text, (6, int(21 * scale / 0.6)), cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, 1)


def fit(img: np.ndarray, h: int) -> np.ndarray:
    return cv2.resize(img, (round(img.shape[1] * h / img.shape[0]), h))


def pad_row(tiles: list[np.ndarray], width: int) -> np.ndarray:
    row = np.hstack(tiles)
    return cv2.copyMakeBorder(row, 0, 6, 0, width - row.shape[1], cv2.BORDER_CONSTANT)


def gt_frame(im: dict, anns: list[dict], image_root: Path, pad_frac: float = 0.0) -> np.ndarray:
    img = cv2.imread(str(image_root / im["file_name"]))
    pad = int(pad_frac * im["height"])
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(50, 50, 50))
    if pad:
        cv2.rectangle(img, (pad, pad), (pad + im["width"], pad + im["height"]), WHITE, 3)
    thick = max(2, im["height"] // 300)
    for a in anns:
        x, y, w, h = (np.array(a["bbox"]) + [pad, pad, 0, 0]).astype(int)
        cv2.rectangle(img, (x, y), (x + w, y + h), GREY if a["iscrowd"] else GREEN, thick)
    return img


def dataset_grid(ann_dir: Path, image_root: Path, out: Path, tile_h: int = 300) -> None:
    rows: dict[str, list[np.ndarray]] = {}
    for split in ("train", "val"):
        coco = json.loads((ann_dir / f"{split}.json").read_text())
        by_img: dict[int, list[dict]] = {}
        for a in coco["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a)
        for v in coco["videos"]:
            frames = [im for im in coco["images"] if im["video_id"] == v["id"]]
            for im in (frames[(2 * k + 1) * len(frames) // 8] for k in range(4)):
                tile = fit(gt_frame(im, by_img.get(im["id"], []), image_root), tile_h)
                n = sum(1 for a in by_img.get(im["id"], []) if not a["iscrowd"])
                put_label(tile, f"{v['name'][:8]} {split} f{im['frame_id']} ({n} ped.)", 0.55)
                rows.setdefault(v["name"], []).append(tile)
    width = max(sum(t.shape[1] for t in r) for r in rows.values())
    grid = np.vstack([pad_row(rows[name], width) for name in sorted(rows)])
    cv2.imwrite(str(out), grid)
    print(f"saved {out} ({grid.shape[1]}x{grid.shape[0]})")


def augmented(dataset, index: int, seed: int, pad: int = 60) -> np.ndarray:
    random.seed(seed)
    torch.manual_seed(seed)
    image, target = dataset[index]
    img = image.permute(1, 2, 0).numpy() * STD + MEAN
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)[:, :, ::-1].copy()
    h, w = img.shape[:2]
    canvas = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(50, 50, 50))
    cv2.rectangle(canvas, (pad, pad), (pad + w, pad + h), WHITE, 2)
    for cx, cy, bw, bh in target["boxes"].numpy():
        x1, y1, x2, y2 = (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h
        out = x1 < -1 or y1 < -1 or x2 > w + 1 or y2 > h + 1
        p1, p2 = (int(x1) + pad, int(y1) + pad), (int(x2) + pad, int(y2) + pad)
        cv2.rectangle(canvas, p1, p2, YELLOW if out else BLUE, 2 if out else 1)
    return canvas


def augmentation_examples(data_root: Path, out: Path, per_row: int = 4, tile_h: int = 330) -> None:
    ann_dir = data_root / "mot17" / "annotations"
    image_root = data_root / "raw" / "MOT17" / "train"
    dataset_dir = data_root / "mot17" / "rfdetr"
    build_dataset(ann_dir, image_root, dataset_dir)
    enable_amodal_boxes()

    from rfdetr import RFDETRSmall
    from rfdetr._namespace import _namespace_from_configs
    from rfdetr.datasets import build_dataset as rf_build_dataset

    model = RFDETRSmall()
    config = model.get_train_config(
        dataset_dir=str(dataset_dir), augmentation_backend="torchvision", class_names=["person"]
    )
    ns = _namespace_from_configs(model.model_config, config)
    dataset = rf_build_dataset("train", ns, model.model_config.resolution)

    coco = json.loads((dataset_dir / "train" / "_annotations.coco.json").read_text())
    raw = json.loads((ann_dir / "train.json").read_text())
    raw_images = {im["id"]: im for im in raw["images"]}
    by_img: dict[int, list[dict]] = {}
    for a in raw["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    # one frame per sequence for 5 sequences, picked in the middle of each sequence
    picks = []
    for v in coco["videos"][:: max(1, len(coco["videos"]) // 5)][:5]:
        idx = [i for i, im in enumerate(coco["images"]) if im["video_id"] == v["id"]]
        picks.append(idx[len(idx) // 2])

    rows = []
    for k, i in enumerate(picks):
        im = raw_images[coco["images"][i]["id"]]
        src = fit(gt_frame(im, by_img.get(im["id"], []), image_root, pad_frac=0.12), tile_h)
        put_label(src, f"source {im['file_name'].split('/')[0][:8]} f{im['frame_id']} (GT)", 0.5)
        aug = [fit(augmented(dataset, i, seed=10 * k + s), tile_h) for s in range(per_row)]
        for s, tile in enumerate(aug):
            put_label(tile, f"augmented #{s + 1} (yellow: kept past border)", 0.5)
        tiles = [src, *aug]
        rows.append(tiles)
    width = max(sum(t.shape[1] for t in r) for r in rows)
    grid = np.vstack([pad_row(r, width) for r in rows])
    cv2.imwrite(str(out), grid)
    print(f"saved {out} ({grid.shape[1]}x{grid.shape[0]})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/checks"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_grid(
        args.data_root / "mot17" / "annotations",
        args.data_root / "raw" / "MOT17" / "train",
        args.out_dir / "dataset_grid.png",
    )
    augmentation_examples(args.data_root, args.out_dir / "augmentations.png")


if __name__ == "__main__":
    main()
