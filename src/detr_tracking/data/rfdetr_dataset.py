"""Lay out MOT17 COCO splits in the folder structure RF-DETR trains on.

    <out>/train/_annotations.coco.json + images
    <out>/valid/_annotations.coco.json + images

Images are symlinked (or copied with `copy=True`) and renamed `<sequence>_<frame>.jpg` so each
split folder is flat. Ignore regions keep `iscrowd=1`: RF-DETR drops them from the training
targets and its validation metric treats them as crowd regions.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

RFDETR_SPLITS = {"train": "train_half", "valid": "val_half"}


def flat_name(file_name: str) -> str:
    """`MOT17-02-FRCNN/img1/000001.jpg` -> `MOT17-02-FRCNN_000001.jpg`."""
    parts = Path(file_name).parts
    return f"{parts[0]}_{parts[-1]}"


def build_split(
    coco: dict, image_root: Path, split_dir: Path, *, copy: bool = False, limit: int | None = None
) -> dict:
    """Write one split folder and return the COCO dict saved in it."""
    split_dir.mkdir(parents=True, exist_ok=True)
    images = coco["images"][:limit] if limit else coco["images"]
    keep = {im["id"] for im in images}

    out_images = []
    for im in images:
        name = flat_name(im["file_name"])
        src, dst = (image_root / im["file_name"]).resolve(), split_dir / name
        if not dst.exists():
            if copy:
                shutil.copy2(src, dst)
            else:
                dst.symlink_to(src)
        out_images.append({**im, "file_name": name})

    out = {
        **coco,
        "images": out_images,
        "annotations": [a for a in coco["annotations"] if a["image_id"] in keep],
        "categories": [{"id": 1, "name": "person", "supercategory": "none"}],
    }
    (split_dir / "_annotations.coco.json").write_text(json.dumps(out))
    return out


def build_dataset(
    ann_dir: Path, image_root: Path, out_dir: Path, *, copy: bool = False, limit: int | None = None
) -> dict[str, dict]:
    """Build `train/` and `valid/` from `<ann_dir>/{train_half,val_half}.json`."""
    built = {}
    for folder, split in RFDETR_SPLITS.items():
        coco = json.loads((ann_dir / f"{split}.json").read_text())
        built[folder] = build_split(coco, image_root, out_dir / folder, copy=copy, limit=limit)
    return built
