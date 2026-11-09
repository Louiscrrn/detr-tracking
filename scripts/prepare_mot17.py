"""Convert MOT17 train ground truth to COCO json for train_half / val_half / full.

Usage: uv run python scripts/prepare_mot17.py [--root data/raw/MOT17] [--out data/mot17]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from detr_tracking.data.mot17 import to_coco


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/MOT17"))
    parser.add_argument("--out", type=Path, default=Path("data/mot17/annotations"))
    parser.add_argument("--detector", default="FRCNN", help="folder suffix; images are shared")
    args = parser.parse_args()

    train_dir = args.root / "train"
    seq_dirs = sorted(train_dir.glob(f"MOT17-*-{args.detector}"))
    if not seq_dirs:
        raise SystemExit(f"no sequences found in {train_dir}")

    args.out.mkdir(parents=True, exist_ok=True)
    for split in ("train_half", "val_half", "full"):
        coco = to_coco(seq_dirs, split, image_root=train_dir)
        path = args.out / f"{split}.json"
        path.write_text(json.dumps(coco))
        positives = sum(1 for a in coco["annotations"] if not a["iscrowd"])
        print(
            f"{split:10s} {len(coco['images']):5d} images  {positives:6d} boxes  "
            f"{len(coco['annotations']) - positives:5d} ignore -> {path}"
        )


if __name__ == "__main__":
    main()
