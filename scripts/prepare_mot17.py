"""Convert MOT17 train ground truth to COCO json, split by sequence: train.json / val.json.

Usage: uv run python scripts/prepare_mot17.py [--root data/raw/MOT17] [--out data/mot17/annotations]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from detr_tracking.data.mot17 import write_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/MOT17"))
    parser.add_argument("--out", type=Path, default=Path("data/mot17/annotations"))
    parser.add_argument("--detector", default="FRCNN", help="folder suffix; images are shared")
    args = parser.parse_args()
    write_splits(args.root / "train", args.out, detector=args.detector)


if __name__ == "__main__":
    main()
