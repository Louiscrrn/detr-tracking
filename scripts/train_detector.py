"""Fine-tune RF-DETR Small on MOT17 pedestrians (train_half), validating on val_half.

Expects the COCO annotations from `scripts/prepare_mot17.py`. The model kept for evaluation is
the last checkpoint (fixed number of epochs): val_half is also our test set, so picking the
"best" epoch on it would inflate the reported mAP.

Usage:
  uv run python scripts/train_detector.py --epochs 20 --batch-size 8
  uv run python scripts/train_detector.py --limit 16 --epochs 1 --batch-size 2   # smoke test
"""

from __future__ import annotations

import argparse
from pathlib import Path

from detr_tracking.data.rfdetr_dataset import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-dir", type=Path, default=Path("data/mot17/annotations"))
    parser.add_argument("--image-root", type=Path, default=Path("data/raw/MOT17/train"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/mot17/rfdetr"))
    parser.add_argument("--output-dir", type=Path, default=Path("weights/rfdetr-small-mot17"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--eval-interval", type=int, default=2, help="validate every N epochs")
    parser.add_argument("--device", default=None, help="cuda / mps / cpu; default = auto")
    parser.add_argument("--limit", type=int, default=None, help="use the first N images per split")
    parser.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    args = parser.parse_args()

    built = build_dataset(
        args.ann_dir, args.image_root, args.dataset_dir, copy=args.copy, limit=args.limit
    )
    for folder, coco in built.items():
        print(f"{folder}: {len(coco['images'])} images, {len(coco['annotations'])} boxes")

    from rfdetr import RFDETRSmall

    model = RFDETRSmall()
    train_kwargs = {
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum,
        "lr": args.lr,
        "resolution": args.resolution,
        "num_workers": args.num_workers,
        "class_names": ["person"],
        "early_stopping": False,
        "eval_interval": args.eval_interval,
    }
    if args.device:
        train_kwargs["device"] = args.device
    model.train(**train_kwargs)
    print(f"checkpoints -> {args.output_dir}")


if __name__ == "__main__":
    main()
