"""Fine-tune RF-DETR Small on MOT17 pedestrians (train_half), validating on val_half.

One entry point for local and Kaggle runs. With `--download`, MOT17 is fetched and converted
first. With `--devices 2`, training runs with DDP on two GPUs (Lightning relaunches this script
once per GPU; data preparation only runs in the main process).

By default, boxes that extend past the image (MOT17 annotates the full person) are kept whole
through RF-DETR's data pipeline, see `detr_tracking.data.amodal`. `--clamp-boxes` restores
RF-DETR's default clamping.

The model kept for evaluation is the last EMA checkpoint (`last_ema.pth`, fixed number of
epochs): val_half is also our test set, so picking the "best" epoch on it would inflate the mAP.

Usage:
  python scripts/train_detector.py --download --data-root /tmp/data --devices 2 --grad-accum 1
  python scripts/train_detector.py                        # local, data already in data/
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from detr_tracking.data.amodal import enable_amodal_boxes  # noqa: E402
from detr_tracking.data.rfdetr_dataset import build_dataset  # noqa: E402

# Module level on purpose: DDP processes and spawned dataloader workers re-import this file.
if "--clamp-boxes" not in sys.argv:
    enable_amodal_boxes()

MOT17_URL = "https://motchallenge.net/data/MOT17.zip"


def is_main_process() -> bool:
    """False in the extra processes Lightning launches for DDP."""
    return os.environ.get("LOCAL_RANK", "0") == "0" and os.environ.get("NODE_RANK", "0") == "0"


def download_mot17(data_root: Path, detector: str = "FRCNN") -> Path:
    """Download MOT17 and extract the train sequences of one detector folder (idempotent)."""
    train_dir = data_root / "MOT17" / "train"
    if any(train_dir.glob(f"MOT17-*-{detector}")):
        print(f"MOT17 already in {train_dir}")
        return train_dir
    data_root.mkdir(parents=True, exist_ok=True)
    zip_path = data_root / "MOT17.zip"
    print(f"downloading {MOT17_URL} (5.9 GB)...", flush=True)
    with urllib.request.urlopen(MOT17_URL) as response, open(zip_path, "wb") as f:
        shutil.copyfileobj(response, f, length=16 * 1024 * 1024)
    with zipfile.ZipFile(zip_path) as archive:
        members = [
            m for m in archive.namelist() if m.startswith("MOT17/train/") and f"-{detector}/" in m
        ]
        archive.extractall(data_root, members=members)
    zip_path.unlink()
    print(f"extracted {len(members)} files to {train_dir}")
    return train_dir


def prepare(args: argparse.Namespace) -> None:
    if args.download:
        download_mot17(args.data_root)
    if not (args.ann_dir / "train_half.json").exists():
        import json

        from detr_tracking.data.mot17 import to_coco

        seq_dirs = sorted(args.image_root.glob("MOT17-*-FRCNN"))
        if not seq_dirs:
            raise SystemExit(f"no MOT17 sequences in {args.image_root} (use --download?)")
        args.ann_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train_half", "val_half"):
            coco = to_coco(seq_dirs, split, image_root=args.image_root)
            (args.ann_dir / f"{split}.json").write_text(json.dumps(coco))
    built = build_dataset(
        args.ann_dir, args.image_root, args.dataset_dir, copy=args.copy, limit=args.limit
    )
    for folder, coco in built.items():
        print(f"{folder}: {len(coco['images'])} images, {len(coco['annotations'])} boxes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--download", action="store_true", help="fetch MOT17 into --data-root")
    parser.add_argument("--output-dir", type=Path, default=Path("weights/rfdetr-small-mot17"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8, help="per GPU")
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--devices", type=int, default=1, help="number of GPUs (DDP if > 1)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--eval-interval", type=int, default=2, help="validate every N epochs")
    parser.add_argument("--limit", type=int, default=None, help="use the first N images per split")
    parser.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    parser.add_argument(
        "--clamp-boxes", action="store_true", help="RF-DETR default: clamp boxes to the image"
    )
    args = parser.parse_args()

    args.image_root = args.data_root / "raw" / "MOT17" / "train"
    if args.download:
        args.image_root = args.data_root / "MOT17" / "train"
    args.ann_dir = args.data_root / "mot17" / "annotations"
    args.dataset_dir = args.data_root / "mot17" / "rfdetr"

    if is_main_process():
        prepare(args)
        print(f"amodal boxes: {not args.clamp_boxes}", flush=True)
        effective = args.batch_size * args.devices * args.grad_accum
        print(f"effective batch size: {effective} ({args.devices} GPU(s))", flush=True)

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
        "eval_interval": args.eval_interval,
        "class_names": ["person"],
        "early_stopping": False,
        # pinned so Kaggle (Albumentations installed) and local runs share the pipeline
        # that check_amodal_pipeline.py verifies
        "augmentation_backend": "torchvision",
    }
    if args.devices > 1:
        train_kwargs.update(device="cuda", devices=args.devices, strategy="ddp")
    model.train(**train_kwargs)

    if is_main_process():
        for extra in args.output_dir.glob("checkpoint_best_*.pth"):
            extra.unlink()  # we report the last epoch, not the best one on val
        print(f"done -> {args.output_dir / 'last_ema.pth'}")


if __name__ == "__main__":
    main()
