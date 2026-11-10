"""Evaluate RF-DETR Small on a MOT17 split (COCO mAP, person class) and log to MLflow.

Usage:
  uv run python scripts/eval_detector.py                      # pretrained COCO baseline
  uv run python scripts/eval_detector.py --weights w.pth      # fine-tuned checkpoint
  uv run python scripts/eval_detector.py --limit 50           # quick check on 50 frames
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlflow
import numpy as np
from PIL import Image
from tqdm import tqdm

from detr_tracking.data.amodal import enable_unclamped_predictions
from detr_tracking.detection.detector import keep_class, load_detector, person_class_id
from detr_tracking.detection.evaluate import detections_to_coco, evaluate, load_gt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val_half")
    parser.add_argument("--ann-dir", type=Path, default=Path("data/mot17/annotations"))
    parser.add_argument("--image-root", type=Path, default=Path("data/raw/MOT17/train"))
    parser.add_argument("--weights", default=None, help="checkpoint; default = COCO pretrained")
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="evaluate the first N images")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    gt_path = args.ann_dir / f"{args.split}.json"
    gt = load_gt(str(gt_path))
    images = gt.loadImgs(sorted(gt.getImgIds()))[: args.limit]

    # MOT17 protocol (TrackEval): boxes are compared as they are, nothing is clamped to the image
    enable_unclamped_predictions()
    model = load_detector(args.weights)
    config = model.model_config
    class_id = person_class_id(args.weights is not None, model.class_names)

    results: list[dict] = []
    batch_times: list[float] = []
    for start in tqdm(range(0, len(images), args.batch_size), desc="predict"):
        batch = images[start : start + args.batch_size]
        pil = [Image.open(args.image_root / im["file_name"]).convert("RGB") for im in batch]
        t0 = time.perf_counter()
        preds = model.predict(pil, threshold=args.threshold)
        batch_times.append((time.perf_counter() - t0) / len(batch))
        preds = preds if isinstance(preds, list) else [preds]
        for im, det in zip(batch, preds, strict=True):
            results.extend(detections_to_coco(im["id"], keep_class(det, class_id)))

    metrics = evaluate(gt, results, image_ids=[im["id"] for im in images])
    # first batch includes warm-up; report the median per-image time of the rest
    ms_per_image = 1000 * float(np.median(batch_times[1:] or batch_times))

    args.out.mkdir(parents=True, exist_ok=True)
    tag = "pretrained" if args.weights is None else Path(args.weights).stem
    pred_path = args.out / f"dets_{tag}_{args.split}.json"
    pred_path.write_text(json.dumps(results))

    mlflow.set_experiment("detection")
    with mlflow.start_run(run_name=args.run_name or f"rfdetr-small-{tag}-{args.split}"):
        mlflow.log_params(
            {
                "model": "rfdetr-small",
                "weights": args.weights or "coco-pretrained",
                "resolution": config.resolution,
                "device": str(config.device),
                "split": args.split,
                "n_images": len(images),
                "score_threshold": args.threshold,
                "batch_size": args.batch_size,
            }
        )
        mlflow.log_metrics({**metrics, "ms_per_image": ms_per_image})
        mlflow.log_artifact(str(pred_path))

    print(f"\n{args.split}: {len(images)} images, {ms_per_image:.1f} ms/image ({config.device})")
    for name in ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR100"):
        print(f"  {name:10s} {metrics[name]:.3f}")


if __name__ == "__main__":
    main()
