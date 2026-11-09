"""Qualitative + per-sequence analysis of saved detections on a MOT17 split.

For each sequence: an MP4 with TP (blue), FP (red), missed GT (yellow) and ignore regions
(grey), plus a per-sequence table (AP, AP50, recall and precision at the display threshold,
recall on small boxes). A mosaic of one frame per sequence is saved and, with --run-id,
logged to that MLflow run.

Usage:
  uv run python scripts/visualize_detections.py --dets outputs/dets_pretrained_val_half.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from detr_tracking.detection.evaluate import evaluate, load_gt
from detr_tracking.detection.matching import match_frame

BLUE, RED, YELLOW, GREY = (255, 140, 0), (40, 40, 230), (0, 220, 255), (150, 150, 150)  # BGR
SMALL_HEIGHT = 50  # px; boxes shorter than this count as "small pedestrians"
OCCLUDED_VISIBILITY = 0.5  # GT visibility below this counts as "occluded"


def xywh_to_xyxy(boxes: list[list[float]]) -> np.ndarray:
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return np.concatenate([b[:, :2], b[:, :2] + b[:, 2:]], axis=1)


def draw(img: np.ndarray, boxes: np.ndarray, color, thickness: int) -> None:
    for x1, y1, x2, y2 in boxes.astype(int):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def legend(img: np.ndarray, title: str, counts: dict[str, int]) -> None:
    items = [
        (f"TP {counts['tp']}", BLUE),
        (f"FP {counts['fp']}", RED),
        (f"missed {counts['missed']}", YELLOW),
        ("ignore", GREY),
    ]
    cv2.rectangle(img, (0, 0), (img.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    x = img.shape[1] - 580
    for text, color in items:
        cv2.putText(img, text, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        x += 140


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dets", type=Path, default=Path("outputs/dets_pretrained_val_half.json"))
    parser.add_argument("--split", default="val_half")
    parser.add_argument("--ann-dir", type=Path, default=Path("data/mot17/annotations"))
    parser.add_argument("--image-root", type=Path, default=Path("data/raw/MOT17/train"))
    parser.add_argument("--threshold", type=float, default=0.5, help="display score threshold")
    parser.add_argument("--max-frames", type=int, default=300, help="per video, 0 = all")
    parser.add_argument("--width", type=int, default=1280, help="output video width")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--run-id", default=None, help="MLflow run to attach the mosaic to")
    args = parser.parse_args()

    out = args.out or Path("outputs/qualitative") / args.dets.stem
    out.mkdir(parents=True, exist_ok=True)
    gt = load_gt(str(args.ann_dir / f"{args.split}.json"))
    all_dets = json.loads(args.dets.read_text())
    dets_by_image: dict[int, list[dict]] = defaultdict(list)
    for d in all_dets:
        dets_by_image[d["image_id"]].append(d)

    videos = {v["id"]: (v["name"], v["frame_rate"]) for v in gt.dataset["videos"]}
    rows, tiles = [], []
    for video_id, (name, fps) in videos.items():
        images = sorted(
            (im for im in gt.dataset["images"] if im["video_id"] == video_id),
            key=lambda im: im["frame_id"],
        )
        ids = [im["id"] for im in images]
        seq_dets = [d for i in ids for d in dets_by_image[i]]
        coco = evaluate(gt, seq_dets, image_ids=ids)

        totals = defaultdict(int)
        writer = None
        tile_index = len(images) // 2
        n_render = len(images) if args.max_frames == 0 else min(args.max_frames, len(images))
        for k, im in enumerate(tqdm(images, desc=name, leave=False)):
            anns = gt.loadAnns(gt.getAnnIds(imgIds=im["id"]))
            gt_boxes = xywh_to_xyxy([a["bbox"] for a in anns if not a["iscrowd"]])
            visibility = np.array([a["visibility"] for a in anns if not a["iscrowd"]])
            ign_boxes = xywh_to_xyxy([a["bbox"] for a in anns if a["iscrowd"]])
            dets = [d for d in dets_by_image[im["id"]] if d["score"] >= args.threshold]
            pred = xywh_to_xyxy([d["bbox"] for d in dets])
            scores = np.array([d["score"] for d in dets])
            m = match_frame(pred, scores, gt_boxes, ign_boxes)

            heights = gt_boxes[:, 3] - gt_boxes[:, 1]
            small = heights < SMALL_HEIGHT
            counts = {"tp": int(m.tp.sum()), "fp": int(m.fp.sum()), "missed": int(m.missed.sum())}
            for key, value in counts.items():
                totals[key] += value
            totals["gt"] += len(gt_boxes)
            totals["small"] += int(small.sum())
            totals["small_found"] += int((small & ~m.missed).sum())
            occluded = visibility < OCCLUDED_VISIBILITY
            totals["occ"] += int(occluded.sum())
            totals["occ_found"] += int((occluded & ~m.missed).sum())
            totals["vis_found"] += int((~occluded & ~m.missed).sum())

            if k >= n_render and k != tile_index:
                continue
            img = cv2.imread(str(args.image_root / im["file_name"]))
            draw(img, ign_boxes, GREY, 2)
            draw(img, gt_boxes[m.missed], YELLOW, 3)
            draw(img, pred[m.fp], RED, 3)
            draw(img, pred[m.tp], BLUE, 2)
            scale = args.width / img.shape[1]
            img = cv2.resize(img, (args.width, round(img.shape[0] * scale)))
            legend(img, f"{name}  frame {im['frame_id']}", counts)
            if k == tile_index:
                tiles.append(cv2.resize(img, (960, 540)))
            if k < n_render:
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(out / f"{name}.mp4"),
                        cv2.VideoWriter_fourcc(*"avc1"),
                        fps,
                        (img.shape[1], img.shape[0]),
                    )
                writer.write(img)
        if writer is not None:
            writer.release()

        tp, fp = totals["tp"], totals["fp"]
        rows.append(
            {
                "sequence": name,
                "frames": len(images),
                "gt_boxes": totals["gt"],
                "AP": coco["AP"],
                "AP50": coco["AP50"],
                "recall": tp / max(totals["gt"], 1),
                "precision": tp / max(tp + fp, 1),
                "small_share": totals["small"] / max(totals["gt"], 1),
                "small_recall": totals["small_found"] / max(totals["small"], 1),
                "occluded_share": totals["occ"] / max(totals["gt"], 1),
                "occluded_recall": totals["occ_found"] / max(totals["occ"], 1),
                "visible_recall": totals["vis_found"] / max(totals["gt"] - totals["occ"], 1),
            }
        )

    blank = np.zeros_like(tiles[0])
    tiles += [blank] * (-len(tiles) % 2)
    mosaic = np.vstack([np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)])
    mosaic_path = out / "mosaic.jpg"
    cv2.imwrite(str(mosaic_path), mosaic, [cv2.IMWRITE_JPEG_QUALITY, 85])
    table_path = out / "per_sequence.json"
    table_path.write_text(json.dumps(rows, indent=2))

    header = f"{'sequence':16s} {'frames':>6s} {'boxes':>6s} {'AP':>6s} {'AP50':>6s} "
    header += f"{'R@' + str(args.threshold):>6s} {'P@' + str(args.threshold):>6s} "
    header += f"{'%<50px':>6s} {'R<50px':>6s} {'%occ':>6s} {'R_occ':>6s} {'R_vis':>6s}"
    print(header)
    for r in rows:
        print(
            f"{r['sequence']:16s} {r['frames']:6d} {r['gt_boxes']:6d} {r['AP']:6.3f} "
            f"{r['AP50']:6.3f} {r['recall']:6.3f} {r['precision']:6.3f} "
            f"{r['small_share']:6.1%} {r['small_recall']:6.3f} {r['occluded_share']:6.1%} "
            f"{r['occluded_recall']:6.3f} {r['visible_recall']:6.3f}"
        )
    print(f"\nvideos + mosaic -> {out}")

    if args.run_id:
        import mlflow

        with mlflow.start_run(run_id=args.run_id):
            mlflow.log_artifact(str(mosaic_path), "qualitative")
            mlflow.log_artifact(str(table_path), "qualitative")


if __name__ == "__main__":
    main()
