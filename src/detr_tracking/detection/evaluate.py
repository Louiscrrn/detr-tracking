"""COCO-style detection evaluation (pycocotools) for the single `person` category."""

from __future__ import annotations

import contextlib
import io

import numpy as np
import supervision as sv
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

METRIC_NAMES = (
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR1",
    "AR10",
    "AR100",
    "AR_small",
    "AR_medium",
    "AR_large",
)


def detections_to_coco(
    image_id: int, detections: sv.Detections, category_id: int = 1
) -> list[dict]:
    """Convert supervision detections (xyxy) to COCO result dicts (xywh)."""
    results = []
    for (x1, y1, x2, y2), score in zip(detections.xyxy, detections.confidence, strict=True):
        results.append(
            {
                "image_id": int(image_id),
                "category_id": category_id,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score),
            }
        )
    return results


def evaluate(gt: COCO, results: list[dict], image_ids: list[int] | None = None) -> dict[str, float]:
    """Run COCOeval (bbox) and return the 12 standard metrics by name.

    `image_ids` restricts the evaluation, e.g. when only part of a split was predicted.
    """
    if not results:
        return dict.fromkeys(METRIC_NAMES, 0.0)
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(results)
        coco_eval = COCOeval(gt, dt, iouType="bbox")
        if image_ids is not None:
            coco_eval.params.imgIds = sorted(image_ids)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    stats = np.asarray(coco_eval.stats, dtype=float)
    return {name: float(v) for name, v in zip(METRIC_NAMES, stats, strict=True)}


def load_gt(path: str) -> COCO:
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(path)
