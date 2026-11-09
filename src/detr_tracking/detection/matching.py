"""Per-frame matching of predictions to ground truth, for error analysis and visualisation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box of `a` (N, 4) and `b` (M, 4), both xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def ioa_matrix(a: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """Fraction of each box of `a` covered by each region (intersection over area of `a`)."""
    if len(a) == 0 or len(regions) == 0:
        return np.zeros((len(a), len(regions)))
    tl = np.maximum(a[:, None, :2], regions[None, :, :2])
    br = np.minimum(a[:, None, 2:], regions[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(axis=2)
    return inter / (a[:, 2:] - a[:, :2]).prod(axis=1)[:, None]


@dataclass
class FrameMatch:
    tp: np.ndarray  # bool per prediction
    fp: np.ndarray  # bool per prediction
    ignored: np.ndarray  # bool per prediction (unmatched, on an ignore region)
    missed: np.ndarray  # bool per ground-truth box


def match_frame(
    pred: np.ndarray,
    scores: np.ndarray,
    gt: np.ndarray,
    ignore: np.ndarray,
    iou_threshold: float = 0.5,
    ignore_ioa: float = 0.5,
) -> FrameMatch:
    """Greedy matching by decreasing score, like COCO at a single IoU threshold."""
    order = np.argsort(-scores)
    ious = iou_matrix(pred, gt)
    taken = np.zeros(len(gt), dtype=bool)
    tp = np.zeros(len(pred), dtype=bool)
    for i in order:
        if len(gt) == 0:
            break
        candidates = np.where(~taken & (ious[i] >= iou_threshold))[0]
        if len(candidates):
            j = candidates[np.argmax(ious[i, candidates])]
            taken[j] = True
            tp[i] = True
    on_ignore = (ioa_matrix(pred, ignore) >= ignore_ioa).any(axis=1) if len(ignore) else tp & False
    ignored = ~tp & on_ignore
    return FrameMatch(tp=tp, fp=~tp & ~ignored, ignored=ignored, missed=~taken)
