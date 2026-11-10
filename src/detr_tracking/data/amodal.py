"""Keep MOT17's amodal boxes through RF-DETR's training pipeline.

MOT17 annotates the full extent of a person, even the part outside the image. RF-DETR clamps
every box to the image twice: when loading COCO annotations (`ConvertCoco`) and in its random
crop (`crop`, via torchvision). Resize, flip and normalisation keep boxes as they are.

`enable_amodal_boxes()` replaces those two steps so that a box is kept whole whenever the model
can predict it: DETR outputs (cx, cy, w, h) through a sigmoid, so the centre must lie in the
image and the size must not exceed it. Other boxes are clamped as before (never dropped: in a
DETR, a dropped box teaches the model that nobody is there).

Call it at module level of the training script, so DDP workers and spawned dataloader workers
(which re-import the main module) patch RF-DETR too.
"""

from __future__ import annotations

from typing import Any

import torch

_PATCHED = False


def representable_or_clamped(boxes: torch.Tensor, width: float, height: float) -> torch.Tensor:
    """Keep each xyxy box whole if a sigmoid-parameterised DETR can predict it, else clamp it."""
    if len(boxes) == 0:
        return boxes
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    bw = boxes[:, 2] - boxes[:, 0]
    bh = boxes[:, 3] - boxes[:, 1]
    ok = (cx >= 0) & (cx <= width) & (cy >= 0) & (cy <= height) & (bw <= width) & (bh <= height)
    clamped = boxes.clone()
    clamped[:, 0::2] = clamped[:, 0::2].clamp(0, width)
    clamped[:, 1::2] = clamped[:, 1::2].clamp(0, height)
    return torch.where(ok[:, None], boxes, clamped)


def visible_mask(boxes: torch.Tensor, width: float, height: float) -> torch.Tensor:
    """True for boxes that keep a non-empty part inside a width x height image."""
    x1, y1 = boxes[:, 0].clamp(0, width), boxes[:, 1].clamp(0, height)
    x2, y2 = boxes[:, 2].clamp(0, width), boxes[:, 3].clamp(0, height)
    return (x2 > x1) & (y2 > y1)


def enable_amodal_boxes() -> None:
    """Patch RF-DETR's annotation loading and random crop to keep amodal boxes (idempotent)."""
    global _PATCHED
    if _PATCHED:
        return
    from rfdetr.datasets import _torchvision as tv
    from rfdetr.datasets import coco

    original_convert = coco.ConvertCoco.__call__
    original_crop = tv.crop

    def convert_call(self: Any, image: Any, target: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        if self.include_masks or self.include_keypoints:
            return original_convert(self, image, target)
        anno = [o for o in target["annotations"] if not o.get("iscrowd", 0)]
        image, out = original_convert(self, image, target)
        w, h = image.size
        raw = torch.as_tensor([o["bbox"] for o in anno], dtype=torch.float32).reshape(-1, 4)
        raw[:, 2:] += raw[:, :2]
        keep = visible_mask(raw, w, h)  # same rows the original kept (non-empty after clamping)
        out["boxes"] = representable_or_clamped(raw[keep], w, h)
        return image, out

    def crop(
        image: Any, target: dict[str, Any] | None, top: int, left: int, height: int, width: int
    ) -> tuple[Any, dict[str, Any] | None]:
        if target is None or "boxes" not in target or "masks" in target or "keypoints" in target:
            return original_crop(image, target, top, left, height, width)
        image = tv.functional.crop(image, top=top, left=left, height=height, width=width)
        boxes = target["boxes"] - torch.tensor([left, top, left, top], dtype=target["boxes"].dtype)
        keep = visible_mask(boxes, width, height)
        boxes = representable_or_clamped(boxes[keep], width, height)
        out = tv._filter_per_instance_fields(target.copy(), keep, boxes)
        return image, tv._update_size(out, height, width)

    coco.ConvertCoco.__call__ = convert_call
    tv.crop = crop
    _PATCHED = True
