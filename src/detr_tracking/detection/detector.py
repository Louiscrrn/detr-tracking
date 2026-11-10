"""RF-DETR loading and person-only inference."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import supervision as sv

if TYPE_CHECKING:
    from PIL import Image
    from rfdetr import RFDETRSmall

PERSON_CLASS_ID = 1  # COCO category id used by RF-DETR pretrained weights


def load_detector(weights: str | None = None) -> RFDETRSmall:
    """Load RF-DETR Small. Without `weights`, use COCO-pretrained weights (downloaded once)."""
    from rfdetr import RFDETRSmall

    return RFDETRSmall(pretrain_weights=weights) if weights else RFDETRSmall()


def person_class_id(fine_tuned: bool, class_names: list[str] | None) -> int:
    """Class id the model outputs for "person".

    COCO-pretrained RF-DETR outputs COCO category ids (person = 1). A fine-tuned checkpoint
    outputs the index in its own `class_names` (person = 0 for our single-class model).
    """
    if not fine_tuned:
        return PERSON_CLASS_ID
    if not class_names or "person" not in class_names:
        raise ValueError(f"fine-tuned model has no 'person' class: {class_names}")
    return class_names.index("person")


def keep_class(detections: sv.Detections, class_id: int = PERSON_CLASS_ID) -> sv.Detections:
    """Return only the detections of one class."""
    if len(detections) == 0 or detections.class_id is None:
        return detections[np.zeros(len(detections), dtype=bool)]
    return detections[detections.class_id == class_id]


def detect_persons(
    model: RFDETRSmall,
    image: Image.Image | np.ndarray,
    threshold: float = 0.5,
    class_id: int = PERSON_CLASS_ID,
) -> sv.Detections:
    """Run the detector on one RGB image and keep persons above `threshold`."""
    detections = model.predict(image, threshold=threshold)
    return keep_class(detections, class_id)
