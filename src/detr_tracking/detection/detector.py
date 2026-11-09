"""RF-DETR loading and person-only inference."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import supervision as sv

if TYPE_CHECKING:
    from PIL import Image
    from rfdetr import RFDETRSmall

PERSON_CLASS_ID = 1  # COCO category id used by RF-DETR pretrained weights


def load_detector() -> RFDETRSmall:
    """Load RF-DETR Small with COCO-pretrained weights (downloaded on first call)."""
    from rfdetr import RFDETRSmall

    return RFDETRSmall()


def keep_class(detections: sv.Detections, class_id: int = PERSON_CLASS_ID) -> sv.Detections:
    """Return only the detections of one class."""
    if len(detections) == 0 or detections.class_id is None:
        return detections[np.zeros(len(detections), dtype=bool)]
    return detections[detections.class_id == class_id]


def detect_persons(
    model: RFDETRSmall, image: Image.Image | np.ndarray, threshold: float = 0.5
) -> sv.Detections:
    """Run the detector on one RGB image and keep persons above `threshold`."""
    detections = model.predict(image, threshold=threshold)
    return keep_class(detections)
