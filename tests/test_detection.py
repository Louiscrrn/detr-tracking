import numpy as np
import pytest
import supervision as sv

from detr_tracking.detection.detector import PERSON_CLASS_ID, keep_class


def _detections(class_ids: list[int]) -> sv.Detections:
    n = len(class_ids)
    return sv.Detections(
        xyxy=np.tile([0.0, 0.0, 10.0, 20.0], (n, 1)),
        confidence=np.linspace(0.9, 0.5, n),
        class_id=np.array(class_ids, dtype=int),
    )


def test_keep_class_filters_other_classes():
    out = keep_class(_detections([PERSON_CLASS_ID, 3, PERSON_CLASS_ID, 18]))
    assert len(out) == 2
    assert (out.class_id == PERSON_CLASS_ID).all()


def test_keep_class_handles_empty():
    assert len(keep_class(sv.Detections.empty())) == 0


@pytest.mark.slow
def test_pretrained_model_finds_a_person():
    from PIL import Image

    from detr_tracking.detection.detector import detect_persons, load_detector

    image = Image.new("RGB", (640, 480), "white")
    detections = detect_persons(load_detector(), image)
    assert isinstance(detections, sv.Detections)
