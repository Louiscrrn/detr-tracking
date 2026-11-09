import json
from pathlib import Path

import numpy as np
import pytest
import supervision as sv

from detr_tracking.detection.evaluate import detections_to_coco, evaluate, load_gt


@pytest.fixture
def gt(tmp_path: Path):
    coco = {
        "images": [{"id": 1, "width": 200, "height": 200}, {"id": 2, "width": 200, "height": 200}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 50, 100],
                "area": 5000,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "bbox": [100, 50, 60, 120],
                "area": 7200,
                "iscrowd": 0,
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 1,
                "bbox": [0, 0, 40, 80],
                "area": 3200,
                "iscrowd": 1,
            },
        ],
        "categories": [{"id": 1, "name": "person"}],
    }
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(coco))
    return load_gt(str(path))


def test_detections_to_coco_converts_xyxy_to_xywh():
    dets = sv.Detections(xyxy=np.array([[10.0, 20.0, 30.0, 60.0]]), confidence=np.array([0.8]))
    (res,) = detections_to_coco(7, dets)
    assert res == {"image_id": 7, "category_id": 1, "bbox": [10.0, 20.0, 20.0, 40.0], "score": 0.8}


def test_perfect_predictions_score_one(gt):
    results = [
        {"image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 100], "score": 0.9},
        {"image_id": 2, "category_id": 1, "bbox": [100, 50, 60, 120], "score": 0.9},
        # lands on the ignore region: must not count as a false positive
        {"image_id": 2, "category_id": 1, "bbox": [0, 0, 40, 80], "score": 0.95},
    ]
    metrics = evaluate(gt, results)
    assert metrics["AP"] == pytest.approx(1.0)
    assert metrics["AP50"] == pytest.approx(1.0)


def test_false_positive_lowers_ap(gt):
    results = [
        {"image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 100], "score": 0.5},
        {"image_id": 2, "category_id": 1, "bbox": [100, 50, 60, 120], "score": 0.5},
        {"image_id": 1, "category_id": 1, "bbox": [150, 150, 40, 40], "score": 0.99},
    ]
    assert evaluate(gt, results)["AP50"] < 1.0


def test_empty_results(gt):
    assert evaluate(gt, [])["AP"] == 0.0
