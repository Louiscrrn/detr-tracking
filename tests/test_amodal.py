import torch
from PIL import Image

from detr_tracking.data.amodal import (
    enable_amodal_boxes,
    enable_unclamped_predictions,
    representable_or_clamped,
    visible_mask,
)

W, H = 200, 100


def test_representable_box_is_kept_whole():
    # centre (190, 65) inside the image, box sticks out on the right and bottom
    box = torch.tensor([[150.0, 20.0, 230.0, 110.0]])
    assert representable_or_clamped(box, W, H).tolist() == box.tolist()


def test_box_with_centre_outside_is_clamped():
    box = torch.tensor([[180.0, 20.0, 260.0, 60.0]])  # centre x = 220 > W
    assert representable_or_clamped(box, W, H).tolist() == [[180.0, 20.0, 200.0, 60.0]]


def test_box_larger_than_image_is_clamped():
    box = torch.tensor([[-10.0, -60.0, 100.0, 90.0]])  # height 150 > H
    assert representable_or_clamped(box, W, H).tolist() == [[0.0, 0.0, 100.0, 90.0]]


def test_visible_mask():
    boxes = torch.tensor([[-30.0, 10.0, -5.0, 50.0], [-30.0, 10.0, 5.0, 50.0]])
    assert visible_mask(boxes, W, H).tolist() == [False, True]


def _target():
    anns = [
        {"bbox": [150, 20, 80, 90], "category_id": 1, "area": 7200, "iscrowd": 0},  # amodal
        {"bbox": [10, 10, 20, 40], "category_id": 1, "area": 800, "iscrowd": 0},  # inside
        {"bbox": [-50, 10, 20, 40], "category_id": 1, "area": 800, "iscrowd": 0},  # fully outside
        {"bbox": [60, 10, 20, 40], "category_id": 1, "area": 800, "iscrowd": 1},  # ignore region
    ]
    return {"image_id": 1, "annotations": anns}


def test_patched_convert_coco_keeps_amodal_boxes():
    enable_amodal_boxes()
    from rfdetr.datasets.coco import ConvertCoco

    _, out = ConvertCoco()(Image.new("RGB", (W, H)), _target())
    assert out["boxes"].tolist() == [[150.0, 20.0, 230.0, 110.0], [10.0, 10.0, 30.0, 50.0]]
    assert len(out["labels"]) == len(out["area"]) == len(out["iscrowd"]) == 2


def test_patched_crop_shifts_without_clamping():
    enable_amodal_boxes()
    from rfdetr.datasets import _torchvision as tv

    target = {
        "boxes": torch.tensor([[150.0, 20.0, 230.0, 110.0], [10.0, 10.0, 30.0, 50.0]]),
        "labels": torch.tensor([0, 0]),
        "size": torch.tensor([H, W]),
    }
    # crop x in [100, 200): the second box falls outside and is dropped, the first is shifted
    _, out = tv.crop(Image.new("RGB", (W, H)), target, top=0, left=100, height=H, width=100)
    assert out["boxes"].tolist() == [[50.0, 20.0, 130.0, 110.0]]
    assert out["labels"].tolist() == [0]


def test_unclamped_predictions_keep_boxes_past_the_border():
    enable_unclamped_predictions()
    from rfdetr.models.postprocess import PostProcess

    out_bbox = torch.tensor([[[0.95, 0.5, 0.3, 0.4]]])  # normalised cxcywh, x2 = 1.10 > 1
    sizes = torch.tensor([[H, W]])
    boxes = PostProcess._gather_and_scale_boxes(out_bbox, torch.tensor([[0]]), sizes)
    assert torch.allclose(boxes[0, 0], torch.tensor([160.0, 30.0, 220.0, 70.0]))
