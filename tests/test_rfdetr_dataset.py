import json
from pathlib import Path

import pytest

from detr_tracking.data.rfdetr_dataset import build_dataset, flat_name


def test_flat_name():
    assert flat_name("MOT17-02-FRCNN/img1/000001.jpg") == "MOT17-02-FRCNN_000001.jpg"


def _write_split(ann_dir: Path, split: str, frames: list[int]) -> None:
    images = [
        {"id": f, "file_name": f"SEQ/img1/{f:06d}.jpg", "width": 4, "height": 4} for f in frames
    ]
    anns = [
        {"id": f, "image_id": f, "category_id": 1, "bbox": [0, 0, 2, 2], "area": 4, "iscrowd": 0}
        for f in frames
    ]
    coco = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "person"}]}
    (ann_dir / f"{split}.json").write_text(json.dumps(coco))


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path, Path]:
    ann_dir, image_root, out = tmp_path / "ann", tmp_path / "img", tmp_path / "out"
    ann_dir.mkdir()
    (image_root / "SEQ" / "img1").mkdir(parents=True)
    for f in range(1, 5):
        (image_root / "SEQ" / "img1" / f"{f:06d}.jpg").write_bytes(b"jpg")
    _write_split(ann_dir, "train", [1, 2])
    _write_split(ann_dir, "val", [3, 4])
    return ann_dir, image_root, out


def test_build_dataset_layout(dirs):
    ann_dir, image_root, out = dirs
    build_dataset(ann_dir, image_root, out)
    for folder, frames in (("train", [1, 2]), ("valid", [3, 4])):
        coco = json.loads((out / folder / "_annotations.coco.json").read_text())
        names = [im["file_name"] for im in coco["images"]]
        assert names == [f"SEQ_{f:06d}.jpg" for f in frames]
        assert all((out / folder / n).read_bytes() == b"jpg" for n in names)
        assert coco["categories"] == [{"id": 1, "name": "person", "supercategory": "none"}]


def test_build_dataset_limit_drops_orphan_annotations(dirs):
    ann_dir, image_root, out = dirs
    built = build_dataset(ann_dir, image_root, out, limit=1)
    assert len(built["train"]["images"]) == 1
    assert [a["image_id"] for a in built["train"]["annotations"]] == [1]
