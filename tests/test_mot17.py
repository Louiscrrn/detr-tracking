from pathlib import Path

import numpy as np
import pytest

from detr_tracking.data.mot17 import classify_rows, split_by_sequence, split_frames, to_coco

SEQINFO = """[Sequence]
name=MOT17-99-FRCNN
imDir=img1
frameRate=30
seqLength=4
imWidth=640
imHeight=480
imExt=.jpg
"""

# frame, id, x, y, w, h, flag, class, visibility
GT = """1,1,10,10,20,40,1,1,1.0
1,2,50,50,20,40,0,7,1.0
1,3,90,90,30,30,0,3,1.0
3,1,12,10,20,40,1,1,0.5
4,1,14,10,20,40,0,1,0.2
"""


@pytest.fixture
def seq_dir(tmp_path: Path) -> Path:
    seq = tmp_path / "train" / "MOT17-99-FRCNN"
    (seq / "gt").mkdir(parents=True)
    (seq / "seqinfo.ini").write_text(SEQINFO)
    (seq / "gt" / "gt.txt").write_text(GT)
    return seq


def test_split_frames_halves_cover_sequence_without_overlap():
    train, val = split_frames(7, "train_half"), split_frames(7, "val_half")
    assert list(train) == [1, 2, 3]
    assert list(val) == [4, 5, 6, 7]
    assert list(split_frames(7, "full")) == list(range(1, 8))


def test_classify_rows():
    gt = np.loadtxt(GT.splitlines(), delimiter=",")
    positive, ignore = classify_rows(gt)
    assert positive.tolist() == [True, False, False, True, False]
    assert ignore.tolist() == [False, True, False, False, True]


def test_to_coco_train_half(seq_dir: Path):
    coco = to_coco([seq_dir], "train_half", image_root=seq_dir.parent)
    assert [im["frame_id"] for im in coco["images"]] == [1, 2]
    assert coco["images"][0]["file_name"] == "MOT17-99-FRCNN/img1/000001.jpg"
    anns = coco["annotations"]
    assert [(a["track_id"], a["iscrowd"]) for a in anns] == [(1, 0), (2, 1)]
    assert anns[0]["bbox"] == [10.0, 10.0, 20.0, 40.0]
    assert anns[0]["area"] == 800.0


def test_to_coco_keeps_boxes_past_the_border(seq_dir: Path):
    # MOT17 / TrackEval protocol: GT boxes are never clamped to the image
    gt = seq_dir / "gt" / "gt.txt"
    gt.write_text(gt.read_text() + "2,4,600,400,80,150,1,1,0.6\n")  # x2=680 > 640, y2=550 > 480
    coco = to_coco([seq_dir], "train_half", image_root=seq_dir.parent)
    assert [600.0, 400.0, 80.0, 150.0] in [a["bbox"] for a in coco["annotations"]]


def test_to_coco_val_half(seq_dir: Path):
    coco = to_coco([seq_dir], "val_half", image_root=seq_dir.parent)
    assert [im["frame_id"] for im in coco["images"]] == [3, 4]
    assert [a["iscrowd"] for a in coco["annotations"]] == [0, 1]
    image_ids = {im["id"] for im in coco["images"]}
    assert all(a["image_id"] in image_ids for a in coco["annotations"])


def test_split_by_sequence_keeps_whole_sequences_apart(tmp_path: Path):
    names = [f"MOT17-{n:02d}-FRCNN" for n in (2, 4, 5, 9, 10, 11, 13)]
    dirs = [tmp_path / n for n in names]
    split = split_by_sequence(dirs, val_sequences=("MOT17-02", "MOT17-05"))
    assert [d.name for d in split["val"]] == ["MOT17-02-FRCNN", "MOT17-05-FRCNN"]
    assert len(split["train"]) == 5
    assert not set(split["train"]) & set(split["val"])


def test_split_by_sequence_rejects_unknown_sequence(tmp_path: Path):
    with pytest.raises(ValueError):
        split_by_sequence([tmp_path / "MOT17-04-FRCNN"], val_sequences=("MOT17-02",))


def test_image_ids_follow_sequence_number(seq_dir: Path):
    coco = to_coco([seq_dir], "full", image_root=seq_dir.parent)
    assert coco["videos"][0]["id"] == 99
    assert coco["images"][0]["id"] == 99 * 100_000 + 1
