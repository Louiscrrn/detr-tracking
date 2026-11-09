"""MOT17 ground truth parsing and conversion to COCO detection format.

MOT17 `gt.txt` columns: frame, track_id, x, y, w, h, consider_flag, class, visibility.
Only the train split has public ground truth, so each train sequence is split in two
halves by frame index (first half -> train, second half -> val), as in ByteTrack.

Box policy for detection evaluation:
- class 1 (pedestrian) with consider_flag 1 -> positive box.
- person-like classes (2 person on vehicle, 7 static person, 8 distractor, 12 reflection)
  and pedestrians with consider_flag 0 -> ignore region (`iscrowd=1`): a detection there is
  neither a true nor a false positive.
- every other class (vehicles, occluders) is dropped.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

PEDESTRIAN = 1
IGNORED_PERSON_LIKE = frozenset({2, 7, 8, 12})
Split = Literal["train_half", "val_half", "full"]


@dataclass(frozen=True)
class SequenceInfo:
    name: str
    path: Path
    length: int
    width: int
    height: int
    frame_rate: int
    image_ext: str = ".jpg"

    def image_path(self, frame: int) -> Path:
        return self.path / "img1" / f"{frame:06d}{self.image_ext}"


def read_seqinfo(seq_dir: Path) -> SequenceInfo:
    parser = configparser.ConfigParser()
    parser.read(seq_dir / "seqinfo.ini")
    seq = parser["Sequence"]
    return SequenceInfo(
        name=seq["name"],
        path=seq_dir,
        length=int(seq["seqLength"]),
        width=int(seq["imWidth"]),
        height=int(seq["imHeight"]),
        frame_rate=int(seq["frameRate"]),
        image_ext=seq.get("imExt", ".jpg"),
    )


def split_frames(length: int, split: Split) -> range:
    """1-based frame indices of a split. Train gets the first `length // 2` frames."""
    half = length // 2
    if split == "train_half":
        return range(1, half + 1)
    if split == "val_half":
        return range(half + 1, length + 1)
    return range(1, length + 1)


def read_gt(path: Path) -> np.ndarray:
    """Load `gt.txt` as an (N, 9) float array."""
    gt = np.loadtxt(path, delimiter=",", ndmin=2)
    return gt[:, :9]


def classify_rows(gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks (positive, ignore) over the rows of a MOT17 gt array."""
    cls = gt[:, 7].astype(int)
    flag = gt[:, 6].astype(int)
    pedestrian = cls == PEDESTRIAN
    positive = pedestrian & (flag == 1)
    ignore = (pedestrian & (flag == 0)) | np.isin(cls, list(IGNORED_PERSON_LIKE))
    return positive, ignore


def to_coco(seq_dirs: list[Path], split: Split, image_root: Path) -> dict:
    """Build a COCO detection dict (single `person` category, id 1) for one split.

    `file_name` is relative to `image_root`. Image ids are unique across sequences.
    """
    images: list[dict] = []
    annotations: list[dict] = []
    videos: list[dict] = []

    for video_id, seq_dir in enumerate(sorted(seq_dirs), start=1):
        info = read_seqinfo(seq_dir)
        frames = split_frames(info.length, split)
        videos.append({"id": video_id, "name": info.name})
        image_ids = {}
        for frame in frames:
            image_id = video_id * 100_000 + frame
            image_ids[frame] = image_id
            images.append(
                {
                    "id": image_id,
                    "file_name": str(info.image_path(frame).relative_to(image_root)),
                    "width": info.width,
                    "height": info.height,
                    "video_id": video_id,
                    "frame_id": frame,
                }
            )

        gt = read_gt(seq_dir / "gt" / "gt.txt")
        positive, ignore = classify_rows(gt)
        for row, is_pos, is_ign in zip(gt, positive, ignore, strict=True):
            frame = int(row[0])
            if not (is_pos or is_ign) or frame not in image_ids:
                continue
            x, y, w, h = (float(v) for v in row[2:6])
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_ids[frame],
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": int(is_ign),
                    "track_id": int(row[1]),
                    "visibility": float(row[8]),
                }
            )

    return {
        "info": {"description": f"MOT17 {split} (pedestrians)"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person"}],
        "videos": videos,
    }
