"""MOT17 ground truth parsing and conversion to COCO detection format.

MOT17 `gt.txt` columns: frame, track_id, x, y, w, h, consider_flag, class, visibility.
Only the 7 train sequences have public ground truth. They are split by sequence (whole
scenes): `VAL_SEQUENCES` go to validation, the others to training, so the two splits share no
camera, environment or person. `split_frames` (first/second half of a sequence) is kept for
other protocols.

Box policy for detection evaluation:
- class 1 (pedestrian) with consider_flag 1 -> positive box.
- person-like classes (2 person on vehicle, 7 static person, 8 distractor, 12 reflection)
  and pedestrians with consider_flag 0 -> ignore region (`iscrowd=1`): a detection there is
  neither a true nor a false positive.
- every other class (vehicles, occluders) is dropped.
"""

from __future__ import annotations

import configparser
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

PEDESTRIAN = 1
IGNORED_PERSON_LIKE = frozenset({2, 7, 8, 12})
Split = Literal["train_half", "val_half", "full"]

# A crowded static scene (MOT17-02, 31 pedestrians/frame) and a sparse moving-camera one
# (MOT17-05, 8/frame, 640x480): 22.7 % of the boxes. See docs/DECISIONS.md, D8.
VAL_SEQUENCES = ("MOT17-02", "MOT17-05")


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


def sequence_number(name: str) -> int:
    """`MOT17-04-FRCNN` -> 4."""
    return int(name.split("-")[1])


def split_by_sequence(
    seq_dirs: list[Path], val_sequences: tuple[str, ...] = VAL_SEQUENCES
) -> dict[str, list[Path]]:
    """Assign whole sequences to `train` or `val` (matched on the `MOT17-XX` prefix)."""
    val = [d for d in seq_dirs if any(d.name.startswith(v) for v in val_sequences)]
    missing = [v for v in val_sequences if not any(d.name.startswith(v) for d in val)]
    if missing:
        raise ValueError(f"validation sequences not found: {missing}")
    return {"train": [d for d in seq_dirs if d not in val], "val": val}


def to_coco(seq_dirs: list[Path], split: Split, image_root: Path) -> dict:
    """Build a COCO detection dict (single `person` category, id 1) for some sequences.

    `split` selects frames within each sequence ("full" for all). `file_name` is relative to
    `image_root`. Video and image ids derive from the sequence number, so they are unique
    across splits.
    """
    images: list[dict] = []
    annotations: list[dict] = []
    videos: list[dict] = []

    for seq_dir in sorted(seq_dirs):
        info = read_seqinfo(seq_dir)
        video_id = sequence_number(info.name)
        frames = split_frames(info.length, split)
        videos.append({"id": video_id, "name": info.name, "frame_rate": info.frame_rate})
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
        "info": {"description": f"MOT17 {', '.join(v['name'] for v in videos)} ({split})"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person"}],
        "videos": videos,
    }


def write_splits(
    train_dir: Path, out_dir: Path, detector: str = "FRCNN", verbose: bool = True
) -> dict[str, Path]:
    """Write `train.json` and `val.json` (whole sequences) for the MOT17 train sequences."""
    seq_dirs = sorted(train_dir.glob(f"MOT17-*-{detector}"))
    if not seq_dirs:
        raise FileNotFoundError(f"no MOT17 sequences in {train_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, dirs in split_by_sequence(seq_dirs).items():
        coco = to_coco(dirs, "full", image_root=train_dir)
        paths[split] = out_dir / f"{split}.json"
        paths[split].write_text(json.dumps(coco))
        if verbose:
            boxes = sum(1 for a in coco["annotations"] if not a["iscrowd"])
            names = ", ".join(v["name"] for v in coco["videos"])
            print(f"{split:5s} {len(coco['images']):5d} images {boxes:6d} boxes  [{names}]")
    return paths
