# detr-tracking

Real-time multi-object tracking with a DETR-family detector, from training to a deployed, tested API.

Work in progress.

## Stack

- Detector: [RF-DETR](https://github.com/roboflow/rf-detr) Small (Apache 2.0), COCO-pretrained, fine-tuned on pedestrians.
- Tracker: ByteTrack from [supervision](https://github.com/roboflow/supervision).
- Data: [MOT17](https://motchallenge.net/data/MOT17/) (CC BY-NC-SA 3.0).

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
make install   # create .venv from uv.lock
make lint      # ruff check + format check
make test      # fast tests (no weights needed)
make test-all  # also runs tests that download the model weights
make smoke     # run RF-DETR Small on a sample image -> outputs/smoke_detect.jpg
make data      # MOT17 train GT -> COCO json, split by sequence (train / val)
make eval      # evaluate the detector on the val sequences, logged to MLflow
make stats     # data statistics (border, visibility, box size) -> docs/DATA_STATS.md
make viz       # per-sequence error analysis + annotated videos -> outputs/qualitative/
make mlflow    # MLflow UI on http://127.0.0.1:5000
```

## Data

Download [MOT17](https://motchallenge.net/data/MOT17/) and unzip `MOT17/train/*-FRCNN` into `data/raw/`.
Only the 7 train sequences have public ground truth. They are split **by sequence**, so
training and validation share no scene, camera or person:

| Split | Sequences | Frames | Boxes |
|---|---|---|---|
| train | MOT17-04, -09, -10, -11, -13 | 3,879 | 86,799 |
| val | MOT17-02 (crowded, static camera), MOT17-05 (sparse, moving camera) | 1,437 | 25,498 (22.7 %) |

The official MOT17 test set (hidden GT, MOTChallenge server) is kept for one final submission.
Statistics: [docs/DATA_STATS.md](docs/DATA_STATS.md).
Pedestrians (class 1) are positives; person-like classes (static person, person on vehicle,
distractor, reflection) are ignore regions; vehicles and occluders are dropped.

Design choices, what is verified and what is assumed (with the ablations planned to check
them): [docs/DECISIONS.md](docs/DECISIONS.md).

## Results

### Detection, MOT17 val sequences (1,437 frames, 25,498 boxes)

| Model | AP | AP50 | AP75 | AP small | AP medium | AP large | AR100 |
|---|---|---|---|---|---|---|---|
| RF-DETR Small, COCO pretrained, 512 px | 0.232 | 0.484 | 0.197 | 0.020 | 0.145 | 0.447 | 0.326 |

`make eval`, original MOT17 GT (no clamping), score threshold 0.01, Apple M4 (MPS), 39 ms/image.
