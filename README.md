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
make data      # MOT17 train GT -> COCO json (train_half / val_half / full)
make eval      # evaluate the detector on val_half, logged to MLflow
make viz       # per-sequence error analysis + annotated videos -> outputs/qualitative/
make mlflow    # MLflow UI on http://127.0.0.1:5000
```

## Data

Download [MOT17](https://motchallenge.net/data/MOT17/) and unzip `MOT17/train/*-FRCNN` into `data/raw/`.
Only the train split has public ground truth, so each train sequence is split by frame index:
first half for training, second half for validation (ByteTrack convention).
Pedestrians (class 1) are positives; person-like classes (static person, person on vehicle,
distractor, reflection) are ignore regions; vehicles and occluders are dropped.

Design choices, what is verified and what is assumed (with the ablations planned to check
them): [docs/DECISIONS.md](docs/DECISIONS.md).

## Results

### Detection, MOT17 val_half (2,659 frames, 54,042 boxes)

| Model | AP | AP50 | AP75 | AP small | AP medium | AP large | AR100 |
|---|---|---|---|---|---|---|---|
| RF-DETR Small, COCO pretrained, 512 px | 0.292 | 0.555 | 0.282 | 0.015 | 0.162 | 0.443 | 0.396 |

`make eval`, score threshold 0.01, Apple M4 (MPS), 42 ms/image.
