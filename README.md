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
```
