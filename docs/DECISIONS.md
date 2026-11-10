# Design decisions

The pipeline is built end to end first, on choices that are reasonable but not all proven.
Each choice is logged here with its status and, when it is an assumption, the ablation that
would confirm or reject it. Ablations are run at the end of the project.

Status: **verified** (measured or checked against a primary source), **assumed** (adopted
without proof, ablation planned), **default** (library or common practice, not questioned yet).

## Evaluation protocol (verified)

MOT17 ground truth is used as released: boxes that extend past the image are **not** clamped,
neither in the GT nor in the predictions.

- MOT16 paper (MOT17 reuses its annotation rules), Milan et al. 2016, [arXiv:1603.00831](https://arxiv.org/abs/1603.00831),
  Sec. II-B: *"If a person is cropped by the image border, the box is estimated beyond the
  original frame to represent the entire person and to estimate the level of cropping."*
  Sec. II-C: *"The box coordinates may exceed the visible area."*
- Official evaluation code, [TrackEval `mot_challenge_2d_box.py`](https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/datasets/mot_challenge_2d_box.py):
  no clipping of GT or tracker boxes before IoU; distractor classes (person on vehicle, static
  person, distractor, reflection) remove the predictions they match (IoU >= 0.5); GT rows with
  flag 0 are dropped; visibility is not used.
- RF-DETR clamps predicted boxes to the image in `PostProcess`; we disable it for every model
  (`enable_unclamped_predictions`). Checked: no change for a model trained on clamped boxes
  (AP 0.399 both ways on 300 val frames).

Tracking: HOTA (primary), MOTA, IDF1 with TrackEval on `val_half` (second half of each MOT17
train sequence). Detection: COCO AP on the original GT, distractor classes as ignore regions.

## D1. Detector: RF-DETR Small, 512 px (default)

Smallest RF-DETR that is still accurate, Apache 2.0, built-in ONNX export. 512 px is the
model's native resolution. Ablation: 640 / 768 px for small pedestrians (MOT17-04, -02),
trading latency.

## D2. Fine-tuning: 10 epochs, last EMA checkpoint (assumed)

A 20-epoch run peaked at epoch 7 (RF-DETR's val mAP 0.543) and slowly declined to 0.535: 10
epochs is enough. The last checkpoint is kept, never the best one on `val_half`, since
`val_half` is also the test set. Ablation: 5 / 10 / 20 epochs.

## D3. Amodal boxes at the image border (assumed)

**Hypothesis H1.** Training on MOT17's full boxes at the image border, when the network can
predict them, instead of RF-DETR's default clamping, improves agreement with the official GT
near the border, hence detection and tracking under the MOT17 protocol, without hurting boxes
inside the image. H0: no difference beyond run-to-run noise.

**Why it matters.** 14.6 % of MOT17 pedestrian boxes extend past the image (26 % in MOT17-04,
22 % in MOT17-05). RF-DETR clamps them in three places: annotation loading (`ConvertCoco`),
random crop (torchvision), and prediction post-processing. A model trained on clamped boxes
scored AP 0.398 on the original GT and 0.531 on a clamped GT: the gap is the convention.

**What we do** (`detr_tracking.data.amodal`): keep a box whole if its centre lies in the image
and its size does not exceed the image (a sigmoid-parameterised DETR can predict it), clamp it
otherwise, never drop it (a dropped box teaches a DETR that nobody is there). This keeps 62 %
of the border boxes whole; 4.3 % of all boxes have their centre outside the image and stay out
of reach. Checked on RF-DETR's real train pipeline: 0 % of target boxes past the border by
default, 8.9 % with the patch (`scripts/check_amodal_pipeline.py`,
`outputs/checks/amodal_pipeline.jpg`).

**Ablation to run.**

| Run | Border boxes | Seed |
|---|---|---|
| A (control) | clamped, `--clamp-boxes` | 42 |
| B (adopted) | amodal | 42 |
| C (noise) | clamped | 43 |

Everything else identical: train_half, RF-DETR Small from COCO, 512 px, torchvision
augmentation, 10 epochs, effective batch 16, lr 1e-4, last EMA checkpoint, same inference
threshold, no clamping of predictions, same ByteTrack parameters fixed in advance.

Predictions to check: B predicts boxes past the border for truncated people (A does not);
higher AP / AP75 on border GT, equal on inner GT; higher HOTA (LocA, DetA) and MOTA, IDF1 not
lower, mostly on MOT17-04 and -05. Primary metric: HOTA on `val_half`. Keep B if HOTA(B) -
HOTA(A) clearly exceeds |HOTA(C) - HOTA(A)| and inner boxes do not degrade.

Limits: `val_half` shares scenes with `train_half`; one seed per condition.

## D4. Ignore regions as `iscrowd` during training (default, side effect noted)

Distractor classes become `iscrowd=1`. RF-DETR drops them from the targets, so a DETR learns
"nobody here" on static persons, reflections, etc. Evaluation ignores them, but it may make
the model hesitant on similar-looking pedestrians. Ablation: drop these images' crops from
the loss instead (needs a loss mask), or keep as is.

## D5. No visibility filter on training boxes (assumed)

Boxes with low visibility (13 % below 0.1 in train_half) are kept: removing them would teach
the DETR not to detect heavily occluded people, while TrackEval counts them. Ablation:
visibility threshold 0 / 0.1 / 0.25.

## D6. Augmentation: RF-DETR's torchvision pipeline, pinned (default)

Multi-scale resize, random crop, horizontal flip. Pinned to torchvision so Kaggle and local
runs share the pipeline that D3 was checked on (Kaggle would otherwise pick Albumentations,
whose crop clamps boxes).

## D7. Tracker: ByteTrack from `supervision` (default)

Parameters fixed in advance, not tuned per detector. Ablation: track and match thresholds,
second association pass on/off.
