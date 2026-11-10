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

Tracking: HOTA (primary), MOTA, IDF1 with TrackEval on the validation sequences (D8). Detection: COCO AP on the original GT, distractor classes as ignore regions.

## D1. Detector: RF-DETR Small, 512 px (default)

Smallest RF-DETR that is still accurate, Apache 2.0, built-in ONNX export. 512 px is the
model's native resolution. Ablation: 640 / 768 px for small pedestrians (MOT17-04, -02),
trading latency.

## D2. Fine-tuning: 10 epochs, last EMA checkpoint (assumed)

On the earlier half split (D8), a 20-epoch run peaked at epoch 7 (RF-DETR's val mAP 0.543) and slowly declined to 0.535: 10
epochs is enough. The last checkpoint is kept, never the best one on the validation
sequences, so validation numbers are not inflated by model selection. Ablation: 5 / 10 / 20 epochs.

## D3. Amodal boxes at the image border (assumed)

**Hypothesis H1.** Training on MOT17's full boxes at the image border, when the network can
predict them, instead of RF-DETR's default clamping, improves agreement with the official GT
near the border, hence detection and tracking under the MOT17 protocol, without hurting boxes
inside the image. H0: no difference beyond run-to-run noise.

**Why it matters.** 14.6 % of MOT17 pedestrian boxes (all 7 train sequences) extend past the image (26 % in MOT17-04,
22 % in MOT17-05). RF-DETR clamps them in three places: annotation loading (`ConvertCoco`),
random crop (torchvision), and prediction post-processing. A model trained on clamped boxes
scored AP 0.398 on the original GT and 0.531 on a clamped GT (half split, D8): the gap is
the convention.

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

Everything else identical: train sequences (D8), RF-DETR Small from COCO, 512 px, torchvision
augmentation, 10 epochs, effective batch 16, lr 1e-4, last EMA checkpoint, same inference
threshold, no clamping of predictions, same ByteTrack parameters fixed in advance.

Predictions to check: B predicts boxes past the border for truncated people (A does not);
higher AP / AP75 on border GT, equal on inner GT; higher HOTA (LocA, DetA) and MOTA, IDF1 not
lower, mostly on MOT17-04 and -05. Primary metric: HOTA on the validation sequences. Keep B if HOTA(B) -
HOTA(A) clearly exceeds |HOTA(C) - HOTA(A)| and inner boxes do not degrade.

Limits: one seed per condition.

**Boxes whose centre is outside the image (4.3 %, assumed acceptable).** They are clamped to
the visible part, never dropped. Options for later, cheapest first:

1. Keep as is: mostly people less than half visible (e.g. only feet at the top of MOT17-04).
2. Tracker-side completion, no retraining: when a detection touches the border, restore the
   size its track had while fully visible, anchored on the visible side. Also covers boxes
   larger than the image; does not help people entering the frame.
3. Pad the image (train and inference) so out-of-image centres fall inside the canvas; costs
   input resolution at a fixed 512 px, hurting small pedestrians.
4. Widen the box parameterisation (e.g. centre in [-0.5, 1.5] instead of a sigmoid's [0, 1]);
   intrusive in RF-DETR (head, target normalisation, post-processing, ONNX export).

Next to try if border errors weigh on HOTA: option 2.

## D4. Ignore regions as `iscrowd` during training (default, side effect noted)

Distractor classes become `iscrowd=1`. RF-DETR drops them from the targets, so a DETR learns
"nobody here" on static persons, reflections, etc. Evaluation ignores them, but it may make
the model hesitant on similar-looking pedestrians. Ablation: drop these images' crops from
the loss instead (needs a loss mask), or keep as is.

## D5. No visibility filter on training boxes (assumed)

Boxes with low visibility (8 % below 0.1 in the train sequences, 36 % in validation) are kept: removing them would teach
the DETR not to detect heavily occluded people, while TrackEval counts them. Ablation:
visibility threshold 0 / 0.1 / 0.25.

## D6. Augmentation: RF-DETR's torchvision pipeline, pinned (default)

Multi-scale resize, random crop, horizontal flip. Pinned to torchvision so Kaggle and local
runs share the pipeline that D3 was checked on (Kaggle would otherwise pick Albumentations,
whose crop clamps boxes).

## D7. Tracker: ByteTrack from `supervision` (default)

Parameters fixed in advance, not tuned per detector. Ablation: track and match thresholds,
second association pass on/off.

## D8. Train / validation split by sequence (decided)

Only MOT17's 7 train sequences have public GT; the official test set (7 other sequences, hidden
GT) is kept for one submission to the MOTChallenge server at the end of the project.

The train sequences are split **by sequence**, so the two splits share no scene, camera or
person (`VAL_SEQUENCES` in `detr_tracking.data.mot17`):

| Split | Sequences | Frames | Boxes | Pedestrians / frame |
|---|---|---|---|---|
| train | MOT17-04, -09, -10, -11, -13 | 3,879 | 86,799 | 45, 10, 20, 10, 16 |
| val | MOT17-02, -05 | 1,437 | 25,498 (22.7 %) | 31, 8 |

Why these two: a crowded static scene (MOT17-02) and a sparse moving-camera one (MOT17-05,
640x480), while training keeps the full density range, including the most crowded scene
(MOT17-04). A three-density validation (crowded, medium, sparse) is not possible near 80/20:
the smallest such set (MOT17-02, -13, -09) already holds 32 % of the boxes. MOT17-09 and -10
look like the same shopping street (Australian shop signs in both), so they stay on the same
side.

Known gaps: validation is more occluded than training (mean visibility 0.44 vs 0.64; 36 % of
val boxes are less than 10 % visible, mostly MOT17-02) and has fewer boxes past the image
border (7.3 % vs 16.7 %). Validation numbers are therefore harder than training-scene numbers,
and weigh occlusion more than the border.

History: the first experiments used the "half" split (first half of each sequence for
training, second half for validation, as in CenterTrack and ByteTrack). It was dropped
because the halves share scenes and 45 % of identities. Numbers quoted in D2 and D3 come from
that split; the first fine-tuned model was trained on the first half of MOT17-02 and -05 and
is not evaluated on the new validation.
