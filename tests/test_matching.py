import numpy as np

from detr_tracking.detection.matching import iou_matrix, match_frame


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=float)
    np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 1 / 3, 0.0]])


def test_match_frame_counts_tp_fp_missed_and_ignored():
    gt = np.array([[0, 0, 10, 20], [50, 50, 60, 70]], dtype=float)
    ignore = np.array([[100, 100, 120, 140]], dtype=float)
    pred = np.array(
        [
            [0, 0, 10, 20],  # TP
            [1, 0, 11, 20],  # duplicate of the same person -> FP
            [102, 102, 118, 138],  # on ignore region -> ignored
            [200, 200, 210, 220],  # FP
        ],
        dtype=float,
    )
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    m = match_frame(pred, scores, gt, ignore)
    assert m.tp.tolist() == [True, False, False, False]
    assert m.fp.tolist() == [False, True, False, True]
    assert m.ignored.tolist() == [False, False, True, False]
    assert m.missed.tolist() == [False, True]


def test_match_frame_empty():
    m = match_frame(np.zeros((0, 4)), np.zeros(0), np.zeros((0, 4)), np.zeros((0, 4)))
    assert len(m.tp) == 0 and len(m.missed) == 0
