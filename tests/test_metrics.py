import numpy as np

from polypseg.metrics import Confusion, binarize, image_metrics, size_bucket, summarize


def test_confusion_counts_and_metrics():
    pred = np.array([[1, 1, 0, 0]])
    target = np.array([[1, 0, 1, 0]])
    c = Confusion.from_masks(pred, target)
    assert (c.tp, c.fp, c.fn, c.tn) == (1, 1, 1, 1)
    m = c.as_dict()
    assert abs(m["dice"] - 0.5) < 1e-6 and abs(m["iou"] - 1 / 3) < 1e-6
    assert abs(m["precision"] - 0.5) < 1e-6 and abs(m["recall"] - 0.5) < 1e-6
    assert abs(m["accuracy"] - 0.5) < 1e-6
    # Dice and IoU are linked: dice = 2 iou / (1 + iou)
    assert abs(m["dice"] - 2 * m["iou"] / (1 + m["iou"])) < 1e-6


def test_empty_prediction_and_target_is_perfect_and_confusions_add():
    empty = Confusion.from_masks(np.zeros((4, 4)), np.zeros((4, 4)))
    assert empty.dice() == 1.0 and empty.iou() == 1.0
    total = empty + Confusion(1, 2, 3, 4)
    assert (total.tp, total.fp, total.fn, total.tn) == (1, 2, 3, 20)


def test_summarize_mean_vs_dataset_level():
    per_image = [image_metrics(np.ones((2, 2)), np.ones((2, 2))),
                 image_metrics(np.zeros((2, 2)), np.ones((2, 2)))]
    pooled = Confusion(4, 0, 4, 0)
    s = summarize(per_image, pooled)
    assert s["num_images"] == 2
    assert abs(s["mean_per_image"]["dice"] - 0.5) < 1e-6
    assert abs(s["dataset_level"]["dice"] - 2 * 4 / (2 * 4 + 4)) < 1e-6


def test_binarize_and_size_buckets():
    probs = np.array([0.2, 0.5, 0.9])
    np.testing.assert_array_equal(binarize(probs), [0, 1, 1])
    assert size_bucket(0.01).startswith("small")
    assert size_bucket(0.1).startswith("medium")
    assert size_bucket(0.5).startswith("large")
