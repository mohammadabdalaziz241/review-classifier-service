import pytest

from review_classifier.metrics import classification_metrics


def test_known_binary_example():
    y_true = [0, 0, 0, 1, 1]
    y_pred = [0, 0, 1, 1, 0]
    m = classification_metrics(y_true, y_pred)
    # class 0: tp=2 fp=1 fn=1 -> p=r=f1=2/3; class 1: tp=1 fp=1 fn=1 -> p=r=f1=1/2
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["class_0_f1"] == pytest.approx(2 / 3)
    assert m["class_1_f1"] == pytest.approx(1 / 2)
    assert m["macro_f1"] == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert m["class_0_support"] == 3
    assert m["class_1_support"] == 2
    assert m["confusion_matrix"] == [[2, 1], [1, 1]]
    assert m["n"] == 5


def test_never_predicted_class_scores_zero_without_error():
    m = classification_metrics([0, 1, 1], [0, 0, 0])
    assert m["class_1_precision"] == 0.0
    assert m["class_1_f1"] == 0.0


def test_macro_average_skips_classes_absent_from_truth_and_predictions():
    # Matches sklearn's average="macro", which only averages over present labels.
    m = classification_metrics([0, 0], [0, 0])
    assert m["macro_f1"] == 1.0


@pytest.mark.parametrize(
    ("y_true", "y_pred", "message"),
    [([0, 1], [0], "2 labels but 1"), ([], [], "zero examples"), ([0, 2], [0, 1], "outside")],
)
def test_rejects_bad_input(y_true, y_pred, message):
    with pytest.raises(ValueError, match=message):
        classification_metrics(y_true, y_pred)


def test_agrees_with_scikit_learn_as_used_in_the_notebook():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    import random

    rng = random.Random(0)
    for _ in range(50):
        n = rng.randint(1, 60)
        y_true = [rng.randint(0, 1) for _ in range(n)]
        y_pred = [rng.randint(0, 1) for _ in range(n)]
        ours = classification_metrics(y_true, y_pred)
        p, r, f1, _ = sklearn_metrics.precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        assert ours["macro_f1"] == pytest.approx(f1)
        assert ours["macro_precision"] == pytest.approx(p)
        assert ours["macro_recall"] == pytest.approx(r)
        assert ours["accuracy"] == pytest.approx(sklearn_metrics.accuracy_score(y_true, y_pred))


def test_balanced_class_weights_match_sklearn():
    from review_classifier.train import balanced_class_weights

    sklearn_utils = pytest.importorskip("sklearn.utils")
    import numpy as np

    labels = [0] * 86 + [1] * 14
    expected = sklearn_utils.compute_class_weight("balanced", classes=np.array([0, 1]), y=labels)
    assert balanced_class_weights(labels) == pytest.approx(list(expected))
    with pytest.raises(ValueError, match="Every class"):
        balanced_class_weights([0, 0, 0])
