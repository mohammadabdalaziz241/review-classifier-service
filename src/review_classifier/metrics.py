"""Classification metrics, shared by training and by service evaluation.

Mirrors ``evaluate_predictions`` in the project notebook, which uses
scikit-learn's ``precision_recall_fscore_support(average="macro",
zero_division=0)``: macro averages are taken over the labels that occur in
either ``y_true`` or ``y_pred``. Implemented without scikit-learn so the
service and its evaluation do not depend on it.
"""

from __future__ import annotations

from collections.abc import Sequence


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], num_classes: int = 2
) -> dict:
    if len(y_true) != len(y_pred):
        raise ValueError(f"{len(y_true)} labels but {len(y_pred)} predictions")
    if not y_true:
        raise ValueError("Cannot compute metrics for zero examples")
    for value in (*y_true, *y_pred):
        if not 0 <= value < num_classes:
            raise ValueError(f"Label {value} is outside 0..{num_classes - 1}")

    # confusion[t][p]: number of examples with true label t predicted as p.
    confusion = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred, strict=True):
        confusion[t][p] += 1

    metrics: dict = {}
    precisions, recalls, f1s = [], [], []
    present = set(y_true) | set(y_pred)
    for c in range(num_classes):
        tp = confusion[c][c]
        fp = sum(confusion[t][c] for t in range(num_classes)) - tp
        fn = sum(confusion[c]) - tp
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[f"class_{c}_precision"] = precision
        metrics[f"class_{c}_recall"] = recall
        metrics[f"class_{c}_f1"] = f1
        metrics[f"class_{c}_support"] = tp + fn
        if c in present:
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

    correct = sum(confusion[c][c] for c in range(num_classes))
    metrics["accuracy"] = correct / len(y_true)
    metrics["macro_precision"] = sum(precisions) / len(precisions)
    metrics["macro_recall"] = sum(recalls) / len(recalls)
    metrics["macro_f1"] = sum(f1s) / len(f1s)
    metrics["n"] = len(y_true)
    metrics["confusion_matrix"] = confusion
    return metrics
