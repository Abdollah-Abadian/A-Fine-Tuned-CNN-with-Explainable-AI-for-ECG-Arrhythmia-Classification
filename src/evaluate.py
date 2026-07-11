"""
Evaluation utilities: metrics, confusion matrix, bootstrap confidence intervals,
McNemar test, classification report.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from scipy.stats import chi2_contingency, norm


def compute_metrics(y_true, y_pred):
    """Return dict with accuracy, precision, recall, f1 (macro & weighted)."""
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
    }


def per_class_metrics(y_true, y_pred, labels):
    """Return dict of per-class precision, recall, f1, support."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    results = {}
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        support = tp + fn
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        results[label] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support,
            "specificity": specificity,
        }
    return results


def bootstrap_confidence_interval(y_true, y_pred, n_bootstrap=1000, metric_func=accuracy_score, alpha=0.05):
    """Compute bootstrap CI for a given metric."""
    rng = np.random.default_rng(42)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        scores.append(metric_func(y_true[idx], y_pred[idx]))
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    return lower, upper, np.std(scores)


def mcnemar_test(y_true, y_pred1, y_pred2):
    """McNemar test for two classifiers."""
    # Compare where they differ
    diff = (y_pred1 != y_pred2)
    y1_correct = (y_pred1 == y_true)
    y2_correct = (y_pred2 == y_true)
    # Counts: both correct, both wrong, 1 correct 2 wrong, 1 wrong 2 correct
    b = np.sum(diff & (y1_correct == True) & (y2_correct == False))   # 1 correct, 2 wrong
    c = np.sum(diff & (y1_correct == False) & (y2_correct == True))   # 1 wrong, 2 correct
    # McNemar chi-square = (b-c)^2 / (b+c)
    if b + c == 0:
        return 0, 1.0  # no difference
    chi2 = ((b - c) ** 2) / (b + c)
    p = 1 - chi2_contingency(np.array([[b, c], [c, b]]))[1]   # approximate
    return chi2, p
