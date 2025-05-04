from typing import Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)


def compute_classification_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    average: str = "binary",
) -> Dict[str, float]:
    if average == "binary" and len(np.unique(labels)) > 2:
        average = "macro"

    accuracy = accuracy_score(labels, predictions)

    precision = precision_score(
        labels, predictions, average=average, zero_division=0
    )

    recall = recall_score(
        labels, predictions, average=average, zero_division=0
    )

    f1 = f1_score(
        labels, predictions, average=average, zero_division=0
    )

    try:
        if len(np.unique(labels)) == 2:
            auc = roc_auc_score(labels, predictions)
        else:
            auc = None
    except:
        auc = None

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    if auc is not None:
        metrics["auc"] = float(auc)

    return metrics


def get_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    return confusion_matrix(labels, predictions)


def calculate_confidence_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    confidence_scores: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> Dict[str, List[float]]:
    if thresholds is None:
        thresholds = np.linspace(0.5, 0.95, 10)

    results = {
        "thresholds": thresholds.tolist(),
        "precision": [],
        "recall": [],
        "accuracy": [],
        "coverage": [],
    }

    for threshold in thresholds:
        mask = confidence_scores >= threshold
        coverage = mask.mean()

        if coverage == 0:
            precision = recall = accuracy = 0
        else:
            precision = precision_score(
                labels[mask], predictions[mask], average="binary", zero_division=0
            )
            recall = recall_score(
                labels[mask], predictions[mask], average="binary", zero_division=0
            )
            accuracy = accuracy_score(labels[mask], predictions[mask])

        results["precision"].append(float(precision))
        results["recall"].append(float(recall))
        results["accuracy"].append(float(accuracy))
        results["coverage"].append(float(coverage))

    return results
