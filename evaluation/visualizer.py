from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def create_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "Blues",
    normalize: bool = False,
) -> plt.Figure:
    if class_names is None:
        class_names = ["Grounded", "Hallucination"]

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap=cmap,
        square=True,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    return fig


def create_confidence_histogram(
    confidences: np.ndarray,
    labels: np.ndarray,
    figsize: Tuple[int, int] = (10, 6),
    bins: int = 20,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    df = pd.DataFrame({
        "Confidence": confidences,
        "Type": ["Hallucination" if label == 1 else "Grounded" for label in labels],
    })

    sns.histplot(
        data=df,
        x="Confidence",
        hue="Type",
        bins=bins,
        alpha=0.6,
        ax=ax,
    )

    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution by Prediction Type")

    return fig


def create_metrics_table(
    metrics: Dict[str, float],
    precision: int = 4,
) -> pd.DataFrame:
    metrics_df = pd.DataFrame({
        "Metric": list(metrics.keys()),
        "Value": [round(value, precision) for value in metrics.values()],
    })

    return metrics_df


def plot_hallucination_distribution(
    steps: List[str],
    hallucination_labels: List[int],
    confidences: List[float],
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    df = pd.DataFrame({
        "Step": [f"Step {i+1}" for i in range(len(steps))],
        "Is Hallucination": hallucination_labels,
        "Confidence": confidences,
        "Step Text": [text[:50] + "..." if len(text) > 50 else text for text in steps],
    })

    # Sort by hallucination status and confidence
    df = df.sort_values(["Is Hallucination", "Confidence"], ascending=[False, False])

    # Plot horizontal bar chart
    bars = sns.barplot(
        data=df,
        y="Step",
        x="Confidence",
        hue="Is Hallucination",
        palette=["green", "red"],
        orient="h",
        ax=ax,
    )

    # Add step text as annotations
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(
            0.02,
            i,
            row["Step Text"],
            va="center",
            color="black",
            fontsize=8,
        )

    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Reasoning Step")
    ax.set_title("Hallucination Detection Results by Reasoning Step")
    ax.legend(title="Is Hallucination", labels=["No", "Yes"])

    return fig
