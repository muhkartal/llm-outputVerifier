import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import set_seed

from config import Config, load_config
from data.dataset import create_dataloaders
from evaluation.metrics import (
    calculate_confidence_metrics,
    compute_classification_metrics,
    get_confusion_matrix,
)
from evaluation.visualizer import (
    create_confidence_histogram,
    create_confusion_matrix_plot,
    create_metrics_table,
)
from models.classifier import create_hallucination_classifier
from models.encoder import load_encoder_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the hallucination detector model")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config file"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--model_name", type=str, default="roberta-base", help="Transformer model name"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Evaluation batch size"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./evaluation", help="Output directory for results"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )

    return parser.parse_args()


def set_random_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def main():
    args = parse_args()

    if args.config:
        config = load_config(args.config)
    else:
        config = Config()

    set_random_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    encoder, tokenizer = load_encoder_model(args.model_name)
    model = create_hallucination_classifier(args.model_name)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    print("Creating dataloaders...")
    dataloaders = create_dataloaders(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        corruption_rate=config.data.corruption_rate,
    )

    test_dataloader = dataloaders["test"]

    print(f"Evaluating on {len(test_dataloader.dataset)} examples...")

    all_predictions = []
    all_labels = []
    all_confidences = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].cpu().numpy()

            outputs = model.predict(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            predictions = outputs["labels"].cpu().numpy()
            confidences = outputs["confidence"].cpu().numpy()

            all_predictions.extend(predictions)
            all_labels.extend(labels)
            all_confidences.extend(confidences)

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)

    print("Computing metrics...")

    metrics = compute_classification_metrics(all_predictions, all_labels)
    print(f"Metrics: {metrics}")

    confidence_metrics = calculate_confidence_metrics(
        all_predictions, all_labels, all_confidences
    )

    results = {
        "metrics": metrics,
        "confidence_metrics": confidence_metrics,
        "model_path": args.model_path,
        "model_name": args.model_name,
        "num_examples": len(all_labels),
    }

    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    cm_fig = create_confusion_matrix_plot(all_labels, all_predictions)
    cm_fig.savefig(output_dir / "confusion_matrix.png", bbox_inches="tight", dpi=300)

    hist_fig = create_confidence_histogram(all_confidences, all_labels)
    hist_fig.savefig(output_dir / "confidence_histogram.png", bbox_inches="tight", dpi=300)

    metrics_df = create_metrics_table(metrics)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)

    print(f"Results saved to {output_dir}")

    print("\nEvaluation Summary:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")

    if "auc" in metrics:
        print(f"  AUC: {metrics['auc']:.4f}")


if __name__ == "__main__":
    main()
