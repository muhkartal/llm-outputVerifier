import argparse
import random
from pathlib import Path

import numpy as np
import torch
from transformers import set_seed

from config import Config, get_default_config, load_config, save_config
from data.dataset import create_dataloaders
from models.classifier import create_hallucination_classifier
from models.encoder import (
    apply_mixed_precision,
    enable_gradient_checkpointing,
    load_encoder_model,
)
from training.trainer import HallucinationTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train the hallucination detector model")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Transformer model name"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--corruption_rate",
        type=float,
        default=None,
        help="Rate of synthetic hallucinations"
    )

    return parser.parse_args()


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def main():
    args = parse_args()

    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()

    if args.model_name:
        config.model.model_name = args.model_name

    if args.batch_size:
        config.data.train_batch_size = args.batch_size

    if args.learning_rate:
        config.training.learning_rate = args.learning_rate

    if args.num_epochs:
        config.training.num_epochs = args.num_epochs

    if args.seed:
        config.training.seed = args.seed

    if args.no_wandb:
        config.training.use_wandb = False

    if args.corruption_rate:
        config.data.corruption_rate = args.corruption_rate

    if args.output_dir:
        config.training.save_dir = args.output_dir

    output_dir = Path(config.training.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_config(config, output_dir / "config.json")

    set_random_seed(config.training.seed)

    encoder, tokenizer = load_encoder_model(config.model.model_name)

    if config.model.gradient_checkpointing:
        encoder = enable_gradient_checkpointing(encoder)

    model = create_hallucination_classifier(
        model_name=config.model.model_name,
        dropout_rate=config.model.dropout_rate,
        freeze_encoder=config.model.freeze_encoder,
    )

    if config.model.mixed_precision:
        model = apply_mixed_precision(model)

    dataloaders = create_dataloaders(
        tokenizer=tokenizer,
        batch_size=config.data.train_batch_size,
        max_length=config.data.max_length,
        corruption_rate=config.data.corruption_rate,
        num_workers=config.data.num_workers,
    )

    trainer = HallucinationTrainer(
        model=model,
        train_dataloader=dataloaders["train"],
        val_dataloader=dataloaders["val"],
        test_dataloader=dataloaders["test"],
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        num_epochs=config.training.num_epochs,
        warmup_ratio=config.training.warmup_ratio,
        log_interval=config.training.log_interval,
        save_dir=config.training.save_dir,
        use_wandb=config.training.use_wandb,
        mixed_precision=config.training.fp16,
        project_name=config.training.project_name,
    )

    print(f"Starting training with config:")
    print(f"Model: {config.model.model_name}")
    print(f"Batch size: {config.data.train_batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Epochs: {config.training.num_epochs}")
    print(f"Corruption rate: {config.data.corruption_rate}")
    print(f"Output directory: {config.training.save_dir}")

    results = trainer.train()

    print(f"Training completed.")
    print(f"Best validation accuracy: {results['best_val_accuracy']:.4f} at epoch {results['best_epoch'] + 1}")


if __name__ == "__main__":
    main()
