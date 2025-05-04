from typing import Dict, List, Optional, Tuple, Union

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from data.preprocessing import extract_step_inputs
from evaluation.metrics import compute_classification_metrics


class HallucinationTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        num_epochs: int = 3,
        warmup_ratio: float = 0.1,
        log_interval: int = 100,
        save_dir: str = "./checkpoints",
        use_wandb: bool = True,
        mixed_precision: bool = True,
        project_name: str = "hallucination-hunter",
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.log_interval = log_interval
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project=project_name)
            wandb.config.update({
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "num_epochs": num_epochs,
                "warmup_ratio": warmup_ratio,
                "batch_size": train_dataloader.batch_size,
            })

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.total_steps = len(train_dataloader) * num_epochs
        self.warmup_steps = int(self.total_steps * warmup_ratio)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )

        self.criterion = nn.CrossEntropyLoss()

        self.mixed_precision = mixed_precision
        self.scaler = GradScaler() if mixed_precision else None

    def train(self) -> Dict[str, Union[float, List[float]]]:
        best_val_accuracy = 0.0
        best_epoch = 0

        train_losses = []
        val_metrics = []

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")

            train_loss = self._train_epoch()
            train_losses.append(train_loss)

            val_results = self.evaluate(self.val_dataloader)
            val_metrics.append(val_results)

            if val_results["accuracy"] > best_val_accuracy:
                best_val_accuracy = val_results["accuracy"]
                best_epoch = epoch
                self._save_checkpoint(f"best_model.pt")

            self._save_checkpoint(f"model_epoch_{epoch + 1}.pt")

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Metrics: {val_results}")

            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_accuracy": val_results["accuracy"],
                    "val_precision": val_results["precision"],
                    "val_recall": val_results["recall"],
                    "val_f1": val_results["f1"],
                })

        if self.test_dataloader:
            self.model.load_state_dict(torch.load(self.save_dir / "best_model.pt"))
            test_results = self.evaluate(self.test_dataloader)

            print(f"Test Metrics: {test_results}")

            if self.use_wandb:
                wandb.log({
                    "test_accuracy": test_results["accuracy"],
                    "test_precision": test_results["precision"],
                    "test_recall": test_results["recall"],
                    "test_f1": test_results["f1"],
                })

        return {
            "train_losses": train_losses,
            "val_metrics": val_metrics,
            "best_val_accuracy": best_val_accuracy,
            "best_epoch": best_epoch,
        }

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_dataloader)

        progress_bar = tqdm(self.train_dataloader, desc="Training")

        for batch_idx, batch in enumerate(progress_bar):
            inputs, labels = extract_step_inputs(batch)

            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if self.mixed_precision:
                with autocast():
                    outputs = self.model(**inputs)
                    loss = self.criterion(outputs["logits"], labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(**inputs)
                loss = self.criterion(outputs["logits"], labels)

                loss.backward()
                self.optimizer.step()

            self.scheduler.step()

            total_loss += loss.item()

            if (batch_idx + 1) % self.log_interval == 0 or (batch_idx + 1) == num_batches:
                progress_bar.set_postfix({"loss": total_loss / (batch_idx + 1)})

                if self.use_wandb:
                    wandb.log({
                        "train_step_loss": loss.item(),
                        "learning_rate": self.scheduler.get_last_lr()[0],
                    })

        return total_loss / num_batches

    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                inputs, labels = extract_step_inputs(batch)

                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = labels.to(self.device)

                outputs = self.model(**inputs)
                logits = outputs["logits"]

                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        metrics = compute_classification_metrics(
            np.array(all_preds),
            np.array(all_labels),
        )

        return metrics

    def _save_checkpoint(self, filename: str) -> None:
        torch.save(self.model.state_dict(), self.save_dir / filename)

        if self.use_wandb:
            wandb.save(str(self.save_dir / filename))
