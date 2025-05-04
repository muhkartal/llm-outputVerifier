from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from data.augmentation import create_synthetic_hallucinations


class HallucinationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        split: str = "train",
        corruption_rate: float = 0.3,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split

        gsm8k = load_dataset("gsm8k", "main")

        if split == "train":
            self.dataset = gsm8k["train"]
        elif split == "val":
            validation_size = max(int(len(gsm8k["train"]) * 0.1), 100)
            splits = gsm8k["train"].train_test_split(test_size=validation_size, seed=42)
            self.dataset = splits["test"]
        else:  # Test split
            self.dataset = gsm8k["test"]

        self.augmented_dataset = create_synthetic_hallucinations(
            self.dataset, corruption_rate=corruption_rate
        )

        self.flattened_data = self._flatten_data()

    def _flatten_data(self) -> List[Dict[str, Union[str, int]]]:
        flattened = []

        for example in self.augmented_dataset:
            steps = example["reasoning_steps"]
            labels = example["hallucination_labels"]
            question = example["question"]

            for step, label in zip(steps, labels):
                flattened.append({
                    "question": question,
                    "step": step,
                    "label": label
                })

        return flattened

    def __len__(self) -> int:
        return len(self.flattened_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.flattened_data[idx]

        question = item["question"]
        step = item["step"]
        label = item["label"]

        encoding = self.tokenizer(
            question,
            step,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
        }


def create_dataloaders(
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 16,
    max_length: int = 512,
    corruption_rate: float = 0.3,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    train_dataset = HallucinationDataset(
        tokenizer=tokenizer,
        max_length=max_length,
        split="train",
        corruption_rate=corruption_rate,
    )

    val_dataset = HallucinationDataset(
        tokenizer=tokenizer,
        max_length=max_length,
        split="val",
        corruption_rate=corruption_rate,
    )

    test_dataset = HallucinationDataset(
        tokenizer=tokenizer,
        max_length=max_length,
        split="test",
        corruption_rate=corruption_rate,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {
        "train": train_dataloader,
        "val": val_dataloader,
        "test": test_dataloader,
    }
