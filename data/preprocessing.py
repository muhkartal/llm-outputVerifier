from typing import Dict, List, Optional, Tuple, Union

import re
import torch
from transformers import PreTrainedTokenizer


def preprocess_reasoning_steps(
    reasoning: str,
    tokenizer: PreTrainedTokenizer
) -> List[Dict[str, torch.Tensor]]:
    steps = re.split(r"\n", reasoning)
    steps = [step for step in steps if step.strip()]

    encoded_steps = []

    for step in steps:
        encoding = tokenizer(
            step,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )

        encoded_steps.append({
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        })

    return encoded_steps


def vectorize_reasoning_chain(
    question: str,
    reasoning_steps: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
) -> Dict[str, torch.Tensor]:
    encodings = []

    for step in reasoning_steps:
        encoding = tokenizer(
            question,
            step,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

        encodings.append({
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        })

    input_ids = torch.stack([encoding["input_ids"] for encoding in encodings])
    attention_mask = torch.stack([encoding["attention_mask"] for encoding in encodings])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def extract_step_inputs(
    batch: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    inputs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
    }

    labels = batch["label"]

    return inputs, labels
