from typing import Dict, List, Optional, Union

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup


def create_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    eps: float = 1e-8,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    no_decay = ["bias", "LayerNorm.weight"]

    # Group parameters for weight decay application
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # Create optimizer based on type
    if optimizer_type.lower() == "adamw":
        return optim.AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            eps=eps,
        )
    elif optimizer_type.lower() == "adam":
        return optim.Adam(
            optimizer_grouped_parameters,
            lr=learning_rate,
            eps=eps,
        )
    elif optimizer_type.lower() == "sgd":
        return optim.SGD(
            optimizer_grouped_parameters,
            lr=learning_rate,
            momentum=momentum,
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "linear",
    num_training_steps: int = 10000,
    num_warmup_steps: Optional[int] = None,
    warmup_ratio: float = 0.1,
    last_epoch: int = -1,
) -> Union[LambdaLR, CosineAnnealingLR]:
    if num_warmup_steps is None:
        num_warmup_steps = int(num_training_steps * warmup_ratio)

    if scheduler_type.lower() == "linear":
        return get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )
    elif scheduler_type.lower() == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )
    elif scheduler_type.lower() == "constant":
        return LambdaLR(optimizer, lambda _: 1.0, last_epoch=last_epoch)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def get_learning_rates(optimizer: torch.optim.Optimizer) -> List[float]:
    return [group["lr"] for group in optimizer.param_groups]
