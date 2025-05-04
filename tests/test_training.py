import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from training.optimizer import create_optimizer, create_scheduler
from training.trainer import HallucinationTrainer


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        hidden_states = torch.randn(batch_size, 10)
        logits = self.linear(hidden_states)
        return {"logits": logits, "hidden_states": hidden_states}


@pytest.fixture
def model():
    return SimpleModel()


@pytest.fixture
def dataloaders():
    batch_size = 4

    train_inputs = torch.randint(0, 100, (20, 10))
    train_masks = torch.ones_like(train_inputs)
    train_labels = torch.randint(0, 2, (20,))

    train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    val_inputs = torch.randint(0, 100, (10, 10))
    val_masks = torch.ones_like(val_inputs)
    val_labels = torch.randint(0, 2, (10,))

    val_dataset = TensorDataset(val_inputs, val_masks, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return {"train": train_dataloader, "val": val_dataloader}


def test_create_optimizer(model):
    optimizer = create_optimizer(model, optimizer_type="adamw", learning_rate=1e-4)
    assert isinstance(optimizer, torch.optim.AdamW)

    optimizer = create_optimizer(model, optimizer_type="adam", learning_rate=1e-4)
    assert isinstance(optimizer, torch.optim.Adam)

    optimizer = create_optimizer(model, optimizer_type="sgd", learning_rate=1e-4)
    assert isinstance(optimizer, torch.optim.SGD)


def test_create_scheduler(model):
    optimizer = create_optimizer(model)

    scheduler = create_scheduler(optimizer, scheduler_type="linear", num_training_steps=1000)
    assert scheduler is not None

    scheduler = create_scheduler(optimizer, scheduler_type="cosine", num_training_steps=1000)
    assert scheduler is not None

    scheduler = create_scheduler(optimizer, scheduler_type="constant", num_training_steps=1000)
    assert scheduler is not None


def test_trainer_initialization(model, dataloaders):
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = HallucinationTrainer(
            model=model,
            train_dataloader=dataloaders["train"],
            val_dataloader=dataloaders["val"],
            num_epochs=1,
            save_dir=temp_dir,
            use_wandb=False,
        )

        assert trainer.model is model
        assert trainer.train_dataloader is dataloaders["train"]
        assert trainer.val_dataloader is dataloaders["val"]
        assert trainer.num_epochs == 1
        assert trainer.save_dir == temp_dir
        assert trainer.device is not None


def test_trainer_training(model, dataloaders):
    def extract_step_inputs(batch):
        input_ids = batch[0]
        attention_mask = batch[1]
        labels = batch[2]

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        return inputs, labels

    with tempfile.TemporaryDirectory() as temp_dir:
        import training.trainer
        original_extract_fn = training.trainer.extract_step_inputs
        training.trainer.extract_step_inputs = extract_step_inputs

        try:
            trainer = HallucinationTrainer(
                model=model,
                train_dataloader=dataloaders["train"],
                val_dataloader=dataloaders["val"],
                num_epochs=1,
                save_dir=temp_dir,
                use_wandb=False,
            )

            results = trainer.train()

            assert "train_losses" in results
            assert "val_metrics" in results
            assert "best_val_accuracy" in results
            assert "best_epoch" in results

            assert os.path.exists(os.path.join(temp_dir, "best_model.pt"))
            assert os.path.exists(os.path.join(temp_dir, "model_epoch_1.pt"))

        finally:
            training.trainer.extract_step_inputs = original_extract_fn
