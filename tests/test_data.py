import pytest
import torch
from transformers import AutoTokenizer

from data.augmentation import (
    parse_gsm8k_reasoning,
    fact_distortion,
    logical_error,
    number_substitution,
    create_synthetic_hallucinations,
)

from data.dataset import (
    HallucinationDataset,
    create_dataloaders,
)

from data.preprocessing import (
    preprocess_reasoning_steps,
    vectorize_reasoning_chain,
    extract_step_inputs,
)


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")


@pytest.fixture
def gsm8k_example():
    return {
        "question": "John has 5 apples. He buys 2 more. How many apples does he have now?",
        "answer": "John starts with 5 apples.\nThen he buys 2 more apples.\nSo in total, he has 5 + 2 = 7 apples.\nThe answer is 7."
    }


def test_parse_gsm8k_reasoning(gsm8k_example):
    steps, labels = parse_gsm8k_reasoning(gsm8k_example)

    assert len(steps) == 4
    assert len(labels) == 4
    assert all(label == 0 for label in labels)
    assert steps[0] == "John starts with 5 apples."
    assert steps[-1] == "The answer is 7."


def test_fact_distortion():
    step = "John increases his apple count by 2."
    distorted = fact_distortion(step)

    assert distorted != step
    assert "decrease" in distorted


def test_logical_error():
    step = "So in total, he has 5 + 2 = 7 apples."
    corrupted = logical_error(step)

    assert corrupted != step
    assert "5 + 2 =" in corrupted
    assert "= 7" not in corrupted


def test_number_substitution():
    step = "John has 5 apples initially."
    corrupted = number_substitution(step)

    assert corrupted != step
    assert "5" not in corrupted


def test_create_synthetic_hallucinations(gsm8k_example):
    mock_dataset = [gsm8k_example]

    def mock_map(func):
        return [func(example) for example in mock_dataset]

    class MockDataset:
        def __init__(self, data):
            self.data = data

        def map(self, func):
            return [func(example) for example in self.data]

    mock_dataset = MockDataset([gsm8k_example])

    result = create_synthetic_hallucinations(
        mock_dataset,
        corruption_rate=1.0
    )

    assert isinstance(result, list)
    assert len(result) == 1
    assert "reasoning_steps" in result[0]
    assert "hallucination_labels" in result[0]
    assert "question" in result[0]
    assert "original_answer" in result[0]

    hallucination_count = sum(result[0]["hallucination_labels"])
    assert hallucination_count > 0


def test_preprocess_reasoning_steps(tokenizer, gsm8k_example):
    reasoning = gsm8k_example["answer"]

    encoded_steps = preprocess_reasoning_steps(reasoning, tokenizer)

    assert isinstance(encoded_steps, list)
    assert len(encoded_steps) > 0

    for step in encoded_steps:
        assert "input_ids" in step
        assert "attention_mask" in step
        assert isinstance(step["input_ids"], torch.Tensor)
        assert isinstance(step["attention_mask"], torch.Tensor)


def test_vectorize_reasoning_chain(tokenizer, gsm8k_example):
    question = gsm8k_example["question"]
    reasoning = gsm8k_example["answer"]
    steps = reasoning.split("\n")

    encodings = vectorize_reasoning_chain(question, steps, tokenizer)

    assert "input_ids" in encodings
    assert "attention_mask" in encodings
    assert isinstance(encodings["input_ids"], torch.Tensor)
    assert isinstance(encodings["attention_mask"], torch.Tensor)
    assert encodings["input_ids"].shape[0] == len(steps)


def test_extract_step_inputs():
    batch = {
        "input_ids": torch.ones((4, 10), dtype=torch.long),
        "attention_mask": torch.ones((4, 10), dtype=torch.long),
        "label": torch.tensor([0, 1, 0, 1], dtype=torch.long),
    }

    inputs, labels = extract_step_inputs(batch)

    assert "input_ids" in inputs
    assert "attention_mask" in inputs
    assert torch.equal(inputs["input_ids"], batch["input_ids"])
    assert torch.equal(inputs["attention_mask"], batch["attention_mask"])
    assert torch.equal(labels, batch["label"])
