import pytest
import torch
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedTokenizer

from inference.predictor import HallucinationPredictor


class MockClassifier(nn.Module):
    def __init__(self):
        super().__init__()

    def predict(self, input_ids, attention_mask, token_type_ids=None):
        batch_size = input_ids.size(0)
        labels = torch.zeros(batch_size, dtype=torch.long)
        labels[0] = 1

        confidence = torch.ones(batch_size) * 0.9

        probabilities = torch.zeros(batch_size, 2)
        probabilities[0, 1] = 0.9
        probabilities[0, 0] = 0.1

        probabilities[1:, 0] = 0.9
        probabilities[1:, 1] = 0.1

        return {
            "labels": labels,
            "confidence": confidence,
            "probabilities": probabilities,
        }


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")


@pytest.fixture
def model():
    return MockClassifier()


@pytest.fixture
def predictor(model, tokenizer):
    return HallucinationPredictor(
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        confidence_threshold=0.7,
    )


def test_predictor_initialization(model, tokenizer):
    predictor = HallucinationPredictor(
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        confidence_threshold=0.8,
    )

    assert predictor.model is model
    assert predictor.tokenizer is tokenizer
    assert predictor.device == torch.device("cpu")
    assert predictor.confidence_threshold == 0.8


def test_predict_single_step(predictor):
    question = "What is 2+2?"
    step = "2+2=4"

    result = predictor.predict_single_step(question, step)

    assert "label" in result
    assert "confidence" in result
    assert "probabilities" in result

    assert isinstance(result["label"], int)
    assert isinstance(result["confidence"], float)
    assert isinstance(result["probabilities"], list)
    assert len(result["probabilities"]) == 2
    assert 0 <= result["probabilities"][0] <= 1
    assert 0 <= result["probabilities"][1] <= 1


def test_predict_reasoning_chain(predictor):
    question = "What is 2+2?"
    reasoning = "First, I need to add 2 and 2.\n2+2=4\nTherefore, the answer is 4."

    results = predictor.predict_reasoning_chain(question, reasoning)

    assert isinstance(results, list)
    assert len(results) == 3

    for result in results:
        assert "step" in result
        assert "label" in result
        assert "is_hallucination" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert "is_reliable" in result

        assert isinstance(result["step"], str)
        assert isinstance(result["label"], int)
        assert isinstance(result["is_hallucination"], bool)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["probabilities"], list)
        assert isinstance(result["is_reliable"], bool)


def test_format_predictions(predictor):
    predictions = [
        {
            "step": "Step 1",
            "label": 0,
            "is_hallucination": False,
            "confidence": 0.9,
            "probabilities": [0.9, 0.1],
            "is_reliable": True,
        },
        {
            "step": "Step 2",
            "label": 1,
            "is_hallucination": True,
            "confidence": 0.8,
            "probabilities": [0.2, 0.8],
            "is_reliable": True,
        },
    ]

    formatted = predictor.format_predictions(predictions)

    assert isinstance(formatted, str)
    assert "Step 1" in formatted
    assert "Step 2" in formatted
    assert "GROUNDED" in formatted
    assert "HALLUCINATION" in formatted
