import torch
import pytest
from transformers import AutoModel, AutoTokenizer

from models.classifier import (
    HallucinationClassifier,
    create_hallucination_classifier,
)
from models.encoder import (
    load_encoder_model,
    enable_gradient_checkpointing,
    apply_mixed_precision,
)


@pytest.fixture
def encoder_tokenizer():
    encoder = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    return encoder, tokenizer


@pytest.fixture
def classifier(encoder_tokenizer):
    encoder, _ = encoder_tokenizer
    return HallucinationClassifier(encoder=encoder, dropout_rate=0.1)


def test_hallucination_classifier_init(encoder_tokenizer):
    encoder, _ = encoder_tokenizer
    classifier = HallucinationClassifier(encoder=encoder, dropout_rate=0.1)

    assert isinstance(classifier, HallucinationClassifier)
    assert hasattr(classifier, "encoder")
    assert hasattr(classifier, "classifier")
    assert isinstance(classifier.classifier, torch.nn.Sequential)


def test_hallucination_classifier_forward(classifier, encoder_tokenizer):
    _, tokenizer = encoder_tokenizer

    inputs = tokenizer(
        "This is a test question",
        "This is a test reasoning step",
        return_tensors="pt",
        padding="max_length",
        max_length=128,
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    outputs = classifier.forward(input_ids=input_ids, attention_mask=attention_mask)

    assert "logits" in outputs
    assert outputs["logits"].shape == (1, 2)


def test_hallucination_classifier_predict(classifier, encoder_tokenizer):
    _, tokenizer = encoder_tokenizer

    inputs = tokenizer(
        "This is a test question",
        "This is a test reasoning step",
        return_tensors="pt",
        padding="max_length",
        max_length=128,
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    predictions = classifier.predict(input_ids=input_ids, attention_mask=attention_mask)

    assert "labels" in predictions
    assert "confidence" in predictions
    assert "probabilities" in predictions

    assert predictions["labels"].shape == (1,)
    assert predictions["confidence"].shape == (1,)
    assert predictions["probabilities"].shape == (1, 2)


def test_create_hallucination_classifier():
    model = create_hallucination_classifier(
        model_name="prajjwal1/bert-tiny",
        dropout_rate=0.1
    )

    assert isinstance(model, HallucinationClassifier)


def test_load_encoder_model():
    encoder, tokenizer = load_encoder_model("prajjwal1/bert-tiny")

    assert isinstance(encoder, torch.nn.Module)
    assert hasattr(tokenizer, "encode")
    assert hasattr(tokenizer, "decode")


def test_enable_gradient_checkpointing(encoder_tokenizer):
    encoder, _ = encoder_tokenizer

    updated_encoder = enable_gradient_checkpointing(encoder)

    assert updated_encoder is encoder


def test_apply_mixed_precision(encoder_tokenizer):
    encoder, _ = encoder_tokenizer

    fp16_encoder = apply_mixed_precision(encoder, dtype=torch.float16)

    assert fp16_encoder is encoder
