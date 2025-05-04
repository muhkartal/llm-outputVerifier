from typing import Optional, Tuple, Union

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


def load_encoder_model(
    model_name: str = "roberta-base",
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    encoder = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return encoder, tokenizer


def enable_gradient_checkpointing(
    model: PreTrainedModel,
) -> PreTrainedModel:
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    return model


def apply_mixed_precision(
    model: PreTrainedModel,
    dtype: Optional[torch.dtype] = None,
) -> PreTrainedModel:
    if dtype is None:
        if torch.cuda.is_available():
            dtype = torch.float16
        else:
            dtype = torch.bfloat16

    model = model.to(dtype=dtype)

    return model
