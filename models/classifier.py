from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModel


class HallucinationClassifier(nn.Module):
    def __init__(
        self,
        encoder: PreTrainedModel,
        dropout_rate: float = 0.1,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout_rate)

        hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 2),
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        return {
            "logits": logits,
            "hidden_states": outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
        }

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, token_type_ids)

            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=-1)
            predicted_labels = torch.argmax(probs, dim=-1)
            confidence_scores = torch.max(probs, dim=-1).values

            return {
                "labels": predicted_labels,
                "confidence": confidence_scores,
                "probabilities": probs,
            }


def create_hallucination_classifier(
    model_name: str = "roberta-base",
    dropout_rate: float = 0.1,
    freeze_encoder: bool = False,
) -> HallucinationClassifier:
    encoder = AutoModel.from_pretrained(model_name)

    return HallucinationClassifier(
        encoder=encoder,
        dropout_rate=dropout_rate,
        freeze_encoder=freeze_encoder,
    )
