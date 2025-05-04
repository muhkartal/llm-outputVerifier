from typing import Dict, List, Optional, Tuple, Union

import re
import torch
from transformers import PreTrainedTokenizer

from data.preprocessing import vectorize_reasoning_chain
from models.classifier import HallucinationClassifier


class HallucinationPredictor:
    def __init__(
        self,
        model: HallucinationClassifier,
        tokenizer: PreTrainedTokenizer,
        device: Optional[torch.device] = None,
        confidence_threshold: float = 0.7,
    ):
        self.model = model
        self.tokenizer = tokenizer

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = self.model.to(self.device)
        self.model.eval()

        self.confidence_threshold = confidence_threshold

    def predict_single_step(
        self,
        question: str,
        reasoning_step: str,
    ) -> Dict[str, Union[int, float, List[float]]]:
        encoding = self.tokenizer(
            question,
            reasoning_step,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )

        inputs = {
            "input_ids": encoding["input_ids"].to(self.device),
            "attention_mask": encoding["attention_mask"].to(self.device),
        }

        with torch.no_grad():
            outputs = self.model.predict(**inputs)

        return {
            "label": outputs["labels"][0].item(),
            "confidence": outputs["confidence"][0].item(),
            "probabilities": outputs["probabilities"][0].tolist(),
        }

    def predict_reasoning_chain(
        self,
        question: str,
        reasoning: str,
    ) -> List[Dict[str, Union[str, int, float, List[float]]]]:
        steps = re.split(r"\n", reasoning)
        steps = [step for step in steps if step.strip()]

        encodings = vectorize_reasoning_chain(
            question=question,
            reasoning_steps=steps,
            tokenizer=self.tokenizer,
        )

        inputs = {
            "input_ids": encodings["input_ids"].to(self.device),
            "attention_mask": encodings["attention_mask"].to(self.device),
        }

        with torch.no_grad():
            outputs = self.model.predict(**inputs)

        labels = outputs["labels"].cpu().numpy()
        confidences = outputs["confidence"].cpu().numpy()
        probabilities = outputs["probabilities"].cpu().numpy()

        results = []

        for i, step in enumerate(steps):
            results.append({
                "step": step,
                "label": int(labels[i]),
                "is_hallucination": bool(labels[i] == 1),
                "confidence": float(confidences[i]),
                "probabilities": probabilities[i].tolist(),
                "is_reliable": confidences[i] >= self.confidence_threshold,
            })

        return results

    def format_predictions(
        self,
        predictions: List[Dict[str, Union[str, int, float, List[float]]]],
    ) -> str:
        result = ""

        for i, pred in enumerate(predictions):
            step = pred["step"]
            is_hallucination = pred["is_hallucination"]
            confidence = pred["confidence"]

            status = "⚠️ HALLUCINATION" if is_hallucination else "✅ GROUNDED"
            reliability = "(HIGH CONFIDENCE)" if pred["is_reliable"] else "(LOW CONFIDENCE)"

            confidence_str = f"{confidence:.2f}"

            result += f"Step {i+1}: {status} {confidence_str} {reliability}\n"
            result += f"{step}\n\n"

        return result
