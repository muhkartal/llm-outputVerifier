from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator


class PredictionRequest(BaseModel):
    question: str = Field(
        ...,
        description="The mathematical question",
        example="John has 5 apples. He buys 2 more. How many apples does he have now?"
    )
    reasoning: str = Field(
        ...,
        description="Chain-of-thought reasoning for the question",
        example="John starts with 5 apples.\nThen he buys 2 more apples.\nSo in total, he has 5 + 2 = 7 apples."
    )

    @validator("question")
    def validate_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty")
        if len(v) > 1000:
            raise ValueError("Question must be less than 1000 characters")
        return v

    @validator("reasoning")
    def validate_reasoning(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Reasoning cannot be empty")
        if len(v) > 10000:
            raise ValueError("Reasoning must be less than 10000 characters")
        return v

    class Config:
        schema_extra = {
            "example": {
                "question": "John has 5 apples. He buys 2 more. How many apples does he have now?",
                "reasoning": "John starts with 5 apples.\nThen he buys 2 more apples.\nSo in total, he has 5 + 2 = 7 apples."
            }
        }


class StepPrediction(BaseModel):
    step: str = Field(..., description="The reasoning step text")
    is_hallucination: bool = Field(..., description="Whether the step is a hallucination")
    confidence: float = Field(..., description="Confidence score (0-1)")
    probabilities: List[float] = Field(..., description="Class probabilities [grounded, hallucination]")
    is_reliable: bool = Field(..., description="Whether the prediction meets confidence threshold")

    @validator("confidence")
    def validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        return v

    @validator("probabilities")
    def validate_probabilities(cls, v: List[float]) -> List[float]:
        if len(v) != 2:
            raise ValueError("Probabilities must have exactly 2 values (grounded, hallucination)")
        if not all(0.0 <= p <= 1.0 for p in v):
            raise ValueError("Probability values must be between 0.0 and 1.0")
        if abs(sum(v) - 1.0) > 1e-5:
            raise ValueError("Probabilities must sum to 1.0")
        return v

    class Config:
        schema_extra = {
            "example": {
                "step": "So in total, he has 5 + 2 = 7 apples.",
                "is_hallucination": False,
                "confidence": 0.95,
                "probabilities": [0.95, 0.05],
                "is_reliable": True
            }
        }


class PredictionResponse(BaseModel):
    predictions: List[StepPrediction] = Field(
        ..., description="Predictions for each reasoning step"
    )
    question: str = Field(..., description="The original question")
    num_steps: int = Field(..., description="Number of reasoning steps")
    num_hallucinations: int = Field(..., description="Number of hallucinated steps")
    hallucination_rate: float = Field(
        0.0, description="Percentage of steps classified as hallucinations"
    )
    average_confidence: float = Field(
        0.0, description="Average confidence score across all steps"
    )
    processing_time_ms: Optional[float] = Field(
        None, description="Time taken to process the request in milliseconds"
    )

    @validator("hallucination_rate", always=True)
    def calculate_hallucination_rate(cls, v: float, values: Dict) -> float:
        if "num_steps" in values and "num_hallucinations" in values:
            if values["num_steps"] > 0:
                return values["num_hallucinations"] / values["num_steps"] * 100.0
        return 0.0

    @validator("average_confidence", always=True)
    def calculate_average_confidence(cls, v: float, values: Dict) -> float:
        if "predictions" in values and values["predictions"]:
            confidences = [pred.confidence for pred in values["predictions"]]
            return sum(confidences) / len(confidences)
        return 0.0

    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "step": "John starts with 5 apples.",
                        "is_hallucination": False,
                        "confidence": 0.97,
                        "probabilities": [0.97, 0.03],
                        "is_reliable": True
                    },
                    {
                        "step": "Then he buys 2 more apples.",
                        "is_hallucination": False,
                        "confidence": 0.95,
                        "probabilities": [0.95, 0.05],
                        "is_reliable": True
                    },
                    {
                        "step": "So in total, he has 5 + 2 = 8 apples.",
                        "is_hallucination": True,
                        "confidence": 0.82,
                        "probabilities": [0.18, 0.82],
                        "is_reliable": True
                    }
                ],
                "question": "John has 5 apples. He buys 2 more. How many apples does he have now?",
                "num_steps": 3,
                "num_hallucinations": 1,
                "hallucination_rate": 33.33,
                "average_confidence": 0.91,
                "processing_time_ms": 156.23
            }
        }


class HealthResponse(BaseModel):
    status: str = Field(..., description="API health status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    version: str = Field("0.1.0", description="API version")
    device: str = Field("cpu", description="Device the model is running on")
    uptime_seconds: Optional[float] = Field(None, description="API uptime in seconds")
    memory_usage_mb: Optional[float] = Field(None, description="Current memory usage in MB")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "version": "0.1.0",
                "device": "cuda",
                "uptime_seconds": 3600.5,
                "memory_usage_mb": 1256.34
            }
        }
