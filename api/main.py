import logging
import time
from typing import Dict, List, Optional

import torch
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from hallucination_hunter.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    StepPrediction,
)
from hallucination_hunter.inference.predictor import HallucinationPredictor
from hallucination_hunter.models.classifier import create_hallucination_classifier
from hallucination_hunter.models.encoder import load_encoder_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("hallucination_hunter")

app = FastAPI(
    title="Hallucination Hunter API",
    description="API for detecting hallucinations in chain-of-thought reasoning",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production to specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(time.time())
    logger.info(f"Request {request_id} started: {request.method} {request.url.path}")
    start_time = time.time()

    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Request {request_id} completed: {response.status_code} ({process_time:.4f}s)")
        return response
    except Exception as e:
        logger.error(f"Request {request_id} failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )

MODEL_PATH = "models/hallucination_classifier"
MODEL_NAME = "roberta-base"
CONFIDENCE_THRESHOLD = 0.7


def get_predictor() -> HallucinationPredictor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading model from {MODEL_PATH} on device {device}")

    try:
        encoder, tokenizer = load_encoder_model(MODEL_NAME)
        logger.info(f"Loaded encoder: {MODEL_NAME}")

        model = create_hallucination_classifier(MODEL_NAME)

        # Check if model path exists
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model path does not exist: {MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

        # Load model with error handling
        try:
            state_dict = torch.load(
                MODEL_PATH,
                map_location=device,
            )
            model.load_state_dict(state_dict)
            logger.info("Successfully loaded model weights")
        except Exception as e:
            logger.error(f"Failed to load model weights: {str(e)}")
            raise ValueError(f"Failed to load model weights: {str(e)}")

        # Create predictor
        predictor = HallucinationPredictor(
            model=model,
            tokenizer=tokenizer,
            device=device,
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )

        # Verify predictor works by making a test prediction
        try:
            test_question = "What is 2+2?"
            test_step = "2+2=4"
            predictor.predict_single_step(test_question, test_step)
            logger.info("Predictor verification successful")
        except Exception as e:
            logger.error(f"Predictor verification failed: {str(e)}")
            raise ValueError(f"Predictor verification failed: {str(e)}")

        return predictor

    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}",
        )


@app.get("/")
def root():
    return {
        "message": "Welcome to the Hallucination Hunter API",
        "version": "0.1.0",
        "endpoints": [
            "/predict",
            "/health",
            "/docs",
        ],
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(
    request: PredictionRequest,
    predictor: HallucinationPredictor = Depends(get_predictor),
):
    start_time = time.time()
    request_id = str(int(start_time))

    logger.info(f"Processing prediction request {request_id}")

    # Validate input
    if not request.question or not request.question.strip():
        logger.warning(f"Request {request_id}: Empty question provided")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty",
        )

    if not request.reasoning or not request.reasoning.strip():
        logger.warning(f"Request {request_id}: Empty reasoning provided")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reasoning cannot be empty",
        )

    # Process reasoning chains with proper error handling
    try:
        # Log request details (truncated for privacy/size)
        logger.info(
            f"Request {request_id}: Question: {request.question[:50]}..., "
            f"Reasoning: {len(request.reasoning.split())} words"
        )

        # Make prediction
        predictions = predictor.predict_reasoning_chain(
            question=request.question,
            reasoning=request.reasoning,
        )

        step_predictions = []

        for pred in predictions:
            step_predictions.append(
                StepPrediction(
                    step=pred["step"],
                    is_hallucination=pred["is_hallucination"],
                    confidence=pred["confidence"],
                    probabilities=pred["probabilities"],
                    is_reliable=pred["is_reliable"],
                )
            )

        # Create response
        response = PredictionResponse(
            predictions=step_predictions,
            question=request.question,
            num_steps=len(step_predictions),
            num_hallucinations=sum(1 for p in step_predictions if p.is_hallucination),
        )

        # Log results summary
        process_time = time.time() - start_time
        logger.info(
            f"Request {request_id} completed in {process_time:.2f}s: "
            f"{response.num_steps} steps analyzed, "
            f"{response.num_hallucinations} hallucinations detected"
        )

        return response

    except ValueError as e:
        logger.error(f"Request {request_id} validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}",
        )
    except torch.cuda.OutOfMemoryError:
        logger.error(f"Request {request_id} failed: GPU OOM error")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service temporarily unavailable due to resource constraints. Please try again later or with shorter input.",
        )
    except Exception as e:
        logger.error(f"Request {request_id} failed with unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )
