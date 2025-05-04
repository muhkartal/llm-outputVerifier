from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from pydantic import BaseModel, Field, validator


class DataConfig(BaseModel):
    gsm8k_dataset: str = "gsm8k"
    gsm8k_subset: str = "main"
    corruption_rate: float = 0.3
    corruption_types: List[str] = ["fact_distortion", "logical_error", "number_substitution"]
    max_length: int = 512
    train_batch_size: int = 16
    eval_batch_size: int = 32
    num_workers: int = 4
    cache_dir: Optional[str] = None
    seed: int = 42

    @validator("corruption_rate")
    def validate_corruption_rate(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("corruption_rate must be between 0.0 and 1.0")
        return v

    @validator("corruption_types")
    def validate_corruption_types(cls, v: List[str]) -> List[str]:
        valid_types = ["fact_distortion", "logical_error", "number_substitution"]
        for corruption_type in v:
            if corruption_type not in valid_types:
                raise ValueError(f"Invalid corruption type: {corruption_type}. Must be one of {valid_types}")
        return v


class ModelConfig(BaseModel):
    model_name: str = "roberta-base"
    dropout_rate: float = 0.1
    freeze_encoder: bool = False
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    precision: str = "fp16"  # Options: fp32, fp16, bf16
    hidden_size: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    attention_heads: Optional[int] = None
    classifier_hidden_size: Optional[int] = None

    @validator("dropout_rate")
    def validate_dropout_rate(cls, v: float) -> float:
        if not 0.0 <= v <= 0.9:
            raise ValueError("dropout_rate must be between 0.0 and 0.9")
        return v

    @validator("precision")
    def validate_precision(cls, v: str) -> str:
        valid_precision = ["fp32", "fp16", "bf16"]
        if v not in valid_precision:
            raise ValueError(f"precision must be one of {valid_precision}")
        return v


class TrainingConfig(BaseModel):
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    log_interval: int = 100
    save_dir: str = "./checkpoints"
    use_wandb: bool = True
    project_name: str = "hallucination-hunter"
    seed: int = 42
    fp16: bool = True
    evaluation_steps: int = 500
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01
    scheduler: str = "linear"  # Options: linear, cosine, constant
    optimizer: str = "adamw"   # Options: adamw, adam, sgd
    save_best_model: bool = True
    save_total_limit: int = 3

    @validator("learning_rate")
    def validate_learning_rate(cls, v: float) -> float:
        if not 1e-7 <= v <= 1e-1:
            raise ValueError("learning_rate must be between 1e-7 and 1e-1")
        return v

    @validator("scheduler")
    def validate_scheduler(cls, v: str) -> str:
        valid_schedulers = ["linear", "cosine", "constant"]
        if v not in valid_schedulers:
            raise ValueError(f"scheduler must be one of {valid_schedulers}")
        return v

    @validator("optimizer")
    def validate_optimizer(cls, v: str) -> str:
        valid_optimizers = ["adamw", "adam", "sgd"]
        if v not in valid_optimizers:
            raise ValueError(f"optimizer must be one of {valid_optimizers}")
        return v


class InferenceConfig(BaseModel):
    model_path: str = "models/hallucination_classifier"
    confidence_threshold: float = 0.7
    batch_size: int = 32


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 4
    log_level: str = "info"


class UIConfig(BaseModel):
    api_url: str = "http://localhost:8000"
    port: int = 8501


class Config(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    ui: UIConfig = Field(default_factory=UIConfig)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_default_config() -> Config:
    return Config()


def save_config(config: Config, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write(config.json(indent=2))


def load_config(path: Union[str, Path]) -> Config:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config_dict = f.read()

    return Config.parse_raw(config_dict)
