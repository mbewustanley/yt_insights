from dataclasses import dataclass
from typing import Tuple
from pathlib import Path


@dataclass
class DataIngestionConfig:
    data_url: str
    test_size: float
    data_path: str


@dataclass
class DataPreprocessingConfig:
    raw_data_dir: str       # folder containing train.csv & test.csv
    interim_data_dir: str   # folder to save processed data


@dataclass
class ModelTrainerConfig:
    root_dir: Path

    trained_model_path: Path
    vectorizer_path: Path

    max_features: int
    ngram_range: Tuple[int, int]

    learning_rate: float
    max_depth: int
    n_estimators: int

    
@dataclass
class ModelEvaluatorConfig:
    root_dir: Path
    metric_output_path: Path
    acceptance_threshold: float
    mlflow_tracking_uri: str
    mlflow_experiment_name: str




