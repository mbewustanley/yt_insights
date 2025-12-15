from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionArtifact:
    train_data_path: str
    test_data_path: str


@dataclass
class DataPreprocessingArtifact:
    train_processed_path: str
    test_processed_path: str


@dataclass
class ModelTrainerArtifact:
    trained_model_path: Path
    vectorizer_path: Path


@dataclass
class ModelEvaluatorArtifact:
    is_model_accepted: bool
    metrics_path: Path
    mlflow_run_id: str


@dataclass
class ModelPusherArtifact:
    saved_model_dir: Path
    model_path: Path
    vectorizer_path: Path