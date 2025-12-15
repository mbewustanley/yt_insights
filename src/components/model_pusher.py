import shutil
from pathlib import Path
import yaml

from src.config.config import ModelPusherConfig
from src.entity.artifact import ModelEvaluatorArtifact, ModelPusherArtifact
from src.utils.logger import get_logger


class ModelPusher:
    def __init__(
        self,
        config: ModelPusherConfig,
        evaluator_artifact: ModelEvaluatorArtifact
    ):
        self.config = config
        self.evaluator_artifact = evaluator_artifact
        self.logger = get_logger("model_pusher")

    def push(self) -> ModelPusherArtifact:
        self.logger.info("Starting model pusher")

        if not self.evaluator_artifact.is_model_accepted:
            self.logger.info("Model not accepted — skipping push")
            return None

        self.config.saved_model_dir.mkdir(parents=True, exist_ok=True)

        model_dest = self.config.saved_model_dir / "model.pkl"
        vectorizer_dest = self.config.saved_model_dir / "vectorizer.pkl"

        shutil.copy(
            self.evaluator_artifact.trained_model_path,
            model_dest
        )

        shutil.copy(
            self.evaluator_artifact.vectorizer_path,
            vectorizer_dest
        )

        self.logger.info("Model pushed successfully")

        return ModelPusherArtifact(
            saved_model_dir=self.config.saved_model_dir,
            model_path=model_dest,
            vectorizer_path=vectorizer_dest
        )



def main():
    logger = get_logger("model_pusher_main")

    try:
        PROJECT_ROOT = Path(__file__).resolve().parents[2]

        with open(PROJECT_ROOT / "src/config/config.yaml", "r") as f:
            config_yaml = yaml.safe_load(f)

        pusher_cfg = config_yaml["model_pusher"]

        config = ModelPusherConfig(
            saved_model_dir=PROJECT_ROOT / pusher_cfg["saved_model_dir"]
        )

        evaluator_artifact = ModelEvaluatorArtifact(
            is_model_accepted=True,  # produced by evaluator stage
            trained_model_path=PROJECT_ROOT / "models/trained_model.pkl",
            vectorizer_path=PROJECT_ROOT / "models/tfidf_vectorizer.pkl",
            metric_path=PROJECT_ROOT / "reports/metrics.json"
        )

        pusher = ModelPusher(config, evaluator_artifact)
        pusher.push()

        logger.info("Model pusher stage completed")

    except Exception as e:
        logger.error("Model pusher failed", exc_info=True)
        raise


if __name__ == "__main__":
    main()
