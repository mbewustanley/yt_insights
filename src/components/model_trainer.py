from pathlib import Path
import yaml
import pickle
import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config.config import ModelTrainerConfig
from src.entity.artifact import DataPreprocessingArtifact, ModelTrainerArtifact
from src.utils.logger import get_logger


class ModelTrainer:
    def __init__(
        self,
        config: ModelTrainerConfig,
        preprocessing_artifact: DataPreprocessingArtifact
    ):
        self.config = config
        self.preprocessing_artifact = preprocessing_artifact
        self.logger = get_logger("model_trainer")

    def load_data(self) -> pd.DataFrame:
        self.logger.debug("Loading processed training data")
        df = pd.read_csv(self.preprocessing_artifact.train_processed_path)
        df.fillna("", inplace=True)
        return df

    def apply_tfidf(self, df):
        self.logger.debug("Applying TF-IDF")

        X = df["clean_comment"].values
        y = df["category"].values

        vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range
        )

        X_tfidf = vectorizer.fit_transform(X)
        return X_tfidf, y, vectorizer

    def train_model(self, X, y):
        self.logger.debug("Training LightGBM model")

        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            metric="multi_logloss",
            is_unbalance=True,
            class_weight="balanced",
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            n_estimators=self.config.n_estimators,
            reg_alpha=0.1,
            reg_lambda=0.1
        )

        model.fit(X, y)
        return model

    def save_artifacts(self, model, vectorizer) -> ModelTrainerArtifact:
        self.logger.debug("Saving model and vectorizer")

        self.config.trained_model_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.config.trained_model_path, "wb") as f:
            pickle.dump(model, f)

        with open(self.config.vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)

        return ModelTrainerArtifact(
            trained_model_path=self.config.trained_model_path,
            vectorizer_path=self.config.vectorizer_path
        )

    def run(self) -> ModelTrainerArtifact:
        self.logger.info("Starting model training")

        df = self.load_data()
        X, y, vectorizer = self.apply_tfidf(df)
        model = self.train_model(X, y)

        artifact = self.save_artifacts(model, vectorizer)

        self.logger.info("Model training completed")
        return artifact


def main():
    logger = get_logger("model_trainer_main")

    try:
        PROJECT_ROOT = Path(__file__).resolve().parents[2]

        # Load global config
        with open(PROJECT_ROOT / "src/config/config.yaml", "r") as f:
            config_yaml = yaml.safe_load(f)

        trainer_cfg = config_yaml["model_trainer"]

        # Build ModelTrainerConfig
        model_trainer_config = ModelTrainerConfig(
            root_dir=PROJECT_ROOT,
            trained_model_path=PROJECT_ROOT / trainer_cfg["trained_model_path"],
            vectorizer_path=PROJECT_ROOT / trainer_cfg["vectorizer_path"],
            max_features=trainer_cfg["max_features"],
            ngram_range=tuple(trainer_cfg["ngram_range"]),
            learning_rate=trainer_cfg["learning_rate"],
            max_depth=trainer_cfg["max_depth"],
            n_estimators=trainer_cfg["n_estimators"],
        )

        # Build preprocessing artifact (output of previous stage)
        preprocessing_artifact = DataPreprocessingArtifact(
            train_processed_path=PROJECT_ROOT / "data/interim/train_processed.csv",
            test_processed_path=PROJECT_ROOT / "data/interim/test_processed.csv",
        )

        # Run trainer
        trainer = ModelTrainer(
            config=model_trainer_config,
            preprocessing_artifact=preprocessing_artifact
        )
        trainer.run()

        logger.info("Model trainer stage completed successfully")

    except Exception as e:
        logger.error("Model trainer stage failed", exc_info=True)
        raise


if __name__ == "__main__":
    main()
