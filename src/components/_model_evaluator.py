import os
import json
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml

from sklearn.metrics import classification_report, confusion_matrix

from src.utils.logger import get_logger
from src.config.config import ModelEvaluatorConfig
from src.entity.artifact import ModelTrainerArtifact, DataPreprocessingArtifact, ModelEvaluatorArtifact



### evaluate → log metrics/artifacts → log model → if accepted → register

class ModelEvaluator:
    def __init__(
        self,
        config: ModelEvaluatorConfig,
        trainer_artifact: ModelTrainerArtifact,
        preprocessing_artifact: DataPreprocessingArtifact
    ):
        self.config = config
        self.trainer_artifact = trainer_artifact
        self.preprocessing_artifact = preprocessing_artifact
        self.logger = get_logger("model_evaluator")

    def load_model_and_vectorizer(self):
        with open(self.trainer_artifact.trained_model_path, "rb") as f:
            model = pickle.load(f)

        with open(self.trainer_artifact.vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)

        return model, vectorizer
    

    def load_test_data(self):
        df = pd.read_csv(self.preprocessing_artifact.test_processed_path)
        df.fillna("", inplace=True)
        return df


    def evaluate(self, model, vectorizer, df):
        X_test = vectorizer.transform(df["clean_comment"].values)
        y_test = df["category"].values

        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        return report, cm

    def log_confusion_matrix(self, cm):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        self.config.metric_output_path.parent.mkdir(parents=True, exist_ok=True)
        cm_path = self.config.metric_output_path.parent/"confusion_matrix.png"
        
        print("Saving confusion matrix to:", cm_path.resolve())

        plt.savefig(cm_path)
        mlflow.log_artifact(str(cm_path))
        plt.close()


    def run(self) -> ModelEvaluatorArtifact:
        self.logger.info("Starting model evaluation")

        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment_name)

        with mlflow.start_run() as run:
            model, vectorizer = self.load_model_and_vectorizer()
            test_df = self.load_test_data()

            report, cm = self.evaluate(model, vectorizer, test_df)

            # Log metrics
            weighted_f1 = report["weighted avg"]["f1-score"]
            mlflow.log_metric("weighted_f1", weighted_f1)

            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"{label}_precision": metrics["precision"],
                        f"{label}_recall": metrics["recall"],
                        f"{label}_f1": metrics["f1-score"]
                    })

            self.log_confusion_matrix(cm)

            # Save metrics locally (DVC-tracked)
            with open(self.config.metric_output_path, "w") as f:
                json.dump(report, f, indent=4)

            is_accepted = weighted_f1 >= self.config.acceptance_threshold

            self.logger.info(
                "Model accepted" if is_accepted else "Model rejected"
            )

            return ModelEvaluatorArtifact(
                is_model_accepted=is_accepted,
                metrics_path=self.config.metric_output_path,
                mlflow_run_id=run.info.run_id
            )



def main():
    logger = get_logger("model_evaluator_main")

    try:
        PROJECT_ROOT = Path(__file__).resolve().parents[2]

        with open(PROJECT_ROOT / "src/config/config.yaml", "r") as f:
            config_yaml = yaml.safe_load(f)

        eval_cfg = config_yaml["model_evaluator"]

        evaluator_config = ModelEvaluatorConfig(
            root_dir=PROJECT_ROOT,
            metric_output_path=PROJECT_ROOT / eval_cfg["metric_output_path"],
            acceptance_threshold=eval_cfg["acceptance_threshold"],
            mlflow_tracking_uri=eval_cfg["mlflow_tracking_uri"],
            mlflow_experiment_name=eval_cfg["mlflow_experiment_name"]
        )

        trainer_artifact = ModelTrainerArtifact(
            trained_model_path=PROJECT_ROOT / "models/trained_model.pkl",
            vectorizer_path=PROJECT_ROOT / "models/tfidf_vectorizer.pkl"
        )

        preprocessing_artifact = DataPreprocessingArtifact(
            train_processed_path=PROJECT_ROOT / "data/interim/train_processed.csv",
            test_processed_path=PROJECT_ROOT / "data/interim/test_processed.csv"
        )

        evaluator = ModelEvaluator(
            config=evaluator_config,
            trainer_artifact=trainer_artifact,
            preprocessing_artifact=preprocessing_artifact
        )

        evaluator.run()
        logger.info("Model evaluation completed successfully")

    except Exception:
        logger.error("Model evaluation failed", exc_info=True)
        raise


if __name__ == "__main__":
    main()
