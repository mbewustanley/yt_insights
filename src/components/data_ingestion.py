import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging
from pathlib import Path

from src.utils.logger import get_logger
from src.config.config import DataIngestionConfig
from src.entity.artifact import DataIngestionArtifact


logger = get_logger("data_ingestion")


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def _load_params(self) -> dict:
        try:
            with open(self.params_path, "r") as file:
                params = yaml.safe_load(file)
            logger.debug("Parameters loaded successfully")
            return params
        except Exception as e:
            logger.error("Failed to load params", exc_info=True)
            raise

    def load_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.config.data_url)
            logger.debug("Data loaded successfully")
            return df
        except Exception:
            logger.error("Failed to load data", exc_info=True)
            raise

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.dropna()
            df = df.drop_duplicates()
            df = df[df["clean_comment"].str.strip() != ""]
            logger.debug("Data preprocessing completed")
            return df
        except Exception:
            logger.error("Preprocessing failed", exc_info=True)
            raise

    

    def split_and_save(self, df: pd.DataFrame) -> DataIngestionArtifact:
        try:
            train_df, test_df = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=42
            )

            raw_data_path = os.path.join(self.config.data_path, "raw")
            os.makedirs(raw_data_path, exist_ok=True)

            train_path = os.path.join(raw_data_path, "train.csv")
            test_path = os.path.join(raw_data_path, "test.csv")

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            logger.debug("Train and test data saved successfully")

            return DataIngestionArtifact(
                train_data_path=train_path,
                test_data_path=test_path
            )

        except Exception:
            logger.error("Failed during split/save", exc_info=True)
            raise


    def run(self) -> DataIngestionArtifact:
        df = self.load_data()
        processed_df = self.preprocess_data(df)
        return self.split_and_save(processed_df)



# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
def main():
    try:
        """params_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../params.yaml"
        )"""

        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        params_path = PROJECT_ROOT / "src/config/config.yaml"


        with open(params_path, "r") as f:
            params = yaml.safe_load(f)

        ingestion_config = DataIngestionConfig(
            data_url=params["data_ingestion"]["data_url"],
            test_size=params["data_ingestion"]["test_size"],
            data_path=params["data_ingestion"]["data_path"]
        )

        ingestion = DataIngestion(ingestion_config)
        artifact = ingestion.run()

        logger.info("Data ingestion completed successfully")
        logger.info("Train data saved at: %s", artifact.train_data_path)
        logger.info("Test data saved at: %s", artifact.test_data_path)

    except Exception:
        logger.error("Data ingestion failed", exc_info=True)
        raise


if __name__ == "__main__":
    main()
