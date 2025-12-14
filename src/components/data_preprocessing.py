import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.config.config import DataPreprocessingConfig
from src.entity.artifact import DataPreprocessingArtifact
from src.utils.logger import get_logger

logger = get_logger("data_preprocessing")

# Download required NLTK data (only once)
nltk.download('wordnet')
nltk.download('stopwords')

class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config
        self.stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_comment(self, comment: str) -> str:
        try:
            comment = comment.lower().strip()
            comment = re.sub(r'\n', ' ', comment)
            comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
            comment = ' '.join([w for w in comment.split() if w not in self.stop_words])
            comment = ' '.join([self.lemmatizer.lemmatize(w) for w in comment.split()])
            return comment
        except Exception as e:
            logger.error(f"Error in preprocessing comment: {e}")
            return comment

    def normalize_text(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df['clean_comment'] = df['clean_comment'].apply(self.preprocess_comment)
            logger.debug('Text normalization completed')
            return df
        except Exception as e:
            logger.error(f"Error during text normalization: {e}")
            raise

    def save_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> DataPreprocessingArtifact:
        try:
            os.makedirs(self.config.interim_data_dir, exist_ok=True)

            train_path = os.path.join(self.config.interim_data_dir, "train_processed.csv")
            test_path = os.path.join(self.config.interim_data_dir, "test_processed.csv")

            train_data.to_csv(train_path, index=False)
            test_data.to_csv(test_path, index=False)

            logger.debug(f"Processed data saved to {self.config.interim_data_dir}")

            return DataPreprocessingArtifact(
                train_processed_path=train_path,
                test_processed_path=test_path
            )
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise

    def run(self) -> DataPreprocessingArtifact:
        try:
            train_data = pd.read_csv(os.path.join(self.config.raw_data_dir, 'train.csv'))
            test_data = pd.read_csv(os.path.join(self.config.raw_data_dir, 'test.csv'))
            logger.debug('Raw data loaded successfully')

            train_processed = self.normalize_text(train_data)
            test_processed = self.normalize_text(test_data)

            return self.save_data(train_processed, test_processed)
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            raise

def main():
    try:
        config = DataPreprocessingConfig(
            raw_data_dir='./data/raw',
            interim_data_dir='./data/interim'
        )

        preprocessor = DataPreprocessing(config)
        artifact = preprocessor.run()

        logger.info("Data preprocessing completed successfully")
        logger.info(f"Train processed: {artifact.train_processed_path}")
        logger.info(f"Test processed: {artifact.test_processed_path}")

    except Exception as e:
        logger.error("Preprocessing pipeline failed", exc_info=True)
        raise


if __name__ == "__main__":
    main()
