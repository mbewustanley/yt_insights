# mlflow, boto3, awscli, optuna, xgboost, imbalanced-learn

import optuna # a hyperparameter optimization framework for ml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


#set mlflow tracking and experiment
mlflow.set_tracking_uri("http://ec2-13-244-203-74.af-south-1.compute.amazonaws.com:5000")
mlflow.set_experiment('Exp 5 - ML Algos with HP Tuning')


# load data
df = pd.read_csv('reddit_preprocessing.csv').dropna(subset=['clean_comment'])

# remap the class labels from [-1,0,1] to [2,0,1]. xgboost does not take negatove values
df['category'] = df['category'].map({-1:2, 0:0, 1:1})

# remove rows where target labels (category) are NAN
df = df.dropna(subset=['category'])

ngram_range = (1,3) #trigram settings
max_features = 1000  # set max features for TF-IDF

# train-test split before vectorization and resampling
X_train, X_test, y_train, y_test = train_test_split(df['clean_comment'], df['category'], test_size=0.2, random_state=42)

#vectorization using TF-IDF, fit on training data
vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
X_train_vec = vectorizer.fit_transform(X_train) # fit on training data
X_test_vec = vectorizer.transform(X_test) # transform test data

smote = SMOTE(random_state=42)
X_train_vec, y_train = smote.fit_resample(X_train_vec, y_train)

# function to log results in MLFLOW
def log_mlflow(model_name, model, X_train, X_test, y_train, y_test):
    with mlflow.start_run():
        #log model type
        mlflow.set_tag('mlflow.runName', f"{model_name}_SMOTE_TFIDF_Trigrams")
        mlflow.set_tag('experiment_type', "algorithm_comparison")

        # log algorithm name as a parameter
        mlflow.log_param("algo_name", model_name)

        # train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        #log accuracy
        accuracy= accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        #log classification report
        classification_rep =classification_report(y_test, y_pred, output_dict=True)
        for label, metrics in classification_rep.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    metric.log(f"{label}_{metric}", value)
        
        #log the model
        mlflow.sklearn.log_model(model, f"{model_name}_model")


#optuna objective function for xgboost
def objective_xgboost(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    max_depth = trial.suggest_int('max_depth', 3, 10)

    model = XGBClassifier(n_estimators-n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_State=42)
    return accuracy_score(y_test, model.fit(X_train_vec, y_train).predict(X_test_vec))


# run optuna for XGBoost, log the best model only
def run_optuna_experiment():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_xgboost, n_trials=30)

    #get the best parameters and log only the best model
    best_params = study.best_params
    best_model = XGBClassifier(n_estimators=best_params['n_estimators'], learning_rate=best_params['learning_rate'], max_depth = best_params['max_depth'], random_state=42)

    #log the nest model with mlflow, passing the algo_name as 'xgboost'
    log_mlflow("XGBoost", best_model, X_train_vec, X_test_vec, y_train, y_test)

#run the experiment for xgboost
run_optuna_experiment()