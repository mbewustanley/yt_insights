# mlflow boto3 awscli
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import mlflow
import mlflow.sklearn


### MAX FFEATURES?

#set mlflow tracking and experiment
mlflow.set_tracking_uri("http://ec2-13-245-71-96.af-south-1.compute.amazonaws.com:5000")
mlflow.set_experiment('Exp 3 - Tfidf Tigram max_features')

#load data
df = pd.read_csv('reddit_preprocessing.csv').dropna(subset=['clean_comment'])


# Function to run the experiment
def run_experiment_tfidf_max_features(max_features):
    ngram_range = (1,3) #Tigram setting

    # vectorization using TF-IDF with varying max_features 
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
    
    X_train, X_test, y_train, y_test = train_test_split(df['clean_comment'], df['category'], test_size=0.2, random_state=42)

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # define and train random forest model
    with mlflow.start_run() as run:
        #set tags for the experiment
        mlflow.set_tag('mlflow.runName', f'TFIDF_Tigrams_max_features_{ngram_range}_RandomForest')
        mlflow.set_tag('experiment_type', 'feature-engineering')
        mlflow.set_tag('model_type', 'RandomForestClassifier')

        # add desctiption
        mlflow.set_tag('description', f"RandomForest with TFIDF tigrams, max_features= {max_features}")

        # Log vectorizer parameters
        mlflow.log_param('vectorizer_type', "TF-IDF")
        mlflow.log_param('ngram_range', ngram_range)
        mlflow.log_param('vectorizer_max_features', max_features)

        #log randomForest parameters
        n_estimators = 200
        max_depth = 15
        
        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_param('max_depth', max_depth)

        #initialize and train model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        #make predictions and log metrics
        y_pred = model.predict(X_test)

        # log accuracy
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric('accuracy', accuracy)

        # log classification report
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        for label, metrics in classification_rep.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    mlflow.log_metric(f'{label}_{metric}', value)
        
        # log confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f"Confusion Matrix: TF-IDF Tigrams, max_features={max_features}")
        plt.savefig("confusion_matrix_3.png")
        mlflow.log_artifact('confusion_matrix_3.png')
        plt.close()


        # Log the model
        mlflow.sklearn.log_model(model, f"random_forest_model_tfidf_tigrams_{max_features}")

# run experiments for varying max features
max_features_values = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

for max_features in max_features_values:
    run_experiment_tfidf_max_features( max_features)

