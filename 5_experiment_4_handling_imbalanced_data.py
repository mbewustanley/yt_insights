# mlflow boto3 awscli
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN


## Handling Imbalanced Data


#set mlflow tracking and experiment
mlflow.set_tracking_uri("http://ec2-13-244-203-74.af-south-1.compute.amazonaws.com:5000")
mlflow.set_experiment('Exp 4 - Handling Imbalanced Data')

# load data
df = pd.read_csv('reddit_preprocessing.csv').dropna(subset=['clean_comment'])

# function to run experiment
def run_imbalanced_experiment(imbalance_method):
    ngram_range = (1,3)
    max_features = 1000

    X_train, X_test, y_train, y_test = train_test_split(df['clean_comment'], df['category'], test_size=0.2, random_state=42)

    #vectorization using TF-IDF, fit on training data
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
    X_train_vec = vectorizer.fit_transform(X_train) # fit on training data
    X_test_vec = vectorizer.transform(X_test) # transform test data

    #Handle class imbalance based on selected mathod (only applying to training set)
    if imbalance_method == 'class_weights':
        #use class_weight in Random Forest
        class_weight = 'balanced'
    else:
        class_weight = None # do not apply class weight if using resampling

        # Resampling techniques (applied only to training set)
        if imbalance_method == 'oversampling':
            smote = SMOTE(random_state=42)
            X_train_vec, y_train = smote.fit_resample(X_train_vec, y_train)
        elif imbalance_method == 'adasyn':
            adasyn = ADASYN(random_state=42)
            X_train_vec, y_train = adasyn.fit_resample(X_train_vec, y_train)
        elif imbalance_method == 'undersampling':
            rus = RandomUnderSampler(random_state=42)
            X_train_vec, y_train = rus.fit_resample(X_train_vec, y_train)
        elif imbalance_method == 'smote_enn':
            smote_enn = SMOTEENN(random_state=42)
            X_train_vec, y_train = smote_enn.fit_resample(X_train_vec, y_train)        

    
    # Define and train a Random Forest model
    with mlflow.start_run() as run:
        #set tags for the experiment and run
        mlflow.set_tag('mlFlow.runName', f'Imbalance_{imbalance_method}_RandomForest_TFIDF_Tigrams')
        mlflow.set_tag('experiment_type', 'imbalance_handling')
        mlflow.set_tag('model_type', 'RandomForestClassifier')

        #add description
        mlflow.set_tag('description', f'RandomForest with TF-IDF Tigrams, imbalance handling method={imbalance_method}')

        # log vectorizer parameters
        mlflow.log_param('vectorizer_type', 'TF-IDF')
        mlflow.log_param('ngram_range', ngram_range)
        mlflow.log_param('vectorizer_max_features', max_features)

        #log Random Forest parameters
        n_estimators = 200
        max_depth = 15

        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_param('max_depth', max_depth)
        mlflow.log_param('imbalance_method', imbalance_method)

        #initialize and train the model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, class_weight='balanced')
        model.fit(X_train_vec, y_train)

        #make predictions and log metrics
        y_pred = model.predict(X_test_vec)

        #log accuracy
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric('accuracy', accuracy)

        #log classification report
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        for label, metrics in classification_rep.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric}", value)


        #log confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f"confusion Matrix: TFIDF Tigrams, Imbalance={imbalance_method}")
        confusion_matrix_filename = f"confusion_matrix_{imbalance_method}.png"
        plt.savefig(confusion_matrix_filename)
        mlflow.log_artifact(confusion_matrix_filename)
        plt.close()


        #log the model
        mlflow.sklearn.log_model(model, f"random_forest_model_tfidf_tigrams_imbalnce_{imbalance_method}")

#run the experiments for different imbalance methods
imbalance_methods = ['class_weights', 'oversampling', 'adasyn', 'undersampling', "smote_enn"]

for method in imbalance_methods:
    run_imbalanced_experiment(method)