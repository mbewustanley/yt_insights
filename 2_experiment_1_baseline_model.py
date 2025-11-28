import re
import nltk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# To test mlflow
"""mlflow.set_tracking_uri('http://ec2-13-244-77-114.af-south-1.compute.amazonaws.com:5000')

with mlflow.start_run():
    mlflow.log_param('param1', 15)
    mlflow.log_metric('metric1', 0.89)"""


# Load data
df = pd.read_csv('https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')
df.head()

# drop NA, duplicate values
df.dropna(inplace=True)

df.drop_duplicates(inplace=True)

df = df[~(df['clean_comment'].str.strip()=='')]


# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet') 

# Preprocessing function
def preprocess_comment(comment):
    comment = comment.lower()
    comment = comment.strip()
    comment = re.sub(r'\n', ' ', comment) # replace newline withspace
    comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment) # replace alphanumeric special char with blank except punctuations

    # remove stopwords but retain important ones for sentimental analysis
    stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
    comment = ' '.join([word for word in comment.split() if word not in stop_words])

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

    return comment

# Apply preprocessing function to the 'clean_comment' column
df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)



## Model Training

# step 1: vectorize the comments using bag of words (countvectorizer)
vectorizer = CountVectorizer(max_features=10000)  #Bag of words model with a limit of 10000 features

X = vectorizer.fit_transform(df['clean_comment']).toarray()
y = df['category'] # Assuming 'sentiment' is the target variable (0 or 1 for binary classification)


# make sure pip install boto3, awscli and then !aws configure on terminal
# Set mlflow tracking
mlflow.set_tracking_uri('http://ec2-13-244-77-114.af-south-1.compute.amazonaws.com:5000')
mlflow.set_experiment('RF Baseline')

#split data(0.80 train)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# define and train a random forest baseline model using a simple train-test split
with mlflow.start_run() as run:
    # Log a description for the run
    mlflow.set_tag('mlflow.runName', 'RandomForest_Baseline_TrainTestSplit')
    mlflow.set_tag('experiment_type', "baseline")
    mlflow.set_tag('model_type', "RandomForestClassifier")

    # Add a description
    mlflow.set_tag('description', 'Baseline RandomForest model for sentiment analysis using bag of words (BOW)')

    # log parameters for the vectorizer
    mlflow.log_param('vectorizer_type', 'CountVectorizer')
    mlflow.log_param('vectorizer_max_features', vectorizer.max_features)

    # log random forest parameters
    n_estimators = 200
    max_depth = 15
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_param('max_depth', max_depth)

    # initialize and train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # make predictions
    y_pred = model.predict(X_test)

    # log metrics for each class and accuracy
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric('accuracy', accuracy)

    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    for label, metrics in classification_rep.items():
        if isinstance(metrics, dict): # i.e. for precision, recall, f1-score ...
            for metric, value in metrics.items():
                mlflow.log_metric(f'{label}_{metric}', value)

    
    # confusion matrix plot
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion matrix')

    # save and log confusion matrix plot
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')

    #log the Random forest model
    mlflow.sklearn.log_model(model, 'random_forest_model')

    # optional - log the dataset too if it's small
    df.to_csv('dataset.csv', index=False)
    mlflow.log_artifact('dataset.csv')


df.to_csv('reddit_preprocessing.csv', index=False)


