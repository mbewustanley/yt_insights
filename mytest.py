"""Ignore this File
This is a side script to test my model performance """




import pickle
import os
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment
    


#def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    #set MLFlow tracking URI to your server
    mlflow.set_tracking_uri("http://ec2-13-247-106-239.af-south-1.compute.amazonaws.com:5000/")
    client = MlflowClient()

    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(model_uri) # sklearn <> pyfunc

    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
    return model, vectorizer

#initialize model and vectorizer
#model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "2", "./tfidf_vectorizer.pkl")



def load_model(model_path, vectorizer_path):
    """Load the trained model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
      
        return model, vectorizer
    except Exception as e:
        raise
model, vectorizer = load_model('./lgbm_model.pkl', './tfidf_vectorizer.pkl')

"""

comments = [
    "This is actually really impressive, great job!",
    "I’m not sure how I feel about this, but it’s interesting.",
    "This didn’t meet my expectations at all.",
    "Wow, this exceeded what I thought it would be.",
    "It’s okay, nothing special.",
    "I love the creativity here.",
    "Honestly, this feels poorly executed.",
    "I don’t really have an opinion on this.",
    "Fantastic work — keep it up!",
    "This looks a bit messy and unorganized.",
    "Seems fine to me.",
    "Super clean and well thought out.",
    "I didn’t enjoy this at all.",
    "Neutral on this, just observing.",
    "This is a solid improvement from the last version.",
    "I think this needs a lot more work.",
    "Pretty decent overall.",
    "This feels rushed and unfinished.",
    "Nice! This really made my day.",
    "Can’t say it’s good or bad — just okay."
]
"""

comments = [input("Please enter your comment: ")]

def test_model(model, vectorizer, comments, islist=False):
  
    init_comments = [preprocess_comment(comment) for comment in comments]
    
    vectorized_comments = vectorizer.transform(init_comments)
    
    dense_comments = vectorized_comments.toarray()

    predictions = model.predict(dense_comments).tolist()

    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    print(response[0:9])

    #return response


test_model(model, vectorizer, comments)