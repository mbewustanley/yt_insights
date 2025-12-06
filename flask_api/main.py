import matplotlib
matplotlib.use('Agg') # Use non-interactive backend before importing pyplot

import io
import re
import pickle
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes



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

# Load the model and vectorizer from the model registry and local storage
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    #set MLFlow tracking URI to your server
    mlflow.set_tracking_uri("http://ec2-13-247-106-239.af-south-1.compute.amazonaws.com:5000/")
    client = MlflowClient()

    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(model_uri) # sklearn <> pyfunc

    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
    return model, vectorizer

#def load_model(model_path, vectorizer_path):
    """Load the trained model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
      
        return model, vectorizer
    except Exception as e:
        raise


#initialize model and vectorizer
model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "2", "./tfidf_vectorizer.pkl")


#initialize first(default) route of flask api to test
@app.route('/')
def home():
    return "Welcome to Stanley's flask api"



# predict route to test api on postman
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')
    # print("i am the comment: ",comments)
    # print("i am the comment type: ",type(comments))
    
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # Convert the sparse matrix to dense format
        dense_comments = transformed_comments.toarray()  # Convert to dense array
        
        # Make predictions
        predictions = model.predict(dense_comments).tolist()  # Convert to list
        
        # Convert predictions to strings for consistency
        # predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Return the response with original comments and predicted sentiments
    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    return jsonify(response)


if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
