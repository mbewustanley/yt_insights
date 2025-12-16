# --------------------------------------------------
# Matplotlib (non-interactive backend)
# --------------------------------------------------
import matplotlib
matplotlib.use("Agg")

# --------------------------------------------------
# Standard libs
# --------------------------------------------------
import io
import re
import os

# --------------------------------------------------
# Third-party
# --------------------------------------------------
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# --------------------------------------------------
# Flask app
# --------------------------------------------------
app = Flask(__name__)
CORS(app)


def preprocess_comment(comment: str) -> str:
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


# --------------------------------------------------
# MLflow model loading (PIPELINE MODEL)
# --------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://ec2-13-244-115-169.af-south-1.compute.amazonaws.com:5000/"
)

MODEL_NAME = "yt_insights_classifier"
MODEL_STAGE = "Production"  # or "Staging"
MODEL_VERSION = 'latest'

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.sklearn.load_model(model_uri)



# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return "Welcome to Stanley's Flask API"


# --------------------------------------------------
# Prediction (RAW TEXT → MODEL)
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    comments = data.get("comments")

    if not comments:
        return jsonify({"error": "No comments provided"}), 400


    print("comments:", comments)
    print("type(comments):", type(comments))


    # Preprocess
    clean_comments = [preprocess_comment(c) for c in comments]

    # MUST be DataFrame (matches MLflow signature)
    # input_df = pd.DataFrame(
    #     {"clean_comment": clean_comments}
    # )


    try:
        #print("input_df shape:", input_df.shape)
        predictions = model.predict(clean_comments).tolist()
        #predictions = model.predict(input_df).tolist()
        print("predictions:", predictions)
        print("len(predictions):", len(predictions))

    except Exception as e:
        return jsonify({"error": f"{str(e)}"}), 500

    response = [
        {"comment": comment, "sentiment": int(pred)}
        for comment, pred in zip(comments, predictions)
    ]

    assert len(comments) == len(predictions), "Prediction count mismatch"


    return jsonify(response)


# --------------------------------------------------
# Prediction with timestamps
# --------------------------------------------------
@app.route("/predict_with_timestamps", methods=["POST"])
def predict_with_timestamps():
    data = request.get_json()
    comments_data = data.get("comments")

    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    comments = [item["text"] for item in comments_data]
    timestamps = [item["timestamp"] for item in comments_data]

    try:
        predictions = model.predict(comments)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    response = [
        {
            "comment": comment,
            "sentiment": int(pred),
            "timestamp": ts
        }
        for comment, pred, ts in zip(comments, predictions, timestamps)
    ]

    return jsonify(response)


# --------------------------------------------------
# Generate sentiment pie chart
# --------------------------------------------------
@app.route("/generate_chart", methods=["POST"])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get("sentiment_counts")

        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        labels = ["Positive", "Neutral", "Negative"]
        sizes = [
            int(sentiment_counts.get("1", 0)),
            int(sentiment_counts.get("0", 0)),
            int(sentiment_counts.get("-1", 0))
        ]

        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")

        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=140
        )
        plt.axis("equal")

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG", transparent=True)
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype="image/png")

    except Exception as e:
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500


# --------------------------------------------------
# Generate word cloud
# --------------------------------------------------
@app.route("/generate_wordcloud", methods=["POST"])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get("comments")

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        text = " ".join(comments)

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="black",
            colormap="Blues",
            stopwords=set(stopwords.words("english")),
            collocations=False
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format="PNG")
        img_io.seek(0)

        return send_file(img_io, mimetype="image/png")

    except Exception as e:
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500


# --------------------------------------------------
# Generate sentiment trend graph
# --------------------------------------------------
@app.route("/generate_trend_graph", methods=["POST"])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get("sentiment_data")

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        df = pd.DataFrame(sentiment_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["sentiment"] = df["sentiment"].astype(int)
        df.set_index("timestamp", inplace=True)

        monthly_counts = df.resample("M")["sentiment"].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        for s in [-1, 0, 1]:
            if s not in monthly_percentages.columns:
                monthly_percentages[s] = 0

        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        plt.figure(figsize=(12, 6))

        colors = {-1: "red", 0: "gray", 1: "green"}
        labels = {-1: "Negative", 0: "Neutral", 1: "Positive"}

        for s in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[s],
                marker="o",
                label=labels[s],
                color=colors[s]
            )

        plt.title("Monthly Sentiment Percentage Over Time")
        plt.xlabel("Month")
        plt.ylabel("Percentage (%)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG")
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype="image/png")

    except Exception as e:
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500


# --------------------------------------------------
# Run app
# --------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
# other ports 8080, 7000