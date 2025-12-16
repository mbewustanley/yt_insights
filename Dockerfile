# base image
FROM python:3.11-slim

# Environment Variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=http://ec2-13-244-115-169.af-south-1.compute.amazonaws.com:5000/

# work directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libffi-dev \
    libssl-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# copy requirements first (cache friendly)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Download NLTK corpora
RUN python -m nltk.downloader stopwords wordnet

# copy all files into work directory
#COPY . /app
COPY flask_api/ flask_api/
COPY src/ src/

# Expose Flask port
EXPOSE 8000

# Run flask app
CMD ["python3", "flask_api/app.py"]
