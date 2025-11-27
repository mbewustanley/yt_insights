# yt_insights
Youtube Sentiments Insight using a simple machine learning model

## Project Plan

- Data Collection
    - yt api (not available)
    - reddit comments sentiment analysis data

- Data Preprocessing
    - Exploratory Data Analysis (EDA)

- Baseline Model Building

- Setup MLFLOW server on AWS for experiment tracking

- Improve Baseline Model
    - for vectorization: Bag of Words (for text data)
    - Max feature identification for entire token of data
    - Handling imbalanced data
    - Hyperparameter tuning with multiple models
    - Model stacking    

- Build ML pipeline using DVC

- Add model to model registory (containerization)

- Implement Chrome plugin

- CICD workflow

- Dockerization

- Deployment using AWS

- Github


## Notes:

- conda env name: YTenv


## MLFLOW on AWS Setup:

- Login to AWS console.
- Create IAM user with AdministratorAccess
- Export the credentials in your AWS CLI by running "aws configure"
- Create a s3 bucket
- Create EC2 machine (Ubuntu) & add Security groups 5000 port

Run the following commands on EC2 machine

` sudo apt update

` sudo apt install python3-pip

` sudo apt install pipenv

` sudo apt install virtualenv

` mkdir mlflow

` cd mlflow

` pipenv install mlflow

` pipenv install awscli

` pipenv install boto3

` pipenv shell


## Then set aws credentials

` aws configure


#Finally 
` mlflow server -h 0.0.0.0 --default-artifact-root s3://stanley-mlflow-bucket-27

#open Public IPv4 DNS to the port 5000


#set uri in your local terminal and in your code 
export MLFLOW_TRACKING_URI = mlflow server -h 0.0.0.0 --default-artifact-root s3://stanley-mlflow-bucket-27 --allowed-hosts "ec2-13-247-179-110.af-south-1.compute.amazonaws.com:5000"

