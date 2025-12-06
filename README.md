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
- Create an s3 bucket
- Create EC2 machine (Ubuntu) & add Security groups 5000 port

Run the following command on EC2 machine
```
sudo apt update

sudo apt install python3-pip

sudo apt install pipenv

sudo apt install virtualenv

mkdir mlflow

cd mlflow

pipenv install mlflow

pipenv install awscli

pipenv install boto3

pipenv shell
```


## Then set aws credentials
```
aws configure
```

## Finally
``` 
mlflow server -h 0.0.0.0 --default-artifact-root s3://stanley-mlflow-bucket-27 --allowed-hosts "ec2-13-244-77-114.af-south-1.compute.amazonaws.com:5000"
```

#open Public IPv4 DNS to the port 5000


## set uri in your local terminal and in your code 

export MLFLOW_TRACKING_URI:
```
http://ec2-13-244-77-114.af-south-1.compute.amazonaws.com:5000
```



# Local

conda create -n youtube python=3.11 -y

conda activate youtube

pip install -r requirements.txt

## DVC
dvc init

dvc repro

dvc dag

## AWS

aws configure

### Json data demo in postman

http://localhost:5000/predict

{
    "comments": ["This video is awsome! I loved a lot", "Very bad explanation. poor video"]
}

chrome://extensions

## How to get youtube api key from GCP

https://www.youtube.com/watch?v=i_FdiQMwKiw


# AWS CICD Deployment with GITHUB ACTIONS

## 1. Login to AWS Console.

## 2. create IAM user for deployment

    # with specific access

    1. EC2 access

    2. ECR: Elastic Container Registry to save our docker image in aws

    # Description About the deployment

    1. Build docker image of the source code

    2. Push the docker image to ECR

    3. Launch EC2

    4. Pull image from ECR in EC2

    5. Launch docker image in EC2

    # Policy:

    1. AmazonEC2ContainerRegistryFullAccess

    AmazonEC2FullAccess

## 3. create ECR repo to store/save docker image

    - save the URI: 

## 4. Create EC2 machine (ubuntu)

## 5. Open EC2 and install Docker in EC2 machine:

    # optional

    sudo apt-get update -y

    sudo apt-get upgrade

    # required

    curl -fsSL https://get.docker.com -o get-docker.sh

    sudo sh get-docker.sh

    sudo usermod -aG docker ubuntu

    newgrp docker

## 6. Configure EC2 as seld-hosted runner:

    settings>actions>runner>new self hosted runner> choose os> then run command one by one

## 7. setup github secrets:

    AWS_ACCESS_KEY_ID=
    AWS_SECRET_ACCESS_KEY=
    AWS_REGION=
    AWS_ECR_LOGIN_URI=
    ECR_REPOSITORY_NAME=



# Docker custom Images

- "docker build -t mcstanleydocker27/youtubeinsights:latest ."

- "docker run -p 5000:5000 mcstanleydocker27/youtubeinsights:latest"

- "docker run -d -p 5000:5000 mcstanleydocker27/youtubeinsights:latest" # this continues to run the container after your local terminal has been shutdown


# Docker Push to hub

- docker login

- docker push mcstanleydocker27/youtubeinsights:latest