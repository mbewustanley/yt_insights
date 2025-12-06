# base image
FROM python:3.11-slim-buster

#work directory
WORKDIR /app

# copy all files into work directory
COPY . /app

# run dependencies
RUN pip install -r requirements.txt

# commands
CMD ["python3", "app.py"]
