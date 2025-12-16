import os
import subprocess
from dotenv import load_dotenv


REQUIRED_ENV_VARS = [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_DEFAULT_REGION",
    "MLFLOW_TRACKING_URI",
    "HOST_PORT",
    "CONTAINER_PORT",
    "DOCKER_IMAGE",
    "CONTAINER_NAME",
]


def validate_env():
    """
    Ensure all required environment variables are set.
    """
    missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]

    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )


def run_docker():
    # Load .env from project root
    load_dotenv()

    # Validate env variables
    validate_env()

    # Read env variables
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_DEFAULT_REGION")

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")

    host_port = os.getenv("HOST_PORT")
    container_port = os.getenv("CONTAINER_PORT")

    image_name = os.getenv("DOCKER_IMAGE")
    container_name = os.getenv("CONTAINER_NAME")

    docker_cmd = [
        "docker", "run", "-it",
        "-p", f"{host_port}:{container_port}",
        "--name", container_name,
        "-e", f"AWS_ACCESS_KEY_ID={aws_access_key}",
        "-e", f"AWS_SECRET_ACCESS_KEY={aws_secret_key}",
        "-e", f"AWS_DEFAULT_REGION={aws_region}",
        "-e", f"MLFLOW_TRACKING_URI={mlflow_uri}",
        image_name
    ]

    print("\nRunning Docker command:")
    print(" ".join(docker_cmd))

    subprocess.run(docker_cmd, check=True)


if __name__ == "__main__":
    run_docker()
