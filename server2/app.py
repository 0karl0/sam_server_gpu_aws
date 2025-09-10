import json
import os
import subprocess
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
import cv2
from flask import Flask, jsonify, request
import logging

s3 = boto3.client("s3")
app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
SHARED_DIR = os.getenv("SHARED_DIR", "/mnt/s3")


def get_single_secret_value(secret_name: str) -> dict:
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=AWS_REGION)
    return client.get_secret_value(SecretId=secret_name)


def _mount_s3_from_secret(mount_point: str) -> None:
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        try:
            secret = get_single_secret_value("s3bucket")
            secret_str = secret["SecretString"]
            try:
                secret_dict = json.loads(secret_str)
                bucket = secret_dict.get("S3_BUCKET", secret_str)
            except json.JSONDecodeError:
                bucket = secret_str
            if bucket.startswith("arn:aws:s3:::"):
                bucket = bucket.split(":::", 1)[1]
        except ClientError as e:  # pragma: no cover
            logger.error("[s3] failed to retrieve bucket secret: %s", e)
            return

    try:
        os.makedirs(mount_point, exist_ok=True)
        if not os.path.ismount(mount_point):
            subprocess.run(["s3fs", bucket, mount_point], check=True)
        logger.info("[s3] mounted s3://%s at %s", bucket, mount_point)
    except Exception as e:  # pragma: no cover
        logger.error("[s3] failed to mount s3://%s at %s: %s", bucket, mount_point, e)


_mount_s3_from_secret(SHARED_DIR)


@app.route("/ping", methods=["GET"])
def ping():
    """Health check endpoint."""
    return "pong", 200


@app.route("/invocations", methods=["POST"])
def invoke():
    """Handle an inference request.

    Expects a JSON payload with at least:
      {"s3": "s3://bucket/path/to/input.png", "output": "s3://bucket/path/out.png"}
    The image is downloaded from S3, processed (placeholder), and uploaded
    to the specified output location.
    """
    evt = request.get_json()
    input_s3 = evt["s3"]
    output_s3 = evt.get("output")

    logger.info("[invoke] received request: input=%s output=%s", input_s3, output_s3)
    bucket, key = input_s3.replace("s3://", "").split("/", 1)
    local_path = Path("/tmp") / Path(key).name
    logger.info("[invoke] downloading %s to %s", input_s3, local_path)
    s3.download_file(bucket, key, str(local_path))
    logger.info("[invoke] download complete for %s", input_s3)

    # Placeholder for real processing logic.
    img = cv2.imread(str(local_path))
    if img is None:
        return jsonify({"error": "failed to read image"}), 400

    # TODO: integrate worker.py processing here.

    if output_s3:
        out_bucket, out_key = output_s3.replace("s3://", "").split("/", 1)
        logger.info("[invoke] uploading result to %s", output_s3)
        s3.upload_file(str(local_path), out_bucket, out_key)
        logger.info("[invoke] result uploaded for server1 at %s", output_s3)

    logger.info("[invoke] processing complete for %s", input_s3)
    return jsonify({"status": "done"})


if __name__ == "__main__":  # pragma: no cover
    app.run(host="0.0.0.0", port=8080, debug=False)
