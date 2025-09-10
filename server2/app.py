import os
from pathlib import Path

import boto3
import cv2
from flask import Flask, jsonify, request
import logging

s3 = boto3.client("s3")
app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
