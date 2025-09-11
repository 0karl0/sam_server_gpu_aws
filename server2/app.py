import os
from pathlib import Path

import boto3
import cv2
from flask import Flask, jsonify, request
import logging

from worker import generate_masks, save_masks, load_settings

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
    The image is downloaded from S3, processed to generate segmentation masks,
    and all outputs are written back to the specified S3 prefix.
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

    img = cv2.imread(str(local_path))
    if img is None:
        return jsonify({"error": "failed to read image"}), 400

    # Run SAM-based segmentation.
    settings = load_settings("/tmp/nonexistent.json")
    masks, image = generate_masks(str(local_path), settings)
    base = local_path.stem
    masks_dir = Path("/tmp/masks")
    masks_dir.mkdir(parents=True, exist_ok=True)
    save_masks(masks, image, base, str(masks_dir))

    uploaded = []
    if output_s3:
        out_bucket, out_prefix = output_s3.replace("s3://", "").split("/", 1)
        if not out_prefix.endswith("/"):
            out_prefix += "/"
        local_files = list(masks_dir.glob("*"))
        for path in local_files:
            key = out_prefix + path.name
            logger.info("[invoke] uploading %s to s3://%s/%s", path, out_bucket, key)
            s3.upload_file(str(path), out_bucket, key)
            uploaded.append(f"s3://{out_bucket}/{key}")

    logger.info("[invoke] processing complete for %s", input_s3)
    return jsonify({"status": "done", "outputs": uploaded})


if __name__ == "__main__":  # pragma: no cover
    app.run(host="0.0.0.0", port=8080, debug=False)
