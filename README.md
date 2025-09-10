## Building

```
docker build -t sam-server1 -f Server1.Dockerfile .
docker build -t sam-server2 -f Server2.Dockerfile .
```

## Running

Server1 is intended to run on AWS ECS (or a standard EC2 instance). It exposes a
small web UI on port `5050` and expects two sets of credentials supplied via
environment variables:

* `APP_USER1` / `APP_PASS1`
* `APP_USER2` / `APP_PASS2`

The web interface presents a login form and only proceeds once a valid
username/password pair is supplied.

Both servers read and write job data and model weights from a shared storage
location. Mount an S3 bucket into each container and ensure both processes
point to the same path. The base directory defaults to `/mnt/s3` but can be
overridden with the `SHARED_DIR` environment variable.

When a user logs in, Server1 scales a SageMaker endpoint variant up to one
instance via the AWS API. Server1 monitors the queue and scales the endpoint
back to zero five minutes after no images remain, restarting it when new files
appear.
It also watches each user's `resized/` directory and invokes the SageMaker
endpoint for any new images found, so files added outside the upload workflow
are processed automatically.

```
docker run -it --rm -p 5050:5050 \
  -v /path/on/s3:/mnt/s3 \
  -e AWS_REGION="us-east-1" \
  -e S3_BUCKET="your-bucket" \
  -e SAGEMAKER_ENDPOINT="sam-server2-endpoint" \
  -e APP_USER1="user_a" -e APP_PASS1="pass_a" \
  -e APP_USER2="user_b" -e APP_PASS2="pass_b" \
  sam-server1
```

Server2 runs as a SageMaker endpoint. The container polls the shared storage for
new images and processes them as they appear. Heavy models
(`segment-anything`, `ultralytics`, `rembg[gpu]`) are imported at module load
time so they remain in memory while the endpoint is active.

### Environment Variables

Both Server1 and the SageMaker container require AWS credentials and region
configuration to interact with services such as S3, EC2, and Secrets Manager.
Provide these via one of the standard AWS mechanisms (environment variables,
credentials files, or IAM roles). When using environment variables, set:

* `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` (and `AWS_SESSION_TOKEN` if using
  temporary credentials)
* `AWS_REGION` or `AWS_DEFAULT_REGION`

If your network uses a proxy, also configure `HTTPS_PROXY`/`HTTP_PROXY` so
`botocore` can reach AWS endpoints. The `SHARED_DIR` variable may be set to change
the mount point for the shared S3 directory.

### Models

When Server1 starts it ensures required model weights are present under
`shared/models`, downloading any missing files without overwriting existing
ones. The following weights are fetched:

* `vit_l.pth` – Segment Anything
* `birefnet-dis.onnx` – rembg
* `yolov8n.pt` and `yolov8n-seg.pt` – Ultralytics YOLO

You may pre-populate `shared/models` to skip the network downloads entirely.
