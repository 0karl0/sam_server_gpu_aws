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
location. Mount an EFS volume into each container and ensure both processes
point to the same path. The base directory defaults to `/mnt/efs` but can be
overridden with the `SHARED_DIR` environment variable.

When a user logs in, Server1 starts a dedicated GPU EC2 instance (e.g.,
`g4dn.xlarge`) via the AWS API. Server1 monitors the queue and stops the
instance five minutes after no images remain, restarting it when new files
appear.

```
docker run -it --rm -p 5050:5050 \
  -v /path/on/efs:/mnt/efs \
  -e AWS_REGION="us-east-1" \
  -e GPU_INSTANCE_ID="i-xxxxxxxx" \
  -e APP_USER1="user_a" -e APP_PASS1="pass_a" \
  -e APP_USER2="user_b" -e APP_PASS2="pass_b" \
  sam-server1
```

Server2 runs on the GPU EC2 instance. The container polls the shared storage for
new images and processes them as they appear. Heavy models
(`segment-anything`, `ultralytics`, `rembg[gpu]`) are imported at module load
time so they remain in memory while the instance is running.

### Models

When Server1 starts it ensures required model weights are present under
`shared/models`, downloading any missing files without overwriting existing
ones. The following weights are fetched:

* `vit_l.pth` – Segment Anything
* `birefnet-dis.onnx` – rembg
* `yolov8n.pt` and `yolov8n-seg.pt` – Ultralytics YOLO

You may pre-populate `shared/models` to skip the network downloads entirely.
