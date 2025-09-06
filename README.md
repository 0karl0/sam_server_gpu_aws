## Building

```
docker build -t sam-server1 -f Server1.Dockerfile .
docker build -t sam-server2 -f Server2.Dockerfile .
```

## Running

Server1 is intended to run on an AWS EC2 instance. It exposes a small web UI on
port `5050` and expects two sets of credentials supplied via environment
variables:

* `APP_USER1` / `APP_PASS1`
* `APP_USER2` / `APP_PASS2`

The web interface presents a login form and only proceeds once a valid
username/password pair is supplied.

Both servers read and write job data and model weights from a shared storage
location on EC2. Mount this storage into each container (or expose it via a
network filesystem like EFS or an S3 bucket mounted with `s3fs`) and ensure both
processes point to the same path. The base directory defaults to `/mnt/shared`
but can be overridden with the `SHARED_DIR` environment variable.

The GPU worker (Server2) is managed through the RunPod API. When a user logs in,
Server1 submits jobs to a RunPod **serverless** endpoint. Each invocation
processes any pending images and exits. Models are loaded once at container
startup so warm runs reuse the weights already in memory, minimizing "cold start"
delays.

Server1 monitors the queue and stops the GPU pod five minutes after no images remain, restarting it when new files appear.

```
docker run -it --rm -p 5050:5050 \
  -v /path/on/ec2/shared:/mnt/shared \
  -e RUNPOD_API_KEY="your_api_key_here" \
  -e GPU_POD_ID="your_gpu_pod_id_here" \
  -e APP_USER1="user_a" -e APP_PASS1="pass_a" \
  -e APP_USER2="user_b" -e APP_PASS2="pass_b" \
  sam-server1
```

Server2 is deployed as a RunPod serverless worker. The container is still built
from `Server2.Dockerfile`, but instead of running continuously it reacts to jobs
from Server1 via the RunPod library. Heavy models (`segment-anything`,
`ultralytics`, `rembg[gpu]`) are imported at module load time so they remain in
memory between invocations when the worker is warm.

### Models

When Server1 starts it ensures required model weights are present under
`shared/models`, downloading any missing files without overwriting existing
ones. The following weights are fetched:

* `vit_l.pth` – Segment Anything
* `birefnet-dis.onnx` – rembg
* `yolov8n.pt` and `yolov8n-seg.pt` – Ultralytics YOLO

You may pre-populate `shared/models` to skip the network downloads entirely.
