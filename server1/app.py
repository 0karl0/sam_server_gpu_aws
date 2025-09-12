import os
import json
import threading
import time
from typing import List, Dict
import io
import queue
import boto3
from botocore.exceptions import ClientError

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    session,
    Response,
)
from werkzeug.utils import secure_filename
import requests

import numpy as np
import cv2
from PIL import Image
import pillow_heif  # enables HEIC/HEIF decode in Pillow

# -------------------------
# Paths / Constants
# -------------------------
# Base directory for shared storage. Defaults to "/mnt/s3" but can be
# overridden via the SHARED_DIR environment variable so both servers can point
# to a common local path. S3 access is handled via boto3 rather than a mounted
# filesystem.
SHARED_DIR   = os.getenv("SHARED_DIR", "/mnt/s3")

# Images and user data live in S3; only model weights are cached locally.
MODELS_DIR   = os.path.join(SHARED_DIR, "models")  # downloaded weights
SETTINGS_KEY = "shared/config/settings.json"  # default S3 location for settings

MAX_RESIZE = 1024  # longest side for SAM
ALLOWED_EXT = {"png", "jpg", "jpeg", "webp", "bmp", "tiff", "heic", "heif"}
MAX_CROPS_PER_IMAGE = 5  # limit crops returned per original

# Only the models directory is needed locally and can be created at import time.
os.makedirs(MODELS_DIR, exist_ok=True)

# Helper to configure per-user S3 paths. This updates module-level variables so
# existing code can continue to reference them.
def set_user_dirs(username: str) -> None:
    global SETTINGS_KEY

    SETTINGS_KEY = f"{username}/config/settings.json"

    if s3_client and S3_BUCKET:
        prefixes = [
            f"{username}/uploads/",
            f"{username}/input/",
            f"{username}/resized/",
            f"{username}/output/masks/",
            f"{username}/output/thumbs/",
            f"{username}/output/crops/",
            f"{username}/output/smalls/",
            f"{username}/output/points/",
            f"{username}/config/",
        ]
        for p in prefixes:
            try:
                s3_client.put_object(Bucket=S3_BUCKET, Key=p)
            except Exception as e:  # pragma: no cover - best effort
                print(f"[s3] ensure prefix failed for {p}: {e}")
# Register HEIF opener for Pillow
pillow_heif.register_heif_opener()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "change-me")

album_update_subscribers: List[queue.Queue] = []

def notify_album_update() -> None:
    """Push a notification to all SSE subscribers."""
    for q in list(album_update_subscribers):
        try:
            q.put_nowait("refresh")
        except Exception:
            pass


@app.route("/album_stream")
def album_stream() -> Response:
    """Server-sent events stream for album updates."""

    def gen():
        q: queue.Queue = queue.Queue()
        album_update_subscribers.append(q)
        try:
            while True:
                data = q.get()
                yield f"data: {data}\n\n"
        finally:
            album_update_subscribers.remove(q)

    return Response(gen(), mimetype="text/event-stream")


def get_single_secret_value(secret_name):
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name='us-east-1'
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
        return get_secret_value_response
    except ClientError as e:
        print(e)
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e
    else:
        secret_string = get_secret_value_response['SecretString']
        return json.loads(secret_string)
USERS: Dict[str, str] = {}

print("getting user")
secret = get_single_secret_value("APP_USER1")

user = secret["SecretString"]
print("getting pw")
secret = get_single_secret_value("APP_PASS1")

pw = secret["SecretString"]

if user and pw:
        USERS[user] = pw


print(f'user: {user}')

# AWS configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
GPU_ACTIVE = False
USER_LOGGED_IN = False
_last_work_time = time.time()

print("getting sage")
secret = get_single_secret_value("sagemaker-endpoint")  # os.getenv("SAGEMAKER_ENDPOINT","sam-server2-endpoint")

secret_str = secret["SecretString"]
# SecretString may be a JSON blob like {"SAGEMAKER_ENDPOINT": "name"}
# so try to decode it; fallback to the raw string if decoding fails.
try:
    secret_dict = json.loads(secret_str)
    SAGEMAKER_ENDPOINT = secret_dict.get("SAGEMAKER_ENDPOINT", secret_str)
except json.JSONDecodeError:
    SAGEMAKER_ENDPOINT = secret_str

print(f'sagemaker_endpoint: {SAGEMAKER_ENDPOINT}')
SAGEMAKER_VARIANT = os.getenv("SAGEMAKER_VARIANT", "AllTraffic")
print("getting s3")
secret = get_single_secret_value("s3bucket")
#S3_BUCKET = secret["SecretString"]

secret_str = secret["SecretString"]
# SecretString may be a JSON blob like {"S3_BUCKET": "arn:aws:s3:::my-bucket"}
# so try to decode it; fallback to the raw string if decoding fails.
try:
    secret_dict = json.loads(secret_str)
    S3_BUCKET = secret_dict.get("S3_BUCKET", secret_str)
except json.JSONDecodeError:
    S3_BUCKET = secret_str
# If the bucket is provided as an ARN, extract the bucket name portion.
if S3_BUCKET.startswith("arn:aws:s3:::"):
    S3_BUCKET = S3_BUCKET.split(":::", 1)[1]

print(f'here is the s3 bucket were trying to load: {S3_BUCKET}')

#S3_BUCKET = os.getenv("S3_BUCKET","sam-server-shared-1757294775")
s3_client = boto3.client("s3", region_name=AWS_REGION) if S3_BUCKET else None
if s3_client:
    try:
        s3_client.head_bucket(Bucket=S3_BUCKET)
        print(f"[s3] bucket '{S3_BUCKET}' accessible")
    except Exception as e:
        print(f"[s3] bucket '{S3_BUCKET}' not accessible: {e}")
        s3_client = None

sm_client = (
    boto3.client("sagemaker-runtime", region_name=AWS_REGION)
    if SAGEMAKER_ENDPOINT
    else None
)

print(f'sm_client: {sm_client}')

sagemaker_client = (
    boto3.client("sagemaker", region_name=AWS_REGION)
    if SAGEMAKER_ENDPOINT
    else None
)

print(f'sagemaker_client: {sagemaker_client}')

# Track which mask files have been processed into crops
_processed_mask_files = set()
# Track which resized images are currently being processed by SageMaker
_processing_jobs = set()


# -------------------------
# Model downloads
# -------------------------
def _download_file(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print("file already exists!")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".tmp"
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        os.replace(tmp, dest)
    except Exception:
        print(f'does {tmp} exist?')
        if os.path.exists(tmp):
            os.remove(tmp)


# Model downloads are disabled by default. Set ENABLE_MODEL_DOWNLOADS=1 to
# fetch required weights at startup if desired.
ENABLE_MODEL_DOWNLOADS = os.getenv("ENABLE_MODEL_DOWNLOADS") == "1"

def ensure_models() -> None:
    models = {
        "vit_l.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "birefnet-dis.onnx": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-DIS-epoch_590.onnx",
        "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "yolov8n-seg.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt",
    }
    for fname, url in models.items():
        print(f'Downloading {fname}')
        _download_file(url, os.path.join(MODELS_DIR, fname))


if ENABLE_MODEL_DOWNLOADS:
    ensure_models()
else:
    print("Model downloads disabled; skipping ensure_models()")


def notify_sagemaker_startup() -> None:
    """Send a simple startup notification to the SageMaker endpoint."""
    if not (SAGEMAKER_ENDPOINT and sm_client):
        return
    try:
        sm_client.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            ContentType="application/json",
            Body=json.dumps({"event": "server1_startup"}),
        )
        print(f"[sagemaker] notified {SAGEMAKER_ENDPOINT} of server1 startup")
    except Exception as e:  # pragma: no cover - best effort
        print(f"[sagemaker] startup notification failed: {e}")


#notify_sagemaker_startup()


# -------------------------
# SageMaker endpoint lifecycle helpers
# -------------------------
def start_gpu_instance() -> None:
    global GPU_ACTIVE, _last_work_time
    if GPU_ACTIVE or not (SAGEMAKER_ENDPOINT and sagemaker_client):
        return
    try:
        sagemaker_client.update_endpoint_weights_and_capacities(
            EndpointName=SAGEMAKER_ENDPOINT,
            DesiredWeightsAndCapacities=[
                {"VariantName": SAGEMAKER_VARIANT, "DesiredInstanceCount": 1}
            ],
        )
        GPU_ACTIVE = True
        _last_work_time = time.time()
    except Exception as e:
        print(f"Failed to scale endpoint: {e}")


def stop_gpu_instance() -> None:
    global GPU_ACTIVE
    if not GPU_ACTIVE or not (SAGEMAKER_ENDPOINT and sagemaker_client):
        return
    try:
        sagemaker_client.update_endpoint_weights_and_capacities(
            EndpointName=SAGEMAKER_ENDPOINT,
            DesiredWeightsAndCapacities=[
                {"VariantName": SAGEMAKER_VARIANT, "DesiredInstanceCount": 0}
            ],
        )
        GPU_ACTIVE = False
    except Exception as e:
        print(f"Failed to scale endpoint: {e}")


def has_unprocessed_files() -> bool:
    """Check S3 for any resized images not yet processed."""
    if not (s3_client and S3_BUCKET):
        return False
    resp = s3_client.list_objects_v2(Bucket=S3_BUCKET, Delimiter="/")
    users = [p["Prefix"].strip("/") for p in resp.get("CommonPrefixes", [])]
    for user in users:
        processed = set()
        proc_key = f"{user}/output/processed.json"
        try:
            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=proc_key)
            processed = set(json.loads(obj["Body"].read()))
        except Exception:
            pass
        resp2 = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=f"{user}/resized/")
        for item in resp2.get("Contents", []):
            fname = os.path.basename(item["Key"])
            if not fname.lower().endswith(".png"):
                continue
            stem, _ = os.path.splitext(fname)
            if stem not in processed:
                return True
    return False


def gpu_monitor_loop() -> None:
    global _last_work_time
    while True:
        pending = has_unprocessed_files()
        if USER_LOGGED_IN:
            if pending and not GPU_ACTIVE:
                start_gpu_instance()
            if pending:
                _last_work_time = time.time()
            elif GPU_ACTIVE and time.time() - _last_work_time > 300:
                stop_gpu_instance()
        else:
            if GPU_ACTIVE:
                stop_gpu_instance()
        time.sleep(30)


@app.before_request
def require_login():
    global USER_LOGGED_IN, _last_work_time
    if request.endpoint in {"login", "static"}:
        return
    if "user" not in session:
        return redirect(url_for("login"))
    set_user_dirs(session["user"])
    ensure_settings_defaults()
    USER_LOGGED_IN = True
    _last_work_time = time.time()


# -------------------------
# Helpers
# -------------------------
def normalize_to_png_bytes(pil_img: Image.Image, longest_side: int | None = None) -> bytes:
    """Optionally resize to longest_side and return an optimized PNG bytestring."""
    img = pil_img.convert("RGBA")
    if longest_side and max(img.size) > longest_side:
        img.thumbnail((longest_side, longest_side), Image.LANCZOS)
    buf = io.BytesIO()
    # optimize=True reduces file size without altering pixel data
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def detect_paper_crop(bgr: np.ndarray) -> np.ndarray | None:
    """Attempt to detect and perspective-crop a sheet of paper in the image."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 75, 200)
    edges = cv2.dilate(edges, None, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    best = None
    best_area = 0
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        area = cv2.contourArea(approx)
        if area < 0.2 * w * h:
            continue
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [approx], -1, 255, -1)
        mean_val = cv2.mean(gray, mask=mask)[0]
        if mean_val < 180:
            continue
        if area > best_area:
            best_area = area
            best = approx
    if best is None:
        return None
    pts = best.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]
    ordered = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    w1 = np.linalg.norm(bottom_right - bottom_left)
    w2 = np.linalg.norm(top_right - top_left)
    h1 = np.linalg.norm(top_right - bottom_right)
    h2 = np.linalg.norm(top_left - bottom_left)
    dst_w = int(max(w1, w2))
    dst_h = int(max(h1, h2))
    dst = np.array([[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(bgr, M, (dst_w, dst_h))


def load_crops_index(username: str) -> Dict[str, List[str]]:
    if not (s3_client and S3_BUCKET):
        return {}
    key = f"{username}/output/crops/index.json"
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        return json.loads(obj["Body"].read())
    except Exception:
        return {}

def save_crops_index(index: Dict[str, List[str]], username: str) -> None:
    if not (s3_client and S3_BUCKET):
        return
    key = f"{username}/output/crops/index.json"
    s3_client.put_object(Bucket=S3_BUCKET, Key=key, Body=json.dumps(index, indent=2))

def ensure_settings_defaults() -> dict:
    print("setting defaults")
    defaults = {
        "model_type": "vit_l",         # allow vit_b / vit_l / vit_h
        "points_per_side": 32,
        "pred_iou_thresh": 0.88,
        "stability_score_thresh": 0.95,
        "crop_n_layers": 1
    }
    if s3_client and S3_BUCKET:
        try:
            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=SETTINGS_KEY)
            data = json.loads(obj["Body"].read())
            defaults.update({k: data.get(k, v) for k, v in defaults.items()})
        except Exception:
            s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=SETTINGS_KEY,
                Body=json.dumps(defaults, indent=2),
            )
    return defaults

def make_rgba_crops(original_bgr: np.ndarray, mask_gray: np.ndarray) -> List[np.ndarray]:
    """Return RGBA crops for each disconnected region in ``mask_gray``.

    The mask is applied as an alpha channel on top of ``original_bgr``. Each
    connected component in the mask becomes its own crop. An empty mask results
    in an empty list.
    """

    if mask_gray.shape != original_bgr.shape[:2]:
        mask_gray = cv2.resize(
            mask_gray,
            (original_bgr.shape[1], original_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    mask_u8 = (mask_gray > 0).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )

    crops: List[np.ndarray] = []
    bgra = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2BGRA)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area == 0:
            continue
        component_mask = (labels == i).astype(np.uint8) * 255
        crop = bgra[y : y + h, x : x + w].copy()
        crop[:, :, 3] = component_mask[y : y + h, x : x + w]
        crops.append(crop)

    return crops


# -------------------------
# Server2 (SageMaker) invocation
# -------------------------
def _mark_processed(stem: str, username: str) -> None:
    if not (s3_client and S3_BUCKET):
        return
    key = f"{username}/output/processed.json"
    processed = set()
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        processed.update(json.loads(obj["Body"].read()))
    except Exception:
        pass
    processed.add(stem)
    s3_client.put_object(
        Bucket=S3_BUCKET, Key=key, Body=json.dumps(sorted(processed))
    )


def run_sagemaker_job(stem: str, username: str) -> None:
    job_key = (username, stem)
    if job_key in _processing_jobs:
        return
    _processing_jobs.add(job_key)
    if not (SAGEMAKER_ENDPOINT and S3_BUCKET and s3_client and sm_client):
        return
    print(f"[sagemaker] starting job for user {username} and file {stem}.png")
    input_key = f"{username}/resized/{stem}.png"
    output_prefix = f"{username}/output/masks/"
    try:
        s3_client.head_object(Bucket=S3_BUCKET, Key=input_key)
    except Exception as e:
        print(f"[sagemaker] resized image missing: {e}")
        return

    payload = {"s3": f"s3://{S3_BUCKET}/{input_key}", "output": f"s3://{S3_BUCKET}/{output_prefix}"}
    try:
        print(f"[sagemaker] invoking endpoint {SAGEMAKER_ENDPOINT} with payload {payload}")
        response = sm_client.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            ContentType="application/json",
            Body=json.dumps(payload),
        )
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status != 200:
            body = response.get("Body")
            body_text = body.read().decode() if body else ""
            print(f"[sagemaker] invoke returned {status}: {body_text}")
        else:
            print(f"[sagemaker] server2 acknowledged request with status {status}")
    except Exception as e:
        print(f"[sagemaker] invoke failed: {e}")
        return
    print(f"[sagemaker] job launched for {stem}")

# -------------------------
# Authentication
# -------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if USERS.get(username) == password:
            session["user"] = username
            set_user_dirs(username)
            ensure_settings_defaults()
            start_gpu_instance()
            return redirect(url_for("index"))
        error = "Invalid credentials"
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.pop("user", None)
    global USER_LOGGED_IN
    USER_LOGGED_IN = False
    stop_gpu_instance()
    return redirect(url_for("login"))

# -------------------------
# Upload & Settings
# -------------------------
@app.route("/", methods=["GET"])
def index():
    # Just render; the page fetches crops and settings via APIs
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    """
    - Accepts most image types (incl. HEIC/HEIF).
    - Uploads original PNG to S3 under <user>/input/<basename>.png
    - Uploads resized PNG (≤1024) to S3 under <user>/resized/<basename>.png
    """
    if "file" not in request.files:
        return "No file part", 400
    file = request.files["file"]
    if not file or file.filename == "":
        return "No selected file", 400

    filename = secure_filename(file.filename)
    stem, ext = os.path.splitext(filename)
    ext = ext.lower().lstrip(".")

    # Ensure user-specific directories exist even if the request bypassed the
    # normal login flow.
    username = session.get("user", "shared")
    set_user_dirs(username)

    if ext not in ALLOWED_EXT:
        return "Unsupported file type", 400

    # Read via Pillow (handles HEIC via pillow-heif)
    try:
        pil_img = Image.open(file.stream)
    except Exception as e:
        return f"Failed to open image: {e}", 400

    # Detect and crop a sheet of paper if present
    cv_img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    paper_crop = detect_paper_crop(cv_img)
    if paper_crop is not None:
        print("Paper detection: cropped to sheet")
        pil_img = Image.fromarray(cv2.cvtColor(paper_crop, cv2.COLOR_BGR2RGB))
    else:
        print("Paper detection: none found")

    # Upload original, resized, and thumbnail images directly to S3
    if s3_client and S3_BUCKET:
        try:
            input_bytes = normalize_to_png_bytes(pil_img, longest_side=None)
            s3_key = f"{username}/input/{stem}.png"
            s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=input_bytes)
            print(f"[upload] uploaded original to s3://{S3_BUCKET}/{s3_key}")
        except Exception as e:
            print(f"[upload] failed to upload original: {e}")

        try:
            resized_bytes = normalize_to_png_bytes(pil_img, longest_side=MAX_RESIZE)
            s3_resized_key = f"{username}/resized/{stem}.png"
            s3_client.put_object(Bucket=S3_BUCKET, Key=s3_resized_key, Body=resized_bytes)
            print(f"[upload] uploaded resized to s3://{S3_BUCKET}/{s3_resized_key}")
        except Exception as e:
            print(f"[upload] failed to upload resized: {e}")

        try:
            thumb_bytes = normalize_to_png_bytes(pil_img, longest_side=256)
            s3_thumb_key = f"{username}/output/thumbs/{stem}.png"
            s3_client.put_object(Bucket=S3_BUCKET, Key=s3_thumb_key, Body=thumb_bytes)
        except Exception as e:
            print(f"[upload] failed to upload thumb: {e}")

    if SAGEMAKER_ENDPOINT and S3_BUCKET:
        threading.Thread(
            target=run_sagemaker_job,
            args=(stem, session.get("user", "shared")),
            daemon=True,
        ).start()

    return jsonify({"status": "ok", "original": f"{stem}.png"})

# --- serve originals and thumbs via S3 presigned URLs ---
@app.route("/input/<path:filename>", methods=["GET"])
def serve_input(filename):
    username = session.get("user", "shared")
    key = f"{username}/input/{filename}"
    url = s3_client.generate_presigned_url(
        "get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=3600
    )
    return redirect(url)

@app.route("/thumbs/<path:filename>", methods=["GET"])
def serve_thumb(filename):
    username = session.get("user", "shared")
    key = f"{username}/output/thumbs/{filename}"
    url = s3_client.generate_presigned_url(
        "get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=3600
    )
    return redirect(url)

# Albums: originals (from S3) + their crops (from index)
@app.route("/list_originals", methods=["GET"])
def list_originals():
    username = session.get("user", "shared")
    index = load_crops_index(username)
    albums = []
    resp = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=f"{username}/input/")
    for item in resp.get("Contents", []):
        if not item["Key"].lower().endswith(".png"):
            continue
        f = os.path.basename(item["Key"])
        crop_files = index.get(f, [])
        crops = []
        for c in crop_files:
            crop_key = f"{username}/output/crops/{c}"
            area = 0
            try:
                head = s3_client.head_object(Bucket=S3_BUCKET, Key=crop_key)
                area = head.get("ContentLength", 0)
            except Exception:
                pass
            crop_url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": S3_BUCKET, "Key": crop_key},
                ExpiresIn=3600,
            )
            thumb_url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": S3_BUCKET, "Key": f"{username}/output/thumbs/{c}"},
                ExpiresIn=3600,
            )
            crops.append({"file": c, "url": crop_url, "thumb_url": thumb_url, "area": area})
        crops.sort(key=lambda x: x.get("area", 0), reverse=True)
        crops = crops[:MAX_CROPS_PER_IMAGE]
        orig_url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": f"{username}/input/{f}"},
            ExpiresIn=3600,
        )
        thumb_url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": f"{username}/output/thumbs/{f}"},
            ExpiresIn=3600,
        )
        albums.append({"original": f, "original_url": orig_url, "thumb_url": thumb_url, "crops": crops})
    return jsonify(albums)




@app.route("/save_settings", methods=["POST"])
def save_settings():
    """
    Saves SAM settings to shared config for Server2.
    Expect JSON: { model_type, points_per_side, pred_iou_thresh, stability_score_thresh, crop_n_layers }
    """
    data = request.get_json(force=True, silent=True) or {}
    current = ensure_settings_defaults()
    current.update({
        "model_type": data.get("model_type", current["model_type"]),
        "points_per_side": int(data.get("points_per_side", current["points_per_side"])),
        "pred_iou_thresh": float(data.get("pred_iou_thresh", current["pred_iou_thresh"])),
        "stability_score_thresh": float(data.get("stability_score_thresh", current["stability_score_thresh"])),
        "crop_n_layers": int(data.get("crop_n_layers", current["crop_n_layers"])),
    })
    if s3_client and S3_BUCKET:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=SETTINGS_KEY,
            Body=json.dumps(current, indent=2),
        )
    return jsonify({"status": "ok", "settings": current})

@app.route("/get_settings", methods=["GET"])
def get_settings():
    return jsonify(ensure_settings_defaults())

# -------------------------
# Serving crops & list API (with original association)
# -------------------------
@app.route("/crops/<path:filename>", methods=["GET"])
def serve_crop(filename):
    username = session.get("user", "shared")
    key = f"{username}/output/crops/{filename}"
    url = s3_client.generate_presigned_url(
        "get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=3600
    )
    return redirect(url)

@app.route("/list_crops", methods=["GET"])
def list_crops():
    """
    Returns JSON like:
    [
      { "file": "penguin_mask0.png", "url": "<presigned>", "original": "penguin.png" },
      ...
    ]
    """
    username = session.get("user", "shared")
    index = load_crops_index(username)
    items = []
    for original, crops in index.items():
        for c in crops:
            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": S3_BUCKET, "Key": f"{username}/output/crops/{c}"},
                ExpiresIn=3600,
            )
            items.append({"file": c, "url": url, "original": original})
    return jsonify(items)


def _delete_user_s3_data(username: str) -> None:
    """Delete all S3 objects belonging to ``username`` while preserving folders."""

    if not (s3_client and S3_BUCKET):
        return

    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=f"{username}/"):
        objs = []
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            # Keys ending with "/" represent the placeholder objects that mimic
            # folder structures in S3. Keep those so subsequent uploads do not
            # need to recreate the directory tree.
            if key.endswith("/"):
                continue
            objs.append({"Key": key})

        if objs:
            s3_client.delete_objects(Bucket=S3_BUCKET, Delete={"Objects": objs})

    # Some workflows cache processing state in ``output/processed.json``.  The
    # above bulk-deletion should remove it, but delete explicitly to avoid stale
    # entries reappearing if the list call missed it for any reason.
    proc_key = f"{username}/output/processed.json"
    try:  # best effort – failure here shouldn't block clearing other files
        s3_client.delete_object(Bucket=S3_BUCKET, Key=proc_key)
    except Exception:
        pass

    # Ensure the expected directory prefixes still exist after the purge.
    # ``set_user_dirs`` creates empty objects for each required prefix.
    set_user_dirs(username)


@app.route("/clear_all", methods=["POST"])
def clear_all():
    """Remove all processed images and clear the user's S3 storage."""
    username = session.get("user", "shared")
    _delete_user_s3_data(username)
    _processed_mask_files.clear()
    notify_album_update()
    return jsonify({"status": "cleared"})

# -------------------------
# Mask watcher → cropper (runs in background)
# -------------------------
def process_mask_file(mask_key: str, username: str):
    """
    Given a mask object key like user/output/masks/<stem>_maskN.png:
    - download original and mask from S3
    - create RGBA crops and thumbnails
    - upload results back to S3 and update index
    """
    fname = os.path.basename(mask_key)
    if "_mask" not in fname:
        return

    base = fname.split("_mask")[0]                # stem
    try:
        orig_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=f"{username}/input/{base}.png")
        mask_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=mask_key)
    except Exception:
        return

    try:
        orig_bytes = orig_obj["Body"].read()
        mask_bytes = mask_obj["Body"].read()
        orig_bgr = cv2.imdecode(np.frombuffer(orig_bytes, np.uint8), cv2.IMREAD_COLOR)
        mask_gray = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if orig_bgr is None or mask_gray is None:
            return

        crops = make_rgba_crops(orig_bgr, mask_gray)
        if not crops:
            return

        index = load_crops_index(username)
        crops_for_original = index.get(f"{base}.png", [])
        for idx, crop_rgba in enumerate(crops):
            out_name = fname[:-4] + f"_{idx}.png"
            _, buf = cv2.imencode(".png", crop_rgba)
            crop_key = f"{username}/output/crops/{out_name}"
            s3_client.put_object(Bucket=S3_BUCKET, Key=crop_key, Body=buf.tobytes())

            try:
                pil_crop = Image.fromarray(cv2.cvtColor(crop_rgba, cv2.COLOR_BGRA2RGBA))
                thumb_bytes = normalize_to_png_bytes(pil_crop, longest_side=256)
                thumb_key = f"{username}/output/thumbs/{out_name}"
                s3_client.put_object(Bucket=S3_BUCKET, Key=thumb_key, Body=thumb_bytes)
            except Exception:
                pass

            if out_name not in crops_for_original:
                crops_for_original.append(out_name)

        index[f"{base}.png"] = crops_for_original
        save_crops_index(index, username)
        _mark_processed(base, username)
        notify_album_update()
    finally:
        _processing_jobs.discard((username, base))

def resized_watcher_loop():
    while True:
        try:
            if not (s3_client and S3_BUCKET):
                time.sleep(0.5)
                continue
            resp = s3_client.list_objects_v2(Bucket=S3_BUCKET, Delimiter="/")
            users = [p["Prefix"].strip("/") for p in resp.get("CommonPrefixes", [])]
            for user in users:
                processed = set()
                proc_key = f"{user}/output/processed.json"
                try:
                    obj = s3_client.get_object(Bucket=S3_BUCKET, Key=proc_key)
                    processed = set(json.loads(obj["Body"].read()))
                except Exception:
                    pass
                resp2 = s3_client.list_objects_v2(
                    Bucket=S3_BUCKET, Prefix=f"{user}/resized/"
                )
                for item in resp2.get("Contents", []):
                    fname = os.path.basename(item["Key"])
                    if not fname.lower().endswith(".png"):
                        continue

                    stem, _ = os.path.splitext(fname)
                    #print(f'found {fname} and launching a thread with run_sagemaker_job using arguments {stem} and {user}')
                    key = (user, stem)
                    if stem in processed or key in _processing_jobs:
                        continue
                    threading.Thread(
                        target=run_sagemaker_job,
                        args=(stem, user),
                        daemon=True,
                    ).start()
        except Exception as e:
            print(f"[resized_watcher] error: {e}")
        time.sleep(0.5)

def mask_watcher_loop():
    while True:
        try:
            if not (s3_client and S3_BUCKET):
                time.sleep(0.5)
                continue
            resp = s3_client.list_objects_v2(Bucket=S3_BUCKET, Delimiter="/")
            users = [p["Prefix"].strip("/") for p in resp.get("CommonPrefixes", [])]
            for user in users:
                resp2 = s3_client.list_objects_v2(
                    Bucket=S3_BUCKET, Prefix=f"{user}/output/masks/"
                )
                for item in resp2.get("Contents", []):
                    fname = os.path.basename(item["Key"])
                    if not fname.lower().endswith(".png"):
                        continue
                    key = item["Key"]
                    if key in _processed_mask_files:
                        continue
                    process_mask_file(key, user)
                    _processed_mask_files.add(key)
        except Exception as e:
            print(f"[mask_watcher] error: {e}")
        time.sleep(0.5)

# Start background watcher before serving
threading.Thread(target=resized_watcher_loop, daemon=True).start()
threading.Thread(target=mask_watcher_loop, daemon=True).start()
threading.Thread(target=gpu_monitor_loop, daemon=True).start()

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    ensure_settings_defaults()
    app.run(host="0.0.0.0", port=8080, threaded=True)
