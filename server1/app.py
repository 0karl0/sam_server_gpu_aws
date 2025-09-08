import os
import json
import threading
import time
from typing import List, Dict
import shutil
import boto3
from botocore.exceptions import ClientError

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory,
    redirect,
    url_for,
    session,
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
# to a common network location (e.g., an S3 mount).
SHARED_DIR   = os.getenv("SHARED_DIR", "/mnt/s3")
INPUT_DIR    = os.path.join(SHARED_DIR, "input")              # originals (PNG-normalized)
RESIZED_DIR  = os.path.join(SHARED_DIR, "resized")            # ≤1024 for SAM
MASKS_DIR    = os.path.join(SHARED_DIR, "output", "masks")    # from Server2
CROPS_DIR    = os.path.join(SHARED_DIR, "output", "crops")    # RGBA crops
SMALLS_DIR   = os.path.join(SHARED_DIR, "output", "smalls")
POINTS_DIR   = os.path.join(SHARED_DIR, "output", "points")
PROCESSED_FILE = os.path.join(SHARED_DIR, "output", "processed.json")
CONFIG_DIR   = os.path.join(SHARED_DIR, "config")
SETTINGS_JSON = os.path.join(CONFIG_DIR, "settings.json")
CROPS_INDEX   = os.path.join(CROPS_DIR, "index.json")         # manifest linking crops to original
THUMBS_DIR   = os.path.join(SHARED_DIR, "output", "thumbs")   # thumbnails for UI
MODELS_DIR   = os.path.join(SHARED_DIR, "models")              # downloaded weights

MAX_RESIZE = 1024  # longest side for SAM
ALLOWED_EXT = {"png", "jpg", "jpeg", "webp", "bmp", "tiff", "heic", "heif"}

for d in [
    INPUT_DIR,
    RESIZED_DIR,
    MASKS_DIR,
    CROPS_DIR,
    SMALLS_DIR,
    THUMBS_DIR,
    CONFIG_DIR,
    POINTS_DIR,
    MODELS_DIR,
]:
    try:
     print(f'making directory {d}')
     os.makedirs(d, exist_ok=True)
    except:
     print(f'could not make {d}')
# Register HEIF opener for Pillow
pillow_heif.register_heif_opener()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "change-me")

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
        return get_secret_value_response["SecretString"]
    except ClientError as e:
        print(e)
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e
    else:
        secret_string = get_secret_value_response['SecretString']
        return json.loads(secret_string)["SecretString"]
USERS: Dict[str, str] = {}


user = get_single_secret_value("APP_USER1")

#user = secret["SecretString"]

pw = get_single_secret_value("APP_PASS1")

#pw = secret["SecretString"]

if user and pw:
        USERS[user] = pw




# AWS configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
GPU_ACTIVE = False
USER_LOGGED_IN = False
_last_work_time = time.time()

SAGEMAKER_ENDPOINT = get_single_secret_value("SAGEMAKER_ENDPOINT") #os.getenv("SAGEMAKER_ENDPOINT","sam-server2-endpoint")

SAGEMAKER_VARIANT = os.getenv("SAGEMAKER_VARIANT", "AllTraffic")

S3_BUCKET = get_single_secret_value("S3_BUCKET")


#S3_BUCKET = os.getenv("S3_BUCKET","sam-server-shared-1757294775")
s3_client = boto3.client("s3", region_name=AWS_REGION) if S3_BUCKET else None
sm_client = (
    boto3.client("sagemaker-runtime", region_name=AWS_REGION)
    if SAGEMAKER_ENDPOINT
    else None
)
sagemaker_client = (
    boto3.client("sagemaker", region_name=AWS_REGION)
    if SAGEMAKER_ENDPOINT
    else None
)

# Track which mask files have been processed into crops
_processed_mask_files = set()


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


ensure_models()


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
    files = [f for f in os.listdir(RESIZED_DIR) if f.lower().endswith(".png")]
    if not files:
        return False
    processed = set()
    if os.path.exists(PROCESSED_FILE):
        try:
            with open(PROCESSED_FILE, "r") as f:
                processed = set(json.load(f))
        except Exception:
            pass
    for f in files:
        if os.path.splitext(f)[0] not in processed:
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
    USER_LOGGED_IN = True
    _last_work_time = time.time()


# -------------------------
# Helpers
# -------------------------
def normalize_to_png_and_save(pil_img: Image.Image, out_path_png: str, longest_side: int | None = None) -> None:
    """Optionally resize to longest_side, then save as PNG (preserve alpha if present)."""
    img = pil_img.convert("RGBA")
    if longest_side and max(img.size) > longest_side:
        img.thumbnail((longest_side, longest_side), Image.LANCZOS)
    img.save(out_path_png, "PNG")


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


def load_crops_index() -> Dict[str, List[str]]:
    if os.path.exists(CROPS_INDEX):
        try:
            with open(CROPS_INDEX, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_crops_index(index: Dict[str, List[str]]) -> None:
    tmp_path = CROPS_INDEX + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(index, f, indent=2)
    os.replace(tmp_path, CROPS_INDEX)

def ensure_settings_defaults() -> dict:
    defaults = {
        "model_type": "vit_b",         # allow vit_b / vit_l / vit_h
        "points_per_side": 32,
        "pred_iou_thresh": 0.88,
        "stability_score_thresh": 0.95,
        "crop_n_layers": 1
    }
    if os.path.exists(SETTINGS_JSON):
        try:
            with open(SETTINGS_JSON, "r") as f:
                data = json.load(f)
            defaults.update({k: data.get(k, v) for k, v in defaults.items()})
        except Exception:
            pass
    else:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(SETTINGS_JSON, "w") as f:
            json.dump(defaults, f, indent=2)
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
def _mark_processed(stem: str) -> None:
    processed = set()
    if os.path.exists(PROCESSED_FILE):
        try:
            with open(PROCESSED_FILE, "r") as f:
                processed.update(json.load(f))
        except Exception:
            pass
    processed.add(stem)
    tmp = PROCESSED_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(sorted(processed), f)
    os.replace(tmp, PROCESSED_FILE)


def run_sagemaker_job(stem: str) -> None:
    if not (SAGEMAKER_ENDPOINT and S3_BUCKET and s3_client and sm_client):
        return
    resized_path = os.path.join(RESIZED_DIR, f"{stem}.png")
    if not os.path.exists(resized_path):
        return
    input_key = f"input/{stem}.png"
    output_key = f"output/{stem}_mask0.png"
    try:
        s3_client.upload_file(resized_path, S3_BUCKET, input_key)
    except Exception as e:
        print(f"[sagemaker] upload failed: {e}")
        return
    payload = {"s3": f"s3://{S3_BUCKET}/{input_key}", "output": f"s3://{S3_BUCKET}/{output_key}"}
    try:
        sm_client.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            ContentType="application/json",
            Body=json.dumps(payload),
        )
    except Exception as e:
        print(f"[sagemaker] invoke failed: {e}")
        return
    local_mask = os.path.join(MASKS_DIR, f"{stem}_mask0.png")
    try:
        s3_client.download_file(S3_BUCKET, output_key, local_mask)
    except Exception as e:
        print(f"[sagemaker] download failed: {e}")
        return
    process_mask_file(local_mask)
    _processed_mask_files.add(local_mask)
    _mark_processed(stem)

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
    - Normalizes original to PNG in /input/<basename>.png
    - Produces resized PNG (≤1024) in /resized/<basename>.png
    """
    if "file" not in request.files:
        return "No file part", 400
    file = request.files["file"]
    if not file or file.filename == "":
        return "No selected file", 400

    filename = secure_filename(file.filename)
    stem, ext = os.path.splitext(filename)
    ext = ext.lower().lstrip(".")

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

    # Normalize original to PNG in /input (use cropped if available)
    input_png = os.path.join(INPUT_DIR, f"{stem}.png")
    normalize_to_png_and_save(pil_img, input_png, longest_side=None)  # keep original resolution

    # Also save a resized (≤1024) copy for SAM in /resized (same basename)
    resized_png = os.path.join(RESIZED_DIR, f"{stem}.png")
    normalize_to_png_and_save(pil_img, resized_png, longest_side=MAX_RESIZE)

    # Save a smaller thumbnail for quick listing in /thumbs
    thumb_png = os.path.join(THUMBS_DIR, f"{stem}.png")
    normalize_to_png_and_save(pil_img, thumb_png, longest_side=256)

    if SAGEMAKER_ENDPOINT and S3_BUCKET:
        threading.Thread(target=run_sagemaker_job, args=(stem,), daemon=True).start()

    return jsonify({"status": "ok", "original": f"{stem}.png"})

# --- serve originals (normalized PNGs) ---
# Serve originals directly from /mnt/shared/input
@app.route("/input/<path:filename>", methods=["GET"])
def serve_input(filename):
    return send_from_directory(INPUT_DIR, filename)

@app.route("/thumbs/<path:filename>", methods=["GET"])
def serve_thumb(filename):
    return send_from_directory(THUMBS_DIR, filename)

# Albums: originals (from /input) + their crops (from crops index)
@app.route("/list_originals", methods=["GET"])
def list_originals():
    """
    Returns:
    [
      {
        "original": "penguin.png",
        "original_url": "/input/penguin.png",
        "thumb_url": "/thumbs/penguin.png",
        "crops": [
          {"file": "penguin_mask0.png", "url": "/crops/penguin_mask0.png", "thumb_url": "/thumbs/penguin_mask0.png"},
          ...
        ]
      },
      ...
    ]
    """
    index = load_crops_index()   # { "penguin.png": ["penguin_mask0.png", ...], ... }
    albums = []

    # Include every normalized original (PNG) in INPUT_DIR
    for f in sorted(os.listdir(INPUT_DIR)):
        if not f.lower().endswith(".png"):
            continue
        crop_files = index.get(f, [])
        crops = []
        for c in crop_files:
            crop_path = os.path.join(CROPS_DIR, c)
            area = 0
            try:
                img = cv2.imread(crop_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    h, w = img.shape[:2]
                    area = int(h * w)
            except Exception:
                pass
            thumb_url = f"/thumbs/{c}" if os.path.exists(os.path.join(THUMBS_DIR, c)) else f"/crops/{c}"
            crops.append({"file": c, "url": f"/crops/{c}", "thumb_url": thumb_url, "area": area})

        points_info = None
        base_name = os.path.splitext(f)[0]
        pts_path = os.path.join(POINTS_DIR, f"{base_name}.json")
        if os.path.exists(pts_path):
            try:
                with open(pts_path, "r") as pf:
                    pts = json.load(pf)
                if isinstance(pts, dict) and "points" in pts:
                    points_info = pts
            except Exception:
                points_info = None

        orig_thumb = f"/thumbs/{f}" if os.path.exists(os.path.join(THUMBS_DIR, f)) else f"/input/{f}"
        albums.append({
            "original": f,
            "original_url": f"/input/{f}",
            "thumb_url": orig_thumb,
            "crops": crops,
            "yolo": points_info
        })

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
    with open(SETTINGS_JSON, "w") as f:
        json.dump(current, f, indent=2)
    return jsonify({"status": "ok", "settings": current})

@app.route("/get_settings", methods=["GET"])
def get_settings():
    return jsonify(ensure_settings_defaults())

# -------------------------
# Serving crops & list API (with original association)
# -------------------------
@app.route("/crops/<path:filename>", methods=["GET"])
def serve_crop(filename):
    return send_from_directory(CROPS_DIR, filename)

@app.route("/list_crops", methods=["GET"])
def list_crops():
    """
    Returns JSON like:
    [
      { "file": "penguin_mask0.png", "url": "/crops/penguin_mask0.png", "original": "penguin.png" },
      ...
    ]
    """
    index = load_crops_index()
    items = []
    for original, crops in index.items():
        for c in crops:
            items.append({
                "file": c,
                "url": f"/crops/{c}",
                "original": original
            })
    return jsonify(items)


@app.route("/clear_all", methods=["POST"])
def clear_all():
    """Remove all processed images and trackers."""
    dirs = [INPUT_DIR, RESIZED_DIR, MASKS_DIR, CROPS_DIR, SMALLS_DIR, THUMBS_DIR]
    for d in dirs:
        for name in os.listdir(d):
            path = os.path.join(d, name)
            try:
                if os.path.isfile(path) or os.path.islink(path):
                    os.remove(path)
                else:
                    shutil.rmtree(path)
            except Exception:
                pass
    for f in [CROPS_INDEX, PROCESSED_FILE]:
        try:
            if os.path.exists(f):
                os.remove(f)
        except Exception:
            pass
    _processed_mask_files.clear()
    return jsonify({"status": "cleared"})

# -------------------------
# Mask watcher → cropper (runs in background)
# -------------------------
def process_mask_file(mask_path: str):
    """
    Given a mask PNG path like .../masks/<stem>_maskN.png
    - find /input/<stem>.png
    - create RGBA crop
    - save to /crops/<stem>_maskN.png
    - update index.json association
    """
    fname = os.path.basename(mask_path)
    if "_mask" not in fname:
        return

    base = fname.split("_mask")[0]                # stem
    original_png = os.path.join(INPUT_DIR, f"{base}.png")
    if not os.path.exists(original_png):
        # original not found; skip
        return

    # Load original + mask
    orig_bgr = cv2.imread(original_png, cv2.IMREAD_COLOR)
    mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if orig_bgr is None or mask_gray is None:
        return

    crops = make_rgba_crops(orig_bgr, mask_gray)
    if not crops:
        return

    index = load_crops_index()
    crops_for_original = index.get(f"{base}.png", [])
    for idx, crop_rgba in enumerate(crops):
        out_name = fname[:-4] + f"_{idx}.png"
        out_path = os.path.join(CROPS_DIR, out_name)
        cv2.imwrite(out_path, crop_rgba)  # PNG with alpha

        # Save thumbnail for UI
        try:
            pil_crop = Image.fromarray(cv2.cvtColor(crop_rgba, cv2.COLOR_BGRA2RGBA))
            thumb_path = os.path.join(THUMBS_DIR, out_name)
            normalize_to_png_and_save(pil_crop, thumb_path, longest_side=256)
        except Exception:
            pass

        if out_name not in crops_for_original:
            crops_for_original.append(out_name)

    index[f"{base}.png"] = crops_for_original
    save_crops_index(index)

def mask_watcher_loop():
    while True:
        try:
            for fname in os.listdir(MASKS_DIR):
                if not fname.lower().endswith(".png"):
                    continue
                fpath = os.path.join(MASKS_DIR, fname)
                if fpath in _processed_mask_files:
                    continue
                process_mask_file(fpath)
                _processed_mask_files.add(fpath)
        except Exception as e:
            # Keep the watcher alive even if one file causes an error
            print(f"[mask_watcher] error: {e}")
        time.sleep(0.5)  # light polling

# Start background watcher before serving
threading.Thread(target=mask_watcher_loop, daemon=True).start()
threading.Thread(target=gpu_monitor_loop, daemon=True).start()

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    ensure_settings_defaults()
    app.run(host="0.0.0.0", port=8080, threaded=True)
