import os, time, threading, requests, json, cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

# -------------------
# Config
# -------------------
SHARED_DIR = "/mnt/shared"
INPUT_DIR = os.path.join(SHARED_DIR, "input")
RESIZED_DIR = os.path.join(SHARED_DIR, "resized")
MASKS_DIR = os.path.join(SHARED_DIR, "output", "masks")
CROPS_DIR = os.path.join(SHARED_DIR, "output", "crops")
CONFIG_FILE = os.path.join(SHARED_DIR, "config", "settings.json")
CROPS_INDEX = os.path.join(CROPS_DIR, "index.json")

for d in [INPUT_DIR, RESIZED_DIR, MASKS_DIR, CROPS_DIR, os.path.dirname(CONFIG_FILE)]:
    os.makedirs(d, exist_ok=True)

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY","RUNPOD_API_KEY")
GPU_POD_ID = os.getenv("GPU_POD_ID","GPU_POD_ID")

app = Flask(__name__)
last_activity = time.time()
GPU_ACTIVE = False


# -------------------
# GPU lifecycle
# -------------------
def runpod_request(query, variables):
    url = "https://api.runpod.io/graphql"
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
        json={"query": query, "variables": variables}
    )
    return r.json()

def start_gpu_pod():
    global GPU_ACTIVE
    if not GPU_ACTIVE:
        query = """
          mutation StartPod($podId: ID!) {
            podStart(input: { podId: $podId }) {
              id
              desiredStatus
            }
          }
        """
        runpod_request(query, {"podId": GPU_POD_ID})
        GPU_ACTIVE = True

def stop_gpu_pod():
    global GPU_ACTIVE
    query = """
      mutation StopPod($podId: ID!) {
        podStop(input: { podId: $podId }) {
          id
          desiredStatus
        }
      }
    """
    runpod_request(query, {"podId": GPU_POD_ID})
    GPU_ACTIVE = False


def activity_monitor():
    global last_activity
    while True:
        if GPU_ACTIVE and (time.time() - last_activity > 300):  # 5 min idle
            stop_gpu_pod()
        time.sleep(60)

threading.Thread(target=activity_monitor, daemon=True).start()


@app.before_request
def track_activity():
    global last_activity
    last_activity = time.time()


# -------------------
# Mask postprocessing
# -------------------
def crop_to_bbox(original, mask):
    """Scale mask to original size, apply, crop bounding box."""
    mask_resized = cv2.resize(mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
    masked = cv2.bitwise_and(original, original, mask=mask_resized)
    coords = cv2.findNonZero(mask_resized)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    return masked[y:y+h, x:x+w]


def process_masks(base_name):
    """Process all masks for one file."""
    original_path = os.path.join(INPUT_DIR, base_name)
    original = cv2.imread(original_path)
    if original is None:
        return []

    results = []
    idx = 0
    while True:
        mask_file = os.path.join(MASKS_DIR, f"{os.path.splitext(base_name)[0]}_mask{idx}.png")
        if not os.path.exists(mask_file):
            break
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        crop = crop_to_bbox(original, mask)
        if crop is not None and crop.size > 0:
            out_file = f"{os.path.splitext(base_name)[0]}_crop{idx}.png"
            out_path = os.path.join(CROPS_DIR, out_file)
            cv2.imwrite(out_path, crop)
            results.append(out_file)
        idx += 1

    # Update crops index
    if results:
        if os.path.exists(CROPS_INDEX):
            with open(CROPS_INDEX) as f:
                index = json.load(f)
        else:
            index = {}
        index[base_name] = results
        with open(CROPS_INDEX, "w") as f:
            json.dump(index, f, indent=2)

    return results


def mask_watcher():
    """Continuously look for new masks and crop them."""
    seen = set()
    while True:
        for f in os.listdir(MASKS_DIR):
            if f.endswith(".png") and f not in seen:
                seen.add(f)
                base_name = f.split("_mask")[0] + os.path.splitext(f)[1]
                process_masks(base_name)
        time.sleep(2)


threading.Thread(target=mask_watcher, daemon=True).start()


# -------------------
# Routes
# -------------------
@app.route("/")
def index():
    start_gpu_pod()
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    filename = secure_filename(file.filename)
    path = os.path.join(INPUT_DIR, filename)
    file.save(path)

    # Resize to max 1024px
    img = Image.open(path)
    img.thumbnail((1024, 1024))
    resized_path = os.path.join(RESIZED_DIR, filename)
    img.save(resized_path)

    return jsonify({"status": "uploaded", "file": filename})


@app.route("/settings", methods=["POST"])
def update_settings():
    data = request.json
    settings = {
        "points_per_side": int(data.get("points_per_side", 32)),
        "pred_iou_thresh": float(data.get("iou_threshold", 0.88)),
        "stability_score_thresh": float(data.get("stability_score_threshold", 0.95)),
        "crop_n_layers": int(data.get("crop_n_layers", 1)),
        "model_type": "vit_h"
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(settings, f, indent=2)
    return jsonify({"status": "ok", "settings": settings})


@app.route("/crops")
def list_crops():
    if not os.path.exists(CROPS_INDEX):
        return jsonify({})
    with open(CROPS_INDEX) as f:
        return jsonify(json.load(f))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
