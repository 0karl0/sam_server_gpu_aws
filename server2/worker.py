import os
import time
import json
import cv2
import gc
import numpy as np
import runpod
from PIL import Image
from rembg import remove, new_session
from segment_anything import (
    SamAutomaticMaskGenerator,
    SamPredictor,
    sam_model_registry,
)

try:  # YOLO is optional
    from ultralytics import YOLO  # type: ignore
    print("ultralytics loaded")
    _YOLO_AVAILABLE = True
except Exception:  # pragma: no cover - ultralytics may not be installed
    YOLO = None  # type: ignore
    _YOLO_AVAILABLE = False
    print("couldn't find ultralytics")

try:
    import torch  # type: ignore
    print("importing torch")
    _TORCH_AVAILABLE = True
    _REMBG_PROVIDERS = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ] if torch.cuda.is_available() else ["CPUExecutionProvider"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:  # pragma: no cover - torch may not be installed
    print("couldn't import torch")
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False
    _REMBG_PROVIDERS = ["CPUExecutionProvider"]
    DEVICE = "cpu"

# -------------------------
# Config / directories
# -------------------------
# Base directory for shared storage, overridable via SHARED_DIR env var to
# point at a network-mounted location (e.g., S3 or EFS) accessible from both
# Server1 on EC2 and this RunPod worker.
SHARED_DIR = os.getenv("SHARED_DIR", "/mnt/shared")
RESIZED_DIR = os.path.join(SHARED_DIR, "resized")
MASKS_DIR = os.path.join(SHARED_DIR, "output", "masks")
SMALLS_DIR = os.path.join(SHARED_DIR, "output", "smalls")
CROPS_DIR = os.path.join(SHARED_DIR, "output", "crops")
CONFIG_FILE = os.path.join(SHARED_DIR, "config", "settings.json")
MODEL_PATH = os.path.join(SHARED_DIR, "models", "vit_l.pth")
PROCESSED_FILE = os.path.join(SHARED_DIR, "output", "processed.json")
YOLO_MODELS_DIR = os.path.join(SHARED_DIR, "models")
POINTS_DIR = os.path.join(SHARED_DIR, "output", "points")
BOXES_DIR = os.path.join(SHARED_DIR, "output", "boxes")

AREA_THRESH = 1000  # pixel area below which masks are treated as "smalls"

# Load BirefNet session from the shared models directory.
#
# ``rembg`` looks for downloaded model weights inside the directory pointed to
# the ``U2NET_HOME`` environment variable.  If the file already exists, it will
# be used directly and no network call is made.  By setting ``U2NET_HOME`` to
# our shared models directory, we ensure the pre-downloaded
# ``birefnet-dis.onnx`` file is picked up automatically.
os.environ.setdefault("U2NET_HOME", os.path.join(SHARED_DIR, "models"))
_REMBG_SESSION = new_session("birefnet-dis", providers=_REMBG_PROVIDERS)
print(f"[Worker] rembg providers: {_REMBG_SESSION.get_providers()}")


os.makedirs(MASKS_DIR, exist_ok=True)
os.makedirs(SMALLS_DIR, exist_ok=True)
os.makedirs(CROPS_DIR, exist_ok=True)
os.makedirs(POINTS_DIR, exist_ok=True)
os.makedirs(BOXES_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(YOLO_MODELS_DIR, exist_ok=True)


def _refine_mask_with_rembg(image_bgr: np.ndarray) -> np.ndarray:
    pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    print("[Decision] running rembg remove")
    result = remove(pil_img, session=_REMBG_SESSION)
    alpha = np.array(result)[..., 3]
    print("[Decision] rembg remove complete")
    return (alpha > 0).astype(np.uint8)


def _refine_mask_with_birefnet(image_bgr: np.ndarray) -> np.ndarray:
    """Refine mask using the BirefNet session.

    This simply delegates to rembg with the preloaded BirefNet model. A
    separate helper makes it easy to catch errors and fall back to the generic
    rembg model if needed.
    """
    print("birefnet")
    return _refine_mask_with_rembg(image_bgr)


def _is_line_drawing(image_bgr: np.ndarray) -> bool:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = float(np.count_nonzero(edges)) / edges.size
    color_std = float(image_bgr.std())
    result = edge_ratio > 0.05 and color_std < 25.0
    print(
        f"[Decision] _is_line_drawing: edge_ratio={edge_ratio:.4f}, color_std={color_std:.2f} -> {result}"
    )
    return result


def _has_long_lines(image_bgr: np.ndarray) -> bool:
    """Return True if prominent straight lines are detected in the image."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=80, minLineLength=60, maxLineGap=10
    )
    count = 0 if lines is None else len(lines)
    result = lines is not None and count > 0
    print(f"[Decision] _has_long_lines: count={count} -> {result}")
    return result


def _is_mostly_one_color(image_bgr: np.ndarray, mask: np.ndarray, std_thresh: float = 5.0) -> bool:
    """Return True if the region defined by mask has little color variation."""
    if mask.shape != image_bgr.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (image_bgr.shape[1], image_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    masked_pixels = image_bgr[mask.astype(bool)]
    if masked_pixels.size == 0:
        return False
    std = float(masked_pixels.std())
    result = std < std_thresh
    print(
        f"[Decision] _is_mostly_one_color: std={std:.2f}, thresh={std_thresh} -> {result}"
    )
    return result


def _crop_with_mask(image_bgr: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    """Return RGBA crops for each disconnected region in ``mask``.

    The incoming ``mask`` is expected to be a boolean or ``0/1`` array where
    non-zero values represent foreground.  We find connected components in the
    mask and return a list of cropped RGBA images, one per component.  Regions
    with no foreground pixels result in an empty list.
    """

    mask_u8 = (mask > 0).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )

    crops: list[np.ndarray] = []
    if num_labels <= 1:
        return crops

    bgra = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area == 0:
            continue
        component_mask = (labels == i).astype(np.uint8) * 255
        crop = bgra[y : y + h, x : x + w].copy()
        crop[:, :, 3] = component_mask[y : y + h, x : x + w]
        crops.append(crop)

    return crops


def _get_yolo_points(image_path: str) -> list[tuple[float, float, int]]:
    """Run all YOLO models and return midpoints labeled for SAM.

    Each returned tuple is ``(x, y, label)`` where ``label`` is ``1`` for
    regular objects (positive point) and ``0`` for detected humans (negative
    point). The center of any ``person``/``human`` box becomes a negative
    selector so SAM can avoid that region.
    """

    points: list[tuple[float, float, int]] = []
    if not _YOLO_AVAILABLE:
        print("[Worker] YOLO models not available, skipping YOLO point generation")
        return points
    if not os.path.isdir(YOLO_MODELS_DIR):
        print(
            f"[Worker] YOLO models directory '{YOLO_MODELS_DIR}' not found, skipping"
        )
        return points

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    img = cv2.imread(image_path)
    combined = img.copy() if img is not None else None

    print(f"[Worker] Running YOLO models on {image_path}")
    for fname in os.listdir(YOLO_MODELS_DIR):
        if not fname.lower().endswith((".pt", ".onnx")):
            continue
        # Skip any non-YOLO models such as the BirefNet weights which share
        # the models directory but are incompatible with the YOLO API.
        if "birefnet" in fname.lower():
            continue
        model_path = os.path.join(YOLO_MODELS_DIR, fname)
        print(f"[Worker] Running YOLO model {fname}")
        try:
            model = YOLO(model_path)
            yolo_device = 0 if DEVICE == "cuda" else "cpu"
            results = model(image_path, device=yolo_device)
            model_img = img.copy() if img is not None else None
            for r in results:
                names = getattr(r, "names", {})
                for box in getattr(r, "boxes", []):
                    cls = int(box.cls[0])
                    label = names.get(cls, "").lower()
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    print(
                        f"[Worker] {fname} detected {label} at ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
                    )
                    if label in {"person", "human"}:
                        points.append((cx, cy, 0))
                    else:
                        points.append((cx, cy, 1))
                    if model_img is not None:
                        cv2.rectangle(
                            model_img,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 255, 0),
                            2,
                        )
                        cv2.circle(model_img, (int(cx), int(cy)), 3, (0, 0, 255), -1)
                    if combined is not None:
                        cv2.rectangle(
                            combined,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 255, 0),
                            2,
                        )
                        cv2.circle(combined, (int(cx), int(cy)), 3, (0, 0, 255), -1)
            if model_img is not None:
                out_file = os.path.join(
                    BOXES_DIR, f"{base_name}-{os.path.splitext(fname)[0]}.png"
                )
                cv2.imwrite(out_file, model_img)
        except Exception as e:  # pragma: no cover - inference may fail
            print(f"[Worker] YOLO model {fname} failed: {e}")

    if combined is not None:
        try:
            out_file = os.path.join(BOXES_DIR, f"{base_name}-combined.png")
            cv2.imwrite(out_file, combined)
        except Exception as e:  # pragma: no cover - best effort only
            print(f"[Worker] Failed to save combined boxes for {base_name}: {e}")

    return points


def _save_yolo_points(
    points: list[tuple[float, float, int]],
    base_name: str,
    width: int,
    height: int,
) -> None:
    """Persist YOLO midpoint data for later display on thumbnails.

    ``points`` is a list of ``(x, y, label)`` tuples. Labels are stored so the
    frontend can distinguish positive and negative selectors if desired.
    """

    try:
        data = {"width": width, "height": height, "points": points}
        out_path = os.path.join(POINTS_DIR, f"{base_name}.json")
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, out_path)
    except Exception as e:  # pragma: no cover - best effort only
        print(f"[Worker] Failed to save YOLO points for {base_name}: {e}")


def load_processed_set():
    """Build a set of base filenames that have already been processed."""
    processed = set()
    # Load from persisted json if present
    if os.path.exists(PROCESSED_FILE):
        try:
            with open(PROCESSED_FILE, "r") as f:
                processed.update(json.load(f))
        except Exception:
            pass
    # Also include any masks that already exist on disk
    for fname in os.listdir(MASKS_DIR):
        if "_mask" in fname:
            base = fname.split("_mask")[0]
            processed.add(base)
    return processed


def save_processed_set(processed_set):
    """Persist processed base filenames to disk atomically."""
    tmp = PROCESSED_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(sorted(processed_set), f)
    os.replace(tmp, PROCESSED_FILE)

# -------------------------
# Load SAM model
# -------------------------
sam = sam_model_registry["vit_l"](checkpoint=MODEL_PATH)
sam.to(DEVICE)

# -------------------------
# Helper functions
# -------------------------
def load_settings():
    """Load SAM settings from Server1 JSON file."""
    default = {
        "points_per_side": 32,
        "pred_iou_thresh": 0.88,
        "stability_score_thresh": 0.95,
        "crop_n_layers": 1,
        "model_type": "vit_l"
    }
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            settings = json.load(f)
        default.update(settings)
    return default

def generate_masks(image_path, settings, points=None):
    """Generate masks for a single image.

    ``points`` is an optional list of ``(x, y, label)`` tuples where ``label``
    is ``1`` for positive selectors and ``0`` for negative selectors. Each
    positive point is evaluated with all negative points supplied to SAM so that
    detections such as humans can be excluded from the resulting masks.
    """
    print("using sam")
    image = cv2.imread(image_path)
    if image is None:
        return [], None

    masks = []
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=settings["points_per_side"],
        pred_iou_thresh=settings["pred_iou_thresh"],
        stability_score_thresh=settings["stability_score_thresh"],
        crop_n_layers=settings["crop_n_layers"]
       # min_mask_region_area=1000
    )

    masks.extend(mask_generator.generate(image))

    if points:
        predictor = SamPredictor(sam)
        predictor.set_image(image)

        pos = [(x, y) for x, y, lbl in points if lbl == 1]
        neg = [(x, y) for x, y, lbl in points if lbl == 0]
        neg_arr = np.array(neg) if neg else np.empty((0, 2))
        neg_labels = np.zeros(len(neg), dtype=np.int32)

        for x, y in pos:
            coords = np.vstack(([[x, y]], neg_arr))
            labels = np.concatenate(([1], neg_labels))
            try:
                mask, _, _ = predictor.predict(
                    point_coords=coords,
                    point_labels=labels,
                    multimask_output=False,
                )
                masks.append({"segmentation": mask[0]})
            except Exception as e:  # pragma: no cover - prediction may fail
                print(f"[Worker] SAM point prediction failed: {e}")

    return masks, image

def save_masks(masks, image, base_name):
    """Save each mask individually without merging.

    The masks generated by SAM are resized to the original image size and
    written out as separate PNG files. Smaller components are stored
    separately from larger ones based on ``AREA_THRESH``.
    """

    h, w = image.shape[:2]

    big_idx = 0
    small_idx = 0
    for m in masks:
        seg = m["segmentation"].astype(np.uint8)
        if seg.shape != (h, w):
            seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)

        area = int(seg.sum())
        out = seg * 255
        if area < AREA_THRESH:
            out_path = os.path.join(SMALLS_DIR, f"{base_name}_small{small_idx}.png")
            small_idx += 1
        else:
            out_path = os.path.join(MASKS_DIR, f"{base_name}_mask{big_idx}.png")
            big_idx += 1
        cv2.imwrite(out_path, out)

def process_new_images() -> int:
    """Process any unprocessed images found in the resized directory.

    Returns the number of images that were processed.
    """

    processed = load_processed_set()
    settings = load_settings()
    files = [f for f in os.listdir(RESIZED_DIR) if f.endswith((".png", ".jpg", ".jpeg"))]
    if not files:
        print("[Worker] No pages found")
        return 0
    print(f"[Worker] Found {len(files)} page(s): {files}")
    count = 0
    for f in files:
        base = os.path.splitext(f)[0]
        if base in processed:
            continue
        file_path = os.path.join(RESIZED_DIR, f)
        start = time.process_time()
        print(f"[Worker] Processing {f} ...")
        try:
            img = cv2.imread(file_path)
            if img is None:
                continue
            simple = _is_line_drawing(img) or _has_long_lines(img)
            yolo_points = [] if simple else _get_yolo_points(file_path)
            if simple or not yolo_points:
                print("[Worker] Using BirefNet for segmentation")
                mask = _refine_mask_with_birefnet(img)
                crops = _crop_with_mask(img, mask)
                mask_file = os.path.join(MASKS_DIR, f"{base}_mask0.png")
                cv2.imwrite(mask_file, mask.astype(np.uint8) * 255)
                for idx, crop in enumerate(crops):
                    crop_file = os.path.join(CROPS_DIR, f"{base}_mask0_{idx}.png")
                    cv2.imwrite(crop_file, crop)
            else:
                print("[Worker] Using SAM for segmentation")
                masks, img = generate_masks(file_path, settings, yolo_points)
                if img is not None and yolo_points:
                    h, w = img.shape[:2]
                    _save_yolo_points(yolo_points, base, w, h)

                if masks:
                    largest = max(
                        masks, key=lambda m: int(np.count_nonzero(m["segmentation"]))
                    )
                    if _is_mostly_one_color(img, largest["segmentation"]):
                        try:
                            print("refining with birefnet")
                            largest["segmentation"] = _refine_mask_with_birefnet(img).astype(bool)
                        except Exception:
                            print("refining with rembg")
                            largest["segmentation"] = _refine_mask_with_rembg(img).astype(bool)
                    h, w = img.shape[:2]
                    total_pixels = h * w
                    center_y, center_x = h // 2, w // 2
                    for m in list(masks):
                        seg = m["segmentation"]
                        seg_resized = seg
                        if seg.shape != (h, w):
                            seg_resized = cv2.resize(
                                seg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                            ).astype(bool)
                        area = np.count_nonzero(seg_resized)
                        if area > 0.9 * total_pixels:
                            if seg_resized[center_y, center_x]:
                                masks.remove(m)
                                continue
                            inverse = m.copy()
                            inverse["segmentation"] = np.logical_not(seg)
                            masks.append(inverse)
                save_masks(masks, img, base)
            processed.add(base)
            count += 1
            gc.collect()
            end = time.process_time()
            total = end - start
            print(f"elapsed time: {total:.6f} seconds")
        except Exception as e:
            print(f"[Worker] Error processing {f}: {e}")
    save_processed_set(processed)
    return count


def handler(job):  # type: ignore
    """RunPod serverless handler."""
    try:
        _ = job.get("input", {})
    except Exception as e:
        return {"status": "error", "message": f"Invalid input: {e}"}
    try:
        processed = process_new_images()
        return {"status": "success", "processed": processed}
    except Exception as e:  # pragma: no cover - best effort
        return {"status": "error", "message": str(e)}


runpod.serverless.start({"handler": handler})
