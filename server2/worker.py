from __future__ import annotations

import os
import time
import json
import cv2
import gc
import logging
import boto3
import numpy as np
from typing import Dict
from botocore.exceptions import ClientError
from PIL import Image
from rembg import remove, new_session
from segment_anything import (
    SamAutomaticMaskGenerator,
    SamPredictor,
    sam_model_registry,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:  # YOLO is optional
    from ultralytics import YOLO  # type: ignore
    logger.info("ultralytics loaded")
    _YOLO_AVAILABLE = True
except Exception:  # pragma: no cover - ultralytics may not be installed
    YOLO = None  # type: ignore
    _YOLO_AVAILABLE = False
    logger.warning("couldn't find ultralytics")

try:
    import torch  # type: ignore
    logger.info("importing torch")
    _TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:  # pragma: no cover - torch may not be installed
    logger.warning("couldn't import torch")
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False
    DEVICE = "cpu"

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
s3_client = boto3.client("s3", region_name=AWS_REGION)
S3_BUCKET = os.getenv("S3_BUCKET")


def get_single_secret_value(secret_name: str) -> dict:
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=AWS_REGION)
    return client.get_secret_value(SecretId=secret_name)


if not S3_BUCKET:
    try:
        secret = get_single_secret_value("s3bucket")
        secret_str = secret["SecretString"]
        try:
            secret_dict = json.loads(secret_str)
            S3_BUCKET = secret_dict.get("S3_BUCKET", secret_str)
            logger.info("found %s", S3_BUCKET)
        except json.JSONDecodeError:
            S3_BUCKET = secret_str
        if S3_BUCKET.startswith("arn:aws:s3:::"):
            S3_BUCKET = S3_BUCKET.split(":::", 1)[1]
    except ClientError as e:  # pragma: no cover - network/permission issues
        logger.error("[s3] failed to retrieve bucket secret: %s", e)
        S3_BUCKET = None
logger.info("found %s", S3_BUCKET)

# -------------------------
# Config / directories
# -------------------------
# Base directory for temporary processing. Files are downloaded from S3 on
# demand via boto3 instead of relying on a mounted S3 filesystem.
SHARED_DIR = os.getenv("SHARED_DIR", "/tmp/shared")
os.makedirs(SHARED_DIR, exist_ok=True)

# All models are shared across users and live in a common directory.
MODELS_DIR = os.path.join(SHARED_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, "vit_l.pth")
YOLO_MODELS_DIR = MODELS_DIR

## WHERE is the path for bire-dis?
def get_user_dirs(username: str) -> Dict[str, str]:
    """Return the set of directories used for a specific user.

    The returned dictionary contains paths for ``resized``, ``masks``,
    ``smalls``, ``crops``, ``config`` (settings.json), ``processed``
    (processed.json), ``points`` and ``boxes``.  All directories are created
    if they do not yet exist so callers can assume they are present.
    """

    base = os.path.join(SHARED_DIR, username)
    dirs = {
        "resized": os.path.join(base, "resized"),
        "masks": os.path.join(base, "output", "masks"),
        "smalls": os.path.join(base, "output", "smalls"),
        "crops": os.path.join(base, "output", "crops"),
        "config": os.path.join(base, "config", "settings.json"),
        "processed": os.path.join(base, "output", "processed.json"),
        "points": os.path.join(base, "output", "points"),
        "boxes": os.path.join(base, "output", "boxes"),
    }

    # Ensure directories exist
    for path in [
        dirs["resized"],
        dirs["masks"],
        dirs["smalls"],
        dirs["crops"],
        os.path.dirname(dirs["config"]),
        dirs["points"],
        dirs["boxes"],
    ]:
        os.makedirs(path, exist_ok=True)

    return dirs

AREA_THRESH = 1000  # pixel area below which masks are treated as "smalls"

# Load BirefNet ONNX model from a shared S3 bucket using ``rembg``.
#
# The model weights live in ``s3://sam-server-shared-1757292440/models`` and are
# downloaded on demand into the shared models directory.  ``rembg`` looks up
# models by *name*, not by file path, so we place the ONNX file in ``U2NET_HOME``
# and request the ``birefnet-dis`` session explicitly.
os.environ["U2NET_HOME"] = MODELS_DIR
_BIRE_NET_ONNX = os.path.join(MODELS_DIR, "birefnet-dis.onnx")
if not os.path.exists(_BIRE_NET_ONNX):
    logger.info("No path found for birefnet-dis.onnx in %s", _BIRE_NET_ONNX)
    try:
        s3_client.download_file(
            "sam-server-shared-1757292440",
            "models/birefnet-dis.onnx",
            _BIRE_NET_ONNX,
        )
        logger.info("BirefNet model downloaded to %s", _BIRE_NET_ONNX)
    except ClientError as e:  # pragma: no cover - network/permission issues
        logger.error("[s3] failed to download BirefNet model: %s", e)

try:
    _REMBG_SESSION = new_session(
        "birefnet-dis", providers=["CUDAExecutionProvider"]
    )
except ValueError:
    logger.warning("[rembg] falling back to default session")
    _REMBG_SESSION = new_session(providers=["CUDAExecutionProvider"])

providers = _REMBG_SESSION.inner_session.get_providers()
if "CUDAExecutionProvider" not in providers:
    raise RuntimeError("rembg is not configured with CUDAExecutionProvider")
logger.info("[Worker] rembg providers: %s", providers)

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(YOLO_MODELS_DIR, exist_ok=True)





def _refine_mask_with_rembg(image_bgr: np.ndarray) -> np.ndarray:
    pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    logger.info("[Decision] running rembg remove")
    result = remove(pil_img, session=_REMBG_SESSION)
    alpha = np.array(result)[..., 3]
    logger.info("[Decision] rembg remove complete")
    return (alpha > 0).astype(np.uint8)


def _refine_mask_with_birefnet(image_bgr: np.ndarray) -> np.ndarray:
    """Refine mask using the BirefNet session.

    This simply delegates to rembg with the preloaded BirefNet model. A
    separate helper makes it easy to catch errors and fall back to the generic
    rembg model if needed.
    """
    logger.info("birefnet")
    return _refine_mask_with_rembg(image_bgr)


def _is_line_drawing(image_bgr: np.ndarray) -> bool:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = float(np.count_nonzero(edges)) / edges.size
    color_std = float(image_bgr.std())
    result = edge_ratio > 0.05 and color_std < 25.0
    logger.info(
        "[Decision] _is_line_drawing: edge_ratio=%.4f, color_std=%.2f -> %s",
        edge_ratio,
        color_std,
        result,
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
    logger.info("[Decision] _has_long_lines: count=%d -> %s", count, result)
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
    logger.info(
        "[Decision] _is_mostly_one_color: std=%.2f, thresh=%.2f -> %s",
        std,
        std_thresh,
        result,
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


def _get_yolo_points(image_path: str, boxes_dir: str) -> list[tuple[float, float, int]]:
    """Run all YOLO models and return midpoints labeled for SAM.

    Each returned tuple is ``(x, y, label)`` where ``label`` is ``1`` for
    regular objects (positive point) and ``0`` for detected humans (negative
    point). The center of any ``person``/``human`` box becomes a negative
    selector so SAM can avoid that region.
    """

    points: list[tuple[float, float, int]] = []
    if not _YOLO_AVAILABLE:
        logger.info(
            "[Worker] YOLO models not available, skipping YOLO point generation"
        )
        return points
    if not os.path.isdir(YOLO_MODELS_DIR):
        logger.warning(
            "[Worker] YOLO models directory '%s' not found, skipping",
            YOLO_MODELS_DIR,
        )
        return points

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    img = cv2.imread(image_path)
    combined = img.copy() if img is not None else None

    logger.info("[Worker] Running YOLO models on %s", image_path)
    try:
        model_files = os.listdir(YOLO_MODELS_DIR)
    except OSError as e:
        logger.error("[Worker] cannot access %s: %s", YOLO_MODELS_DIR, e)
        return points
    for fname in model_files:
        if not fname.lower().endswith((".pt", ".onnx")):
            continue
        # Skip any non-YOLO models such as the BirefNet weights which share
        # the models directory but are incompatible with the YOLO API.
        if "birefnet" in fname.lower():
            continue
        model_path = os.path.join(YOLO_MODELS_DIR, fname)
        logger.info("[Worker] Running YOLO model %s", fname)
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
                    logger.info(
                        "[Worker] %s detected %s at (%.1f, %.1f, %.1f, %.1f)",
                        fname,
                        label,
                        x1,
                        y1,
                        x2,
                        y2,
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
                    boxes_dir, f"{base_name}-{os.path.splitext(fname)[0]}.png"
                )
                cv2.imwrite(out_file, model_img)
        except Exception as e:  # pragma: no cover - inference may fail
            logger.error("[Worker] YOLO model %s failed: %s", fname, e)

    if combined is not None:
        try:
            out_file = os.path.join(boxes_dir, f"{base_name}-combined.png")
            cv2.imwrite(out_file, combined)
        except Exception as e:  # pragma: no cover - best effort only
            logger.error(
                "[Worker] Failed to save combined boxes for %s: %s", base_name, e
            )

    return points


def _save_yolo_points(
    points: list[tuple[float, float, int]],
    base_name: str,
    width: int,
    height: int,
    points_dir: str,
) -> None:
    """Persist YOLO midpoint data for later display on thumbnails.

    ``points`` is a list of ``(x, y, label)`` tuples. Labels are stored so the
    frontend can distinguish positive and negative selectors if desired.
    """

    try:
        data = {"width": width, "height": height, "points": points}
        out_path = os.path.join(points_dir, f"{base_name}.json")
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, out_path)
    except Exception as e:  # pragma: no cover - best effort only
        logger.error(
            "[Worker] Failed to save YOLO points for %s: %s", base_name, e
        )


def load_processed_set(processed_file: str, masks_dir: str):
    """Build a set of base filenames that have already been processed."""
    processed = set()
    # Load from persisted json if present
    if os.path.exists(processed_file):
        try:
            with open(processed_file, "r") as f:
                processed.update(json.load(f))
        except Exception:
            pass
    # Also include any masks that already exist on disk
    try:
        mask_files = os.listdir(masks_dir)
    except OSError as e:
        logger.error("[Worker] cannot access %s: %s", masks_dir, e)
        mask_files = []
    for fname in mask_files:
        if "_mask" in fname:
            base = fname.split("_mask")[0]
            processed.add(base)
    return processed


def save_processed_set(processed_set, processed_file: str):
    """Persist processed base filenames to disk atomically."""
    tmp = processed_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(sorted(processed_set), f)
    os.replace(tmp, processed_file)

# -------------------------
# Load SAM model
# -------------------------
if not os.path.exists(MODEL_PATH):
    logger.info("SAM model not found at %s", MODEL_PATH)
    try:
        logger.info(
            "downloading SAM model from sam-server-shared-1757292440 to %s",
            MODEL_PATH,
        )
        s3_client.download_file(
            "sam-server-shared-1757292440",
            "models/vit_l.pth",
            MODEL_PATH,
        )
        logger.info("SAM model downloaded to %s", MODEL_PATH)
    except ClientError as e:  # pragma: no cover - network/permission issues
        logger.error("[s3] failed to download SAM model: %s", e)
else:
    logger.info("SAM model found at %s", MODEL_PATH)
logger.debug("Files in models dir %s: %s", MODELS_DIR, os.listdir(MODELS_DIR))

sam = sam_model_registry["vit_l"](checkpoint=MODEL_PATH)
sam.to(DEVICE)

# -------------------------
# Helper functions
# -------------------------
def load_settings(config_file: str):
    """Load SAM settings from Server1 JSON file."""
    default = {
        "points_per_side": 32,
        "pred_iou_thresh": 0.88,
        "stability_score_thresh": 0.95,
        "crop_n_layers": 1,
        "model_type": "vit_l",
    }
    if os.path.exists(config_file):
        with open(config_file) as f:
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
    logger.info("using sam")
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
                logger.error("[Worker] SAM point prediction failed: %s", e)

    return masks, image

def save_masks(masks, image, base_name, masks_dir: str, smalls_dir: str):
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
            out_path = os.path.join(smalls_dir, f"{base_name}_small{small_idx}.png")
            small_idx += 1
        else:
            out_path = os.path.join(masks_dir, f"{base_name}_mask{big_idx}.png")
            big_idx += 1
        cv2.imwrite(out_path, out)

def process_new_images(username: str) -> int:
    """Process any unprocessed images found for ``username``.

    Returns the number of images that were processed.
    """

    dirs = get_user_dirs(username)
    processed = load_processed_set(dirs["processed"], dirs["masks"])
    settings = load_settings(dirs["config"])
    try:
        files = [
            f
            for f in os.listdir(dirs["resized"])
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]
    except OSError as e:
        logger.error("[Worker] cannot access %s: %s", dirs["resized"], e)
        files = []
    new_files = [f for f in files if os.path.splitext(f)[0] not in processed]
    if not new_files:
        logger.info("[Worker] No new files found for %s", username)
        return 0
    logger.info(
        "[Worker] Found %d new file(s) for %s: %s",
        len(new_files),
        username,
        new_files,
    )
    count = 0
    for f in new_files:
        base = os.path.splitext(f)[0]
        file_path = os.path.join(dirs["resized"], f)
        start = time.process_time()
        logger.info("[Worker] Processing %s for %s ...", f, username)
        try:
            img = cv2.imread(file_path)
            if img is None:
                continue
            simple = _is_line_drawing(img) or _has_long_lines(img)
            yolo_points = [] if simple else _get_yolo_points(file_path, dirs["boxes"])
            if simple or not yolo_points:
                logger.info("[Worker] Using BirefNet for segmentation")
                mask = _refine_mask_with_birefnet(img)
                crops = _crop_with_mask(img, mask)
                mask_file = os.path.join(dirs["masks"], f"{base}_mask0.png")
                cv2.imwrite(mask_file, mask.astype(np.uint8) * 255)
                for idx, crop in enumerate(crops):
                    crop_file = os.path.join(dirs["crops"], f"{base}_mask0_{idx}.png")
                    cv2.imwrite(crop_file, crop)
            else:
                logger.info("[Worker] Using SAM for segmentation")
                masks, img = generate_masks(file_path, settings, yolo_points)
                if img is not None and yolo_points:
                    h, w = img.shape[:2]
                    _save_yolo_points(yolo_points, base, w, h, dirs["points"])

                if masks:
                    largest = max(
                        masks, key=lambda m: int(np.count_nonzero(m["segmentation"]))
                    )
                    if _is_mostly_one_color(img, largest["segmentation"]):
                        try:
                            logger.info("refining with birefnet")
                            largest["segmentation"] = _refine_mask_with_birefnet(img).astype(bool)
                        except Exception:
                            logger.info("refining with rembg")
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
                save_masks(masks, img, base, dirs["masks"], dirs["smalls"])
            processed.add(base)
            count += 1
            gc.collect()
            end = time.process_time()
            total = end - start
            logger.info("elapsed time: %.6f seconds", total)
        except Exception as e:
            logger.error("[Worker] Error processing %s: %s", f, e)
    save_processed_set(processed, dirs["processed"])
    return count


def main() -> None:
    mod_dir = os.path.join(SHARED_DIR, "models")
    logger.info("Files in SHARED_DIR %s: %s", mod_dir, os.listdir(mod_dir))
    while True:
        try:
            users = [
                d
                for d in os.listdir(SHARED_DIR)
                if os.path.isdir(os.path.join(SHARED_DIR, d)) and d != "models"
            ]
        except OSError as e:
            logger.error("[Worker] cannot list users in %s: %s", SHARED_DIR, e)
            users = []
        total_processed = 0
        for user in users:
            logger.info("looking at user: %s", user)
            total_processed += process_new_images(user)
        if total_processed == 0:
            time.sleep(5)
        else:
            time.sleep(1)


if __name__ == "__main__":
    main()
