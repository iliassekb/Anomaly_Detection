"""
Feedback endpoints — confirm or correct model predictions,
persisting them as training data in MinIO.
"""

import base64
import json
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException

from api.minio_client import (
    get_classes,
    list_objects,
    object_exists,
    put_object,
)

router = APIRouter(prefix="/api/feedback", tags=["feedback"])

FEEDBACK_PREFIX = "feedback/"


# ── helpers ───────────────────────────────────────────────────────

def _decode_image(image_b64: str) -> np.ndarray:
    data = base64.b64decode(image_b64)
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image")
    return img


def _mask_b64_to_yolo(mask_b64: str, class_id: int, img_w: int, img_h: int) -> str:
    """Base64-encoded PNG mask → YOLO segmentation line."""
    data = base64.b64decode(mask_b64)
    arr = np.frombuffer(data, np.uint8)
    mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return ""
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return ""
    contour = max(cnts, key=cv2.contourArea).reshape(-1, 2)
    coords = " ".join(f"{x / img_w:.6f} {y / img_h:.6f}" for x, y in contour)
    return f"{class_id} {coords}"


def _polygons_to_yolo(polygons: list[dict], class_id: int, img_w: int, img_h: int) -> str:
    lines = []
    for poly in polygons:
        points = poly.get("points", [])
        if len(points) < 3:
            continue
        coords = " ".join(f"{p[0] / img_w:.6f} {p[1] / img_h:.6f}" for p in points)
        lines.append(f"{class_id} {coords}")
    return "\n".join(lines)


def _save_feedback_record(record: dict) -> str:
    fid = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
    key = f"{FEEDBACK_PREFIX}{fid}.json"
    put_object(key, json.dumps(record, indent=2).encode(), "application/json")
    return fid


def _check_retrain() -> None:
    try:
        from api.training import should_retrain, trigger_training_background
        if should_retrain():
            trigger_training_background()
    except Exception:
        pass


# ── endpoints ─────────────────────────────────────────────────────

@router.post("/confirm")
async def confirm_feedback(payload: dict):
    """
    User confirms that the model predictions are correct.
    Saves the image + converted YOLO labels to MinIO annotated/.

    payload: {
      image_b64: str,
      filename: str,
      class_name: str,
      detections: [{ mask_b64?: str, polygon?: [[x_norm, y_norm],...], label: str }]
    }
    """
    image_b64 = payload.get("image_b64", "")
    filename = payload.get("filename") or f"{uuid.uuid4().hex}.jpg"
    class_name = payload.get("class_name", "")
    detections = payload.get("detections", [])

    classes = get_classes()
    if class_name not in classes:
        raise HTTPException(404, f"Class '{class_name}' not found")

    try:
        img = _decode_image(image_b64)
    except ValueError:
        raise HTTPException(400, "Cannot decode image_b64")

    img_h, img_w = img.shape[:2]
    class_id = classes[class_name]

    yolo_lines = []
    for det in detections:
        det_label = det.get("label", class_name)
        det_class_id = classes.get(det_label, class_id)

        if "mask_b64" in det and det["mask_b64"]:
            line = _mask_b64_to_yolo(det["mask_b64"], det_class_id, img_w, img_h)
            if line:
                yolo_lines.append(line)
        elif "polygon" in det and det["polygon"]:
            pts = det["polygon"]
            if len(pts) >= 3:
                coords = " ".join(f"{p[0]:.6f} {p[1]:.6f}" for p in pts)
                yolo_lines.append(f"{det_class_id} {coords}")

    if not yolo_lines:
        raise HTTPException(422, "No valid detection polygons to save")

    yolo_txt = "\n".join(yolo_lines)

    # Encode image bytes back
    _, buf = cv2.imencode(".jpg", img)
    img_bytes = buf.tobytes()

    stem = Path(filename).stem
    img_key = f"annotated/{class_name}/images/{filename}"
    lbl_key = f"annotated/{class_name}/labels/{stem}.txt"

    put_object(img_key, img_bytes, "image/jpeg")
    put_object(lbl_key, yolo_txt.encode(), "text/plain")

    fid = _save_feedback_record({
        "type": "confirm",
        "class_name": class_name,
        "filename": filename,
        "num_detections": len(yolo_lines),
        "timestamp": int(time.time()),
    })

    _check_retrain()
    return {"saved": filename, "feedback_id": fid}


@router.post("/correct")
async def correct_feedback(payload: dict):
    """
    User corrects the model predictions by providing new polygons.

    payload: {
      image_b64: str,
      filename: str,
      class_name: str,
      polygons: [{ points: [[x, y],...] }],   # pixel coords
      img_w: int,
      img_h: int
    }
    """
    image_b64 = payload.get("image_b64", "")
    filename = payload.get("filename") or f"{uuid.uuid4().hex}.jpg"
    class_name = payload.get("class_name", "")
    polygons = payload.get("polygons", [])
    img_w = payload.get("img_w", 1)
    img_h = payload.get("img_h", 1)

    classes = get_classes()
    if class_name not in classes:
        raise HTTPException(404, f"Class '{class_name}' not found")

    if not polygons:
        raise HTTPException(422, "No polygons provided")

    class_id = classes[class_name]
    yolo_txt = _polygons_to_yolo(polygons, class_id, img_w, img_h)
    if not yolo_txt.strip():
        raise HTTPException(422, "No valid polygons (need ≥ 3 points each)")

    try:
        img = _decode_image(image_b64)
    except ValueError:
        raise HTTPException(400, "Cannot decode image_b64")

    _, buf = cv2.imencode(".jpg", img)
    img_bytes = buf.tobytes()

    stem = Path(filename).stem
    img_key = f"annotated/{class_name}/images/{filename}"
    lbl_key = f"annotated/{class_name}/labels/{stem}.txt"

    put_object(img_key, img_bytes, "image/jpeg")
    put_object(lbl_key, yolo_txt.encode(), "text/plain")

    fid = _save_feedback_record({
        "type": "correct",
        "class_name": class_name,
        "filename": filename,
        "num_polygons": len(polygons),
        "timestamp": int(time.time()),
    })

    _check_retrain()
    return {"saved": filename, "feedback_id": fid}


@router.get("/list")
def list_feedback():
    keys = list_objects(FEEDBACK_PREFIX)
    return {"count": len(keys), "keys": keys}
