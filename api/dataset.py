import io
import json
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from api.minio_client import (
    copy_object,
    delete_object,
    get_classes,
    get_object,
    list_objects,
    object_exists,
    put_object,
    update_classes,
)

router = APIRouter(prefix="/api/dataset", tags=["dataset"])


# ── helpers ───────────────────────────────────────────────────────

def _next_class_id(classes: dict) -> int:
    return max(classes.values(), default=-1) + 1


def _annotated_image_key(class_name: str, filename: str) -> str:
    return f"annotated/{class_name}/images/{filename}"


def _annotated_label_key(class_name: str, filename: str) -> str:
    stem = Path(filename).stem
    return f"annotated/{class_name}/labels/{stem}.txt"


def _pending_key(filename: str) -> str:
    return f"pending/{filename}"


def _count_images(class_name: str) -> dict:
    annotated = len(list_objects(f"annotated/{class_name}/images/"))
    pending = [
        k for k in list_objects("pending/")
        if k.startswith(f"pending/{class_name}_")
    ]
    return {"annotated": annotated, "pending": len(pending)}


# ── annotation format converters ─────────────────────────────────

def _mask_png_to_yolo(mask_bytes: bytes, class_id: int, img_w: int, img_h: int) -> str:
    """Binary mask PNG → YOLO segmentation .txt line."""
    arr = np.frombuffer(mask_bytes, np.uint8)
    mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return ""
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return ""
    contour = max(cnts, key=cv2.contourArea).reshape(-1, 2)
    coords = " ".join(
        f"{x / img_w:.6f} {y / img_h:.6f}" for x, y in contour
    )
    return f"{class_id} {coords}"


def _coco_to_yolo(coco_json: dict, class_name_to_id: dict, img_w: int, img_h: int) -> str:
    """COCO segmentation JSON → YOLO segmentation .txt lines."""
    lines = []
    categories = {c["id"]: c["name"] for c in coco_json.get("categories", [])}
    for ann in coco_json.get("annotations", []):
        cat_name = categories.get(ann.get("category_id"), "")
        class_id = class_name_to_id.get(cat_name)
        if class_id is None:
            continue
        for seg in ann.get("segmentation", []):
            if len(seg) < 6:
                continue
            coords = " ".join(
                f"{seg[i] / img_w:.6f} {seg[i+1] / img_h:.6f}"
                for i in range(0, len(seg), 2)
            )
            lines.append(f"{class_id} {coords}")
    return "\n".join(lines)


def _polygons_to_yolo(polygons: list[dict], class_id: int, img_w: int, img_h: int) -> str:
    """[{points: [[x,y],...]}] → YOLO segmentation .txt lines."""
    lines = []
    for poly in polygons:
        points = poly.get("points", [])
        if len(points) < 3:
            continue
        coords = " ".join(
            f"{p[0] / img_w:.6f} {p[1] / img_h:.6f}" for p in points
        )
        lines.append(f"{class_id} {coords}")
    return "\n".join(lines)


def _get_image_size(image_bytes: bytes) -> tuple[int, int]:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image")
    h, w = img.shape[:2]
    return w, h


def _check_retrain_threshold() -> None:
    """Fire background training if any class has ≥ 50 new annotated images."""
    try:
        from api.training import should_retrain, trigger_training_background
        if should_retrain():
            trigger_training_background()
    except Exception:
        pass


# ── class endpoints ───────────────────────────────────────────────

@router.get("/classes")
def list_classes():
    classes = get_classes()
    result = []
    for name, cid in classes.items():
        counts = _count_images(name)
        result.append({"name": name, "id": cid, **counts})
    return {"classes": result}


@router.post("/classes", status_code=201)
def create_class(payload: dict):
    name = payload.get("name", "").strip().lower().replace(" ", "_")
    if not name:
        raise HTTPException(400, "Class name is required")
    classes = get_classes()
    if name in classes:
        raise HTTPException(409, f"Class '{name}' already exists")
    classes[name] = _next_class_id(classes)
    update_classes(classes)
    return {"name": name, "id": classes[name]}


@router.delete("/classes/{name}")
def delete_class(name: str):
    classes = get_classes()
    if name not in classes:
        raise HTTPException(404, f"Class '{name}' not found")
    for key in list_objects(f"annotated/{name}/"):
        delete_object(key)
    for key in list_objects("pending/"):
        if key.startswith(f"pending/{name}_"):
            delete_object(key)
    del classes[name]
    update_classes(classes)
    return {"deleted": name}


# ── image listing & serving ───────────────────────────────────────

@router.get("/images/{class_name}")
def list_images(class_name: str):
    classes = get_classes()
    if class_name not in classes:
        raise HTTPException(404, f"Class '{class_name}' not found")

    annotated_keys = list_objects(f"annotated/{class_name}/images/")
    pending_keys = [
        k for k in list_objects("pending/")
        if Path(k).name.startswith(f"{class_name}_")
    ]

    images = []
    for key in annotated_keys:
        filename = Path(key).name
        has_label = object_exists(_annotated_label_key(class_name, filename))
        images.append({"filename": filename, "status": "annotated" if has_label else "pending", "key": key})
    for key in pending_keys:
        images.append({"filename": Path(key).name, "status": "pending", "key": key})

    return {"images": images}


@router.get("/image/{class_name}/{filename}")
def get_image(class_name: str, filename: str):
    key = _annotated_image_key(class_name, filename)
    if not object_exists(key):
        key = f"pending/{filename}"
    if not object_exists(key):
        raise HTTPException(404, "Image not found")
    data = get_object(key)
    ext = Path(filename).suffix.lower()
    media = "image/png" if ext == ".png" else "image/jpeg"
    return Response(content=data, media_type=media)


# ── upload endpoints ──────────────────────────────────────────────

@router.post("/upload/raw")
async def upload_raw(
    class_name: str = Form(...),
    file: UploadFile = File(...),
):
    classes = get_classes()
    if class_name not in classes:
        raise HTTPException(404, f"Class '{class_name}' not found")

    image_bytes = await file.read()
    unique_name = f"{class_name}_{uuid.uuid4().hex[:8]}_{file.filename}"
    put_object(_pending_key(unique_name), image_bytes, file.content_type or "image/jpeg")
    return {"filename": unique_name, "status": "pending"}


@router.post("/upload/annotated")
async def upload_annotated(
    class_name: str = Form(...),
    image: UploadFile = File(...),
    annotation: UploadFile = File(...),
):
    classes = get_classes()
    if class_name not in classes:
        raise HTTPException(404, f"Class '{class_name}' not found")

    class_id = classes[class_name]
    image_bytes = await image.read()
    ann_bytes = await annotation.read()

    try:
        img_w, img_h = _get_image_size(image_bytes)
    except ValueError:
        raise HTTPException(400, "Cannot decode image")

    ann_name = annotation.filename or ""
    if ann_name.endswith(".txt"):
        yolo_txt = ann_bytes.decode()
    elif ann_name.endswith(".png"):
        yolo_txt = _mask_png_to_yolo(ann_bytes, class_id, img_w, img_h)
    elif ann_name.endswith(".json"):
        coco = json.loads(ann_bytes.decode())
        yolo_txt = _coco_to_yolo(coco, classes, img_w, img_h)
    else:
        raise HTTPException(400, "Unsupported annotation format (.txt, .png, .json)")

    if not yolo_txt.strip():
        raise HTTPException(422, "Annotation produced no valid polygons")

    filename = image.filename or f"{uuid.uuid4().hex}.jpg"
    put_object(_annotated_image_key(class_name, filename), image_bytes, image.content_type or "image/jpeg")
    put_object(_annotated_label_key(class_name, filename), yolo_txt.encode(), "text/plain")

    _check_retrain_threshold()
    return {"filename": filename, "status": "annotated"}


# ── annotation endpoints ──────────────────────────────────────────

@router.get("/annotation/{class_name}/{filename}")
def get_annotation(class_name: str, filename: str):
    key = _annotated_label_key(class_name, filename)
    if not object_exists(key):
        raise HTTPException(404, "Annotation not found")
    return Response(content=get_object(key), media_type="text/plain")


@router.post("/annotation/{class_name}/{filename}")
async def save_annotation(class_name: str, filename: str, payload: dict):
    """
    payload: {
      polygons: [{points: [[x,y],...]}],
      img_w: int,
      img_h: int,
      source_key?: str   (pending key to move to annotated/)
    }
    """
    classes = get_classes()
    if class_name not in classes:
        raise HTTPException(404, f"Class '{class_name}' not found")

    class_id = classes[class_name]
    polygons = payload.get("polygons", [])
    img_w = payload.get("img_w", 1)
    img_h = payload.get("img_h", 1)
    source_key = payload.get("source_key")

    yolo_txt = _polygons_to_yolo(polygons, class_id, img_w, img_h)
    if not yolo_txt.strip():
        raise HTTPException(422, "No valid polygons provided")

    # move image from pending to annotated if needed
    if source_key and source_key.startswith("pending/"):
        copy_object(source_key, _annotated_image_key(class_name, filename))
        delete_object(source_key)

    put_object(_annotated_label_key(class_name, filename), yolo_txt.encode(), "text/plain")
    _check_retrain_threshold()
    return {"saved": filename}
