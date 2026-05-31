"""
MobileSAM endpoint — multi-point prompting → polygon contour.
Supports positive (label=1) and negative (label=0) points to refine the mask.
"""

import base64
import os
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/sam", tags=["sam"])

_predictor = None
WEIGHTS_PATH = os.getenv("SAM_WEIGHTS", "weights/mobile_sam.pt")


def _load_predictor():
    global _predictor
    if _predictor is not None:
        return _predictor

    weights = Path(WEIGHTS_PATH)
    if not weights.exists():
        weights.parent.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id="dhkim2810/MobileSAM",
                filename="mobile_sam.pt",
                local_dir=str(weights.parent),
            )
            weights = Path(path)
        except Exception as e:
            raise RuntimeError(f"Cannot load MobileSAM weights: {e}")

    from mobile_sam import SamPredictor, sam_model_registry
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_t"](checkpoint=str(weights))
    sam.to(device)
    sam.eval()
    _predictor = SamPredictor(sam)
    return _predictor


def _mask_to_polygon(mask: np.ndarray, img_w: int, img_h: int) -> list[list[float]]:
    """Largest contour of a binary mask → list of [x_norm, y_norm] points."""
    mask_u8 = (mask * 255).astype(np.uint8)
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    contour = max(cnts, key=cv2.contourArea).reshape(-1, 2)
    return [[float(x) / img_w, float(y) / img_h] for x, y in contour]


@router.post("/segment")
def segment(payload: dict):
    """
    Input:  {
      image_b64: str,
      points: [{ x: float, y: float, label: int }]
              label 1 = foreground (extend), 0 = background (reduce)
    }
    Output: { polygon: [[x_norm, y_norm], ...], mask_b64: str }
    """
    image_b64 = payload.get("image_b64", "")
    points = payload.get("points", [])

    if not image_b64 or not points:
        raise HTTPException(400, "image_b64 and points are required")

    try:
        img_data = base64.b64decode(image_b64)
        arr = np.frombuffer(img_data, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        raise HTTPException(400, "Cannot decode image_b64")

    img_h, img_w = img_rgb.shape[:2]

    try:
        predictor = _load_predictor()
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    point_coords = np.array([[p["x"], p["y"]] for p in points], dtype=np.float32)
    point_labels = np.array([p["label"] for p in points])

    predictor.set_image(img_rgb)
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    best_mask = masks[int(np.argmax(scores))]
    polygon = _mask_to_polygon(best_mask, img_w, img_h)

    if not polygon:
        raise HTTPException(422, "SAM produced an empty mask")

    mask_u8 = (best_mask * 255).astype(np.uint8)
    _, buf = cv2.imencode(".png", mask_u8)
    mask_b64 = base64.b64encode(buf.tobytes()).decode()

    return JSONResponse({"polygon": polygon, "mask_b64": mask_b64})
