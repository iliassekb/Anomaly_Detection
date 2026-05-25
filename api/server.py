"""
FastAPI inference server for DefectVision Pro.
Run from project root: python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""

import base64
import functools
import io
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

torch.load = functools.partial(torch.load, weights_only=False)

DEFAULT_WEIGHTS = "weights/best_m.pt"
HF_SPACE_REPO = "imad9/defectvision"

app = FastAPI(title="DefectVision API", version="1.0")


@app.on_event("startup")
def startup():
    try:
        from api.minio_client import init_buckets
        init_buckets()
    except Exception as e:
        import logging
        logging.warning(f"MinIO init skipped: {e}")

from api.dataset import router as dataset_router
from api.sam import router as sam_router
from api.training import router as training_router
from api.feedback import router as feedback_router
app.include_router(dataset_router)
app.include_router(sam_router)
app.include_router(training_router)
app.include_router(feedback_router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Stats"],
)

_model_cache: dict[str, object] = {}


def get_model(weights_path: str):
    from ultralytics import YOLO
    if weights_path not in _model_cache:
        _model_cache[weights_path] = YOLO(weights_path)
    return _model_cache[weights_path]


def _is_lfs_pointer(p: Path) -> bool:
    return p.exists() and p.stat().st_size < 1_000_000


def _ensure_weights(weights_path: str) -> None:
    p = Path(weights_path)
    if p.exists() and not _is_lfs_pointer(p):
        return
    try:
        from huggingface_hub import hf_hub_download
        import shutil
        p.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_SPACE_REPO, repo_type="space",
            filename=weights_path, local_dir=".",
        )
        dl = Path(downloaded)
        if dl.resolve() != p.resolve() and dl.exists():
            shutil.copy2(dl, p)
    except Exception:
        pass


_ensure_weights(DEFAULT_WEIGHTS)


def post_process_mask(raw_mask, target_w, target_h, threshold=0.45):
    soft = cv2.resize(raw_mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    soft = cv2.GaussianBlur(soft, (7, 7), sigmaX=2.0)
    binary = (soft > threshold).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_cc > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        binary = np.where(labels == largest, np.uint8(255), np.uint8(0))
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return (binary > 0).astype(np.uint8), None
    outer = max(cnts, key=cv2.contourArea)
    filled = np.zeros((target_h, target_w), dtype=np.uint8)
    cv2.drawContours(filled, [outer], -1, 255, cv2.FILLED)
    smooth = cv2.approxPolyDP(outer, 0.002 * cv2.arcLength(outer, True), True)
    return (filled > 0).astype(np.uint8), smooth


def extract_product_mask(frame):
    h, w = frame.shape[:2]
    bw = max(20, min(h, w) // 8)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    bdr = np.zeros((h, w), dtype=bool)
    bdr[:bw, :] = bdr[-bw:, :] = bdr[:, :bw] = bdr[:, -bw:] = True
    bg = lab[bdr]
    bg_mean, bg_std = np.mean(bg, axis=0), np.std(bg, axis=0) + 1.0
    diff = (lab - bg_mean) / bg_std
    diff[:, :, 0] *= 0.5; diff[:, :, 1] *= 1.3; diff[:, :, 2] *= 1.3
    dist = np.sqrt((diff ** 2).sum(axis=2))
    fg = (dist > 2.5).astype(np.uint8) * 255
    fg[bdr] = 0
    if fg.sum() / 255 / (h * w) < 0.02:
        fg = (dist > 1.8).astype(np.uint8) * 255
        fg[bdr] = 0
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    fg_e = cv2.erode(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg_e, connectivity=8)
    if n > 1:
        cy_img, cx_img = h / 2.0, w / 2.0
        best, best_score = 1, -1.0
        for i in range(1, n):
            x_, y_, bw_, bh_, area = stats[i]
            d = np.sqrt((x_ + bw_ / 2 - cx_img) ** 2 + (y_ + bh_ / 2 - cy_img) ** 2)
            score = area / (1.0 + d)
            if score > best_score:
                best_score, best = score, i
        fg_e = np.where(labels == best, np.uint8(255), np.uint8(0))
    fg = cv2.dilate(fg_e, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)))
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        hull = cv2.convexHull(np.vstack([c.reshape(-1, 2) for c in cnts]))
        fg = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(fg, [hull], -1, 255, cv2.FILLED)
    fg = cv2.dilate(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    return (fg > 0).astype(np.uint8)


def annotate_frame(frame, result, class_names, conf_thr, product_mask=None, mask_overlap_thr=0.20):
    detections = []
    if product_mask is not None:
        pm_cnts, _ = cv2.findContours(product_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if pm_cnts:
            outer = max(pm_cnts, key=cv2.contourArea)
            smooth = cv2.approxPolyDP(outer, 0.005 * cv2.arcLength(outer, True), True)
            cv2.polylines(frame, [smooth], True, (255, 0, 0), 2, cv2.LINE_AA)
    if result.masks is None:
        return frame, detections
    masks_data = result.masks.data.cpu().numpy()
    boxes = result.boxes
    h, w = frame.shape[:2]
    overlay = frame.copy()
    for i in range(len(boxes)):
        conf = float(boxes.conf[i].item())
        if conf < conf_thr:
            continue
        mask_bin, smooth_cnt = post_process_mask(masks_data[i], w, h)
        if product_mask is not None:
            det_area = int(mask_bin.sum())
            if det_area > 0 and int((mask_bin & product_mask).sum()) / det_area < mask_overlap_thr:
                continue
        cls_id = int(boxes.cls[i].item())
        name = class_names.get(cls_id, f"class_{cls_id}")
        is_good = name.lower() == "good"
        detections.append({"class": name, "class_id": cls_id, "conf": round(conf, 4), "is_good": is_good})
        color = (0, 200, 80) if is_good else (0, 0, 255)
        overlay[mask_bin == 1] = color
        if smooth_cnt is not None:
            cv2.polylines(frame, [smooth_cnt], True, color, 2, cv2.LINE_AA)
        else:
            cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, cnts, -1, color, 2)
        M = cv2.moments(mask_bin)
        if M["m00"] > 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        else:
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        label = f"{name}  {conf*100:.0f}%"
        (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
        lx = max(0, min(cx - lw // 2, w - lw - 4))
        ly = max(lh + bl + 4, cy)
        cv2.rectangle(frame, (lx - 4, ly - lh - bl - 4), (lx + lw + 4, ly + bl + 2), (10, 10, 30), -1)
        cv2.putText(frame, label, (lx, ly - bl // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    bh = 44
    anomaly_dets = [d for d in detections if not d["is_good"]]
    if anomaly_dets:
        unique = list(dict.fromkeys(d["class"] for d in anomaly_dets))
        banner = f"  ANOMALY  |  {', '.join(unique).upper()}"
        cv2.rectangle(frame, (0, 0), (min(len(banner) * 13 + 20, w), bh), (10, 10, 140), -1)
        cv2.putText(frame, banner, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (0, 0), (240, bh), (10, 50, 20), -1)
        cv2.putText(frame, "  PASS — NORMAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return frame, detections


def to_b64(frame_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode()


@app.get("/api/health")
def health():
    return {"status": "ok", "device": "cuda" if torch.cuda.is_available() else "cpu"}


@app.get("/api/models")
def list_models():
    candidates = ["weights/best_m.pt", "weights/best_s.pt"]
    return {"models": [c for c in candidates if Path(c).exists()]}


@app.post("/api/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    weights: str = Form(DEFAULT_WEIGHTS),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    imgsz: int = Form(800),
    isolate: bool = Form(True),
    mask_overlap_thr: float = Form(0.20),
    return_original: bool = Form(True),
):
    if not Path(weights).exists():
        raise HTTPException(400, f"Weights not found: {weights}")
    arr = np.frombuffer(await file.read(), np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Cannot decode image")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(weights)
    class_names = dict(model.names)
    t0 = time.time()
    prod_mask = extract_product_mask(frame) if isolate else None
    results = model.predict(frame, imgsz=imgsz, conf=conf, iou=iou, device=device, save=False, verbose=False)
    elapsed_ms = (time.time() - t0) * 1000
    annotated, detections = annotate_frame(
        frame.copy(), results[0], class_names, conf,
        product_mask=prod_mask, mask_overlap_thr=mask_overlap_thr,
    )
    return {
        "original_b64": to_b64(frame) if return_original else None,
        "annotated_b64": to_b64(annotated),
        "detections": detections,
        "inference_ms": round(elapsed_ms, 1),
        "is_anomaly": any(not d["is_good"] for d in detections),
        "class_names": list(class_names.values()),
    }


def _get_ffmpeg() -> str:
    """Return ffmpeg executable: system install or imageio-bundled binary."""
    import shutil
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    try:
        from imageio.plugins.ffmpeg import get_exe
        return get_exe()
    except Exception:
        return "ffmpeg"


def _reencode_h264(src: str) -> str:
    """Re-encode to H.264 using ffmpeg for browser compatibility. Returns output path."""
    dst = src.replace(".mp4", "_h264.mp4")
    try:
        r = subprocess.run(
            [_get_ffmpeg(), "-y", "-i", src,
             "-vcodec", "libx264", "-crf", "23", "-preset", "fast",
             "-movflags", "+faststart",
             dst],
            capture_output=True, timeout=300,
        )
        if r.returncode == 0 and Path(dst).exists():
            return dst
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return src  # fall back to original if re-encoding fails


@app.post("/api/predict/video")
async def predict_video(
    file: UploadFile = File(...),
    weights: str = Form(DEFAULT_WEIGHTS),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    imgsz: int = Form(640),
    step: int = Form(2),
):
    if not Path(weights).exists():
        raise HTTPException(400, f"Weights not found: {weights}")
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        in_path = tmp.name
    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    raw_out = out_tmp.name
    out_tmp.close()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(weights)
    class_names = dict(model.names)
    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    all_dets, anomaly_frames, total_frames = [], 0, 0
    last_annotated = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if total_frames % step == 0:
            results = model.predict(frame, imgsz=imgsz, conf=conf, iou=iou, device=device, save=False, verbose=False)
            ann, dets = annotate_frame(frame.copy(), results[0], class_names, conf)
            all_dets.extend(dets)
            if dets:
                anomaly_frames += 1
            last_annotated = ann
        writer.write(last_annotated if last_annotated is not None else frame)
        total_frames += 1
    cap.release()
    writer.release()
    os.unlink(in_path)

    # Re-encode to H.264 so the browser <video> tag can play it
    out_path = _reencode_h264(raw_out)
    if out_path != raw_out:
        os.unlink(raw_out)

    counts: dict[str, int] = {}
    for d in all_dets:
        counts[d["class"]] = counts.get(d["class"], 0) + 1
    stats = {
        "total_frames": total_frames,
        "anomaly_frames": anomaly_frames,
        "anomaly_rate": round(anomaly_frames / max(total_frames, 1) * 100, 1),
        "total_detections": len(all_dets),
        "detections_by_class": counts,
    }

    def _stream():
        try:
            with open(out_path, "rb") as f:
                yield from iter(lambda: f.read(65536), b"")
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass

    return StreamingResponse(
        _stream(), media_type="video/mp4",
        headers={"X-Stats": json.dumps(stats)},
    )
