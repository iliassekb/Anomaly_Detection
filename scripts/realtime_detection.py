# Usage examples:
#   python realtime_detection.py --source 0
#   python realtime_detection.py --source video.mp4 --save-video
#   python realtime_detection.py --source rtsp://192.168.1.100/stream --conf 0.3
#   python realtime_detection.py --weights runs/segment_multiclass/mvtec3d_anomaly_type/weights/best.pt --source 0

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# PyTorch >= 2.6 defaults weights_only=True which blocks ultralytics checkpoints.
torch.load = functools.partial(torch.load, weights_only=False)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_palette(n_classes: int):
    """Return a list of BGR uint8 tuples, one per class, using tab20 colormap."""
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab20")
    palette = []
    for i in range(n_classes):
        r, g, b, _ = cmap(i % 20)
        palette.append((int(b * 255), int(g * 255), int(r * 255)))  # BGR
    return palette


def load_class_names(json_path: str) -> dict:
    """Load class_ids.json and return {id: name} mapping."""
    with open(json_path, "r") as f:
        name_to_id: dict = json.load(f)
    return {v: k for k, v in name_to_id.items()}


def draw_banner(frame, text: str, is_anomaly: bool):
    """Draw a colored status banner in the top-left corner."""
    color = (0, 0, 200) if is_anomaly else (0, 180, 0)   # red : green (BGR)
    cv2.rectangle(frame, (0, 0), (520, 40), color, -1)
    cv2.putText(frame, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)


def draw_hud(frame, fps: float, source_name: str, frame_idx: int):
    """Draw FPS (top-right), source (bottom-left), frame counter (bottom-right)."""
    h, w = frame.shape[:2]

    # Top-right: FPS
    fps_text = f"FPS: {fps:04.1f}"
    (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(frame, (w - tw - 16, 4), (w - 4, th + 12), (30, 30, 30), -1)
    cv2.putText(frame, fps_text, (w - tw - 10, th + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    # Bottom-left: source
    src_text = f"SRC: {source_name}"
    cv2.rectangle(frame, (0, h - 28), (len(src_text) * 10 + 10, h), (30, 30, 30), -1)
    cv2.putText(frame, src_text, (6, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # Bottom-right: frame counter
    fc_text = f"Frame: {frame_idx:04d}"
    (fw, fh), _ = cv2.getTextSize(fc_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (w - fw - 14, h - 28), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, fc_text, (w - fw - 8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)



def extract_product_mask(frame: np.ndarray) -> np.ndarray:
    """
    Two-strategy product segmentation:
    - Achromatic bg (white/grey): LAB distance + shadow filter.
    - Chromatic/textured bg (wood): edge-shape fallback.
    Returns a uint8 binary mask (1 = product, 0 = background).
    """
    h, w = frame.shape[:2]
    bw   = max(8, min(h, w) // 15)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)

    border_pixels = np.concatenate([
        lab[:bw, :].reshape(-1, 3),
        lab[-bw:, :].reshape(-1, 3),
        lab[:, :bw].reshape(-1, 3),
        lab[:, -bw:].reshape(-1, 3),
    ])
    bg_lab = np.median(border_pixels, axis=0)

    bg_chroma = float(np.sqrt((bg_lab[1] - 128) ** 2 + (bg_lab[2] - 128) ** 2))

    diff    = lab - bg_lab
    dist    = np.sqrt((diff * diff).sum(axis=2))
    dist_u8 = np.clip(dist / (dist.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
    _, fg   = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if bg_chroma < 15:
        ab_dist = np.sqrt(diff[:, :, 1] ** 2 + diff[:, :, 2] ** 2)
        fg[(fg == 255) & (ab_dist < 8)] = 0

    coverage = fg.sum() / 255 / (h * w)

    if bg_chroma > 15 or not (0.05 < coverage < 0.85):
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=15)
        edges   = cv2.Canny(blurred, 15, 50)

        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        closed  = cv2.dilate(edges, k_close, iterations=2)

        tmp  = closed.copy()
        step = max(1, min(h, w) // 20)
        for x in range(0, w, step):
            for sy in (0, h - 1):
                if tmp[sy, x] == 0:
                    cv2.floodFill(tmp, None, (x, sy), 128)
        for y in range(step, h - step, step):
            for sx in (0, w - 1):
                if tmp[y, sx] == 0:
                    cv2.floodFill(tmp, None, (sx, y), 128)

        edge_fg  = (tmp == 0).astype(np.uint8) * 255
        edge_cov = edge_fg.sum() / 255 / (h * w)
        if 0.05 < edge_cov < 0.85:
            fg = edge_fg

    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fg = np.where(labels == largest, np.uint8(255), np.uint8(0))

    k_close  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE,  k_close)
    fg = cv2.morphologyEx(fg, cv2.MORPH_DILATE, k_dilate)

    return (fg > 0).astype(np.uint8)


def draw_detections(frame, result, class_names: dict, palette: list,
                    best_only: bool, product_mask: np.ndarray | None = None):
    """
    Overlay segmentation masks and bounding-box labels onto frame.
    product_mask is used only for the visual cyan outline — never blocks detections.
    Returns a list of detected class names (may contain duplicates).
    """
    detected_names = []

    # draw product outline on every frame (visual only)
    if product_mask is not None:
        pm_cnts, _ = cv2.findContours(product_mask, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, pm_cnts, -1, (0, 220, 255), 2)

    if result.masks is None:
        return detected_names

    masks_data = result.masks.data.cpu().numpy()
    boxes      = result.boxes
    h, w       = frame.shape[:2]

    conf_vals = boxes.conf.cpu().numpy()
    order     = np.argsort(conf_vals)[::-1]
    if best_only:
        order = order[:1]

    overlay = frame.copy()

    for idx in order:
        raw_mask = masks_data[idx]
        mask_bin = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bin = (mask_bin > 0.5).astype(np.uint8)

        # Skip detections that are mostly outside the product zone
        if product_mask is not None:
            det_area = int(mask_bin.sum())
            if det_area > 0 and int((mask_bin & product_mask).sum()) / det_area < 0.20:
                continue

        cls_id = int(boxes.cls[idx].item())
        conf   = float(conf_vals[idx])
        name   = class_names.get(cls_id, f"class_{cls_id}")
        color  = palette[cls_id % len(palette)]

        detected_names.append(name)

        overlay[mask_bin == 1] = color
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, color, 2)

        # ---- label at mask centroid ----
        M = cv2.moments(mask_bin)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x1, y1, x2, y2 = boxes.xyxy[idx].cpu().numpy().astype(int)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        label = f"{name} {conf*100:.0f}%"
        (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        lx = max(0, cx - lw // 2)
        ly = max(lh + bl, cy)
        cv2.rectangle(frame, (lx - 2, ly - lh - bl - 2),
                      (lx + lw + 2, ly + bl), (0, 0, 0), -1)
        cv2.putText(frame, label, (lx, ly - bl // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # blend mask overlay at 45% opacity
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    return detected_names


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    DEFAULT_WEIGHTS = "mvtec3d_anomaly_type3/weights/best.pt"

    p = argparse.ArgumentParser(
        description="Real-time multiclass anomaly detection with YOLO segmentation"
    )
    p.add_argument("--source",     default="0",
                   help="Webcam index (int), video file path, or RTSP URL")
    p.add_argument("--weights",    default=DEFAULT_WEIGHTS,
                   help="Path to best.pt YOLO weights")
    p.add_argument("--class-json", default=None,
                   help="Optional path to class_ids.json. If omitted, class names "
                        "are read directly from the model weights.")
    p.add_argument("--conf",       type=float, default=0.25,
                   help="Confidence threshold (default 0.25)")
    p.add_argument("--iou",        type=float, default=0.45,
                   help="IoU NMS threshold (default 0.45)")
    p.add_argument("--imgsz",      type=int,   default=320,
                   help="Inference image size (default 320 for speed; use 640/800 for accuracy)")
    p.add_argument("--skip",       type=int,   default=2,
                   help="Run inference every N frames, reuse last result in between "
                        "(default 2; set to 1 to infer every frame)")
    p.add_argument("--no-isolate", action="store_true",
                   help="Disable product outline detection (on by default).")
    p.add_argument("--margin",     type=float, default=0.10,
                   help="GrabCut border margin — fraction of image edge treated as "
                        "background (default 0.10)")
    p.add_argument("--gc-skip",    type=int,   default=30,
                   help="Re-run GrabCut every N frames (default 30). "
                        "Product is static so the mask stays valid for many frames.")
p.add_argument("--save-video", action="store_true",
                   help="Record annotated output to output.mp4")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- device ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_label = device
    if device == "cuda":
        device_label = f"cuda ({torch.cuda.get_device_name(0)})"

    # ---- model ----
    weights_path = Path(args.weights)
    if not weights_path.exists():
        sys.exit(f"[ERROR] Weights not found: {weights_path}")
    model = YOLO(str(weights_path))

    # ---- class names ----
    # Primary source: names embedded in the weights (model.names = {id: name})
    # Override: explicit --class-json if provided and the file exists
    class_names: dict = dict(model.names)   # {int -> str}
    class_source = "model weights"

    if args.class_json is not None:
        json_path = Path(args.class_json)
        if not json_path.exists():
            sys.exit(f"[ERROR] class_ids.json not found: {json_path}")
        class_names  = load_class_names(str(json_path))
        class_source = json_path.name

    n_classes = max(class_names.keys()) + 1
    palette   = build_palette(n_classes)

    # ---- source label ----
    try:
        src_index = int(args.source)
        source    = src_index
        src_label = f"webcam:{src_index}"
    except ValueError:
        source    = args.source
        src_label = Path(args.source).name if Path(args.source).exists() else args.source

    # ---- startup banner ----
    print("=" * 60)
    print("  Real-time Multiclass Anomaly Detection")
    print("=" * 60)
    print(f"  Model   : {weights_path}")
    print(f"  Classes : {len(class_names)} from {class_source} -> {list(class_names.values())}")
    print(f"  Source  : {src_label}")
    print(f"  Device  : {device_label}")
    print(f"  Conf    : {args.conf}   IoU: {args.iou}   ImgSz: {args.imgsz}   Skip: {args.skip}")
    isolate = not args.no_isolate
    print(f"  Isolate : {'ON' if isolate else 'OFF'}")
    print("-" * 60)
    print("  Controls:")
    print("    Q / ESC  → quit")
    print("    S        → save screenshot")
    print("    P        → pause / resume")
    print("    C        → toggle best-only detection")
    print("=" * 60)

    # ---- capture ----
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open source: {source}")

    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ---- video writer (optional) ----
    writer      = None
    writer_init = False   # lazily initialised after first frame so we know exact size

    # ---- state ----
    paused       = False
    best_only    = False
    frame_idx    = 0
    screenshot_n = 0
    fps          = 0.0
    t_prev       = time.time()
    paused_frame = None
    last_result   = None   # cached inference result reused for skipped frames
    last_prod_mask = None  # cached GrabCut mask, refreshed on same cadence as inference

    try:
        while True:
            # ---- key handling ----
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):          # Q or ESC
                break
            elif key == ord("s"):
                fname = f"screenshot_{screenshot_n:03d}.jpg"
                save_frame = paused_frame if paused else None
                if save_frame is not None:
                    cv2.imwrite(fname, save_frame)
                    print(f"[INFO] Saved {fname}")
                screenshot_n += 1
            elif key == ord("p"):
                paused = not paused
                print(f"[INFO] {'Paused' if paused else 'Resumed'}")
            elif key == ord("c"):
                best_only = not best_only
                mode = "best detection only" if best_only else "all detections"
                print(f"[INFO] Mode: {mode}")

            # ---- pause: redisplay frozen frame ----
            if paused:
                if paused_frame is not None:
                    cv2.imshow("Anomaly Detection", paused_frame)
                continue

            # ---- read frame ----
            ok, frame = cap.read()
            if not ok:
                print("[INFO] End of stream.")
                break

            # ---- FPS ----
            t_now  = time.time()
            fps    = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now

# ---- GrabCut (every gc_skip frames, independent of inference) ----
            if isolate and (frame_idx % args.gc_skip == 0 or last_prod_mask is None):
                last_prod_mask = extract_product_mask(frame)

            # ---- inference (every skip frames) ----
            if frame_idx % args.skip == 0 or last_result is None:
                results     = model.predict(
                    frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=device,
                    half=(device == "cuda"),   # FP16 on GPU for speed
                    save=False,
                    verbose=False,
                )
                last_result = results[0]
            result    = last_result
            prod_mask = last_prod_mask if isolate else None

            # ---- annotate ----
            detected_names = draw_detections(
                frame, result, class_names, palette, best_only,
                product_mask=prod_mask,
            )

            if detected_names:
                unique_types = list(dict.fromkeys(detected_names))   # deduplicated, ordered
                banner_text  = f"  ANOMALY: {', '.join(unique_types)}"
                draw_banner(frame, banner_text, is_anomaly=True)
            else:
                draw_banner(frame, "  NORMAL", is_anomaly=False)

            draw_hud(frame, fps, src_label, frame_idx)

            # ---- video writer init (first frame) ----
            if args.save_video and not writer_init:
                h_f, w_f = frame.shape[:2]
                raw_fps  = cap.get(cv2.CAP_PROP_FPS)
                out_fps  = raw_fps if raw_fps > 0 else 30.0
                fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
                writer   = cv2.VideoWriter("output.mp4", fourcc, out_fps, (w_f, h_f))
                writer_init = True
                print(f"[INFO] Recording to output.mp4 ({w_f}x{h_f} @ {out_fps:.0f} fps)")

            if writer is not None:
                writer.write(frame)

            # ---- display ----
            cv2.imshow("Anomaly Detection", frame)
            paused_frame = frame.copy()   # keep last good frame for screenshot on pause
            frame_idx   += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print("[INFO] Video saved to output.mp4")
        cv2.destroyAllWindows()
        print("[INFO] Exited cleanly.")


if __name__ == "__main__":
    main()
