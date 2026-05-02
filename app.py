"""
Streamlit app — MVTec 3D-AD Multiclass Anomaly Detection
Run:  streamlit run app.py
"""

import functools
import io
import tempfile
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO

torch.load = functools.partial(torch.load, weights_only=False)

# ── constants ────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = "mvtec3d_anomaly_type3/weights/best.pt"
HF_SPACE_REPO   = "imad9/defectvision"


def _is_lfs_pointer(p: Path) -> bool:
    """Return True if the file is a Git LFS pointer (~134 bytes) not the real binary."""
    return p.exists() and p.stat().st_size < 1_000_000


def _ensure_weights(weights_path: str) -> None:
    """Download weights via HF Hub API if the file is absent or an LFS pointer."""
    p = Path(weights_path)
    if p.exists() and not _is_lfs_pointer(p):
        return
    try:
        from huggingface_hub import hf_hub_download
        import shutil
        p.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_SPACE_REPO,
            repo_type="space",
            filename=weights_path,
            local_dir=".",
        )
        dl = Path(downloaded)
        if dl.resolve() != p.resolve() and dl.exists():
            shutil.copy2(dl, p)
    except Exception:
        pass


_ensure_weights(DEFAULT_WEIGHTS)
CONF_DEFAULT    = 0.25
IOU_DEFAULT     = 0.45
IMGSZ_DEFAULT   = 800

# ── theme palettes ────────────────────────────────────────────────────────────
THEMES: dict[str, dict] = {
    "Dark": {
        "bg":           "#060c18",
        "card":         "#0a1428",
        "border":       "#162035",
        "sidebar":      "#07101f",
        "t1":           "#dce8fc",
        "t2":           "#4e73a8",
        "acc":          "#1f46a8",
        "acc_l":        "#6899e0",
        "tab_sel_bg":   "#152d6e",
        "tab_sel_fg":   "#dce8fc",
        "chart_bg":     "#060c18",
        "grid_c":       "#162035",
        "bar_c":        "#a01c1c",
        "bar_e":        "#6b0d0d",
        "pass_bg":      "#071a0e",
        "pass_border":  "#0d4a22",
        "pass_fg":      "#27c96a",
        "brand_g1":     "#0d2260",
        "brand_g2":     "#152d82",
        "brand_border": "#1f46a8",
        "brand_name":   "#dce8fc",
        "brand_sub":    "#6899e0",
        "section_c":    "#1f46a8",
        "inf_bg":       "#0a1428",
        "inf_border":   "#162035",
        "inf_fg":       "#4e73a8",
        "code_bg":      "#0a1428",
        "code_fg":      "#6899e0",
        "head_eyebrow": "#1f46a8",
        "head_title":   "#dce8fc",
        "head_border":  "#162035",
    },
    "Light": {
        "bg":           "#f4f6fb",
        "card":         "#ffffff",
        "border":       "#dce3ef",
        "sidebar":      "#ffffff",
        "t1":           "#0d1b3e",
        "t2":           "#4e6ba0",
        "acc":          "#1f46a8",
        "acc_l":        "#3462d4",
        "tab_sel_bg":   "#1f46a8",
        "tab_sel_fg":   "#ffffff",
        "chart_bg":     "#ffffff",
        "grid_c":       "#dce3ef",
        "bar_c":        "#c0392b",
        "bar_e":        "#922b21",
        "pass_bg":      "#edfbf3",
        "pass_border":  "#6fcf97",
        "pass_fg":      "#1a6b3a",
        "brand_g1":     "#1a3a8c",
        "brand_g2":     "#2450b4",
        "brand_border": "#4a7af0",
        "brand_name":   "#ffffff",
        "brand_sub":    "#b8d0fc",
        "section_c":    "#1f46a8",
        "inf_bg":       "#ffffff",
        "inf_border":   "#dce3ef",
        "inf_fg":       "#4e6ba0",
        "code_bg":      "#eef1f8",
        "code_fg":      "#1f46a8",
        "head_eyebrow": "#1f46a8",
        "head_title":   "#0d1b3e",
        "head_border":  "#dce3ef",
    },
}


def make_css(t: dict) -> str:
    return f"""
<style>
#MainMenu, footer, header {{ visibility: hidden; }}

.stApp,
[data-testid="stAppViewContainer"] > .main {{ background: {t['bg']}; }}

[data-testid="stSidebar"] {{
    background: {t['sidebar']};
    border-right: 1px solid {t['border']};
}}

[data-testid="metric-container"] {{
    background: {t['card']};
    border: 1px solid {t['border']};
    border-radius: 10px;
    padding: 16px !important;
}}
[data-testid="metric-container"] label {{
    color: {t['t2']} !important;
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    font-weight: 600;
}}
[data-testid="stMetricValue"] {{
    color: {t['t1']} !important;
    font-weight: 700 !important;
}}
[data-testid="stMetricDelta"] {{ font-size: 11px !important; }}

[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    background: {t['card']};
    border-radius: 10px;
    padding: 4px;
    border: 1px solid {t['border']};
    display: flex !important;
    justify-content: space-evenly !important;
    gap: 0px;
}}
[data-testid="stTabs"] [data-baseweb="tab"] {{
    border-radius: 7px;
    color: {t['t2']} !important;
    font-weight: 500;
    font-size: 13px;
    flex: 1 1 0 !important;
    text-align: center !important;
    justify-content: center !important;
}}
[data-testid="stTabs"] [aria-selected="true"] {{
    background: {t['tab_sel_bg']} !important;
    color: {t['tab_sel_fg']} !important;
}}
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] {{ display: none; }}

button[kind="primary"] {{
    background: linear-gradient(135deg, {t['acc']}, {t['acc_l']}) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.4px !important;
    transition: all 0.2s !important;
}}
button[kind="primary"]:hover {{
    filter: brightness(1.15) !important;
    box-shadow: 0 0 14px rgba(31,70,168,0.45) !important;
}}

[data-testid="stDownloadButton"] button {{
    background: {t['card']} !important;
    color: {t['acc']} !important;
    border: 1px solid {t['acc']} !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}}
[data-testid="stDownloadButton"] button:hover {{
    background: {t['border']} !important;
}}

[data-testid="stFileUploader"] section {{
    border: 2px dashed {t['acc']} !important;
    border-radius: 12px !important;
    background: {t['card']} !important;
    padding: 40px 24px !important;
    transition: border-color 0.2s, background 0.2s !important;
}}
[data-testid="stFileUploader"] section:hover {{
    border-color: {t['acc_l']} !important;
    background: {t['border']} !important;
}}
[data-testid="stFileUploaderDropzoneInstructions"] div span {{
    color: {t['t1']} !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}}
[data-testid="stFileUploaderDropzoneInstructions"] div small {{
    color: {t['t2']} !important;
    font-size: 11px !important;
}}

[data-testid="stProgressBar"] > div > div {{
    background: linear-gradient(90deg, {t['acc']}, {t['acc_l']}) !important;
    border-radius: 4px !important;
}}

[data-testid="stDataFrame"] {{
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid {t['border']};
}}

[data-testid="stTextInput"] label {{ color: {t['t1']} !important; }}
[data-testid="stTextInput"] input {{
    background: {t['card']} !important;
    border: 1px solid {t['border']} !important;
    color: {t['t1']} !important;
    border-radius: 8px !important;
}}

[data-testid="stSlider"] label,
[data-testid="stSelectSlider"] label {{ color: {t['t1']} !important; }}
[data-testid="stToggle"] p {{ color: {t['t1']} !important; }}
[data-testid="stRadio"] label {{ color: {t['t1']} !important; }}
[data-testid="stNumberInput"] label {{ color: {t['t1']} !important; }}

[data-testid="stAlert"] {{ border-radius: 10px !important; }}
hr {{ border-color: {t['border']} !important; }}
h1, h2, h3, h4 {{ color: {t['t1']} !important; }}
p, .stMarkdown p {{ color: {t['t2']}; }}
</style>
"""


# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DefectVision Pro",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── resolve theme before any rendering ───────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

T = THEMES[st.session_state.theme]
st.markdown(make_css(T), unsafe_allow_html=True)

# ── backend helpers ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model...")
def load_model(weights_path: str) -> YOLO:
    return YOLO(weights_path)


def build_palette(n: int) -> list[tuple[int, int, int]]:
    cmap = plt.get_cmap("tab20")
    return [
        (int(b * 255), int(g * 255), int(r * 255))
        for r, g, b, _ in (cmap(i % 20) for i in range(n))
    ]


def post_process_mask(
    raw_mask: np.ndarray,
    target_w: int,
    target_h: int,
    threshold: float = 0.45,
) -> tuple[np.ndarray, np.ndarray | None]:
    soft = cv2.resize(raw_mask, (target_w, target_h),
                      interpolation=cv2.INTER_LINEAR).astype(np.float32)
    soft = cv2.GaussianBlur(soft, (7, 7), sigmaX=2.0)
    binary = (soft > threshold).astype(np.uint8) * 255

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)

    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_cc > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        binary = np.where(labels == largest, np.uint8(255), np.uint8(0))

    cnts_raw, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts_raw:
        return (binary > 0).astype(np.uint8), None

    outer = max(cnts_raw, key=cv2.contourArea)
    filled = np.zeros((target_h, target_w), dtype=np.uint8)
    cv2.drawContours(filled, [outer], -1, 255, cv2.FILLED)
    perimeter = cv2.arcLength(outer, True)
    smooth_cnt = cv2.approxPolyDP(outer, 0.002 * perimeter, True)
    return (filled > 0).astype(np.uint8), smooth_cnt


def extract_product_mask(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    bw   = max(8, min(h, w) // 15)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0].astype(np.uint8)).astype(np.float32)

    border_pixels = np.concatenate([
        lab[:bw, :].reshape(-1, 3), lab[-bw:, :].reshape(-1, 3),
        lab[:, :bw].reshape(-1, 3), lab[:, -bw:].reshape(-1, 3),
    ])
    bg_lab    = np.median(border_pixels, axis=0)
    bg_chroma = float(np.sqrt((bg_lab[1] - 128) ** 2 + (bg_lab[2] - 128) ** 2))

    diff    = lab - bg_lab
    dist    = np.sqrt((diff * diff).sum(axis=2))
    dist_u8 = np.clip(dist / (dist.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
    _, fg   = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if bg_chroma < 15:
        ab_dist = np.sqrt(diff[:, :, 1] ** 2 + diff[:, :, 2] ** 2)
        l_dist  = np.abs(diff[:, :, 0])
        fg[(fg == 255) & (ab_dist < 8) & (l_dist < 30)] = 0

    coverage = fg.sum() / 255 / (h * w)

    if bg_chroma > 15 or not (0.05 < coverage < 0.85):
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=15)
        edges   = cv2.Canny(blurred, 15, 50)
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        closed  = cv2.dilate(edges, k_close, iterations=2)
        tmp     = closed.copy()
        step    = max(1, min(h, w) // 20)
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

    k_open_pre = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k_open_pre)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fg = np.where(labels == largest, np.uint8(255), np.uint8(0))

    k_close  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE,  k_close)
    fg = cv2.morphologyEx(fg, cv2.MORPH_DILATE, k_dilate)
    return (fg > 0).astype(np.uint8)


def annotate_frame(
    frame: np.ndarray,
    result,
    class_names: dict,
    conf_thr: float,
    product_mask: np.ndarray | None = None,
    mask_overlap_thr: float = 0.20,
) -> tuple[np.ndarray, list[dict]]:
    detections = []

    if product_mask is not None:
        pm_cnts, _ = cv2.findContours(product_mask, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
        if pm_cnts:
            pm_outer  = max(pm_cnts, key=cv2.contourArea)
            pm_perim  = cv2.arcLength(pm_outer, True)
            pm_smooth = cv2.approxPolyDP(pm_outer, 0.005 * pm_perim, True)
            cv2.polylines(frame, [pm_smooth], isClosed=True, color=(255, 0, 0),
                          thickness=2, lineType=cv2.LINE_AA)

    if result.masks is None:
        return frame, detections

    masks_data = result.masks.data.cpu().numpy()
    boxes      = result.boxes
    h, w       = frame.shape[:2]
    overlay    = frame.copy()

    for i in range(len(boxes)):
        conf = float(boxes.conf[i].item())
        if conf < conf_thr:
            continue

        raw_mask = masks_data[i]
        mask_bin, smooth_cnt = post_process_mask(raw_mask, w, h)

        if product_mask is not None:
            det_area = int(mask_bin.sum())
            if det_area > 0 and int((mask_bin & product_mask).sum()) / det_area < mask_overlap_thr:
                continue

        cls_id = int(boxes.cls[i].item())
        name   = class_names.get(cls_id, f"class_{cls_id}")
        color  = (0, 0, 255)

        detections.append({"class": name, "class_id": cls_id, "conf": conf})

        overlay[mask_bin == 1] = color
        if smooth_cnt is not None:
            cv2.polylines(frame, [smooth_cnt], isClosed=True, color=color,
                          thickness=2, lineType=cv2.LINE_AA)
        else:
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, color, 2)

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
        cv2.rectangle(frame, (lx - 4, ly - lh - bl - 4), (lx + lw + 4, ly + bl + 2),
                      (10, 10, 30), -1)
        cv2.rectangle(frame, (lx - 4, ly - lh - bl - 4), (lx + lw + 4, ly + bl + 2),
                      (0, 0, 180), 1)
        cv2.putText(frame, label, (lx, ly - bl // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    bh = 44
    if detections:
        unique = list(dict.fromkeys(d["class"] for d in detections))
        banner = f"  ANOMALY  |  {', '.join(unique).upper()}"
        bw_px  = min(len(banner) * 13 + 20, w)
        cv2.rectangle(frame, (0, 0), (bw_px, bh), (10, 10, 140), -1)
        cv2.rectangle(frame, (0, 0), (bw_px, bh), (0, 0, 210), 1)
        cv2.putText(frame, banner, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (0, 0), (240, bh), (10, 50, 20), -1)
        cv2.rectangle(frame, (0, 0), (240, bh), (0, 170, 60), 1)
        cv2.putText(frame, "  PASS — NORMAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), 2, cv2.LINE_AA)

    return frame, detections


def bgr_to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def tiled_inference(
    frame: np.ndarray,
    model,
    rows: int,
    cols: int,
    imgsz: int,
    conf_thr: float,
    iou_thr: float,
    device: str,
    class_names: dict,
) -> tuple[np.ndarray, list[dict]]:
    h, w    = frame.shape[:2]
    result  = frame.copy()
    overlay = frame.copy()
    cell_results: list[dict] = []
    contour_list: list[tuple] = []
    label_list:   list[tuple] = []

    for r in range(rows):
        for c in range(cols):
            pid = r * cols + c + 1
            ty1 = r * h // rows;  ty2 = (r + 1) * h // rows
            tx1 = c * w // cols;  tx2 = (c + 1) * w // cols
            th  = ty2 - ty1;      tw  = tx2 - tx1

            tile    = frame[ty1:ty2, tx1:tx2]
            results = model.predict(tile, imgsz=imgsz, conf=conf_thr, iou=iou_thr,
                                    device=device, save=False, verbose=False)
            res = results[0]
            cell_dets: list[dict] = []

            if res.masks is not None:
                masks_data = res.masks.data.cpu().numpy()
                boxes      = res.boxes
                for i in range(len(boxes)):
                    conf   = float(boxes.conf[i].item())
                    cls_id = int(boxes.cls[i].item())
                    name   = class_names.get(cls_id, f"class_{cls_id}")
                    color  = (0, 0, 255)
                    cell_dets.append({"class": name, "class_id": cls_id, "conf": conf})

                    raw_mask      = masks_data[i]
                    tile_mask_bin, tile_cnt = post_process_mask(raw_mask, tw, th)
                    full_mask = np.zeros((h, w), dtype=np.uint8)
                    full_mask[ty1:ty2, tx1:tx2] = tile_mask_bin
                    overlay[full_mask == 1] = color

                    if tile_cnt is not None:
                        full_cnt = tile_cnt + np.array([[[tx1, ty1]]])
                        contour_list.append(([full_cnt], color))
                    else:
                        cnts, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
                        contour_list.append((cnts, color))

                    M = cv2.moments(full_mask)
                    if M["m00"] > 0:
                        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    else:
                        bx1, by1, bx2, by2 = boxes.xyxy[i].cpu().numpy().astype(int)
                        cx = tx1 + (bx1 + bx2) // 2
                        cy = ty1 + (by1 + by2) // 2

                    det_label = f"{name}  {conf*100:.0f}%"
                    (lw_, lh_), bl_ = cv2.getTextSize(det_label,
                                                      cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
                    lx = max(tx1, min(cx - lw_ // 2, tx2 - lw_ - 4))
                    ly = max(ty1 + lh_ + bl_ + 4, min(cy, ty2 - 4))
                    label_list.append((det_label, lx, ly, color))

            is_anomaly   = len(cell_dets) > 0
            border_color = (0, 0, 200) if is_anomaly else (0, 160, 60)
            chip_color   = (10, 10, 120) if is_anomaly else (10, 50, 20)
            status_text  = (
                f"P{pid}: {', '.join(dict.fromkeys(d['class'] for d in cell_dets))} "
                f"{max((d['conf'] for d in cell_dets), default=0)*100:.0f}%"
                if is_anomaly else f"P{pid}: PASS"
            )
            cv2.rectangle(result, (tx1, ty1), (tx2 - 1, ty2 - 1), border_color, 2)
            cv2.putText(result, f"P{pid}", (tx1 + 6, ty1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
            (sw, sh), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cx1 = tx1 + 4;  cy2 = ty2 - 4;  cy1 = cy2 - sh - 10
            cv2.rectangle(result, (cx1, cy1), (cx1 + sw + 12, cy2), chip_color, -1)
            cv2.rectangle(result, (cx1, cy1), (cx1 + sw + 12, cy2), border_color, 1)
            cv2.putText(result, status_text, (cx1 + 6, cy1 + sh + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

            cell_results.append({
                "product_id": pid, "cell": (r, c),
                "bbox": (tx1, ty1, tx2, ty2),
                "detections": cell_dets, "is_anomaly": is_anomaly,
            })

    cv2.addWeighted(overlay, 0.35, result, 0.65, 0, result)
    for cnts, color in contour_list:
        cv2.drawContours(result, cnts, -1, color, 2)
    for det_label, lx, ly, color in label_list:
        (lw_, lh_), bl_ = cv2.getTextSize(det_label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
        cv2.rectangle(result, (lx - 3, ly - lh_ - bl_ - 3), (lx + lw_ + 3, ly + bl_),
                      (10, 10, 30), -1)
        cv2.putText(result, det_label, (lx, ly - bl_ // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    return result, cell_results


def detection_stats(detections: list[dict]) -> None:
    if not detections:
        st.markdown(
            f'<div style="background:{T["pass_bg"]};border:1px solid {T["pass_border"]};'
            f'border-radius:10px;padding:14px 20px;color:{T["pass_fg"]};font-weight:700;'
            f'font-size:13px;letter-spacing:1.5px;">PASS &nbsp;·&nbsp; No anomalies detected</div>',
            unsafe_allow_html=True,
        )
        return

    counts: dict[str, int]   = {}
    best:   dict[str, float] = {}
    for d in detections:
        counts[d["class"]] = counts.get(d["class"], 0) + 1
        best[d["class"]]   = max(best.get(d["class"], 0.0), d["conf"])

    cols = st.columns(len(counts))
    for col, (name, cnt) in zip(cols, counts.items()):
        col.metric(label=name.upper(), value=cnt, delta=f"{best[name]*100:.0f}% conf")

    fig, ax = plt.subplots(figsize=(max(4, len(counts) * 1.5), 3.4))
    names  = list(counts.keys())
    values = [best[n] * 100 for n in names]
    bars   = ax.bar(names, values, color=T["bar_c"], edgecolor=T["bar_e"],
                    linewidth=0.8, width=0.45)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Max confidence (%)", color=T["t2"], fontsize=10)
    ax.set_title("Defect confidence by class", color=T["t1"],
                 fontsize=11, fontweight="bold", pad=14)
    ax.bar_label(bars, fmt="%.0f%%", padding=5, fontsize=10,
                 color=T["t1"], fontweight="bold")
    ax.set_facecolor(T["chart_bg"])
    fig.patch.set_facecolor(T["chart_bg"])
    ax.tick_params(colors=T["t2"], labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(T["border"])
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=T["grid_c"], linewidth=0.8)
    ax.xaxis.grid(False)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _inf_pill(ms: float, extra: str = "") -> str:
    return (
        f'<div style="background:{T["inf_bg"]};border:1px solid {T["inf_border"]};'
        f'border-radius:8px;padding:8px 14px;font-size:12px;color:{T["inf_fg"]};'
        f'display:inline-block;margin-bottom:16px;">'
        f'Inference &nbsp;{ms:.0f} ms{extra}</div>'
    )


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    # Brand bar
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{T['brand_g1']},{T['brand_g2']});
         border-radius:12px;padding:20px 18px;margin-bottom:4px;
         border:1px solid {T['brand_border']};">
      <div style="font-size:16px;font-weight:800;color:{T['brand_name']};
           letter-spacing:2.5px;text-transform:uppercase;">DefectVision</div>
      <div style="font-size:10px;color:{T['brand_sub']};margin-top:5px;
           letter-spacing:1px;font-weight:500;">PRO &nbsp;·&nbsp; AI Quality Control</div>
    </div>
    """, unsafe_allow_html=True)

    # Theme toggle
    st.markdown(
        f'<p style="font-size:9px;color:{T["section_c"]};font-weight:700;'
        f'letter-spacing:2.5px;text-transform:uppercase;margin:16px 0 4px 0;">Theme</p>',
        unsafe_allow_html=True,
    )
    chosen = st.radio("Theme", ["Dark", "Light"], horizontal=True,
                      index=0 if st.session_state.theme == "Dark" else 1,
                      label_visibility="collapsed")
    if chosen != st.session_state.theme:
        st.session_state.theme = chosen
        st.rerun()

    # Model
    st.markdown(
        f'<p style="font-size:9px;color:{T["section_c"]};font-weight:700;'
        f'letter-spacing:2.5px;text-transform:uppercase;margin:18px 0 6px 0;">Model</p>',
        unsafe_allow_html=True,
    )
    weights_input = st.text_input("Weights path", value=DEFAULT_WEIGHTS,
                                  label_visibility="collapsed")
    weights_ok = Path(weights_input).exists()
    if not weights_ok:
        st.error("Weights file not found")
    else:
        st.success("Model ready")

    # Detection
    st.markdown(
        f'<p style="font-size:9px;color:{T["section_c"]};font-weight:700;'
        f'letter-spacing:2.5px;text-transform:uppercase;margin:18px 0 6px 0;">Detection</p>',
        unsafe_allow_html=True,
    )
    conf_thr = st.slider("Confidence threshold", 0.05, 0.95, CONF_DEFAULT, 0.05)
    iou_thr  = st.slider("IoU (NMS) threshold",  0.10, 0.95, IOU_DEFAULT,  0.05)
    imgsz    = st.select_slider("Inference size",
                                options=[320, 480, 640, 800, 1024], value=IMGSZ_DEFAULT)

    # Isolation
    st.markdown(
        f'<p style="font-size:9px;color:{T["section_c"]};font-weight:700;'
        f'letter-spacing:2.5px;text-transform:uppercase;margin:18px 0 6px 0;">Isolation</p>',
        unsafe_allow_html=True,
    )
    isolate = st.toggle(
        "Product outline", value=True,
        help="Segments the product shape and drops detections outside it.",
    )
    mask_overlap_thr = st.slider(
        "Min. product overlap", 0.00, 0.50, 0.20, 0.05, disabled=not isolate,
        help="Lower (~0.05) for dark products on dark backgrounds.",
    )

    # System
    st.markdown(
        f'<p style="font-size:9px;color:{T["section_c"]};font-weight:700;'
        f'letter-spacing:2.5px;text-transform:uppercase;margin:18px 0 6px 0;">System</p>',
        unsafe_allow_html=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_label = device
    if device == "cuda":
        device_label = f"cuda  {torch.cuda.get_device_name(0)}"
    st.markdown(
        f'<div style="background:{T["card"]};border:1px solid {T["border"]};'
        f'border-radius:8px;padding:10px 14px;font-size:12px;">'
        f'<span style="color:{T["t2"]};font-size:9px;letter-spacing:1.5px;'
        f'text-transform:uppercase;">Device</span><br>'
        f'<span style="color:{T["t1"]};font-weight:600;">{device_label}</span></div>',
        unsafe_allow_html=True,
    )

# ── model load ────────────────────────────────────────────────────────────────

if not weights_ok:
    st.warning("Set a valid weights path in the sidebar to continue.")
    st.stop()

model       = load_model(weights_input)
class_names = dict(model.names)
n_classes   = max(class_names.keys()) + 1
palette     = build_palette(n_classes)

# ── header ────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div style="padding:24px 0 20px 0;border-bottom:1px solid {T['head_border']};
     margin-bottom:28px;">
  <span style="font-size:10px;color:{T['head_eyebrow']};font-weight:700;
  letter-spacing:3px;text-transform:uppercase;">Industrial Vision AI</span>
  <h1 style="margin:6px 0 0 0;font-size:26px;font-weight:800;
  color:{T['head_title']};letter-spacing:0.3px;">Anomaly Detection Platform</h1>
  <div style="margin-top:10px;display:flex;gap:24px;flex-wrap:wrap;">
    <span style="font-size:11px;color:{T['t2']};">
      Model &nbsp;<code style="background:{T['code_bg']};color:{T['code_fg']};
      padding:2px 6px;border-radius:4px;font-size:10px;">{Path(weights_input).name}</code>
    </span>
    <span style="font-size:11px;color:{T['t2']};">
      Classes &nbsp;<code style="background:{T['code_bg']};color:{T['code_fg']};
      padding:2px 6px;border-radius:4px;font-size:10px;">{', '.join(class_names.values())}</code>
    </span>
    <span style="font-size:11px;color:{T['t2']};">
      Conf &nbsp;<code style="background:{T['code_bg']};color:{T['code_fg']};
      padding:2px 6px;border-radius:4px;font-size:10px;">{conf_thr}</code>
    </span>
    <span style="font-size:11px;color:{T['t2']};">
      IoU &nbsp;<code style="background:{T['code_bg']};color:{T['code_fg']};
      padding:2px 6px;border-radius:4px;font-size:10px;">{iou_thr}</code>
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── tabs ──────────────────────────────────────────────────────────────────────

tab_img, tab_vid, tab_cam = st.tabs(["Image", "Video", "Webcam snapshot"])

# ── IMAGE tab ─────────────────────────────────────────────────────────────────

with tab_img:
    # ── Upload zone ──────────────────────────────────────────────────────────
    st.markdown(
        f'<p style="font-size:9px;color:{T["section_c"]};font-weight:700;'
        f'letter-spacing:2.5px;text-transform:uppercase;margin:8px 0 10px 0;">'
        f'Upload Images</p>',
        unsafe_allow_html=True,
    )
    uploaded_imgs = st.file_uploader(
        "Drop PNG / JPG images here or click to browse — multiple files supported",
        type=["png", "jpg", "jpeg"], key="img_up",
        accept_multiple_files=True,
    )

    for img_idx, uploaded_img in enumerate(uploaded_imgs or []):
        if len(uploaded_imgs) > 1:
            st.markdown(
                f'<div style="border-top:1px solid {T["border"]};margin:28px 0 16px 0;'
                f'padding-top:16px;display:flex;align-items:center;gap:10px;">'
                f'<span style="font-size:9px;color:{T["section_c"]};font-weight:700;'
                f'letter-spacing:2px;text-transform:uppercase;">Image</span>'
                f'<span style="font-size:12px;color:{T["t1"]};font-weight:600;">'
                f'{uploaded_img.name}</span></div>',
                unsafe_allow_html=True,
            )

        file_bytes = np.frombuffer(uploaded_img.read(), np.uint8)
        frame_raw  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        frame_bgr  = frame_raw

        with st.spinner("Running inference..."):
            t0        = time.time()
            prod_mask = extract_product_mask(frame_bgr) if isolate else None
            results   = model.predict(frame_bgr, imgsz=imgsz, conf=conf_thr,
                                      iou=iou_thr, device=device,
                                      save=False, verbose=False)
            elapsed   = time.time() - t0

        annotated, detections = annotate_frame(
            frame_bgr.copy(), results[0], class_names, conf_thr,
            product_mask=prod_mask, mask_overlap_thr=mask_overlap_thr,
        )

        col_orig, col_ann = st.columns(2)
        with col_orig:
            st.markdown(
                f'<p style="font-size:10px;font-weight:700;letter-spacing:1.5px;'
                f'text-transform:uppercase;color:{T["t2"]};margin-bottom:6px;">Original</p>',
                unsafe_allow_html=True,
            )
            st.image(bgr_to_pil(frame_raw), use_column_width=True)
        with col_ann:
            st.markdown(
                f'<p style="font-size:10px;font-weight:700;letter-spacing:1.5px;'
                f'text-transform:uppercase;color:{T["acc_l"]};margin-bottom:6px;">Annotated</p>',
                unsafe_allow_html=True,
            )
            st.image(bgr_to_pil(annotated), use_column_width=True)

        st.markdown(_inf_pill(elapsed * 1000), unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-size:9px;color:{T["section_c"]};font-weight:700;'
            f'letter-spacing:2.5px;text-transform:uppercase;margin:18px 0 10px 0;">'
            f'Detection Results</p>',
            unsafe_allow_html=True,
        )
        detection_stats(detections)

        buf = io.BytesIO()
        bgr_to_pil(annotated).save(buf, format="PNG")
        st.download_button("Download annotated image", buf.getvalue(),
                           file_name=f"annotated_{uploaded_img.name}",
                           mime="image/png", key=f"dl_{img_idx}")

# ── VIDEO tab ─────────────────────────────────────────────────────────────────

with tab_vid:
    uploaded_vid = st.file_uploader(
        "Upload a video (MP4 / AVI / MOV)", type=["mp4", "avi", "mov"], key="vid_up"
    )

    if uploaded_vid:
        suffix = Path(uploaded_vid.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_vid.read())
            tmp_path = tmp.name

        cap      = cv2.VideoCapture(tmp_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w_src    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_src    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        st.markdown(
            f'<div style="background:{T["card"]};border:1px solid {T["border"]};'
            f'border-radius:10px;padding:14px 18px;margin-bottom:12px;">'
            f'<span style="color:{T["t1"]};font-weight:600;">{uploaded_vid.name}</span>'
            f'<span style="color:{T["t2"]};font-size:12px;margin-left:16px;">'
            f'{n_frames} frames &nbsp;|&nbsp; {src_fps:.0f} fps &nbsp;|&nbsp; '
            f'{w_src}x{h_src}</span></div>',
            unsafe_allow_html=True,
        )

        step = st.slider("Process every N-th frame", min_value=1, max_value=10, value=1,
                         help="2-5 for faster processing; 1 for every frame.")

        col_btn1, _ = st.columns([1, 4])
        run_video   = col_btn1.button("Run detection", type="primary")

        if run_video:
            cap     = cv2.VideoCapture(tmp_path)
            out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            out_tmp.close()
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_tmp.name, fourcc, src_fps, (w_src, h_src))

            frame_ph  = st.empty()
            prog_bar  = st.progress(0, text="Processing...")
            all_dets2: list[dict] = []
            frame_idx = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % step == 0:
                    results = model.predict(frame, imgsz=imgsz, conf=conf_thr,
                                            iou=iou_thr, device=device,
                                            save=False, verbose=False)
                    annotated, dets = annotate_frame(
                        frame.copy(), results[0], class_names, conf_thr
                    )
                    all_dets2.extend(dets)
                    last_annotated = annotated
                else:
                    last_annotated = frame

                writer.write(last_annotated)
                pct = min((frame_idx + 1) / max(n_frames, 1), 1.0)
                prog_bar.progress(pct, text=f"Frame {frame_idx+1} / {n_frames}")
                if frame_idx % max(step, 3) == 0:
                    frame_ph.image(bgr_to_pil(last_annotated),
                                   caption=f"Frame {frame_idx+1}",
                                   use_column_width=True)
                frame_idx += 1

            cap.release()
            writer.release()
            prog_bar.empty()
            frame_ph.empty()

            st.success(f"Done — {frame_idx} frames processed.")
            st.markdown("#### Overall detections")
            detection_stats(all_dets2)

            with open(out_tmp.name, "rb") as f:
                st.download_button("Download annotated video", f.read(),
                                   file_name="annotated_output.mp4", mime="video/mp4")

# ── WEBCAM tab ────────────────────────────────────────────────────────────────

def _run_inference_on_frame(frame_bgr: np.ndarray) -> None:
    frame_raw = frame_bgr.copy()
    with st.spinner("Running inference..."):
        t0        = time.time()
        prod_mask = extract_product_mask(frame_bgr) if isolate else None
        results   = model.predict(frame_bgr, imgsz=imgsz, conf=conf_thr,
                                  iou=iou_thr, device=device,
                                  save=False, verbose=False)
        elapsed   = time.time() - t0
    annotated, detections = annotate_frame(
        frame_bgr.copy(), results[0], class_names, conf_thr,
        product_mask=prod_mask, mask_overlap_thr=mask_overlap_thr,
    )
    col_orig, col_ann = st.columns(2)
    with col_orig:
        st.markdown(
            f'<p style="font-size:10px;font-weight:700;letter-spacing:1.5px;'
            f'text-transform:uppercase;color:{T["t2"]};margin-bottom:6px;">Original</p>',
            unsafe_allow_html=True,
        )
        st.image(bgr_to_pil(frame_raw), use_column_width=True)
    with col_ann:
        st.markdown(
            f'<p style="font-size:10px;font-weight:700;letter-spacing:1.5px;'
            f'text-transform:uppercase;color:{T["acc_l"]};margin-bottom:6px;">Annotated</p>',
            unsafe_allow_html=True,
        )
        st.image(bgr_to_pil(annotated), use_column_width=True)
    st.markdown(_inf_pill(elapsed * 1000), unsafe_allow_html=True)
    st.markdown(
        f'<p style="font-size:9px;color:{T["section_c"]};font-weight:700;'
        f'letter-spacing:2.5px;text-transform:uppercase;margin:18px 0 10px 0;">'
        f'Detection Results</p>',
        unsafe_allow_html=True,
    )
    detection_stats(detections)


with tab_cam:
    # ── Mode selector ─────────────────────────────────────────────────────────
    st.markdown(
        f'<p style="font-size:9px;color:{T["section_c"]};font-weight:700;'
        f'letter-spacing:2.5px;text-transform:uppercase;margin:8px 0 10px 0;">'
        f'Camera Source</p>',
        unsafe_allow_html=True,
    )
    cam_mode = st.radio(
        "Camera source",
        ["Browser / Smartphone", "Connected camera (USB / external)"],
        label_visibility="collapsed",
    )

    # ── MODE 1 : Browser camera ───────────────────────────────────────────────
    if cam_mode == "Browser / Smartphone":
        st.markdown(
            f'<div style="background:{T["card"]};border:1px solid {T["border"]};'
            f'border-radius:10px;padding:14px 18px;margin-bottom:16px;">'
            f'<div style="font-size:13px;font-weight:600;color:{T["t1"]};">Browser camera</div>'
            f'<div style="font-size:11px;color:{T["t2"]};margin-top:4px;">'
            f'Works on any device that opens this app — laptop webcam, tablet, or smartphone.'
            f' On a phone, open the Network URL shown in the terminal and tap the capture button.'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        cam_image = st.camera_input("Capture", label_visibility="collapsed")
        if cam_image:
            file_bytes = np.frombuffer(cam_image.read(), np.uint8)
            frame_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            _run_inference_on_frame(frame_bgr)

    # ── MODE 2 : Connected camera (OpenCV) ────────────────────────────────────
    else:
        st.markdown(
            f'<div style="background:{T["card"]};border:1px solid {T["border"]};'
            f'border-radius:10px;padding:14px 18px;margin-bottom:16px;">'
            f'<div style="font-size:13px;font-weight:600;color:{T["t1"]};">Connected camera</div>'
            f'<div style="font-size:11px;color:{T["t2"]};margin-top:4px;">'
            f'Captures directly from a camera connected to this machine via USB or driver '
            f'(e.g. DroidCam, EpocCam, virtual camera). '
            f'Device 0 = built-in, 1 = first external, 2 = second external, etc.'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        dev_col, btn_col = st.columns([2, 3])
        cam_idx  = dev_col.number_input("Device index", min_value=0, max_value=10,
                                        value=0, step=1,
                                        help="0 = built-in webcam, 1 = first USB camera, etc.")
        capture  = btn_col.button("Capture snapshot", type="primary",
                                  use_container_width=True)

        if capture:
            cap = cv2.VideoCapture(int(cam_idx))
            if not cap.isOpened():
                st.error(f"Could not open camera device {int(cam_idx)}. "
                         f"Check the index or make sure the camera is connected.")
            else:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ok, frame = cap.read()
                cap.release()
                if not ok or frame is None:
                    st.error("Camera opened but failed to capture a frame. Try again.")
                else:
                    _run_inference_on_frame(frame)
