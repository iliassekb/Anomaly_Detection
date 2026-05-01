# Usage:
#   python scripts/make_test_video.py
#   python scripts/make_test_video.py --out my_test.mp4 --hold 40 --fps 30

"""
Assembles the YOLO bagel dataset images into a single test MP4 that you can
feed to realtime_detection.py:

    python scripts/realtime_detection.py --source test_video.mp4

Layout per frame
----------------
  - Image fills the frame at 800x800
  - Top-left corner: ground-truth category chip  (grey = good, yellow = defect)
  - Between categories: a 0.5-second black title card names the incoming type

Order: good → crack → hole → contamination → combined  (all train + val images)
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

# ── config ──────────────────────────────────────────────────────────────────
DATASET_ROOT = Path("c:/Users/imqd/Desktop/myAPPS/pfe/data/yolo_bagel/images")
OUTPUT_DEFAULT = Path("test_video.mp4")
FRAME_SIZE = (800, 800)   # width, height — matches IMG_SIZE used during training
FPS_DEFAULT = 30
HOLD_DEFAULT = 45          # frames to show each image  (45 / 30 fps ≈ 1.5 s)
FADE_FRAMES = 8            # cross-fade between consecutive images
TITLE_FRAMES = 20          # black title card between categories  (≈ 0.67 s)

# display order
CATEGORY_ORDER = ["good", "crack", "hole", "contamination", "combined"]

# per-category chip colours  (BGR)
CHIP_COLORS = {
    "good":          (80,  180,  80),   # green
    "crack":         (40,  100, 220),   # orange-red
    "hole":          (220, 100,  40),   # blue
    "contamination": (40,  200, 200),   # yellow
    "combined":      (200,  40, 200),   # magenta
}


# ── helpers ──────────────────────────────────────────────────────────────────

def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read {path}")
    return cv2.resize(img, FRAME_SIZE, interpolation=cv2.INTER_AREA)


def draw_gt_chip(frame: np.ndarray, category: str):
    color = CHIP_COLORS.get(category, (120, 120, 120))
    label = f"GT: {category}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    pad = 8
    cv2.rectangle(frame, (0, 0), (tw + pad * 2, th + pad * 2), color, -1)
    cv2.putText(frame, label, (pad, th + pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def make_title_card(category: str) -> np.ndarray:
    card = np.zeros((*FRAME_SIZE[::-1], 3), dtype=np.uint8)  # black
    color = CHIP_COLORS.get(category, (200, 200, 200))
    text = category.upper()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.2, 4)
    x = (FRAME_SIZE[0] - tw) // 2
    y = (FRAME_SIZE[1] + th) // 2
    cv2.putText(card, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                2.2, color, 4, cv2.LINE_AA)
    sub = "ground-truth category"
    (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    cv2.putText(card, sub, ((FRAME_SIZE[0] - sw) // 2, y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 1, cv2.LINE_AA)
    return card


def natural_sort_key(p: Path):
    parts = re.split(r"(\d+)", p.stem)
    return [int(x) if x.isdigit() else x.lower() for x in parts]


def collect_images(category: str) -> list[Path]:
    paths = []
    for split in ("train", "val"):
        d = DATASET_ROOT / split
        if d.exists():
            paths += sorted(
                (f for f in d.iterdir()
                 if f.stem.startswith(category + "_") or f.stem == category),
                key=natural_sort_key,
            )
    return paths


def write_frames(writer: cv2.VideoWriter, frames_iter):
    """Write an iterable of numpy frames to the VideoWriter."""
    for f in frames_iter:
        writer.write(f)


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out",   default=str(OUTPUT_DEFAULT), help="Output mp4 path")
    p.add_argument("--fps",   type=int, default=FPS_DEFAULT)
    p.add_argument("--hold",  type=int, default=HOLD_DEFAULT,
                   help="Frames to hold each image")
    p.add_argument("--max",   type=int, default=0,
                   help="Max images per class (0 = all)")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    fps      = args.fps
    hold     = args.hold

    # ── gather image list grouped by category ───────────────────────────────
    groups: dict[str, list[Path]] = {}
    total = 0
    for cat in CATEGORY_ORDER:
        imgs = collect_images(cat)
        if args.max > 0:
            imgs = imgs[:args.max]
        groups[cat] = imgs
        total += len(imgs)
        print(f"  {cat:>14s}: {len(imgs):3d} images")

    if total == 0:
        raise RuntimeError(f"No images found under {DATASET_ROOT}")

    print(f"\n  Total: {total} images -> ~{total * hold // fps}s video at {fps} fps")

    # ── video writer ─────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, FRAME_SIZE)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {out_path}")

    # ── build video ──────────────────────────────────────────────────────────
    prev_frame: np.ndarray | None = None
    total_written = 0

    for cat in CATEGORY_ORDER:
        paths = groups[cat]
        if not paths:
            continue

        # title card
        card = make_title_card(cat)
        for _ in range(TITLE_FRAMES):
            writer.write(card)
        total_written += TITLE_FRAMES

        for idx, img_path in enumerate(paths):
            try:
                img = load_image(img_path)
            except Exception as e:
                print(f"  [WARN] {e}")
                continue

            draw_gt_chip(img, cat)

            # cross-fade from previous frame
            if prev_frame is not None:
                for t in range(FADE_FRAMES):
                    alpha = t / FADE_FRAMES
                    blended = cv2.addWeighted(prev_frame, 1 - alpha, img, alpha, 0)
                    writer.write(blended)
                total_written += FADE_FRAMES

            # hold current image
            for _ in range(hold):
                writer.write(img)
            total_written += hold

            prev_frame = img.copy()
            print(f"  [{cat}] {idx+1}/{len(paths)}  {img_path.name}", end="\r")

        print()  # newline after category

    writer.release()
    duration = total_written / fps
    print(f"\n  Saved: {out_path}  ({total_written} frames, {duration:.1f}s)")
    print(f"\n  Run detection on it:")
    print(f"    python scripts/realtime_detection.py --source {out_path}")


if __name__ == "__main__":
    main()
