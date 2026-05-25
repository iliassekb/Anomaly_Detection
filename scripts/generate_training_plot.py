import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

df = pd.read_csv(r"c:\Users\imqd\Desktop\myAPPS\pfe\results.csv")
df.columns = df.columns.str.strip()

epochs = df["epoch"].values

BLUE      = "#4682C3"
BLUEDARK  = "#234891"
GREEN     = "#27AE60"
ORANGE    = "#E67E22"
RED       = "#C0392B"
PURPLE    = "#8E44AD"
TEAL      = "#16A085"
GRAY      = "#7F8C8D"
BG        = "#F5F7FA"
GRID      = "#D5E4F5"

plots = [
    # (title, [(col, label, color), ...], ylabel)
    ("Train — Box Loss",
     [("train/box_loss", "train", BLUE),
      ("val/box_loss",   "val",   ORANGE)],
     "Loss"),

    ("Train — Seg Loss",
     [("train/seg_loss", "train", BLUE),
      ("val/seg_loss",   "val",   ORANGE)],
     "Loss"),

    ("Train — Cls Loss",
     [("train/cls_loss", "train", BLUE),
      ("val/cls_loss",   "val",   ORANGE)],
     "Loss"),

    ("Train — DFL Loss",
     [("train/dfl_loss", "train", BLUE)],
     "Loss"),

    ("Mask — Précision & Rappel",
     [("metrics/precision(M)", "Precision", GREEN),
      ("metrics/recall(M)",    "Recall",    RED)],
     "Score"),

    ("Mask — mAP@0.5",
     [("metrics/mAP50(M)", "mAP@0.5 (Mask)", BLUEDARK)],
     "mAP"),

    ("Mask — mAP@0.5:0.95",
     [("metrics/mAP50-95(M)", "mAP@0.5:0.95 (Mask)", PURPLE)],
     "mAP"),

    ("Learning Rate",
     [("lr/pg0", "pg0", TEAL)],
     "LR"),
]

fig = plt.figure(figsize=(18, 10), facecolor=BG)
fig.suptitle(
    "YOLO26s-seg — Courbes d'Entraînement (MVTec 3D-AD, 17 epochs)",
    fontsize=14, fontweight="bold", color=BLUEDARK, y=0.98
)

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35,
                       left=0.06, right=0.98, top=0.92, bottom=0.08)

for idx, (title, series, ylabel) in enumerate(plots):
    row, col = divmod(idx, 4)
    ax = fig.add_subplot(gs[row, col])
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="--", alpha=0.7)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

    for col_name, label, color in series:
        if col_name in df.columns:
            y = df[col_name].values
            ax.plot(epochs, y, color=color, linewidth=1.8,
                    label=label, marker="o", markersize=3.5,
                    markerfacecolor="white", markeredgewidth=1.2,
                    markeredgecolor=color)
            # best value annotation
            best_idx = np.argmin(y) if "loss" in col_name.lower() or "lr" in col_name.lower() else np.argmax(y)
            ax.axvline(x=epochs[best_idx], color=color, linewidth=0.8,
                       linestyle=":", alpha=0.5)

    ax.set_title(title, fontsize=9, fontweight="bold", color=BLUEDARK, pad=4)
    ax.set_xlabel("Epoch", fontsize=8, color=GRAY)
    ax.set_ylabel(ylabel, fontsize=8, color=GRAY)
    ax.tick_params(labelsize=7, colors=GRAY)
    ax.set_xlim(epochs[0] - 0.3, epochs[-1] + 0.3)

    if len(series) > 1:
        ax.legend(fontsize=7, framealpha=0.85, edgecolor=GRID,
                  facecolor=BG, loc="best")

out_path = r"c:\Users\imqd\Desktop\myAPPS\pfe\photos\results_training.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out_path}")
plt.close()
