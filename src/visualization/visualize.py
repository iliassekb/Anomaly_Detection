"""Visualize point clouds, anomaly heatmaps, and results."""
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def show_point_cloud(pcd: o3d.geometry.PointCloud, title: str = "Point Cloud"):
    o3d.visualization.draw_geometries([pcd], window_name=title)


def colorize_by_score(pcd: o3d.geometry.PointCloud, scores: np.ndarray) -> o3d.geometry.PointCloud:
    """Color each point by its anomaly score (low=blue, high=red)."""
    normed = (scores - scores.min()) / (scores.ptp() + 1e-8)
    colors = cm.jet(normed)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float, save_path: str = None):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curve")
    ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()
