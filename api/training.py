"""
Training backend — fine-tunes best_m.pt with YOLOv8 segmentation,
tracked by MLflow. Background thread to avoid blocking the API.
"""

import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from api.minio_client import get_classes, get_object, list_objects

router = APIRouter(prefix="/api/training", tags=["training"])

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
BASE_WEIGHTS = os.getenv("MODEL_WEIGHTS", "weights/best_m.pt")
RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", "50"))

_training_state = {
    "running": False,
    "current_epoch": 0,
    "total_epochs": 0,
    "run_id": None,
    "error": None,
}
_state_lock = threading.Lock()


# ── helpers ───────────────────────────────────────────────────────

def _count_annotated_total() -> int:
    total = 0
    classes = get_classes()
    for cls in classes:
        total += len(list_objects(f"annotated/{cls}/images/"))
    return total


def _get_last_trained_count() -> int:
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name("DefectVision")
        if exp is None:
            return 0
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="tags.trained_on_count != ''",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            return 0
        return int(runs[0].data.tags.get("trained_on_count", 0))
    except Exception:
        return 0


def should_retrain() -> bool:
    current = _count_annotated_total()
    last = _get_last_trained_count()
    return (current - last) >= RETRAIN_THRESHOLD


def trigger_training_background() -> None:
    with _state_lock:
        if _training_state["running"]:
            return
        _training_state["running"] = True
        _training_state["error"] = None
    t = threading.Thread(target=_run_training, daemon=True)
    t.start()


def _run_training() -> None:
    tmp_dir = tempfile.mkdtemp(prefix="defectvision_train_")
    try:
        _do_training(tmp_dir)
    except Exception as e:
        with _state_lock:
            _training_state["error"] = str(e)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        with _state_lock:
            _training_state["running"] = False
            _training_state["current_epoch"] = 0
            _training_state["total_epochs"] = 0


def _do_training(tmp_dir: str) -> None:
    import mlflow
    from ultralytics import YOLO

    mlflow.set_tracking_uri(MLFLOW_URI)

    classes = get_classes()
    if not classes:
        raise RuntimeError("No classes in registry — nothing to train on")

    # Build sorted class list by id so YOLO indices align
    sorted_classes = sorted(classes.items(), key=lambda kv: kv[1])
    class_names = [name for name, _ in sorted_classes]
    num_classes = len(class_names)

    # Download annotated images + labels from MinIO
    train_img_dir = Path(tmp_dir) / "images" / "train"
    train_lbl_dir = Path(tmp_dir) / "labels" / "train"
    val_img_dir = Path(tmp_dir) / "images" / "val"
    val_lbl_dir = Path(tmp_dir) / "labels" / "val"
    for d in (train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir):
        d.mkdir(parents=True, exist_ok=True)

    total_images = 0
    for cls_name in class_names:
        img_keys = list_objects(f"annotated/{cls_name}/images/")
        for i, img_key in enumerate(img_keys):
            filename = Path(img_key).name
            # 80 / 20 split
            split = "val" if (i % 5 == 0) else "train"
            img_dir = val_img_dir if split == "val" else train_img_dir
            lbl_dir = val_lbl_dir if split == "val" else train_lbl_dir

            img_bytes = get_object(img_key)
            (img_dir / filename).write_bytes(img_bytes)

            stem = Path(filename).stem
            lbl_key = f"annotated/{cls_name}/labels/{stem}.txt"
            try:
                lbl_bytes = get_object(lbl_key)
                (lbl_dir / f"{stem}.txt").write_bytes(lbl_bytes)
            except Exception:
                pass
            total_images += 1

    if total_images == 0:
        raise RuntimeError("No annotated images found in MinIO")

    # Write data.yaml
    data_yaml = {
        "path": tmp_dir,
        "train": "images/train",
        "val": "images/val",
        "nc": num_classes,
        "names": {i: n for i, n in enumerate(class_names)},
    }
    yaml_path = Path(tmp_dir) / "data.yaml"
    yaml_path.write_text(yaml.dump(data_yaml))

    # Determine epoch count: 50 fine-tune epochs
    epochs = int(os.getenv("TRAIN_EPOCHS", "50"))

    with _state_lock:
        _training_state["total_epochs"] = epochs

    with mlflow.start_run(run_name=f"finetune_{int(time.time())}",
                          experiment_id=_get_or_create_experiment()) as run:
        run_id = run.info.run_id
        with _state_lock:
            _training_state["run_id"] = run_id

        mlflow.set_tag("trained_on_count", str(total_images))
        mlflow.log_params({
            "base_weights": BASE_WEIGHTS,
            "epochs": epochs,
            "num_classes": num_classes,
            "class_names": ",".join(class_names),
            "total_images": total_images,
        })

        model = YOLO(BASE_WEIGHTS)

        # Epoch callback to update progress state
        def on_epoch_end(trainer):
            with _state_lock:
                _training_state["current_epoch"] = trainer.epoch + 1
            metrics = trainer.metrics or {}
            mlflow.log_metrics(
                {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                step=trainer.epoch,
            )

        model.add_callback("on_train_epoch_end", on_epoch_end)

        results = model.train(
            data=str(yaml_path),
            task="segment",
            imgsz=800,
            epochs=epochs,
            batch=4,
            project=tmp_dir,
            name="run",
            exist_ok=True,
            verbose=False,
        )

        # Copy best weights
        best_pt = Path(tmp_dir) / "run" / "weights" / "best.pt"
        if best_pt.exists():
            # Versioned copy
            weights_dir = Path(BASE_WEIGHTS).parent
            existing = list(weights_dir.glob("best_m_v*.pt"))
            version = len(existing) + 1
            versioned = weights_dir / f"best_m_v{version}.pt"
            shutil.copy(str(best_pt), str(versioned))
            shutil.copy(str(best_pt), BASE_WEIGHTS)
            mlflow.log_artifact(str(versioned), artifact_path="weights")

        # Log final metrics
        if hasattr(results, "results_dict"):
            final = {k: float(v) for k, v in results.results_dict.items()
                     if isinstance(v, (int, float))}
            mlflow.log_metrics(final, step=epochs)


def _get_or_create_experiment() -> str:
    import mlflow
    exp = mlflow.get_experiment_by_name("DefectVision")
    if exp is not None:
        return exp.experiment_id
    return mlflow.create_experiment(
        "DefectVision",
        artifact_location=f"s3://defectvision-mlflow/",
    )


# ── endpoints ─────────────────────────────────────────────────────

@router.get("/status")
def get_status():
    with _state_lock:
        return dict(_training_state)


@router.post("/trigger", status_code=202)
def trigger_training():
    with _state_lock:
        if _training_state["running"]:
            raise HTTPException(409, "Training already in progress")
    trigger_training_background()
    return {"message": "Training started"}


@router.get("/runs")
def list_runs():
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name("DefectVision")
        if exp is None:
            return {"runs": []}
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=20,
        )
        result = []
        for r in runs:
            result.append({
                "run_id": r.info.run_id,
                "status": r.info.status,
                "start_time": r.info.start_time,
                "end_time": r.info.end_time,
                "metrics": r.data.metrics,
                "params": r.data.params,
                "tags": r.data.tags,
            })
        return {"runs": result}
    except Exception as e:
        raise HTTPException(503, f"MLflow unavailable: {e}")


@router.get("/runs/{run_id}")
def get_run(run_id: str):
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()
        r = client.get_run(run_id)
        return {
            "run_id": r.info.run_id,
            "status": r.info.status,
            "start_time": r.info.start_time,
            "end_time": r.info.end_time,
            "metrics": r.data.metrics,
            "params": r.data.params,
            "tags": r.data.tags,
        }
    except Exception as e:
        raise HTTPException(404, str(e))


@router.post("/runs/{run_id}/activate")
def activate_run(run_id: str):
    """Copy the weights from an MLflow run to weights/best_m.pt."""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id, path="weights")
        if not artifacts:
            raise HTTPException(404, "No weight artifacts in this run")
        # Download first .pt artifact
        pt_artifact = next((a for a in artifacts if a.path.endswith(".pt")), None)
        if pt_artifact is None:
            raise HTTPException(404, "No .pt artifact found")
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=pt_artifact.path,
            dst_path=str(Path(BASE_WEIGHTS).parent),
            tracking_uri=MLFLOW_URI,
        )
        shutil.copy(local_path, BASE_WEIGHTS)
        return {"activated": run_id, "weights": BASE_WEIGHTS}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
