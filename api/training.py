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
    "stop_requested": False,
    "current_metrics": {},
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
        _training_state["stop_requested"] = False
        _training_state["current_metrics"] = {}
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

    # Disable YOLO's built-in MLflow integration before the trainer is created.
    # YOLO checks ultralytics.settings["mlflow"] when registering integration
    # callbacks — setting it False is the only way to prevent it from opening
    # its own nested MLflow run and corrupting our active-run context.
    try:
        from ultralytics import settings as _yolo_settings
        _yolo_settings.update({"mlflow": False})
    except Exception:
        pass

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

    # Collect all (cls_name, img_key) pairs first so the val split is global
    all_pairs: list[tuple[str, str]] = []
    for cls_name in class_names:
        for img_key in list_objects(f"annotated/{cls_name}/images/"):
            all_pairs.append((cls_name, img_key))

    if not all_pairs:
        raise RuntimeError("No annotated images found in MinIO")

    # Only hold out a val split when there are enough images; otherwise
    # train on everything and point val at the same directory.
    use_val = len(all_pairs) >= 5
    total_images = 0

    for i, (cls_name, img_key) in enumerate(all_pairs):
        filename = Path(img_key).name
        in_val = use_val and (i % 5 == 0)
        img_dir = val_img_dir if in_val else train_img_dir
        lbl_dir = val_lbl_dir if in_val else train_lbl_dir

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

    # Write data.yaml — use images/train as val when dataset is too small
    val_dir_has_images = use_val and any(val_img_dir.iterdir())
    data_yaml = {
        "path": tmp_dir,
        "train": "images/train",
        "val": "images/val" if val_dir_has_images else "images/train",
        "nc": num_classes,
        "names": class_names,
    }
    yaml_path = Path(tmp_dir) / "data.yaml"
    yaml_path.write_text(yaml.dump(data_yaml))

    # Determine epoch count: 50 fine-tune epochs
    epochs = int(os.getenv("TRAIN_EPOCHS", "5"))

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

        model = YOLO("yolov8n-seg.pt")

        # Callback only tracks live UI state — no MLflow inside the callback.
        # MLflow metric history is written from results.csv after training ends.
        def on_epoch_end(trainer):
            import math
            raw: dict = dict(trainer.metrics or {})
            if hasattr(trainer, "label_loss_items") and getattr(trainer, "tloss", None) is not None:
                try:
                    raw.update(trainer.label_loss_items(trainer.tloss, prefix="train"))
                except Exception:
                    pass
            live = {k: round(float(v), 4) for k, v in raw.items()
                    if isinstance(v, (int, float)) and math.isfinite(float(v))}
            with _state_lock:
                _training_state["current_epoch"] = trainer.epoch + 1
                if live:
                    _training_state["current_metrics"] = live
                stop = _training_state["stop_requested"]
            if stop:
                trainer.stop = True

        model.add_callback("on_fit_epoch_end", on_epoch_end)

        # Pre-create run dir so YOLO's background plot thread doesn't race it
        (Path(tmp_dir) / "run").mkdir(parents=True, exist_ok=True)

        results = model.train(
            data=str(yaml_path),
            task="segment",
            imgsz=800,
            epochs=epochs,
            batch=4,
            project=tmp_dir,
            name="run",
            exist_ok=True,
            plots=False,
            verbose=False,
        )

        # ── Log per-epoch metrics from YOLO's results.csv ────────────
        import csv, math as _math

        # YOLO saves results.csv inside save_dir; use the trainer path when available
        save_dir = Path(getattr(getattr(model, "trainer", None), "save_dir", None) or Path(tmp_dir) / "run")
        results_csv = save_dir / "results.csv"

        # Fallback: search the whole tmp_dir tree
        if not results_csv.exists():
            found = list(Path(tmp_dir).rglob("results.csv"))
            if found:
                results_csv = found[0]

        print(f"[training] results.csv → {results_csv}  exists={results_csv.exists()}", flush=True)

        if results_csv.exists():
            try:
                with open(results_csv, newline="") as f:
                    content = f.read()
                print(f"[training] CSV preview:\n{content[:300]}", flush=True)
                for row in csv.DictReader(content.splitlines()):
                    row = {k.strip(): v.strip() for k, v in row.items()}
                    try:
                        step = int(float(row.pop("epoch")))
                    except (KeyError, ValueError):
                        continue
                    to_log = {}
                    for k, v in row.items():
                        try:
                            fv = float(v)
                            if _math.isfinite(fv):
                                # MLflow forbids parentheses: precision(B) → precision_B
                                safe_k = k.replace("(", "_").replace(")", "")
                                to_log[safe_k] = fv
                        except (ValueError, TypeError):
                            pass
                    if to_log:
                        mlflow.log_metrics(to_log, step=step)
                print(f"[training] CSV metrics logged OK ({results_csv.stem})", flush=True)
            except Exception as exc:
                import traceback
                print(f"[training] CSV logging failed: {exc}\n{traceback.format_exc()}", flush=True)

        # Copy best weights
        best_pt = Path(tmp_dir) / "run" / "weights" / "best.pt"
        if best_pt.exists():
            try:
                weights_dir = Path(BASE_WEIGHTS).parent
                existing = list(weights_dir.glob("best_m_v*.pt"))
                version = len(existing) + 1
                versioned = weights_dir / f"best_m_v{version}.pt"
                shutil.copy(str(best_pt), str(versioned))
                shutil.copy(str(best_pt), BASE_WEIGHTS)
            except Exception:
                pass


def _get_or_create_experiment() -> str:
    import mlflow
    exp = mlflow.get_experiment_by_name("DefectVision")
    if exp is not None:
        return exp.experiment_id
    return mlflow.create_experiment(
        "DefectVision",
        artifact_location=f"s3://defectvision-mlflow/",
    )


def cleanup_stale_runs() -> None:
    """On startup, mark any RUNNING MLflow runs as FAILED.
    They were interrupted by a container restart and will never complete."""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name("DefectVision")
        if exp is None:
            return
        stale = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="attributes.status = 'RUNNING'",
        )
        for run in stale:
            client.set_terminated(run.info.run_id, status="FAILED")
    except Exception:
        pass


# ── endpoints ─────────────────────────────────────────────────────

@router.get("/status")
def get_status():
    with _state_lock:
        return dict(_training_state)


@router.post("/stop", status_code=200)
def stop_training():
    with _state_lock:
        if not _training_state["running"]:
            raise HTTPException(400, "No training in progress")
        _training_state["stop_requested"] = True
    return {"message": "Stop requested — will finish after current epoch"}


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


@router.get("/runs/{run_id}/metric-history")
def get_metric_history(run_id: str):
    """Return full per-epoch history for every metric in a run."""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        history: dict[str, list] = {}
        for key in run.data.metrics:
            points = client.get_metric_history(run_id, key)
            history[key] = [{"step": p.step, "value": p.value} for p in points]
        return {"run_id": run_id, "history": history}
    except Exception as e:
        raise HTTPException(404, str(e))


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
