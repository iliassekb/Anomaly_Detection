import io
import json
import os

from minio import Minio
from minio.error import S3Error

BUCKET = os.getenv("MINIO_BUCKET", "defectvision")

_client: Minio | None = None


def get_client() -> Minio:
    global _client
    if _client is None:
        _client = Minio(
            os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            secure=False,
        )
    return _client


def init_buckets() -> None:
    client = get_client()
    for bucket in (BUCKET, "defectvision-mlflow"):
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
    _seed_classes_from_model()


def _seed_classes_from_model() -> None:
    """Populate classes.json from best_m.pt model.names if not yet present."""
    if object_exists("classes.json"):
        return
    weights_path = os.getenv("MODEL_WEIGHTS", "weights/best_m.pt")
    if not os.path.exists(weights_path):
        return
    try:
        from ultralytics import YOLO
        model = YOLO(weights_path)
        names: dict = model.names  # {0: "good", 1: "crack", ...}
        classes = {name: idx for idx, name in names.items()}
        update_classes(classes)
    except Exception:
        pass


# ── classes.json helpers ──────────────────────────────────────────

def get_classes() -> dict:
    """Return {name: id} class registry. Empty dict if not yet created."""
    try:
        data = get_object("classes.json")
        return json.loads(data.decode())
    except S3Error:
        return {}


def update_classes(classes: dict) -> None:
    """Persist {name: id} registry to MinIO."""
    raw = json.dumps(classes, indent=2).encode()
    put_object("classes.json", raw, "application/json")


# ── generic object helpers ────────────────────────────────────────

def put_object(key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    get_client().put_object(
        BUCKET, key,
        io.BytesIO(data), len(data),
        content_type=content_type,
    )


def get_object(key: str) -> bytes:
    response = get_client().get_object(BUCKET, key)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def object_exists(key: str) -> bool:
    try:
        get_client().stat_object(BUCKET, key)
        return True
    except S3Error:
        return False


def delete_object(key: str) -> None:
    get_client().remove_object(BUCKET, key)


def list_objects(prefix: str) -> list[str]:
    """Return list of object keys under the given prefix."""
    objects = get_client().list_objects(BUCKET, prefix=prefix, recursive=True)
    return [obj.object_name for obj in objects]


def copy_object(src_key: str, dst_key: str) -> None:
    from minio.commonconfig import CopySource
    get_client().copy_object(BUCKET, dst_key, CopySource(BUCKET, src_key))
