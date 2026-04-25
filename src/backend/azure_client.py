import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Container / bucket names ──────────────────────────────
BUCKET_XRAY_IMAGES = "xray-images"
BUCKET_MODELS = "models"
BUCKET_CALIBRATION = "calibration-jobs"

# ── Local fallback directory ──────────────────────────────
LOCAL_STORAGE_ROOT = Path(__file__).resolve().parent / "local_storage"

# ── Azure Blob client (None if connection string is missing) ──
blob_service_client = None

try:
    from azure.storage.blob import BlobServiceClient, ContentSettings

    conn_str = os.environ.get("STORAGE_ACCOUNT_CONNECTION_STRING", "")
    if conn_str:
        blob_service_client = BlobServiceClient.from_connection_string(conn_str)
        logger.info("Azure Blob Storage connected.")
    else:
        logger.warning("STORAGE_ACCOUNT_CONNECTION_STRING not set — using local fallback.")
except ImportError:
    logger.warning("azure-storage-blob not installed — using local fallback.")


def use_azure() -> bool:
    return blob_service_client is not None


def _allow_runtime_fallback(bucket: str) -> bool:
    """
    Only public image uploads may fall back to local storage after an Azure error.

    For private calibration/model artifacts, a runtime Azure failure must surface
    immediately so we do not silently split state between Azure and local disk.
    """
    return bucket == BUCKET_XRAY_IMAGES



def local_path(bucket: str, path: str) -> Path:
    return LOCAL_STORAGE_ROOT / bucket / path


def local_upload(bucket: str, path: str, file_bytes: bytes) -> None:
    dest = local_path(bucket, path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(file_bytes)


def local_download(bucket: str, path: str) -> bytes:
    dest = local_path(bucket, path)
    if not dest.exists():
        raise FileNotFoundError(f"Local file not found: {dest}")
    return dest.read_bytes()


def local_delete(bucket: str, paths: list[str]) -> None:
    for p in paths:
        dest = local_path(bucket, p)
        if dest.exists():
            dest.unlink()



def ensure_buckets():
    """Create Azure containers or local directories."""
    if use_azure():
        for name, public_access in [
            (BUCKET_XRAY_IMAGES, "blob"),   # public — frontend loads images directly
            (BUCKET_MODELS, None),           # private
            (BUCKET_CALIBRATION, None),      # private
        ]:
            try:
                blob_service_client.create_container(name, public_access=public_access)
            except Exception:
                pass
    else:
        for name in [BUCKET_XRAY_IMAGES, BUCKET_MODELS, BUCKET_CALIBRATION]:
            (LOCAL_STORAGE_ROOT / name).mkdir(parents=True, exist_ok=True)


def upload_to_bucket(
    bucket: str,
    path: str,
    file_bytes: bytes,
    content_type: str = "application/octet-stream",
) -> str:
    """Upload bytes to Azure Blob Storage, falling back to local disk."""
    if use_azure():
        try:
            blob_client = blob_service_client.get_blob_client(container=bucket, blob=path)
            blob_client.upload_blob(
                file_bytes,
                overwrite=True,
                content_settings=ContentSettings(content_type=content_type),
            )
            return path
        except Exception as exc:
            logger.exception("Azure upload failed for %s/%s", bucket, path)
            if not _allow_runtime_fallback(bucket):
                raise RuntimeError(
                    f"Azure upload failed for {bucket}/{path}"
                ) from exc

    local_upload(bucket, path, file_bytes)
    return path


def download_from_bucket(bucket: str, path: str) -> bytes:
    """Download from Azure Blob Storage, falling back to local disk."""
    if use_azure():
        try:
            blob_client = blob_service_client.get_blob_client(container=bucket, blob=path)
            return blob_client.download_blob().readall()
        except Exception as exc:
            logger.exception("Azure download failed for %s/%s", bucket, path)
            if not _allow_runtime_fallback(bucket):
                raise RuntimeError(
                    f"Azure download failed for {bucket}/{path}"
                ) from exc

    return local_download(bucket, path)


def delete_from_bucket(bucket: str, paths: list[str]) -> None:
    """Delete from Azure Blob Storage, falling back to local disk."""
    if use_azure():
        try:
            container_client = blob_service_client.get_container_client(bucket)
            for p in paths:
                try:
                    container_client.delete_blob(p)
                except Exception:
                    pass
            return
        except Exception as exc:
            logger.exception("Azure delete failed for bucket %s", bucket)
            if not _allow_runtime_fallback(bucket):
                raise RuntimeError(
                    f"Azure delete failed for bucket {bucket}"
                ) from exc

    local_delete(bucket, paths)


def get_public_url(bucket: str, path: str) -> str:
    """Get the public URL for a blob, or a local file path as fallback."""
    if use_azure():
        account_name = blob_service_client.account_name
        return f"https://{account_name}.blob.core.windows.net/{bucket}/{path}"
    return f"/local_storage/{bucket}/{path}"



def upload_image(file_bytes: bytes, path: str, content_type: str) -> str:
    """Upload image bytes to the xray-images container. Returns public URL."""
    upload_to_bucket(BUCKET_XRAY_IMAGES, path, file_bytes, content_type)
    return get_public_url(BUCKET_XRAY_IMAGES, path)
