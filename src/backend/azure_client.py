import logging
import os
from collections.abc import Iterator

logger = logging.getLogger(__name__)

# ── Container / bucket names ─────────
BUCKET_XRAY_IMAGES = "xray-images"
BUCKET_MODELS = "models"
BUCKET_CALIBRATION = "calibration-jobs"
DEFAULT_DOWNLOAD_CHUNK_SIZE = 1024 * 1024

blob_service_client = None

try:
    from azure.storage.blob import BlobServiceClient, ContentSettings

    conn_str = os.environ.get("STORAGE_ACCOUNT_CONNECTION_STRING", "")
    if conn_str:
        blob_service_client = BlobServiceClient.from_connection_string(conn_str)
        logger.info("Azure Blob Storage connected.")
    else:
        logger.warning("STORAGE_ACCOUNT_CONNECTION_STRING not set.")
except ImportError:
    logger.warning("azure-storage-blob not installed.")


def is_azure_configured() -> bool:
    return blob_service_client is not None


def require_azure() -> None:
    if not is_azure_configured():
        raise RuntimeError(
            "Azure Blob Storage is not configured. "
            "Set STORAGE_ACCOUNT_CONNECTION_STRING and ensure azure-storage-blob is installed."
        )



def ensure_buckets():
    """Create Azure containers."""
    require_azure()
    for name, public_access in [
        (BUCKET_XRAY_IMAGES, "blob"),   
        (BUCKET_MODELS, None),         
        (BUCKET_CALIBRATION, None),     
    ]:
        try:
            blob_service_client.create_container(name, public_access=public_access)
        except Exception:
            pass


def upload_to_bucket(
    bucket: str,
    path: str,
    file_bytes: bytes,
    content_type: str = "application/octet-stream",
) -> str:
    """Upload bytes to Azure Blob Storage."""
    require_azure()
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
        raise RuntimeError(
            f"Azure upload failed for {bucket}/{path}"
        ) from exc


def iter_blob_chunks(
    bucket: str,
    path: str,
    chunk_size: int = DEFAULT_DOWNLOAD_CHUNK_SIZE,
) -> Iterator[bytes]:
    """Yield blob bytes incrementally from Azure Blob Storage."""
    require_azure()
    try:
        blob_client = blob_service_client.get_blob_client(container=bucket, blob=path)
        stream = blob_client.download_blob(max_concurrency=1)
        _ = chunk_size
        yield from stream.chunks()
    except Exception as exc:
        logger.exception("Azure chunked download failed for %s/%s", bucket, path)
        raise RuntimeError(
            f"Azure chunked download failed for {bucket}/{path}"
        ) from exc


def download_from_bucket(bucket: str, path: str) -> bytes:
    """Download from Azure Blob Storage into memory using chunked reads."""
    return b"".join(iter_blob_chunks(bucket, path))


def delete_from_bucket(bucket: str, paths: list[str]) -> None:
    """Delete from Azure Blob Storage."""
    require_azure()
    try:
        container_client = blob_service_client.get_container_client(bucket)
        for p in paths:
            try:
                container_client.delete_blob(p)
            except Exception:
                pass
    except Exception as exc:
        logger.exception("Azure delete failed for bucket %s", bucket)
        raise RuntimeError(
            f"Azure delete failed for bucket {bucket}"
        ) from exc


def get_public_url(bucket: str, path: str) -> str:
    """Get the public URL for a blob."""
    require_azure()
    account_name = blob_service_client.account_name
    return f"https://{account_name}.blob.core.windows.net/{bucket}/{path}"



def upload_image(file_bytes: bytes, path: str, content_type: str) -> str:
    """Upload image bytes to the xray-images container. Returns public URL."""
    upload_to_bucket(BUCKET_XRAY_IMAGES, path, file_bytes, content_type)
    return get_public_url(BUCKET_XRAY_IMAGES, path)
