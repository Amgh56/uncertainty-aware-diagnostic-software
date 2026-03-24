import os

from supabase import create_client, Client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

# Bucket names
BUCKET_XRAY_IMAGES = "xray-images"
BUCKET_MODELS = "models"
BUCKET_CALIBRATION = "calibration-jobs"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def ensure_buckets():
    """Create all storage buckets if they don't exist."""
    for bucket_name, public in [
        (BUCKET_XRAY_IMAGES, True),
        (BUCKET_MODELS, False),
        (BUCKET_CALIBRATION, False),
    ]:
        try:
            supabase.storage.get_bucket(bucket_name)
        except Exception:
            supabase.storage.create_bucket(
                bucket_name,
                options={"public": public},
            )


def upload_to_bucket(
    bucket: str,
    path: str,
    file_bytes: bytes,
    content_type: str = "application/octet-stream",
) -> str:
    """Upload bytes to a Supabase Storage bucket. Returns the storage path."""
    supabase.storage.from_(bucket).upload(
        path,
        file_bytes,
        file_options={"content-type": content_type},
    )
    return path


def download_from_bucket(bucket: str, path: str) -> bytes:
    """Download a file from Supabase Storage and return raw bytes."""
    return supabase.storage.from_(bucket).download(path)


def delete_from_bucket(bucket: str, paths: list[str]) -> None:
    """Delete one or more files from a Supabase Storage bucket."""
    supabase.storage.from_(bucket).remove(paths)


def get_public_url(bucket: str, path: str) -> str:
    """Get the public URL for a file in a bucket."""
    return supabase.storage.from_(bucket).get_public_url(path)


# ── X-ray image upload helper ─────────────────────────────

def upload_image(file_bytes: bytes, path: str, content_type: str) -> str:
    """Upload image bytes to the xray-images bucket. Returns public URL."""
    upload_to_bucket(BUCKET_XRAY_IMAGES, path, file_bytes, content_type)
    return get_public_url(BUCKET_XRAY_IMAGES, path)
