import os

from supabase import create_client, Client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

STORAGE_BUCKET = "xray-images"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def ensure_bucket():
    """Create the storage bucket if it doesn't exist."""
    try:
        supabase.storage.get_bucket(STORAGE_BUCKET)
    except Exception:
        supabase.storage.create_bucket(
            STORAGE_BUCKET,
            options={"public": True},
        )


def upload_image(file_bytes: bytes, path: str, content_type: str) -> str:
    """Upload image bytes to Supabase Storage and return the public URL."""
    supabase.storage.from_(STORAGE_BUCKET).upload(
        path,
        file_bytes,
        file_options={"content-type": content_type},
    )
    return supabase.storage.from_(STORAGE_BUCKET).get_public_url(path)
