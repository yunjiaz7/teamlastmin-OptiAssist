from __future__ import annotations

import imghdr
import re
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, UploadFile, status
from PIL import Image

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_UPLOAD_SIZE_BYTES = 15 * 1024 * 1024


@dataclass
class SavedUpload:
    path: Path
    filename: str
    content_type: str
    size_bytes: int


def _sanitize_filename(name: str) -> str:
    base_name = Path(name or "upload_image").name
    clean_name = re.sub(r"[^A-Za-z0-9._-]", "_", base_name)
    return clean_name or "upload_image"


def _ensure_valid_image(image_bytes: bytes) -> None:
    detected = imghdr.what(None, h=image_bytes)
    if detected not in {"png", "jpeg", "webp"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported or invalid image file.",
        )
    try:
        Image.open(Path("/dev/null"))  # pragma: no cover - warm import path
    except Exception:
        pass
    try:
        from io import BytesIO

        Image.open(BytesIO(image_bytes)).verify()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read uploaded image: {exc}",
        ) from exc


async def save_upload_file(upload: UploadFile, destination_dir: Path) -> SavedUpload:
    filename = _sanitize_filename(upload.filename or "upload_image")
    extension = Path(filename).suffix.lower()
    content_type = (upload.content_type or "").lower()

    if content_type not in ALLOWED_CONTENT_TYPES and extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PNG, JPG, JPEG, and WEBP images are allowed.",
        )

    image_bytes = await upload.read()
    if not image_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")
    if len(image_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size is {MAX_UPLOAD_SIZE_BYTES} bytes.",
        )

    _ensure_valid_image(image_bytes)
    destination_dir.mkdir(parents=True, exist_ok=True)
    target_path = destination_dir / filename
    target_path.write_bytes(image_bytes)

    return SavedUpload(
        path=target_path,
        filename=filename,
        content_type=content_type or "application/octet-stream",
        size_bytes=len(image_bytes),
    )
