"""Validation for UTKFace records and uploaded images."""

from pathlib import Path

from PIL import Image, UnidentifiedImageError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


def parse_utkface_filename(path: str | Path) -> dict[str, object]:
    item = Path(path)
    fields = item.name.split("_")
    if len(fields) < 4:
        raise ValueError(f"Invalid UTKFace filename: {item.name}")
    try:
        age, gender, ethnicity = map(int, fields[:3])
    except ValueError as exc:
        raise ValueError(f"Invalid UTKFace labels: {item.name}") from exc
    if not 0 <= age <= 116:
        raise ValueError(f"Age outside supported range: {age}")
    if gender not in (0, 1) or ethnicity not in range(5):
        raise ValueError(f"Invalid demographic label: {item.name}")
    return {"path": str(item), "age": age, "gender": gender, "ethnicity": ethnicity}


def validate_image(image: Image.Image, size_bytes: int | None = None) -> None:
    if size_bytes is not None and size_bytes > MAX_UPLOAD_BYTES:
        raise ValueError("Image exceeds the 10 MB upload limit")
    if image.format not in ALLOWED_FORMATS:
        raise ValueError("Only JPEG, PNG, and WebP images are supported")
    try:
        image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid image") from exc
