"""Save user-uploaded documents and turn them into SourceRecords."""
from __future__ import annotations
import re
from datetime import datetime, timezone
from pathlib import Path

from ingestion.connectors.base import SourceRecord
from ingestion.sources import UPLOAD_DIR

MAX_UPLOAD_BYTES = 2 * 1024 * 1024
ALLOWED_SUFFIXES = {".md", ".txt", ".html", ".htm"}
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class UploadError(ValueError):
    pass


def ensure_upload_dir() -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_DIR


def safe_filename(name: str) -> str:
    raw = Path(name or "").name
    cleaned = _SAFE_NAME.sub("_", raw).strip("._") or "untitled"
    return cleaned[:180]


def suffix_ok(name: str) -> bool:
    return Path(name).suffix.lower() in ALLOWED_SUFFIXES


def _title_from_text(filename: str, content: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
        if stripped:
            break
    return Path(filename).stem.replace("-", " ").replace("_", " ").strip() or filename


def record_from_upload(filename: str, text: str) -> SourceRecord:
    suffix = Path(filename).suffix.lower()
    if suffix in {".html", ".htm"}:
        source_type = "help_center"
        url = f"uploaded://html/{filename}"
    else:
        source_type = "markdown_docs"
        url = f"uploaded://docs/{filename}"
    return SourceRecord(
        source_type=source_type,  # type: ignore[arg-type]
        source_id=filename,
        title=_title_from_text(filename, text),
        content=text,
        url=url,
        updated_at=datetime.now(timezone.utc),
        extra_metadata={"uploaded": True, "filename": filename},
    )


def save_upload(filename: str, data: bytes) -> tuple[Path, SourceRecord]:
    if not suffix_ok(filename):
        raise UploadError(
            "Unsupported file type. Use .md, .txt, .html, or .htm."
        )
    if not data:
        raise UploadError("File is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError(f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise UploadError("File must be UTF-8 text.") from e

    dest_dir = ensure_upload_dir()
    dest = dest_dir / safe_filename(filename)
    dest.write_bytes(data)
    return dest, record_from_upload(dest.name, text)


def list_uploads() -> list[dict]:
    if not UPLOAD_DIR.exists():
        return []
    out = []
    for path in sorted(UPLOAD_DIR.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        if not suffix_ok(path.name):
            continue
        stat = path.stat()
        out.append(
            {
                "filename": path.name,
                "bytes": stat.st_size,
                "updated_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    return out
