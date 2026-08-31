"""Save user-uploaded documents and turn them into SourceRecords."""
from __future__ import annotations
import io
import re
from datetime import datetime, timezone
from pathlib import Path

from ingestion.connectors.base import SourceRecord
from ingestion.sources import UPLOAD_DIR

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_SUFFIXES = {".md", ".txt", ".html", ".htm", ".pdf", ".docx"}
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_SAFE_VISITOR = re.compile(r"[^A-Za-z0-9._-]+")


class UploadError(ValueError):
    pass


def ensure_upload_dir() -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_DIR


def safe_filename(name: str) -> str:
    raw = Path(name or "").name
    cleaned = _SAFE_NAME.sub("_", raw).strip("._") or "untitled"
    return cleaned[:180]


def safe_visitor_id(visitor_id: str) -> str:
    cleaned = _SAFE_VISITOR.sub("_", (visitor_id or "anon").strip())[:80] or "anon"
    return cleaned


def suffix_ok(name: str) -> bool:
    return Path(name).suffix.lower() in ALLOWED_SUFFIXES


def visitor_dir(visitor_id: str) -> Path:
    return ensure_upload_dir() / safe_visitor_id(visitor_id)


def _title_from_text(filename: str, content: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
        if stripped:
            break
    return Path(filename).stem.replace("-", " ").replace("_", " ").strip() or filename


def _bytes_to_text(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise UploadError("PDF support is not installed on this server.") from e
        reader = PdfReader(io.BytesIO(data))
        text = "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()
        if not text:
            raise UploadError("Could not extract text from this PDF.")
        return text
    if suffix == ".docx":
        try:
            from docx import Document
        except ImportError as e:
            raise UploadError("Word (.docx) support is not installed on this server.") from e
        doc = Document(io.BytesIO(data))
        text = "\n\n".join(p.text for p in doc.paragraphs).strip()
        if not text:
            raise UploadError("Could not extract text from this Word file.")
        return text
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise UploadError("File must be UTF-8 text.") from e


def record_from_upload(filename: str, text: str, visitor_id: str) -> SourceRecord:
    suffix = Path(filename).suffix.lower()
    owner = safe_visitor_id(visitor_id)
    source_id = f"{owner}/{filename}"
    if suffix in {".html", ".htm"}:
        source_type = "help_center"
        url = f"uploaded://html/{source_id}"
    else:
        source_type = "markdown_docs"
        url = f"uploaded://docs/{source_id}"
    return SourceRecord(
        source_type=source_type,  # type: ignore[arg-type]
        source_id=source_id,
        title=_title_from_text(filename, text),
        content=text,
        url=url,
        updated_at=datetime.now(timezone.utc),
        extra_metadata={"uploaded": True, "filename": filename, "owner": owner},
    )


def save_upload(filename: str, data: bytes, visitor_id: str = "anon") -> tuple[Path, SourceRecord]:
    if not suffix_ok(filename):
        raise UploadError(
            "Unsupported file type. Use .md, .txt, .html, .htm, .pdf, or .docx."
        )
    if not data:
        raise UploadError("File is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError(f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")

    name = safe_filename(filename)
    text = _bytes_to_text(name, data)
    dest_dir = visitor_dir(visitor_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    dest.write_bytes(data)
    return dest, record_from_upload(name, text, visitor_id)


def list_uploads(visitor_id: str = "anon") -> list[dict]:
    folder = visitor_dir(visitor_id)
    if not folder.exists():
        return []
    out = []
    for path in sorted(folder.iterdir()):
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


def delete_upload_file(filename: str, visitor_id: str) -> Path | None:
    dest = visitor_dir(visitor_id) / safe_filename(filename)
    if not dest.is_file():
        return None
    dest.unlink()
    return dest
