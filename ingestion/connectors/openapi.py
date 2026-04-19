"""OpenAPI reference connector.

One SourceRecord per endpoint (method + path). Chunking happens later,
but typically each endpoint is one chunk — which is exactly what we want
for API-reference queries like "how do I call POST /projects".

Accepts OpenAPI 3.x JSON or YAML.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import AsyncIterator

from ingestion.connectors.base import BaseConnector, SourceRecord


class OpenAPIConnector(BaseConnector):
    source_type = "openapi"

    def __init__(self, spec_path: str | Path, base_url: str = "https://docs.example.com/api"):
        self.path = Path(spec_path).resolve()
        self.base_url = base_url.rstrip("/")

    def _load_spec(self) -> dict:
        text = self.path.read_text(encoding="utf-8")
        if self.path.suffix in {".yaml", ".yml"}:
            import yaml  # optional dep; import lazily
            return yaml.safe_load(text)
        return json.loads(text)

    async def list_records(self) -> AsyncIterator[SourceRecord]:
        if not self.path.exists():
            return
        try:
            spec = self._load_spec()
        except Exception:
            return

        paths = spec.get("paths", {})
        mtime = self.path.stat().st_mtime
        from datetime import datetime, timezone
        updated_at = datetime.fromtimestamp(mtime, tz=timezone.utc)

        for path, ops in paths.items():
            if not isinstance(ops, dict):
                continue
            for method, op in ops.items():
                if method.lower() not in {"get", "post", "put", "patch", "delete"}:
                    continue

                op_id = op.get("operationId") or f"{method.upper()}_{path}"
                summary = op.get("summary", "").strip()
                description = op.get("description", "").strip()
                params = op.get("parameters", []) or []
                request_body = op.get("requestBody", {}) or {}
                responses = op.get("responses", {}) or {}

                body_parts: list[str] = [f"# {method.upper()} {path}"]
                if summary:
                    body_parts.append(f"\n**Summary:** {summary}")
                if description:
                    body_parts.append(f"\n{description}")

                if params:
                    body_parts.append("\n## Parameters")
                    for p in params:
                        pname = p.get("name", "")
                        pin = p.get("in", "")
                        required = "required" if p.get("required") else "optional"
                        pdesc = p.get("description", "")
                        body_parts.append(f"- `{pname}` ({pin}, {required}): {pdesc}")

                if request_body:
                    body_parts.append("\n## Request body")
                    body_parts.append(f"```json\n{json.dumps(request_body, indent=2)[:2000]}\n```")

                if responses:
                    body_parts.append("\n## Responses")
                    for code, r in responses.items():
                        body_parts.append(f"- **{code}**: {r.get('description', '')}")

                content = "\n".join(body_parts)

                yield SourceRecord(
                    source_type="openapi",
                    source_id=op_id,
                    title=f"{method.upper()} {path}",
                    content=content,
                    url=f"{self.base_url}#{op_id}",
                    updated_at=updated_at,
                    extra_metadata={"method": method.upper(), "path": path, "operation_id": op_id},
                )
