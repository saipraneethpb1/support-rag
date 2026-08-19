# Add your own files

Replace the example files under `data/` with yours, then ingest.

## Folders the app already watches

| Path | What to put there |
|------|-------------------|
| `data/sample_docs/` | Markdown (`.md`). Nested folders are fine. |
| `data/sample_help_center/` | HTML articles (`.html`). |
| `data/sample_tickets/tickets.jsonl` | One JSON object per line (id, subject, status, messages). |
| `data/CHANGELOG.md` | A Keep-a-Changelog file; each version becomes its own document. |
| `data/openapi.json` | OpenAPI 3 JSON; each endpoint becomes its own document. |

You can keep some of these empty. Connectors skip missing folders.

## After you add or edit files

```
python -m scripts.bootstrap_index
```

Or, with the API running and your API key:

```
curl -X POST http://localhost:8000/ingest/run -H "x-api-key: YOUR_KEY"
```

Unchanged files are skipped (content hash). Deleted files are removed
from the index on a full ingest pass.
