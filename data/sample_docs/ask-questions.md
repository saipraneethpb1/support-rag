# Ask questions

Open http://localhost:8000. Paste the same `API_KEY` you set in `.env`
into the key field (it is empty on purpose). Type a question.

## Chat API

Blocking JSON:

```
curl -X POST http://localhost:8000/chat \
  -H "content-type: application/json" \
  -H "x-api-key: YOUR_KEY" \
  -d '{"question": "How do I add my own files?"}'
```

Streaming (server-sent events): `POST /chat/stream` with the same body.
Tokens arrive as `event: token`; citations arrive as a final `event: meta`.

## If you get 401

The `x-api-key` header (or the UI field) must match `API_KEY` in `.env`.
The demo UI does not pre-fill the key.

## Optional filters

You can limit retrieval to source types: `markdown_docs`, `help_center`,
`tickets`, `changelog`, `openapi` via the `source_types` field on the
chat request.
