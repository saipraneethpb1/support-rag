# Ask questions

Open http://localhost:8000. Upload a file or ask a question — the UI
does not ask for a key. **New chat** starts a blank thread. Past chats
stay in the left sidebar on this browser (they are not stored on the
server).

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

The `x-api-key` header must match `API_KEY` in `.env`. The chat page
does not ask for a key; opening it sets a cookie so upload and chat
work in the browser.

## Optional filters

You can limit retrieval to source types: `markdown_docs`, `help_center`,
`tickets`, `changelog`, `openapi` via the `source_types` field on the
chat request.
