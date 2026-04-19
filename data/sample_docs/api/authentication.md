# API authentication

The Flowpoint API uses bearer tokens. Generate a token at
**Settings > API > Personal tokens** or create a workspace-scoped token
at **Settings > API > Workspace tokens** (Business plan).

## Making a request

```
curl https://api.flowpoint.io/v1/projects \
  -H "Authorization: Bearer fpk_live_abc123"
```

## Token scopes

- `read` — read-only access to projects, tasks, comments
- `write` — create and update tasks and comments
- `admin` — manage members, fields, automations (workspace tokens only)

## Rate limits

- Personal tokens: 60 requests per minute
- Workspace tokens: 600 requests per minute

Rate-limited responses return `429 Too Many Requests` with a
`Retry-After` header in seconds.
