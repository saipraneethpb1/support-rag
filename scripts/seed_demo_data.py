"""Seed realistic demo data.

Creates a plausible corpus for a fictional project-management SaaS
called "Flowpoint". Not Lorem Ipsum — the content reads like real docs
so retrieval evaluation is meaningful.

Usage:
    python -m scripts.seed_demo_data
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path("data")


DOCS = {
    "getting-started/quickstart.md": """# Quickstart

Get your first Flowpoint project running in under five minutes.

## 1. Create your workspace

Sign in at https://app.flowpoint.io and click **Create workspace**. Pick a
name that reflects your team (you can rename later). Each workspace has
its own billing, members, and projects.

## 2. Invite your team

From the workspace sidebar, open **People > Invite members**. Paste a list
of email addresses (up to 50 at once). Invitees get an email with a
magic-link signup — no password required. Free plans are capped at 10
members per workspace.

## 3. Create a project

Click **New project**. Choose a template (Kanban, Sprint, Roadmap) or
start blank. Projects are where tasks, documents, and discussions live.

## 4. Add your first task

Inside a project, press `C` to create a task. Give it a title, assignee,
and due date. That's it — you're live.
""",
    "getting-started/pricing.md": """# Pricing and plans

Flowpoint has three plans: Free, Team, and Business.

## Free

- Up to 10 members
- Unlimited projects
- 2 GB file storage per workspace
- 30-day activity history
- Community support

## Team — $8 per member per month

- Unlimited members
- 100 GB file storage
- Unlimited activity history
- Custom fields and views
- Email support (24h response)
- SSO via Google and Microsoft

## Business — $16 per member per month

- Everything in Team
- 1 TB file storage
- SAML SSO and SCIM provisioning
- Audit log and data residency controls
- Priority support (4h response)
- 99.9% uptime SLA

Billing is monthly or annual (annual saves 20%). You can switch plans
anytime; downgrades apply at the end of the current billing period.
""",
    "account/cancel-subscription.md": """# Cancel your subscription

You can cancel your paid subscription at any time. Your workspace stays
active until the end of the current billing period, then automatically
moves to the Free plan.

## How to cancel

1. Open **Settings > Billing** (workspace owner only).
2. Click **Cancel subscription**.
3. Confirm by typing your workspace name.

## What happens to your data

Nothing is deleted. If your workspace exceeds Free-plan limits (for
example, more than 10 members or 2 GB storage), those features become
read-only until you're back within limits or upgrade again.

## Refunds

Annual plans are prorated on request within 30 days of renewal. Monthly
plans are not refunded, but you keep access until the period ends.

## Trouble canceling

If the Cancel button is grayed out, you're probably not the workspace
owner. Ask the owner to cancel, or transfer ownership first via
**Settings > People > Transfer ownership**.
""",
    "account/sso-setup.md": """# Set up SSO

Single sign-on is available on Team (Google, Microsoft) and Business
(SAML, Okta, OneLogin) plans.

## SAML SSO (Business plan)

1. In Flowpoint, open **Settings > Security > SSO** and copy the ACS URL
   and Entity ID.
2. In your identity provider, create a new SAML app with those values.
3. Paste the IdP metadata URL back into Flowpoint.
4. Click **Test connection**. A successful test shows your email from
   the IdP.
5. Enable **Require SSO** to force all members to sign in via SSO.

## Common errors

- **"Invalid audience"** — The Entity ID in your IdP doesn't match
  Flowpoint's. Copy it again from the SSO settings page.
- **"User not provisioned"** — SCIM isn't enabled or the user doesn't
  exist in Flowpoint yet. Either enable SCIM auto-provisioning or
  invite the user manually first.
- **Redirect loop after sign-in** — Your IdP is returning a NameID
  format Flowpoint doesn't recognize. Set NameID format to
  `emailAddress`.
""",
    "projects/custom-fields.md": """# Custom fields

Custom fields let you add structured data to tasks beyond the built-in
fields. Available on Team and Business plans.

## Field types

- **Text** — short single-line strings
- **Number** — integers or decimals, with optional unit label
- **Select** — single choice from a fixed list of options
- **Multi-select** — multiple choices
- **Date** — calendar date, optional time
- **Person** — workspace member
- **URL** — validated link
- **Formula** — computed from other fields (Business plan only)

## Create a custom field

Open a project, click the **+** at the end of the column headers, choose
**Custom field**, pick a type, and name it. Fields can be project-scoped
(visible only in this project) or workspace-scoped (reusable).

## Limits

- 50 fields per project
- 200 workspace-scoped fields total
- Formula depth capped at 5 levels to prevent runaway computation
""",
    "projects/automations.md": """# Automations

Automations run when a trigger fires and execute one or more actions.
Think "when a task moves to Done, notify the reporter on Slack".

## Triggers

- Task created, updated, moved, completed, or deleted
- Due date approaching (1 day, 1 hour)
- Custom field changed
- Comment added
- Schedule (hourly, daily, weekly, cron)

## Actions

- Update a task field
- Create a task
- Post to Slack or Microsoft Teams
- Send an email
- Call a webhook

## Limits and quotas

Free plan: no automations. Team: 100 runs per member per month.
Business: 1,000 runs per member per month. Overages are throttled, not
billed — queued runs execute when quota resets.

## Debugging

Every automation run appears in **Automations > Run history** with input,
output, and status. Failed runs can be retried manually. If a webhook
destination returns 5xx, Flowpoint retries up to 3 times with
exponential backoff.
""",
    "api/authentication.md": """# API authentication

The Flowpoint API uses bearer tokens. Generate a token at
**Settings > API > Personal tokens** or create a workspace-scoped token
at **Settings > API > Workspace tokens** (Business plan).

## Making a request

```
curl https://api.flowpoint.io/v1/projects \\
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
""",
}

CHANGELOG = """# Changelog

All notable changes to Flowpoint will be documented here.

## [2.14.0] - 2026-03-12

### Added
- Formula custom fields on Business plan
- Bulk task import from CSV (up to 10,000 rows)
- SCIM 2.0 provisioning for Okta and OneLogin

### Fixed
- SSO redirect loop when NameID format was unset
- Automations occasionally firing twice on rapid status updates

## [2.13.0] - 2026-02-04

### Added
- Slack unfurl for Flowpoint task links
- Keyboard shortcut `G D` to jump to dashboard

### Changed
- API rate limit for workspace tokens raised from 300 to 600 req/min

### Fixed
- Attachments over 50 MB failing silently in Safari

## [2.12.1] - 2026-01-15

### Fixed
- Workspace owner transfer failing when target had a pending invite
- Timezone off-by-one on due date reminders for users in UTC-negative zones
"""

TICKETS = [
    {
        "id": "T-10021",
        "subject": "How do I cancel my subscription mid-cycle?",
        "status": "resolved",
        "created_at": "2026-03-01T10:12:00Z",
        "updated_at": "2026-03-01T11:05:00Z",
        "tags": ["billing", "cancellation"],
        "messages": [
            {"author": "user", "body": "I want to cancel my Team plan today. Will I get a refund for the rest of the month?", "ts": "2026-03-01T10:12:00Z"},
            {"author": "agent", "body": "Monthly plans aren't prorated on cancellation, but you keep full access until your next billing date. After that your workspace moves to Free automatically. To cancel: Settings > Billing > Cancel subscription.", "ts": "2026-03-01T10:44:00Z"},
            {"author": "user", "body": "Thanks, that worked.", "ts": "2026-03-01T11:05:00Z"},
        ],
    },
    {
        "id": "T-10088",
        "subject": "SSO redirect loop with Okta",
        "status": "resolved",
        "created_at": "2026-03-05T14:22:00Z",
        "updated_at": "2026-03-05T16:40:00Z",
        "tags": ["sso", "okta"],
        "messages": [
            {"author": "user", "body": "After clicking Sign in with SSO our users get bounced back to the login page repeatedly. Okta shows the auth as successful.", "ts": "2026-03-05T14:22:00Z"},
            {"author": "agent", "body": "Almost always a NameID format mismatch. In Okta, open the Flowpoint app, go to General > SAML Settings > Edit, and set Name ID format to EmailAddress. Save and re-test.", "ts": "2026-03-05T15:30:00Z"},
            {"author": "user", "body": "That fixed it, thank you.", "ts": "2026-03-05T16:40:00Z"},
        ],
    },
    {
        "id": "T-10142",
        "subject": "Automation ran twice on the same task",
        "status": "resolved",
        "created_at": "2026-02-20T08:00:00Z",
        "updated_at": "2026-02-21T09:10:00Z",
        "tags": ["automations", "bug"],
        "messages": [
            {"author": "user", "body": "Our Slack notify automation posted two messages when I moved a task to Done.", "ts": "2026-02-20T08:00:00Z"},
            {"author": "agent", "body": "Known issue with rapid consecutive status updates firing the trigger twice. Fixed in 2.14.0 released March 12. Please upgrade if you're on an older self-hosted version; cloud customers are already on the fix.", "ts": "2026-02-21T09:10:00Z"},
        ],
    },
    {
        "id": "T-10203",
        "subject": "Can't upload 80MB video attachment in Safari",
        "status": "resolved",
        "created_at": "2026-01-28T12:00:00Z",
        "updated_at": "2026-01-29T10:00:00Z",
        "tags": ["attachments", "safari"],
        "messages": [
            {"author": "user", "body": "Uploads over ~50MB silently fail in Safari. Works in Chrome.", "ts": "2026-01-28T12:00:00Z"},
            {"author": "agent", "body": "Confirmed — fixed in 2.13.0. If you're still hitting it, hard-refresh (Cmd-Shift-R) to clear the old client.", "ts": "2026-01-29T10:00:00Z"},
        ],
    },
]

OPENAPI = {
    "openapi": "3.0.3",
    "info": {"title": "Flowpoint API", "version": "1.0.0"},
    "paths": {
        "/v1/projects": {
            "get": {
                "operationId": "listProjects",
                "summary": "List projects",
                "description": "Returns all projects visible to the authenticated token.",
                "parameters": [
                    {"name": "limit", "in": "query", "required": False, "description": "Max results (default 50, cap 200)."},
                    {"name": "cursor", "in": "query", "required": False, "description": "Pagination cursor from previous response."},
                ],
                "responses": {"200": {"description": "A paginated list of projects."}, "401": {"description": "Invalid token."}},
            },
            "post": {
                "operationId": "createProject",
                "summary": "Create a project",
                "description": "Creates a new project in the workspace. Requires write scope.",
                "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {"name": {"type": "string"}, "template": {"type": "string"}}}}}},
                "responses": {"201": {"description": "Project created."}, "400": {"description": "Invalid request."}, "403": {"description": "Insufficient scope."}},
            },
        },
        "/v1/tasks/{task_id}": {
            "get": {
                "operationId": "getTask",
                "summary": "Get a task",
                "parameters": [{"name": "task_id", "in": "path", "required": True, "description": "Task ID."}],
                "responses": {"200": {"description": "The task."}, "404": {"description": "Not found."}},
            },
            "patch": {
                "operationId": "updateTask",
                "summary": "Update a task",
                "parameters": [{"name": "task_id", "in": "path", "required": True, "description": "Task ID."}],
                "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
                "responses": {"200": {"description": "Updated task."}, "404": {"description": "Not found."}},
            },
        },
    },
}


def main() -> None:
    (ROOT / "sample_docs").mkdir(parents=True, exist_ok=True)
    for rel, body in DOCS.items():
        path = ROOT / "sample_docs" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")

    (ROOT / "CHANGELOG.md").write_text(CHANGELOG, encoding="utf-8")

    (ROOT / "sample_tickets").mkdir(parents=True, exist_ok=True)
    with (ROOT / "sample_tickets" / "tickets.jsonl").open("w", encoding="utf-8") as f:
        for t in TICKETS:
            f.write(json.dumps(t) + "\n")

    (ROOT / "openapi.json").write_text(json.dumps(OPENAPI, indent=2), encoding="utf-8")

    print(f"Seeded: {len(DOCS)} docs, 1 changelog, {len(TICKETS)} tickets, 1 openapi spec")


if __name__ == "__main__":
    main()
