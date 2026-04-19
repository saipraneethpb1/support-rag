# Automations

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
