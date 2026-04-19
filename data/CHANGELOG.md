# Changelog

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
