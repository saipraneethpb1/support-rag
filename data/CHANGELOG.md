# Changelog

All notable changes to this app are listed here.

## [0.4.0] - 2026-08-26

### Changed
- Chat UI no longer asks visitors for an API key (cookie on GET /)

## [0.3.0] - 2026-08-25

### Added
- Upload `.md`, `.txt`, and `.html` from the chat UI (`POST /ingest/upload`)

## [0.2.0] - 2026-08-17

### Added
- Generic document Q&A (not tied to a company support desk)
- Example corpus that explains how to add your own files

### Changed
- Semantic cache invalidates when ingest actually changes the corpus

## [0.1.0] - 2026-08-09

### Added
- Hybrid retrieval (vector + BM25 + RRF)
- Streaming chat and citation audit
- Markdown, HTML, tickets, changelog, and OpenAPI connectors
