# Source types

The pipeline is the same for every source: clean → chunk → embed →
store. Only the connector differs.

## Markdown

Best for hand-written docs. Headings become chunk boundaries. The
first `#` heading is the title.

## HTML

For exported help-center pages. Nav, footer, and scripts are stripped
so the model sees the article body.

## Tickets

JSONL of resolved conversations. Useful for "we already answered this"
style questions. Unresolved tickets in a webhook payload are ignored.

## Changelog

One document per `##` version heading. Ask "what changed in 0.2.0?"
and you should retrieve that section, not the whole file.

## OpenAPI

One document per HTTP method + path. Ask "how do I call POST /chat?"
and you should retrieve that operation.
