# Custom fields

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
