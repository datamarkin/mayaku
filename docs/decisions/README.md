# Architecture decision records

Each ADR is a short, dated, immutable note explaining a non-obvious choice
or a target we dropped. Filename pattern: `NNN-<short-slug>.md`, numbered
sequentially.

## When to write one
- Dropping a best-effort export target (`NNN-drop-<backend>.md`) — required
  per `03_d2_reimplementation_prompt.md` §"Drop policy".
- Picking one of several defensible defaults when the spec or portability
  report is ambiguous.
- Departing from a recommendation in `BACKEND_PORTABILITY_REPORT.md`.

## When *not* to write one
- Routine implementation choices already implied by the spec.
- Anything that can be read out of the code (file structure, naming).

## Template

```markdown
# NNN — <decision title>

- **Status:** accepted | superseded by NNN
- **Date:** YYYY-MM-DD

## Context
Why this came up.

## Decision
What we picked.

## Consequences
What we gain, what we give up, what would need to change to revisit.
```

(no ADRs yet)
