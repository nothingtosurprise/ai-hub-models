# Triage Knowledge Base

This directory contains the knowledge base for automated issue triage.
It is used by AI agents (Breeze, Claude) to generate soft triage recommendations
on newly created GitHub issues.

## Files

| File | Purpose |
|------|---------|
| [teams.md](teams.md) | Team ownership mapping — labels, keyword signals |
| [error-patterns.md](error-patterns.md) | Error signature → label/team routing with confidence levels |
| [historical-patterns.md](historical-patterns.md) | Recurring failure patterns from 6 months of nightly triage |
| [runtime-guide.md](runtime-guide.md) | Runtime architecture context for disambiguation |
| [examples.md](examples.md) | Real triage decisions with reasoning |

## How It Works

1. A nightly failure or bug is filed as a GitHub issue
2. An AI agent reads the issue title, body, and error logs
3. The agent consults this knowledge base to classify the issue
4. A **triage card** is posted as a comment with:
   - Suggested label(s)
   - Suggested team
   - Confidence level (HIGH / MEDIUM / LOW)
   - Similar past issues
5. The release czar reviews the suggestion and assigns accordingly

## Maintaining This Knowledge Base

- **Anyone can edit** — this is version-controlled markdown
- Update `teams.md` when team ownership changes or new teams form
- Add new patterns to `error-patterns.md` as you discover them
- Add interesting triage decisions to `examples.md` (especially ambiguous ones)
- Data source: analysis of 1000 issues from qcom-ai-hub/tetracode (Nov 2025 – Apr 2026)

## Key Principles

1. **Soft recommendations only** — never auto-assign, always suggest
2. **Labels are the strongest signal** — route by label first, keyword second
3. **Confidence matters** — flag ambiguous cases so the czar knows to investigate
4. **Czar rotation** — never hardcode a person for nightly failure triage
