---
name: studio-dev
description: Switch into Studio development mode for working on the NeMo Studio frontend in web/. Use when starting frontend work, building UI features, or working on React components.
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---

# Studio Development Mode

You are now in **Studio development mode**. Read the Studio project instructions to load full context:

@web/CLAUDE.md

## What This Means

- **Most changes will be in `web/`** — this is a React + TypeScript monorepo for NeMo Studio
- **The rest of the repo is useful for understanding** backend APIs, data models, and SDK types, but changes should generally stay within `web/`
- **Studio-specific skills are now available** — use `/unit-test`, `/e2e-test`, and `/visual-dev` for testing and development workflows
- **Run frontend commands from `web/`** — pnpm, vitest, and other Node.js tooling lives there

## Quick Reference

| What | Where |
|------|-------|
| Main app | `web/packages/studio/` |
| Shared code | `web/packages/common/` |
| API SDK | `web/packages/sdk/` |
| Storybook | `web/packages/storybook/` |
| Dev tools | `web/packages/sandbox/` |
| Backend APIs | `services/` (read-only context) |
| OpenAPI spec | `openapi/openapi.yaml` (read-only context) |
