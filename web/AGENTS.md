<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Studio Agent Instructions

## Project Context

- NeMo Studio — React + TypeScript monorepo for the NeMo Helix UI
- Tech Stack: React 18, TypeScript, Vite, Vitest, Playwright, pnpm workspaces
- Uses KUI (Kaizen UI) React components, TanStack Query

## Agent skills

Cursor/Claude skills for this monorepo live under **`web/.agents/skills/`** (for example `unit-test`, `e2e-test`, `feature-flags`, `ux-guidelines`, `visual-dev`, `test-coverage-improvement`).

The `kaizen-ui` skill is not committed — it is synced from the `@nvidia/foundations-react-core` design system package on `pnpm install` (root `sync-skills` script) and is gitignored. To refresh it after upgrading the package, run `pnpm sync-skills` from `web/`.

## Package Overview

- **packages/studio** — Main React frontend application
- **packages/common** — Shared code between UI and service
- **packages/sdk** — Generated API clients and types
- **packages/sandbox** — Development tools
- **packages/storybook** — Component stories and documentation
- **packages/scripts** — Build scripts and utilities

## Import Path Rules

- **Never use relative imports.** This applies to sibling files in the same directory too. Always use the absolute alias path — even `./useSomething` or `./types` should be written as `@studio/components/Foo/useSomething` or `@nemo/common/src/utils/types`. ESLint (`no-relative-import-paths/no-relative-import-paths`) enforces this and will warn if you slip.
- Import path mappings:
  - `@studio/` → `packages/studio/src/` (inside the studio package)
  - `@nemo/common/` → `packages/common/src/`
  - `@nemo/sdk` → `packages/sdk/`
- Use `import type` for type-only imports

## Importing the generated SDK

`packages/sdk/generated/<service>/` is regenerated per service into one file per
OpenAPI tag (e.g. `generated/platform/files.ts`, `generated/agents/agents.ts`) —
not a single `api.ts`. Import a hook from its specific tag file:

```ts
import { useFilesDeleteFileset } from '@nemo/sdk/generated/platform/files';
```

- **Never** `import * as sdk from '@nemo/sdk/generated/<service>/index'` (or the
  old `.../api'` path — it no longer exists) just to reach one or two hooks.
  That barrel re-exports the entire service and defeats the point of the
  per-tag split; it exists only for the two call sites that genuinely need the
  whole surface (`packages/studio/src/plugins/PluginRenderer.tsx` and
  `plugins/types.ts`'s `PluginSdk`).
- You don't know which tag file a hook lives in ahead of time — grep
  `packages/sdk/generated/<service>/*.ts` for `export const <hookName>`, or let
  your editor's auto-import resolve it.
- `vi.mock('@nemo/sdk/generated/<service>/api', ...)` from before this split no
  longer resolves. Mock the specific tag file(s) the test actually imports from;
  if a test's mocked hooks span more than one tag file, that's multiple
  `vi.mock()` calls, one per file — don't merge them back into a single call
  against a nonexistent path.
- The generated tree is gitignored and gets rebuilt by `pnpm gen:<service>` /
  `pnpm gen:all` (see `packages/sdk/package.json`) — file layout can shift on
  regeneration if the OpenAPI spec's tags change.

## Package Management

- Use **pnpm** exclusively — never npm or yarn
- Run frontend commands from `web/`, not from repo root
- Node.js and pnpm come from the root Flox environment. Run `flox -q activate` for interactive work; Corepack provides the repository-pinned pnpm.
- Install dependencies: `pnpm add <package>`
- Run scripts: `pnpm <script-name>`

### Bumping a shared singleton also means bumping the plugins

Everything in `VENDOR_IMPORT_MAP` (`packages/studio/vite.config.ts`) is served to
plugin bundles from Studio's vendor build, so plugins must leave it external and
never ship their own copy:

| Shared                                                                                          | In the plugin's `package.json`?                                                                                                                                                  |
| ----------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `react`, `react-dom`, `react-router`, `@nvidia/foundations-react-core`, `@tanstack/react-query` | Yes — versions must match the `catalog:` block                                                                                                                                   |
| `@nemo/common` (Studio's shared UI)                                                             | **No** — unpublished, has no catalog entry, and must not be added as a dependency. Externalized in the plugin's `vite.config.ts`; types come from `paths` in its `tsconfig.json` |

Plugin web dirs are **standalone pnpm roots** and cannot reference the
`catalog:` block, so they restate the versions by hand. When you change one of
those catalog entries, update the matching `dependencies` in every
`plugins/*/web/package.json` in the same change — otherwise a plugin keeps
typechecking against the old version while Studio serves the new one, and
nothing fails until it breaks at runtime. See
[plugins/example-plugin/web/AGENTS.md](../plugins/example-plugin/web/AGENTS.md).

## Running Tests Locally

- **Never invoke `vitest` directly** (e.g. `pnpm vitest --run`). Always go through a package's `test` script so env/config (e.g. `NODE_OPTIONS=--max-old-space-size=10240` in `studio`) is applied.
- Use one of these patterns from `web/`:
  - Whole package: `pnpm --filter <package-name> test` (e.g. `pnpm --filter nemo-studio-ui test`, `pnpm --filter @nemo/common test`)
  - Targeted file: `pnpm --filter <package-name> test path/to/file.test.tsx`

## CI Scripts Convention

CI automatically discovers and runs specific scripts from each package's `package.json`. To opt a package into CI checks, add these scripts:

- **`test:ci`** — Runs tests in CI. Use this for CI-specific config like coverage, reporters, and non-watch mode (e.g., `vitest run --coverage`).
- **`typecheck`** — Runs type checking (`tsc --noEmit`). The same script runs locally and in CI — no separate `typecheck:ci` needed.

If a package defines these scripts, CI will pick them up automatically. No additional CI yaml configuration is required.

## TypeScript Standards

### Type Safety

- **Never use `any`** — use `unknown` and type guards instead
- Prefer `interface` over `type` for object shapes and contracts
- Use `type` for unions, intersections, and computed types
- Implement strict null checks and handle undefined/null explicitly
- Use type assertions sparingly — prefer type guards and narrowing

### Naming Conventions

- `camelCase` for variables, functions, and methods
- `PascalCase` for classes, interfaces, types, and React components
- `SCREAMING_SNAKE_CASE` for constants and environment variables

### Interface and Type Definitions

- Use descriptive property names with clear types
- Mark optional properties with `?` and provide defaults when appropriate
- Use `readonly` for immutable properties

### Functions

- Prefer arrow functions for short, pure functions
- Use function declarations for hoisted functions and methods
- Keep functions small and focused (Single Responsibility Principle)
- Use explicit return types for public APIs and complex functions

### Error Handling

- Use Result types or discriminated unions for predictable errors
- Throw exceptions only for truly exceptional circumstances
- Implement proper error boundaries in React components

### Imports and Exports

- Use named exports over default exports
- Use absolute imports via tsconfig path mapping (never relative)
- **Never hand-sort imports — run `pnpm lint:fix` from `web/` and leave the result alone.**
  `import/order` runs as an _error_ (`eslint.config.js`) with
  `groups: [['builtin', 'external'], 'internal', ['parent', 'sibling', 'index']]`
  and `alphabetize: { order: 'asc' }`. The `@nemo` / `@nvidia` / `@studio` aliases
  resolve as **external**, so they share a single alphabetized group with `react`,
  `lucide-react` and friends — which sorts `react` near the end. That reads
  backwards if you expect "third-party first, ours second", but it is the rule's
  output, and reordering fails CI's `--max-warnings 0`.

### React Patterns

- Define explicit props interfaces for all components
- Use generic components for reusable UI patterns
- Use discriminated unions for variant props

### Code Organization

- One class or main function per file
- Co-locate related types with their implementations
- Organize by feature/domain, not by technical layer
