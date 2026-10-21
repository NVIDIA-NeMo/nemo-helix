---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: unit-test
description: Unit testing workflow with Vitest and React Testing Library. Use when writing, running, or debugging unit tests for Studio components and utilities.
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---

# Unit Testing Workflow

## React Component Testing

- Test components using React Testing Library
- Focus on user behavior, not implementation details
- Use `screen.getByRole()` over `getByTestId()` when possible
- Test accessibility by querying with roles and labels
- Avoid testing internal component state directly
- **Use `await findBy*()` methods for async elements** — never wrap `getBy*()` in `waitFor()`
- Use `findBy*` for elements that appear after async operations (API calls, state updates)
- Use `getBy*` for elements that should be immediately present

```typescript
// DON'T
await waitFor(() => {
  expect(screen.getByText('Data loaded')).toBeInTheDocument();
});

// DO
expect(await screen.findByText('Data loaded')).toBeInTheDocument();
```

## Test Structure

- Follow Arrange-Act-Assert pattern
- Use descriptive test names that explain the scenario
- Group related tests with `describe` blocks
- Use `test.each` for parameterized tests
- Keep tests focused and independent

## Imports (by package)

- **`packages/studio`:** Use tsconfig path aliases — **`@studio/...`** for application code and test utilities under `src/`; **`@nemo/common/...`** for the linked common package (see Studio **`tsconfig.json`** **`paths`**). Prefer aliases over long **`../../../`** chains; import the **module under test** with **`@studio/...`** when that matches app code style and ESLint rules.
- **Other packages (e.g. `packages/common`, `packages/sdk`):** Use **relative** imports for code and specs **inside the same package** (e.g. **`./index`**, **`../form/Widget`**). Those packages do not use **`@studio/...`**; their **`tsconfig`** typically has no `@studio` path map.
- **Cross-package:** In Studio only, use **`@nemo/common/...`** / **`@nemo/sdk/...`** where **`paths`** allow it. In **`packages/common`**, import sibling modules with relatives; use workspace package boundaries as defined by that package’s config (often no internal `@nemo/common` self-alias in specs).

## Shared mocks and factories (centralize)

- **Do not** duplicate large fixture builders or handler factories inside a single spec when the same shape is useful elsewhere.
- **Studio:** Prefer **`@studio/mocks/`** — domain fixtures and helpers live under `packages/studio/src/mocks/` (e.g. `intake/`, `entity-store/`, `handlers/*.ts`). Default HTTP behavior is composed in **`handlers.ts`** and registered via **`@studio/mocks/node`** (`server`).
- **Add** reusable `makeX` / `mockXHandler` functions next to related mocks (or in an existing handler module) and **import** them into specs.
- **Shared test utilities:** e.g. **`@studio/tests/util/render`** (`renderRoute`, etc.) — use these instead of re-creating providers and router wiring in every file.
- **`packages/common`:** colocate small factories next to the module under test or in a nearby `__tests__` helper only when the mock is file-local; if multiple specs need it, extract to a shared test helper under that package.

## MSW handlers (prefer over ad-hoc fetch mocks)

- **Studio:** **Default handlers** run from **`vitest.setup.tsx`**: `server.listen`, **`afterEach` → `server.resetHandlers()`** (restores defaults), `server.close` in `afterAll`. Unhandled requests error by default — add or override handlers instead of silencing unless intentional. Other packages follow their own Vitest setup if configured.
- **Per-test overrides (Studio):** `import { server } from '@studio/mocks/node'` then **`server.use(http.get(...), ...)`** for that scenario (errors, empty lists, slow responses). After the test, reset is automatic.
- **Prefer MSW** for HTTP/API behavior over **`vi.mock` of `fetch`** or low-level axios mocks when the code path hits real request URLs.
- **Extend the graph:** new stable API contracts should get handlers in **`src/mocks/handlers/`** or domain **`src/mocks/<area>/`** and be merged into **`handlers.ts`** when they are broadly needed; one-off overrides stay in the spec with `server.use`.
- Use **`http` / `HttpResponse`** from **`msw`**; align URL prefixes with **`@studio/constants/environment`** (e.g. `INTAKE_MICROSERVICE_URL`, `PLATFORM_BASE_URL`) where the app does.

## Test constants and defaults

- **Workspace string:** use **`DEFAULT_WORKSPACE`** from **`@nemo/common/src/models/constants`** in Studio (and **`./models/constants`** or the same import path from `@nemo/common` when inside **`packages/common`**). Avoid scattering the literal **`'default'`** for workspace slugs, routes, and entity refs unless the test explicitly asserts a non-default workspace.
- **Route params / memory history:** build paths with **`DEFAULT_WORKSPACE`** (or **`generatePath(ROUTES.workspace.*, { workspace: DEFAULT_WORKSPACE, ... })`**) so defaults stay consistent with app constants.
- **Other defaults:** reuse existing exported constants from **`@nemo/common`** or the feature module (prompt templates, etc.) instead of re-hardcoding the same magic values.
- When a test **must** use a non-default workspace to prove isolation, define a **named constant** at the top of the file (e.g. `const OTHER_WORKSPACE = 'other-ws'`) rather than inline strings repeated across cases.

## Mocking Strategy

- Mock external dependencies, never internal modules
- **Prefer MSW** for HTTP (see above); use **`vi.mock()`** for non-HTTP modules (browser APIs, optional native deps), never `jest.mock()`
- Mock only what's necessary — avoid over-mocking
- Reset mocks between tests using `vi.clearAllMocks()`

## API Integration Testing

- Use MSW to mock API responses consistently (shared handlers + `server.use` for variants)
- Test both success and error scenarios
- Validate request payloads and response handling
- Use generated types to ensure API contract compliance
- Test loading states and error boundaries

## Vitest Configuration

- **Do not import test globals** — `describe`, `it`, `expect`, `vi` are available implicitly
- Configure jsdom or happy-dom for component testing environment

## Running Tests

**Always go through the package's `test` script.** Never invoke `vitest` directly — it skips per-package env/config (e.g. `NODE_OPTIONS=--max-old-space-size=10240` in `studio`).

Two acceptable patterns from `web/`:

```bash
# Pattern A — filter from web/ root
pnpm --filter nemo-studio-ui test                                # whole package
pnpm --filter nemo-studio-ui test -- src/components/Button.test.tsx   # specific file
pnpm --filter nemo-studio-ui test -- --reporter=verbose Button   # pattern match
pnpm --filter nemo-studio-ui test -- --coverage                  # with coverage

# Pattern B — cd into the package
cd packages/studio
pnpm test                                                        # whole package
pnpm test -- src/components/Button.test.tsx                      # specific file
```

`pnpm test` already passes `--run` (no watch mode) — tests run to completion so results can be read and iterated on.

## Best Practices

- Write tests alongside feature development
- Test edge cases and error conditions
- Use **central factories** and **MSW handlers** before inventing new inline blobs
- Clean up after tests (reset handlers is automatic for MSW; clear other mocks as needed)
- Keep tests fast — avoid unnecessary async operations
