<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# UI Package Agent Instructions

## About This Package

- Main React frontend application for NeMo Studio
- Built with React 18, TypeScript, Vite
- Handles user interfaces for model customization, evaluation, and deployment
- Integrates with backend APIs through generated SDK clients

## Design System

- **Use NVIDIA Foundations React as the primary design system** — preferred over custom components
- NVIDIA Foundations React provides consistent styling, accessibility, and user experience patterns
- Follow NVIDIA Foundations React's design tokens and theming system for consistent visual design
- Leverage NVIDIA Foundations React's built-in accessibility features and ARIA patterns

## Key Dependencies

- **NVIDIA Foundations React** (@nvidia/foundations-react-core) — Primary component library
- **TanStack Query** — Data fetching and caching
- **React Hook Form + Zod** — Form handling and validation
- **React Router 6** — Client-side routing

## Import Path Rules

- **Never use relative imports** — always use absolute imports
- Import path mappings:
  - `@studio/` → `packages/studio/src/`
  - `@e2e-tests/` → `packages/studio/e2e-tests/`
- Other local packages are imported via pnpm workspaces

### KUI v1 Select: always wrap items in SelectListbox

In KUI v1, `SelectContent` is a transparent popover host — it has no background.
Background, border, and shadow come from `SelectListbox` → `MenuRoot` internally.

**Required pattern:**

```tsx
<SelectContent>
  <SelectListbox>
    <SelectItem value="a">A</SelectItem>
  </SelectListbox>
</SelectContent>
```

Wrong — transparent dropdown:

```tsx
<SelectContent>
  <SelectItem value="a">A</SelectItem> {/* no SelectListbox = no background */}
</SelectContent>
```

Exception: call sites that fill SelectContent with custom children containing their
own background (e.g. a sticky Block with bg-surface) are fine as-is.

## Loading a page's data and handling errors

Every workspace route renders inside `SuspenseOutlet` (loading fallback) and has a
`RouteErrorPanel` as its `errorElement`. The panel shows "Not Found" for a 404,
"No Access" for a 403, and otherwise the error with Retry and Go Back.

- **Load the resource the URL names with the generated `use<Name>Suspense` hook**
  (e.g. `useJobsGetJobSuspense`), not `use<Name>`. Its `data` is always defined,
  and a failed load or missing resource falls through to the route's error panel,
  so the page needs no loading, not-found, or error branches of its own.
- Suspense hooks share their query key with the non-suspense hook, so existing
  invalidations and other components reading the same query keep working. They
  don't accept `enabled`; a background refetch error keeps the loaded data.
- Secondary data (a tab's table, logs, results) stays on the regular hook with an
  inline state: `<ErrorPanel error={error} onRetry={refetch} />` or a DataView's
  `renderErrorState`.
- Set a DataView's `requestStatus` with `getRequestStatus(query)` from
  `@nemo/common/src/utils/query`. It shows loading only on the first load or while
  placeholder rows are shown for a new page, never on a background refetch or
  poll. ESLint rejects `isFetching` in `requestStatus` because it blanks a loaded
  table on every refetch.
- In tests, give the route `errorElement: <RouteErrorPanel />` to assert error
  states, and mock the `...Suspense` hook (returning `{ data }`) when the test
  mocks hooks instead of using MSW.
- The query client doesn't retry 4xx responses (other than 408/429), so a
  not-found page shows immediately.
