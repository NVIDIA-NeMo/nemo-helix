<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# UI Package Agent Instructions

## About This Package

- Main React frontend application for NeMo Studio
- **Read [README.md](README.md) before proposing or building a feature.** It is Studio's product charter: mission, responsibilities, principles, and direction.
- Built with React 18, TypeScript, Vite
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
