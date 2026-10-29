<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Screenshots Setup - Agent Instructions

## About

This directory contains a Playwright-based screenshot capture script for Studio documentation. It runs via `pnpm screenshots` from `web/packages/studio/`.

## Key Patterns for Writing Setup Functions

### KUI Select Dropdowns (NvSelect)

KUI selects use Radix primitives. The trigger has `data-testid="nv-select-trigger"` and options render in a portal with `role="option"`.

```ts
// Open the dropdown
await page.getByTestId('nv-select-trigger').click();

// Pick an option by its visible text
await page.getByRole('option', { name: 'option-name' }).click();
```

- Do NOT use `getByText()` for dropdown options — the text may match elsewhere in the DOM.
- If there are multiple select triggers on the page, scope to a parent locator first.

### Table Row Checkboxes

File tables use `<tr>` rows with `nv-checkbox-box` test IDs. Checkboxes are not labeled — they all share the same `aria-labelledby`, so you must scope to the row first.

```ts
const row = page.locator('tr', { hasText: 'filename.jsonl' });
await row.getByTestId('nv-checkbox-box').click();
```

- Use `.first()` if the text could match multiple rows (e.g., `evaluation.jsonl` vs `evaluation-offline-simple.jsonl`).

### Table Row Action Menus ("..." Button)

Action menu triggers have `data-testid="quick-actions-menu-trigger"`. The menu renders in a Radix portal.

```ts
const row = page.locator('tr', { hasText: 'filename.jsonl' }).first();
const menuTrigger = row.getByTestId('quick-actions-menu-trigger');
await menuTrigger.waitFor({ timeout: 3_000 });
await menuTrigger.click();
```

### Animated Menus and Overlays

Radix menus and modals use CSS animations. Text content appears in the DOM before the animation completes, so `waitFor()` and `isVisible()` return true while the element is still animating (e.g., opacity 0 -> 1). **Always add a delay after detecting content** to let animations finish:

```ts
// Wait for menu content to exist, then let animation complete
await page.getByText('Menu Item Text').waitFor({ timeout: 5_000 });
await page.waitForTimeout(500);
```

This applies to:

- Dropdown menus (Radix DropdownMenu)
- Modal overlays (Radix Dialog) — both opening and closing animations
- Select content popovers

### Modal Open/Close Assertions

Prefer asserting on structural elements (headings, roles) rather than page-specific body text that may change:

```ts
// Wait for modal to open
await page.getByRole('heading', { name: 'Modal Title' }).waitFor({ timeout: 5_000 });

// Wait for modal to close
await page.getByRole('heading', { name: 'Modal Title' }).waitFor({
  state: 'hidden',
  timeout: 10_000,
});

// Always wait for dismiss animation after modal closes
await page.waitForTimeout(500);
```

### Privacy Replacements and Setup Order

Privacy replacements (DOM text node walker) run **before** setup functions. This prevents the text walker from triggering React re-renders that close open menus/popups. If you open a menu in setup, it will still be open when the screenshot is taken.

### Locators vs Role Queries

- **Never use `page.locator()` directly.** Always prefer user-facing queries: `page.getByRole()`, `page.getByTestId()`, `page.getByText()`, `page.getByLabel()`.
- These queries follow Playwright's best practices and are more resilient to DOM changes.
- For scoping within a container (e.g., a table row), use `page.getByRole('row', { name: '...' })` and chain queries on it.
- If `getByRole('row')` is not feasible, `page.locator('tr', { hasText: '...' })` is acceptable as a last resort — but always try role-based queries first.

## Build Exclusions

This directory is excluded from:

- TypeScript compilation (`tsconfig.json` exclude)
- Vite production builds (only `src/` is bundled)

Imports use the `@screenshots/*` path alias (defined in `tsconfig.json`), following the same pattern as `@e2e-tests/*`. Console output uses `/* eslint-disable no-console */` since this is a CLI script.

## Ad-hoc Screenshots (`pnpm screenshot`)

`screenshots/screenshot.ts` is a separate headless, non-interactive capture of any page or `data-testid` element, meant for agents (PR descriptions, visual checks, bug reports). See the `screenshot` skill in `web/.agents/skills/screenshot/SKILL.md` for usage.
