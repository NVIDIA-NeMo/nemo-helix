---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: screenshot
description: Capture a headless Playwright screenshot of a Studio page or a single element by data-testid. Use whenever you need an image of the web UI — attaching to a PR, verifying a visual change, comparing light/dark themes, or showing a bug.
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---

# Screenshot

`pnpm screenshot` (`packages/studio/screenshots/screenshot.ts`) loads a Studio route in headless Chromium and saves a PNG of the viewport, the full page, or one `data-testid` element. Run it from `web/packages/studio/`.

## Prerequisites

- Studio must be running. Default target is the Vite dev server at `https://localhost:5173/` (`pnpm start`); point elsewhere with `--base-url` or `STUDIO_URL` (e.g. `http://localhost:8080/studio/`).
- Pages that fetch data need the platform backend up, otherwise they render Studio's error page.
- In a sandboxed agent shell, `tsx` fails with `listen EPERM ... tsx-*.pipe`. Rerun outside the sandbox.

## Workflow

1. **Find the route** from the router (`packages/studio/src/routes/`). Routes are relative to the base URL, e.g. `workspaces/default/agents`.
2. **Find the test id**: grep the component for `data-testid`. If the element you need has none, add one to the component rather than capturing the whole page.
3. **Capture**:

   ```sh
   # Viewport
   pnpm screenshot --route workspaces/default/agents

   # One element, with padding, in both themes
   pnpm screenshot --route workspaces/default/agents/my-agent --testid agent-detail-header --padding 16
   pnpm screenshot --route workspaces/default/agents/my-agent --testid agent-detail-header --padding 16 --theme dark

   # Click through to a state first (each --click is a data-testid, run in order)
   pnpm screenshot --route workspaces/default/jobs --click quick-actions-menu-trigger --delay 800
   ```

4. **Look at the output** with your image-reading tool before using it. The path is printed on success; it defaults to `screenshots/out/<testid|page>-<theme>.png` (gitignored). Use `--out` for a specific name.

Run `pnpm screenshot --help` for all options (`--width`, `--height`, `--full-page`, `--timeout`, ...). Captures are 2x device scale.

## Behavior

- The welcome tour is suppressed and the theme is set through localStorage (`--theme`, default `light`), so no setup clicks are needed for those.
- React Query devtools are hidden.
- On failure, the current page is saved next to the output as `*.error.png`. Read it — it usually shows the cause (wrong route, error page, signed-out screen, missing test id).
- Radix menus and dialogs animate in; increase `--delay` if a `--click`-opened overlay looks half-transparent.

## Auth

If the platform has auth enabled, a human must sign in once:

```sh
pnpm screenshot --login
```

This opens a headed browser, saves the session to `screenshots/out/auth.json`, and every later run reuses it. Agents cannot complete this step — ask the user to run it (`! pnpm screenshot --login` from `web/packages/studio/`).

## Not this skill

- Documentation images in `docs/studio/_images/` use `pnpm screenshots` (`screenshots/capture.ts`, configured in `screenshots/config.ts`).
- Visual regression assertions belong in E2E tests (see `e2e-test`).
