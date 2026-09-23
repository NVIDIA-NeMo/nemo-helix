<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Studio

NeMo Studio is a UI built on the NeMo Helix, which is aimed at improving agents and making LLM customization much more accessible.

## Getting Started

1. For first setup, install Flox from the repository root and run `make bootstrap-studio`; it provides the pinned Node.js and pnpm versions without requiring global installation. For interactive web work, activate Flox:

   ```bash
   flox -q activate   # from the repository root
   ```

   Or install the Node.js and pnpm versions printed by `make toolchain-versions`, then use `TOOLCHAIN=system make bootstrap-studio` from the repository root.

2. Copy `.env` files from `web/`. `make bootstrap-studio` already installs the
   workspace dependencies. In an activated Flox environment, Corepack provides
   the repository-pinned pnpm:

   ```bash
   pnpm install
   cp packages/studio/env/.env.dev.local.sample packages/studio/env/.env.dev.local && \
     cp packages/studio/env/.env.e2e packages/studio/env/.env.e2e.local
   ```

   `pnpm install` also syncs the KUI `kaizen-ui` agent skill from the `@nvidia/foundations-react-core` design system package into `web/.agents/skills/` (via the root `sync-skills` script). The synced skill is gitignored; re-run `pnpm sync-skills` from `web/` to refresh it after upgrading the package.

## Running Studio Locally

Run the following script from `web/`:

```bash
pnpm dev
```

This runs the vite dev server, which has hot module reloading. This server is accessible at `http://localhost:5173`.

By default your `.env.dev.local` (copied from `.env.dev.local.sample`) points your locally running Studio to a NeMo Helix deployment. You can configure this to point to a local or remote deployment as needed.

## Development

Studio is a React + TypeScript SPA built with [Vite](https://vite.dev/), styled with [Tailwind](https://tailwindcss.com/) and [NVIDIA UI Foundations](https://www.npmjs.com/package/@nvidia/foundations-react-core), and uses [Orval](https://orval.dev/) + [TanStack Query](https://tanstack.com/query/latest) for API integration.

### Adding Dependencies

Use `pnpm add` with a filter for the package name. [More information here.](https://pnpm.io/cli/add)

```bash
pnpm add <pkg> --filter <workspace>
```

### eslint / prettier

We use `eslint` and `prettier` for linting and formatting. `pnpm install` will download those for you, and you can run the `pnpm` scripts found in `package.json` directly.

It's highly recommend you set up your IDE to run these tools on save, so you don't have to worry about manually formatting/linting, and failing CI because your code isn't formatted/linted.

### Orval

[Orval](https://orval.dev/) generates TypeScript types and TanStack Query hooks (`useListCustomizations`, `useCreateProject`, etc.) from each microservice's OpenAPI spec. Scripts and generated code live in `packages/sdk`.

### Logging

Use the `websiteLogger` global from `packages/studio/src/util/logger.ts`. Logs go to the console, and in deployed envs are also exported via OpenTelemetry to a Datadog agent. See `packages/studio/src/telemetry/telemetry.ts`.

### Testing

Stack: [Vitest](https://vitest.dev/) + [React Testing Library](https://testing-library.com/docs/react-testing-library/intro/) + [msw](https://mswjs.io/) for unit/component tests, and [Playwright](https://playwright.dev/) for E2E.

Conventions:

- Co-locate unit tests next to source: `Chat/index.tsx` ↔ `Chat/index.test.tsx`.
- Larger user-workflow tests (create/delete project, chat, etc.) live in `packages/studio/src/tests`, e.g. `create-a-model.test.tsx`.
- E2E specs live in `packages/studio/e2e-tests`.

From `/packages/studio`:

```bash
pnpm test            # run unit tests
pnpm test:watch      # watch mode
pnpm test:visual     # browser UI at http://localhost:51204/__vitest__/#
pnpm test -- --coverage  # coverage report
```

### Environment Variables

Studio is configured using environment variables. How they make their way into the React app depends on the environment Studio is running in. For the full list of environment variables Studio accepts, look at `packages/studio/env/.env.fastapi`.

When the bundle is built for the **Studio FastAPI** app (`pnpm build --mode fastapi`), `STUDIO_UI_*` placeholders come from `packages/studio/env/.env.fastapi` and are resolved at runtime using **`services/studio/src/nhx/studio/env_mappings.py`** at the Platform repository root (alongside `web/`). Keep `.env.fastapi` and `env_mappings.py` in sync (see the parity comment in `.env.fastapi`).

#### Local Development

`.env` files in `packages/studio/env` control which microservice deployments your local Studio points to. `.local` files are gitignored — edit them freely to point at different deployments.

| File                    | Used by                                                               |
| ----------------------- | --------------------------------------------------------------------- |
| `.env.dev.local`        | `pnpm dev` - copied from `.env.dev.local.sample` during setup         |
| `.env.dev.local.sample` | Template for `.env.dev.local`                                         |
| `.env.fastapi`          | `pnpm build --mode fastapi`; must stay in sync with `env_mappings.py` |
| `.env.test`             | Vitest, locally and in CI                                             |
| `.env.e2e`              | E2E tests in CI                                                       |
| `.env.e2e.local`        | E2E tests locally - copied from `.env.e2e`                            |
