<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Studio plugin web UI — agent instructions

This is the web UI for a NeMo Studio plugin. Studio loads the built bundle at
runtime and renders it **inside its own React tree**. This dir is also the
canonical template other plugins copy — keep it exemplary.

Runtime contract: `../../../web/packages/studio/src/plugins/types.ts`.

## Mental model

- **One root, not two.** Export a `Root` **component**; Studio renders `<Root/>`
  under its own Router / QueryClient / KaizenThemeProvider. Never call
  `createRoot`, never create a `BrowserRouter`.
- **Share the singletons that carry context** — React, react-dom, react-router,
  `@nvidia/foundations-react-core`, `@tanstack/react-query`, and `@nemo/common`
  (Studio's shared UI) resolve via Studio's import map. This dir **externalizes**
  them so the plugin uses Studio's one instance (shared router + theme + query
  cache + tables). Everything else bundles privately.

## Rules — DO / DON'T

| Concern | DO | DON'T |
| --- | --- | --- |
| Entry | `export { Root }` (component) + `export { navItems }` from `src/index.ts` | export `mount()` or call `createRoot` |
| Routing | Studio's shared router — `Routes`/`Route`/`NavLink`/`Navigate`/`Outlet`/`useNavigate`. Route `path`s are relative; every `to` / `href` must be **absolute** (see below) | `BrowserRouter`, `history.pushState` patching, hardcoded `basename`, relative `to` |
| Components | KUI from `@nvidia/foundations-react-core` — `Text`, `Stack`, `Flex`, `Button` | hand-rolled styled `<div>`s or native `<button>` |
| Tables, forms, status | Studio's shared UI from `@nemo/common` — `StudioDataView`, `useStudioDataViewState`, `ControlledTextInput`, `StatusBadge`, … | re-implement a table/empty state/relative timestamp, or deep-import `@nemo/common/src/...` |
| Styling | Studio's theme-aware tokens: `bg-surface-base/raised/sunken/hover`, `text-subtle/muted/primary`, `border-subtle` | hardcoded Tailwind palette (`bg-gray-100`, `text-blue-700`) — not compiled for the plugin, not theme-aware |
| Auth | `host.auth.getAccessToken()` **per request** → `Authorization: Bearer …` | `react-oidc-context` / `useAuth` (refresh token must not cross the boundary) |
| Deps | externalize the shared set in `vite.config.ts`; bundle the rest | bundle react / react-dom / react-router / foundations |

Studio injects everything a plugin needs through a **single `host` prop**
(`host.workspaceId`, `host.apiBaseUrl`, `host.auth`, `host.sdk`, `host.navigation`,
`host.notifications`, `host.telemetry`, `host.breadcrumbs`) — grouped so new capabilities extend the
handle without changing `Root`'s signature. Destructure what you use. All are
backed by Studio's own singletons: `notifications` fires into Studio's shared
toaster, `telemetry` logs to Studio's OTEL pipeline (auto-scoped to the plugin),
`navigation` drives Studio's shared router, and `breadcrumbs` writes Studio's
breadcrumb bar — which lives in GlobalNav, *outside* the plugin's subtree, so a
plugin cannot render it itself. Studio clears the trail when the plugin
unmounts, but **not between pages within the plugin**: return a cleanup that
clears it, or the trail follows you to the next page. See `src/SharedUiPage.tsx`.

`@tanstack/react-query` **is** shared — call `useQuery`/`useMutation` and it reads
Studio's `QueryClientProvider` (one cache across Studio and every plugin). Put the
`host.auth.getAccessToken()` Bearer token in your `queryFn`.

Studio's **SDK** arrives on `host.sdk`, not via an import. `host.sdk.platform` and
`host.sdk.agents` are those services' generated hooks and functions; call the hooks
at the top level like any hook
(e.g. `host.sdk.platform.useEntitiesListWorkspaces({ page: 1, page_size: 100 })`,
see `src/Root.tsx`). They run on Studio's one configured axios instance (base URL +
OIDC interceptor) and the shared QueryClient — a plugin never bundles or
configures the SDK itself. This works because React is a shared singleton, so a
hook from Studio's module graph dispatches into the plugin's tree. `@nemo/sdk` is
a private, unpublished package, so it is **not** a dependency here: `src/types.ts`
declares a minimal structural `PluginSdk` covering only the hooks this example
calls. A real plugin either mirrors the calls it needs the same way or, if it can
resolve the SDK's types, types `host.sdk` as Studio does.

`host.sdk` covers **platform** services only. A plugin that calls *its own*
service ships its own client, and must prefix every request with
`host.apiBaseUrl` — Studio's dev-server `/apis` proxy is opt-in, so a bare
`/apis/...` request hits the dev server rather than the platform whenever
`VITE_PLATFORM_BASE_URL` is set. See `plugins/nemo-agent-hardener/web` for a
generated client wired this way.

## Shared UI (`@nemo/common`)

Studio's own table, form, and status components are shared the same way KUI is.
Studio builds the curated surface in
`../../../web/packages/common/src/plugin.ts` into `public/vendor/common.js` and
maps the bare specifier to it, so a plugin's `StudioDataView` **is** the module
instance Studio renders — same behavior, same styles, no second copy in the
bundle. See `src/SharedUiPage.tsx`.

```ts
import { AssistantChat, StudioDataView, useStudioDataViewState } from '@nemo/common';
```

- **Bare specifier only.** A deep `@nemo/common/src/...` import is not
  externalized, so it silently bundles a *second copy* of the component instead
  of sharing Studio's — it does not error, it just quietly stops being shared.
  The `reject-deep-shared-imports` plugin in `vite.config.ts` fails the build on
  one; keep it when you copy this template. Importing a name the barrel doesn't
  export is already a tsc error, so `pnpm typecheck` covers that half.
- **`plugin.ts` is the API.** Need something Studio has but the barrel doesn't
  export? Add it there — additions are cheap, removals are breaking.
- **Nothing in the barrel may call the API.** The vendor build resolves
  `@nemo/sdk` to source rather than externalizing it, and defines only
  `process.env.NODE_ENV` — so a bundled fetcher would read an undefined
  `import.meta.env` for its base URL and OIDC keys. Type-only SDK imports are
  fine (they erase). A shared component that needs data takes it as a prop or
  callback and the caller supplies it from `host.sdk`; `CreateSecretModal`'s
  `onCreate` and `fetchAllPages`' page fetcher are the pattern.
- **`AssistantChat` is shared.** A plugin can point it at an authenticated,
  OpenAI-compatible `baseURL`; Studio supplies its current access token and
  chat runtime. Use `messageContentProps.markdownLinkComponent` when a plugin
  owns trusted, in-app citation targets. The plugin should still own the panel,
  prompts, endpoint, and citation behavior specific to its feature.
- **Types come from source**, via `paths` in `tsconfig.json`; `@nemo/common` is
  unpublished, so there is nothing to install. `src/env.d.ts` declares the `*.css`
  side-effect imports those sources carry.
- **Out-of-tree plugins vendor a generated `.d.ts` instead.** A plugin living in
  its own repository can't use `paths` into these sources — resolving them needs
  this workspace's `node_modules`, including the unpublished `@nemo/sdk`. The
  whole surface is rolled up into `packages/common/plugin-types/plugin.d.ts`,
  which is committed here; a plugin repo copies that file in and points its
  `paths` at it. Whatever the rolled-up file still imports, the consumer has to
  resolve — check the `import` lines at the top of it rather than assuming this
  list is current. Published packages (`class-variance-authority`,
  `@assistant-ui/react` once the chat surface lands) are plain type-only
  devDependencies. Only `@nemo/sdk/generated/platform/schema` needs a local stub,
  because it is unpublished; it contributes `PlatformJobLog` and
  `PlatformJobStatus`, reached solely through `LogViewer` and the job-status
  constants, so a dozen structural lines cover it.
- **Regenerate with `pnpm --filter @nemo/common types:plugin` when you change
  `plugin.ts`.** The `web-plugin-types` CI job regenerates and fails on a diff,
  so the artifact can't drift from the surface it describes. It can still drift
  from a plugin's *copy* — nothing in this repo knows about those — and a stale
  copy compiles happily against a surface that no longer exists, so refresh it
  deliberately when the surface moves.
- **CSS is already loaded.** The vendor build stubs stylesheet imports because
  Studio bundles the same files through its own graph. A plugin adds no CSS.
- **`useStudioDataViewState` syncs to URL search params** on Studio's shared
  router — two DataViews on one route will fight over them.
- **Toasts need `onNotify`.** `ToastProvider` is *not* shared: Studio mounts it
  by deep import, so this bundle carries its own `ToastContext` with nothing in
  it. `ConfirmationModal`, `DeleteConfirmationModal`, `CreateSecretModal` and
  `LogViewer` therefore take an `onNotify` prop — pass `host.notifications.notify`
  and the message lands in Studio's toaster. Omit it and the message is dropped
  with a `logger.warn`; nothing throws, so the miss is silent in the UI.

  ```tsx
  <DeleteConfirmationModal onNotify={host.notifications.notify} ... />
  ```

## Contract

```ts
// PluginRootProps (from Studio's types.ts)
{
  host: {
    workspaceId: string;
    apiBaseUrl: string;
    auth: { accessToken: string; getAccessToken: () => string };
    sdk: {
      platform: /* @nemo/sdk platform hooks */;
      agents: /* @nemo/sdk agents hooks and functions */;
    };
    navigation: { navigate: (to: string) => void; back: () => void };
    notifications: {
      notify: (
        message: string,
        type?: 'success'|'error'|'info'|'warning',
        options?: { durationMs?: number | false },
      ) => void;
    };
    telemetry: { info; warn; error: (m, cause?) => void; event: (name, attrs?) => void };
    breadcrumbs: { set: (trail: { label: string; href?: string }[]) => void };
  };
}
```

`src/index.ts` must export `Root` (a `ComponentType<PluginRootProps>`) and
`navItems(workspaceId) => PluginNavGroup[]`. See `src/Root.tsx` and `src/Nav.tsx`.

## Externals & versions

The `external` list in `vite.config.ts` **must match the keys of** Studio's
`VENDOR_IMPORT_MAP` in `../../../web/packages/studio/vite.config.ts` — that map,
not `VENDOR_EXTERNALS`, is what the browser resolves at runtime. (`@nemo/common`
is in the map but deliberately not in `VENDOR_EXTERNALS`: Studio imports Common
by deep path and bundles it normally, so externalizing the bare name would only
add a dead import to every Studio chunk.) If Studio shares something new, add it
here too or the plugin bundles its own copy and loses the shared
instance/theme/cache. The SDK is **not** in this list — it comes in on the `sdk`
prop (see above), so there is nothing to externalize for it.

**Shared deps must also match Studio's versions**, which is a separate
obligation from externalizing them. Studio *serves* these at runtime, so the
plugin only ever types and builds against them — declare a version Studio does
not ship and you get types that describe a different library than the one
executing. Keep the `dependencies` below in sync with the `catalog:` block in
`../../../web/pnpm-workspace.yaml`, copying the range verbatim:

| Dep | Why exact-match matters |
| --- | --- |
| `@nvidia/foundations-react-core` | Studio pins an **exact** version (no caret). A caret here silently floats the plugin ahead of the KUI Studio actually serves. |
| `react`, `react-dom`, `react-router` | Hooks, context, and router internals must be one instance. |
| `@tanstack/react-query` | Shares Studio's QueryClient; the hook and the provider must agree. |
| `@nemo/common` | Not a package — resolved from Studio's vendor bundle, so it is never a dependency. Types come from `paths` in `tsconfig.json`. |

This dir is a standalone pnpm root, so it cannot use `catalog:` references and
has to restate the versions.

Build-only tooling (`vite`, `typescript`, `@vitejs/plugin-react`) and the
`@types/*` packages are not served by Studio, so drift there is a correctness
issue only for the types. They are still kept catalog-aligned — every dep in
this `package.json` currently matches the catalog exactly, which makes an audit
a diff rather than a judgement call.

## Build & verify

This dir is its own pnpm root (`pnpm-workspace.yaml`), separate from `web/`.
Use pnpm — `pnpm-lock.yaml` is the only lockfile, and an `npm install` here
writes a competing `package-lock.json` that resolves different versions.

```bash
pnpm install         # first time — pulls @nvidia/foundations-react-core etc.
pnpm build           # emits ../src/<pkg>/web/dist/index.js (shipped in the wheel)
pnpm typecheck       # tsc --noEmit, incl. the shared-UI types from @nemo/common

# shared deps must stay external (bare specifiers, not bundled):
grep -oE 'from *"[^"]*"' ../src/<pkg>/web/dist/index.js | sort -u
```

The bundle is registered via the plugin's `nemo.studio` entry point
(`studio.py` → `StudioSpec`), served at `/plugin-ui/<name>/index.js`, and the
UI route is gated behind the `pluginsEnabled` flag (on by default).

## Gotchas

- **Links must be absolute — a relative `to` silently appends.** Studio mounts
  plugins at a splat route (`/workspaces/:workspaceId/plugin/:pluginName/*`),
  and React Router resolves a relative `to` against the splat's *full* matched
  pathname, not the mount point (`getResolveToMatches` uses `match.pathname` for
  the last match, not `match.pathnameBase`). So on `/plugin/example/auth`,
  `<NavLink to="shared-ui">` navigates to `/plugin/example/auth/shared-ui`, and
  each further click appends again. Build hrefs from `host.workspaceId` — see
  `src/paths.ts`, used by both `Root.tsx` and `Nav.tsx`. Route `path`s are
  unaffected; only `to` / `href` resolution is.

- **Token classes must be ones Studio already compiles.** Studio's Tailwind only
  scans `web/packages/**`, not this dir. Stick to the semantic tokens Studio uses
  (grep `web/packages/studio/src` for `bg-surface`, `text-subtle`) and KUI
  components (self-styled). Arbitrary utility classes won't have CSS.
- **Never expose refresh tokens.** Only `accessToken` / `getAccessToken` cross the
  boundary — never reach for Studio's OIDC context.
- **New shared singletons** with internal dynamic imports need `codeSplitting:
  false` in Studio's vendor build; here you only add the name to `external`.
