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
  (Studio's shared UI) resolve via Studio's import map. `studioPlugin()` in
  `vite.config.ts` **externalizes** them so the plugin uses Studio's one instance
  (shared router + theme + query cache + tables). Everything else bundles
  privately.

## Rules — DO / DON'T

| Concern | DO | DON'T |
| --- | --- | --- |
| Entry | `export { Root }` (component) + `export { navItems }` from `src/index.ts` | export `mount()` or call `createRoot` |
| Routing | Studio's shared router — `Routes`/`Route`/`NavLink`/`Navigate`/`Outlet`/`useNavigate`. Route `path`s are relative; every `to` / `href` must be **absolute** (see below) | `BrowserRouter`, `history.pushState` patching, hardcoded `basename`, relative `to` |
| Components | KUI from `@nvidia/foundations-react-core` — `Text`, `Stack`, `Flex`, `Button` | hand-rolled styled `<div>`s or native `<button>` |
| Tables, forms, status | Studio's shared UI from `@nemo/common` — `StudioDataView`, `useStudioDataViewState`, `ControlledTextInput`, `StatusBadge`, … | re-implement a table/empty state/relative timestamp, or deep-import `@nemo/common/src/...` |
| Styling | Studio's theme-aware tokens: `bg-surface-base/raised/sunken/hover`, `text-subtle/muted/primary`, `border-subtle` | hardcoded Tailwind palette (`bg-gray-100`, `text-blue-700`) — not compiled for the plugin, not theme-aware |
| Auth | `host.auth.getAccessToken()` **per request** → `Authorization: Bearer …` | `react-oidc-context` / `useAuth` (refresh token must not cross the boundary) |
| Deps | `studioPlugin()` in `vite.config.ts` externalizes the shared set; bundle the rest | bundle react / react-dom / react-router / foundations |

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
  externalized, so it would bundle a *second copy* of the component instead of
  sharing Studio's. `studioPlugin()` in `vite.config.ts` fails the build on one
  (and on deep imports of the other shared packages); keep it when you copy this
  template. Importing a name the barrel doesn't export is already a tsc error, so
  `pnpm typecheck` covers that half.
- **`plugin.ts` is the API.** Need something Studio has but the barrel doesn't
  export? Add it there — additions are cheap, removals are breaking.
- **Nothing in the barrel may call the API.** The vendor build resolves
  `@nemo/sdk` to source rather than externalizing it, and defines only
  `process.env.NODE_ENV` — so a bundled fetcher would read an undefined
  `import.meta.env` for its base URL and OIDC keys. Type-only SDK imports are
  fine (they erase). A shared component that needs data takes it as a prop or
  callback and the caller supplies it from `host.sdk`; `CreateSecretModal`'s
  `onCreate` and `fetchAllPages`' page fetcher are the pattern.
- **Model pickers take their fetcher.** `WorkspaceModelSelect` and
  `useModelSearch` search the server and load models a page at a time as the
  list scrolls; pass `listModels={host.sdk.platform.modelsListModels}`.
- **`AssistantChat` is shared.** A plugin can point it at an authenticated,
  OpenAI-compatible `baseURL`; Studio supplies its current configured bearer and
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
  devDependencies. Only the unpublished `@nemo/sdk` schemas need local stubs:
  stub the names on the file's `@nemo/sdk/generated/*/schema` import lines with
  structural types.
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

Plugin builds use `studioPlugin()` from
`../../../web/packages/common/plugin-build/studio-plugin.mjs`. It externalizes
every specifier in Studio's plugin contract (`PLUGIN_CONTRACT`, in
`../../../web/packages/common/src/pluginContract/contract.ts`), so there is no
hand-kept `external` list to drift from Studio's `VENDOR_IMPORT_MAP` in
`../../../web/packages/studio/vite.config.ts` (which the browser resolves at
runtime). In-repo plugins import the helper by relative path; out-of-tree plugins
vendor a copy of that file. It also fails the build when a shared package is
imported by a deep path or bundled from `node_modules`, and emits
`plugin.manifest.json` next to `index.js`, recording the contract version, the
bare imports left in the bundle, and the versions built against. The SDK is
**not** in the contract — it comes in on the `sdk` prop (see above), so there is
nothing to externalize for it.

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

## Shared-dependency contract

Studio runs your bundle in its own page, so some packages must be Studio's copy.

- **Singleton** (identity matters: the plugin must use Studio's instance): `react`,
  `react-dom`, `react-router`, KUI (`@nvidia/foundations-react-core`),
  `@tanstack/react-query`.
- **Provided** (`@nemo/common`): Studio serves a separate vendor build of Common's
  source, so module state is **not** shared with Studio's own copy. State crosses
  via the `host` prop.
- Everything else is bundled into the plugin and is out of the contract.

**Version rule.** Only a different **major** between the version you built against
and the version Studio ships is an error. The contract has its own semver: it is
supported iff the major is the same and the plugin's minor is <= Studio's minor.

**Enforcement.**

1. Build gate: `studioPlugin()` fails the build on deep imports and on bundled singletons.
2. Sidecar: the build emits `plugin.manifest.json` next to `index.js` (commit it next
   to a committed bundle).
3. Load time: Studio fetches the sidecar and checks it before the bundle runs.

**Error codes.**

| Code                          | Cause                                                      | Fix                                                                                         |
| ----------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `PLUGIN_MANIFEST_MISSING`     | `plugin.manifest.json` not found next to `index.js`        | Add `studioPlugin()` from `studio-plugin.mjs` to the plugin's Vite config, rebuild          |
| `PLUGIN_SPECIFIER_UNMAPPED`   | The bundle imports a name Studio does not provide, or deep-imports `@nemo/common` (e.g. `@nemo/common/components/Foo`) | Import it as a normal dependency so it is bundled, or request it be added to the contract; for `@nemo/common`, import from the bare `@nemo/common` specifier, and if the name is missing add it to `web/packages/common/src/plugin.ts` |
| `PLUGIN_SINGLETON_BUNDLED`    | A singleton's source was bundled into the plugin           | Import only contract specifiers (no deep paths, no `node_modules`), rebuild                 |
| `PLUGIN_CONTRACT_UNSUPPORTED` | Built against a contract version Studio does not support   | Update the vendored `studio-plugin.mjs` to the current one, rebuild                         |
| `PLUGIN_VERSION_INCOMPATIBLE` | A shared package has a different major than Studio ships   | Set that package to Studio's version in the plugin's `web/package.json`, rebuild            |
| `PLUGIN_LOAD_FAILED`          | Untrusted URL, unreadable sidecar, or the bundle threw/lacks `Root`/`navItems` | Check the browser console for the underlying error, rebuild, refresh Studio |

**Where a failure shows up.** The plugin's route panel, a `FAILED PLUGINS` sidebar
group, and one line in the platform terminal (reported through
`POST /studio/api/plugin-load-errors`; on only when
`studio.plugin_error_reports_enabled` is true, or unset with auth disabled).

**Development loop.** Run `vite build --watch` in the plugin's `web/`, then refresh
Studio (the plugin UI is a prebuilt bundle, no HMR). `NODE_ENV=development` builds
fail (the dev JSX runtime is not provided); build in the default production mode.

**Out-of-tree plugins** vendor `web/packages/common/plugin-build/studio-plugin.mjs`
and refresh it when the contract changes (`pnpm --filter @nemo/common build:plugin-helper`
regenerates it).

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
- **Never expose refresh tokens.** Only the configured platform bearer (an access
  token or ID token) crosses through `accessToken` / `getAccessToken` — never
  reach for Studio's OIDC context.
- **New shared singletons** with internal dynamic imports need `codeSplitting:
  false` in Studio's vendor build; here you only add the name to the plugin
  contract.
