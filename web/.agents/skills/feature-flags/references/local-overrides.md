<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Local Feature-Flag Overrides

A localStorage layer on top of the normal env cascade. Lets a developer flip any
Studio flag without a rebuild or a redeploy.

**Changes require a page reload.** Flags are parsed once at module load so
`featureFlags` stays a plain immutable value rather than reactive state every
consumer would have to subscribe to.

## Enabling

Gated by `flagOverridesEnabled` (`VITE_FF_FLAG_OVERRIDES_ENABLED`), which
defaults to `false`. It is read **only** from the build environment, never from
an override — otherwise anyone could set it to `true` in production and unlock
every other flag.

| Environment             | How to enable                                                                                                 |
| ----------------------- | ------------------------------------------------------------------------------------------------------------- |
| Vite dev server         | `VITE_FF_FLAG_OVERRIDES_ENABLED='true'` in `env/.env.dev.local`                                               |
| Local served build      | Already on — `studio.feature_flags.flag_overrides_enabled: true` in `packages/nhx_platform/config/local.yaml` |
| Deployed (dev/internal) | `studio.feature_flags.flag_overrides_enabled: true` in that target's values yaml                              |
| Production              | Leave off                                                                                                     |

## Devtools: `window.__flags`

Attached only when the gate is on.

```js
__flags.list(); // every flag: type, base value, effective value, overridden
__flags.set('dashboardEnabled', 'preview');
__flags.unset('dashboardEnabled');
__flags.reset(); // clear all overrides
__flags.manifest(); // full manifest object
location.reload(); // required for any change to take effect
```

`set` rejects unknown flag names and values that fail the flag's schema, so a
typo can't brick the app.

## Chrome extension contract

A Chrome content script runs in an **isolated world** and cannot see page-world
globals — `window.__flags` is for humans only. The extension instead uses two
localStorage keys, which it _can_ read and write directly (same origin).

### `feature-flag-manifest` — written by Studio, read-only to the extension

```jsonc
{
  "version": 1,
  "overridesEnabled": true,
  "overridesKey": "feature-flag-overrides",
  "flags": [
    {
      "key": "dashboardEnabled", // use this in the overrides blob
      "envVar": "VITE_FF_DASHBOARD_ENABLED",
      "type": "preview", // boolean | preview | string | number
      "baseValue": false, // value with no override — what "reset" restores
      "effectiveValue": "preview", // what the app is actually running with
      "overridden": true,
    },
  ],
}
```

`type` tells the extension which control to render: `boolean` → toggle,
`preview` → tri-state (`true` / `preview` / `false`), `string` / `number` → text
input.

When the gate is off, Studio still writes a stub
(`{"version":1,"overridesEnabled":false,"overridesKey":"","flags":[]}`) so the
extension can distinguish "overrides are disabled here" from "not a Studio page".

Check `version` against `FLAG_MANIFEST_VERSION` and refuse to render on a
mismatch rather than guessing at an unfamiliar shape.

### `feature-flag-overrides` — written by the extension, read by Studio

```jsonc
{ "dashboardEnabled": "preview", "toolCallingEnabled": "true" }
```

Values are **raw strings**, keyed by the manifest's `key`. Studio feeds them
through the same Zod schemas as real env vars, so `"preview"`, `"true"`,
`"false"`, numbers, and strings all behave identically to setting the env var.

Remove the key entirely to clear all overrides rather than storing `{}`.

After writing, call `chrome.tabs.reload()` — nothing takes effect until the page
reloads. The manifest only refreshes on the next load, so the extension should
treat its own just-written blob as the source of truth for pending state.

### Failure handling in Studio

Deliberately forgiving, because a dev tool must never white-screen the app:

| Input                             | Result                              |
| --------------------------------- | ----------------------------------- |
| Malformed JSON                    | Entire blob ignored, warning logged |
| Not a JSON object (array, scalar) | Entire blob ignored, warning logged |
| Unknown flag key                  | That entry dropped, warning logged  |
| Non-string value                  | That entry dropped, warning logged  |
| Value failing the flag's schema   | That entry dropped, warning logged  |
| localStorage unavailable (throws) | Treated as no overrides             |

That last column matters: `parseFlags` treats a schema failure as a fatal
"missing required flag", so bad overrides are filtered out _before_ they reach it.

## No UI indicator

By design, nothing in Studio's chrome signals that overrides are active. A stale
override will silently alter behavior until someone runs `__flags.list()` or
opens the extension — check this first when triaging a confusing bug report from
someone who has the extension installed.

## Source

- `packages/studio/src/constants/featureFlags/overrides.ts` — read, validate, manifest, global
- `packages/studio/src/constants/featureFlags/index.ts` — two-pass parse cascade
- `packages/studio/src/main.tsx` — `initFlagOverrideTooling()` at boot
