<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Local Feature-Flag Overrides

A localStorage layer on top of the normal env cascade, edited from a panel in the
global header. Lets a developer flip any Studio flag without a rebuild or a
redeploy.

**Changes apply on reload.** Flags are parsed once at module load so
`featureFlags` stays a plain immutable value rather than reactive state that all
~30 derived constants in `environment.ts` would have to subscribe to. The panel's
primary action is "Apply and reload" for that reason.

## Enabling

Gated by `flagOverridesEnabled` (`VITE_FF_FLAG_OVERRIDES_ENABLED`), which
defaults to `false`. It is read **only** from the build environment, never from
an override — otherwise anyone could set it to `true` in production and unlock
every other flag.

| Environment             | How to enable                                                                                                |
| ----------------------- | ------------------------------------------------------------------------------------------------------------ |
| Vite dev server         | `VITE_FF_FLAG_OVERRIDES_ENABLED='true'` in `env/.env.dev.local`                                               |
| Local served build      | Already on — `studio.feature_flags.flag_overrides_enabled: true` in `packages/nhx_platform/config/local.yaml` |
| Deployed (dev/internal) | `studio.feature_flags.flag_overrides_enabled: true` in that target's values yaml                              |
| Production              | Leave off                                                                                                     |

When the gate is off the panel is never rendered, and its chunk is never fetched
— `GlobalNav` imports it lazily for exactly that reason.

## Using the panel

The flag icon in the global header opens a modal listing every flag with its
type, its deployment default, and its current effective value.

Each control has an explicit **Default** option. That is not the same as setting
a flag to `false`: "default" means no entry in the override blob, so the flag
falls through to whatever the deployment ships. A plain on/off toggle cannot
express that distinction, which is why the controls are selects rather than
switches.

Edits accumulate in a draft and commit as one write plus one reload, so changing
several flags costs a single reload.

Three bulk actions act on the rows the filter is currently showing, so narrowing
the filter first scopes them:

| Action      | Effect                                                                     |
| ----------- | -------------------------------------------------------------------------- |
| All on      | Writes an explicit `true` override for every listed boolean/preview flag   |
| All off     | Writes an explicit `false` override for the same                            |
| Reset all   | Removes overrides entirely, so flags fall back to the deployment default   |

"All off" and "Reset all" are not the same: the first pins flags off regardless
of what the deployment ships, the second stops overriding them at all. String
and number flags are left alone by the first two, since `true`/`false` is not a
meaningful value for them.

The gate flag itself is deliberately absent from the list. Since it is read only
from the build environment, a row for it would render a control that silently
does nothing.

## Storage

One key, `feature-flag-overrides`:

```jsonc
{ "dashboardEnabled": "preview", "toolCallingEnabled": "true" }
```

Values are **raw strings** keyed by the camelCase flag name. Studio feeds them
through the same Zod schemas as real env vars, so `"preview"`, `"true"`,
`"false"`, numbers, and strings behave exactly as if the env var were set. An
empty blob removes the key outright rather than storing `{}`.

### Recovery

If an override enables a feature that breaks rendering, the panel goes with it.
Clear overrides from devtools:

```js
localStorage.removeItem('feature-flag-overrides');
location.reload();
```

## Failure handling

Deliberately forgiving. Most of what gets dropped is **stale, not hostile**: the
blob outlives the code that wrote it, so flags get deleted and change type while
overrides persist in a browser indefinitely.

| Input                             | Result                              |
| --------------------------------- | ----------------------------------- |
| Malformed JSON                    | Entire blob ignored, warning logged |
| Not a JSON object (array, scalar) | Entire blob ignored, warning logged |
| Unknown or removed flag key       | That entry dropped, warning logged  |
| Non-string value                  | That entry dropped, warning logged  |
| Value failing the flag's schema   | That entry dropped, warning logged  |
| localStorage unavailable (throws) | Treated as no overrides             |

That last column matters: `parseFlags` treats a schema failure as a fatal
"missing required flag", so bad entries are filtered out *before* they reach it.
Reading `window.localStorage` is itself a throwing operation when storage is
blocked, so a `typeof` guard is not safe — all access funnels through
`safeStorage()`.

## Source

- `packages/studio/src/constants/featureFlags/overrides.ts` — storage, validation
- `packages/studio/src/constants/featureFlags/index.ts` — two-pass parse cascade
- `packages/studio/src/components/FeatureFlagPanel/` — the modal
- `packages/studio/src/components/Layouts/GlobalNav/index.tsx` — lazy mount behind the gate
