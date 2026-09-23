---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: feature-flags
description: Inspect and manage Studio feature flags. Use when working with feature flags, adding new flags, checking flag state across environments, or debugging why a feature is enabled/disabled.
---

# Feature Flags

## Inspect Current State

Run the matrix script to see all flags and their effective values across environments:

```bash
cd web/packages/studio && pnpm feature-flags --json
```

The JSON output shows each flag with two cascades:

- **`local`**: what the dev server sees
  - `override` — `.env.dev.local` (developer's local overrides)
  - `default` — `featureFlags.ts` (code defaults)
- **`deployments`**: what each deployment target sees
  - Each target (DEV, INTERNAL, NEMOLLM, etc.) has an `override` from its Helm values yaml
  - `default` — `env_mappings.py` (shared deployment default)

In both cascades, `override.value` wins when non-null; otherwise `default.value` is used.

## Tri-State Values

Flags use three states:

| Value       | Meaning                  | UI Behavior                          |
| ----------- | ------------------------ | ------------------------------------ |
| `"true"`    | Feature is GA            | Enabled, no badge                    |
| `"preview"` | Feature is early preview | Enabled, shows "Early Preview" badge |
| `"false"`   | Feature is disabled      | Hidden                               |

Both `true` and `"preview"` are truthy, so route gating (`if (flag)`) works without changes.

The `FeatureFlagBadge` component (`src/components/FeatureFlagBadge/index.tsx`) renders the badge when a flag equals the `PREVIEW` constant.

## Adding a New Flag

Four files must be updated:

### 1. Define the flag in `featureFlags.ts`

Path: `web/packages/studio/src/constants/featureFlags/featureFlags.ts`

```ts
myNewFlag: previewFlag('VITE_FF_MY_NEW_FLAG', false),
```

Use `previewFlag()` for features that may show an "Early Preview" badge. Use `booleanFlag()` only for flags that will never need the preview state (e.g. `tourEnabled`).

### 2. Add the env mapping in `env_mappings.py`

Path: `services/studio/src/nhx/studio/env_mappings.py`

```python
EnvMapping(
    marker="STUDIO_UI_VITE_FF_MY_NEW_FLAG",
    config_path="studio.feature_flags.my_new_flag",
    default="false",
),
```

This controls the deployment default. The `marker` must match `STUDIO_UI_` + the env var name. The `config_path` uses snake_case.

### 3. Add to local env file (optional)

Path: `web/packages/studio/env/.env.dev.local`

```
VITE_FF_MY_NEW_FLAG='preview'
```

### 4. Add Helm overrides (optional)

Path: `deploy/helm/values/ci/<target>-values.yaml`

Add under the `feature_flags:` block using snake_case:

```yaml
feature_flags:
  my_new_flag: true
```

## Graduating a Flag

To move a feature from preview to GA:

1. Change `env_mappings.py` default from `"false"` or `"preview"` to `"true"`
2. Update any Helm values yaml overrides if needed
3. The `FeatureFlagBadge` component auto-hides when the flag is no longer `"preview"`

## Removing a Flag

1. Remove all usage from the codebase (search for the camelCase flag name)
2. Remove from `featureFlags.ts`
3. Remove from `env_mappings.py`
4. Remove from `.env` files and Helm values yamls
5. Remove any `<FeatureFlagBadge flag="..." />` for that flag

## Naming Conventions

- Env var: `VITE_FF_<SCREAMING_SNAKE_CASE>`
- Config key in code: `camelCase` (e.g. `myNewFlag`)
- Config path in env_mappings.py: `snake_case` (e.g. `my_new_flag`)
- Helm values yaml key: `snake_case` (e.g. `my_new_flag`)
