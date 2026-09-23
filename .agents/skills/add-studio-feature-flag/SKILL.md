---
name: add-studio-feature-flag
description: >-
  Adds a NeMo Studio (Vite) feature flag end-to-end: typed definition, runtime
  injection mapping, FastAPI build env markers, and optional Helm config. Use when
  the user asks for a new feature flag, VITE_FF_ variable, studio.feature_flags
  setting, toggling UI behind a flag, or preview/boolean flags for Studio.
---

# Add a Studio feature flag

Studio flags live in `web/packages/studio` and are wired into the **FastAPI Studio** bundle via **marker replacement** at runtime. Follow every step that applies so local dev, FastAPI builds, and cluster deploys stay consistent.

For broader Studio context, see [.cursor/skills/studio-dev/SKILL.md](../studio-dev/SKILL.md).

## 1. Choose flag shape

| Helper                          | TypeScript value                 | Use when                                                                     |
| ------------------------------- | -------------------------------- | ---------------------------------------------------------------------------- |
| `previewFlag(envVar, default?)` | `true` \| `'preview'` \| `false` | User-facing features that may show the **Early Preview** badge (`'preview'`) |
| `booleanFlag(envVar, default?)` | `boolean`                        | Internal toggles (no preview mode)                                           |
| `stringFlag` / `numberFlag`     | string / number                  | Rare; see `web/packages/studio/src/constants/featureFlags/utils.ts`          |

**Preview behavior:** `true` and `'preview'` are both truthy for `if (featureFlags.myFlag)`. Only `'preview'` triggers `FeatureFlagBadge` (see step 6).

**Naming**

- Env var: `VITE_FF_<SCREAMING_SNAKE_CASE>`
- `flagDefinitions` key: **camelCase** (e.g. `jobQueueEnabled`)
- `studio.feature_flags` / `config_path`: **snake_case** (e.g. `job_queue_enabled`)

## 2. Register the flag (TypeScript)

**File:** `web/packages/studio/src/constants/featureFlags/featureFlags.ts`

Add one entry to `flagDefinitions`:

```ts
jobQueueEnabled: previewFlag('VITE_FF_JOB_QUEUE_ENABLED'),
// or
jobQueueEnabled: booleanFlag('VITE_FF_JOB_QUEUE_ENABLED'),
```

Alphabetical order with sibling keys is preferred.

**Consume**

```ts
import { featureFlags } from "@studio/constants/featureFlags";

if (featureFlags.jobQueueEnabled) {
  // enabled (true or 'preview')
}
```

## 3. Optional: `environment.ts` re-export

**File:** `web/packages/studio/src/constants/environment.ts`

If the codebase prefers `SOME_FEATURE_ENABLED` constants (see existing `MEMBERS_ENABLED`, `SECRETS_ENABLED`, etc.), add:

```ts
export const JOB_QUEUE_ENABLED = featureFlags.jobQueueEnabled !== false;
```

Use this only when it matches an established pattern for that area of the app.

## 4. Runtime injection (FastAPI Studio service)

**File:** `services/studio/src/nhx/studio/env_mappings.py`

Add an `EnvMapping` in `ENV_MAPPINGS` (keep the feature-flag block grouped):

```python
EnvMapping(
    marker="STUDIO_UI_VITE_FF_JOB_QUEUE_ENABLED",
    config_path="studio.feature_flags.job_queue_enabled",
    default="false",
),
```

- `marker` is always `STUDIO_UI_` + the **Vite** env name (`VITE_FF_...`).
- `default` is a **string** (`"true"`, `"false"`, or `"preview"` for preview flags).

## 5. FastAPI build env markers (parity)

**File:** `web/packages/studio/env/.env.fastapi`

Add a line so the **built** bundle contains the placeholder the Studio service replaces:

```bash
VITE_FF_JOB_QUEUE_ENABLED=STUDIO_UI_VITE_FF_JOB_QUEUE_ENABLED
```

The file header requires **parity** with `env_mappings.py`—do not skip this for flags used in FastAPI mode.

## 6. Local development

Update **both** files so the sample stays in lockstep with the feature-flag inventory — devs copy the sample to bootstrap `.env.dev.local`, and a missing entry means the flag silently falls back to its default.

**File:** `web/packages/studio/env/.env.dev.local.sample` (always — committed reference)

Add the flag in the `# Feature Flags (VITE_FF_* prefix)` block, alphabetized with sibling `VITE_FF_*` lines. Use the value a new developer should start with (usually the same as the `default` in `env_mappings.py`):

```bash
VITE_FF_JOB_QUEUE_ENABLED='false'
```

**File:** `web/packages/studio/env/.env.dev.local` (only if you want it on for your own machine — gitignored)

```bash
VITE_FF_JOB_QUEUE_ENABLED='true'
# or for preview badge behavior:
VITE_FF_JOB_QUEUE_ENABLED='preview'
```

## 7. Gate UI and routes

- **Routes:** lazy imports / `children` arrays in `web/packages/studio/src/routes/index.tsx`, or conditional wrappers—mirror existing flags (e.g. `membersEnabled`, `secretsEnabled`).
- **Nav:** `WorkspaceSideNav` and similar—search for `featureFlags.` patterns.
- **Preview badge:** For `previewFlag` entries, add `<FeatureFlagBadge flag="jobQueueEnabled" />` next to the feature title when appropriate (`web/packages/studio/src/components/FeatureFlagBadge/index.tsx`).

## 8. Helm / platform config (deployed environments)

Under `studio.feature_flags` in values (e.g. `deploy/helm/values/ci/dev-values.yaml`), add the **snake_case** key with string value `true`, `false`, or `preview` as needed.

Global settings schema may need updating if the new key is not yet defined—follow how existing `studio.feature_flags.*` keys are declared in the repo.

## Checklist

Copy and track:

- [ ] `flagDefinitions` in `featureFlags.ts` (correct helper + defaults)
- [ ] `EnvMapping` in `env_mappings.py`
- [ ] `VITE_FF_*=STUDIO_UI_VITE_FF_*` line in `env/.env.fastapi`
- [ ] `env/.env.dev.local.sample` updated (committed reference for new devs)
- [ ] Local `env/.env.dev.local` (only if developers need the flag on for their own machine)
- [ ] UI/route/nav gated with `featureFlags.<key>` or `environment.ts` constant
- [ ] `FeatureFlagBadge` if the flag uses `previewFlag` and should show Early Preview
- [ ] Helm / values or docs updated for environments that should see the flag
- [ ] `pnpm typecheck` (from `web/packages/studio` or monorepo Studio package) after TS edits

## Removing a flag (reference)

1. Remove all usages.
2. Remove from `flagDefinitions`, `env_mappings.py`, `.env.fastapi`, Helm values, and samples.
3. Remove any `environment.ts` export.
