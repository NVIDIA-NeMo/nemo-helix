---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: refresh-data-designer-skill
description: Refresh the vendored data-designer skill bundle in this plugin from the upstream NVIDIA-NeMo/DataDesigner repo at the pinned library version, adapting CLI commands and platform-specific guidance to the NeMo plugin context. Use when bumping the `data-designer` library pin in pyproject.toml, when upstream ships skill changes that need to land here, or when auditing drift. Trigger keywords - refresh data designer skill, update data designer skill, vendor data designer skill, sync data designer skill, bump data-designer version.
---

# Refresh Data Designer Skill

The user-facing skill at `skills/nemo-data-designer-plugin/` is derived from the upstream skill bundle at `https://github.com/NVIDIA-NeMo/DataDesigner` (path `skills/data-designer/`).

This skill is for plugin contributors. End users do not run it — they get the already-vendored output.

## The governing principle

**The shipped skill describes NeMo Platform execution only.** It is not a copy of the upstream skill with caveats bolted on.

Upstream's skill documents the standalone library, where models come from a local YAML registry, seed data is a file on disk, and persona locales are downloaded to `~/.data-designer/managed-assets/`. **None of that is true here.** Data Designer on NeMo Platform resolves inference through Inference Gateway, seed data through HuggingFace or the Files service, and persona data through `system` workspace filesets.

Earlier versions of this playbook copied upstream wholesale and tried to teach agents the delta in a separate additions file. That failed: agents followed the standalone-library instructions that were still sitting in the bundle. Do not reintroduce that pattern. When upstream content contradicts platform behavior, **replace it**, don't annotate it.

## When to use

- After bumping the `data-designer==X.Y.Z` pin in `plugins/nemo-data-designer/pyproject.toml`.
- When upstream ships skill changes that should land here.
- When auditing drift between the vendored copy and upstream.

## File ownership

Refreshing is not a whole-tree overwrite. Each file is either tracked from upstream or owned by this plugin:

| File                                  | Ownership | On refresh                                    |
| ------------------------------------- | --------- | --------------------------------------------- |
| `SKILL.md`                            | upstream  | Re-fetch, then re-apply the adaptations below. |
| `workflows/interactive.md`            | upstream  | Re-fetch, then re-apply the adaptations below. |
| `workflows/autopilot.md`              | upstream  | Re-fetch, then re-apply the adaptations below. |
| `references/preview-review.md`        | upstream  | Take verbatim.                                 |
| `references/seed-datasets.md`         | **plugin** | Do not overwrite. Review only.                |
| `references/person-sampling.md`       | **plugin** | Do not overwrite. Review only.                |
| `references/platform-execution.md`    | **plugin** | Do not overwrite. Review only.                |
| `references/retrieval-sdg.md`         | **plugin** | No upstream counterpart.                      |
| `scripts/get_person_object_schema.py` | **plugin** | Reads the locale's platform fileset, not a local path. Do not overwrite. |
| `skill-card.md`, `BENCHMARK.md`, `evals/`, `skill.oms.sig` | **plugin** | Release artifacts. Do not overwrite. |

For plugin-owned files, "review only" means: read upstream's version to see whether it documents a *new capability* worth covering, then write that capability in platform terms yourself. Never paste upstream's text in.

## Procedure

### Step 1 — Determine the pinned version

```bash
grep -E '"data-designer==' plugins/nemo-data-designer/pyproject.toml | head -1
```

Capture the version string (e.g., `0.9.1`) — this is `$VERSION` for the rest of the procedure.

### Step 2 — Fetch the upstream bundle to a temp directory

```bash
gh api "repos/NVIDIA-NeMo/DataDesigner/git/trees/v$VERSION?recursive=1" \
  | jq -r '.tree[] | select(.path | startswith("skills/data-designer/")) | select(.type=="blob") | .path'
```

Fetch each path from `https://raw.githubusercontent.com/NVIDIA-NeMo/DataDesigner/v$VERSION/<path>` into a scratch directory. **Stage it — do not write into `skills/nemo-data-designer-plugin/` yet.** You are diffing against the shipped bundle, not replacing it.

### Step 3 — Diff and triage

Diff each staged upstream file against its shipped counterpart. For every difference, decide:

- **A new or changed capability** (new column type, new sampler, new flag) → adopt it, translated to platform terms.
- **Standalone-library mechanics** (local paths, `~/.data-designer/`, the local model registry, artifact folders on disk) → drop it. It does not apply here.
- **Cosmetic upstream churn** → take it if it costs nothing.

If upstream adds a file with no platform meaning, do not vendor it. If upstream deletes a file the plugin still owns, keep the plugin's.

### Step 4 — Apply adaptations to upstream-tracked files

**Prefix CLI invocations.** Every shell-shaped invocation of the `data-designer` binary becomes `nemo data-designer …`. Subcommand names, flags, and positional args are otherwise identical. Do not rewrite prose mentions of the project name ("the Data Designer library", "Data Designer columns") — only shell-shaped invocations.

**`create` takes no `--dataset-name` or `--artifact-path`.** Artifacts live at a Jobs-service-managed path, so there is no local folder to name or relocate. Strip those flags and their values; keep others like `--num-records`.

**Resolve-CLI-command step.** Upstream's workflows open by detecting the standalone `data-designer` binary. Replace with `nemo` detection:

```
1. **Resolve CLI command** — Run `command -v nemo 2>/dev/null || (test -x .venv/bin/nemo && realpath .venv/bin/nemo) || echo CLI_NOT_FOUND`.
  - If the output is a path, use `<path> data-designer` as the command prefix for all `nemo data-designer …` invocations in this workflow.
  - If the output is `CLI_NOT_FOUND`, STOP and follow the Troubleshooting section in SKILL.md. Do not continue to the next step.
```

**Learn step.** Upstream tells the agent to stop when `agent context` lists no model aliases, and to trust its output generally. Both are wrong here. `agent context` prints three sections that describe a standalone local install — **Model Aliases** (a local YAML registry), **Persona Datasets** (an `installed` column reflecting local downloads), and **Commands** (unprefixed `data-designer …`). Its `config_root` path and type tables *are* correct and are the reason to run it at all. Replace the stop-on-no-aliases line with an instruction to read schemas from `config_root` and ignore those three sections, pointing at `references/platform-execution.md`.

**Clarify / Infer step.** Where upstream picks among locally configured aliases, have the agent run `nemo inference providers list` and choose an Inference Gateway provider instead.

**Output Template.** Preserve the plugin's template shape:
- `model_configs=[dd.ModelConfig(...)]` declared inline in the `DataDesignerConfigBuilder(...)` constructor, with a comment pointing at `nemo inference providers list`. Upstream's bare `DataDesignerConfigBuilder()` produces configs that cannot run here.
- The commented seed line uses `FilesetFileSeedSource`, imported from `data_designer_nemo.fileset_file_seed_source`, **not** `dd.LocalFileSeedSource`.
- The PEP 723 dependency list includes `data-designer-nemo` for the seed source import.

**Rules section.** Keep the bullet pointing at `references/platform-execution.md`, and the one routing retrieval SDG to `references/retrieval-sdg.md`.

### Step 5 — Write the bundle

Write adapted upstream-tracked files to `skills/nemo-data-designer-plugin/`, mirroring the upstream tree (`workflows/`, `references/`, `scripts/`). Leave plugin-owned files alone unless Step 3 turned up a capability change.

### Step 6 — Verify

Resolve every failure before considering the refresh complete.

**No untranslated CLI invocations.**

```bash
SKILLS=skills/nemo-data-designer-plugin
grep -RnE '(\s|`)data-designer\s' "$SKILLS" | grep -v 'nemo data-designer'
```

Clean run = no output. Prose mentions are fine; CLI-shaped invocations are not.

**Every referenced subcommand resolves.**

```bash
grep -RhoE 'nemo data-designer( [a-z][a-z-]*)+' "$SKILLS" \
  | sort -u \
  | while read -r cmd; do
      eval "$cmd --help" > /dev/null 2>&1 && echo "OK: $cmd" || echo "MISSING: $cmd"
    done
```

Any `MISSING:` means upstream changed the command surface or `nemo data-designer` drifted — investigate before merging. In particular, `personas` exposes **only** `make-fileset`; a `personas download` reference is always a bug.

**No standalone-library guidance leaked back in.**

```bash
grep -RniE '~/\.data-designer|managed-assets|LocalFileSeedSource|personas download|data-designer config' "$SKILLS"
```

Every hit must be either absent or an explicit statement that the thing does *not* apply on NeMo Platform. A hit that reads as instruction is a failure.

**Non-obvious claims are checked against the source**, not against upstream's docs:

- Supported seed types: `packages/data_designer_nemo/src/data_designer_nemo/seed.py`
- Persona locales and fileset naming: `packages/data_designer_nemo/src/data_designer_nemo/nemotron_personas.py`
- CLI surface: `plugins/nemo-data-designer/src/nemo_data_designer_plugin/cli/`

## Notes

- Upstream may add files under `skills/data-designer/`. Pick them up via the Step 2 tree enumeration, then triage per Step 3 — enumeration is not adoption.
- Don't edit `references/preview-review.md` beyond CLI prefixing; it is genuinely upstream's.
- `scripts/get_person_object_schema.py` diverges from upstream on purpose: it reads the locale's `system` fileset through the plugin's `FilesetFileSystem` and fails with instructions to run `personas make-fileset` when that fileset is missing. Upstream's version reads a local download path that never exists here.
