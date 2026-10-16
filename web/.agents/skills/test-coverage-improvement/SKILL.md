---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: test-coverage-improvement
description: >-
  Autonomous Vitest coverage workflow for a chosen package under web/packages:
  baseline report, agent-curated shortlist of ~10 files, then per-file context from
  reading source, inferred behaviors, implemented tests, and verification (vitest +
  typecheck) before the next file; after the loop, **eslint only the files you created or
  edited** and fix issues.
  Human review happens outside this skill (e.g. MR,
  local diff). Use when improving unit coverage in the web monorepo. Trigger keywords
  — package coverage, vitest coverage, pnpm workspace tests, coverage baseline,
  raise coverage.
disable-model-invocation: true
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---

# Package test coverage improvement (autonomous)

**Scope:** A **single** package under **`web/packages/*`** that uses **Vitest** (TypeScript). Run commands from **`web/`** (pnpm workspace) or **`cd web/packages/<pkg>`**.

**Conventions:** Obey **`web/AGENTS.md`**. For **React** UI in `studio` / `common`, follow **`web/.agents/skills/unit-test/SKILL.md`**. For **plain TS**, use Vitest only; no watch until runs are green.

**Human role:** The skill does **not** block on questions per file. The **user reviews** changes separately (PR, `git diff`, local run). If the user stated constraints up front (areas to focus, avoid), honor those; otherwise proceed using judgment below.

## 0. Choose the target package

1. If the user named a package, use it (e.g. `studio`, `common`, `sdk`).
2. Otherwise infer from open files or cwd under `web/packages/`.
3. Read **`package.json`**: **`name`**, **`test`**, **`test:ci`**, **`typecheck`**, **`lint`** / **`lint:fix`** (if present).
4. Read Vitest config (**`vite.config.ts`** / **`vitest.config.ts`**) → **`test.coverage`**: **`reportsDirectory`**, **`include` / `exclude`**.

If there is **no** coverage tooling, add **`--coverage`** + reporters on the CLI for baseline, or build the shortlist without JSON (search + file importance only).

## 1. Baseline — current coverage

1. `cd web/packages/<target>`.
2. Run (**no watch**):
   ```bash
   pnpm test:coverage
   ```
3. Summarize **briefly** for the transcript: overall line (or statement) % from text or **`total`** in **`coverage-final.json`** (under **`reportsDirectory`**).

## 2. Shortlist ~10 files

Combine **`coverage-final.json`** with judgment—**do not** use only the bottom ten by %:

- Sort by low **`lines.pct`** / **`statements.pct`** among included source files.
- **Prefer:** logic-heavy modules, hooks, API glue, non-trivial components, error paths.
- **Skip or defer:** thin re-exports, empty barrels, generated-only files, trivial constants (unless user asked otherwise).
- **Exclude:** `*.test.*`, `e2e-tests/**`, `node_modules/**`, outside **`coverage.include`**.

Output a **numbered list** (path, %, one-line rationale). **Do not wait for approval** unless the user explicitly asked to confirm the list; proceed to the loop.

## 3. Per-file loop

For **each** shortlist file, in order:

### 3a. Contextualize (agent)

Read the implementation and, when useful, **callers or types** (imports, sibling modules). Note **exports**, **side effects**, **async** boundaries, **feature flags**, and **error handling**.

### 3b. Decide what to test (agent)

Infer **high-value behaviors** without asking the user—for example:

- Public functions / component outcomes users depend on.
- Branches: success vs error, empty data, validation failures.
- Integration with mocked **fetch** / **MSW** / **TanStack Query** where the module touches the network.
- Stable contracts: inputs → outputs, not implementation trivia.

If the file is **untestable without refactor**, add a **short skip note** in the summary and continue.

### 3c. Implement

- **React:** **`web/.agents/skills/unit-test/SKILL.md`** (RTL, MSW, `findBy*`, `vi.mock`).
- **Non-React:** Vitest + **`vi.mock`** as needed.
- Colocate **`*.test.ts` / `*.test.tsx`** unless the package uses another established pattern.

### 3d. Verify before next file

1. `pnpm vitest --run path/to/File.test.ts` (and any related specs) until **exit 0**.
2. **`pnpm typecheck`**. Fix TypeScript issues (TanStack context, SDK types, etc.).
3. Optionally re-run coverage for the package and note improvement for that file.

Keep a **running list** of every file path you **create or edit** during §3 (new **`*.test.*`**, and any production files you touch). You will pass that list to eslint in §4.

You may fix obvious **ESLint** issues (e.g. **`import/order`**) during §3d when **`typecheck`** is already clean.

## 4. Lint after the loop (changed files only)

When **all** shortlist iterations in §3 are finished (not after each file):

1. `cd web/packages/<target>`.
2. From **`package.json`**, read the **`lint`** script and reuse the **same eslint flags** as the project (everything after the `eslint` command—e.g. **`--report-unused-disable-directives --max-warnings 0`** for Studio), but **replace the path glob** (e.g. `.`) with **only your tracked file paths**:
   ```bash
   pnpm exec eslint --report-unused-disable-directives --max-warnings 0 path/to/A.test.ts path/to/B.test.ts
   ```
3. If the package uses **`lint:fix`** for local workflow, you may run **`pnpm exec eslint --fix ...`** with the **same file list** first, then re-run without **`--fix`** if needed to confirm **exit 0**.
4. Fix any reported issues until eslint exits **0** on that list.

**Do not** run **`pnpm lint`** / **`eslint .`** on the whole package unless the user asked for a full-package lint or a changed file list is empty (then note “no files to lint”).

If eslint is not available in the package, note that lint was skipped.

## 5. Stop

- After **~10** files (or fewer if the shortlist was shorter), **§4 lint**, and a final sanity check.
- If the user capped scope in the initial message, respect that.
- End with a **short recap**: files touched, specs added/updated, any skips and why, and confirm **lint** passed.

## 6. Quick reference

| Item              | Notes                                                                                |
| ----------------- | ------------------------------------------------------------------------------------ |
| Workspace root    | `web/` — optional **`pnpm --filter <package.json name> <script>`**                   |
| Package root      | `web/packages/<directory>`                                                           |
| Coverage artifact | `<reportsDirectory>/coverage-summary.json`                                           |
| Memory            | Match package scripts (e.g. Studio **`NODE_OPTIONS`** on **`test`** / **`test:ci`**) |
| Lint              | **`pnpm exec eslint …<paths>`** on **created/edited files only** after §3 (§4)       |
