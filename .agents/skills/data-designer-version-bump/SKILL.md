---
name: data-designer-version-bump
description: Bump the upstream NVIDIA NeMo Data Designer library (`data-designer`, `data-designer-config`, `data-designer-engine`) to a new release in nemo-helix. Covers upstream research, pin edits, relocking, dependency-conflict triage, wheel constraints, versioned doc links, generated license/OpenAPI/CLI artifacts, API-change adoption, the vendored plugin skill refresh, and verification. Use when asked to bump, upgrade, or update Data Designer or the data-designer dependency to a new version. Not for nemo-anonymizer, which releases on its own cadence.
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Bump Data Designer

Move nemo-helix from one `data-designer` release to another. Read this whole file before starting. Prior bumps for reference: `f30fe66fb` (0.9.1), `788d59da1` (0.9.3, also shows adopting an upstream API change).

## Scope

- **In scope:** `data-designer`, `data-designer-config`, `data-designer-engine` (always released together and pinned to each other with `==`), and whatever transitive dependency changes the bump forces.
- **Out of scope:** `nemo-anonymizer`. It does not bump in lockstep with Data Designer. Leave its pin alone unless the user asks. You still must confirm the pinned anonymizer accepts the new DD version (Step 1).
- `data-designer-retrieval-sdg` is versioned separately. Leave it unless the relock forces it.

Throughout, `$OLD` is the currently pinned version and `$NEW` is the target. Use a scratch directory for anything downloaded.

## Step 0: Preflight

```bash
git status --short                      # must be clean
git branch --show-current               # must not be main; branch as <name>/<username>
OLD=$(grep -oE '"data-designer==[^"]+"' plugins/nemo-data-designer/pyproject.toml | head -1 | sed -E 's/.*==([^"]+)"/\1/')
echo "$OLD -> $NEW"
```

## Step 1: Research upstream before touching anything

Do all of this first. It tells you whether the bump is mechanical or needs a decision from the user.

1. **Release notes** for every release in `($OLD, $NEW]`, not just `$NEW`:
   ```bash
   gh release list -R NVIDIA-NeMo/DataDesigner -L 10
   gh release view v<each> -R NVIDIA-NeMo/DataDesigner --json body -q .body
   ```
   Flag `feat:` and `fix:` entries that touch config models, column types, seed readers, model facades, traces/results serialization, or the CLI. These are the platform's integration points.

2. **Dependency ranges.** Compare `requires_dist` for all three packages at `$OLD` and `$NEW`:
   ```bash
   for p in data-designer data-designer-config data-designer-engine; do
     for v in $OLD $NEW; do
       curl -s https://pypi.org/pypi/$p/$v/json \
         | python3 -c "import json,sys; d=json.load(sys.stdin); print('$p', '$v', d['info']['requires_dist'])"
     done
   done
   ```
   Note any **raised floors** and **removed caps**. Raised floors (often CVE fixes) are the usual source of lock conflicts. To find why a floor moved, look at the upstream pyproject diffs:
   ```bash
   gh api repos/NVIDIA-NeMo/DataDesigner/compare/v$OLD...v$NEW \
     --jq '.files[] | select(.filename|endswith("pyproject.toml")) | .filename + "\n" + .patch'
   ```

3. **Anonymizer compatibility.** The pinned `nemo-anonymizer` declares its own `data-designer` range:
   ```bash
   ANON=$(grep -oE '"nemo-anonymizer==[^"]+"' plugins/nemo-anonymizer/pyproject.toml | sed -E 's/.*==([^"]+)"/\1/')
   curl -s https://pypi.org/pypi/nemo-anonymizer/$ANON/json \
     | python3 -c "import json,sys; print([r for r in json.load(sys.stdin)['info']['requires_dist'] if 'designer' in r])"
   ```
   If `$NEW` falls outside that range, stop and tell the user. Anonymizer must move too, which is a separate decision.

4. **Upstream skill changes.** Did anything under `skills/data-designer/` change?
   ```bash
   gh api repos/NVIDIA-NeMo/DataDesigner/compare/v$OLD...v$NEW \
     --jq '.files[] | select(.filename|startswith("skills/")) | "\(.status) \(.filename)"'
   ```
   Empty output means the Step 8 refresh only needs its verification pass.

5. **Versioned docs.** Confirm upstream published docs for `$NEW`:
   ```bash
   curl -s -o /dev/null -w '%{http_code}\n' -L -A 'Mozilla/5.0' \
     https://docs.nvidia.com/nemo/datadesigner/v$NEW/getting-started/welcome
   ```
   Expect `200`. Step 5 covers what to do if it isn't.

## Step 2: Edit the pins

Find every pin. Don't rely on a hardcoded list, because files move:

```bash
git grep -n -E 'data-designer(-config|-engine)?==' -- ':!uv.lock' ':!third_party/'
```

As of 0.9.5 the hand-edited sources are:

| File | Notes |
| --- | --- |
| `packages/data_designer_nemo/pyproject.toml` | runtime dependency |
| `plugins/nemo-data-designer/pyproject.toml` | runtime dependency, plus a generated `data-designer-nemo` extra block |
| `plugins/nemo-anonymizer/pyproject.toml` | the anonymizer plugin pins `data-designer` too. Bump it even though the anonymizer library itself stays put. |
| `packages/nemo_helix/pyproject.toml` | **generated** by `make vendor`. Don't hand-edit. Regenerate. |
| `.github/wheel-constraints/*.txt` | handled in Step 4 |

```bash
sed -i '' "s/\"data-designer==$OLD\"/\"data-designer==$NEW\"/" \
  packages/data_designer_nemo/pyproject.toml \
  plugins/nemo-data-designer/pyproject.toml \
  plugins/nemo-anonymizer/pyproject.toml
make vendor                               # regenerates packages/nemo_helix/pyproject.toml and the generated blocks
```

(`sed -i ''` is BSD/macOS syntax. On Linux use `sed -i`.)

## Step 3: Relock

Upgrade only the DD packages. Don't do a blanket `uv lock --upgrade`:

```bash
uv lock --upgrade-package data-designer --upgrade-package data-designer-config --upgrade-package data-designer-engine
```

Record the `Updated X vA -> vB` lines. Every package listed there feeds Step 4.

**If resolution fails**, read the uv explanation and find which of our constraints blocks the new floor. Check the root `pyproject.toml` `override-dependencies` and `constraint-dependencies` first, then plugin pyprojects. Then:

- Test a fix without committing to it: edit the cap, run `uv lock --dry-run --upgrade-package …`, then restore the files.
- Check whether the blocking package's own upstream has already moved. For example, does the latest release of the plugin library that motivated the cap still need it?
- **Stop and ask the user** before loosening a cap that affects other plugins. Give them the options: loosen the cap, take an earlier DD release that resolves cleanly, or stop.
- If a cap changes, change it everywhere it's declared (`git grep` the old specifier), then rerun `make vendor` and the relock.

Past conflict (0.9.5): `data-designer-engine` raised `fsspec>=2026.6.0` for a CVE, and the root override pinned `datasets<=4.3.0`, which caps fsspec at 2025.9.0. The fix was to raise `datasets` to `<6` in the root override and in `plugins/nemo-safe-synthesizer`. Upstream `nemo-safe-synthesizer` had already moved to `datasets>=4.8.4`.

**Overrides hide transitive floors.** An entry in `override-dependencies` *replaces* every other package's requirement on that dependency, so uv will happily lock a version that a newly upgraded package can't import. A clean `uv lock` doesn't prove compatibility. For every package that moved, compare its new `requires_dist` against any root override on the same dependency, and `--upgrade-package` the overridden dependency when its locked version is below the real floor. In 0.9.5, `datasets` 5.1.0 needs `huggingface-hub>=1.31.0`, but the root override (`>=1.0.1,<2`) kept the lock at 1.26.0. Safe-synthesizer tests then failed with `ImportError: cannot import name 'httpx' from 'huggingface_hub.utils'` until `uv lock --upgrade-package huggingface-hub` moved it to 1.33.0.

Then sync so tests run against the new lock:

```bash
uv sync --frozen --all-packages
```

## Step 4: Wheel constraints

`.github/wheel-constraints/*.txt` pin each wheel's direct external dependencies for CI's install smoke test. The full regeneration (`script/compile-wheel-constraints.sh`) needs built wheels, including the Studio toolchain. For a dependency bump, hand-editing is the established practice. For every package in the Step 3 `Updated` list, update its line in any constraints file that lists it:

```bash
grep -n -E '^(data-designer|datasets|fsspec)==' .github/wheel-constraints/*.txt   # extend with the Step 3 packages
```

Constraints files can disagree with each other and with `uv.lock` before you start. In 0.9.5 the services file listed `huggingface-hub==1.27.0` while the lock had 1.26.0. Make every line you touch match `uv.lock`; don't just swap old for new.

## Step 5: Versioned doc links

```bash
git grep -n "datadesigner/v$OLD" -- docs/
```

If Step 1.5 returned `200` for `$NEW`, replace `v$OLD` with `v$NEW` in those files. Before committing, check every rewritten URL (not just `welcome`) and confirm the page actually contains `v$NEW`, so a "not found" page served with a 200 doesn't pass:

```bash
for u in $(git grep -h -oE "https://docs.nvidia.com/nemo/datadesigner/v$NEW/[a-z/-]+" -- docs/ | sort -u); do
  curl -s -L -A 'Mozilla/5.0' "$u" | grep -q "v$NEW" && echo "OK   $u" || echo "FAIL $u"
done
```

Any `404` is unexpected. Tell the user so they can review and potentially raise it with the DD team, leave the links on the newest version that resolves, and recheck before the PR merges.

## Step 6: Adopt upstream API changes

Read the upstream diff for modules the platform imports, then decide whether the platform should adopt a new behavior:

```bash
git grep -h -oE 'from data_designer[a-z_.]*|import data_designer[a-z_.]*' -- '*.py' | sort | uniq -c | sort -rn
gh api repos/NVIDIA-NeMo/DataDesigner/compare/v$OLD...v$NEW --jq '.files[] | "\(.status) \(.filename)"' | grep -v -E '^(modified|added) (docs|\.github|plans|fern)/'
```

Example: the 0.9.3 bump adopted upstream's trace DataFrame encoding/decoding in `plugins/nemo-anonymizer/.../tasks/anonymizer/run.py` and added tests. Keep adoptions in their own commit so reviewers can tell them apart from the mechanical bump.

## Step 7: Regenerate generated artifacts

Run each target, then inspect the diff. Don't blindly commit noise.

```bash
make refresh-openapi                 # plugin openapi.yaml files pick up DD/anonymizer schema changes
make generate-cli-reference-docs     # docs/cli/reference.mdx: CLI flags generated from DD/anonymizer Pydantic schemas
make update-licenses                 # third_party/licenses.jsonl, requirements-main.txt, osv-licenses*.json (needs osv-scanner from Flox)
```

- `third_party/requirements-main.txt` should show the new DD hashes and any transitive moves from Step 3.
- `third_party/osv-licenses.json` can churn a lot because the advisory database moves. That's expected. These files are already excluded from trufflehog (`.github/trufflehog-exclude.txt`) because advisory text can contain placeholder credentials.
- `third_party/osv-licenses-callouts.json` and `osv-licenses-jobs-launcher.json` are Go-module scans that a Python bump doesn't affect. Restore them (`git restore` both files). Past bumps never committed them.
- Sanity check: `git diff third_party/requirements-main.txt | grep -E '^[-+][a-z]' | cut -d' ' -f1` should list exactly the Step 3 `Updated` packages.
- If `licenses.jsonl` didn't change, `tools/lint/lint-fix-licenses.sh` treats the rest of `third_party/` as noise and restores it. A dependency bump should change `requirements-main.txt`, so keep it.

## Step 8: Refresh the vendored plugin skill

Follow the `data-designer-skill-refresh` skill (`.agents/skills/data-designer-skill-refresh/SKILL.md`) with `$VERSION=$NEW`.

- If Step 1.4 showed upstream skill changes, run the full procedure.
- If not, run only its **Step 6: Verify**. The `nemo data-designer` command surface can drift even when upstream's skill doesn't change.
- Keep the `eval` in its subcommand loop and prefix `uv run --frozen` when `nemo` isn't on `PATH`. Under zsh, an unquoted `$cmd` isn't word-split, so every command reports a false `MISSING:`.

## Step 9: Verify

```bash
uv run --frozen pytest packages/data_designer_nemo plugins/nemo-data-designer plugins/nemo-anonymizer -n auto -q
```

Also test every package whose dependencies moved in Step 3. For example, a `datasets` change means also running `plugins/nemo-safe-synthesizer`. Find the importers (use POSIX classes, because `git grep -E` doesn't support `\s` or `\b`):

```bash
git grep -l -E '^[[:space:]]*(import|from) (datasets|fsspec|huggingface_hub)([. ]|$)' -- '*.py' \
  | grep -v '^e2e/' | awk -F/ '{print $1"/"$2}' | sort -u
```

Then:

```bash
uv run nemo data-designer --help > /dev/null
uv run --frozen ty check
uv run pre-commit run -a
```

`ty check` reports hundreds of unrelated diagnostics repo-wide. Filter for the moved packages (`grep -iE 'data_designer|datasets|fsspec|huggingface'`) and compare against `main` rather than expecting zero. The pre-commit `ty` hook is the gate. In a worktree without `make bootstrap-studio`, the `studio-lint-staged` hook fails with `lint-staged not found`. That's an environment gap, not a bump problem.

Report test failures with their output. Don't claim the bump is done if any of these fail.

## Step 10: Commit and PR

- Title: `chore: bump data-designer to $NEW` (add `and <pkg> to <ver>` when Step 3 forced a notable cap change).
- Commit with `-s`. Separate commits for (a) the mechanical bump plus regenerated artifacts, (b) any API adoption from Step 6, and (c) any skill refresh content changes.
- In the PR body, list the upstream releases covered, any forced transitive moves and why (link the upstream PR or CVE), and doc links held back from Step 5.
- Use the `contributor-create-pr` skill to open the PR.
