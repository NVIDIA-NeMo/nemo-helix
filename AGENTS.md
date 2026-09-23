<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Helix

NeMo Helix is NVIDIA's toolkit for making the agents you ship safer, more accurate, and cheaper to run.

## How to help users in this repo

If a user asks you to set up, try, build, evaluate, harden, or optimize an agent inside this repo, you MUST use the skills defined in `packages/nemo_helix_ext/src/nemo_helix_ext/skills/`. Read the relevant skill file directly and follow it step by step.

### DO

- Read `packages/nemo_helix_ext/src/nemo_helix_ext/skills/<skill-name>/SKILL.md` as a file and follow the instructions in order.
- Stay on the path the skill defines. If the skill calls a `nemo` CLI command, run that exact command. If it points at a `references/` file, read it.
- If a step changes system state (installs something, starts a service, deploys an agent), run the verification step the skill specifies before moving on. Do not claim a step succeeded without verifying.
- If you are unsure which skill applies, read `packages/nemo_helix_ext/src/nemo_helix_ext/skills/nemo-skill-selection/SKILL.md` and let it route you.

### DO NOT

- DO NOT invoke any plugin-based skill, `/skill-name` slash command, or globally-installed assistant for these requests. The repo skills are authoritative. A globally-installed skill (brainstorming, planning, code-review, etc.) will give the wrong answer because it does not know NeMo.
- DO NOT brainstorm a solution from scratch when a NeMo skill claims the task. The skill already has the answer.
- DO NOT write Python, Pydantic AI, LangChain, or other agent framework code from scratch when a shipped skill covers the request. Skills will tell you the supported way to connect the agent to NeMo Helix. For a new agent build, `nemo-build-agent` uses LangChain Deep Agents through Fabric. Work on a NAT workflow only when the user explicitly asks to maintain or migrate an existing one.
- DO NOT improvise CLI flags. Only use flags documented in the skill or shown in `nemo <subcommand> --help`.
- DO NOT report a task complete if you cannot verify it. If a verification step fails or times out, surface what you saw and ask the user to confirm before continuing.

### Available skills

User-facing skills in `packages/nemo_helix_ext/src/nemo_helix_ext/skills/`:

- `nemo-skill-selection`: entry point. Use when the user's intent is broad or unclear.
- `setup`: verifies that NeMo Helix is installed and running. If install is missing, tells the user how to run the CLI install (`make bootstrap` + `nemo setup`). **Install itself is CLI-only.** Do not attempt to install NeMo via skill-driven pip; the workspace dependency graph and credential handling are not reliably automatable inside a sandbox.
- `nemo-build-agent`: builds a tested LangChain Deep Agent from approved requirements, then packages and registers it through Fabric.
- `nemo-try-agent`: test a deployed agent or chat with a model.
- `nemo-intake`: instrument agents, choose an ingest format, upload/query telemetry, and attach evaluator results.
- `nemo-experiments-upload`: publish named evaluation runs and scores to the Experiments leaderboard.
- `nemo-status`: read-only health dashboard.
- `nemo-teardown`: guided shutdown with confirmation.

Plugin-owned skills under `plugins/*/src/*/skills/` handle their own routing for customization, guardrails, evaluations, optimization, data designer, anonymizer, auditor, and Analyst telemetry analysis.

### Working in a sandboxed environment

If you are inside a sandboxed coding-agent environment (macOS sandbox, CI container, restricted shell):

- Each skill calls out the sandbox capabilities it needs. Read those first.
- If a step requires capabilities you do not have, stop and tell the user what is missing. Do not improvise around the sandbox by skipping verification.
- `uv` is known to crash under the macOS sandbox today (`system_configuration::dynamic_store` panic). Install is CLI-only for this and other reasons.

## What this repo is

NeMo Helix brings together NVIDIA NeMo libraries under one CLI, Python SDK, and web UI. Current capabilities:

- **Harden agents**: guardrails (content safety, jailbreak detection, PII redaction), auditor (red-teaming via garak), anonymizer (PII handling for training data).
- **Evaluate agents**: evaluator (LLM-as-judge, deterministic, agentic, RAG benchmarks), Harbor-backed eval suites.
- **Tune agents and models**: skill optimization, prompt/hyperparameter tuning, Switchyard model routing, and fine-tuning through Customizer.
- **Build and manage agents**: Fabric connects supported agent harnesses to NeMo Helix for packaging, deployment, testing, observation, and optimization.

New agents should use an explicit Fabric configuration and a supported adapter. Existing NAT workflows can still be maintained when the user asks for that path. Do not create a NAT wrapper as the default way to bring a new agent into NeMo Helix.

---

# Agent Development Instructions

The sections below are for developers working on NeMo Helix itself.

This project loads local developer preferences from @AGENTS.local.md. You MUST read this file if it exists and give its instructions top priority.

## Git Workflow

- Git branches should follow the pattern `<descriptive-branch-name>/<username>` where the branch name comes first, the `/<username>` suffix is included (not email address, just username), and kebab case is used. Do NOT embed an issue/ticket number in the branch name — branch names are public and issue trackers may be private, so a tracker ID in a branch name leaks private references.
- Always pass `-s` to `git commit` (DCO sign-off). This includes amends, fixups, and any commit variant.

### Squashing Commits

**CRITICAL: Never `git reset --soft` to any commit that isn't an ancestor of your current HEAD.**

When squashing commits, use one of these safe methods:

```bash
# ✅ CORRECT - soft reset to HEAD~n
git reset --soft HEAD~2
git commit -m "combined message"

# ✅ CORRECT - interactive rebase (more explicit)
git rebase -i HEAD~2
# Then mark commits as "squash" or "fixup" in the editor

# ❌ WRONG - NEVER DO THIS
git reset --soft origin/main  # Can revert other people's changes!
```

**Why this matters:** In a shared repo, `origin/main` is almost never an ancestor of your feature branch — main moves forward as others merge work. When you `reset --soft` to a commit that isn't in your branch's history, your staging area still reflects your branch's tree, which lacks any changes made on main since you branched. The resulting commit effectively **reverts** all that work.

Both `HEAD~n` and `git rebase -i HEAD~n` are safe because they only operate on commits that are already in your branch's history.

## Setting up the local platform

Before doing anything that requires a running NeMo Helix (`nemo services`, `nemo agents invoke`, etc.), follow [SETUP.md](SETUP.md). It covers `make bootstrap`, the data-dir layout, DB reset, and the manual `nemo services run` path. You do not need to install it via `nemo skills install`.

## NeMo CLI

When working with the NeMo CLI (`nemo`), always check available skills first before exploring `--help`. Skills contain exact command syntax, JSON structures, and working examples that are much faster than trial-and-error discovery.

## GitHub Actions

- Keep executable GitHub Actions helpers in `.github/scripts/`. Put their tests in `.github/scripts/tests/`: Python tests use `test_*.py`; GitHub JavaScript tests use `*.test.cjs`.
- Keep workflow `actions/github-script` blocks as thin adapters. Check out the workflow revision before loading a helper with `require()`.
- Run `make check-github-scripts` to test, lint, and format-check GitHub JavaScript helpers locally.
- Run `flox activate --dir tools/actionlint -- actionlint` to validate GitHub workflow and action syntax.

## Building a Studio plugin web UI

A plugin can ship a web UI that Studio loads at runtime and renders **inside its own React tree** — sharing Studio's React, router, and KUI design system rather than bundling its own. Before writing or reviewing plugin web code, read [plugins/example-plugin/web/AGENTS.md](plugins/example-plugin/web/AGENTS.md): it holds the contract (a `Root` component + `navItems`, externalized shared deps, KUI + theme tokens, auth rules) and is the canonical template to copy.

## Writing Python Code

- Don't put `__init__.py` files in packages. Instead prefer implicit namespace packages.

### Python Style notes

- Always prefer concrete type hints over string based ones. DO NOT import these types under TYPE_CHECKING. Instead prefer to import the types a regular import when possible.
- Keep NeMo Helix SDK and typed client naming distinct. Variables holding generated `NeMoHelix` or `AsyncNeMoHelix` instances should be named `sdk` or `async_sdk`. Variables holding `nemo_helix_plugin` typed clients should be named `client`, `async_client`, or service-specific names such as `files_client`, `jobs_client`, or `models_client`. Do not name typed clients `sdk`, and do not merge generated SDKs and typed clients into one public type; adapt at the boundary with `client_from_platform`.

### Python Package Management

- Use uv exclusively for Python package management in all projects.

#### Package Management Commands

- All Python dependencies **must be installed, synchronized, and locked** using uv
- Never use pip, pip-tools, poetry, or conda directly for dependency management

**Use these commands:**
- Install dependencies: `uv add <package>`
- Remove dependencies: `uv remove <package>`
- Sync dependencies: `uv sync`

#### Running Python Code

- Run a Python script with `uv run <script-name>.py`
- Run Python tools like Pytest with `uv run pytest` or `uv run ruff`
- Launch a Python repl with `uv run python`

### SDK

The Python SDK (`sdk/python/nemo-helix`) is checked in to this repository and hand-maintained. The `nemo` CLI is hand-written in `nemo_helix_ext` and bundled into the SDK package at build time.

The OpenAPI spec at `openapi/openapi.yaml` is the source of truth for the platform's HTTP API routes. It is regenerated locally from the FastAPI service code (no cloud credentials required).

**Update the OpenAPI spec:**
- `make refresh-openapi` - Regenerate `openapi/openapi.yaml` from API definitions

**Update web SDK / CLI:**
- `make update-sdk` - Regenerate the OpenAPI spec and TypeScript web SDK (Orval), then vendor `nemo_helix_ext` into the SDK and regenerate the CLI reference docs.

**When to run `make refresh-openapi`:**
Run it whenever you modify:
- API endpoints (routes, methods, parameters, responses)
- Data models or schemas
- Files in these paths:
  - `packages/nhx_common/src/nhx_common/datamodel/`
  - `packages/nhx_common/src/nhx_common/api/`
  - Service API files: `services/*/src/*/api/`

**Note:** OpenAPI generation also runs as a pre-commit hook (manual stage) when API files change.

#### Changing SDK types

The SDK package is not regenerated. If a previously generated type or client needs to change, do not edit it in `sdk/python/nemo-helix`: use the corresponding typed client from `nemo_helix_plugin` instead and migrate consumers to it.

#### Testing Python Code

- When verifying solutions, prefer to write unit tests instead of executing python snippets.
- Don't put an `__init__.py` file in test directories. tests are not modules.

### Type Checking

Use the `ty` tool for type checking:
- Check all files: `uv run --frozen ty check`

Type checking runs automatically in CI via the `lint:uv` job.

### Linting

**Python Style (Ruff):**
- Lint all files: `uv run ruff check`
- Format check all files: `uv run ruff format --check`
- Lint single file: `uv run ruff check path/to/file.py`
- Format single file: `uv run ruff format path/to/file.py`

**Licenses:**
- Update and validate licenses: `make update-licenses`

**OpenAPI Spec:**
- Validate OpenAPI spec: `script/generate-openapi-spec.sh`
- Or use: `make refresh-openapi`

**SDK Validation:**
- Update SDK: `make update-sdk`

**OPA Policy:**
- Check policy WASM is up-to-date: `make check-policy`
- Build policy WASM: `make build-policy`

Linting runs automatically in CI via GitHub Actions in `.github/workflows/ci.yaml`; Studio web linting is covered by `.github/workflows/studio-ci.yaml`.

### Pre-commit Hooks

Pre-commit hooks run automatically before commits and pushes to ensure code quality. They can also be run manually.

**Run all hooks manually:**
- `uv run pre-commit run -a` - Run all pre-commit hooks

**What the hooks do:**
- **Ruff linter** - Automatically fixes linting issues in Python code (excludes SDK)
- **Ruff formatter** - Formats Python code (excludes SDK)
- **Type checking (ty)** - Runs type checks on Python code (may need manual fixes)
- **uv lock** - Automatically updates `uv.lock` with the platform uv version when `pyproject.toml` changes
- **uv lock check** - Verifies `uv.lock` is in sync with `pyproject.toml`
- **Helm Docs Container** - Runs `helm-docs` container to regenerate Helm documentation in `k8s/helm/README.md`
- **Check merge conflicts** - Detects merge conflict markers
- **OpenAPI generator** (manual stage) - Regenerates OpenAPI spec when API files change
- **Check policy WASM** (pre-push only) - Verifies OPA policy WASM is up-to-date

**Before attempting to commit:**
Ensure all pre-commit hooks pass by running `uv run pre-commit run -a`. A clean run (no changes) means you're ready to commit. Type checking errors may require manual fixes.

## Testing services

### Running Tests

**All tests:**
- All unit tests: `make test-unit`
- Integration tests: `make test-integration`
- All tests (unit + integration): `make test-all`

**Specific tests:**
- Specific service tests: `make test-unit-<service>` (e.g., `make test-unit-evaluator`)
- Specific package: `make test-package PACKAGE=<package_name>` (e.g., `make test-package PACKAGE=nhx_common`)
- Single test file: `uv run --frozen pytest path/to/test_file.py -v`
- Single test function: `uv run --frozen pytest path/to/test_file.py::test_function_name -v`

**Test utilities:**
- Watch mode (re-run on changes): `make test-watch`
- With coverage report: `make test-coverage` (generates HTML report in `htmlcov/index.html`)
- Debug mode: `make test-debug`
- Re-run failed tests: `make test-failed`

**Note:** E2E tests are currently disabled. Use `make test-unit` iteratively, then `make test-integration` for comprehensive verification.

## Cursor Cloud specific instructions

### Bootstrap prerequisites

- **Toolchain:** Install Flox before running `make bootstrap`; Make activates it as needed. For a preinstalled host toolchain, install the versions printed by `make toolchain-versions` and use `TOOLCHAIN=system`.
- **uv version:** Root `pyproject.toml` specifies the supported minimum for source checkout bootstrap. `make toolchain-versions` reports the exact version used for `uv.lock` updates, Flox, platform containers, and CI.
- **Native build deps:** `make bootstrap-python` builds `annoy` (via `nemoguardrails`). Install system headers once per VM image: `sudo apt-get install -y python3-dev build-essential`.
- **Python bootstrap:** Run `make bootstrap-python` from repo root (creates `.venv`, runs `uv sync --frozen --all-packages`). See [SETUP.md](SETUP.md) for the full playbook.
- **Studio (optional):** `make bootstrap-studio` uses the Node.js/pnpm versions pinned in the Flox environment, so a VM shipping an older Node does not need upgrading. API services still run without Studio assets. For a preinstalled host toolchain, install the versions printed by `make toolchain-versions` and use `TOOLCHAIN=system`.
- **Docker:** Not needed for dependency bootstrap. It is required before running `nemo setup` or local services.

### Running the platform

Before any `nemo` CLI command against a local instance, set:

```bash
export NHX_BASE_URL=http://localhost:8080
```

Check for an existing instance before starting (`lsof -iTCP:8080 -sTCP:LISTEN` or `nemo workspaces list`). To start all default services in the background, use a tmux session:

```bash
tmux -f /exec-daemon/tmux.portal.conf new-session -d -s nemo-helix -c /workspace -- \
  'export NHX_BASE_URL=http://localhost:8080 && uv run nemo services run --service-group all --controller-group all --port 8080'
```

Wait for readiness: `curl -sf http://localhost:8080/health/ready` → `{"status":"ready"}`.

Minimal subset for inference-focused work (documented in SETUP.md): `uv run nemo services run --services entities,models,inference-gateway,secrets --controllers models`.

### Lint and test

Standard commands from repo root (see sections above): `uv run ruff check`, `uv run --frozen ty check`, `make test-unit`, `make test-package PACKAGE=<name>`.

- **Docker:** Not required for core platform smoke tests. One `nhx_common` config test expects Docker and sets runtime to `none` when Docker is unavailable.
- **Inference providers:** `nemo setup`, `nemo chat`, and agent invoke require an API key (`NVIDIA_API_KEY`, `OPENAI_API_KEY`, etc.). Core APIs (workspaces, secrets, hello-world, entities) work without provider credentials.

### Quick smoke verification

```bash
export NHX_BASE_URL=http://localhost:8080
curl -sf http://localhost:8080/health/ready
curl -sf http://localhost:8080/apis/hello-world/v2/workspaces/default/hello
uv run nemo workspaces list
```
