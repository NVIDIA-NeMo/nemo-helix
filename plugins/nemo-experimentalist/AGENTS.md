<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

## Conventions

Inherited from the NeMo Platform monorepo that now hosts this plugin:

- Use `uv` exclusively (`uv add`, `uv sync`, `uv run`). No pip/poetry/conda.
- No `__init__.py` files — use implicit namespace packages.
- Concrete type hints, not string-based. Don't hide imports under `TYPE_CHECKING`.
- Lint/format with `uv run ruff check` and `uv run ruff format`.
- All files need the SPDX header (`Copyright (c) ... NVIDIA CORPORATION & AFFILIATES`, `Apache-2.0`).
- Use `nemo_platform_plugin.nooa_model_client` for provider routing, Platform
  authentication, and configured model selection.

## Active migrations

### 2026-07-31: Command group nested under `nemo agents`

The only path is `nemo agents experimentalist <verb>`. `ExperimentalistCLI` is
registered under the `nemo.cli.agents` entry-point group, which the `nemo-agents`
plugin's `AgentsCLI` discovers and mounts. There is no top-level
`nemo experimentalist` alias.

Analysis is submitted through `nemo insights analysis-runs create --agent <agent> --wait`. Prefer `ctx.command_path` over a hardcoded path when a message quotes the
command back to the user.

### 2026-07-24: Optimizer renamed to Experimentalist

Ahead of the move into the `nemo-platform` monorepo, the plugin was renamed from
Optimizer to Experimentalist. This is a breaking rename with no compatibility
aliases:

- distribution `nemo-optimizer-plugin` → `nemo-experimentalist-plugin`, source
  path `src/nemo_optimizer_plugin` → `src/nemo_experimentalist_plugin`
- `OptimizerCLI` → `ExperimentalistCLI`, and the `nemo.cli.agents` and
  `nemo.skills` entry-point keys are now `experimentalist`, so the command is
  `nemo agents experimentalist ...` (historically also briefly exposed as a
  top-level `nemo experimentalist` alias; that alias is gone)
- the `experiment` verb is now `run`: `nemo agents experimentalist run`
- `OPTIMIZER_API_BASE`, `OPTIMIZER_API_KEY`, `OPTIMIZER_{SMART,MID,FAST}_MODEL_NAME`,
  `OPTIMIZER_MODEL`, `NEMO_OPTIMIZER_E2E`, and `NEMO_OPTIMIZER_RUNTIME_CACHE` are
  now `NEMO_EXPERIMENTALIST_*`
- the dataset cache moved from `~/.cache/nemo-optimizer/` to
  `~/.cache/nemo-experimentalist/`, so cached datasets re-download once

Two names deliberately did **not** change: `optimizer.yaml` and the
`.nemo-optimizer/` state directory. The Experimentalist's profile helpers,
`PROFILE_FILENAME` and `discover_profile()`, still live in
`nemo_insights_plugin.contracts.profile`. Existing local insight files can still
be read from `<profile-dir>/.nemo-optimizer/insights.yaml`; AnalysisRuns write to
Platform, so pass the resulting Insight ID explicitly with `--insight`.


`EvolutionaryOptimizer` is now `EvolutionaryStrategy`, in
`experimentalist/strategies/evolutionary.py`, and is resolved by name like any other
component (`strategy: evolutionary`).

Run `nemo agents experimentalist components` to see everything this install can resolve,
including components registered by a separately installed package.

## Development environment

- The root `.venv` is the only Experimentalist environment; use the README for the
  standard `uv` lint, test, and run commands.
- NeMo Experimentalist consumes Insights produced by the Platform Insights plugin.
  Configure Platform services, trace storage, and Insights analysis according
  to the Platform documentation; this repository does not own a service,
  scheduler, or testbed setup.
- `nooa` comes from PyPI. This plugin's `pyproject.toml` declares the floor
  (`nooa>=0.0.9`), which is the first release carrying the callable
  `@strategy(llm=...)` support the components depend on. When raising the floor,
  move it in this plugin, `nemo-insights`, and
  `examples/tau3-nooa-agent/pyproject.toml`, update the tag quoted in
  `framework-skills/nooa/SKILL.md`, and relock both lock files together. Keep the
  Platform-supplied Insights plugin separate.
- This branch uses merged Platform PR 718 contracts only. After the Platform
  handoff lands, rebase and repin before adopting any new Platform testbed or
  installer interfaces.
