<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

## Conventions

Inherited from the NeMo Helix monorepo that now hosts this plugin:

- Use `uv` exclusively (`uv add`, `uv sync`, `uv run`). No pip/poetry/conda.
- No `__init__.py` files — use implicit namespace packages.
- Concrete type hints, not string-based. Don't hide imports under `TYPE_CHECKING`.
- Lint/format with `uv run ruff check` and `uv run ruff format`.
- All files need the SPDX header (`Copyright (c) ... NVIDIA CORPORATION & AFFILIATES`, `Apache-2.0`).

## Active migrations

### 2026-08-19: Eval Author agent moved back into this plugin

`nemo_experimentalist_plugin.eval_author` owns the Eval Author agent again, along with
`eval_author/traces.py`. The 2026-07-28 extraction aimed at a standalone Eval Author
plugin. That goal is retired: the customer-facing path is a skill rather than a plugin
install, and insight mode was the agent's only caller.

**The dependency is one arrow, and it points out of the Eval Author plugin.**
Experimentalist imports nothing from `nemo-eval-author-plugin`, so the package cycle that
`uv` had to resolve is gone.

- The agent's borrows of the evaluator, staging, trace, tools, and reporting helpers are
  intra-package imports. The Eval Author plugin's boundary ratchet existed to drive those
  ten borrows to zero, so it went away with them.
- `tests/test_contract_dependency.py` asserts that this plugin never declares
  `nemo-eval-author-plugin` as a dependency, which is what keeps the cycle broken.
- Agent tests live in `tests/eval_author/`. This plugin's `conftest.py` already covers the
  isolation those tests need, so the Eval Author copy went away.

### 2026-07-31: Command group nested under `nemo agents`

The only path is `nemo agents experimentalist <verb>`. `ExperimentalistCLI` is
registered under the `nemo.cli.agents` entry-point group, which the `nemo-agents`
plugin's `AgentsCLI` discovers and mounts. There is no top-level
`nemo experimentalist` alias.

The Analyst follows the same rule: `nemo agents analyst run` (was `nemo insights
analyze`). Prefer `ctx.command_path` over a hardcoded path when a message quotes the
command back to the user.

### 2026-07-28: Eval Author extracted to its own plugin, heading for standalone (superseded)

Superseded by the 2026-08-19 entry, which moved the agent back.

`plugins/nemo-eval-author/` took ownership of the `eval_author/` package, targeting a
standalone plugin that imported nothing from Experimentalist. Both arrows pointed at each
other for the duration, and `uv` resolved the cycle.

One rule from that entry still holds: both plugins use
`nemo_helix_plugin.nooa_model_client`, so provider routing, Platform authentication,
and configured model selection have one owner outside either plugin.

### 2026-07-24: Optimizer renamed to Experimentalist

Ahead of the move into the `nemo-helix` monorepo, the plugin was renamed from
Optimizer to Experimentalist. Like the Eval Author rename below, this is a
breaking rename with no compatibility aliases:

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

Two names deliberately did **not** change. `optimizer.yaml` and the
`.nemo-optimizer/` state directory are a shared contract with
`nemo-insights-plugin`: `PROFILE_FILENAME` and `discover_profile()` live in
`nemo_insights_plugin.contracts.profile`, and `nemo agents analyst run` writes
`<profile-dir>/.nemo-optimizer/insights.yaml`, which this plugin reads as the
default insight. Rename them only in lockstep with a Platform change to that
contract.

`EvolutionaryOptimizer` is now `EvolutionaryStrategy`, in
`experimentalist/strategies/evolutionary.py`, and is resolved by name like any other
component (`strategy: evolutionary`).

Run `nemo agents experimentalist components` to see everything this install can resolve,
including components registered by a separately installed package.

### 2026-07-21: Curator renamed to Eval Author

The Curator agent was renamed directly to Eval Author in ASE-643. This is a
breaking rename with no compatibility aliases or migration layer:

- `nemo_experimentalist_plugin.curator` → `nemo_experimentalist_plugin.eval_author` (briefly `nemo_eval_author_plugin.eval_author` while the agent had its own plugin)
- `Curator`, `CuratorConfig`, and `CuratorResult` → `EvalAuthor`,
  `EvalAuthorConfig`, and `EvalAuthorResult`
- `run_curator(...)` and `build_curator_agent(...)` →
  `run_eval_author(...)` and `build_eval_author_agent(...)`
- the `curator` optimizer configuration block → `eval_author`
- Curator-specific artifact paths, cache keys, helpers, tests, and documentation
  now use `eval_author` / Eval Author terminology

Agents working on in-flight branches that touch these surfaces should rebase
before resolving conflicts, keep the Eval Author names, and update their code
instead of restoring Curator imports, configuration, or aliases. The obsolete
`curator` configuration key is intentionally rejected with a validation error.

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
