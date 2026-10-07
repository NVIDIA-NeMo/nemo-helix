<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# skill_eval — A/B evaluation of an injected agent skill

Runs one taskset twice through the Fabric agent-eval runtime — once **without** a
skill (baseline) and once **with** an injected [agentskills.io](https://agentskills.io)
skill (treated, via `runtime.with_skill(skill)`) — then compares the scores. The
two arms differ in *exactly* the skill, so the delta is attributable to it.

## The setup: the skill is *required* to pass

Each task asks the agent to *"write a Python function that ..., following the
Supercool Coding Guidelines."* Those guidelines live **only** in the injected
`supercool-guidelines` skill and are **not inferable from the prompt**:

- function names must start with `supercool_`;
- a positional-index parameter must be named `enieme` (French for "nth").

So the `follows_guidelines` metric can pass **only when the agent actually uses
the skill** — giving a clean, measured baseline-vs-treated difference rather than
a fuzzy "did it help".

## What it demonstrates

- Injecting an agentskills bundle into `FabricAgentRuntime` and running a
  confound-free A/B with `with_skill(...)` (baseline vs. treated as distinct
  `run_id`s).
- Task-authored guidelines metrics whose `follows_guidelines` output is
  skill-dependent by construction, scored off the **parsed function signature**
  (AST) so a convention only *mentioned* in prose doesn't count. Each task gets the
  metric matching its parameters: `GuidelinesMetric` (index tasks) checks for the
  `enieme` positional-index name — the unguessable discriminator — while
  `GcdGuidelinesMetric` (gcd, no index param) checks its parameters are French
  words. `supercool_prefix` is weakly guessable and reported on its own.
- `SkillUsedMetric` — `skill_present` / `skill_used` surface whether the agent
  used the injected skill.

The bundled skill is `skills/supercool-guidelines/` (a spec-compliant `SKILL.md`).

## Run it

**Prerequisites** — this example imports `nhx_evals_sdk` and drives the native
Fabric stack, so run it from the project virtualenv (a bare repo-root `python`
won't have the workspace on its import path):

- `make bootstrap-python` — creates `.venv` and `uv sync --all-packages`, which
  installs the workspace packages (including `nhx_evals_sdk`);
- `uv sync --extra fabric` — the `nemo-fabric` SDK, its adapters, and the `nemo-relay`
  Python bindings, all from the lock;
- `script/dev-install-fabric.sh` — the `nemo-relay` gateway binary, which is not on PyPI;
- `NVIDIA_API_KEY` for an account **provisioned for** `MODEL` in `run_skill_eval.py`.

Then, from the repo root, run with the venv interpreter:

```bash
NVIDIA_API_KEY=... ADAPTER_PYTHON="$(pwd)/.venv/bin/python" \
  .venv/bin/python -m packages.nhx_evals_sdk.examples.skill_eval.run_skill_eval
```

`ADAPTER_PYTHON` is required whenever the `python3` on your `PATH` is not this venv
(common on macOS/Homebrew, pyenv, etc.): the Fabric Hermes adapter runs as a
subprocess and otherwise falls back to a bare `python3` off `PATH`, which won't
have `nemo_fabric_adapters` installed (`python_adapter_exit_nonzero` /
`ModuleNotFoundError`).

Each arm writes a run bundle under `skill-eval-output/<arm>/`, with per-task
Fabric evidence under `evidence/fabric/<run_id>/`. If a trial fails (bad model
id, missing credential, harness crash), the run prints a `⚠️ N trial(s) FAILED`
block and exits non-zero rather than showing an empty-but-tidy table.

Example output (with `nvidia/nemotron-3-super-120b-a12b`):

```text
Harness: nvidia.fabric.hermes   model: nvidia/nemotron-3-super-120b-a12b   tasks: 2
runs: baseline (baseline) vs treated (treated)

  metric.output                             baseline   with-skill
  agent_phase_success.agent_phase_success   2/2        2/2
  follows_guidelines.enieme_param           0/1        1/1
  follows_guidelines.follows_guidelines     0/2        2/2
  follows_guidelines.french_params          0/1        1/1
  follows_guidelines.supercool_prefix       0/2        2/2
  skill_used.skill_present                  0/2        2/2
  skill_used.skill_used                     0/2        2/2
```

(`enieme_param` and `french_params` each total `/1` — they are the per-task checks,
emitted only by the index task and the gcd task respectively.)

## Notes

- `follows_guidelines` is the causal metric — it is what the skill directly
  controls. `skill_used` is the *mechanism* signal; it detects the skill's staged
  `location` in the trajectory and can under-report for the **Hermes** harness
  (in-context loading), so treat `follows_guidelines` as the source of truth for
  "did the skill take effect". See `SkillUsedMetric` for the detail.
- The model is the `MODEL` constant in `run_skill_eval.py`. It must be
  **provisioned for your account** — some catalog-listed models return HTTP 404
  (`Function ... not found for account`). It also has to be capable enough to
  *obey* the injected skill: weaker models read the guidelines but ignore them,
  producing a flat A/B (a valid, if undramatic, result).
  `nvidia/nemotron-3-super-120b-a12b` produces the lift shown above.
- The Hermes agent loop budget is `runtime.max_turns` (set to 50
  here). The Fabric Hermes adapter defaults it to **1**, which starves any
  multi-step task — leave it set.
- The guidelines metrics score the **parsed function signature** (`ast`, with a
  `def`-regex fallback), scoping to the `supercool_`-named answer function so a
  helper definition or a convention mentioned only in prose doesn't count. The gcd
  `french_params` check uses a small illustrative French wordlist (`_FRENCH_PARAMS`)
  — extend it if a run picks a French word not listed.
