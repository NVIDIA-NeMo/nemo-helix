<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# prompt-master — optimize a platform agent's system prompt

The `prompt-master` strategy for `nemo agents optimize`. It runs the vendored
[Prompt Master](https://github.com/nidhinjs/prompt-master) skill once through a
Fabric agent, hands it the selected platform agent's current system prompt as
inert data, and writes back a complete `nemo-agents-spec-v1` config carrying the
optimized `instructions.system.content`. The stored agent is never modified.

```
nemo agents optimize run-strategy --strategy prompt-master ...
```

The job is discovered by the router in `plugins/nemo-agent-optimization` the way
every strategy is: `PromptMasterOptimizeJob` is a plain `nemo.jobs` entry that
declares `nemo_agent_optimization_strategy`. Nothing outside this directory
knows the strategy exists.

## Install

This plugin is not a member of the root uv workspace, so `uv sync` does not
install it. Install it into the repo venv as an editable package after the
normal bootstrap:

```bash
make bootstrap-plugins BOOTSTRAP_LOCAL_PLUGIN_DIRS=plugins/prompt-master
# or, equivalently:
uv pip install -e plugins/prompt-master
```

Re-run that after any `uv sync`, which removes packages the lockfile does not
know about. `nemo agents optimize list-strategies` shows `prompt-master` once
the platform process has the package installed.

The job step prefers the `subprocess` execution profile, which runs in the
platform host's own venv. Deployments without one fall back to the `cpu`
profile's `nhx-tasks` image; because this plugin is not part of the stock
`cpu-tasks` dependency group, that image must be built with
`prompt-master-plugin` installed for the step to import.

## Configure the optimizer

The optimizer is itself a platform agent, defined once in
[src/prompt_master_plugin/agent.yaml](src/prompt_master_plugin/agent.yaml): a
Deep Agents harness carrying the vendored skill, a default NVIDIA model, and
the one-shot system instructions. It runs as-is with no config.

To change any of it, pass a partial `agent.yaml` in the same
`nemo-agents-spec-v1` format. Only the fields you give replace the bundled
ones; everything else is kept:

```yaml
models:
  default:
    provider: nvidia
    model: nvidia-nemotron-3-5-lightning-30b-a3b
    temperature: 0.0
runtime:
  timeout_seconds: 600
```

Models are served through the platform's Inference Gateway: the job binds the
gateway URL for its workspace at run time and authenticates its inference
calls with its own identity, so neither file names a provider URL or an API
key. `models.default.model` is a model ID from `nemo models list`. To replace
the optimizer's own system instructions, set `instructions.system.content`.
[examples/prompt-master.yaml](examples/prompt-master.yaml) is a ready-to-use
copy of the above.

## Run

`run-strategy` submits a platform job, so the job cannot read paths on the
submitting client: an optimizer config, if any, has to be in a fileset, and the
agent has to be registered on the platform.

```bash
# 1. Register the agent whose system prompt you want to optimize (skip if it exists).
uv run nemo agents create \
  --name calculator-agent \
  --agent-config plugins/nemo-agents/examples/nemo-agent-config/calculator-agent/agent.yaml

# 2. (Optional) Stage optimizer overrides in a fileset.
uv run nemo files filesets create prompt-master-bundle
uv run nemo files upload plugins/prompt-master/examples/prompt-master.yaml prompt-master-bundle

# 3. Submit. --optimize-config is relative to the fileset root and is omitted, together
#    with --optimize-config-fileset, to run the bundled optimizer unchanged.
uv run nemo agents optimize run-strategy \
  --strategy prompt-master \
  --agent calculator-agent \
  --optimize-config-fileset default/prompt-master-bundle \
  --optimize-config prompt-master.yaml \
  --workspace default

# 4. Fetch the optimized agent config from the job's results and register it as a new agent.
#    A directory result downloads as a tarball holding agent.yaml and prompt-master-result.json.
uv run nemo jobs results download prompt_master --job <job-name> -o prompt_master.tar.gz
mkdir -p prompt-master-results && tar -xzf prompt_master.tar.gz -C prompt-master-results --strip-components=1
uv run nemo agents create \
  --name calculator-agent-optimized \
  --agent-config ./prompt-master-results/agent.yaml
```

Stage the config with `nemo files` rather than `nemo agents optimize
prepare-fileset`: that verb belongs to the `legacy` strategy, preflights a
hyperparameter bundle, and rejects a config with no `optimizer:` section.

## What the run produces

The artifacts are registered as the job's `prompt_master` result, in the job's
own fileset on the platform:

| File | Contents |
| --- | --- |
| `agent.yaml` | The source agent's full `nemo-agents-spec-v1` config with the optimized `instructions.system.content`. Registers as-is with `nemo agents create --agent-config`. |
| `prompt-master-result.json` | Source agent, optimizer model, original and optimized prompts, and Prompt Master's full reply. |

The job result also carries `optimized_prompt` directly, so the new prompt can
be read without downloading anything.

## Spec

The router forwards every submitted field but `strategy` to this strategy's
own schema, which is what decides what is required. Fields it does not declare,
including the router's `--output`, are refused: the artifacts only ever land in
the job's results.

| Field | Flag | Required | Meaning |
| --- | --- | --- | --- |
| `agent` | `--agent` | yes | Platform agent to optimize, `name` or `workspace/name`. |
| `optimize_config` | `--optimize-config` | no | Optimizer overrides (a partial `agent.yaml`), relative to the fileset root. |
| `optimize_config_fileset` | `--optimize-config-fileset` | with `optimize_config` | Fileset holding the overrides. |

## Layout

```
src/prompt_master_plugin/
  agent.yaml            the optimizer agent; a run's overrides merge over it
  jobs/optimize.py      PromptMasterOptimizeJob — the strategy (compile + run)
  schemas/optimize.py   PromptMasterOptimizeSpec
  runner.py             the one-shot Fabric invocation and prompt extraction
  tasks/optimize.py     task entrypoint: python -m prompt_master_plugin.tasks.optimize
  vendor/prompt-master/ the vendored skill (MIT; see LICENSE there)
```

The vendored skill files are byte-identical to upstream; NeMo-specific one-shot
instructions live in `agent.yaml` so the upstream boundary stays explicit.
