<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# strands-harness-optimizer — tune a platform agent's system prompt against a dataset

The `strands-harness-optimizer` strategy for `nemo agents optimize`. It wraps
[Strands Harness Optimizer](https://github.com/strands-labs/harness-optimizer):
the selected platform agent's `nemo-agents-spec-v1` config is converted into a
Strands agent on the same model (reached through the Inference Gateway), the
library's rollout-and-reflect loop runs it over a labelled dataset for a few
epochs, a reflection model proposes an improved system prompt after each epoch,
and the final prompt is written back as a complete `nemo-agents-spec-v1`
`agent.yaml` in the job's results. The stored agent is never modified.

```
nemo agents optimize run-strategy --strategy strands-harness-optimizer ...
```

The job is discovered by the router in `plugins/nemo-agent-optimization` the way
every strategy is: `StrandsHarnessOptimizeJob` is a plain `nemo.jobs` entry that
declares `nemo_agent_optimization_strategy`.

## Install

This plugin is opt-in. It is not part of `nemo-helix[all]` and not a root
uv-workspace member, so a bare `uv sync` does not install it. It is locked under
its own dependency group:

```bash
uv sync --frozen --all-packages --group strands-harness-optimizer
```

Re-run that after any sync without the group, which removes packages outside the
requested groups. `nemo agents optimize list-strategies` shows `strands-harness-optimizer`
once the platform process has the package installed.

The job step prefers the `subprocess` execution profile, which runs in the
platform host's own venv. Deployments without one fall back to the `cpu`
profile's `nhx-tasks` image; because this plugin is opt-in, that image must be
built with `strands-harness-optimizer-plugin` installed for the step to import.

## The bundle

The strategy reads one YAML and the JSONL dataset it names, both uploaded to
one fileset. `dataset` is relative to the YAML and must stay in its directory
or below. [examples/](examples/) holds a ready-to-use pair:

```yaml
# strands-harness-optimizer.yaml
dataset: dataset.jsonl
epochs: 2            # rollout-and-reflect passes over the dataset
batch_size: 8        # rows per rollout batch
num_workers: 1       # parallel agent invocations per batch
# max_samples: 16    # use only the first N rows
# optimizer_model: nvidia-nemotron-3-5-lightning-30b-a3b   # reflection model; defaults to the agent's own
```

```jsonl
{"id": "add", "input": "What is 17 + 26?", "expected_output": "43"}
{"id": "sub", "input": "What is 100 - 58?", "expected_output": "42"}
```

Each row is `{"input": str, "expected_output": str}` with an optional `"id"`.
A rollout earns reward 1.0 when the whitespace- and case-normalized
`expected_output` occurs in the agent's reply, else 0.0.

## Run

`run-strategy` submits a platform job, so the job cannot read paths on the
submitting client: the bundle has to be in a fileset and the agent registered on
the platform.

```bash
# 1. Register the agent whose system prompt you want to optimize (skip if it exists).
nemo agents create \
  --name calculator-agent \
  --agent-config plugins/nemo-agents/examples/nemo-agent-config/calculator-agent/agent.yaml

# 2. Stage the bundle.
nemo files filesets create sho-bundle
nemo files upload plugins/strands-harness-optimizer/examples/strands-harness-optimizer.yaml sho-bundle
nemo files upload plugins/strands-harness-optimizer/examples/dataset.jsonl sho-bundle

# 3. Submit. --optimize-config is relative to the fileset root.
nemo agents optimize run-strategy \
  --strategy strands-harness-optimizer \
  --agent calculator-agent \
  --optimize-config strands-harness-optimizer.yaml \
  --optimize-config-fileset default/sho-bundle

# 4. Fetch the optimized agent config from the job's results and register it as a new agent.
#    A directory result downloads as a tarball holding agent.yaml and the run summary.
nemo jobs results download strands_harness_optimizer --job <job-name> -o strands_harness_optimizer.tar.gz
mkdir -p sho-results && tar -xzf strands_harness_optimizer.tar.gz -C sho-results --strip-components=1
nemo agents create --name calculator-agent-optimized --agent-config ./sho-results/agent.yaml
```

Stage the bundle with `nemo files` rather than `nemo agents optimize
prepare-fileset`: that verb belongs to the `legacy` strategy and rejects a
config with no `optimizer:` section.

## What the run produces

The artifacts are registered as the job's `strands_harness_optimizer` result:

| File | Contents |
| --- | --- |
| `agent.yaml` | The source agent's full `nemo-agents-spec-v1` config with the optimized `instructions.system.content`. Registers as-is with `nemo agents create --agent-config`. |
| `strands-harness-optimizer-result.json` | Agent, dataset size, epochs, per-epoch average reward, and the original and optimized prompts. |

The job result also carries `optimized_prompt` directly.

## Spec

The router forwards every submitted field but `strategy` to this strategy's own
schema. Fields it does not declare, including the router's `--output`, are
refused: the artifacts only ever land in the job's results.

| Field | Flag | Required | Meaning |
| --- | --- | --- | --- |
| `agent` | `--agent` | yes | Platform agent to optimize, `name` or `workspace/name`. |
| `optimize_config` | `--optimize-config` | yes | The strategy YAML, relative to the fileset root. |
| `optimize_config_fileset` | `--optimize-config-fileset` | yes | Fileset holding the YAML and its dataset. |

## Limitations

- Only the system prompt is optimized. The agent's tools, skills and MCP servers
  are not forwarded to the Strands agent, so rollouts of a tool-using agent
  measure the model and prompt alone.
- The reward is a substring match on `expected_output`; tasks whose answers
  cannot be checked that way need a different reward.
- Models are reached through the Inference Gateway only: the agent's
  `models.default` (or its default harness's `model`) must name a gateway model.

## Layout

```
src/strands_harness_optimizer_plugin/
  config.py             StrandsHarnessConfig — the bundle YAML
  schemas/optimize.py   StrandsHarnessOptimizeSpec — the submitted spec
  strands_bridge.py     Fabric config -> Strands agent, reward, reflection optimizer, training loop
  jobs/optimize.py      StrandsHarnessOptimizeJob — the strategy (compile + run)
  tasks/optimize.py     task entrypoint: python -m strands_harness_optimizer_plugin.tasks.optimize
examples/               a ready-to-upload bundle
```
