<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# nemo-eval-author — author a platform agent's first evals

`nemo evaluator author first-eval` runs the vendored
[NeMo Eval Author](https://github.com/NVIDIA-NeMo/labs-eval-author) `eval-author-first-eval`
skill once through a Fabric Deep Agents run and hands it a registered platform agent as the
repository to author evals for. The stored agent is never modified.

```bash
nemo evaluator author first-eval --agent my-agent
```

## What the job stages and produces

The job fetches the agent's stored config from the platform and writes it to `/agent.yaml` in the
author's workspace. When the agent has a companion `<agent>-ethos` fileset (or `--agent-fileset`
names one), that fileset is downloaded into the workspace first, so `ETHOS.md` and any docs it holds
are visible to the skill. The vendored skills library is staged at `/.agents/skills`.

The skill runs non-interactively: it establishes the Ethos (creating `/ETHOS.md` when missing),
plans the evaluation scope, and drafts Harbor task cases. Everything it writes under
`/.eval-author/` plus `/ETHOS.md` is collected together with `eval-author-result.json` (agent,
author model, output fileset, file list, the author's final response).

## Where outputs land

- The `<agent>-evals` fileset in the agent's workspace (or `--output <fileset>`), created when missing:
  `nemo files download <agent>-evals -o ./evals`.
- The job's `eval_author` result:
  `nemo jobs results download eval_author --job <job-name> -o eval-author.tar.gz`.

## Spec

| Field | Flag | Required | Meaning |
| --- | --- | --- | --- |
| `agent` | `--agent` | yes | Platform agent, `name` or `workspace/name`. |
| `agent_fileset` | `--agent-fileset` | no | Fileset staged as the agent's repository; defaults to `<agent>-ethos` when it exists. |
| `output` | `--output` | no | Fileset receiving the outputs; defaults to `<agent>-evals`. |
| `author_config` | `--author-config` | with `author_config_fileset` | Partial `agent.yaml` overriding the bundled author agent, relative to the fileset root. |
| `author_config_fileset` | `--author-config-fileset` | with `author_config` | Fileset holding `author_config`. |

The author agent is defined in [src/nemo_eval_author_plugin/agent.yaml](src/nemo_eval_author_plugin/agent.yaml):
a Deep Agents harness, a default NVIDIA model served through the workspace's Inference Gateway,
and the non-interactive system instructions. An overrides file is a partial file in the same
format; only the fields it gives replace the bundled ones:

```yaml
models:
  default:
    model: nvidia-nemotron-3-5-lightning-30b-a3b
runtime:
  timeout_seconds: 900
```

Stage it with `nemo files upload author.yaml my-bundle` and pass
`--author-config author.yaml --author-config-fileset my-bundle`.

## Limitations

- Only the `eval-author-first-eval` skill is exposed; the other vendored skills ship because it links into them. Upstream skills it never references are not vendored (see `vendor/UPSTREAM.md`).
- There is no evaluation runtime (Harbor, Gym, Docker) inside the job: the deliverable is the Ethos,
  the evaluation plan, and task drafts, not an executed suite.

## Layout

```
src/nemo_eval_author_plugin/
  agent.yaml              the author agent; a run's overrides merge over it
  cli.py                  EvalAuthorCLI, mounted at `nemo evaluator author`
  jobs/first_eval.py      FirstEvalJob (compile + run)
  runner.py               the one-shot Fabric invocation
  schemas/first_eval.py   FirstEvalSpec
  service.py              EvalAuthorService: /apis/eval-author/v2/workspaces/{ws}/jobs/first-eval
  tasks/first_eval.py     task entrypoint: python -m nemo_eval_author_plugin.tasks.first_eval
  vendor/                 upstream skills, LICENSE, NOTICE (byte-identical; see UPSTREAM.md)
```
