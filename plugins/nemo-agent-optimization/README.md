<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# nemo-agent-optimization

Owns `nemo agents optimize` — the command group every agent optimization
strategy shares — and the router job behind it. The nemo-agents plugin 
doesn't have any optimization options without this plugin installed.

```
nemo agents optimize run-strategy     # submit a run (--strategy picks the implementation)
nemo agents optimize list-strategies  # what the platform has installed
nemo agents optimize prepare-fileset  # contributed by the `nat` strategy
```

`RunStrategyJob` resolves `--strategy` to an installed strategy job and
delegates: `compile` returns that job's steps verbatim, so the strategy's work
*is* the run — one job record, no child submission, no polling. Submissions go
to this plugin's own API segment:

```
POST /apis/agent-optimization/v2/workspaces/{workspace}/jobs/run-strategy
GET  /apis/agent-optimization/v2/strategies
```

The strategies listing is not workspace-scoped: which strategies exist is a
property of what the deployment installed, identical in every workspace.

## The strategy contract

A strategy is an ordinary `nemo.jobs` entry whose class declares one extra
class variable:

```python
class OptimizeJob(NemoJob):
    nemo_agent_optimization_strategy: ClassVar[OptimizationStrategy] = OptimizationStrategy(
        name="nat",
        description="Hyperparameter and GA prompt optimization.",
    )
```

That is the whole contract _for now_. 

The declared value is the same `OptimizationStrategy` the listing route returns,
so `list-strategies` shows what the owning plugin wrote rather than a
description borrowed from the job — the job's own `description` introduces the
job to CLI users, which is a different sentence.

To add a strategy:

1. Write a `NemoJob` with `compile` and `run` as usual, and set
   `nemo_agent_optimization_strategy` to an `OptimizationStrategy` naming what
   users will pass to `--strategy` and saying what it optimizes.
2. Declare it under `nemo.jobs` in your plugin's `pyproject.toml`. The key
   prefix is yours; it decides where (and whether) the job gets its own route
   and CLI command, and discovery here ignores it either way:

   ```toml
   [project.entry-points."nemo.jobs"]
   "my-plugin.optimize" = "my_plugin.jobs.optimize:MyOptimizeJob"
   ```

3. Optionally contribute companion verbs to the shared `optimize` group by
   declaring a plain `def register(group: typer.Typer) -> None` under
   `nemo.cli.agents.optimize`. The group is shared, so prefix your strategy name
   onto anything that is not plainly generic. (The `nat` strategy holds the
   generic `prepare-fileset`.)

   ```toml
   [project.entry-points."nemo.cli.agents.optimize"]
   my-plugin-stage = "my_plugin.cli:register_stage_command"
   ```

   This is independent of step 2: a plugin can contribute a verb without
   shipping a strategy job, and can ship a strategy job without contributing a verb.

The router forwards every field but `strategy` to your job's own `spec_schema`
and validates it there, so your job keeps full ownership of what its inputs
mean — a config that must be fileset-relative, an output target you cannot
write, and so on. A mismatch surfaces as a 422 on the submit.

## Discover the installed strategies

`--strategy` takes any name this prints, one per line:

```bash
$ nemo agents optimize list-strategies
Targeting http://localhost:8080
nat
```

Only the names go to stdout (the target line is on stderr), so the output is
safe to loop over in a shell; when the platform has no strategies, stdout is
empty and a note goes to stderr.

It asks the platform — `GET /apis/agent-optimization/v2/strategies` — because
the platform is what resolves `--strategy` when the run is submitted. A client
venv missing a strategy plugin still submits to a server that has it, and a
client that has one can submit to a server that does not, so answering from the
client would be answering the wrong question. When the platform cannot be
reached the command reports that on stderr and exits non-zero rather than
listing this environment, because this environment describes a different
machine.

Only the names go to stdout, one per line, so piping stays clean.

The list is whatever the platform has installed: each entry is a job's
declared `nemo_agent_optimization_strategy`, so installing a plugin that ships
one adds it without any change to this package.
