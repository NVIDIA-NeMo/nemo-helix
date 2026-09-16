<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# nemo-agent-optimization

Entry point for `nemo agents optimize`. `OptimizeJob` (this plugin) resolves
`--strategy` to an installed strategy job, delegates `compile`/`run` to it, and
returns its steps — one job record, no child submission, no polling. See
`nemo_agent_optimization_plugin.job_base` for the base class every strategy
implements.

## The strategy contract

A strategy is not a separate plugin protocol; it is an ordinary `nemo.jobs`
entry whose class happens to subclass `AgentOptimizeJob`. Discovery
(`nemo_agent_optimization_plugin.discovery.discover_agent_optimize_jobs`)
walks every installed job and keeps the ones that are `AgentOptimizeJob`
subclasses, keyed by their `strategy` class variable — membership is
type-checked, not name-matched, so nothing has to register under a
strategy-specific entry-point group.

To add a strategy:

1. Subclass `AgentOptimizeJob` (`nemo_agent_optimization_plugin.job_base`) and set:
   - `strategy: ClassVar[str]` — the name users pass to `--strategy`.
   - `task_module: ClassVar[str]` — the module the platform runs as
     `python -m <task_module>` for a remote submission.
   - `name`, `description`, and other `NemoJob` class variables as normal.
2. Implement `optimize(self, *, source_agent_config, config, ctx, workspace, sdk) -> dict`,
   returning an optimized `nemo-agents-spec-v1` config dict. The base class
   already resolved the Agent under Test (`source_agent_config`), staged the
   bundle referenced by `--optimize-config-fileset`/`--optimize-config` into
   the process working directory, and will register whatever this method
   returns as a new agent entity — `optimize()` only has to do the
   strategy-specific transform.
3. Ship a task entry point, `tasks/agent_optimize.py`, that calls
   `nemo_platform_plugin.tasks.dispatcher.run_task` with your job class (see
   any of the installed strategies for the ~20-line pattern).
4. Register the class under `nemo.jobs` in your plugin's `pyproject.toml`,
   e.g.:

   ```toml
   [project.entry-points."nemo.jobs"]
   "my-strategy.agent_optimize" = "my_plugin.jobs.agent_optimize:MyAgentOptimizeJob"
   ```

Every strategy job's `spec_schema` is the same normalized
`nemo_agent_optimization_plugin.schemas.optimize.AgentOptimizeSpec`:
`agent`, `optimize_config_fileset`, `optimize_config`, `output_agent`, and
`workspace` are all required (`workspace` defaults to `"default"`). `agent`
is the platform agent under test; `optimize_config_fileset` +
`optimize_config` locate the strategy's own bundle (stage one with
`nemo agents optimize prepare-fileset`); `output_agent` names the new agent
entity the run creates.

`nemo agents optimize` itself (the router, `OptimizeJob`) is CLI-facing and
takes one extra field, `strategy`, via its own
`OptimizeSubmitSpec` — that's where `--strategy` comes from. `OptimizeJob`
strips `strategy` back off and re-validates the rest as `AgentOptimizeSpec`
(`_child_spec()`) before delegating to the resolved strategy job, so a
strategy's `optimize()` never sees `strategy` on its spec.

## Discover the installed strategies

`nemo agents optimize --strategy` takes any name this command prints, one per
line:

```bash
$ nemo agents optimization-strategies list
nat
prompt-master
```

The list is whatever is installed in the current environment: each name comes
from an installed `AgentOptimizeJob` subclass's `strategy` class variable, so
installing a plugin that ships one adds it here without any change to this
package.
