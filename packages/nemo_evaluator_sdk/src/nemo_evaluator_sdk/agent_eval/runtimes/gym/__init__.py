# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo Gym-backed :class:`AgentTaskRunner` for the agent-eval pipeline.

Runs an *existing* NeMo Gym environment through its ``gym`` CLI and adapts the
rollout bundle into SDK :class:`AgentEvalTrial` objects, so an
:class:`AgentEvaluator` can score and report Gym runs through the same seam as
Harbor/Fabric/Model. Gym owns execution *and* scoring; this runtime imports an
existing Gym environment as-is — Gym is treated as a self-scoring engine and the
runtime only adapts its rollout bundle, it does not re-derive rewards.

**Mapping** (Gym → Evaluator): one Gym dataset → one run; each distinct row →
one :class:`AgentEvalTask` (id = content hash of the row); each attempt
(``_ng_rollout_index``) → one :class:`AgentEvalTrial`; the per-attempt verifier
``reward`` → a :class:`GymRewardMetric` score. ``num_repeats=R`` therefore yields
up to R trials per task. Row duplication is *not* a way to ask for repeated
attempts — ``num_repeats`` is (see :func:`discover_gym_tasks`).

**Attribution** is by ``_ng_task_index``, which this runtime *assigns* rather
than infers. Gym only auto-assigns an index when a row doesn't already carry one
(``rollout_collection._preprocess_rows_from_config``), and its own fallback
dedup keys off the **raw jsonl line text** — a rule we cannot reproduce from
parsed rows. So instead of guessing, :meth:`GymAgentTaskRunner.run_tasks`
materializes a normalized dataset (one line per requested task, ``_ng_task_index``
stamped explicitly) and feeds *that* to Gym. Gym echoes the index back on every
rollout record, giving a total, order-independent ``index → task`` map. This also
means a caller can run a **subset** of tasks without Gym rolling out the rest.

**Execution** is the two-step Gym flow (the one that reads a dataset directly
without triggering Gym's split-driven data-prep), preceded by a pre-flight:
``gym env validate`` merges the composed config and reports unset ``???`` values,
bad paths, and dangling cross-references without starting anything; then ``gym env
start`` brings up the resources-server + agent + model servers, and ``gym eval run
--no-serve --input <materialized dataset>`` collects rollouts against them. Both
commands receive the identical selection arguments, so what is validated is what
runs. The runtime shells out to the ``gym`` CLI on PATH, so this SDK never imports
``nemo_gym``. Subprocess
output is streamed to log files under the run's work dir *and* mirrored to this
module's logger at ``DEBUG``, so callers choose terminal visibility through
ordinary ``logging`` configuration.

**Where Gym finds things.** NeMo Gym must be installed and its ``gym`` on PATH,
along with the target environment's own dependencies. Generally that means a
*separate* environment: Gym imports Ray at module load, and nemo-helix
excludes Ray by constraint over an unfixed CVE, so the two cannot share one. In a
job image the image owns PATH and this is unremarkable. There is deliberately no
config field naming a checkout, a venv, or a search root — these runner configs
become serialized job specs, and a local filesystem path means nothing on the
other side of that boundary. Environments themselves ship in the ``nemo-gym``
wheel (``resources_servers`` and friends install beside ``nemo_gym``, configs and
example data included), so no checkout is needed to reach them.

The subprocesses inherit this process's working directory, which is where Gym
looks for the gitignored ``env.yaml`` holding the collector's credentials before
falling back to its install root — so credentials never pass through this SDK.
Run from the directory holding that file; a Gym checkout there also has its
components take precedence, which is how you reach an environment the wheel does
not carry.

**Boundaries**: the caller is responsible for a
Gym runtime whose deps are installed (each Gym env ships its own
``requirements.txt``), and for handing a *ready-to-run* dataset file (``--no-serve
--input`` bypasses Gym's prompt-templating/materialization). Service-side
provisioning (docker/k8s, Ray) is out of scope here — that is the plugin's job.

A consequence of that bypass: an environment whose rows carry no rendered prompt
(``responses_create_params.input == []``, the prompt supplied by data-prep or by the
environment's own agent) is still supported — the row travels through this runtime intact
and the task simply has no ``inputs['instruction']``. See :func:`discover_gym_tasks`.
"""

from nemo_evaluator_sdk.agent_eval.runtimes.gym.config import DEFAULT_REWARD_KEY, GymRuntimeConfig
from nemo_evaluator_sdk.agent_eval.runtimes.gym.dataset import discover_gym_tasks, validate_gym_task_row
from nemo_evaluator_sdk.agent_eval.runtimes.gym.runtime import GymAgentTaskRunner
from nemo_evaluator_sdk.metrics.runner_rewards import GymRewardMetric

__all__ = [
    "DEFAULT_REWARD_KEY",
    "GymAgentTaskRunner",
    "GymRewardMetric",
    "GymRuntimeConfig",
    "discover_gym_tasks",
    "validate_gym_task_row",
]
