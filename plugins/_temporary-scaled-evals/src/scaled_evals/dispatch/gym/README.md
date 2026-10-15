<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dispatch via NeMo Gym

Gym-based sandbox paths exist in scaled-evals. With ``GYM_RUNNER_IMAGE`` set,
they launch one-shot **gym-runner** containers via the Docker socket (see
[`docs/RUNNER.md`](../../../../docs/RUNNER.md)). ``sandbox_k8s`` is unchanged.

| Runtime | Agent | Sandbox seam | Harness |
|---------|-------|--------------|---------|
| ``gym_daytona`` | ``harbor_agent`` | Harbor's built-in Daytona environment | ``examples/gym-daytona/`` |
| ``gym_sandbox_daytona`` | ``mini_swe_agent_2`` | ``nemo_gym.sandbox`` → Daytona SDK (PR #1377/#1513) | ``examples/gym-sandbox-daytona/`` |
| ``gym_sandbox_opensandbox`` | ``mini_swe_agent_2`` | ``nemo_gym.sandbox`` → OpenSandbox cell | ``examples/gym-sandbox-opensandbox/`` |

See also [`docs/RUNNER.md`](../../../../docs/RUNNER.md) for
requirement coverage vs the Sandbox Requirements PDF.

## LaunchSpec contract

Gym backends honor the immutable `LaunchSpec` inputs at dispatch time instead of
falling back to chart-wide defaults:

- `framework_config` from a `gym` config profile is materialized into `GYM_*`
  harness environment variables for both host-process and Docker runners.
- `image_ref` / `image_digest` are exposed as `TASK_IMAGE` and
  `SCALED_EVALS_TASK_IMAGE_DIGEST` so the runner uses the requested task image.
- `tarball_object_key` stages the uploaded task pack into the per-evaluation work
  directory and points `TASK_PATH` at that staged tree.
- `extra_skill_object_keys`, `instruction_prefix`, and `instruction_postfix`
  require a staged uploaded task pack with `task.toml` and fail closed if the
  mutation cannot be applied.
- `parallelism` and `n_attempts` are passed through as Gym Hydra overrides
  (`num_samples_in_parallel` and `num_repeats`) and recorded in provenance.

Gym currently rejects `network_policy=default_deny`, `network_policy=scoped_egress`,
non-empty `network_policy_config`, and `initial_user_turns` because these providers
cannot enforce or consume those inputs safely. Use `sandbox_k8s` when Kubernetes
egress policy is required.

## gym_daytona (Harbor harbor_agent)

Harbor trials run through Gym's ``harbor_agent`` with
``harbor_environment_type: "daytona"``. Works on Gym ``main`` today.

```
GYM_DAYTONA_ENABLED=true
GYM_DIR=~/src/Gym-daytona-refresh
GYM_DAYTONA_ENV_FILE=examples/gym-daytona/targets/daytona.env
```

Oracle Terminal-Bench smoke does not require a policy model.

Compose dispatch (recommended):

```
GYM_DAYTONA_ENABLED=true
GYM_RUNNER_IMAGE=scaled-evals-gym-runner:dev
GYM_DAYTONA_ENV_FILE=/harness/gym-daytona/targets/daytona.env
GYM_DAYTONA_WORK_DIR=/tmp/gym-daytona
GYM_DAYTONA_DOCKER_VOLUME=scaled-evals-gym-daytona-work
```

Host dev (``run.sh`` / ``run_and_collect.sh``) still needs ``GYM_DIR`` on the host.

## gym_sandbox_daytona (nemo_gym.sandbox)

SWE-bench tasks run through ``mini_swe_agent_2``, which creates sandboxes via
Gym's provider-neutral ``nemo_gym.sandbox`` facade and the Daytona provider.

Requires Gym with merged [PR #1377](https://github.com/NVIDIA-NeMo/Gym/pull/1377)
plus open [PR #1513](https://github.com/NVIDIA-NeMo/Gym/pull/1513). The reviewed
personal refresh SHA is ``b6199c4c00dd55a356b7bacd0b01f342858d2298``; install
it with ``uv sync --frozen --extra sandbox``.

```
GYM_SANDBOX_DAYTONA_ENABLED=true
GYM_DIR=~/src/Gym-daytona-refresh
GYM_SANDBOX_DAYTONA_ENV_FILE=examples/gym-sandbox-daytona/targets/daytona.env
```

Requires a policy model endpoint (``openai_model`` config or Hydra overrides).

Dispatch and ``run.sh`` use **``run_and_collect``** (``ng_run`` → ``ng_collect_rollouts`` →
shutdown). Do not point ``gym_sandbox_daytona`` at ``ng_e2e_collect_rollouts`` for
custom JSONL — that command needs ``++split=`` and dataset prep from YAML configs.

## gym_sandbox_opensandbox (nemo_gym.sandbox)

SWE-bench tasks run through ``mini_swe_agent_2`` and Gym's OpenSandbox provider.
Dev uses a user-owned ``kubectl port-forward`` to the NeMo RL cell; prod uses a
stable cell URL (in-cluster DNS or corpnet) and service-owned secrets.

```
GYM_SANDBOX_OPENSANDBOX_ENABLED=true
GYM_RUNNER_IMAGE=scaled-evals-gym-runner:dev
GYM_SANDBOX_OPENSANDBOX_ENV_FILE=/harness/gym-sandbox-opensandbox/targets/opensandbox.env
GYM_SANDBOX_OPENSANDBOX_WORK_DIR=/tmp/gym-sandbox-opensandbox
```

Target env files (same ``runtime``, different substrate wiring):

| File | Use |
|------|-----|
| ``opensandbox-dev.env`` | Compose + port-forward |
| ``opensandbox-nrl-colocated.env`` | Prod: runner on NeMo RL EKS |
| ``opensandbox-nrl-remote.env`` | Prod: runner on a remote cluster |

For compose on macOS, use ``OPENSANDBOX_DOMAIN=http://host.docker.internal:18080``.
For host-only smoke scripts, use ``http://127.0.0.1:18080``.

See ``examples/gym-sandbox-opensandbox/README.md`` for compose vs cluster notes.

## Shared gaps

- Hosted Kubernetes has no working Gym path yet (AALGO-906). Evaluations run as Platform Jobs in the plugin's app image, without the snapshot-pinned runner image, `GYM_RUNNER_MODE=process`, Gym resources, or `/dev/shm`, so the default docker runner mode fails in the Job pod. See the plugin README's "Background execution".
- Provider quota/capacity can fail sandbox creation mid-run

## Three-path comparison (including K8s)

| | ``sandbox_k8s`` | ``gym_daytona`` | ``gym_sandbox_daytona`` | ``gym_sandbox_opensandbox`` |
|---|---|---|---|---|
| Needs cluster | Yes | No | No | Yes |
| Needs Gym | No (Harbor CLI) | Yes | Yes (+ sandbox extra) | Yes (+ OpenSandbox provider) |
| Needs model (oracle smoke) | No | No | Yes (SWE-bench agent) | Yes (SWE-bench agent) |
| Sandbox isolation | K8s Sandbox CR | Daytona SDK | Daytona SDK via Gym API | OpenSandbox cell |
