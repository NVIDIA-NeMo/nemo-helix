<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# rl backend — Kubernetes job execution requirement

The `rl` (DPO **and GRPO**) backend runs **each job step as a Kubernetes pod** via the
`kubernetes_job` execution backend. This is different from `automodel` / `unsloth`,
which run on either the `docker` or the `kubernetes_job` backend, depending on how
the platform is configured. rl has no docker fallback, so the platform you submit
against must be deployed/configured for Kubernetes job execution — the docker job
backend cannot run rl, and `rl --job-json` fails fast (`require_distributed_runtime`)
on a docker-runtime platform.

Deployment model is the same as automodel/unsloth: **run the platform locally**
(`nemo services run` — never anything else). The only difference for rl is the
**execution backend** the local platform dispatches jobs to: it must be
`kubernetes_job` (pointing at a Kubernetes GPU cluster via its kubeconfig), never `docker`.
What matters is that the platform's jobs backend is `kubernetes_job` and the job
pods can reach the platform's APIs.

`SKILL.md` § **Plugin pick** (the **platform check**) sends you here when the GPU execution profiles report `backend: docker` or `subprocess`. Configure the local platform as below. If **no** Kubernetes cluster is available to point it at, stop and tell the user rl needs one.

## Configuring the local platform for rl

When you start the platform locally (`nemo services run`) for an rl job, it must
be configured with all of:

1. `platform.runtime: kubernetes`.
2. `jobs` `kubernetes_job` executors registered for **both** providers the
   customizer stamps — `cpu` (download / upload / model-entity steps) and `gpu`
   (DPO training) — at the resolved profile.
3. `platform.loopback_address` set to a platform address the **job pods can reach**
   (the platform rewrites the `NHX_*_URL` it injects into pods to this, so the
   download/upload steps can call the files/jobs APIs).
4. The target GPU cluster has, available as pullable/loaded images: the job-step
   images (`nhx-customizer-tasks`, `nhx-rl-training`), the **jobs-launcher** image (each
   step runs a launcher init container), and a **job-storage PVC** the steps share.
5. Multi-node only (`parallelism.num_nodes > 1`): `NHX_RL_MULTINODE_SHARED_STORAGE_PATH`
   (a shared filesystem for Ray's cross-node coordination).

If a job pod shows `ErrImagePull` / `ImagePullBackOff` on the launcher init
container or a step image, that image isn't available in the cluster — surface it;
do not build/pull it as part of the customization workflow.

## Sandboxed Gym (GRPO)

GRPO adds one more cluster requirement on top of everything above: the Gym servers
that run user environment code execute in an **OpenSandbox** pod, separate from the
training pod. Only serialized JSON crosses the boundary; the sandbox reaches the
job's vLLM engine over scoped egress.

`sandboxed_gym_default` is **true**, so this is the default path for GRPO — not an
opt-in. Two settings gate it, and **both fail at submit, before any GPU is claimed**:

| Setting | Env var | Owner | Why |
|---|---|---|---|
| `platform.sandbox_cluster_capable` | `NHX_SANDBOX_CLUSTER_CAPABLE` (Helm: `sandboxClusterCapable`) | platform | Declares OpenSandbox is installed. The platform Helm chart does **not** install it. `false` → `OpenSandbox is not yet available on this cluster (sandbox_cluster_capable=false)` |
| `rl.job_storage_pvc_claim` | `NHX_RL_JOB_STORAGE_PVC_CLAIM` | rl service | The sandbox re-mounts the job-storage PVC to read the downloaded environment and dataset, and only learns the claim by name. Unset → `Sandboxed GRPO requires the job-storage PVC claim name` |

DPO needs neither. Neither is settable per job — if a submit fails on one, it is a
platform configuration gap, not a problem with the user's package. Say so and stop.

Installing OpenSandbox is an operator task with its own guide:
`docs/set-up/helm/opensandbox.mdx` (and `opensandbox-kata.mdx` for the Kata runtime).
Once installed, the platform also needs:

| Setting | Required? |
|---|---|
| `platform.sandbox_server_domain` | Yes — the in-cluster OpenSandbox Service DNS, no scheme |
| `platform.sandbox_server_protocol` | **Set it to `http` for the normal in-cluster install.** Unset, NeMo-RL builds health URLs as `https`, which an in-cluster Service does not speak — the job then stalls waiting for a sandbox that never reports ready. Only leave it unset when OpenSandbox is genuinely reachable over HTTPS |
| `platform.sandbox_api_key_secret` | Yes — the OpenSandbox API-key Secret, copied from `opensandbox-system` into the release namespace (key: `platform.sandbox_api_key_secret_key`, default `api-key`) |

### Egress from the sandbox

Deny-default. An environment whose per-server venv build has to reach a package
index needs the operator to open it:

| Setting | Env var | Notes |
|---|---|---|
| `rl.sandbox_allow_internet` | `NHX_RL_SANDBOX_ALLOW_INTERNET` | Default `false`. Required for every `native-v1` package and any wheels package whose closure is incomplete. Converter-built `adapter-wheels-v1` packages do not need it at job start. |
| `rl.sandbox_public_dns_allow` | `NHX_RL_SANDBOX_PUBLIC_DNS_ALLOW` | Extra suffixes/FQDNs, consulted only when the above is true. NeMo-RL's built-in list covers `*.com` and `*.org`, so e.g. `hub.primeintellect.ai` must be named here |

Cluster-private, node-local and metadata destinations stay denied either way. A
complete `wheels-v1` closure avoids needing any of this — see
`../../nemo-gym-environments/references/dependencies.md`.

### Ingress to the training pod

The sandbox must reach the training pod's vLLM port and the episode broker, and must not reach its Ray ports. That isolation is a per-step ingress NetworkPolicy, and it is **off by default**.

| Setting | Default | Notes |
|---|---|---|
| `networkPolicies.jobSteps.enabled` | `false` | Helm value. Set `true` to write `jobs.executor_defaults.kubernetes_job.step_network_policies_enabled` and the same flag on `volcano_job`. The jobs controller then creates and deletes an ingress NetworkPolicy for each step that declares `network`, which sandboxed GRPO does. |
| `jobs.executor_defaults.{kubernetes_job,volcano_job}.step_network_policies_enabled` | `false` | The resolved flag. A `platformConfig` override turns the controller, its NetworkPolicy RBAC, and the ValidatingAdmissionPolicy on or off together. Enabling it needs a CNI that enforces NetworkPolicy (Cilium on the minikube Docker driver; the default CNI accepts the resources and does not enforce them) and Kubernetes 1.30+ for the admission policy, or 1.28+ with the ValidatingAdmissionPolicy beta API. |

Leave it off and GRPO still runs, but a Gym sandbox can open the training pod's Ray ports. Both
`networkPolicies.enabled` and `networkPolicies.jobSteps.enabled` must be true. The operator setup is
`docs/set-up/helm/sandboxed-grpo.mdx`.

### Rollout transport and sandbox sizing (operator-scoped)

All jobs-invisible; jobs cannot set them. Reach for these only when a running GRPO
job fails in a way that matches:

| Setting | Default | Symptom it addresses |
|---|---|---|
| `rl.sandbox_rollout_chunk_size` | NeMo-RL's 8 | Every rollout fails with HTTP 500 after a long POST — the OpenSandbox proxy caps how long one request may stay open, and a large `max_new_tokens` overruns it. Read elapsed time on the failing POSTs and divide down |
| `rl.sandbox_rollout_max_in_flight` | NeMo-RL's 8 | In-flight rollouts are chunk × this. Raise in proportion when lowering the chunk, or step throughput falls with it |
| `rl.sandbox_ttl_s` | NeMo-RL's 14400 (4h) | A run outliving the reap timer loses its sandbox mid-rollout and fails with a proxy 502. Must exceed the longest accepted run; capped by the server's `max_sandbox_timeout_seconds` |
| `rl.sandbox_resources` | `cpu: "4"`, `memory: 8Gi` | Sandbox OOMKilled mid-rollout, surfacing as a proxy 502 rather than a memory error. The pod runs one Gym server process per config entry plus its own Ray. The OpenSandbox SDK default (1 CPU / 2Gi) is too small for that |
| `rl.gym_runtime_image` | the `nhx-rl-training` image | Only if the sandbox must run a different image |

Reference (local platform config): `docs/set-up/manage-jobs.mdx` (execution
backends — `kubernetes_job`), `docs/set-up/config-reference.mdx`
(`platform.runtime`, `loopback_address`, `kubernetes_job` executor config),
`docs/set-up/helm/opensandbox.mdx` (OpenSandbox install).
