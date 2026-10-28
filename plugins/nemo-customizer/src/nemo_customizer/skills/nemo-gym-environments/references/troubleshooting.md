<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `Missing nemo-environment.yaml at environment root` | No manifest, or uploaded one directory too deep | Manifest at the FileSet root, not inside a subdirectory |
| `Invalid nemo-environment.yaml: ... extra_forbidden` | Unknown manifest key | Remove it; the schema is closed |
| `config_paths reference files that are not in the package` | Upload flattened the tree, or a typo | Re-upload preserving relative paths |
| `Prompt JSONL must not live in the environment package` | `.jsonl` inside the package | Move rows to the dataset FileSet |
| `native-v1 config_paths must be under (...)` | Config outside the three Gym server dirs | Move it, or switch to `wheels-v1` |
| `adapter-wheels-v1 config_paths should live under configs/` | Config elsewhere | Move it under `configs/` |
| `format ... installs its dependencies offline, so the package must carry a non-empty wheels/ directory` | Wheels format with no wheels | Vendor the closure |
| `Non-wheel files in wheels/` | Stray file (sdist, README) | `wheels/` holds `*.whl` only |
| `adapter.agent ... is not built into the training image` | Harness not on the allowlist | Use `verifiers_agent`, or switch format |
| `ServerRefNotFoundError: ... Available responses_api_models: (none)` | A config references `policy_model` with nothing defining it | Add a `policy_model` config — **every** format needs one; see `gym-config.md` |
| `Symlinks are not allowed` | Symlinked config | Ship real files |
| `Environment fileset '…' has purpose 'dataset'; expected purpose='environment'` | Package uploaded to a fileset created with the wrong purpose | Recreate with `--purpose environment` |
| `AlmostServerError` / "Almost-Servers Detected" banner | A server block failed validation but looks like one — most often a `domain` that is empty or not in the closed set | Set a valid `domain` on every `resources_servers` block; remove `domain` from agent/model blocks |
| `Missing pyproject.toml or requirements.txt for uv venv setup in server dir` | Custom implementation shipped without an install marker | Add `requirements.txt` to `{server_type}/{implementation}/` |
| `Found both pyproject.toml and requirements.txt … Please only use one or the other!` | Both markers in one server directory | Keep one |
| Job runs but scores the wrong environment | Custom server dir has no install marker **and** shares a name with a Gym built-in, so Gym resolved the built-in | Add `requirements.txt`, or rename the implementation |
| `ModuleNotFoundError` at the first rollout | Server venv missing an import — empty/incomplete `requirements.txt`, or the wheelhouse missed it at venv-build time | List real imports; vendor the step-1 closure (`dependencies.md`) |
| `your requirements are unsatisfiable` on the cluster | `wheels/` vendors two versions of one project | Clear `wheels/` and rebuild the closure |
| `has no wheels with a matching platform tag` | Closure built for the build host (macOS / arm64), not the image | Re-download with the `--python-version` / `--platform` flags in `dependencies.md` |
| Spin-up stalls, then fails resolving a package | The per-server venv build cannot reach an index | Vendor it (`wheels-v1`), or ask the operator for `NHX_RL_SANDBOX_ALLOW_INTERNET` — see `dependencies.md` |
| `OpenSandbox is not yet available on this cluster (sandbox_cluster_capable=false)` | Operator has not enabled sandboxed Gym | Platform config, not the package — `../../nemo-customizer/references/rl-kubernetes-runtime.md` § **Sandboxed Gym (GRPO)** |
| `Sandboxed GRPO requires the job-storage PVC claim name` | `NHX_RL_JOB_STORAGE_PVC_CLAIM` unset | Same — operator config |

The last two fail **at submit**, before any GPU is claimed, and no change to the package will fix them.
