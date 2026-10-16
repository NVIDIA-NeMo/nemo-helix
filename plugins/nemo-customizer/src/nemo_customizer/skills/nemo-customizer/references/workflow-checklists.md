<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Workflow checklists by backend

Step-by-step checklists. The common steps come first, then one branch per backend. Pick the branch from **Plugin pick** in `SKILL.md`.

```text
- [ ] Resolve CLI (SKILL.md § Pre-flight — CLI resolution); cd nemo-helix
- [ ] export NHX_BASE_URL (if user provided endpoint); note whether base URL is user-overridden
- [ ] nemo auth status — skip login if auth disabled; if auth enabled and unsigned JWT allowed, `nemo auth login --unsigned-token --email <…>`; if OIDC, `nemo auth login`
- [ ] nemo jobs list-execution-profiles -f json — apply the Plugin pick rules in SKILL.md (retry login on 401/403)
- [ ] On connection error: default URL → ask to start platform (see Platform unreachable in references/gotchas.md); custom URL → report unreachable and stop
- [ ] Convert HF dataset → /tmp/train-data/*.jsonl (see references/hf-conversion.md)
- [ ] Create dataset fileset (new name when the data changes; no --exist-ok, which can leave stale files — see references/gotchas.md), upload the JSONL files, nemo files list to verify — automodel/unsloth: train.jsonl (+ validation.jsonl); rl: training.jsonl + validation.jsonl (see rl branch)
- [ ] Gated HF base model? → confirm `hf-token` exists; ask user and stop if missing (see SKILL.md § HuggingFace token + references/troubleshooting.md § Gated HuggingFace models)
- [ ] Create HF weights fileset + model entity if missing (--exist-ok; gated repos need `token_secret` on fileset — see references/troubleshooting.md)
- [ ] Find the model's reference recipe in references/recipes.md; take its hyperparameters and parallel layout into the job JSON (skip max_steps / ci settings)

# automodel branch (submit → platform GPU job: docker or kubernetes_job backend)
- [ ] Write /tmp/job.json (batch sizing for ≥48 GB GPU; else the Defaults table in SKILL.md)
- [ ] nemo customization automodel --job-json /tmp/job.json --workspace default
- [ ] Poll until top-level terminal (`poll_customization_job.sh <name>` — pass the "name" field from submit stdout verbatim; it already carries the `automodel-` prefix; default 15s interval, or 30–60s manual polls)
- [ ] Report using the template in `references/reporting.md`
- [ ] Optional: compare base vs adapter on validation — `references/eval_helpers.py …` (LoRA only; CHAT format; adapters hot-reload automatically; see `references/post-training-eval.md`)

# unsloth branch (submit → platform GPU job: docker or kubernetes_job backend)
- [ ] Write /tmp/job.json using the UnslothJobInput shape (see references/fast-path-sft.md § Fast path — unsloth)
- [ ] nemo customization unsloth --job-json /tmp/job.json --workspace default [--profile <gpu-profile>]
- [ ] Poll until top-level terminal (`poll_customization_job.sh <name>` — pass the "name" field from submit stdout verbatim; it already carries the `unsloth-` prefix; default 15s interval)
- [ ] Report using the template in `references/reporting.md`
- [ ] Optional: compare base vs adapter on validation — `references/eval_helpers.py …` (LoRA only; CHAT format; adapters hot-reload automatically; see `references/post-training-eval.md`)

# rl branch (DPO; submit → Kubernetes/Ray job) — requires platform.runtime: kubernetes
- [ ] Verify execution backend: `nemo jobs list-execution-profiles -f json` shows cpu/gpu at `backend: kubernetes_job` (NOT docker/subprocess). If not → stop; do not start a docker platform; configure per references/rl-kubernetes-runtime.md
- [ ] Dataset is PREFERENCE data: upload training.jsonl + validation.jsonl ({prompt,chosen,rejected}) to ONE fileset
- [ ] Write /tmp/job.json using the RlJobInput shape (see references/fast-path-rl.md § Fast path — rl (DPO))
- [ ] nemo customization rl --job-json /tmp/job.json --workspace default [--profile <gpu-profile>]
- [ ] Read job id from the "name" field in submit stdout (JSON) — submit has no --name flag; do NOT pick the newest rl-* from `nemo jobs list`
- [ ] Poll until top-level terminal (`poll_customization_job.sh <name>` — pass that same "name" value verbatim; it already carries the `rl-` prefix; default 15s interval)
- [ ] Report using the template in `references/reporting.md`

# rl branch (GRPO; submit → Kubernetes/Ray job) — requires platform.runtime: kubernetes
- [ ] Verify execution backend (same gate as DPO above)
- [ ] Confirm the cluster runs sandboxed Gym: sandbox_cluster_capable + job-storage PVC claim are operator config and fail AT SUBMIT (references/rl-kubernetes-runtime.md § Sandboxed Gym (GRPO)); ask the operator about egress before choosing native-v1
- [ ] Pick the ENVIRONMENT format — verifiers/hub env → adapter-wheels-v1; Gym tree WITH egress → native-v1; everything else (incl. Gym tree, no egress) → wheels-v1 (see references/gym-environments.md)
- [ ] Build the ENV package in a dedicated env, not the project `.venv` (`UV_PROJECT_ENVIRONMENT=.venv-conversion uv sync --frozen --package nhx-rl --extra conversion`, then that env's python). `wheels-v1` runs `pip` as `sys.executable`; doing that in the project env diverges from `uv.lock` and the next `flox activate` fails, which blocks commits. Gym server → `scripts/grpo-examples/gym_to_env_package.py --gym-root <Gym checkout> --nemo-rl-root <NeMo-RL checkout> --server resources_servers/<name> --format <wheels-v1 or native-v1> --arch <x86_64 or aarch64>` (NeMo Gym is NOT vendored here: `git clone https://github.com/NVIDIA-NeMo/Gym`). Hub/verifiers env → `pi-to-gym-conversion --hub-id … --hub-version … --out-dir …` on an internet-capable host. Both are EXAMPLES; hand-building the manifest + configs/policy_model.yaml + server dirs works too
- [ ] Build the DATASET rows. math_with_judge → `scripts/grpo-examples/prepare_math_with_judge.py --out-dir … --train-size …` (adds agent_ref + expected_answer). Otherwise convert per references/dataset-formats.md, reading the env's *VerifyRequest first
- [ ] Create the MODEL entity (HF weights fileset + `nemo models create`) — same as step 2 in references/fast-path-sft.md; see scripts/grpo-examples/README.md § 4
- [ ] Validate BEFORE upload: `pi-to-gym-conversion --validate-only <pkg-dir>` — cheapest possible failure. It checks LAYOUT ONLY: not wheel tags, not closure completeness, not rows
- [ ] Upload environment (--purpose environment — enforced at submit) and dataset (--purpose dataset) as TWO filesets; trailing slash on the local dir; no .jsonl inside the env package; `nemo files list` to confirm nothing nested
- [ ] Dataset rows are GYM ROLLOUT ROWS (prompt under responses_create_params.input + agent_ref object) — see references/dataset-formats.md § NeMo-RL (GRPO)
- [ ] Write /tmp/job.json with training.type "grpo" + the `environment` string ref (see references/fast-path-rl.md)
- [ ] nemo customization rl --job-json /tmp/job.json --workspace default [--profile <gpu-profile>]
- [ ] Read job id from the "name" field in submit stdout (JSON) — submit has no --name flag
- [ ] Poll until top-level terminal (`poll_customization_job.sh <name>` — pass that same "name" value verbatim; it already carries the `rl-` prefix)
- [ ] Report on REWARD, not loss (references/reporting.md)
```

