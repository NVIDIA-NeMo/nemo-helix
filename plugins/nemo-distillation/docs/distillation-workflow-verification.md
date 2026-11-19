<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Distillation workflow v0: workbox verification

Target: the workbox platform at `NHX_BASE_URL` (`http://10.2.183.187:8080`, port assumed).
Teacher: `Qwen/Qwen3.6-35B-A3B-FP8`. Students: `Qwen/Qwen3.5-2B` and `Qwen/Qwen3.5-4B`.
Source agent: the packaged `email-security-triage` sample agent.

Status key: `[ ]` not started, `[~]` in progress, `[x]` passed, `[!]` failed (note the cause under the step).

---

## 0. Preconditions

- [ ] `nemo-platform` tree on the workbox includes `plugins/nemo-distillation` and the `customization.atif-convert` job.
- [ ] Platform image rebuilt and deployed (loads `distillation` and `distillation-workflows`).
- [ ] `nhx-customizer-tasks` image rebuilt and deployed. It must contain `nemo_customizer.tasks.atif_convert`. The convert step fails without it.
- [ ] Workbox GPUs: two RTX 5880 Ada (48GB, sm89), plus a GT 1030 that is hidden from the platform. One 5880 holds the hand-deployed judge (see Judge model below). The other is the workflow's, so every workflow below sets `gpu_budget: 1`. The budget decides how many jobs run at once (`gpu_budget // GPUs per job`). At 1 the workflow runs one job at a time, in this order, with no help from outside: teacher deploy, agent deploy, original evals, agent undeploy, teacher undeploy, convert, then baseline (student deploy, agent deploy, evals, undeploy), then each candidate (train, deploy, agent deploy, evals, undeploy).
- [ ] Nothing but the judge holds a GPU when a run starts. Delete the pre-flight `default/qwen36-teacher` deployment (steps 4 and 6 left it and an agent deployment up). The workflow deploys and removes its own.
- [ ] The judge is up: `default/judge-qwen36` is `READY` (see Judge model below).
- [ ] `export NHX_BASE_URL=http://10.2.183.187:8080`

## Judge model (hand-deployed, outside the workflow)

Both example eval configs score with `llm-judge` metrics whose `model` is `default/nvidia-nemotron-3-super-120b-a12b`. That model is not on the workbox, and a 120B judge does not fit. The first run failed on it: both original evals returned 422 `Model reference ... not found` at submit.

For testing, the judge is the 35B itself, deployed once by hand on its own GPU and left up for every arm. The workflow's teacher is a second deployment of the same model on the other GPU. It is not a workflow node, so the workflow never undeploys it and every eval, including the baseline and candidate arms, has a judge to call.

```bash
nemo inference deployment-configs create judge-qwen36-config --workspace default --engine vllm \
  --model-spec '{"model_namespace":"default","model_name":"qwen3-6-35b-a3b-fp8"}' \
  --executor-config '{"gpu":1}'
nemo inference deployments create judge-qwen36 --workspace default --config judge-qwen36-config
```

- [x] `default/judge-qwen36` reached `READY` at 17:45:49 MST (puller about 2.5 minutes, server about 3 minutes). A test chat through it returned text.
- [ ] The uploaded copies of the eval configs name the judge `default/qwen3-6-35b-a3b-fp8` (step 7 does the swap with `sed`). The repo examples are not changed.
- vLLM logs "Your GPU does not have native support for FP8 computation ... Marlin" on these sm89 cards. The warning is unconditional text. The cause is kernel selection: for block-quantized FP8 on CUDA, vLLM `main` lists DeepGEMM and CUTLASS block kernels first (both need sm90 or newer), then Marlin, whose only requirement is sm75, ahead of the Triton block-FP8 kernel. sm89 therefore lands on Marlin. This is vLLM `main` at `34f1fc8`; the image's version was not checked. Effect: weight-only FP8, slower but correct.

## 1. Platform is reachable

```bash
curl -sf "$NHX_BASE_URL/health/ready" && echo ready
```

- [x] Returns `{"status":"ready"}` (HTTP 200, 2026-11-18 ~15:12 MST). The first probe dropped with an empty reply; the retry succeeded.

## 2. Distillation routes are mounted

```bash
curl -s -X POST "$NHX_BASE_URL/apis/distillation/v2/workspaces/default/workflow-templates/distillation:render" \
  -H 'content-type: application/json' \
  -d '{"source_agent":"default/email-security-triage","student_model":"default/qwen3-5-2b"}'
```

- [x] HTTP 200.
- [x] `counts.nodes` is 10, `counts.candidates` is 1 (`counts.gpu_hours` is 2). This was the graph before evals were moved in front of `prepare`.
- [ ] Re-render after the platform image includes the workflow-owned teacher and agent deployments. The render body needs `teacher` (or `trajectory_filesets`), `agent_deployment`, and `evals`. With both example evals, one LoRA candidate, and `gpu_budget: 1`, `counts.nodes` is 21, `counts.gpu_hours` is 4 (teacher, base deploy, LoRA deploy, LoRA train, each 1 GPU), and `counts.parallel_jobs` is 1. `deploy_teacher` has no dependencies. `deploy_agent_original` waits on it, and both original evals wait on that agent deployment. `prepare` depends on `eval_original[tasks]` and `eval_original[dataset]`. `undeploy_teacher` and `undeploy_agent_original` depend on those evals. `deploy_base` waits on `undeploy_teacher`. `train[lora]` waits on `prepare`, `undeploy_teacher`, and the baseline's `undeploy_agent_base` and `undeploy_base`.
- [ ] Re-render with `trajectory_filesets` and no `teacher`: no `deploy_teacher`, `deploy_agent_original`, `eval_original[*]`, `undeploy_teacher`, or `undeploy_agent_original`. `prepare` has no dependencies and its `trajectory_filesets` are the ones supplied.
- [ ] Re-render with teacher and `gpu_budget: 1` and `2` candidates: the second candidate's `train` depends on the first candidate's `undeploy_agent[...]` and `undeploy[...]`. A body with both `teacher` and `trajectory_filesets`, or `teacher.gpu` above `gpu_budget`, returns 422.
- [x] Graph has `train[lora]`, plus `deploy[lora]`, `eval_distilled[lora]`, `undeploy[lora]`. The new graph names the distilled evals `eval_distilled_tasks[lora]` and `eval_distilled_dataset[lora]`.
- [ ] Render writes nothing: the workflow list is unchanged afterwards. (Not yet compared against a before-listing.)

HTTP 404 means the running image does not include the plugin.

## 3. Register the three checkpoints as model entities

Each `nemo files filesets create` and `nemo models create` must succeed (409 on re-run is fine).

- [x] `default/qwen3-6-35b-a3b-fp8` (fileset `Qwen/Qwen3.6-35B-A3B-FP8`, pinned revision `c653bba46dfa06e9658dcc0b0dc87fe4685f14a2`)
- [x] `default/qwen3-5-2b` (fileset `Qwen/Qwen3.5-2B`, pinned revision `15852e8c16360a2fea060d615a32b45270f8a8fc`)
- [x] `default/qwen3-5-4b` (fileset `Qwen/Qwen3.5-4B`, pinned revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`)
- Created 2026-11-18 ~15:13 MST. Before this step the workbox had six entities and no Qwen 3.5/3.6 models.

## 4. Teacher pre-flight (done by hand once; the workflow owns this deploy)

This step only proved the 35B FP8 checkpoint loads on the workbox. The workflow's `deploy_teacher` node creates the same deployment (`teacher.model`, `teacher.engine`, `teacher.gpu`) and `undeploy_teacher` removes it. Do not repeat these commands for a workflow run.

```bash
nemo inference deployment-configs create qwen36-teacher-config --workspace default \
  --engine vllm \
  --model-spec '{"model_namespace":"default","model_name":"qwen3-6-35b-a3b-fp8"}' \
  --executor-config '{"gpu":1}'
nemo inference deployments create qwen36-teacher --workspace default --config qwen36-teacher-config
nemo wait inference deployment qwen36-teacher --workspace default
```

- [x] Config `qwen36-teacher-config` (vLLM, gpu 2) and deployment `qwen36-teacher` created 2026-11-18 ~15:13 MST, status `CREATED`. That config requests 2 GPUs. Recreate it with `gpu: 1` before the single-GPU run. The 35B FP8 checkpoint fits on one 48GB GPU.
- [x] Deployment reaches `READY` (22:26:41 UTC, `qwen36-teacher-server` ready after puller SUCCEEDED at 22:21:56).
  - 22:18 UTC: `qwen36-teacher` is `PENDING`, waiting on prerequisite `qwen36-teacher-puller`.
  - `qwen36-teacher-puller` is `STARTING`: k8s job `dep-default-qwen36-teacher-puller-10c73f59` has 1 active pod since 22:13:41. Its weight download is the wait. Status fields only change at job start and exit, so no progress shows in the API while it downloads.
  - Pod log (`weight-puller`, reported by user 22:19 UTC): `Fetching 56 files: 0%` then `Still waiting to acquire lock on /model-store/.cache/huggingface/.gitignore.lock`. The download is blocked by a filesystem lock in the shared model store. Nothing is transferring.
  - 22:21 UTC, pod `download/` listing (user): per-shard `.lock` and `.metadata` files for layers 0 to 39, plus one in-flight blob `iG3qE-...incomplete`. The puller got past the `.gitignore.lock` wait and is fetching. Presence of `.lock` files alone does not mean they are held. Progress is confirmed only if the `.incomplete` size grows.
  - 22:21:56 UTC: `qwen36-teacher-puller` SUCCEEDED (exit 0). Weights are on `/model-store`.
  - 22:22 UTC: `qwen36-teacher-server` STARTING, pod phase Running.
  - Earlier note on a likely stale holder is superseded by this listing unless the log stalls again.
  - Likely holder (superseded): another process or stale lock on `/model-store`, possibly left by an earlier job. huggingface_hub's filelock waits without a timeout by default, so this will not clear on its own.
  - No progress telemetry is exposed on the API. Pod logs live on the workbox cluster. The local kubectl context (`nv-prd-nemo.teleport.sh-nemo-testing-red`) does not reach it.
- [x] `nemo chat -m default/qwen3-6-35b-a3b-fp8 'Hello!' --provider qwen36-teacher --max-tokens 50` returns text.

Leave the registered model entity. Delete the pre-flight `default/qwen36-teacher` deployment before creating `qwen2b` (it was created at `gpu: 2`).

## 5. Package the sample agent image

The workbox only has `k8s` agent deployments enabled, and that mode needs an image. There is no shipped agent runtime image. Build one from the sample agent config, on a host with a Docker daemon, and push it to a registry the workbox nodes can pull without an image pull secret (the k8s backend does not use pull secrets).

From a source checkout, packaging cannot resolve `nemo-helix` from an index. Build the wheel first:

```bash
cd "$NEMO_PLATFORM"
uv build --package nemo-helix --wheel --out-dir dist
export NEMO_AGENTS_WHEEL=LATEST
```

```bash
nemo agents package \
  --agent plugins/nemo-agents/examples/nemo-agent-config/email-security-triage/agent.yaml \
  --tag email-security-triage:1.0.0 \
  --publish \
  --registry "$AGENT_REGISTRY"
```

- [x] Image built on the workbox: `nvcr.io/0921617854601259/nemo-platform-dev/email-security-triage:1.0.0` (`eca5633332ca`). `$AGENT_IMAGE` is that reference. `$AGENT_REGISTRY` is `nvcr.io/0921617854601259/nemo-platform-dev`.
- No Helm `default_image`. Pass `$AGENT_IMAGE` on the deploy in step 6. Image order is `--image`, then the agent's `reference.container_image`, then `agents.deployments.default_image`.

## 6. Register the sample agent (the workflow owns its deployments)

Registering the agent and giving the workflow the image is all that happens outside it. The deploy command below was a one-time pre-flight that proved the image runs. The workflow deploys and removes a clone of this agent for every arm.

**Result 22:30 UTC: BLOCKED at deployment.**
- Workspace `sample-8727ff98` created. Agent `email-security-triage` (`nemo-agents-spec-v1`) registered there.
- Stream ended with `kind: error`, status 400: "No deployment mode can run the sample agent on this platform. Turn on subprocess deployments ('agents.subprocess_enabled: true'), or configure a container mode with 'agents.deployments.default_image'."
- Deployment modes on the workbox: `subprocess` disabled, `docker` disabled, `openshell` disabled, `k8s` enabled but `requires_image: true` with no default image configured.
- The sample-agent endpoint registers the agent and then creates a deployment with no `image` field, so it can only use `agents.deployments.default_image`. That is why this call stopped. Deploy takes the image directly and does not need the Helm value.
- `$SAMPLE_WS` is `sample-8727ff98`. The agent is already registered there. Confirm its stored config points `models.default.model` at `default/qwen3-6-35b-a3b-fp8` before deploying.

```bash
export SAMPLE_WS=sample-8727ff98
export AGENT_IMAGE=nvcr.io/0921617854601259/nemo-platform-dev/email-security-triage:1.0.0

nemo agents deploy \
  --agent email-security-triage \
  --workspace "$SAMPLE_WS" \
  --mode k8s \
  --image "$AGENT_IMAGE"
```

- [x] Deploy reaches `running`. `email-security-triage-7d72be56` was `running` in 14s at `http://dep-sample-8727ff98-email-security-triage-7d72be56-618222da.default.svc.cluster.local:8000`.
- [x] Agent config is `nemo-agents-spec-v1`. `models.default.model` is `qwen3-6-35b-a3b-fp8`, and `base_url` is the inference gateway path for that model in `default`.

Leave the registered agent. Delete deployment `email-security-triage-7d72be56` before creating `qwen2b`. The workflow takes the image, mode, and entrypoint flag from `agent_deployment`, and the dataset eval calls the workflow's deployment, not this one.

## 7. Distill the 2B student (first full workflow)

This is entry point 1: the workflow generates the trajectories. The create body (full JSON in [Appendix A](#appendix-a-full-create-request-body)) names `gpu_budget`, the teacher model, the agent deployment, the deployment and candidate settings, and both example evals by fileset reference. `tasks` is the task-driven config file, submitted as an agent eval against the cloned agent. `dataset` is the dataset-driven config file, submitted as `evaluate/jobs` against that clone's deployment chat endpoint. Upload the two config files and `dataset.jsonl` once before creating the workflow (they are eval inputs, not workflow steps). The workflow never carries the tasks: the request and the stored entity hold only the file references. `prepare` does not start until both original-arm evals have succeeded, and it receives their job filesets.

The `qwen2b` workflow already on the workbox (now `cancelled`) was created from the old graph and the service has no delete route for workflows, so change `name` in the body (for example `qwen2b-r2`) rather than reusing it. Create this one only after the platform image and `nhx-customizer-tasks` image include this change.

```bash
nemo files filesets create esec-eval-data --workspace "$SAMPLE_WS"
nemo files upload \
  plugins/nemo-agents/examples/nemo-agent-config/email-security-triage/dataset.jsonl \
  esec-eval-data --workspace "$SAMPLE_WS"

nemo files filesets create esec-eval-config --workspace "$SAMPLE_WS"
EX=plugins/nemo-agents/examples/nemo-agent-config/email-security-triage
mkdir -p /tmp/eval-configs
for f in eval-config.task-driven.json eval-config.dataset-driven.yml; do
  # The examples judge with a 120B NIM model that is not on the workbox; judge with the 35B on vLLM instead.
  # NIM-only `extra_body.nvext.max_thinking_tokens` is ignored by vLLM, so Qwen3.6 reasons until max_tokens and
  # returns no content. vLLM's equivalent is the `thinking_token_budget` request parameter; keep max_tokens above it.
  sed 's#default/nvidia-nemotron-3-super-120b-a12b#default/qwen3-6-35b-a3b-fp8#g' "$EX/$f" > "/tmp/eval-configs/$f"
  python3 - "/tmp/eval-configs/$f" <<'PY'
import re, sys
p = sys.argv[1]
t = open(p).read()
t = re.sub(r'"nvext": \{\s*"max_thinking_tokens": 256\s*\}', '"thinking_token_budget": 256', t)
t = re.sub(r"( +)nvext:\n +max_thinking_tokens: 256", r"\1thinking_token_budget: 256", t)
t = t.replace('"max_tokens": 1024', '"max_tokens": 2048').replace("max_tokens: 1024", "max_tokens: 2048")
open(p, "w").write(t)
PY
  nemo files upload "/tmp/eval-configs/$f" esec-eval-config --workspace "$SAMPLE_WS"
done

# Save the request body from Appendix A as /tmp/qwen2b.json (the whole JSON object).

curl -s -X POST "$NHX_BASE_URL/apis/distillation/v2/workspaces/$SAMPLE_WS/workflows" \
  -H 'content-type: application/json' \
  -d @/tmp/qwen2b.json
```

First create at 22:45 UTC failed: `prepare` job `qwen2b-t1` could not pull `nhx-customizer-tasks:bbf3f21dbd`. After the image rebuild, the workflow entity and that job were deleted (22:54 UTC) and the workflow was created again.

Live attempts of this graph (the 22:45 and 22:55 UTC notes above are from the earlier graph):

| Workflow | When (MST) | Outcome |
| -------- | ---------- | ------- |
| `qwen2b-r2` | 17:15 | Teacher and agent deployed. Both original evals failed at submit with 422: judge model `default/nvidia-nemotron-3-super-120b-a12b` not found. Cleanup ran, GPU freed. Fix: judge model (see Judge model). |
| `qwen2b-r3` | 17:38 | Cancelled by hand while the teacher loaded, to free the GPU before the judge deploy. Cancel cleaned up, no deployments left. |
| `qwen2b-r4` | 17:46 | Teacher on the second GPU and agent deployment succeeded. Both original evals failed: vLLM answered every agent request with 400 `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set`. The task eval ended with 8 of 8 trials errored. The dataset eval saw it as 502 Bad Gateway from the agent deployment. Cleanup ran, nothing left but the judge. |
| `qwen2b-r5` | after image rebuild | Same 400 as r4 although the teacher carried the tool flags. Cause (consistent with r6, where the 400 vanished after the swap): the hand-deployed judge `judge-qwen36` had no flags and shares the gateway route for the 35B model with the teacher. Judge recreated as `judge-qwen36-tools` with the same flags plus `--reasoning-parser qwen3`. |
| `qwen2b-r6` | 21:19 | Tool-choice 400 gone: agent requests complete and the judge scores. `eval_original[dataset]` failed: judge returned `finish_reason: length`, 1024 reasoning tokens and no content, so `llm-judge` raised on row 2 (`EvaluationError`), which tore down the pipeline. The "failed after 6 attempts" agent errors in that log are `ResilienceCancelledError` from the teardown, not agent faults. The tasks eval logged the same judge warning per trial. Fix: judge configs use `extra_body.thinking_token_budget: 256` and `max_tokens: 2048` (the NIM `nvext.max_thinking_tokens` hint is ignored by vLLM). Probed live on the judge route (3 samples each, `scratch/budget_probe.py`): budget 256 gave 255 reasoning tokens and `{"accuracy": 1.0}`; adding `reasoning_effort: low` changed nothing (this template ignores it); `enable_thinking: false` gave 0 reasoning tokens and the same score. Transport checked: `InferenceParams` is `extra="allow"`, so `inference.extra_body` reaches the request dict (the r6 job log shows it there); `make_inference_request` spreads that dict into `AsyncOpenAI.chat.completions.create(**request_body)`, where `extra_body` is the OpenAI client's own parameter and is merged into the top level of the JSON body. Confirmed on the wire with the same client call (`scratch/sdk_probe.py`): body keys `max_tokens, messages, model, response_format, thinking_token_budget`, no literal `extra_body` key, 255 reasoning tokens, content returned. Still to see in a real eval run: the judge actually scoring through the job pipeline. |
| `qwen2b-r7` | 21:35 | First clean pass through the evals with the patched judge configs (`thinking_token_budget: 256`, `max_tokens: 2048`). Original arm: agent eval 2 tasks, 0 errors, every metric scored 2 of 2; dataset eval 28 rows, 0 NaN, `llm-judge.accuracy` mean 0.756. `prepare` succeeded (2 examples). Base arm: agent eval 8 trials, 0 errors; dataset eval 28 rows, 0 NaN, `llm-judge.accuracy` mean 0.756 (per-row scores differ from the original arm, so the equal mean is a coincidence); `string-check` mean 0.0. Undeploys ran. `train[lora]` failed at submit with 422: the body did not match `AutomodelJobInput` (dataset must be `{"training": ref}`, `training.type` is `training_type`, `parallelism` is a sibling of `training`, no `peft` key). Fixed in `CustomizationAdapter.start`; needs a platform image rebuild before the next run. The stage `metrics.score` is still null for the eval stages, so the results projection does not read these scores yet. |
| `qwen2b-r8` | 22:10 | Training body accepted. `train[lora]` succeeded (2 training rows, `global_batch_size: 2`, adapter fileset `qwen2b-r8-t19`), and `deploy[lora]` and `deploy_agent[lora]` reported success. Both distilled evals then failed: the agent eval errored on 8 of 8 trials and the dataset eval errored after retries. vLLM logged `load_lora_adapter failed (status=500)`: the adapter's target modules are not in the set vLLM expects for this model. The default LoRA (no module list) adapted 194 modules, among them `mtp.layers.*`, `eh_proj`, and the linear-attention `in_proj_b` and `out_proj`; vLLM's full error (captured later from the sidecar, now that its log cap is 2000): expected target modules in {qkv, lm_head, up_proj, in_proj_qkv, linear_fc2, o_proj, v_proj, down_proj, embed_tokens, out_proj, linear_fc1, gate_proj, q_proj, in_proj_b, in_proj_a, proj, conv1d, k_proj, in_proj_z} but received ['mtp.layers.0.eh_proj', 'mtp.layers.0.eh_proj']. So only the MTP head's `eh_proj` is rejected; `in_proj_b` and `out_proj` are fine (an earlier guess in this row was wrong). Without a module list the compiler defaults `target_modules` to `["*proj"]`, which also adapts the `mtp.layers.*` blocks, and the deployed model has no MTP layers. Deployment and agent were undeployed. Fix to try (r9): `lora.exclude_modules: ["*mtp*"]` on the candidate. After training, `adapter_config.json` should list no `mtp.*` modules. |
| `qwen2b-r9` | 23:20 | Candidate `lora.exclude_modules: ["*mtp*"]` plus `global_batch_size: 2`. Everything through `deploy_agent[lora]` succeeded, and the adapter (`qwen2b-r9-t19`'s fileset) has no `mtp.*` modules. It does carry 98 `model.visual.*` modules and `lm_head`, because an explicit exclude list replaces the customizer's default exclusions. Both distilled evals still failed: the agent eval errored on 8 of 8 trials (`No usable scores across 8 trials`). The per-trial errors were not retrievable (files API returned 502 for the results artifact) and the deployment logs were gone after cleanup, so whether the adapter loaded this time is unconfirmed. Cleanup left nothing behind. |

What `qwen2b-r4` already proved: fileset-driven eval configs are read and submitted, the teacher lands on the free GPU while the judge holds the other, the agent deployment answers through the gateway route, and `always_run` cleanup leaves nothing behind after a failed run.

**The miss behind `qwen2b-r4`:** the workflow deployed vLLM models with no way to pass serve args. Tool calling on vLLM is not a typed field; it is `executor_config.additional_args` (`--enable-auto-tool-choice --tool-call-parser <parser>`). `tool_call_config` is NIM-only and the plugin copied that rule, so `teacher` and `deployment` could not carry tool flags at all. An agent that calls tools cannot be evaluated on such a deployment, whichever arm it is. The plugin now takes `additional_args` on `teacher` and `deployment` and writes it into every deployment config it creates (teacher, baseline, candidates); Appendix A sets it. Unit tests cover the plumbing, so the miss was a gap in the design, not something a unit test could see: nothing in the fakes starts vLLM.

- [ ] Rebuilt platform image includes `additional_args`.
- [ ] The teacher deployment starts with the flags and the `tasks` eval gets scored trials instead of tool-choice 400s. `qwen3_coder` and `--reasoning-parser qwen3` are not yet verified against these checkpoints. If a deployment fails to start naming an unknown parser, or evals still get 400s, change the list in the request body.
- [ ] The same flags are on `deploy_base` and `deploy[lora]` (check the deployment configs the workflow created).

22:55 UTC snapshot (earlier graph): workflow `running`. `prepare` succeeded. `experiment`, `deploy_base`, and `train[lora]` are `running`. The eval nodes are still `pending`.

Poll: `curl -s "$NHX_BASE_URL/apis/distillation/v2/workspaces/$SAMPLE_WS/workflows/qwen2b"`

Check while it runs:

With `gpu_budget: 1` the workflow runs these in order, one at a time. Only one model deployment exists at any moment.

1. [ ] `experiment` and `deploy_teacher` start with no dependencies. The teacher is 1 GPU of `default/qwen3-6-35b-a3b-fp8`. The experiment has one Evaluation per arm: `original`, `base`, `lora`.
2. [ ] `deploy_agent_original` starts once the teacher is ready. It clones `source_agent`, points the clone at the teacher, and deploys it with the supplied image and mode.
3. [ ] `eval_original[tasks]` and `eval_original[dataset]` run against that deployment, only after it is running.
4. [ ] `undeploy_agent_original` and `undeploy_teacher` run after the original evals. The teacher and the original agent deployment are gone before anything below starts.
5. [ ] `prepare` consumes the original evals' output filesets (`trajectory_filesets`) and produces `dataset` under `nodes/prepare/` (`train.jsonl`, `validation.jsonl`, `output.json`).
6. [ ] Baseline: `deploy_base` (1-GPU LoRA deployment of the base student), `deploy_agent_base`, `eval_base[tasks]` and `eval_base[dataset]` against that deployment (the same evals that ran against the teacher), then `undeploy_agent_base` and `undeploy_base`.
7. [ ] Candidate: `train[lora]` starts only after the baseline is undeployed. It consumes `${nodes.prepare.outputs.dataset}` and requests 1 GPU. Then `deploy[lora]` (gateway `default/qwen3-5-2b&adapters/<ws>/<adapter>`), `deploy_agent[lora]`, `eval_distilled_tasks[lora]` and `eval_distilled_dataset[lora]`, then `undeploy_agent[lora]` and `undeploy[lora]`.

End state:

- [ ] Workflow `status` is `completed`.
- [ ] `results` has `original`, `base`, `distilled`. Scores are null (expected for stub rows).
- [ ] `deploy_base` and `deploy[lora]` deployments are deleted.
- [ ] Temporary agents and their deployments (`*-tN` names) are deleted, with their `-ethos` filesets. The source agent entity is still registered.
- [ ] The workflow's teacher deployment is gone. The single GPU is free of the teacher before any student deployment.

## 8. Distill the 4B student

Run only after step 7 reaches `completed`. Same entry point and body as step 7. Only `name` and `student_model` change, and `jq` makes that edit so nothing else drifts.

```bash
jq '.name = "qwen4b" | .inputs.student_model = "default/qwen3-5-4b"' \
  /tmp/qwen2b.json > /tmp/qwen4b.json

curl -s -X POST "$NHX_BASE_URL/apis/distillation/v2/workspaces/$SAMPLE_WS/workflows" \
  -H 'content-type: application/json' \
  -d @/tmp/qwen4b.json
```

- [ ] Same checks as step 7, with `qwen4b` and `default/qwen3-5-4b`.
- [ ] Ends `completed`, deployments and temporary agents removed.

## 9. Cancel removes the deployments

Start a third workflow from the same body (only `name` changes) and cancel it once `deploy_teacher` exists.

```bash
jq '.name = "qwencancel"' /tmp/qwen2b.json > /tmp/qwencancel.json

curl -s -X POST "$NHX_BASE_URL/apis/distillation/v2/workspaces/$SAMPLE_WS/workflows" \
  -H 'content-type: application/json' \
  -d @/tmp/qwencancel.json
curl -s -X POST "$NHX_BASE_URL/apis/distillation/v2/workspaces/$SAMPLE_WS/workflows/qwencancel/cancel" -o /dev/null -w '%{http_code}\n'
```

- [ ] Cancel returns 202.
- [ ] Workflow ends `cancelled`.
- [ ] `eval_distilled_tasks[lora]` and `eval_distilled_dataset[lora]` end `cancelled`.
- [ ] Undeploy nodes ran (`always_run`), and no deployment created by this workflow is still listed.

## 10. Retry rejects non-failed nodes

- [ ] `POST .../workflows/qwen2b/nodes/prepare/retry` on the completed workflow returns 409.

---

## Known limits for this pass

- The convert step records the eval output filesets and still writes synthetic training rows. Scores stay null. No quality claim comes from this run.
- The dataset-driven eval calls the workflow-owned agent deployment. Each arm deploys its clone before its evals and deletes that deployment afterwards.
- One 48GB GPU, `gpu_budget: 1`. Every stage of a run is a workflow node. The teacher, the baseline, and each candidate run one after another, and each stage's undeploy finishes before the next stage deploys or trains. The pre-flight `default/qwen36-teacher` (created with `gpu: 2`) and the pre-flight agent deployment are not part of any run. Delete them and do not recreate them.
- Parallel lanes (`gpu_budget` of at least twice the GPUs per job) and several candidates are covered by unit tests only. The live runs use one LoRA candidate. Candidates now also carry training hyperparameters (see `candidates` below), so a sweep is several candidates with different `hyperparameters`; none of that has run live.
- The training body (`CustomizationAdapter.start`) is built to `AutomodelJobInput`: `dataset: {training: <ref>}`, `training.training_type`, `training.finetuning_type`, `parallelism` beside `training`, and the candidate's `max_seq_length`, `lora`, `schedule`, `batch` and `optimizer`. r7 got a 422 on the earlier shape (dataset as a string, `training.type`, `peft`). The current output is checked against the live OpenAPI schema (`scratch/validate_specs.py`), but a training job has not run live yet.
- A deployment reports READY before its LoRA adapter is loaded. The adapter loads afterwards and a failure shows only in the deployment's own logs, so the workflow cannot see it until an eval fails. Worth a check in `deploy[...]` that the adapter model answers.
- The stub convert writes only 2 training rows, and the customizer defaults to `global_batch_size: 8` and a 0.1 validation split. Set small batch sizes on the candidate for live runs, or expect the job to fail on batch size.
- Judge: a hand-deployed 35B on its own GPU judges every arm. It is outside the workflow by design, so the workflow does not count its GPU in `gpu_budget`. Scores come from a same-model judge and say nothing about quality.
- vLLM tool calling is configured through `additional_args`, not `tool_call_config` (NIM only). Every vLLM deployment the workflow creates needs it if the agent calls tools.
- Live results prove plumbing: ordering, naming, deployment lifecycle, and cleanup.

---

## Appendix A: full create request body

`POST /apis/distillation/v2/workspaces/{workspace}/workflows`. This is the exact body step 7 sends. The eval definitions are not in it: each `evals[]` entry points at a file in a fileset, and the eval node reads that file when it starts. Steps 8 and 9 reuse it and change only the fields listed after it. Validated against the plugin's `DistillInputs`: it renders to 21 nodes, `parallel_jobs` 1. The referenced config files must exist before the workflow starts an eval; a missing or malformed file fails that eval node with a message naming the file.

Eval config files used here, both from `plugins/nemo-agents/examples/nemo-agent-config/email-security-triage/`: `eval-config.task-driven.json` (8 tasks, each with its own metrics and views, `max_concurrent_tasks: 1`) and `eval-config.dataset-driven.yml` (`dataset`, `prompt_template`, two metrics). Their shape is the shape the evals service takes, so what you upload is what an eval job would be submitted with.

Top-level `inputs` fields:

| Field | Required | Meaning |
| ----- | -------- | ------- |
| `source_agent` | yes | `workspace/name` of the registered agent. The workflow clones it per arm. |
| `student_model` | yes | `namespace/name` of the registered model to distill into. |
| `teacher` | one of `teacher` or `trajectory_filesets` | Entry point 1. Model the workflow deploys to produce trajectories. `engine` is `vllm` or `nim`, `gpu` defaults to 1, and `additional_args` is appended to the serve command. |
| `trajectory_filesets` | one of `teacher` or `trajectory_filesets` | Entry point 2. Existing evaluation output filesets (`workspace/name`). Skips the teacher stage, deploy, and original evals. |
| `gpu_budget` | no, default 1 | GPUs usable at once. Candidate jobs run `gpu_budget // gpus-per-job` at a time. 1 means fully sequential. Must be at least the teacher's and every job's GPUs. |
| `agent_deployment` | yes | `mode` (`k8s` default, `docker`, `openshell`, `subprocess`), `image` (required except for `subprocess`), `use_image_entrypoint` (default false). |
| `deployment` | no | How student models deploy (baseline and candidates): `engine`, `gpu`, `additional_args`, and `tool_call_config` (`nim` only). |
| `candidates` | no | Training candidates, the sweep axis. Each has `name`, `finetuning_type` (`lora` or `all_weights`), `gpu`, and optional `hyperparameters`. Omitted means one LoRA candidate named `lora` with no hyperparameters set. Each candidate trains on the same generated dataset, then deploys and evaluates on its own. |
| `candidates[].hyperparameters` | no | Training settings in the shape of the customizer's automodel job spec. Every field is optional, and one left out is not sent, so the customizer's default applies. `max_seq_length` (default 2048). `lora`: `rank` (16), `alpha` (32), `dropout` (0), `target_modules` or `exclude_modules` (one of the two, as the customizer requires; with neither set the adapter covers every linear layer); only with `finetuning_type: lora`. `schedule`: `epochs` (1), `max_steps`, `validation_split` (0.1), `checkpoint_selection` (`best`, `last` or `both`; default `best`), `seed`. `batch`: `global_batch_size` (8), `micro_batch_size` (1), `sequence_packing` (false). `optimizer`: `learning_rate` (5e-5), `min_learning_rate`, `weight_decay` (0.01), `warmup_steps` (0), `lr_decay_style` (`cosine`, `linear` or `constant`; default `cosine`). Unknown names and out-of-range values are rejected when the workflow is created. |
| `evals` | yes, at least one | Each has `name`, `kind`, `config`, and for `dataset` an optional `dataset`. `config` is one file, `workspace/fileset#path`, JSON or YAML. `agent` reads a task-driven agent-eval config (a non-empty `tasks` list, optional `max_concurrent_tasks`). `dataset` reads a dataset-driven config (`prompt_template`, `metrics`, and a `dataset` reference). `dataset` on the eval replaces the config's `dataset`, which is a placeholder in the shipped example. The workflow fills in the per-arm target and publication. Workspace may be omitted to mean the workflow's. |

`additional_args` matters for agent evals. The sample agent calls tools, and vLLM rejects tool choice with a 400 (`"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser`) unless the deployment sets those flags. Every deployment the workflow creates (teacher, baseline, candidates) gets the same list. `qwen3_coder` is the parser the repo's deploy docs use for Qwen3-family tool calls, and `--reasoning-parser qwen3` keeps the model's thinking out of the answer the evals read. Neither is verified against these checkpoints yet. A run that gets 400s on tool choice, or a startup failure naming an unknown parser, means these args are wrong for the model.

Trajectory entry point: delete `teacher` and add `"trajectory_filesets": ["sample-8727ff98/<eval-output-fileset>"]`. Every other field stays as it is.

```json
{
  "name": "qwen2b",
  "inputs": {
    "source_agent": "sample-8727ff98/email-security-triage",
    "student_model": "default/qwen3-5-2b",
    "gpu_budget": 1,
    "teacher": {
      "model": "default/qwen3-6-35b-a3b-fp8",
      "engine": "vllm",
      "gpu": 1,
      "additional_args": [
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
        "--reasoning-parser",
        "qwen3"
      ]
    },
    "agent_deployment": {
      "mode": "k8s",
      "image": "nvcr.io/0921617854601259/nemo-platform-dev/email-security-triage:1.0.0",
      "use_image_entrypoint": false
    },
    "deployment": {
      "engine": "vllm",
      "gpu": 1,
      "additional_args": [
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
        "--reasoning-parser",
        "qwen3"
      ]
    },
    "candidates": [
      {
        "name": "lora",
        "finetuning_type": "lora",
        "gpu": 1,
        "hyperparameters": {
          "max_seq_length": 2048,
          "lora": { "rank": 16, "alpha": 32 },
          "schedule": { "epochs": 1 },
          "batch": { "global_batch_size": 2, "micro_batch_size": 1 },
          "optimizer": { "learning_rate": 0.00005 }
        }
      }
    ],
    "evals": [
      {
        "name": "tasks",
        "kind": "agent",
        "config": "sample-8727ff98/esec-eval-config#eval-config.task-driven.json"
      },
      {
        "name": "dataset",
        "kind": "dataset",
        "config": "sample-8727ff98/esec-eval-config#eval-config.dataset-driven.yml",
        "dataset": "sample-8727ff98/esec-eval-data#dataset.jsonl"
      }
    ]
  }
}
```

Changes for the other runs:

| Run | `name` | `inputs.student_model` |
| --- | ------ | ---------------------- |
| Step 7 | `qwen2b` | `default/qwen3-5-2b` |
| Step 8 | `qwen4b` | `default/qwen3-5-4b` |
| Step 9 | `qwencancel` | `default/qwen3-5-2b` |

## Run log

| Step | Result | Notes |
| ---- | ------ | ----- |
| 2 render | pass | 21 nodes, `gpu_hours` 4, `parallel_jobs` 1. 422 on no entry point and on a config that is not a file reference. |
| judge deploy | pass | `default/judge-qwen36` `READY` 17:45:49 MST. |
| 7 `qwen2b-r2` | fail | Judge model missing (422). |
| 7 `qwen2b-r4` | fail | vLLM tool choice 400 (no `additional_args`). Fixed in code, awaiting image rebuild and `qwen2b-r5`. |

## SWE-smith local stress data (2026-11-19)

- Local run: teacher `default/qwen3-6-35b-a3b-fp8` (Qwen3.6 35B-A3B), student `default/qwen3-5-4b`. Cluster run: GLM -> `nemotron-3-5-lightning-30b-a3b`; backup teacher Ultra.
- `max_seq_length` must be set to 32768. The chat path never truncates (`ChatDataset` default `do_not_truncate`, customizer does not override), and packing drops over-length rows. The converter filters to <=32k rendered tokens so neither applies.
- **Qwen3.5 template rejects string tool-call `arguments`** (`Can only get item pairs from a mapping`). Automodel `ChatDataset._normalize_messages` json.dumps'd dicts. Fixed by `docker/automodel/patches/automodel-chat-dataset-tool-call-arguments.patch` (applied in `Dockerfile.nhx-automodel-base`). Needs an automodel base + training image rebuild before the 4B smoke.
- SWE-smith `tool` split, resolved claude-3-7: 6,906 rows = 2,322 native `tool_calls` + 4,584 text-format (inline `<function=...>`, `OBSERVATION:` user turns). Converter keeps native only. Tools used: `bash`, `str_replace_editor` (view/create/str_replace), `submit`.
- `scratch/swe/convert_smith.py` -> `scratch/swe/out/c37-native-32k/` (1,656 train / 85 val, 31.9M train tokens, median 18.6k, max 32.8k; split by instance_id). Slices: `smoke-3`, `slice-200`.
- Not done: text-format -> native conversion (would add ~4.5k rows), 3-row smoke on the 4B (blocked on image rebuild).

## SWE image and taskset-run steps (2026-11-19)

Teacher-stage agent for the SWE demo runs as a Kubernetes taskset run: one Job and one OpenSandbox sandbox per task. OpenSandbox is running on the workbox and the target cluster (confirmed by the user). The demo itself uses the `trajectory_filesets` entry point, so this only gets the live agent running. A model-calling Harbor agent (`openhands` is closest to the SWE-smith tools: bash, str_replace_editor, finish) reaches the teacher through the gateway with `source.model = {"ref": "default/qwen3-6-35b-a3b-fp8"}`.

Image work, all on the workbox side (the laptop has no workbox kube context, so the user runs these):
- [ ] Rebuild the automodel base and training images. The base now applies `docker/automodel/patches/automodel-chat-dataset-tool-call-arguments.patch` in the `automodel-clone` stage. (In progress per the user.)
- [ ] Rebuild `nhx-customizer-tasks` only if the converter moves into the customizer (not yet).
- [x] Harbor task images build in-cluster through Builder (`nemo evals tasksets build-images`); no manual image push is needed. Working after the fixes in "Resolution timeline" (hello suite 40 s, DeepSWE task 56 s, `swesmith` tasks 3 to 5.5 min when 8 build at once). Registries a task image may copy from are limited by `platformConfig.builder.import_registries`.

Steps (run from `nemo-platform` with `NHX_BASE_URL=http://10.2.183.187:8080`, workspace `sample-8727ff98`):
1. `flox activate -- uv run nemo evals runs submit --workspace sample-8727ff98 --harbor-dir <suite> --list-tasks`
2. `... runs submit --harbor-dir <suite> --tasks 0 --agent oracle` (no model; checks publish, image build, Job, sandbox, results). Copy of the docs hello-world suite is at `scratch/swe/hello-suite`.
3. `nemo evals runs status <run-id> --workspace sample-8727ff98 --watch`, then `runs results`.
4. Model-calling run: same submit with `--spec-file` (`target.source = {name: openhands, model: {ref: ...}}`, `agent_kwargs.version: 1.6.0`). With platform auth off, no `env_secrets`. Check the sandbox profile's `agent_setup_egress` allows the install hosts.

Result of step 2, first attempt (17:22-17:23 UTC): **blocked at image build.**
- Published task `sample-8727ff98/hello-suite-hello-world#4c933385...` and taskset `sample-8727ff98/hello-suite-efa2a1cc#d4326921...`.
- Four build attempts failed (image-builds ids 1-4, jobs `task-image-build-1a0c8930d7f9da06-1..4`), each with `Builder build harbor-env-aff02de19f92 kept being submitted by other requests; try again`. `build-images --wait` did not recover it.
- The job log shows `adopt_or_build` getting a `ConflictError` on `submit` of `harbor-env-aff02de19f92-1` three times, then raising. But Builder has no such image: `GET container-images/harbor-env-aff02de19f92-{1,2,3}` return 404 and the workspace's `container-images` list is empty. So the 409 is not a race with another request; the message is misleading.
- The 409 reason is swallowed (logged at info, not recorded). Builder `POST /builds` returns 409 for `BackendRejectedError` (backend cannot build the request, or cannot build at all, or is misconfigured), `BuildConflict` (name/revision submitted with a different request, or the job name is taken) and `MissingSecrets`.
- **Resolved** (see "Resolution timeline" below): replaying the Builder `POST` returned `409 {"detail":"builder.registry is not configured; there is nowhere to publish the result"}`. The push-secret hypothesis in the earlier draft of this section was wrong.
- No run was created (`GET agent-evaluate/runs` is empty).

## Builder build and deployment steps (2026-11-19)

Source: `plugins/nemo-builder/README.md`, `docs/set-up/config-reference.mdx` (`builder:`), `docs/set-up/helm/harbor-sandbox-profiles.mdx`. Nothing below was checked against the workbox or the target cluster from the laptop (no kube context for either; the user runs the cluster-side steps). The Harbor task image builds that taskset runs need go through this Builder.

### What the deployment must have (operator side)

- [ ] Platform image includes the builder (every platform image does). Chart value `builder.enabled: true`.
- [ ] `platformConfig.builder.registry` set, a host only (`http://host` for plain HTTP). **Unset, every submit returns 409.** Also `repository_prefix` (a path dedicated to builds) and `default_platform` (`linux/amd64` unless the nodes differ).
- [ ] Images the cluster can pull, all at the same tag: `nhx-api`, `nhx-builder-tasks` (step image) and `nhx-kaniko` (sandbox). Build with `docker buildx bake -f docker-bake.hcl nhx-api-docker nhx-builder-tasks-docker nhx-kaniko-docker --set '*.platform=linux/amd64'`, or use the CI images from `ghcr.io/nvidia-nemo/nemo-helix` (tagged by commit, amd64 only). Point `platformConfig.platform.image_registry` and `image_tag` at them; the steps and sandboxes take `nhx-builder-tasks` and `nhx-kaniko` from there. A rebuilt image needs a new tag (`BAKE_TAG=<tag>`) on minikube.
- [ ] Build namespace `<release-namespace>-builds` (baseline Pod Security) with the three step ServiceAccounts, a ReadWriteMany work volume (`builder.workVolume.storageClass`, must bind immediately), a LimitRange and a ResourceQuota. Image pull secrets are not copied there by the chart.
- [ ] Three Jobs execution profiles, `build-fetch`, `build-control`, `build-push` (`cpu` on `kubernetes_job`). They must name the same namespace, fetch and push the same `storage.pvc_name`, and use `service_account_name` `nhx-build-fetch` / `nhx-build-control` / `nhx-build-push`. A mismatch is a 409. `builder.maxConcurrentBuilds` sets each profile's `max_active_jobs`.
- [ ] Sandboxes resolve names with `8.8.8.8` and `1.1.1.1`. If a pod cannot reach them (`kubectl run dnscheck --rm -i --restart=Never --image=docker.io/library/busybox:1.36 -- nslookup pypi.org 8.8.8.8`), set `platformConfig.builder.sandbox.dns_nameservers`.
- [ ] A registry credential that can push under `<repository_prefix>/<workspace>/`, plus a signing key (`openssl ecparam -name prime256v1 -genkey -noout | openssl pkcs8 -topk8 -nocrypt -out signing.key`).
- [ ] `builder-settle` controller running (the chart refuses a controller selection without it).
- [ ] Build namespace pods can pull the step images (`imagePullSecrets` in the build namespace if the registry is private).

### What each workspace needs (per workspace that submits builds)

The push step reads three platform secrets from the submitting workspace. **A workspace without them gets a 409 naming what is missing.** The workbox workspace is `sample-8727ff98`.

- [ ] `nemo secrets create builder-registry-username --workspace sample-8727ff98 --from-file <file>`
- [ ] `nemo secrets create builder-registry-password --workspace sample-8727ff98 --from-file <file>`
- [ ] `nemo secrets create builder-signing-key --workspace sample-8727ff98 --from-file signing.key`

(All three are required unless the deployment set `registry_username_secret`, `registry_password_secret` or `signing_key_secret` to null.)

### Smoke test the Builder on its own (before any taskset image build)

Direct submit returns the real 409 `detail`, which the evals plugin hides. From a shell with `NHX_BASE_URL=http://10.2.183.187:8080`:

```bash
CTX=$(mktemp -d) && cat > "$CTX/Dockerfile" <<'EOT'
FROM docker.io/library/python:3.13-slim
RUN echo "hello from nemo-builder" > /hello.txt
EOT
nemo files filesets create hello-context --workspace sample-8727ff98
nemo files upload "$CTX/" hello-context --workspace sample-8727ff98
curl -s -X POST "$NHX_BASE_URL/apis/builder/v2/workspaces/sample-8727ff98/builds" \
  -H 'Content-Type: application/json' \
  -d '{"name":"hello","revision":1,"build_specs":[{"source":{"fileset":"hello-context"},"output":{"repository":"demo/hello","tag":"v1"}}]}'
curl -s "$NHX_BASE_URL/apis/builder/v2/workspaces/sample-8727ff98/container-images/hello-1" | jq '{status, digest, status_detail}'
```

- [ ] `201` with a job name and `pending` rows. A `409` body is the answer to the open question below.
- [ ] The image reaches `ready` with a digest in a minute or two (`kubectl -n <release-namespace>-builds get pods -w` shows fetch, build with its sandbox, push).
- [ ] Optional: crane `digest` of `<registry>/<prefix>/sample-8727ff98/demo/hello:v1` equals the recorded digest, and `cosign verify --key signing.pub --insecure-ignore-tlog=true <ref>@<digest>` passes.
- To build again, use `"revision": 2`. A failed row fails once its job ended; within five minutes of submit, the same request again creates the job.

### 409 reasons from `POST /builds` (what to check)

| 409 reason | Check |
| --- | --- |
| `registry` unset | `platformConfig.builder.registry` |
| Execution profiles missing, not on Kubernetes, or disagree | the three `build-*` profiles: namespace, PVC, service accounts |
| Push step secrets missing in the workspace, or the caller cannot read them | the three secrets above, in the submitting workspace |
| Image to copy is on a registry not in `import_registries` | `builder.import_registries` |
| Same `name` and `revision` submitted with a different request, or the job name is held by a job this request did not create (rows are then failed) | new `revision`; look for a leftover job named `builder-<name>-<revision>` |
| Backend rejects the request or is misconfigured (`BackendRejectedError`) | Builder config and `default_platform` |

### Resolution timeline: the oracle run's image build and sandbox start (2026-11-19)

Every layer below was a workbox configuration gap, found one at a time. Cluster-side fixes were applied by the user (no kube context for the workbox on the laptop). Reads from the laptop used the workbox REST API only.

| # | Symptom (where seen) | Cause | Fix (who) |
| --- | --- | --- | --- |
| 1 | `build-images`: `Builder build harbor-env-aff02de19f92 kept being submitted by other requests` on 4 attempts | `adopt_or_build` treats every 409 as a lost race and hides the detail. Replaying `POST /apis/builder/v2/workspaces/sample-8727ff98/builds` (throwaway name `probe-hello-env`, request refused, nothing started) returned `409 builder.registry is not configured; there is nowhere to publish the result` | Helm: `builder.enabled=true`, `builder.devRegistry.enabled=true`, `--set-file builder.devRegistry.password=<file>`. The chart then sets `platformConfig.builder.registry` to `http://nemo-helix-builder-registry.default.svc.cluster.local:5000` (user) |
| 1b | (same, before the first build could run) | Push step reads three platform secrets from the submitting workspace; the chart does not create them | `builder-registry-username` (`builder`), `builder-registry-password`, `builder-signing-key` in `sample-8727ff98` (user) |
| 2 | `build-images`: `main: not built: its build job ended (error) without publishing it`; Builder images `harbor-env-aff02de19f92-1` and `-2` `failed`, job `builder-harbor-env-aff02de19f92-2` has no logs | The `fetch` step pod in `default-builds` could not pull `nvcr.io/0921617854601259/nemo-platform-dev/nhx-builder-tasks:255fc001db`: `pull access denied`, `Unable to retrieve some image pull secrets (nvcrimagepullsecret)`. The chart does not copy the pull secret into the build namespace, and the API image only worked because it was already on the node | Copied `nvcrimagepullsecret` from `default` to `default-builds`; `nhx-builder-tasks` and `nhx-kaniko` at `255fc001db` present on the node (user) |
| 3 | After the build succeeded (`1 image, 40s`, digest `sha256:b2c3af8d...`): oracle run `run-51feace54d70730d` stuck, trial `starting`, sandbox `ImagePullBackOff` | The node's Docker daemon resolved the registry name with node DNS and tried HTTPS: `Get "https://nemo-helix-builder-registry.default.svc.cluster.local:5000/v2/": dial tcp: lookup ...: Temporary failure in name resolution`. Cancelled that run | On node `918b990-lcedt`: `/etc/hosts` maps the registry name to its ClusterIP (`10.212.113.186`); `insecure-registries` added to `daemon.json` and applied with `kill -HUP` (no container restart). `docker info` lists the insecure registry (user) |
| 4 | Next pull (pod `9b689ea6-...-0`) got past DNS and HTTPS: `Get "http://nemo-helix-builder-registry...:5000/v2/.../manifests/sha256:b2c3af8d...": no basic auth credentials`. Run `run-b136c48fbb7da07a` cancelled | The pod did reference `nvcrimagepullsecret` (service account `default`), but the secret's `.auths` held only `nvcr.io`; the registry login added earlier had not landed. The registry password itself was good (`curl -u builder:<pw> http://10.212.113.186:5000/v2/` returned 200, a wrong `$KEYS` path first returned 401) | Merged an `auths["nemo-helix-builder-registry.default.svc.cluster.local:5000"]` entry (`builder:<registry-password>`, base64) into `nvcrimagepullsecret`, keeping `nvcr.io`. `jq '.auths \| keys'` then listed both hosts (user) |
| 5 | Resubmitted `nemo evals runs submit --taskset 'sample-8727ff98/hello-suite-efa2a1cc#d4326921...' --tasks 0 --agent oracle` (the CLI republished it as `...hello-suite-efa2a1cc-sel-efa2a1cc#d4326921...`, 1 of 1 tasks): `run-0b6948c4d403476f` | | **Passed.** Run `completed`, `result_state: ready`. Task 0 `harbor/hello-world` `succeeded`, job `run-0b6948c4d403476f-task-0` `completed`, 1 trial `hello-world__b8XeJPY` (1 attempt, 15 s, 18:04:39 to 18:04:54 UTC), reward 1.0, no errors. |

Check commands used to separate the causes in row 4 (run on the workbox): pod `.spec.imagePullSecrets` and `.spec.serviceAccountName`; secret `.dockerconfigjson` `.auths | keys`; `curl -u` against the registry `/v2/`.

State after the fixes: taskset images build (`build-images --wait` ready in 40 s). The pull path is now name resolution (hosts entry), HTTP (insecure registry), and credentials (merged secret). With that, the Harbor pipeline is confirmed end to end on the workbox for a no-model agent: publish taskset, Builder image build and push, K8s Job, OpenSandbox sandbox pull and start, trial, reward. The model-calling follow-up is in the next section.

Caveats to carry forward:
- The chart's dev registry is local-only: plain HTTP, one password, images lost on pod restart. Anything that needs a second node, or a restart, needs the node-level steps again (or a real registry for `builder.registry`).
- The `nvcrimagepullsecret` edit is a manual change. If something syncs that secret, it can be overwritten, which would also break nvcr.io pulls.
- Agent-layer builds, if they `FROM` the registry image from the `default-builds` namespace, need the registry credential there too. Not yet exercised (oracle installs nothing).
- Each failed build leaves a `failed` Builder revision (`-1`, `-2`); the plugin submits the next revision, nothing to clean up.
- Follow-up in code (not made): `adopt_or_build` (`plugins/nemo-evals/src/nemo_evals/image_builds/builder.py`) should keep Builder's 409 `detail` in the build error instead of reporting a race. It cost four attempts and a log dig here.
- The first two oracle runs, `run-51feace54d70730d` (DNS/HTTPS pull failure) and `run-b136c48fbb7da07a` (no registry credentials), were cancelled while stuck in `starting`.
- Setup needed on any new node or cluster that pulls Builder images from the dev registry: the registry name resolvable on the node, the registry marked insecure in the node's Docker config, and its login in `nvcrimagepullsecret` of the sandbox namespace. Check `.auths | keys` includes the registry host rather than assuming the merge happened.

## Model-calling agent: openhands on one DeepSWE task (2026-11-19)

Goal: get a real agent run (teacher stage mechanism) before wiring `taskset_run` into the workflow. Scope is deliberately one task, not the 113-task DeepSWE set.

### Setup used
- Teacher: `default/qwen3-6-35b-a3b-fp8`, served by the READY deployment `judge-qwen36-tools` (vLLM, 1 GPU, `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`).
- Platform auth is disabled on the workbox (`nemo auth status`), so no gateway key and no `env_secrets`.
- Spec file `scratch/swe/openhands-run.json`: `source = {name: openhands, model: {ref: default/qwen3-6-35b-a3b-fp8}}`, `agent_kwargs {version: "1.6.0", temperature: "0.7", top_p: "0.8", disable_vision: true}` (sampling values are my choice, not from the docs), `n_attempts: 1`, `max_retries: 1`, `execution: {}`.
- Task: `datacurve/deep-swe` `python-statemachine-state-data-scoping` (`--tasks`, so the published taskset holds 1 of 113 tasks). Command: `nemo evals runs submit --workspace sample-8727ff98 --harbor-hub datacurve/deep-swe --tasks python-statemachine-state-data-scoping --spec-file scratch/swe/openhands-run.json`. Submit builds the image itself; `--list-tasks` works from the laptop.

### What failed and was fixed
| # | Symptom | Cause | Fix |
| --- | --- | --- | --- |
| 1 | `image builds failed ... Builder refused the build: HTTP 409: harbor-img-a41884ff4f2a-1 copies from public.ecr.aws, which this deployment doesn't allow (allowed: docker.io, ghcr.io, nvcr.io, quay.io)` (twice, then the task was marked failed; no run created) | The DeepSWE task Dockerfile starts `FROM public.ecr.aws/...`; the Builder's `import_registries` default is the four hosts listed. Here the 409 detail did reach the CLI message | `helm upgrade nemo-helix k8s/helm -n default --reuse-values --set-json 'platformConfig.builder.import_registries=["docker.io","ghcr.io","nvcr.io","quay.io","public.ecr.aws"]'` (user). A list in `--set-json` replaces the whole default list, so it repeats the four hosts. Other DeepSWE tasks may need further registries; the 409 names the registry |
| 2 | Resubmit right after the upgrade: `Error: [Errno 61] Connection refused` | API pods still rolling | Waited until `GET .../agent-evaluate/runs` returned 200; no run created |

### Result so far
- Image build for the DeepSWE task: `ready`, 1 image, 56 s, through the same dev registry.
- Run `run-24467b95db0c33a8` accepted (1 task), submit warned it is open-book (no-network task, model provider may still use hosted tools; closed-book needs a virtual model with the middleware, not set up).
- Trial `python-statemachine-state-data-s__syZM6Sv` started 18:15:11 UTC and reached `agent_running`. That confirms the whole path past the oracle test: task image pull from the dev registry, sandbox start, openhands 1.6.0 install under `agent_setup_egress`, and model calls through the Inference Gateway.
- Job log at 18:17-18:18 UTC (pasted by user): openhands `CodeActAgent` loaded with 6 tools and 0 MCP tools; agent controller state changes about every 10 s. `[Errno 2]` lines are optional config and microagent lookups; `[Errno 21] Is a directory: '/app'` lines are the agent reading a directory as a file (returned to the model as an observation). Neither is a trial failure.
- vLLM on the deployment pod at 18:23:47 UTC (pasted by user): `Running: 1 reqs, Waiting: 0`, generation about 109 tokens/s, GPU KV cache usage 40%, prefix cache hit rate 93.6%.
- **Final status: cancelled by us at 18:55 UTC after about 40 minutes of `agent_running`** (no error, 1 attempt, never reached a reward). The run API exposes no step-level progress, one stream ran at about 110 tok/s, and the task's agent timeout was not known, so waiting was unbounded. No result for this task; it is not evidence that the model fails it.

### Observations for the real run
- One agent is one serial request stream: 109 tok/s on 1 GPU, plus Qwen3 thinking tokens on every step, makes a single task slow. Aggregate throughput needs many concurrent tasks; one task is a pipeline test, not a throughput test.
- KV cache is already 40% for one conversation. The deployment's `--max-model-len` is not known; a context overflow would show as `ContextWindowExceeded` or `BadRequest` in the task job log.
- Thinking mode: decided to keep it on. Note that the reasoning does not reach the ATIF output (no `reasoning_content` on agent steps; see the later section), so it affects the teacher's behavior and speed but is not in the training data.
- Trial recording: set up afterwards as virtual model `default/qwen36-teacher-rec` (`nemo-evals-trial-recording` in both middleware lists) and used by every later run. The trial bundle then carries the trajectory, completions and token usage (see the later section).
- The API exposes only phase and reward per trial (`/runs/{id}`, `/tasks/{n}`); there is no step-level progress. Progress needs `kubectl logs job/<run>-task-<n>` and the vLLM pod log.

### Still unproven
- Whether the model solves a hard DeepSWE task: not measured (run cancelled). A scored result needs an easier task set and a working verifier; see the later section.
- Agent-layer builds (needs the registry credential in `default-builds` if they `FROM` the dev-registry image). Not exercised: the runs so far install the agent in the sandbox.
- Wiring `taskset_run` into the workflow as the teacher node, and running real trial output through `convert_atif`. The ATIF output now exists (later section); the conversion has not been run on it.

## ATIF to SFT conversion: `customization.atif-convert` (2026-11-19)

The job now reads evaluation trajectories and writes one SFT row per assistant turn. It replaces the two synthetic rows. The output fileset layout and `output.json` keys are unchanged, with additions below.

### What the job reads
`trajectory_filesets` entries are the Jobs fileset of each evaluation job, `workspace/job-fileset-<job>`. Checked against a real fileset on the workbox (`job-fileset-qwen2b-r9-t9`):

```
results/<platform-job-attempt-…>/agent-eval-results/
    tasks.jsonl  trials.jsonl  scores.jsonl  summary.json  run.json  metadata.json
    evidence/<runtime>/<run>/<trial>/…/*.atif.json
```

- Trials come from `trials.jsonl`. The trajectory is found through `evidence.descriptors` (`trace:atif`, kind `trace`, format `atif`; fallback kind `atif`), never by file name. Refs are bundle-relative.
- Scores come from `scores.jsonl`: `metric_type`, `status`, `outputs[{name, value}]`. Harbor rewards are `metric_type: harbor_reward`, output `reward`.
- A retried job attempt repeats trial ids; the first bundle (sorted path) wins.

### Conversion rules
- One row per agent turn: `messages` = conversation so far + that turn, `tools` = `agent.tool_definitions`. Each row extends the previous one. The tool result of a turn is context for the next row, never part of the target.
- `reasoning_content` becomes its own assistant `thinking` message (the schema keeps `thinking` and `content` apart). Tool-call `arguments` are dicts.
- `subagent_trajectories` convert as separate conversations. A conversation with no system/user step before its first agent turn is skipped (`no_prompt`). That is what happens to the Fabric email-agent wrapper step, whose real prompt/reply sit in a subagent trajectory.
- Tool calls with no `tool_definitions` anywhere are skipped (`no_tool_definitions`). Definitions are not invented.
- Train/validation split is by task (stable hash), `validation_fraction` default 0.1. With one task, validation is a copy of train and `validation_from_train` is true.
- `pass_filter` (unknown keys are an error): `{"metric_type": "harbor_reward", "output": "reward", "min_value": 1.0}`. A trial passes if at least one completed matching score exists and every matched value is `>= min_value` (default 1.0 once any key is set). Empty filter keeps all completed trials.

### `output.json` additions
`validation_examples`, `validation_from_train`, `trials_seen`, `sources` (converted trials per input, `completions` or `atif`; see "Real ATIF through `convert_atif`"), `skipped` (counts per reason: `trial_not_completed`, `filtered_by_score`, `no_atif_trace`, `trace_missing`, `trace_invalid`, `completions_unreadable`, `truncated_completion`, `no_prompt`, `no_tool_definitions`, `bad_tool_arguments`). `examples` is the train row count, `runs` the trials that produced rows, `held_out` the validation task ids.

### Local check on synthetic Agent Eval output
`scratch/swe/smith_to_atif.py` writes SWE-smith native trajectories as a job fileset in the layout above (40 trials: 30 resolved, 10 not; ATIF-v1.7, all validated with `nhx_evals_sdk.values.atif.Trajectory`). `scratch/swe/run_convert.py` runs the job's conversion on it with a local-disk source.

```
cd ~/work/scratch/swe && python3 smith_to_atif.py --out atif --resolved 30 --unresolved 10
cd ~/work/nemo-platform && flox activate -- uv run --no-sync python ~/work/scratch/swe/run_convert.py \
    ~/work/scratch/swe/atif/job-fileset-swesmith-c37 /tmp/out '{"metric_type":"harbor_reward","output":"reward","min_value":1.0}'
```

| | trials | train rows | validation rows | schema failures |
|---|---|---|---|---|
| no filter | 40 | 1,224 | 93 | 0 |
| reward filter | 30 (10 filtered) | 828 | 68 | 0 |

Rows equal agent steps (1,317 each). Token lengths with the Qwen3.5 template: median 18,265, p90 46,735, max 120,029. **246 of 1,317 rows (18.7%) exceed 32,768 tokens**: they are the late turns of long trajectories, so packing would drop them and the dataset would lean toward early turns. Without a limit the converter does not cap length, and the packer then drops those rows one at a time without reporting it. The `max_tokens` option below fixes that.

Size: per-turn rows repeat their prefix, so the dataset is much larger than the trajectories. Measured on this sample: 77 KB per row (104 MB for 1,317 rows) and 32.9 rows per trajectory. Extrapolated to the 1.7k-trajectory SWE-smith set that is 25k to 56k rows, about 2 to 4 GB of JSONL, so the job streams rows to local disk and uploads from there instead of holding them in memory. (The extrapolation is rough: this sample was drawn without the 32k whole-trajectory cap.)

### Length limit: `max_tokens` (2026-11-19)
Spec fields on `customization.atif-convert`, set from `DistillInputs.max_tokens` and `student_model` by the workflow's `prepare` node:

```json
"student_model": "default/qwen3-5-4b",
"max_tokens": {"enabled": true, "behavior": "drop-chain", "tokens": "32k"}
```

`tokens` is an integer or a `k` suffix (`32k` is 32768). `enabled: false` keeps the block but turns it off. A limit with no `student_model` is rejected. Unset means no limit and the trainer drops long rows silently, as before.

How a row is measured: the job resolves `student_model` to its model entity, downloads only the tokenizer and chat-template files from the weights fileset (no weights, no GPU; `transformers` is already in `nhx-customizer-tasks`), and renders each row with that chat template. This happens before any trajectory is read, so a missing tokenizer fails the job in seconds. Training adds a few tokens of its own, so set `tokens` a little under the training sequence length.

| behavior | a row over the limit |
|---|---|
| `drop-chain` (default) | every row of its chain goes. The chain is judged by its last row, its longest. |
| `drop-longest` | each over-limit row goes; the shorter earlier rows of the chain stay. |
| `truncate-earlier-turns` | the row stays, with its oldest middle turns cut (an assistant turn plus its tool results, never apart) until it fits. System prompt, task prompt and the target turn always stay. A row that cannot fit with the middle gone is dropped. |

A chain is the rows of one conversation; subagents are their own chains. Rendered length does not decrease along a chain, so a chain is measured with about log2(n) renders, not one per row. That assumes the template does not strip earlier reasoning; if it does, a row can be off by a few tokens and the packer drops the stray one.

`output.json` gains `max_tokens`, `truncated_rows`, `dropped_trials` (first 50 trial ids of dropped chains) and, in `skipped`: `chain_over_max_tokens` (chains), `row_over_max_tokens` (rows, including the rows of dropped chains) and `token_count_failed` (chains whose length could not be measured, dropped).

Measured on the 40-trial synthetic fileset with the real Qwen3.5 tokenizer, limit 32k (`scratch/swe/run_convert.py`, `MAX_TOKENS=... TOKENIZER_DIR=scratch/swe/q35`):

| behavior | chains dropped | rows dropped | rows truncated | rows kept |
|---|---|---|---|---|
| none | 0 | 0 | 0 | 1,317 |
| `drop-chain` | 16 of 40 | 727 | 0 | 590 |
| `drop-longest` | 0 | 246 | 0 | 1,071 |
| `truncate-earlier-turns` | 0 | 0 | 246 | 1,317 |

Every kept row is at or under 32,768 tokens (checked with a separate renderer). For `truncate-earlier-turns`, compared row by row with the unbudgeted output: 1,071 rows unchanged, 246 truncated, row identity and target preserved, retained context always the most recent turns, no tool call separated from its result. Context left after truncation: 169 rows keep 20+ turns, 55 keep 5-19, 17 keep 1-4 and 5 keep none, so those 5 train a late turn with only the task prompt as context.

### Still unproven
- The job against a real fileset through `FilesClient` (needs the `nhx-customizer-tasks` image rebuilt, a workbox-side step).
- `load_student_counter` against a real model entity (unit-tested with fake clients; the real tokenizer path was run on the local Qwen3.5 files). Check the weights fileset has `tokenizer_config.json` at its root.
- Real Harbor ATIF: tool names (OpenHands: `execute_bash`, …) differ from SWE-smith. **Update:** `agent.tool_definitions` is present in real OpenHands ATIF-v1.5 output (6 tools, `scratch/swe/bundle-swesmith2/task{0,1}/`); `run-24467b95db0c33a8` was cancelled, the data came from `run-d24c567a1ada4bf0`. Still to do: run that file through `convert_atif`.
- `scores.jsonl` metric name for Harbor rewards: **confirmed** `harbor_reward.reward` (and `harbor_reward.reward.pass@1` in the summary), from `run-d24c567a1ada4bf0`. See "Model-calling runs, continued" at the end.

## Model-calling runs, continued: teacher trials, ATIF output, verifier network (2026-11-19, 11:50 to 12:20 MST)

### Decisions and mistakes
- The one-task DeepSWE run (`run-24467b95db0c33a8`, `python-statemachine-state-data-scoping`) was cancelled by us after about 40 minutes in `agent_running`: no step-level progress is exposed, one stream ran at about 110 tok/s with thinking on, and its task timeout is unknown. No reward. It proved the path (image pull from the dev registry, sandbox, openhands install, gateway calls).
- Mistake: while that run was going I submitted 8 `swesmith` tasks at once (`run-a98467ac4f96a9e0`, accepted 18:43 UTC) against the instruction to stay at one task; vLLM KV cache was already at 40% for one conversation. Cancelled 2 minutes later (8 of 8 tasks cancelled). The 8 images and the `swesmith` taskset (`harbor-datasets-0e4d75e8#f9dbc49e...`) stay built.
- Mistake: one command copied a task directory and a second submitted from it in the same step, so the submit saw no directory. Resubmitted after the copy.
- Mistake: after the egress change I resubmitted the same 2 `swesmith` tasks (`run-a8ddd76b214339cf`, accepted 19:31 UTC) without noticing that another agent had already submitted them (`run-b191d8dbf55721df`, 19:27 UTC; CLI warned `already running`). My run was a duplicate and, if it was not cancelled, doubles the agents sharing the one vLLM. The evals belong to the other agent; this document is the only thing this agent owns. Its status is not recorded here.
- Thinking stays on (user decision).
- Guardrails for later runs: `execution.max_concurrent_task_jobs`, run-level `timeout_seconds` (a top-level sibling of `spec` and `execution`), and `target.agent_setup_timeout_multiplier`.

### Trial recording and the teacher model
- Created virtual model `default/qwen36-teacher-rec` (default model entity `default/qwen3-6-35b-a3b-fp8`, `nemo-evals-trial-recording` in request and response middleware). The runs below use it as `model.ref`.
- Spec files in `scratch/swe/`: `openhands-rec-hello.json`, `openhands-rec-swesmith.json` (openhands 1.6.0, temperature 0.7, top_p 0.8, `agent_setup_timeout_multiplier: 3`, `max_retries: 0`).

### Hello-world through the real model: alpine fails, ubuntu passes
| Run | Image base | Result |
| --- | --- | --- |
| `run-425665a5c78958b1` (`max_retries: 1`) | `alpine:3.23` | trial `errored`, 2 attempts, `NonZeroAgentExitCodeError`. Visible error: `uv venv /opt/openhands-venv ... A virtual environment already exists` |
| `run-ac20aa086fb76ad5` (`max_retries: 0`, setup timeout x3) | `alpine:3.23` | same error on attempt 1: `Python 3.13 is already installed`, then the venv error |
| `run-de404357bb122617` | `ubuntu:24.04` + `bash curl ca-certificates git` (`scratch/swe/hello-suite-glibc`) | **succeeded, reward 1.0, 1 attempt, 1 min 50 s** (19:09:10 to 19:11:01 UTC), no errors |

Reading: the "venv already exists" error is the second pass of an agent setup that failed once before. The first failure is not visible to us, so the musl cause is a hypothesis, supported by the contrast (the DeepSWE and swesmith images are glibc and install fine; the same task on ubuntu passes). Models and registry path are fine. Use a glibc base for any local task that runs a model-calling agent.

### 2 swesmith tasks through the real model: agent ran, verifier broke
Run `run-d24c567a1ada4bf0` (taskset indexes 0 and 1 of the 8 built `swesmith` tasks, 2 concurrent, thinking on): both trials `succeeded`, reward 0.0, no errors.

| trial | agent runtime | model calls | prompt tokens | completion tokens | agent's final message |
| --- | --- | --- | --- | --- | --- |
| `oauthlib...09vlzwgc` | 119 s | 37 | 876,845 | 7,932 | "All OAuth2 tests pass (187 passed). The OAuth1 test failure is an unrelated environment issue..." |
| `oauthlib...1utlssis` | 103 s | 31 | 618,557 | 6,299 | "All done! What's next on the agenda?" |

**The 0.0 rewards are not valid results.** `verifier/test-stdout.txt` in both trials shows the verifier's own setup failing: `uv add ... swebench==4.0.3 ... swesmith==0.0.6` could not fetch `https://pypi.org/simple/...` (`dns error ... Name or service not known`), after which `pytest: error: unrecognized arguments: --ctrf` (the `pytest-json-ctrf` plugin never installed), so `reward.txt` is 0. The model was never graded. The task's `tests/test.sh` runs `apt-get update`, the uv installer from astral.sh and `uv add` from PyPI at verification time; the tasks are `public` network mode, which reaches only `egress_allow` (the API host) under `harbor-standard`. The install hosts in `agent_setup_egress` are open only during agent install.

Fix (cluster side, **applied by the user**, reported at 12:30 MST; whether the validator accepted the wildcards and whether the verifier now installs its dependencies is not confirmed here, the reruns are tracked by a separate agent): replace `harbor-standard.egress_allow`. Values file `scratch/swe/swesmith-verifier-egress.values.yaml`; apply with `helm upgrade nemo-helix k8s/helm -n default --reuse-values -f <file>`. The user chose TLD wildcards over listing hosts: the file holds the API host (`nemo-helix-api.default.svc.cluster.local`, from the chart's `<release>-api` name; lists are replaced, so it must be repeated) plus `*.com`, `*.org`, `*.net`, `*.io`, `*.dev`, `*.ai`, `*.sh`. These cover what the verifier uses (`pypi.org`, `files.pythonhosted.org`, `github.com`, `*.githubusercontent.com`, the Ubuntu and Debian mirrors, and `astral.sh`, which is why `*.sh` is there).
- Docs host format: hostname, `*.` wildcard hostname, IP or CIDR, no port; there is no bare `*`. A wildcard covers subdomains at any depth but not the bare name, and a name matched only by a wildcard still cannot reach a private or link-local address, so cluster-internal addresses stay blocked (unlike `0.0.0.0/0`, which was rejected for that reason).
- **Untested:** whether the config validator accepts a single-label wildcard such as `*.com`. If the helm upgrade fails with a `ValidationError` on `egress_allow`, fall back to the explicit host list (`pypi.org`, `files.pythonhosted.org`, `archive.ubuntu.com`, `security.ubuntu.com`, `*.debian.org`, `github.com`, `api.github.com`, `raw.githubusercontent.com`, `objects.githubusercontent.com`, `release-assets.githubusercontent.com`, `astral.sh`, `releases.astral.sh`).
- Side effects: every `public` task, agent included, can reach these TLDs while the agent runs, so those runs are open-book (the earlier note about closed-book for them no longer holds). `no-network` tasks (DeepSWE) are unaffected. If `egress_verification: strict` fails a trial on its own rules before the agent starts, suspect the wildcards.
- Alternative without a cluster change: bake the verifier's dependencies into each task image and edit `test.sh`, which changes the verifier script.

The 100 Harbor `swesmith` tasks are all `oauthlib` (one repo); 9 of them share instance ids with our training trajectories and are excluded in `swesmith-eval-overlap.json`. 194 other `oauthlib` instances are in the training data, so this set tests held-out bugs in a seen repo, not a new repo.

### What the trial record contains (confirms earlier open items)
`runs download <run> -o <dir>` gives `agent-eval-results/` with `trials.jsonl`; each trial's `evidence.descriptors` lists artifacts by fileset ref, and `nemo jobs results download agent-eval-results --job <run>-task-<i> --workspace <ws> --output-file x.tar.gz` fetches the files. Per trial: `agent/trajectory.json`, OpenHands `events/*.json` (67 to 83), `completions/*.json` (one per model call), `verifier/test-stdout.txt`, `verifier/reward.txt`, `trial.log`.
- `agent/trajectory.json` is **ATIF-v1.5**, `agent {name: openhands, version: 1.6.0}`, 35 steps for one trial (3 system, 1 user, 31 agent). Agent steps carry `tool_calls` with `function_name` and `arguments` as a dict, plus `observation.results` linked by `source_call_id`.
- **`agent.tool_definitions` is present** (6 tools: `execute_bash`, `think`, `finish`, `execute_ipython_cell`, `task_tracker`, `str_replace_editor`). Tools used in the trial: `execute_bash` 15, `str_replace_editor` 13, `think` 2, `finish` 1. This answers the "tool_definitions presence unconfirmed" item above for OpenHands.
- **No `reasoning_content` on any agent step** (0 of 31), although thinking was on at the model server; the reasoning parser strips it from content and OpenHands does not carry it into ATIF. If reasoning is wanted in the training data, it is not in this output.
- Score metric name for the Harbor reward is `harbor_reward.reward` (summary also shows `harbor_reward.reward.pass@1`), confirming the "assumed `harbor_reward` / `reward`" item above.
- Speed: about 3 s per model call and about 105 to 120 s per trial on these tasks, versus more than 40 minutes with no end for the hard DeepSWE task. Prefix caching and thinking-on were fine at this size.

### Rerun after the verifier egress change (`run-b191d8dbf55721df`, 2 `swesmith` tasks, 19:27 to 19:31 UTC)
- Values applied by the user: `harbor-standard.egress_allow` = API host plus `*.com`, `*.org`, `*.net`, `*.io`, `*.dev`, `*.ai`, `*.sh` (`scratch/swe/swesmith-verifier-egress.values.yaml`). The config accepted single-label wildcards, and the verifier then reached apt, the uv installer and PyPI (it resolved 154 packages and loaded `pytest-json-ctrf`). `egress_verification: strict` did not fail any trial.
- Both trials `succeeded`, reward 0.0, no errors, 1 attempt. This time the verifier ran: `test_state.py::test_patch_resolved` failed with "All the tests did not pass" (`ctrf.json`: 1 failed). These are real verdicts, but only the aggregate is visible: the verifier runs the repo's whole suite in the `testbed` conda env into `/logs/test_output.log`, which is not collected, so which tests failed is unknown.
- Task 0 (`...09vlzwgc`, 21 steps): the agent fixed three bugs in `oauth2/rfc6749/utils.py` and ran only `tests/oauth2` (167 passed) before `finish`. The verifier's `git checkout HEAD~1` worked, and the full suite still did not pass; the task's `combine_file` bug may span more than what the agent found, or tests outside `tests/oauth2` fail. Not distinguishable from the artifacts.
- Task 1 (`...1utlssis`, 59 steps): the agent's last run had 4 failing tests (`4 failed, 219 passed`) and it still reported success. It also created `tests/oauth2/rfc6749/test_request_validator.py`, the path the verifier restores from `HEAD~1`: the verifier's `git checkout HEAD~1` aborted with `untracked working tree files would be overwritten by checkout`, so the real tests were never restored. That is a harness fragility in the swesmith task script (an agent file at a hidden test path blocks the restore), on top of the agent's own failures.
- Teacher pass rate so far: 0 of 2 on `swesmith` with a working verifier (the earlier 0 of 2 were invalid). Too few tasks to say anything about the rate. Failed trajectories are not for SFT; filter on reward 1.0.
- Bundles: `scratch/swe/bundle-swesmith3/task{0,1}/`.

### Six more `swesmith` tasks, 3 at a time (`run-b98d022abf0b77c4`, 19:36 to 19:49 UTC)
Taskset indexes 2 to 7 of `harbor-datasets-0e4d75e8#f9dbc49e...` (`--tasks 2-7`), spec `scratch/swe/openhands-rec-swesmith-c3.json` (`max_concurrent_task_jobs: 3`, `timeout_seconds: 1800`, thinking on). All 6 trials `succeeded` with no errors, 1 attempt, in two waves, about 12 minutes for the run (4.5 to 6 minutes per trial including install). Bundles: `scratch/swe/bundle-swesmith6/task{0..5}/`.

| task (selection index) | reward | agent steps | prompt tokens (sum over calls) | completion tokens |
| --- | --- | --- | --- | --- |
| 0 | 0 | 58 | 1,635,111 | 10,593 |
| 1 | **1** | 41 | 962,018 | 9,972 |
| 2 | 0 | 55 | 1,722,313 | 11,614 |
| 3 | 0 | 70 | 2,200,958 | 20,223 |
| 4 | 0 | 64 | 2,450,105 | 16,003 |
| 5 | 0 | 31 | 487,647 | 8,771 |

- Verifier output checked for all six: pytest ran, no `would be overwritten by checkout` abort, no network errors. Task 1's verifier passed (1 passed); the other five failed the full-suite check (`1 failed`). These are real misses, with the same caveat as above (the failing test names are not collected).
- Every trial ended in `finish`, including the failures; the model reports success whether or not it resolved the task, so its own claim is not a signal.
- Teacher pass rate on the `swesmith` subset with a working verifier: 1 of 8 (this run's 6 plus the 2 above), about 12%. Eight tasks is a small sample; the interval is wide.
- First reward-1.0 trajectory: task 1, 41 agent steps (12 `execute_bash`, 19 `str_replace_editor`, 6 `task_tracker`, 2 `think`, 1 `finish`). OpenHands tools used include `task_tracker`, which is not in the SWE-smith tool set.
- Capacity: 3 concurrent task jobs against the single vLLM replica completed without errors or retries. KV cache and prefix-cache figures during this run were not read.
- Data yield: at about 1 in 8, the 91 non-overlapping `oauthlib` tasks would give on the order of 10 passing trajectories after all are run. That is far too little for SFT; this dataset and this teacher are enough for a pipeline demo, not for a training set. Yield needs a higher pass rate, easier tasks, or many more tasks.

### Real ATIF through `convert_atif`: tools work, content is lossy (2026-11-19, 12:50 MST)
The reward-1.0 trial (`bundle-swesmith6/task1`) through the existing ATIF path gave 41 rows, nothing skipped, with all six tool definitions on every row (`execute_bash`, `think`, `finish`, `execute_ipython_cell`, `task_tracker`, `str_replace_editor`) and all 41 tool calls. So tool definitions and calls were never the problem. Compared with the agent's raw model calls (`agent/completions/*.json`, 39 files, saved in the same bundle), the ATIF is lossy:

| | Harbor OpenHands ATIF | Raw model calls |
| --- | --- | --- |
| Assistant text of a step | a label: `Running command: find /testbed …`, `Reading file: /testbed` | what the model said: "I'll investigate this OAuth1 signature issue…" (on 13 of 39 calls; the rest are tool-call-only) |
| Reasoning | none (`reasoning_content` absent) | `reasoning_content` non-blank on 24 of 39 calls (12 more are whitespace only) |
| System messages | 5 in the last row, including notices ("Retrieving content for…", "Added workspace context") that were not in the first request | exactly what the model was sent: 1 system, 1 user |
| Steps | 41 | 39 calls (2 of them returned two tool calls, which probably explains the 2 extra ATIF steps; not checked) |

Training on the ATIF rows would teach the student to write the labels as its text and would drop the thinking. In the raw requests, earlier assistant turns have empty `content` (OpenHands does not resend its earlier text), and the student runs behind the same agent, so the converter keeps request messages as they were sent and takes text and reasoning only from the turn being trained on.

**Change (customizer):** `atif-convert` now builds rows from `agent/completions/*.json` when a trial has them, and falls back to the ATIF trace when they are missing or unreadable (`nemo_customizer.atif_sft.convert_completions`, `completion_refs`; `jobs/atif_convert.py::_convert_candidate`).
- Completion files are found through the trial's `evidence.descriptors` (`artifact:agent/completions/*.json`), put in time order by their `timestamp`, and each call becomes one row: request `messages` as context, response `reasoning_content` as a `thinking` message, then text and `tool_calls`; `tools` from the request. A call whose request does not extend the previous one starts a new chain. A response with `finish_reason: length` is not a target (`truncated_completion`).
- A trial needs a completions set or an ATIF trace; only when both are absent is it counted `no_atif_trace`. Unreadable or malformed completions are counted `completions_unreadable` and the trial falls back to ATIF (or is skipped if it has none).
- `output.json` gains `sources`, the count of converted trials per input (`completions` or `atif`). The pass filter, task hold-out, and `max_tokens` are unchanged. The trial-recording virtual model is not needed for this.
- Same trial, both paths (`convert_filesets` over `bundle-swesmith6`, reward filter): 39 rows from completions (24 with a `thinking` message, 13 with text, 1 system message in the last row) against 41 rows from ATIF (0 thinking, all 41 with a label as the text, 5 system messages). 259 customizer tests pass, ruff clean.
- Known limit: rows in one chain are not strictly increasing in length here, because earlier thinking is in a row's target but not in later rows' history. On this trial one step shrinks by 7 characters out of 92,506. `max_tokens` assumes lengths never decrease along a chain (`_fit_chain` checks only the last row, then binary-searches). With a limit set, a row could slip over it only if an earlier turn's thinking is longer than every later row's growth. It is unlikely, not impossible, and the budget code is unchanged.

### Next
1. Render rows from the reward-1.0 trial with the Qwen3.5 template and the 4B tokenizer; check row lengths against the 32k limit (rows run up to about 131,000 characters here, so expect many over it).
2. Decide whether `_fit_chain` should stop assuming monotone chains (see the known limit above).
3. Only then widen to more `swesmith` tasks, 2 to 4 at a time.
