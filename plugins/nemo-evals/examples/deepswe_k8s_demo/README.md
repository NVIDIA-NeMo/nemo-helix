<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepSWE on Kubernetes: run, watch, understand

[`deepswe_k8s_demo.ipynb`](deepswe_k8s_demo.ipynb) runs 10 DeepSWE tasks against a deployed NeMo Helix as a Kubernetes taskset run, shows each task's progress while you wait, then charts which tasks the agent solved, where the time went, and why a task failed.

## What you need

- A notebook kernel whose environment has the `nemo` CLI with the evals plugin: `pip install 'nemo-helix[nemo-evals-plugin]'`, or this repository's `.venv`. The first code cell installs pandas and matplotlib into it if they're missing.
- A CLI context for the deployment: `nemo config set --context <name> --base-url <url>`. Set `CONTEXT` in the notebook's first code cell.
- A model for the agent (`codex` or `mini-swe-agent`), picked with `MODEL_SOURCE`. The notebook puts a closed-book virtual model in front of it: DeepSWE tasks have no network, so the platform accepts a model-calling agent only through one.
  - `"custom-provider"` (default): your own OpenAI-compatible endpoint and key, so calls are billed to you. Set `PROVIDER_KEY_ENV` to the environment variable that holds the key, `PROVIDER_URL` to the endpoint and `PROVIDER_MODEL` to a model id it lists at `/v1/models`; for OpenAI use `OPENAI_API_KEY`, `https://api.openai.com` and a model such as `gpt-5`. Section 1 keeps the key in the secret `<provider>-key` in `WORKSPACE`, points a provider named after the URL's host (e.g. `inference-api-nvidia-com`) at the endpoint with it, and uses the model it serves as `MODEL`.
  - `"gateway"`: a model the deployment already serves without a key of yours, such as one deployed in the cluster. Set `MODEL` to it.
- A sandbox profile, set by the platform admin in the Helm values (`evals.runs.harbor.sandbox_profiles`), whose `agent_setup_egress` lists the hosts codex and mini-swe-agent install from (Debian, Node.js, npm, GitHub). The admin can change a profile or add one; pick it with `SANDBOX_PROFILE`.
- Platform auth on: nothing to set up. Section 1 runs in your personal workspace (named after your email, created if missing; set `WORKSPACE` to use another). For a model-calling agent it keeps your own Scoped Access Key (`--scope inference`) in the secret `evals-inference-key` there, creating it if missing and replacing it when it has less than a day left, so only you can use it. Set `INFERENCE_KEY_SECRET` to use a secret of your own instead.

## What it does

| Section | Commands |
| --- | --- |
| 1. Connect and prepare | `pip install pandas matplotlib`, `workspaces create`, `experiments create`, `inference virtual-models create`, and with platform auth on `auth access-keys create \| secrets create` (each only if missing); with `PROVIDER_KEY_ENV`, `secrets create` (or `update`) and `inference providers update` |
| 2. Meet the tasks | Harbor Hub `task.toml` metadata: title, language, category |
| 3. Submit, `MODE = "taskset"` | `evals tasksets create` → `evals tasksets build-images --wait` → `evals runs submit --taskset` |
| 3. Submit, `MODE = "submit"` | `evals runs submit --harbor-hub` (publishes, builds and submits) |
| 4. Watch it run | `evals runs status --json`, every 15 s, as one tile per task |
| 5. Results | `evals runs download`, `jobs results download agent-eval-results` per task |
| 6. What happened | Headline numbers, 3 charts, generated findings, one failed task's drill-down |
| 7. Next | Studio leaderboard links; compare runs (`COMPARE`) |

- Every command is printed before it runs.
- Re-running cells in the same kernel returns the run already started instead of starting another; a new kernel starts a new run.
- `RUN_ID` skips submitting and analyzes an existing run.

## Time

All 10 tasks run at once, so the slowest task sets the pace.

- `codex gpt-6-luna`: about 20 min. Our full run took 19.5 min, its tasks 7–15 min of agent time each. A slow attempt stretches it: `fastapi-implicit-head-options` once took 34 min.
- `oracle` (each task's reference solution, no model): about 2 min. Use it to check the deployment, or as a ceiling next to an agent's run.
- Codex and mini-swe-agent install themselves in each sandbox during setup. A trial that ends in an error, such as a failed install, reruns once in a new sandbox (`max_retries: 1`); a second failure ends it before the agent runs. The notebook reports those separately from tasks the agent didn't solve.

## Run it headless

```bash
papermill deepswe_k8s_demo.ipynb out.ipynb -p CONTEXT <context> -p AGENT oracle -p MODE submit
```
