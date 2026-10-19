# Fine-tune your own fast decision model on an open stack

This tutorial goes with the GTC Berlin session "Post Training Open Models with Synthetic Data for Specialized Agents" (Yamini Kagal and Stefan Gavrilovic, NVIDIA).

Agents make the same decisions over and over, usually by calling a large general model. In this tutorial you train a small open model to take over one of those decisions: you generate the training data, fine-tune on a single GPU and measure the result. Everything runs on [NeMo Helix](https://github.com/NVIDIA-NeMo/nemo-helix) with open source tools.

The example is an incident agent at Meridian Compute, a made-up company that runs GPU clusters. For each alert, Slack message or ticket, the agent's triage step picks which of 12 runbooks owns it and which of 4 actions to take. The agent then checks status and acts. Meridian's internal service names and response policy aren't public, so a general model has to be told them, and it still gets the action right only 70% of the time. A 4B model fine-tuned on Meridian's history gets 82%. The full numbers are in [RESULTS.md](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/RESULTS.md).

## Before you start

**Hardware.** Steps 0 to 5 and the agent baseline run on a laptop, macOS or Linux. Steps 6 to 10 train, serve and use a model and need a Linux x86_64 machine with an NVIDIA GPU and the NVIDIA Container Toolkit. Helix asks for at least 40 GB of GPU memory and CUDA 13. We've run every step on a single A100 40GB with a local Helix: training peaked at 14 GB of GPU memory and serving at 30 GB. Our published results used one A100 80GB per job. Plan for about 120 GB of free disk for the container images.

**Accounts.** An inference provider for the agent's own model and, if you generate data, for the writer and judge models. `nemo setup` supports NVIDIA Build, OpenAI, Anthropic, Google Gemini, Ollama and any OpenAI-compatible endpoint. Base models come from Hugging Face; no token is needed for the two used here. Step 10 sends about 16,000 prompt tokens per incident to the hosted model, so a free build.nvidia.com key gets rate limited partway through ([E7](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#e7-a-free-buildnvidiacom-key-cant-keep-up-with-the-agent-evaluations)).

**Time and cost.** Times below are from our runs. With the packaged data on one A100 80GB, plan on about 2 to 4 hours end to end plus the image download in step 7a: steps 0 to 6 about 30 minutes, training about 30 minutes, deploying 20 to 45 minutes, evaluations about 15 minutes and the agent comparison 30 minutes for 50 incidents or about 2 hours for all 200. Everything we ran cost about $13 to $40 at public prices, with an A100 80GB at $1.66 to $6.03 an hour (Brev listings, October 2026).

| Step | What you do | Where it runs | Our time |
|---|---|---|---|
| [0](#step-0-preflight) | Check the machine | Any | 1 min |
| [1](#step-1-install-nemo-helix-and-connect-a-provider) | Install NeMo Helix, connect a provider | Any | 10 to 15 min |
| [2](#step-2-run-the-agent-once-on-a-general-model) | Register the agent, run it once | Any | 5 min |
| [3](#step-3-get-the-incident-history) | Use the packaged data, or generate it | Any | 1 min, or 21 min to generate (about 14 h on a free tier) |
| [4](#step-4-check-the-data) | Check the data | Any | 1 min |
| [5](#step-5-upload-the-data-as-filesets) | Upload the data | Any | 1 min |
| [6](#step-6-register-the-base-model) | Register the base model | GPU machine | A few minutes |
| [7](#step-7-pull-the-training-images-and-fine-tune) | Pull images, fine-tune | GPU machine | About 21 GB to download, 30 min to train Qwen |
| [8](#step-8-deploy-the-base-model-and-the-adapter) | Deploy base plus adapter | GPU machine | 21 min (Qwen), 30 min (Nemotron) |
| [9](#step-9-evaluate-before-and-after) | Evaluate | GPU machine | 5 to 15 min |
| [10](#step-10-put-the-fine-tuned-model-behind-the-triage-step) | Swap the model in the agent, compare | GPU machine | 30 min (50 incidents) to 2 h (200) |
| [11](#step-11-adapt-it-to-your-own-agent) | Adapt it to your agent | Any | Your call |

Run every command from the repo root. The commands assume a local Helix at `http://localhost:8080` and the `default` workspace.

## Step 0: Preflight

**Goal:** find out what's missing before you spend time on it.

On a fresh machine, clone the repo first with the `git clone` line in step 1 and run this from the repo root. Until step 1 is done, FAILs for the nemo CLI, Helix services, inference provider, script packages and agent tool server are expected.

```bash
bash scripts/preflight.sh            # on the GPU machine
bash scripts/preflight.sh --no-gpu   # on a laptop, for steps 0 to 5
```

Add `--model nemotron` to check for Nemotron Nano 9B v2 instead of Qwen3.5-4B. The script only reads; it never prints keys or tokens.

**You should see:** one line per check, then a result. On a ready GPU machine every line says PASS and the last line is `Result: ready (0 warning(s)).` The script exits 0. On a 40 GB GPU the GPU line is a WARN and the last line is `Result: ready (1 warning(s)).`, which is fine for Qwen3.5-4B. On a Mac without a GPU, `--no-gpu` skips the GPU checks:

```
PASS  OS                        Darwin arm64 (fine without a GPU)
PASS  Python                    3.13.13 (.venv/bin/python)
PASS  nemo CLI                  version 0.6.0...
PASS  Helix services            http://localhost:8080/health/ready says ready
SKIP  GPU                       no NVIDIA GPU visible (...) (not needed with --no-gpu)
```

**If it fails:** each FAIL line prints its fix. Run step 1 first if the nemo CLI, Helix services or provider checks fail, then run the preflight again. An `NVIDIA container runtime` FAIL blocks steps 7 to 10; see [P6](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#p6-preflight-fails-on-nvidia-container-runtime--not-listed-in-docker-info).

## Step 1: Install NeMo Helix and connect a provider

**Goal:** a running local Helix with one inference provider, in a Python environment the repo scripts and the agent's tool server share.

```bash
git clone -b gtc-berlin-2026-tutorial --single-branch https://github.com/NVIDIA-NeMo/nemo-helix.git meridian-tutorial && cd meridian-tutorial
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt   # NeMo Helix plus what the scripts import
uv pip install -e agent              # the agent's tool server, meridian-ops-tools
nemo setup
```

`nemo setup` starts the local services, asks for a provider and key, and picks a default model for agents. The agents in this repo run their loop on Nemotron 3 Ultra (`models.default.model` in the agent YAML), so the default `nemo setup` picks doesn't affect them. Run it from this activated shell so the agent you deploy later finds the same Python.

To set up without prompts, with a build.nvidia.com key:

```bash
export NVIDIA_API_KEY=...   # your key; never commit it
nemo setup --auto --start-services --no-install-skills --ready-timeout 360
```

To add OpenRouter as a second provider:

```bash
printf '%s' "$OPENROUTER_API_KEY" | nemo secrets create openrouter-key --from-file -
nemo inference providers create openrouter --host-url https://openrouter.ai/api/v1 --api-key-secret-name openrouter-key
```

Check it:

```bash
curl -s http://localhost:8080/health/ready
nemo inference providers list -f table
nemo models list -f table
```

**You should see:** `{"status":"ready"}`, your provider with status `READY`, and the models it serves. Check that Nemotron 3 Ultra is in the list (`nvidia-nemotron-3-ultra-550b-a55b` on build.nvidia.com); the agents' loop and the baseline both use it. If your gateway names it differently, use that name in the agent YAML files. For step 3, also note a writer and a judge.

**If it fails:** if an old `nemo` answers, see [P1](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#p1-an-older-nemo-on-your-path-reads-a-different-config). If you have more than one CLI context, see [P2](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#p2-nemo-setup-changed-the-wrong-context). Local services need Docker running.

## Step 2: Run the agent once on a general model

**Goal:** see the agent work end to end with a general model in its triage step. This is the baseline you'll beat.

The agent is defined in `agent/agent-with-context.yaml`. Nemotron 3 Ultra runs its loop, and its `route_incident` tool asks a model to triage, with Meridian's notes in the prompt. In this baseline that model is Nemotron 3 Ultra too. Set it to the name from `nemo models list`:

```yaml
# agent/agent-with-context.yaml, under mcp.servers.meridian-ops.env
ROUTER_MODEL: nvidia-nemotron-3-ultra-550b-a55b
```

Then register, deploy and send it one incident:

```bash
nemo agents create --name infra-ops-agent-with-context --agent-config "$PWD/agent/agent-with-context.yaml"
nemo agents deploy --agent infra-ops-agent-with-context --name infra-ops-agent-with-context-deployment --mode subprocess
nemo agents invoke -d infra-ops-agent-with-context-deployment --no-progress --input-file incidents/slack-pod-c.json
```

`incidents/slack-pod-c.json` is a Slack message from the test set: the dispatch queue on pod-c is lagging, but nothing is broken for users yet.

**You should see:** a runbook ID on the first line, an action on the second, then two to four sentences on what the agent checked and did. In our run, Nemotron 3 Ultra with the notes answered `RB-03` and `page-oncall`. Meridian's policy says `ticket-business-hours` for this one, because nothing is broken for users yet. The agent also shows up in NeMo Studio.

**If it fails:** a 503 that ends in "the 'deepagents' package is required" is [A8](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#a8-the-agent-returns-503-the-deepagents-package-is-required). `[Errno 17] File exists` means a relative path ([A1](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#a1-nemo-agents-create-fails-with-errno-17-file-exists)). `No module named nemo_fabric_adapters` means the services started outside this `.venv` ([A2](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#a2-subprocess-agent-fails-with-no-module-named-nemo_fabric_adapters)). A 409 on redeploy is [A4](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#a4-nemo-agents-redeploy-fails-with-409-deployment-already-exists). A `host_timeout` is [A5](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#a5-host_timeout-when-the-agent-starts). A blank or reasoning-only answer from the router is [P4](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#p4-reasoning-models-return-only-their-reasoning).

## Step 3: Get the incident history

**Goal:** labeled incidents to train and test on. Meridian has no real history, so it's generated with Data Designer.

### Recommended for a first run: use the packaged data

`data/meridian-example/` holds the exact dataset behind our results: 2,503 training incidents, 237 for validation, 240 for calibration and 416 for testing. Skip to [step 4](#step-4-check-the-data).

### Or generate your own

`data/incidents.py` is the Data Designer recipe. It reads `taxonomy.json`, picks the right answer first (runbook, scenario, severity, how the incident is reported), then has a writer model write the incident without naming the answer, and a judge model score it. Point it at your provider and models (the defaults are OpenRouter names):

```bash
export MERIDIAN_PROVIDER=default/openrouter                  # default/<your provider name>
export MERIDIAN_WRITER_MODEL=nvidia/nemotron-3.5-lightning
export MERIDIAN_JUDGE_MODEL=nvidia/nemotron-3-super-120b-a12b
nemo data-designer validate data/incidents.py
nemo data-designer check-models data/incidents.py
nemo data-designer create data/incidents.py -n 10            # smoke test, runs as a job
MERIDIAN_SPLIT=test  nemo data-designer create data/incidents.py -n 500
MERIDIAN_SPLIT=train nemo data-designer create data/incidents.py -n 3600
```

Each `create` prints a job ID like `nemo-data-designer-p-xxxxxxxx`. Check on it with `nemo jobs get-status <job>`. Use different models for the writer and the judge, and neither should be the model you compare against later. `MERIDIAN_SPLIT=train` leaves out one scenario type per runbook, so the test measures how well the model handles incidents it never saw.

**You should see:** the validate and check-models commands pass, and each job ends `completed`. Our full run took 20 min 40 s with both jobs in parallel on a provider with high rate limits, and cost $1.29 at OpenRouter list prices. On the free build.nvidia.com tier we saw about 0.07 rows a second, so the 3,600 training rows alone would take about 14 hours ([D1](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#d1-generation-is-slow-on-a-rate-limited-provider)).

**If it fails:** slow or rate-limited, see [D1](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#d1-generation-is-slow-on-a-rate-limited-provider). "Template is invalid" after you edit the recipe, see [D2](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#d2-every-template-mistake-says-prompt-generation-template-is-invalid). `check-models` times out, see [D4](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#d4-check-models-times-out-on-reasoning-models). Use `create -n 10` for smoke tests; `preview` times out on slow models ([D3](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#d3-preview-times-out-with-slow-models)).

## Step 4: Check the data

**Goal:** make sure no incident gives away its answer, every runbook has examples and the test set holds scenario types the model never trained on.

If you used the packaged data:

```bash
python scripts/check_data.py
```

If you generated your own, apply the quality gates and split first, then check (`--no-upload` keeps the files local until step 5):

```bash
python scripts/split_dataset.py --job <test-job> --split test --fileset default/meridian-incidents-test --no-upload
python scripts/split_dataset.py --job <train-job> --split train --fileset default/meridian-incidents-train --no-upload
python scripts/check_data.py runs/datasets/train runs/datasets/test
```

`split_dataset.py` drops rows the judge scored low on label support or on giving away the answer, rows the keyword check flagged and rows that match the pattern sweep. It splits the training run into train, validation and calibration, balanced by runbook.

**You should see,** for the packaged data:

```
Rows per file
  train          5006 records (2503 incidents, two questions each)
  validation      474 records (237 incidents)
  calibration     240 incidents
  test            416 incidents

Training incidents per runbook
  RB-01 214  RB-02 246  RB-03 272  RB-04 367  RB-05 171  RB-06 195  RB-07 150  RB-08 219  RB-09 213  RB-10 216  RB-11 172  RB-12 68
...
Test incidents from scenario types held out of training: 116 of 416
Test incidents that also appear in training: 0

Incidents whose text gives away the answer (runbook ID, team, action or scenario name):
  train 0  calibration 0  test 0

PASS: data looks ready to train on
```

For a generated run, `split_dataset.py` prints what each gate dropped. Ours kept 2,980 of 3,600 training rows and 416 of 500 test rows; 80% of the drops gave away the answer.

**If it fails:** incidents that give away the answer, see [D5](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#d5-the-writer-gives-away-the-answer-and-the-judge-misses-it). Download trouble, see [D6](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#d6-downloading-the-generated-dataset).

## Step 5: Upload the data as filesets

**Goal:** put the training and test files where the Helix post-training plugin and the Helix Evals plugin can read them. Each training record is a prompt and a one-letter answer, so the model is trained on the answer only.

Packaged data:

```bash
gunzip -k data/meridian-example/*.gz
cp data/meridian-example/test.jsonl data/meridian-example/eval.jsonl   # the eval specs read eval.jsonl
nemo files filesets create meridian-incidents-train --workspace default --description "Meridian incident triage"
for f in train validation calibration; do
  nemo files upload data/meridian-example/$f.jsonl meridian-incidents-train --workspace default
done
nemo files filesets create meridian-incidents-test --workspace default --description "Meridian incident triage"
nemo files upload data/meridian-example/eval.jsonl meridian-incidents-test --workspace default
nemo files list meridian-incidents-train --workspace default
```

Generated data: rerun the two `split_dataset.py` commands from step 4 without `--no-upload`. They create both filesets and upload the same file names.

**You should see:** `nemo files list` shows `train.jsonl`, `validation.jsonl` and `calibration.jsonl`. Elsewhere you refer to them as `default/meridian-incidents-train`, and to the test file as `default/meridian-incidents-test#eval.jsonl`.

**If it fails:** a 404 on upload, see [F1](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#f1-nemo-files-upload-returns-404).

## Step 6: Register the base model

**Goal:** tell Helix which Hugging Face model to fine-tune. The weights download when a job or deployment first needs them.

On the GPU machine. If you did the earlier steps on a laptop, repeat steps 0, 1, 2 and 5 here first. Step 10 compares against the step 2 agent deployment, so it has to run on the same Helix as the rest.

```bash
python training/register_base_model.py    # Qwen/Qwen3.5-4B-Base -> default/qwen3.5-4b-base
# or Nemotron Nano 9B v2, a small model from NVIDIA's open Nemotron family:
python training/register_base_model.py --repo nvidia/NVIDIA-Nemotron-Nano-9B-v2 --name nemotron-nano-9b-v2
nemo models get qwen3.5-4b-base --workspace default
```

**You should see:**

```
created fileset qwen3.5-4b-base
created model entity qwen3.5-4b-base
model spec resolved: True
use in training/job.json as model.name: default/qwen3.5-4b-base
```

**If it fails:** HTTP 401 from a remote Helix, see [P5](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#p5-python-scripts-get-http-401-from-a-helix-that-requires-sign-in). `model spec resolved: False` usually means Hugging Face isn't reachable from the machine running Helix; rerun `scripts/preflight.sh`.

## Step 7: Pull the training images and fine-tune

**Goal:** a LoRA adapter that knows Meridian's runbooks and policy.

### 7a. Pull the training images

The Docker job backend runs training in two images, `nhx-customizer-tasks` and `nhx-unsloth-training`, and deployments use a third, `nhx-api`. NVIDIA publishes all three on `nvcr.io/nvidia/nemo-helix/`, and they pull without an NGC key or `docker login` (checked 2026-10-11 for tags `0.7.0` and `0.7.1`). Use the tag that matches the Helix version step 1 installed.

A pip-installed Helix doesn't look on nvcr.io by itself. In `nemo-helix` 0.7.1 the platform's `image_registry` defaults to `my-registry` and `image_tag` to `local`, so a local Helix looks for `my-registry/nhx-unsloth-training:local`. Pull the published images and tag them with those local names. Step 8 needs two of them under their local names: `nhx-api` downloads the model and `nhx-customizer-tasks` loads the adapter. A local Helix doesn't pull images for deployments, so pull the vLLM image here too:

```bash
HELIX_VERSION=$(python -c "import importlib.metadata as m; print(m.version('nemo-helix'))")
echo "$HELIX_VERSION"   # 0.7.1 in October 2026; the image tag must be the same number
for img in nhx-customizer-tasks nhx-unsloth-training nhx-api; do
  docker pull "nvcr.io/nvidia/nemo-helix/$img:$HELIX_VERSION"
  docker tag "nvcr.io/nvidia/nemo-helix/$img:$HELIX_VERSION" "my-registry/$img:local"
done
docker pull docker.io/vllm/vllm-openai:v0.22.1   # the serving image step 8 uses
docker images --format "{{.Repository}}:{{.Tag}}\t{{.Size}}" | grep -E "nhx-|vllm"
```

**You should see:** each pull ends `Status: Downloaded newer image`, and the last command lists each `nhx-` image twice (the `nvcr.io` name and the `my-registry/...:local` name share one image ID), plus the vLLM image. Docker lists `nhx-customizer-tasks` at about 36 GB, `nhx-unsloth-training` at 41 GB, `nhx-api` at 3.6 GB and vLLM at 35 GB.

**If `docker pull` fails** with `manifest unknown` or `not found`, there's no image for your Helix version. Run `echo $HELIX_VERSION` and install a Helix version that has images (`uv pip install "nemo-helix[all]==0.7.1"`), or build the images from source as below.

**If the job in 7b still fails pulling `nhx-unsloth-training` or `nhx-customizer-tasks`,** point the Helix post-training plugin at the published images by their full names and restart the services from this repo's activated shell. Delete any agent deployments first, or the restart can hang ([P3](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#p3-shutting-down-the-services-hangs)). These settings come from the Helix source:

```bash
export NHX_UNSLOTH_TRAINING_IMAGE=nvcr.io/nvidia/nemo-helix/nhx-unsloth-training:$HELIX_VERSION
export NHX_CUSTOMIZER_TASKS_IMAGE=nvcr.io/nvidia/nemo-helix/nhx-customizer-tasks:$HELIX_VERSION
nemo services restart
curl -s http://localhost:8080/health/ready
```

The services read these variables when they start, so set them again in any shell you later restart the services from. They cover training only; step 8 still looks for `my-registry/nhx-customizer-tasks:local`, so keep that tag.

**Fallback: build the images from source.** Use this only if you can't pull, for example on a Helix version with no published images or a machine that can't reach nvcr.io. The script builds `my-registry/nhx-customizer-tasks:local` and `my-registry/nhx-unsloth-training:local` with Helix's own bake file and pulls the vLLM image. The clone checks out the Helix release tag that matches your installed version. Expect 30 to 60 minutes the first time, because it compiles CUDA wheels for the hybrid Nemotron models:

```bash
git clone --branch "$HELIX_VERSION" https://github.com/NVIDIA-NeMo/nemo-helix.git ~/nemo-helix
HELIX_DIR=~/nemo-helix bash scripts/gpu-build-images.sh
```

### 7b. Submit the training job

```bash
nemo customization unsloth submit training/job.json -w default
nemo jobs watch <job> --workspace default
```

`training/job.json` trains Qwen3.5-4B: LoRA rank 16, two epochs, one GPU, loss on the answer only, seed 29. For Nemotron Nano use `training/job-nemotron-nano.json`. Keep the LoRA target lists in both files ([T2](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#t2-nemotron-nano-9b-v2-fails-late-with-module-out_proj-is-incompatible-with-mamba-based-models), [T3](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#t3-qwen35-trains-only-part-of-the-model-with-no-warning)).

Check the loss curve and the GPU it ran on:

```bash
nemo jobs get-status <job> --workspace default -f json | jq -r '"status: \(.status)",
  (.status_details.metrics.train_loss[] | select(.step == 10 or .step % 100 == 0 or .step == 310) | "step \(.step)  loss \(.value*1000|round/1000)"),
  "validation loss: \(.status_details.metrics.val_loss[-1].value*1000|round/1000)"'
nemo jobs get-logs <job> --workspace default --all-pages -f raw \
  | grep -o -E "NVIDIA [A-Z0-9-]+\. Num GPUs = [0-9]+|'train_runtime': '[0-9]+'" | sort -u
```

On a local Helix, `get-logs` returns no lines, because the Docker job backend doesn't keep the training container's output. Read the step times instead: `nemo jobs get-status <job> --workspace default -f json | jq -r '.steps[] | "\(.name)  \(.created_at)  \(.updated_at)"'`.

**You should see:** `submit` prints the job as JSON, and its `name` (like `unsloth-4bd924999dd1`) is the job ID. The job ends `completed`. In our Qwen run the loss fell from 0.474 at step 10 to 0.103 at step 310, with a validation loss of 0.106. `train_runtime` was 1,303 seconds (22 minutes) on one A100 80GB, and the whole job took 30 minutes. On an A100 40GB with a local Helix the job took 27 minutes, with the same loss curve. Nemotron Nano jobs took about 41 minutes. At $1.66 to $6.03 an hour, the GPU time for one Qwen job costs about $0.75 to $2.70.

**If it fails:** an image pull error means the images from 7a aren't on this machine under the names Helix expects; use the override in 7a ([T1](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#t1-the-training-images-werent-public-resolved-2026-10-11) has the history). `out_proj` is incompatible is [T2](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#t2-nemotron-nano-9b-v2-fails-late-with-module-out_proj-is-incompatible-with-mamba-based-models). The `warmup_ratio` warning in the logs is expected ([T4](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#t4-warmup_ratio-is-accepted-and-then-ignored)). If you run out of GPU memory, lower `batch.per_device_train_batch_size` and raise `gradient_accumulation_steps` to keep their product the same.

## Step 8: Deploy the base model and the adapter

**Goal:** serve the untuned base and your fine-tuned adapter from one vLLM deployment, so you can compare them.

```bash
python training/deploy_model.py --model qwen3.5-4b-base --name meridian-qwen --lora-rank 16 --timeout-minutes 45
nemo inference providers get meridian-qwen --workspace default -f json | jq -r '.status, (.served_models[].served_model_name)'
```

Ask both models the triage question for the incident from step 2, and read the probability of each action (W fix it automatically, X page on-call now, Y ticket for business hours, Z escalate to the incident commander):

```bash
for m in default/qwen3.5-4b-base default--meridian-router-qwen3.5-4b-seed29; do
  echo "== $m"
  nemo inference gateway provider post v1/completions meridian-qwen --workspace default \
    --body "$(jq -nc --arg m "$m" --rawfile p incidents/slack-pod-c.decision-prompt.txt '{model:$m,prompt:$p,max_tokens:1,temperature:0,logprobs:4}')" -f json \
    | jq -r '.choices[0].logprobs.top_logprobs[0] | to_entries[] | "\(.key|@json)  \((.value|exp)*100|round)%"'
done
```

**You should see:** `deploy_model.py` prints `status:` lines, then `gateway ready: True`. Ours took 7 minutes for Qwen on a local Helix, and 21 for Qwen and 30 for Nemotron on a cluster. `providers get` prints `READY`, `default/qwen3.5-4b-base` and `default--meridian-router-qwen3.5-4b-seed29`. If it prints only `READY`, wait a minute and run it again; the adapter loads after the gateway is up. In our run the base model's top answer was a blank line (about 70%), so it didn't pick an action. The fine-tuned model put about 97% on `" Y"`, a ticket for business hours, which is Meridian's policy.

A deployment holds its GPU until you delete it. When you're done with steps 9 and 10:

```bash
nemo inference deployments delete meridian-qwen --workspace default
```

**If it fails:** failing within a minute with "Prerequisite '...-puller' failed" is [S6](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#s6-the-deployment-fails-at-once-with-prerequisite-name-puller-failed). Stuck in PENDING with no logs is [S2](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#s2-a-deployment-sits-in-pending-for-20-minutes-or-more-with-no-logs). A 422 or a missing adapter is [S3](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#s3-the-adapter-returns-422-or-doesnt-appear-in-model-lists). `--input-file` errors are [S4](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#s4---input-file-rejects-request-bodies).

## Step 9: Evaluate before and after

**Goal:** score the untuned and fine-tuned models on the same 416 test incidents with the Helix Evals plugin, then compute accuracy, macro-F1 and calibration.

```bash
QUESTIONS="runbook action" scripts/run-evals.sh "default/meridian-incidents-test#eval.jsonl" B-qwen-untuned C-qwen-finetuned
```

The script writes one spec per model and question, submits them, waits and downloads row scores to `runs/eval/`. The specs call the deployment's provider route.

Score the calibration split too, so you can fit the confidence:

```bash
python eval/make_eval_specs.py --dataset "default/meridian-incidents-train#calibration.jsonl" \
  --out runs/eval/specs-cal --arms C-qwen-finetuned --question action
nemo evaluator evaluate --spec-file runs/eval/specs-cal/C-qwen-finetuned.action.json --workspace default
nemo jobs watch <job> --workspace default
nemo jobs results download row-scores --job <job> --workspace default \
  --output-file runs/eval/cal-C-qwen-finetuned.action.rows.jsonl
```

Then compare:

```bash
python eval/analyze_results.py runs/eval/B-qwen-untuned.action.rows.jsonl
python eval/analyze_results.py runs/eval/C-qwen-finetuned.action.rows.jsonl \
  --calibration runs/eval/cal-C-qwen-finetuned.action.rows.jsonl
```

To add the frontier baseline, set `FRONTIER_MODEL` to its name on your gateway and add `A-frontier-no-context A-plus-frontier-with-context` to the `run-evals.sh` arms. Our scores for every model are in `results/`, so you can also compare against those without calling them.

**You should see:** `evaluator evaluate` prints a job ID like `nemo-evaluator-<id>`. `analyze_results.py` prints JSON per model. Ours for the fine-tuned Qwen action question: `"accuracy": 0.8197`, `"macro_f1": 0.7659`, `"ece": 0.0476`, `"accuracy_heldout_families": 0.8017`. The untuned base: 37.7% accuracy. Each small-model job took under a minute; a 416-row job on Nemotron 3 Ultra took about 2.5 minutes. The fine-tuned model should beat the untuned one by a wide margin on the action.

To see what a confidence threshold buys you, and why escalating uncertain cases to the big model didn't help in our run:

```bash
python eval/cascade.py --small runs/eval/C-qwen-finetuned.action.rows.jsonl \
  --cal runs/eval/cal-C-qwen-finetuned.action.rows.jsonl \
  --large results/frontier/A-plus-frontier-with-context.action.rows.jsonl.gz
```

To measure speed on the model alone: `FRONTIER_MODEL=<name> python eval/bench_latency.py --n 50 --arms A-plus-frontier-with-context C-qwen-finetuned`. Ours: 301 ms typical for the fine-tuned model, 670 ms for Ultra with the notes.

**If it fails:** a spec rejected over `name` is [E1](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#e1-spec-rejected-mismatched-fields-name). `'item' is undefined` is [E2](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#e2-item-is-undefined-in-a-generic-target-body). Many zeros with no answer are [E3](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#e3-dropped-requests-count-as-wrong-answers). An untuned base scoring near zero is expected ([S5](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#s5-an-untuned-base-model-scores-near-zero-on-text-matching)); `analyze_results.py` reads probabilities instead.

## Step 10: Put the fine-tuned model behind the triage step

**Goal:** the same agent, with only the triage step's model changed, scored against the baseline from step 2.

`agent/agent-decision-model.yaml` is the agent from step 2 with `route_incident` pointed at your deployment. The settings are under `mcp.servers.meridian-ops.env`:

```yaml
ROUTER_URL: http://localhost:8080/apis/inference-gateway/v2/workspaces/default/provider/meridian-qwen/-/v1
ROUTER_MODEL: default--meridian-router-qwen3.5-4b-seed29
ROUTER_MODE: logits                  # read the probability of each option
ROUTER_TEMPERATURE_RUNBOOK: "1.40"   # ours, fitted on the calibration split
ROUTER_TEMPERATURE_ACTION: "0.95"    # ours, fitted on the calibration split
```

Replace our two temperatures with yours before you create the agent. Keep the quotes; the tool server reads them as strings.

- `ROUTER_TEMPERATURE_ACTION`: the `temperature=` value on the `calibration:` line that `analyze_results.py` printed for `C-qwen-finetuned.action.rows.jsonl` in step 9. The same number is the `"temperature"` field in that model's JSON.
- `ROUTER_TEMPERATURE_RUNBOOK`: step 9 only calibrates the action question. To fit this one, rerun step 9's calibration commands with `--question runbook` (the spec is then `runs/eval/specs-cal/C-qwen-finetuned.runbook.json`; name the download `cal-C-qwen-finetuned.runbook.rows.jsonl`), then run `python eval/analyze_results.py runs/eval/C-qwen-finetuned.runbook.rows.jsonl --calibration runs/eval/cal-C-qwen-finetuned.runbook.rows.jsonl` and use its `temperature=`. If you skip this, keep ours.

If your deployment or adapter has a different name, change `ROUTER_URL` and `ROUTER_MODEL` too. Check the edit with `git diff agent/agent-decision-model.yaml`.

```bash
nemo agents create --name infra-ops-agent-decision-model --agent-config "$PWD/agent/agent-decision-model.yaml"
nemo agents deploy --agent infra-ops-agent-decision-model --name infra-ops-agent-decision-model-deployment --mode subprocess
nemo agents invoke -d infra-ops-agent-decision-model-deployment --no-progress --input-file incidents/slack-pod-c.json
```

Score both agents on the same incidents, 50 per job:

```bash
BATCHES="1 2 3 4" scripts/run-agent-evals.sh agent-finetuned agent-frontier-with-context
python eval/analyze_agent.py --paired runs/eval/agent/*.rows.jsonl
```

**You should see:** the single incident comes back `RB-03` and `ticket-business-hours` in our run. `analyze_agent.py` prints one line per agent. Ours, on the 68 incidents both agents answered: fine-tuned `"action_accuracy": 0.87`, Nemotron 3 Ultra with notes `0.76`, both 100% on the runbook. Our first 50 incidents were close (86% against 84%), so use all four batches to see a difference. How long each 50-incident job takes depends on your provider's rate limits.

The agent's own loop on Nemotron 3 Ultra takes most of each run's time, so swapping the triage model saves little end to end ([A7](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#a7-the-agent-adds-a-lot-of-tokens-to-every-call)). The fine-tuned model answers faster measured on its own (step 9), and it costs less per answer when its GPU is busy.

**If it fails:** many unanswered rows with "429" in the agent's errors is [E7](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#e7-a-free-buildnvidiacom-key-cant-keep-up-with-the-agent-evaluations). An evaluation that loses all its rows hit the 1-hour job limit ([E6](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#e6-an-agent-evaluation-hits-the-1-hour-job-limit-and-every-row-is-lost)). Agent deployment errors are the same as step 2.

When you finish, clean up (see [Clean up](#clean-up)). The deployment from step 8 holds its GPU until you delete it.

## Step 11: Adapt it to your own agent

**Goal:** the same method on a decision your own agent makes.

First pick the step. Look for one that repeats at volume, where the right answer depends on knowledge that shows up in past examples more than in documents, and where a fixed list of options covers the answers.

Then you change three things.

1. **The options and the policy, in [`taxonomy.json`](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/taxonomy.json).** It holds everything Meridian-specific: the options (`runbooks` and `actions`), their definitions, the `policy` that maps each runbook and severity to an action, the internal service names, `private_context` (the notes only your team knows), the scenario types and which ones to hold out for testing. Data generation, training records, the agent prompt and evaluation all read from it. The prompts in [`agent/mcps/prompting.py`](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/agent/mcps/prompting.py) say "Meridian Compute" and "incident", so change that wording too. As the code stands, the first question takes up to 12 options and the second up to 4 (`LETTERS` and `ACTION_LETTERS` in the same file). Extra options get dropped without a warning, so extend those strings if you have more.
2. **The data: the recipe in [`data/incidents.py`](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/data/incidents.py), or your own labeled tickets.** To generate, change the reporting styles, the weights per option, the writer's rules and the judge's questions. Either way, update `LEAK_PATTERNS` in that file to your own option names, because `scripts/check_data.py` uses them to catch text that gives away the answer. If you have real tickets, skip generation and follow [Bring your own labeled tickets](#bring-your-own-labeled-tickets) below.
3. **The agent's tool and names.** Replace the instructions in [`agent/agent-with-context.yaml`](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/agent/agent-with-context.yaml) and the tools in [`agent/mcps/ops_tools.py`](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/agent/mcps/ops_tools.py) with your own. Keep the triage step as a tool driven by `ROUTER_MODEL` and `ROUTER_MODE`, so you can swap its model with a setting. The model, deployment and adapter names are written into `eval/make_eval_specs.py`, `agent/agent-decision-model.yaml`, `eval/specs-agent/` and `scripts/run-agent-evals.sh`; rename them to match yours.

Then rerun steps 4 to 10. Compare against the general model with your notes in its prompt, since that's what it would get in production. Train more than once with different seeds before you trust a difference.

The [adapt guide](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/skills/helix-specialized-agent-tutorial/reference/adapt.md) in the skill lists the questions to answer first and the other code limits to plan around, such as the top-20 limit on logprobs when you have many options.

Our dataset had about 200 synthetic examples per runbook to keep the demo cheap. For production, plan for a few hundred real examples per option, including the rare ones, labels checked by people, a test set from a later period than the training data, and retraining when your teams or policy change.

### Bring your own labeled tickets

`scripts/split_dataset.py` reads a Data Designer job, so for real tickets you build the four files yourself with the code below, then pick up at step 4.

1. **Check the labels.** Each ticket needs its text as the step saw it at decision time, a first answer and a second answer. The answers must match the `id` values in `taxonomy.json` exactly; the lookup fails on anything else. Leave out anything added after the decision, such as resolution notes or the final queue, since it gives the answer away.
2. **Split by date, by hand.** Sort the tickets by date. Keep the latest period (for example the last two or three months) as your test set. Split the earlier tickets into train, validation and calibration; the tutorial uses 84%, 8% and 8%. Save each as a JSON Lines file with one ticket per line.
3. **Format them as prompt and completion rows.** Training and validation get two rows per ticket, one per question, the way `train_rows()` in [`scripts/split_dataset.py`](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/scripts/split_dataset.py) builds them. That function renders the prompts with `render_runbook_question()` and `render_action_question()` and the answers with `letter_for_runbook()` and `letter_for_action()`, all from [`agent/mcps/prompting.py`](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/agent/mcps/prompting.py). Calibration and test get one row per ticket with the fields `eval_row()` writes, since the eval specs and `eval/analyze_results.py` read them. The simplest way is to call those two functions. From the repo root with `.venv` active, adjusting the field names in `load()` to your files:

   ```python
   import json, sys
   from pathlib import Path
   sys.path.insert(0, "scripts")
   from split_dataset import train_rows, eval_row, write_jsonl

   def load(path):
       rows = [json.loads(line) for line in open(path)]
       return [{"incident_text": r["text"], "runbook_id": r["runbook_id"], "action_id": r["action_id"],
                "scenario_family": None, "reporter_style": None} for r in rows]

   out = Path("runs/my-data"); out.mkdir(parents=True, exist_ok=True)
   write_jsonl([row for r in load("my-train.jsonl") for row in train_rows(r)], out / "train.jsonl")
   write_jsonl([row for r in load("my-validation.jsonl") for row in train_rows(r)], out / "validation.jsonl")
   write_jsonl([eval_row(r) for r in load("my-calibration.jsonl")], out / "calibration.jsonl")
   write_jsonl([eval_row(r) for r in load("my-test.jsonl")], out / "eval.jsonl")
   ```

   This keeps your labels as they are. `split_dataset.py` also runs `relabel_actions()`, which overwrites every action with Meridian's policy table; the code above doesn't call it.
4. **Check the files:** `python scripts/check_data.py runs/my-data`. Real tickets have no scenario types, so the held-out line says 0, which is fine. The other checks still apply: every option needs training examples, no test ticket can also be in training and no text should give away the answer.
5. **Continue from step 5.** Run its upload commands with `runs/my-data` in place of `data/meridian-example`, and skip the `gunzip` and `cp` lines. Then follow steps 6 to 10 as written.

## Clean up

**Goal:** give back the GPU and the disk space. Every command here deletes something, so check each name before you run it. Skip anything you want to keep for another run.

Delete the agent deployments first; stopping the services with an agent still deployed can hang ([P3](https://github.com/NVIDIA-NeMo/nemo-helix/blob/gtc-berlin-2026-tutorial/KNOWN-ISSUES.md#p3-shutting-down-the-services-hangs)). Then the agents, the model deployment and its config:

```bash
nemo agents deployments delete infra-ops-agent-decision-model-deployment -y   # step 10
nemo agents deployments delete infra-ops-agent-with-context-deployment -y     # step 2
nemo agents delete infra-ops-agent-decision-model -y
nemo agents delete infra-ops-agent-with-context -y
nemo inference deployments delete meridian-qwen --workspace default           # step 8; frees the GPU
nemo inference deployments get meridian-qwen --workspace default              # wait for DELETED
nemo inference deployment-configs delete meridian-qwen-cfg --workspace default
```

The adapter, the base model entity and the filesets (steps 5 to 7). A fileset can't be deleted while a model entity points at it, so delete the model first. Read the adapter's name from `nemo models get` rather than assuming ours:

```bash
nemo models get qwen3.5-4b-base --workspace default
nemo models adapters delete meridian-router-qwen3.5-4b-seed29 --model-name qwen3.5-4b-base --workspace default
nemo models delete qwen3.5-4b-base --workspace default
nemo files filesets list --workspace default --all-pages -f table
nemo files filesets delete qwen3.5-4b-base --workspace default
nemo files filesets delete meridian-incidents-train --workspace default
nemo files filesets delete meridian-incidents-test --workspace default
for b in 1 2 3 4; do nemo files filesets delete meridian-incidents-test-b$b --workspace default; done   # step 10's batches
nemo files filesets delete meridian-router-qwen3.5-4b-seed29 --workspace default                          # the adapter's files
nemo files filesets delete infra-ops-agent-with-context-ethos --workspace default                          # made by nemo agents create
nemo files filesets delete infra-ops-agent-decision-model-ethos --workspace default
```

If you trained Nemotron Nano, use `meridian-nemotron`, `meridian-nemotron-cfg` and `nemotron-nano-9b-v2` in place of the Qwen names.

Stop the local Helix services:

```bash
nemo services stop
```

Remove the images from step 7a. Check their sizes first:

```bash
docker images --format "{{.Repository}}:{{.Tag}}\t{{.Size}}" | grep -E "nhx-|vllm"
docker rmi my-registry/nhx-unsloth-training:local my-registry/nhx-customizer-tasks:local my-registry/nhx-api:local docker.io/vllm/vllm-openai:v0.22.1
docker rmi nvcr.io/nvidia/nemo-helix/nhx-unsloth-training:<helix version> nvcr.io/nvidia/nemo-helix/nhx-customizer-tasks:<helix version> nvcr.io/nvidia/nemo-helix/nhx-api:<helix version>
```

The services also pulled `clickhouse/clickhouse-server:26.3` (about 1.2 GB) and deployments pulled `busybox`; remove them with `docker rmi` if nothing else on the machine uses them.

Each image has two names after 7a, and Docker frees the space only once both are removed. Remove any other image the first command listed with `docker rmi <image>`.

If you used the build fallback in 7a, also remove the Helix checkout (`du -sh ~/nemo-helix` shows its size, `rm -rf ~/nemo-helix` removes it). The build leaves a Docker build cache: `docker buildx du` shows its size, and `docker buildx prune` clears it for every project on the machine, not only this one.

If you rented the GPU machine, stop or delete it with your provider when you're done; a running instance bills by the hour. On Brev that's `brev stop <instance>` (keeps the disk) or `brev delete <instance>` (removes it).

If this helped, please star [NeMo Helix on GitHub](https://github.com/NVIDIA-NeMo/nemo-helix).
