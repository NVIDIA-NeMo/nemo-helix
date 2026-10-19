# Known issues

Problems we hit while building this tutorial on NeMo Helix (main branch, October 2026) and how to get past each one. Each issue has an ID that the [tutorial](docs/tutorial.md) points to. Newer Helix releases may have fixed some of them.

## Setup and the CLI

### P1. An older `nemo` on your PATH reads a different config

If you installed an earlier NeMo release, its `nemo` can sit ahead of the Helix one on your PATH. It reads a different config folder and won't see your Helix contexts (for example "Context ... not found").

Fix: run `which -a nemo` and use the one inside this repo's `.venv`, or call it by full path. `scripts/preflight.sh` prints the version it found.

### P2. `nemo setup` changed the wrong context

`nemo setup` edits the active CLI context. If that context points at a remote Helix, setup can write local settings into it.

Fix: pass `--base-url http://localhost:8080` to `nemo setup` and to any command where you have more than one context.

### P3. Shutting down the services hangs

With an agent deployed in subprocess mode, stopping the Helix services can wait more than a minute for the agent's child process.

Fix: stop the agent deployment first (`nemo agents deployments delete <name> -y`), or end the process after a minute.

### P4. Reasoning models return only their reasoning

Nemotron 3 models think by default. With a short `max_tokens`, the reply is reasoning text and no answer.

Fix: the repo already sends `chat_template_kwargs: {"enable_thinking": false}` in the Data Designer config, the eval specs and the agent's tool server. Keep it if you change models; set `MERIDIAN_EXTRA_BODY='{}'` for providers that reject the field.

### P5. Python scripts get HTTP 401 from a Helix that requires sign-in

Only matters if you point the scripts at a remote Helix with authentication. The SDK doesn't read the CLI's login.

Fix: `export NHX_ACCESS_TOKEN=$(nemo auth token)` before running a script. The token lasts about an hour, and `nemo auth token` prints an expired token without complaint, so run `nemo auth refresh` first.

### P6. Preflight FAILs on `NVIDIA container runtime  not listed in docker info`

Docker on this machine can't hand a GPU to a container. Training (step 7) and serving (step 8) run in containers, and steps 9 and 10 call the step 8 deployment, so steps 7 to 10 can't work until this passes. Don't skip it.

Fix: install the NVIDIA Container Toolkit and configure Docker with it by following NVIDIA's [install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for your Linux distribution, then restart Docker and rerun `bash scripts/preflight.sh`. Restarting Docker can stop the local Helix services, so check `curl -s http://localhost:8080/health/ready` afterwards and start them again if needed.

## Data Designer

### D1. Generation is slow on a rate-limited provider

On the free build.nvidia.com tier we saw about 0.07 rows a second, so 3,600 training rows would take about 14 hours. With a provider that has higher rate limits, our full run (3,600 training and 500 test rows) took 21 minutes.

Fix: use the packaged dataset in `data/meridian-example/` for a first run. For your own data, use a paid or self-hosted provider for `create`. A free tier is fine for a 10-row smoke test.

### D2. Every template mistake says "prompt generation template is invalid"

The service hides the real cause, which makes Jinja errors hard to find when you edit `data/incidents.py`.

Fix: reproduce locally with the Data Designer library's `DataDesigner().preview()` and read the chained exception (`__context__`). Causes we hit, with their fixes:

| Cause | Fix |
|---|---|
| A conditional sampler with 12 conditions on one column ("Jinja template too complex", depth limit of 10) | Use a subcategory sampler and repeat entries to set weights |
| A subcategory sampler that depends on another subcategory | Use a uniform roll plus an expression column |
| The `format` filter in an expression | Pad by hand: `{% if n < 10 %}0{% endif %}{{ n }}` |
| Double quotes inside a Jinja expression in a schema transform processor | Use single quotes |
| Reading judge scores inside a processor (`quality.score.score` fails because the column is a JSON string there) | Filter on judge scores at split time, as `scripts/split_dataset.py` does |
| `@dd.custom_column_generator` from the Data Designer skill | The Helix service rejects custom columns; use expression columns |

### D3. `preview` times out with slow models

`nemo data-designer preview` waits on one request and times out with a slow provider such as Ollama on a laptop.

Fix: run the smoke test as a job: `nemo data-designer create data/incidents.py -n 10`.

### D4. `check-models` times out on reasoning models

Fix: set `timeout=180` on the model config, or use a model without reasoning for the writer and judge.

### D5. The writer gives away the answer and the judge misses it

About 14% of our generated rows (494 of 3,600) named the runbook, team or action, and the judge often scored those rows as fine. 136 of them got past both the judge and the keyword check.

Fix: keep all three gates in this repo: the judge, the keyword columns in `data/incidents.py` and the pattern sweep in `scripts/split_dataset.py`. Then run `python scripts/check_data.py`. A row that gives away the answer makes a fine-tune look better than it is.

### D6. Downloading the generated dataset

A finished `create` job exposes the data inside a gzipped `artifacts` archive.

Fix: `scripts/split_dataset.py` downloads and unpacks it, applies the gates, splits and uploads the files.

## Files

### F1. `nemo files upload` returns 404

Upload takes the bare fileset name and doesn't create the fileset.

Fix: create it first, then upload with the bare name:

```bash
nemo files filesets create meridian-incidents-train --workspace default
nemo files upload data/meridian-example/train.jsonl meridian-incidents-train --workspace default
```

Everywhere else (job specs, eval specs) the fileset is `default/meridian-incidents-train`, and one file in it is `default/meridian-incidents-train#calibration.jsonl`.

## Fine-tuning with the Helix post-training plugin

### T1. The training images weren't public (resolved 2026-10-11)

The Docker job backend needs `nhx-customizer-tasks` and `nhx-unsloth-training`. Earlier in October 2026 every tag we tried on `nvcr.io/nvidia/nemo-helix/` returned UNAUTHORIZED with a standard NGC key, so the tutorial built them from source.

`nhx-customizer-tasks`, `nhx-unsloth-training` and `nhx-automodel-training` are public on `nvcr.io/nvidia/nemo-helix/` with tags `0.7.0` and `0.7.1`, and they pull without an NGC key. Step 7a pulls the tag that matches your installed Helix and tags it with the local names a pip-installed Helix looks for (`my-registry/<image>:local`; `nemo-helix` 0.7.1 still defaults to those).

If a pull fails for your Helix version, or the machine can't reach nvcr.io, build them once from a Helix checkout with `scripts/gpu-build-images.sh`. Expect 30 to 60 minutes the first time, because it compiles CUDA wheels for the hybrid Nemotron models. If a job can't find the images either way, step 7a shows how to point the Helix post-training plugin at them by full name.

### T2. Nemotron Nano 9B v2 fails late with `Module 'out_proj' is incompatible with Mamba-based models`

The job downloads 18 GB and starts training before it fails, because LoRA targets aren't checked against the architecture at submit time.

Fix: use the target list in `training/job-nemotron-nano.json`: `q_proj k_proj v_proj o_proj up_proj down_proj in_proj`.

### T3. Qwen3.5 trains only part of the model, with no warning

The default LoRA targets skip Qwen3.5's 24 linear-attention layers.

Fix: use the target list in `training/job.json`, which adds `in_proj_qkv in_proj_z in_proj_a in_proj_b out_proj`. If you switch to another hybrid model, check its layer names.

### T4. `warmup_ratio` is accepted and then ignored

The trainer logs that `warmup_ratio` was renamed and ignores it, so the runs have no warmup. Our results were trained this way.

### T5. Training uses 4-bit loading unless you say otherwise

The backend defaults to `load_in_4bit: true`, so these runs were effectively QLoRA. The submitted spec doesn't show it.

### T6. How to train on the answer only

The docs describe chat-format rows. For a model that answers with one letter, you want the loss on that letter only.

Fix: use `prompt`/`completion` JSONL rows with `apply_chat_template: false`, as in `data/meridian-example/train.jsonl` and `training/job.json`.

### T7. The loss curve is buried in a long status record

`nemo jobs get-status` prints about 1,900 lines of JSON.

Fix: pull the curve with `jq` (the command is in step 7 of the tutorial).

### T8. Registering a Hugging Face base model has no documented CLI path

Fix: `training/register_base_model.py` creates the Hugging Face-backed fileset and the model entity. Newer CLIs also offer `nemo customization unsloth submit --upload-model <repo>`.

## Serving with Deployments

### S1. Serving a fine-tuned model is SDK-only

Fix: `training/deploy_model.py` creates the vLLM deployment config and the deployment, then waits for the gateway.

### S2. A deployment sits in PENDING for 20 minutes or more with no logs

Our deployments took 21 minutes (Qwen) and 30 minutes (Nemotron) from create to ready. There's no logs command for a model deployment yet.

Fix: wait. `deploy_model.py` waits up to `--timeout-minutes` (use 45).

### S3. The adapter returns 422, or doesn't appear in model lists

A LoRA adapter answers only on its deployment's provider route, `/apis/inference-gateway/v2/workspaces/default/provider/<deployment>/-/v1`. The standard OpenAI route returns 422 for an adapter name, and `nemo inference models list` and `deployments list-models` don't show adapters.

Fix: list what the deployment serves with `nemo inference providers get <deployment> -f json | jq -r '.served_models[].served_model_name'`, and call the provider route. The eval specs and the agent config in this repo already do. The base model is served as `default/<model>` and an adapter as `default--<output name>`.

### S4. `--input-file` rejects request bodies

`nemo inference gateway provider post ... --input-file body.json` fails with "Unknown input fields".

Fix: pass the body inline with `--body`.

### S5. An untuned base model scores near zero on text matching

Base models often answer with a newline instead of a letter.

Fix: read the option probabilities, as `eval/analyze_results.py` does.

## Evaluations with the Helix Evals plugin

### E1. Spec rejected: "mismatched fields: ['name']"

The metric output `name` must equal the metric type, `exact-match`. The specs from `eval/make_eval_specs.py` already do this.

### E2. `'item' is undefined` in a generic target body

Target request bodies only see `{{ prompt }}`. Put row fields into `prompt_template` instead, as the specs here do.

### E3. Dropped requests count as wrong answers

With `ignore_request_failure`, a request that fails scores 0 in the aggregate.

Fix: keep parallelism low with retries (the specs here do). `eval/analyze_results.py` and `eval/analyze_agent.py` report unanswered rows separately.

### E4. No macro-F1, calibration or confidence threshold built in

Fix: `eval/analyze_results.py` and `eval/cascade.py` compute them from the downloaded row scores.

### E5. No latency or token use per row

Fix: `eval/bench_latency.py` measures speed and tokens with direct calls.

### E6. An agent evaluation hits the 1-hour job limit and every row is lost

Results are written only at the end, so a timed-out job leaves nothing to download.

Fix: score 50 incidents per job. `scripts/run-agent-evals.sh` splits the test set into batches of 50 for you. Run agent evaluations when the machine is otherwise idle.

## Agents

### A1. `nemo agents create` fails with `[Errno 17] File exists`

Happens with a relative `--agent-config` path.

Fix: pass an absolute path, for example `--agent-config "$PWD/agent/agent-with-context.yaml"`.

### A2. Subprocess agent fails with `No module named nemo_fabric_adapters`

The agent's adapter host picks up `python3` from PATH, which may not be the environment Helix runs in.

Fix: start the Helix services from a shell where this repo's `.venv` is active, so its `bin` comes first on PATH. Then redeploy the agent.

### A3. Deployment fails when `artifacts` or `workspace` sits next to the agent YAML

Those names are reserved runtime folders, and the whole agent folder is uploaded.

Fix: keep runtime output outside `agent/`. The tool server writes its action log to `runs/`.

### A4. `nemo agents redeploy` fails with 409 "Deployment already exists"

A failed deployment record survives the teardown.

Fix: `nemo agents deployments delete <name> -y`, then deploy again.

### A5. `host_timeout` when the agent starts

"adapter host did not complete start within 90000 ms" shows up when the machine is under heavy load.

Fix: run agent work when the machine is idle, and redeploy.

### A6. Warnings about `__pycache__` files on `nemo agents create`

Harmless. Delete `agent/mcps/__pycache__` first to quiet them.

### A7. The agent adds a lot of tokens to every call

The Deep Agents harness adds 12,000 to 16,000 prompt tokens to each call of this three-tool agent. That's why a whole agent run takes about the same time whichever model handles the triage step. The speed and cost numbers in [RESULTS.md](RESULTS.md) are for the triage step's model.

## Using a coding agent

The Helix skills get a coding agent most of the way. When we gave one only the skills and a short description, it still needed these project facts, so write them into your project's agent instructions:

- Which Helix URL and workspace to use, and that every reference (`default/...`) must match the workspace.
- Which provider and exact model names to use for the writer, judge, agent loop and baseline.
- Which training backend and spec worked (`training/job.json`, Unsloth, prompt/completion rows) and that the LoRA target lists are required for Qwen3.5 and Nemotron Nano.
- Which files are train, validation and calibration.
- To show a long job's status once and stop, instead of polling in a loop, and to run one command per shell call.
