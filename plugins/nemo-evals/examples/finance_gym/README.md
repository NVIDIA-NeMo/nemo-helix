<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Finance evaluation as NeMo Gym jobs

Score the Nemotron finance cookbook's test sets with the cookbook's own scoring code, as `agent-evaluate` jobs that
run a custom NeMo Gym environment in a sandbox. The cookbook is
[`usage-cookbook/Nemotron-3.5-Super-VL/lora-finance`](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/usage-cookbook/Nemotron-3.5-Super-VL/lora-finance)
in NVIDIA-NeMo/Nemotron. It trains one LoRA adapter per capability: visual OCR, financial reasoning, and code
generation.

Each test row is one Gym task. The model answers through the Inference Gateway, and the `finance_qa` resources
server scores the answer with the cookbook's `evaluation.py` functions, unchanged. Every score the cookbook records
for a row becomes a reward component, so each row gets `exact`, `relaxed`, `task_correct`, `answer_f1`,
`trace_closed` and `hit_max_new_tokens`, plus `program_syntax_valid` for code. `reward` is `task_correct`, the
cookbook's headline score. The report script then combines the jobs into the cookbook's `metrics()` shape: overall,
`by_capability` and `by_task`.

## Files

- `environment/`: the `native-v1` environment package to upload as a FileSet.
  - `resources_servers/finance_qa/cookbook.py`: the cookbook's scoring functions, copied unchanged.
  - `resources_servers/finance_qa/scoring.py`: rebuilds the cookbook's inputs from a Gym response and returns
    the reward components.
  - `resources_servers/finance_qa/app.py`: the Gym resources server.
- `convert_rows.py`: turns one capability's test rows into Gym rows.
- `finance_report.py`: combines the jobs' results bundles into the cookbook's report.

## Prerequisites

- A NeMo Helix deployment with sandboxed Gym for the Evals plugin. See
  [Configure Sandboxed Gym](https://docs.nvidia.com/nemo-helix/latest/documentation/evaluate-models/agent-eval/gym-sandbox-configuration).
  `native-v1` installs the server's `requirements.txt` when the sandbox starts, so the sandbox needs egress to
  PyPI (`pypi.org:443` and `files.pythonhosted.org:443`).
- A NeMo Helix version that scores Gym reward components per row
  ([#4024](https://github.com/NVIDIA-NeMo/nemo-helix/pull/4024)); the report needs them. With
  [#4032](https://github.com/NVIDIA-NeMo/nemo-helix/pull/4032), a row whose request fails is a failed trial
  instead of failing the whole job.
- The model behind the Inference Gateway, with thinking on, as the cookbook evaluates it. Thinking is a provider
  setting, for example `--default-extra-body '{"chat_template_kwargs": {"enable_thinking": true}}'` on
  `nemo inference providers create`. The rows set the cookbook's other settings: temperature 0, and 256 new
  tokens for arithmetic and count or 512 for every other task.
- The cookbook's data, as the Customizer skill's
  [`import_finance.py`](../../../nemo-customizer/src/nemo_customizer/skills/nemo-customizer/scripts/import_finance.py)
  writes it. That script rebuilds the cookbook's splits and prompts and writes one directory per capability as
  chat-transcript JSONL: `visual_ocr/test.jsonl` with its page images under `visual_ocr/images/`,
  `reasoning/test.jsonl`, and `code_generation/test.jsonl`. Each row is a user turn whose first line is
  `Task: <task>`, then an assistant turn with the reference answer. Customizer trains the adapters on the same
  directories, so `convert_rows.py` reshapes the test rows into Gym rows instead of the script writing Gym rows.

## Run it

The commands run from this directory and assume the `default` workspace. Set `DATA` to the directory
`import_finance.py` writes, or already wrote, then run the import if you haven't yet:

```bash
export DATA=/tmp/finance
uv run --no-project --with datasets --with pillow --with pymupdf --with huggingface_hub \
  ../../../nemo-customizer/src/nemo_customizer/skills/nemo-customizer/scripts/import_finance.py --output "$DATA"
```

1. Upload the environment. Keep the trailing slash, so `nemo-environment.yaml` is at the FileSet's root:

   ```bash
   nemo files filesets create finance-qa-env --workspace default --purpose environment
   nemo files upload ./environment/ finance-qa-env --workspace default
   ```

2. Convert each capability's rows. OCR rows reference their page images in the FileSet that
   `--image-fileset` names. Copy just the images the test rows use, as below: the dataset's `images/` directory
   holds every split's images, about eight times as many. If you already uploaded `$DATA/visual_ocr/` as the
   `finance-ocr` FileSet, as `import_finance.py` suggests, the images are already there at the paths the rows
   use: pass `--image-fileset default/finance-ocr`, drop `--copy-images-to`, and skip the image upload in step 3.

   ```bash
   mkdir -p rows
   python3 convert_rows.py "$DATA/visual_ocr/test.jsonl" --image-fileset default/finance-images \
     --copy-images-to ocr-images --output rows/visual_ocr.jsonl
   python3 convert_rows.py "$DATA/reasoning/test.jsonl" --output rows/reasoning.jsonl
   python3 convert_rows.py "$DATA/code_generation/test.jsonl" --output rows/code_generation.jsonl
   ```

3. Upload the images, unless they're already in `finance-ocr`, and the rows. Keep the trailing slashes: the
   images then keep the `images/` prefix the rows reference, and the rows files sit at the FileSet's root.

   ```bash
   nemo files filesets create finance-images --workspace default --purpose dataset
   nemo files upload ./ocr-images/ finance-images --workspace default
   nemo files filesets create finance-gym-rows --workspace default --purpose dataset
   nemo files upload ./rows/ finance-gym-rows --workspace default
   ```

4. Run one job per capability. Each points at that capability's model through the gateway: the base model, or
   the capability's adapter for the cookbook's comparison. This is `visual_ocr.json`; the others differ only in
   the rows file, the model, and the label:

   ```json
   {
     "tasks": {"fileset": "default/finance-gym-rows#visual_ocr.jsonl"},
     "target": {
       "kind": "gym",
       "environment": "default/finance-qa-env",
       "source": {
         "component": "simple_agent",
         "config": "responses_api_agents/simple_agent/configs/simple_agent.yaml"
       },
       "resources_server": "finance_qa",
       "concurrency": 8,
       "hydra_params": {
         "policy_base_url": "http://<api-service>:8080/apis/inference-gateway/v2/workspaces/default/model/<model>/-/v1",
         "policy_api_key": "not-used",
         "policy_model_name": "<model>"
       }
     },
     "labels": {"capability": "visual_ocr"}
   }
   ```

   ```bash
   nemo evals agent-evaluate --spec-file visual_ocr.json --workspace default
   ```

   `<api-service>` is the NeMo Helix API's in-cluster address, which must be on the sandbox's egress list, and
   `<model>` is the model's name in the gateway. The sandbox sends `policy_api_key`; with auth off the gateway
   ignores it.

5. When the jobs finish, download each one's results bundle and build the report:

   ```bash
   nemo jobs results download agent-eval-results --job <ocr-job> --output-file ocr.tar.gz
   nemo jobs results download agent-eval-results --job <reasoning-job> --output-file reasoning.tar.gz
   nemo jobs results download agent-eval-results --job <code-job> --output-file code.tar.gz
   python3 finance_report.py ocr.tar.gz reasoning.tar.gz code.tar.gz --output report.json
   ```

## Reading the report

- `task_accuracy` and `answer_f1` are the cookbook's headline scores. `closed_trace_rate`,
  `python_syntax_rate`, `coverage` and `hit_max_new_tokens` are diagnostics.
- A failed trial isn't scored, as the cookbook skips a failed request, but it counts in `expected_samples`, so
  `coverage` falls below 1. So does a row with no trial, even from a job that produced none.
- `hit_max_new_tokens` counts rows that used their whole token limit. With thinking on, a model that writes a long
  trace runs out of tokens before it answers, and those rows score 0.
- Each job's own summary also has every component's mean, as `gym_reward.<component>`, over that capability.

## Not yet checked

- A job whose model is a LoRA adapter served through the gateway's model route.
- A deployment with auth on: which key the sandbox needs, and whether the gateway can read the image FileSet with
  it.
- Images read from `finance-ocr` instead of an images-only FileSet. The gateway reads any FileSet the caller can
  read, whatever its purpose, so this should work, but the checked runs used `finance-images`.
