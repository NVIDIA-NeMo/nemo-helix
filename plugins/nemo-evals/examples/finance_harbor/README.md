<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Finance evaluation as Harbor shard tasks

Score the Nemotron finance cookbook's test sets with the cookbook's own scoring code, as a Harbor taskset run.
This is the Harbor counterpart of the Gym example in `../finance_gym/`: the same rows, the same scorer
(`cookbook.py` and `scoring.py` from its `finance_qa` server, which `make_tasks.py` copies into each task unchanged), a different way to run them.

- **One task per shard of rows.** `make_tasks.py` splits each capability's rows into tasks of `--shard-size` rows.
  The rows travel in `instruction.md`; the references in `tests/references.jsonl`.
- **A trusted runner calls the model.** `DatasetRunnerAgent`
  (`nhx_evals_sdk.agent_eval.runtimes.harbor.dataset_runner`) is an external Harbor agent: it runs in the task Job's
  dispatcher pod, so the model endpoint and key never enter the sandbox. It answers every row, a bounded number at a
  time, and writes `/logs/artifacts/responses.jsonl` into the sandbox.
- **The metric runs in the sandbox.** The task has no `[verifier.environment]`, so Harbor's verifier runs in the same
  sandbox, started from the metric image. It uploads `tests/`, runs `test.sh`, and the metric writes `reward.json`
  (each component's mean; `reward` is `task_correct`), `row_scores.jsonl` and `summary.json`.
- **One image for the dataset.** Every task's `environment/` holds the same files, so Builder builds it once.

The metric reads each answer as Gym's `inference_provider` model server does: the message's `content`, with
`reasoning_content` dropped, since the Gym route runs without a reasoning parser. Both routes score identical input.

## Run it

Prepare the Gym rows as in `../finance_gym/README.md` (steps 2 and 3: `convert_rows.py`, and the image FileSet for
OCR rows). Then:

1. Write the tasks:

   ```bash
   python3 make_tasks.py rows/visual_ocr.jsonl rows/reasoning.jsonl rows/code_generation.jsonl \
     --shard-size 100 --output finance-harbor
   ```

2. Write `run.json`. `<api-service>` is the API's in-cluster address and `<model>` the model's name in the gateway:

   ```json
   {
     "spec": {
       "target": {
         "kind": "harbor",
         "source": {
           "import_path": "nhx_evals_sdk.agent_eval.runtimes.harbor.dataset_runner:DatasetRunnerAgent",
           "model_name": "<model>"
         },
         "env_vars": {
           "DATASET_RUNNER_BASE_URL": "http://<api-service>:8080/apis/inference-gateway/v2/workspaces/default/model/<model>/-/v1"
         },
         "agent_kwargs": {"max_concurrency": 8},
         "n_attempts": 1
       }
     },
     "execution": {"max_concurrent_task_jobs": 3}
   }
   ```

3. Submit, watch, download and report:

   ```bash
   nemo evals runs submit --workspace default --harbor-dir finance-harbor --spec-file run.json
   nemo evals runs status "$RUN_ID" --workspace default --watch
   nemo evals runs download "$RUN_ID" --workspace default
   python3 harbor_report.py "$RUN_ID" --output report.json
   ```

`harbor_report.py` pools every shard's `row_scores.jsonl`, so its means are over rows, not shards.

## Limits of this prototype

- The runner's endpoint comes from `env_vars`, and its key, if any, from `env_secrets` as `DATASET_RUNNER_API_KEY`.
  The gateway's per-trial keys, closed-book checks and trial recording cover built-in agents only today.
- A no-network task's sandbox still gets the profile's agent channel, which this runner doesn't need.
- A retry re-runs the whole shard, model calls included.
