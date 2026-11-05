# Harbor on Kubernetes

`nemo evals runs` runs a Harbor task suite on a NeMo Helix deployment on Kubernetes: one Kubernetes Job and
one OpenSandbox sandbox per task, one run ID, one merged result bundle. In a repository checkout, prefix the
commands with `uv run`.

## Preconditions

- A CLI context for the deployment; `WS` names a workspace the user can write to.
- The deployment has OpenSandbox and Builder enabled (`sandboxClusterCapable`, `builder.enabled`). If submit
  says Kubernetes runs aren't configured, stop and tell the user to ask their cluster admin.
- Sandbox profile: omit `--sandbox-profile` for the deployment's default. An unknown name returns a 422 that
  lists the valid names. There is no command that lists profiles. Admins define profiles:
  https://docs.nvidia.com/nemo-helix/latest/documentation/kubernetes-deployment/setup/helm/harbor-sandbox-profiles

## Oracle smoke run

`oracle` applies each task's reference solution and calls no model. Run it first on a new deployment.

Run this setup from the root of a source checkout. Without a checkout, set `SUITE_DIR` to an existing local
Harbor suite and skip both the temporary-directory assignment and the copy below.

```bash
SUITE_DIR="$(mktemp -d)/hello-world-suite"
cp -R packages/nhx_evals_sdk/examples/harbor/hello_world_dataset "$SUITE_DIR"
nemo evals runs submit --workspace "$WS" --harbor-dir "$SUITE_DIR" --list-tasks
nemo evals runs submit --workspace "$WS" --harbor-dir "$SUITE_DIR" --tasks 0 --agent oracle
```

- Submit publishes the tasks and a taskset, builds images, then prints `Run run-<16 hex> accepted`
  and `Rerun these tasks: nemo evals runs submit --taskset <ref> …` with the same run options, which skips the upload
  and builds.
- Task sources: `--harbor-dir`, `--harbor-hub org/name[@sha256:…]`, `--harbor-git <url>@<tag-or-commit>`,
  `--taskset ws/name#digest`. `--tasks` takes zero-based indexes or names, comma-separated.

## Watch, read and download

`RUN_ID` is the ID submit prints. `OUT_DIR` is any directory (default `./$RUN_ID`). `TASKSET` is the last line
`tasksets create` prints.

```bash
nemo evals runs status "$RUN_ID" --workspace "$WS" --watch
nemo evals runs results "$RUN_ID" --workspace "$WS"
nemo evals runs download "$RUN_ID" --workspace "$WS" -o "$OUT_DIR"
```

- Done = terminal phase (`completed`, `failed`, `cancelled`, `expired`); `result_state` is final at the same time.
- A `completed` run can have result `failed` (merge didn't finish). No task succeeded → `failed` (or `cancelled` /
  `expired`) with result `none`.
- `--watch` blocks until the run ends, longer than a shell call's timeout for model-calling agent runs. Poll instead:
  `nemo evals runs status "$RUN_ID" --workspace "$WS" --json | jq -r '.run.phase, .run.result_state'`.
- `runs results --json` has one summary per task (`.tasks[].result.summary.trial_count`), not trials.
- `runs download` writes `<OUT_DIR>/agent-eval-results/` with `report.html`; `-o` defaults to `./<run-id>`.
  Read trials with `nhx_evals_sdk.agent_eval.persistence.read_trials("<OUT_DIR>/agent-eval-results")`:
  `trial.error` is `None` on success; tokens are in `trial.measurements.prompt_tokens` / `completion_tokens`.
- Submit may print `<run> is already running <taskset>`: another run uses the same taskset. Informational.

## openhands with a closed-book model

`--agent` takes a built-in agent name only. A model-calling agent needs `--spec-file`.

1. The model: either one the deployment serves (`nemo models list --workspace "$WS" | grep <name>`; pass
   it as `<ws>/<name>` to `--default-model-entity`), or the user's own provider. For build.nvidia.com, the
   user picks a model ID from https://build.nvidia.com/models and creates a key at
   https://build.nvidia.com/settings/api-keys. For a model other than the example, replace its ID here and, in
   step 2, the entity name `nemo inference providers get nvidia-build` lists under `served_models`. Never put a key in a command or file:

   ```bash
   printenv NVIDIA_BUILD_API_KEY | nemo secrets create nvidia-build-api-key --workspace "$WS" --from-file -
   nemo inference providers create nvidia-build --workspace "$WS" \
     --host-url https://integrate.api.nvidia.com --api-key-secret-name nvidia-build-api-key \
     --enabled-models nvidia/nemotron-3.5-super-vl-120b-a12b \
     --default-extra-body '{"chat_template_kwargs": {"enable_thinking": true}}'
   ```

2. A closed-book virtual model that records trials. Closed-book is required for no-network tasks (all
   DeepSWE tasks); without it submit returns a 422 naming those tasks. Trial recording puts each trial's
   model calls and token usage in Intake; without it the job logs a warning and runs unrecorded. Closed-book
   comes last in `request_middleware`:

   ```bash
   nemo inference virtual-models create nemotron-super-closed-book --workspace "$WS" \
     --default-model-entity "$WS/nvidia-nemotron-3-5-super-vl-120b-a12b" \
     --request-middleware '[{"name": "nemo-evals-trial-recording", "config_type": "trial_recording", "config": {}},
                            {"name": "nemo-evals-closed-book", "config_type": "closed_book", "config": {}}]' \
     --response-middleware '[{"name": "nemo-evals-trial-recording", "config_type": "trial_recording", "config": {}}]'
   ```

   With platform auth off, this model returns 401 to calls that don't carry an evaluation's credential. To
   chat with the model, call the model entity instead. Closed-book holds against an agent that tries to get
   around it only with platform auth on: with auth off, an agent can forge its credential or call another model.

3. Platform auth on: an inference-scoped gateway key in a secret. Submit checks for it; trials don't use it.
   Each trial gets its own Scoped Access Key, created and revoked by the job, valid 24 hours from the task
   Job's start, so the deployment must have Scoped Access Keys enabled. Platform auth off: skip this and
   remove `env_secrets` from the spec.

   ```bash
   nemo auth access-keys create --name evals-inference-key --scope inference --expires-in 604800 \
     | nemo secrets create evals-inference-key --workspace "$WS" --from-file -
   ```

4. Copy `assets/specs/harbor_taskset_run_openhands.json` and set `source.model.ref` to the closed-book model in
   your workspace, `<WS>/nemotron-super-closed-book` (the file says `default/…`; JSON doesn't expand `$WS`).
   `agent_kwargs.temperature` and `top_p` set sampling; without them openhands uses 0.0 and 1.0 (greedy). For
   another model, ask the user for its provider's recommended values.
   Create the Experiment, then submit. The task source flag supplies `spec.tasks`:

   ```bash
   nemo experiments create harbor-runs --workspace "$WS" --exist-ok
   nemo evals runs submit --workspace "$WS" --harbor-hub datacurve/deep-swe \
     --tasks anko-default-function-arguments,geo-shapeindex-serialization \
     --spec-file harbor_taskset_run_openhands.json --experiment harbor-runs
   ```

   Drop `--experiment` to skip Intake; otherwise create the Experiment first (see [Experiment](#experiment)).

- `execution` is required in the spec; `{}` takes the deployment's defaults.
- With `--spec-file`, `--sandbox-profile`, `--cpus`, `--memory-mb` and `--storage-mb` are rejected: set
  `execution.sandbox` in the file.
- Expect 5–20 minutes for DeepSWE tasks with openhands.
- openhands reads its gateway key from `LLM_API_KEY`; evals sets `LLM_BASE_URL` and the model name from `model.ref`.
- Harbor's built-in agents aren't guaranteed to work with every agent release, model or Harbor version. Try a new
  combination on one task first, and pin `agent_kwargs.version` once it works.

## Publish once, run many times

With `jq` installed, capture the published taskset and its source Intake label:

```bash
TASKSET_INFO="$(nemo evals tasksets create --workspace "$WS" --harbor-hub datacurve/deep-swe --tasks 0,1,2 --json)"
TASKSET="$(printf '%s\n' "$TASKSET_INFO" | jq -er '.taskset_ref')"
INTAKE_DATASET="$(printf '%s\n' "$TASKSET_INFO" | jq -er '.intake_dataset')"
nemo evals tasksets build-images "$TASKSET" --wait
nemo evals runs submit --workspace "$WS" --taskset "$TASKSET" --spec-file harbor_taskset_run_openhands.json \
  --experiment harbor-runs --intake-dataset "$INTAKE_DATASET"
```

- Keep these values for recovery. Without `--json`, `tasksets create` prints the taskset ref last and
  `Intake label: --intake-dataset …` for Hub and Git sources; copy that complete label, not the taskset digest.
- `--intake-dataset` is an Intake label on the run's Evaluation. It requires `--taskset` and `--experiment`.

## Experiment

```bash
nemo experiments create harbor-runs --workspace "$WS" --exist-ok
```

Add `--experiment harbor-runs` to the submit. One Evaluation per run, named after the run ID; each task's
trial appears when its task Job finishes.

- Confirm: submit prints `Intake: Evaluation <run-id> in Experiment harbor-runs`, and
  `nemo experiments get harbor-runs --workspace "$WS"` shows `evaluation_count` going up.

## Failures and retries

- No failure threshold flag. A run ends `completed` if at least one task succeeded, `failed` if none did,
  unless a cancel or the deadline stopped it first: then `cancelled` or `expired`. A run whose stored plan
  can't be read ends `failed` with a `plan_unreadable: …` error, even if tasks succeeded: ask the operator.
- Trial errors (agent install fails, agent crashes) are stored on the trial; the task Job still succeeds.
  `max_retries` in the spec reruns an errored trial up to that many times, each in a new sandbox.
- Failed task Jobs aren't rerun. Find them with
  `nemo evals runs tasks "$RUN_ID" --workspace "$WS" --phase failed --page-size 1000` (if it ends with
  `next: --page N`, list that page too), then resubmit those indexes with `--taskset <taskset_ref>`;
  `runs status --json` has `taskset_ref`.
  This starts a new run; use a fresh key if the spec pins an `idempotency_key`.
- Same `--idempotency-key` and identical request → the existing run. Same key, changed request → HTTP 409.
  After an unclear submit failure, use the already-published taskset, the original settings and publication
  label, the same workspace, and the key it printed. For the publish-once example above:

  ```bash
  nemo evals runs submit --workspace "$WS" --taskset "$TASKSET" --spec-file harbor_taskset_run_openhands.json \
    --experiment harbor-runs --intake-dataset "$INTAKE_DATASET" --idempotency-key '<original key>'
  ```

  For a direct-source submit, replace its source and selection flags with its published taskset reference;
  do not import a source that may have changed. Keep the original spec content, including when resupplying
  JSON that came from stdin. Ctrl-C during submit prints a full recovery command; transport/HTTP errors may
  print only a key. Retain the original settings and source label: without them an identical replay is not guaranteed.

## Full example

The DeepSWE notebook runs 10 tasks end to end with progress and charts:
https://github.com/NVIDIA-NeMo/nemo-helix/tree/main/plugins/nemo-evals/examples/deepswe_k8s_demo
