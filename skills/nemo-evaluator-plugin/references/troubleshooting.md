# Evaluation Troubleshooting

The plugin CLI surface is `nemo evaluator`. In a repository checkout, prefix
the commands below with `uv run`.

## Inspect the installed contracts

```bash
nemo evaluator info
nemo evaluator metric-types
nemo evaluator evaluate explain
nemo evaluator agent-evaluate explain
```

## Common failures

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `No such command 'evaluation'` | The legacy generated CLI group is not the plugin surface | Use `nemo evaluator ...` |
| Guidance or `--help` references a local plugin `run` verb | That execution path is being retired; `client.evaluator.run()` is already gone | Use `submit`, or the standalone SDK for local iteration |
| Agent-eval metric fails every trial with a missing template key | The metric uses the dataset-driven `item.*` context in a task-driven run | Use `inputs.*`, `reference.*`, `task.*`, `trial.*`, or `sample.output_text` |
| Metric validation reports a missing required output | `compute_scores` omitted an output whose spec defaults to `required=True` | Emit the output on every scoreable trial, or set `required=False` only when absence means unmeasured or not expected on every trial |
| An optional output has `missing > 0` or `nan_count > 0` | The output was omitted, the metric or trial failed, or an emitted value was non-finite | Compare `missing` and `failed` coverage, then inspect score diagnostics and the effective `count`; do not fill omissions with zero |
| Metric bundle rejects `required` or `bundle_format_version` | `required` is not Boolean, the bundle format is not v1, or the reader is an evaluator release that predates optional outputs (`outputs.N.required: Extra inputs are not permitted`) | Regenerate with `bundle_metric`; it omits `required=True` and writes `required=False` explicitly. If the error names an extra `required` input, upgrade the service or job image instead |
| Spec validation error | Fields do not match the current job schema | Run the matching `explain` command and validate against the spec class before submission |
| Dataset row has missing fields | Jinja templates or `field_mapping` do not match row keys | Inspect one row and every referenced template before rerunning |
| Standalone model/agent authentication fails | `api_key_secret` names a platform secret instead of an environment variable, or the variable is unset | Use the name of a populated local environment variable |
| Remote submission returns 409 | The response may describe a missing platform secret, not a duplicate job | Read the response body and verify the workspace secret |
| Built-in metric bundle contains cloudpickle | A legacy or explicit packager was used | Regenerate with `InlineMetricBundlePackager` or the current default |
| `cloudpickle metric payload was created with Python ...` (HTTP 422) | The bundle was created with a different Python major/minor runtime | For a built-in metric, regenerate the checked inline JSON spec; for an intentional custom metric, recreate the bundle with the worker's Python major/minor version |
| Custom metric submission rejects the default packager | Shipping custom code requires explicit opt-in | Pass `HybridMetricBundlePackager()` (preferred) or `CloudpickleMetricBundlePackager()` |
| `ModelRef` fails with the standalone SDK | Model references are resolved by the platform submission path | Use a concrete `Model` with the standalone SDK or use `submit` with `ModelRef` |
| Fileset evaluation cannot load data | The reference, fragment, or workspace is wrong | Verify the `FilesetRef` and access it through the same workspace |
| Result download fails while progress shows 100% | Metric progress finished before the platform job finalized artifacts | Call `job.wait_until_done()` before `get_result()` or `download_artifacts()` |
| `AttributeError` on `get_result()` or `download_artifacts()` after `submit(tasks=...)` | A taskset submission returns `AgentEvaluatorJobResource`, which publishes agent-eval results rather than row scores and carries neither method | Wait with `job.wait_until_done()`, then read scores through `client.evaluator.agent_eval_results` |
| `TypeError` naming `config`, `field_mapping`, `prompt_template`, or `metric_bundle_packager` on `submit(tasks=...)` | Those configure a *row* evaluation; a taskset run is configured by its runner | Drop them and configure the runner passed as `target` |
| `UnsubmittableRunnerError` for a non-Gym runner | Only a Gym runner has a wire form today | Write the job input by hand with the matching runner target and submit it |
| `UnsubmittableRunnerError` for a Gym runner | A `hydra_params` value has no JSON form; a hand-built target or CLI payload cannot carry it either | Replace the value with something JSON-representable, or run in-process with `AgentEvaluator()` |
| Agent-eval rejects the spec | Both or neither of `target` and `trials` were provided | Provide exactly one |
| Runner target fails to start | The runtime dependency, CLI, config, credentials, or Docker access is missing | Check the selected Fabric, Harbor, or Gym runner prerequisites. Gym resolves the `gym` CLI from `PATH` only, and installs into its own environment because it requires Ray |
| Gym environment FileSet requires sandboxed execution | A `GymRunnerTarget.environment` was submitted to a deployment that runs Gym colocated | Enable `sandboxed_gym_default` and configure the sandbox prerequisites, or omit the environment FileSet |
| Sandboxed Gym reports a missing capability, runtime image, PVC, or egress route | The Evaluator deployment cannot provision a usable Gym host | Configure `sandbox_cluster_capable`, `sandbox_runtime_image`, `sandbox_job_storage_pvc_claim`, and at least one model or additional egress destination |
| FileSet-backed Gym reports a PVC mismatch | The Jobs execution profile stages onto a different claim from the OpenSandbox host | Set `sandbox_job_storage_pvc_claim` to the execution profile's job-storage PVC |
| Sandboxed Gym rejects a credential-shaped `env_vars` entry | Plaintext credentials would be readable by environment code | Store the value in NeMo Platform Secrets and map it through `GymRunnerTarget.env_secrets` |
| Gym environment FileSet is invalid | The FileSet has the wrong purpose, lacks a root `nemo-environment.yaml`, or violates the `native-v1` / `wheels-v1` layout | Use `purpose=environment`, upload the directory contents at the FileSet root, and fix the named manifest or package error |
| Sandboxed Gym returns no rollout for the selected agent | The environment registers the agent under a different instance name | Set `GymRunnerTarget.agent_ref_name` to the registered instance |

## Debug in the smallest scope

1. Validate one expected pass and one expected failure.
2. Inspect row scores or task trials before aggregates.
3. Reproduce metric behavior with the standalone SDK before diagnosing platform infrastructure.
4. For submitted jobs, inspect terminal status and error details.
5. Retry only the failed row, task, or runner configuration when possible.
