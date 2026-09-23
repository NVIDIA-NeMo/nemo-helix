---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: plugin-job
description: Creates schedulable NemoJob surfaces for NeMo Helix plugins. Use when adding a job, declaring spec_schema / input_spec_schema / to_spec / compile, mounting job routes with add_job_routes, understanding the three CLI verbs (run / submit / explain), or running jobs in containers. Trigger keywords - job, NemoJob, spec_schema, input_spec_schema, to_spec, compile, add_job_routes, nemo_helix_plugin.jobs, three verbs, run, submit, explain, NemoJobScheduler.
---

# Plugin Jobs (NemoJob)

A `NemoJob` drives three CLI verbs that the platform auto-generates from the class:

```
nemo <plugin> <job> run      [--spec '{...}' | --spec-file FILE]
nemo <plugin> <job> submit   [--profile <p>] [--cluster <c>] \
                             [--spec '{...}' | --spec-file FILE] \
                             [-o <backend>.<key>=<value> ...] [--options-file FILE]
nemo <plugin> <job> explain  [--profile <p>]
```

`run` is in-process (no platform); `submit` POSTs to the plugin service, which compiles the spec and hands it off to the Jobs service for cluster execution; `explain` prints the schemas locally.

## Class Signature

```python
from typing import ClassVar
from nemo_helix_plugin.job import NemoJob
from pydantic import BaseModel

class GenerateSpec(BaseModel):
    num_records: int
    model: str

class GenerateJob(NemoJob):
    name: ClassVar[str] = "generate"                     # suffix only — NOT "data-designer.generate"
    description: ClassVar[str] = "Generate rows."
    spec_schema: ClassVar[type[BaseModel]] = GenerateSpec
    container: ClassVar[str] = "cpu-tasks"

    def run(self, config: dict) -> dict:
        cfg = GenerateSpec.model_validate(config)
        return {"rows": _generate(cfg)}

    @classmethod
    async def compile(cls, *, workspace, spec, entity_client, job_name, sdk, profile=None, options=None):
        # See Compilation section.
        ...
```

Required: `name`, `spec_schema`, `run()`. Required if the job is remote-capable: `compile()`. Optional: `description`, `container`, `input_spec_schema`, `to_spec()`, `job_collection_path`.

### Method colours

`NemoJob` mixes sync and async methods. Each method's colour is fixed by where it executes — plugin authors never make a class-level sync/async choice:

| Method | Colour | Where it runs |
|---|---|---|
| `to_spec`, `compile` | `async classmethod` | API process (plugin service) |
| `run`, `report_progress` | `def` (sync) | Task container |

`name` is a `ClassVar[str]`; empty value raises `TypeError` at class-definition time. Container keys: `"cpu-tasks"` (default), `"gpu-tasks"`. The container is used for remote execution; ignored for local `run`.

## Entry-Point Key Format

Dot-separated: `"<plugin-name>.<job-name>"`.

```toml
[project.entry-points."nemo.jobs"]
"data-designer.generate" = "nemo_data_designer.jobs.generate:GenerateJob"
```

`NemoJob.name` is the **suffix** after the dot. For `"data-designer.generate"` the class sets `name = "generate"`.

## Submitter-facing vs. canonical spec

When the submitter's input shape differs from what `compile()` needs (e.g., names vs resolved IDs), declare both schemas and override `to_spec()`:

```python
class InputSpec(BaseModel):
    model_name: str

class CanonicalSpec(BaseModel):
    model_id: str

class TrainJob(NemoJob):
    name = "train"
    spec_schema = CanonicalSpec
    input_spec_schema = InputSpec

    @classmethod
    async def to_spec(cls, input_spec, *, workspace, entity_client, sdk):
        model = await entity_client.get(Model, name=input_spec.model_name, workspace=workspace)
        return CanonicalSpec(model_id=model.id)
```

When `input_spec_schema` isn't declared, `to_spec()` defaults to the identity and the submitter shape is the canonical shape.

## Mounting job routes — `add_job_routes(job_cls)`

This is the one-liner that mounts submit/list/get/delete endpoints on your plugin service:

```python
from nemo_helix_plugin.authz import AuthzScope
from nemo_helix_plugin.jobs.routes import add_job_routes
from fastapi import FastAPI
from .jobs.generate import GenerateJob

app = FastAPI()
router = add_job_routes(GenerateJob, authz=AuthzScope("data-designer"))
app.include_router(
    router,
    prefix="/v2/workspaces/{workspace}",
)
```

> **Pass `authz=` — it is required.** Without it the job routes are *unruled*, and under the default `on_invalid_plugin=hard_fail` the auth service refuses to build the OPA bundle (the platform 502s) rather than silently fence them. Give it the plugin's `AuthzScope` — the same object the plugin's other routes use — and the factory mints the submit/list/get/delete permissions and scope gate from it. See the `plugin-authz` skill for the full rule model.

`add_job_routes` derives `service_name` / `job_type` / `job_input` / `job_output` / `input_to_output` / `platform_job_config_compiler` from the `NemoJob` subclass. It also derives the job collection path from `job_collection_path` or `/jobs/{NemoJob.name}`, so `GenerateJob.name = "generate"` maps to `/jobs/generate`. Returns a standard `APIRouter` — same routes as the legacy `job_route_factory`.

Passthrough kwargs:

- `authz: AuthzScope` — the scope whose permissions and OAuth scope-gate are minted onto the job routes. Required in practice (see the note above); it defaults to `None`, which leaves the routes unruled.
- `route_options: list[JobRouteOption]` — which standard routes to enable.
- `job_result_routes: list[HelixJobResultRoute]` — custom result download endpoints.
- `generate_job_name: Callable[..., str]` — name generator for unnamed submissions.
- `default_profile: str = "default"` — profile stamped onto compiled steps when the plugin's `compile()` didn't set one.

## Compilation

`compile` is an `async classmethod` —
`compile(workspace, spec, entity_client, job_name, sdk, profile, options) -> HelixJobSpec` —
that turns the validated spec into the concrete step / container / resources description that runs in the cluster.

> **Placeholder — full compilation reference is handled separately.**
> Covers: `HelixJobSpec`, `HelixJobStep`, `ContainerSpec`, `ResourcesSpec`, env var plumbing, how `profile` and `options` flow into per-step settings, result serializers.

## Submit body wire shape (informational)

When the CLI runs `nemo ... submit`, the body POSTed to the plugin service looks like:

```json
{
  "name": "my-run",
  "description": "...",
  "project": "...",
  "spec":    { "num_records": 1000, "model": "gpt-oss-120b" },
  "profile": "research",
  "options": { "slurm": { "partition": "gpu-long" } }
}
```

The plugin service handler (generated by `add_job_routes`) validates `spec` against `input_spec_schema` or `spec_schema`, runs `to_spec()`, calls `compile()`, stamps the profile onto each step, and hands the `HelixJobSpec` to the Jobs service.

## `run()` contract

- **Input**: plain `dict` (validated internally with Pydantic).
- **Output**: plain `dict`, JSON-serializable (no datetime, no Pydantic models).
- **Stateless**: a fresh instance per call.
- **Synchronous**: do NOT define `async def run()`.

```python
from pydantic import BaseModel, model_validator

class GenerateSpec(BaseModel):
    num_records: int
    model: str

    @model_validator(mode="after")
    def check_positive(self) -> "GenerateSpec":
        if self.num_records <= 0:
            raise ValueError("num_records must be positive")
        return self

class GenerateJob(NemoJob):
    name = "generate"
    spec_schema = GenerateSpec

    def run(self, config: dict) -> dict:
        cfg = GenerateSpec.model_validate(config)
        return {"rows": _generate(cfg)}
```

Wrap async work with `asyncio.run()`:

```python
def run(self, config: dict) -> dict:
    return asyncio.run(self._async_run(config))
```

## Programmatic invocation

```python
from nemo_helix_plugin.discovery import discover_jobs
from nemo_helix_plugin.scheduler import NemoJobScheduler

job_cls = discover_jobs()["data-designer.generate"]
result = NemoJobScheduler().run_local(
    job_cls,
    {"num_records": 10, "model": "gpt-oss-120b"},
)
```

`run_local` is synchronous. It drives `NemoJob.to_spec` once via
`asyncio.run` to produce the canonical spec, then calls `run` directly.
`KeyError` if no job is registered under the key.

## In Job Containers

When the platform runs a compiled step in a container, it sets env vars:

```python
from nhx.common.jobs.config import get_task_config, get_job_id, get_workspace

class GenerateJob(NemoJob):
    name = "generate"
    spec_schema = GenerateSpec
    container = "gpu-tasks"

    def run(self, config: dict) -> dict:
        cfg = get_task_config(GenerateSpec)   # reads NEMO_JOB_STEP_CONFIG_FILE_PATH
        job_id = get_job_id()                  # NEMO_JOB_ID
        workspace = get_workspace()            # NEMO_JOB_WORKSPACE
        return {"job_id": job_id, "status": "completed"}
```

Storage env vars:

- `NEMO_JOB_EPHEMERAL_TASK_STORAGE_PATH` — per-step scratch space.
- `NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH` — shared across job steps.
- `NEMO_JOB_STEP_CONFIG_STORAGE_PATH` — config files (read-only).

## Testing Jobs

Jobs have no platform dependency — instantiate and call `run()` directly:

```python
def test_generate_defaults():
    result = GenerateJob().run({"num_records": 10, "model": "gpt-oss-120b"})
    assert len(result["rows"]) == 10
```

No mocking, no FastAPI, no platform needed. See the `plugin-testing` skill for service-route and container-scope patterns.

## Gotchas

- **`name` is the suffix only**: For entry-point key `"data-designer.generate"`, `NemoJob.name = "generate"`. Setting the full key logs a warning at startup.
- **`spec_schema` is required**: `add_job_routes` raises `TypeError` if `spec_schema` is `None`. Declare it before mounting routes.
- **`compile()` default raises `NotImplementedError`**: Jobs that haven't overridden it fail at `submit` time with a clear 422. Local `run` doesn't need `compile`.
- **`run()` must be synchronous**: Use `asyncio.run()` for async work inside.
- **`run()` returns a JSON-serializable dict**: No datetime, no Pydantic models, no custom classes.
- **Jobs are stateless**: A new instance is constructed per call; don't store state on `self`.
