<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Job Surface (NemoJob)

A `NemoJob` is a unit of work you can execute locally, submit to a cluster, or introspect — the same class drives all three. The platform auto-generates three CLI verbs per job: `run`, `submit`, `explain`.

```
nemo <plugin> <job> run      [--spec '{...}' | --spec-file FILE]
nemo <plugin> <job> submit   [--profile <p>] [--cluster <c>] \
                             [--spec '{...}' | --spec-file FILE] \
                             [-o <backend>.<key>=<value> ...] [--options-file FILE]
nemo <plugin> <job> explain  [--profile <p>]
```

- `run` — executes `job.run()` in-process. No platform needed.
- `submit` — POSTs the job to the plugin service, which compiles it into a `PlatformJobSpec` and hands it off to the Jobs service for cluster execution.
- `explain` — prints the job's schemas and submit route. Reads locally, no network.

## Declaring a NemoJob

```python
from typing import ClassVar
from nemo_platform_plugin.job import NemoJob
from pydantic import BaseModel


class GenerateSpec(BaseModel):
    num_records: int
    model: str
    seed: int | None = None


class GenerateJob(NemoJob):
    name: ClassVar[str] = "generate"          # suffix only — NOT "data-designer.generate"
    description: ClassVar[str] = "Generate synthetic rows."
    spec_schema: ClassVar[type[BaseModel]] = GenerateSpec
    container: ClassVar[str] = "cpu-tasks"

    def run(self, config: dict) -> dict:
        cfg = GenerateSpec.model_validate(config)
        return {"rows": _generate(cfg)}

    @classmethod
    async def compile(cls, *, workspace, spec, entity_client, job_name, sdk, profile=None, options=None):
        # See the Compilation reference for full details.
        ...
```

Every job must declare `spec_schema`. Every job that participates in `submit` must override `compile()`. Running locally only requires `run()`.

### Method colours

`NemoJob` mixes sync and async methods. Each method's colour is fixed by where it executes:

- `to_spec`, `compile` — `async classmethod`. Run in the API process (plugin service); async I/O is the right default.
- `run`, `report_progress` — sync `def`. Run in the task container, where there is no event loop and most work calls into sync library protocols.

Plugin authors never make a class-level sync/async choice.

## Entry-point key

Format: `"<plugin-name>.<job-name>"`, dot-separated.

```toml
[project.entry-points."nemo.jobs"]
"data-designer.generate" = "nemo_data_designer.jobs.generate:GenerateJob"
```

`NemoJob.name` is the **suffix** after the dot (`"generate"` above — not the full key).

## Submitter-facing input vs. canonical spec

A job can accept a different shape from what `compile()` sees internally — useful when the submitter passes names but the compiler needs resolved IDs. Declare `input_spec_schema` and override `to_spec()`:

```python
class InputSpec(BaseModel):
    model_name: str          # submitter types this

class CanonicalSpec(BaseModel):
    model_id: str            # compiler needs this

class TrainJob(NemoJob):
    name = "train"
    spec_schema = CanonicalSpec
    input_spec_schema = InputSpec

    @classmethod
    async def to_spec(cls, input_spec, *, workspace, entity_client, sdk):
        model = await entity_client.get(Model, name=input_spec.model_name, workspace=workspace)
        return CanonicalSpec(model_id=model.id)

    @classmethod
    async def compile(cls, *, workspace, spec, ...):
        # spec is a CanonicalSpec instance here
        ...
```

When `input_spec_schema` isn't declared, `to_spec()` is the identity — `spec_schema` covers both sides.

## Mounting the routes — `add_job_routes`

The plugin service mounts per-job endpoints in one line:

```python
from nemo_platform_plugin.jobs.routes import add_job_routes
from .jobs.generate import GenerateJob

router = add_job_routes(GenerateJob)
app.include_router(
    router,
    prefix="/v2/workspaces/{workspace}",
)
```

`add_job_routes(job_cls)` derives `service_name`, `job_type`, `job_input`, `job_output`, the job collection path, and the `to_spec` / `compile` adapters from the class. It returns a standard `APIRouter` from the underlying `job_route_factory` — same POST/GET/LIST/DELETE/results surface as before, just without the boilerplate. By default, `GenerateJob.name = "generate"` maps to `/jobs/generate`.

Passthrough kwargs: `route_options`, `job_result_routes`, `generate_job_name`, `default_profile`.

## Compilation

`compile` is an `async classmethod` —
`compile(workspace, spec, entity_client, job_name, sdk, profile, options) -> PlatformJobSpec` —
that turns the validated spec into the concrete step / container / resources description that the Jobs service executes.

> **Placeholder — full compilation reference is covered separately.**
> Topics to cover there: `PlatformJobSpec` shape, `PlatformJobStep`, `ContainerSpec`, `ResourcesSpec`, environment variables, how `profile` and `options` flow into per-step settings, result serializers.

## Container context

When the platform runs a compiled job step in a container, it sets these env vars — read them from inside `run()` using helpers from `nmp.common.jobs.config`:

```python
from nmp.common.jobs.config import get_task_config, get_job_id, get_workspace

class MyJob(NemoJob):
    name = "heavy-compute"
    spec_schema = MySpec
    container = "gpu-tasks"

    def run(self, config: dict) -> dict:
        cfg = get_task_config(MySpec)   # reads NEMO_JOB_STEP_CONFIG_FILE_PATH
        return {"job_id": get_job_id(), "workspace": get_workspace(), "status": "done"}
```

Storage env vars:

- `NEMO_JOB_EPHEMERAL_TASK_STORAGE_PATH` — per-step scratch space
- `NEMO_JOB_PERSISTENT_JOB_STORAGE_PATH` — shared across job steps
- `NEMO_JOB_STEP_CONFIG_STORAGE_PATH` — config files (read-only)

Container image keys: `"cpu-tasks"` (default) or `"gpu-tasks"`. Ignored for local `run`.

## `run()` contract

`run()` stays synchronous with a plain `dict` input / output — the same signature every existing job uses today. Validate internally with Pydantic:

```python
def run(self, config: dict) -> dict:
    cfg = MySpec.model_validate(config)
    return {"status": "done", "result": _do(cfg)}
```

For async work, wrap with `asyncio.run()`:

```python
def run(self, config: dict) -> dict:
    return asyncio.run(self._async_run(config))
```

Do **not** define `async def run()`. Jobs are stateless — a fresh instance is created per call.

## Testing

Jobs have no platform dependency — instantiate and call:

```python
def test_generate_defaults():
    result = GenerateJob().run({"num_records": 10, "model": "gpt-oss-120b"})
    assert len(result["rows"]) == 10
```

See [`plugin-testing` skill](../.agents/skills/plugin-testing/SKILL.md) for service-route and container-scope testing patterns.

---

## CLI Surface (NemoCLI)

```python
from nemo_platform_plugin.cli import NemoCLI
import typer

class NemoCLI(_NamedPlugin):
    name: ClassVar[str]              # REQUIRED — becomes `nemo <name>`
    description: ClassVar[str] = "" # shown in CLI help

    @abstractmethod
    def get_cli(self) -> typer.Typer: ...  # MUST implement
```

The platform calls `get_cli()` once at startup and mounts the result as `nemo <name> <command>`.

### Organizing commands

```python
from nemo_platform_plugin.cli_state import resolve_cli_workspace

def get_cli(self) -> typer.Typer:
    app = typer.Typer(help=self.description, no_args_is_help=True)

    @app.command(rich_help_panel="Local (no platform required)")
    def invoke(config_file: str = typer.Argument(...)) -> None:
        ...

    @app.command(rich_help_panel="Platform-managed")
    def create(name: str = typer.Argument(...)) -> None:
        ...

    return app
```

### Nested command groups

```python
def get_cli(self) -> typer.Typer:
    app = typer.Typer(name="agents", help=self.description, no_args_is_help=True)

    deps_app = typer.Typer(name="deployments", help="Manage deployments.", no_args_is_help=True)

    @deps_app.command()
    def list(typer_ctx: typer.Context, workspace: str | None = typer.Option(None)) -> None:
        workspace = resolve_cli_workspace(typer_ctx, workspace)
        ...

    app.add_typer(deps_app, rich_help_panel="Platform-managed")
    return app
```

Default ``--workspace`` to ``None`` and resolve it through
:func:`~nemo_platform_plugin.cli_state.resolve_cli_workspace`, never to a
literal ``"default"``. A literal default is indistinguishable from an omitted
flag, so the command silently discards the workspace the user selected with
``nemo config use-context`` (or ``$NMP_WORKSPACE``) and writes to the wrong
workspace. The generated commands already do this for you.

### Environment-based defaults

```python
base_url: str = typer.Option("http://localhost:8080", envvar="NMP_BASE_URL")
```
