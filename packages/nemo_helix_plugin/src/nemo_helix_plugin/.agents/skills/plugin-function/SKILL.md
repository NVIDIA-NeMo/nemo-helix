---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: plugin-function
description: Creates in-process NemoFunction surfaces for NeMo Helix plugins. Use when adding a function, declaring spec_schema, mounting function routes with add_function_routes, understanding the two CLI verbs (run / submit), or streaming NDJSON frames. Trigger keywords - function, NemoFunction, spec_schema, add_function_routes, nemo_helix_plugin.functions, two verbs, run, submit, streaming, NDJSON, FunctionContext.
---

# Plugin Functions (NemoFunction)

A `NemoFunction` is the third primitive on a plugin, alongside `NemoResource` and `NemoJob`. It's an in-process request handler — no scheduler, no backend dispatch — that the platform exposes as both a CLI subcommand and an HTTP route automatically.

```text
nemo <plugin> <fn> run    [--spec '{...}' | --spec-file FILE] [--workspace W] [<spec-flag>...]
nemo <plugin> <fn> submit [--spec '{...}' | --spec-file FILE] \
                          [--base-url URL | --cluster URL] \
                          [--workspace W] [--request-id ID] [<spec-flag>...]
```

`run` is local (in-process); `submit` POSTs to the plugin service's auto-derived route. Two verbs only — no `explain`. A function's only schema is `spec_schema`, and `--help` is the introspection surface.

## CLI introspection — auto-generated per-field flags

Every scalar leaf in `spec_schema` becomes a Typer flag automatically. Nested submodels recurse with dotted paths (`--target.url`, `--target.timeout-seconds`). For a function with `spec_schema = GreetSpec(name: str)`:

```text
$ nemo my-plugin greet run --help
...
Function Spec:
  --name <NAME>  Name to greet.

Spec Source:
  --spec <SPEC>            Spec as a JSON string. [default: {}]
  --spec-file <SPEC_FILE>  Path to a YAML or JSON spec file (used as base; per-flag values override).
  --workspace <WORKSPACE>  Workspace identity passed to the function as ctx.workspace. [default: default]
...
Function Spec flags are generated from the GreetSpec Pydantic schema. Precedence: --spec-file (base) → --spec JSON (overlay) → per-flag values (top).
```

Precedence at runtime is `--spec-file` (base) → `--spec` JSON (overlay) → per-field flag (top), then validation against `spec_schema` happens once on the merged value. Lists, dicts, discriminated unions, and other types the walker can't render fall through to `--spec` / `--spec-file`. The plumbing lives in `nemo_helix_plugin._spec_flags` and is shared with the jobs CLI; if a leaf is missing from `--help`, check that the field type is `str` / `int` / `float` / `bool` (or `Optional` of one of those) and that its name doesn't collide with a static verb flag.

## Class Signature

```python
from typing import ClassVar
from nemo_helix_plugin.function import NemoFunction
from pydantic import BaseModel

class GreetSpec(BaseModel):
    name: str

class GreetResponse(BaseModel):
    message: str

class GreetFunction(NemoFunction[GreetSpec]):
    name:        ClassVar[str] = "greet"
    description: ClassVar[str] = "Say hello to a name."
    spec_schema: ClassVar[type[BaseModel]] = GreetSpec

    async def run(self, spec: GreetSpec) -> GreetResponse:
        return GreetResponse(message=f"Hello, {spec.name}!")
```

Required: `name`, `spec_schema`, `async def run()`. Optional: `description`, `endpoint`.

## Method colours — `run` is **always** `async def`

`NemoFunction` is single-coloured. `run` is the request handler, so it must be `async def` to avoid blocking the FastAPI event loop while a stream is open. The base class enforces this at class-definition time:

```python
class BadFunction(NemoFunction[GreetSpec]):
    name = "bad"
    spec_schema = GreetSpec

    def run(self, spec):  # raises TypeError immediately at class definition
        return {}
```

Both shapes are accepted:

| Shape | What it returns | Wire content type |
|---|---|---|
| `async def run(self, spec)` | A value (typically a Pydantic model) | `application/json` |
| `async def run(self, spec): yield ...` | An `AsyncIterator` of frames | `application/x-ndjson` |

Pure sync work goes through `await asyncio.to_thread(...)`:

```python
async def run(self, spec: GreetSpec) -> GreetResponse:
    rendered = await asyncio.to_thread(_blocking_template_render, spec.name)
    return GreetResponse(message=rendered)
```

There is no `streaming: ClassVar[bool]` toggle — the route adapter detects the iterator return at request time.

## Streaming

Yield Pydantic models from an `async def` body:

```python
from collections.abc import AsyncIterator
from nemo_helix_plugin.functions.frames import Done, Heartbeat
from pydantic import BaseModel

class TickSpec(BaseModel):
    upto: int

class Tick(BaseModel):
    kind: str = "tick"
    n: int

class CountFunction(NemoFunction[TickSpec]):
    name = "count"
    spec_schema = TickSpec

    async def run(self, spec: TickSpec) -> AsyncIterator[BaseModel]:
        for i in range(spec.upto):
            yield Tick(n=i)
        yield Done()
```

Each yielded `BaseModel` becomes one NDJSON line (`model_dump_json() + "\n"`) on the wire. The route adapter wraps the iterator with a heartbeat injector that emits a `Heartbeat()` frame every 5 seconds of idle time so proxies don't time the connection out. Plugin authors don't need to emit heartbeats themselves — the framework adds them.

Convention frames live in `nemo_helix_plugin.functions.frames`:

| Class | `kind` | When to emit |
|---|---|---|
| `Heartbeat` | `"heartbeat"` | Framework injects on idle; plugins rarely emit explicitly |
| `Done` | `"done"` | Last frame of a successful stream |
| `Error` | `"error"` | Last frame of a failed stream |

The discriminator is the `kind` field. Use it in your own frames so consumers can branch with `jq -r 'select(.kind == "tick") | .n'` etc.

## Entry-Point Key Format

Dot-separated: `"<plugin-name>.<function-name>"`.

```toml
[project.entry-points."nemo.functions"]
"my-plugin.greet" = "nemo_my_plugin.functions.greet:GreetFunction"
```

`NemoFunction.name` is the **suffix** after the dot. For `"my-plugin.greet"` the class sets `name = "greet"`.

## Mounting function routes — `add_function_routes(fn_cls)`

This is the one-liner that mounts the function as a `POST` route on your plugin service:

```python
from fastapi import FastAPI
from nemo_helix_plugin.authz import AuthzScope
from nemo_helix_plugin.functions.routes import add_function_routes

from .functions.greet import GreetFunction

app = FastAPI()
router = add_function_routes(
    GreetFunction,
    authz=AuthzScope("my-plugin"),
    permission_description="Invoke the greet function",
)
app.include_router(
    router,
    prefix="/apis/my-plugin/v2/workspaces/{workspace}",
)
# POST /apis/my-plugin/v2/workspaces/{workspace}/greet
```

`authz=` is **required** — pass your plugin's `AuthzScope` or the route is left unruled, and under the default `on_invalid_plugin=hard_fail` the auth service refuses to build the OPA bundle (the platform 502s) rather than silently fencing the route. See the `plugin-authz` skill for the full path-rule model.

`add_function_routes`:

- Validates the body against `spec_schema` (422 on bad input).
- Builds a `FunctionContext` from the workspace path parameter and the optional `X-Request-ID` header.
- Resolves keyword-only DI parameters on `run` by name — `ctx`, `async_sdk` (sync `sdk` injection lands in a follow-up).
- Awaits `run(spec, **resolved)` and serialises the result; returns `application/x-ndjson` with heartbeat injection if the return is an async iterator.
- Stamps the route's authz from `authz=` — a `PRINCIPAL` `@path_rule` carrying a write-action invoke permission (`<namespace>.<function-name>`, e.g. `my-plugin.greet`) plus `@AuthzScope.write`. `permission_description` overrides that permission's description (it defaults to the function's `description`); passing it without `authz=` is a `ValueError`.

Optional kwarg `heartbeat_interval_seconds` overrides the 5-second default — useful in tests.

## Endpoint override

The default route path is `/{name}` mounted under the workspace prefix — i.e. `/apis/<plugin>/v2/workspaces/{workspace}/<name>`. Override the trailing segment per-class via `endpoint`:

```python
class GreetFunction(NemoFunction[GreetSpec]):
    name = "greet"
    spec_schema = GreetSpec
    # Mount at /apis/.../v2/workspaces/{workspace}/greet/v1 to keep
    # an existing client URL working.
    endpoint: ClassVar[str | None] = "/{name}/v1"
```

Only `{name}` is substituted; the workspace placeholder stays as a live FastAPI route parameter. The override doesn't let a function relocate itself outside its plugin's URL namespace — that's intentional.

## DI by signature

`run` accepts framework-managed dependencies as keyword-only parameters. The route adapter and the local CLI both resolve them by parameter name:

```python
from nemo_helix_plugin.function import NemoFunction
from nemo_helix_plugin.function_context import FunctionContext

class WhoamiFunction(NemoFunction[GreetSpec]):
    name = "whoami"
    spec_schema = GreetSpec

    async def run(
        self,
        spec: GreetSpec,
        *,
        ctx: FunctionContext,
        async_sdk: object | None = None,
    ) -> dict:
        return {
            "workspace": ctx.workspace,
            "request_id": ctx.request_id,
            "sdk_present": async_sdk is not None,
        }
```

| Parameter | Source | Notes |
|---|---|---|
| `ctx: FunctionContext` | route adapter / CLI | `workspace` from URL or `--workspace`, `request_id` from `X-Request-ID` |
| `async_sdk` | route adapter only (today) | Plugin services need `app.dependency_overrides[get_sdk_client]` to inject a real handle |
| `sdk` | declared but currently bound to `None` | Sync placeholder lands with the SDK-builder follow-up |

Parameters not declared on `run` aren't injected — there's no implicit context. Keep `run` signatures narrow.

## Programmatic invocation

```python
import asyncio
from nemo_helix_plugin.discovery import discover_functions
from nemo_helix_plugin.function_context import FunctionContext

fn_cls = discover_functions()["my-plugin.greet"]
spec = fn_cls.spec_schema.model_validate({"name": "Ada"})
result = asyncio.run(fn_cls().run(spec, ctx=FunctionContext(workspace="default")))
```

`discover_functions()` returns the typed mapping. There is no `FunctionScheduler` — the function is the runtime; instantiate and `await`.

## Testing

Unit-test functions directly — no FastAPI, no platform required:

```python
import pytest

@pytest.mark.asyncio
async def test_greet_returns_message():
    fn = GreetFunction()
    result = await fn.run(GreetSpec(name="Ada"))
    assert result.message == "Hello, Ada!"
```

Route-test with `TestClient` when you want to exercise validation, DI, and streaming:

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_helix_plugin.authz import AuthzScope
from nemo_helix_plugin.functions.routes import add_function_routes

def test_greet_route_returns_json():
    app = FastAPI()
    app.include_router(
        add_function_routes(GreetFunction, authz=AuthzScope("my-plugin")),
        prefix="/apis/my-plugin/v2/workspaces/{workspace}",
    )
    client = TestClient(app)
    response = client.post(
        "/apis/my-plugin/v2/workspaces/default/greet",
        json={"name": "Ada"},
    )
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, Ada!"}
```

For streaming functions, set a low heartbeat interval so tests aren't flaky:

```python
add_function_routes(CountFunction, heartbeat_interval_seconds=0, authz=AuthzScope("my-plugin"))
```

## Gotchas

- **`run` must be `async def`**: The class enforces this at definition time. Sync work uses `await asyncio.to_thread(...)`.
- **`name` is the suffix only**: For entry-point key `"my-plugin.greet"`, `NemoFunction.name = "greet"`. The class declares the suffix; the dot-prefix lives in `pyproject.toml`.
- **`spec_schema` is required**: `add_function_routes` raises `TypeError` if it's missing. Declare it before mounting.
- **`authz=` is required**: pass your plugin's `AuthzScope` to `add_function_routes`, or the route is unruled and the OPA bundle build fails closed under `hard_fail`. See the `plugin-authz` skill.
- **Streaming is detected from the return type, not declared**: An `async def` with `yield` is an async generator — the route emits NDJSON. An `async def` with `return` is a coroutine — the route emits JSON. Don't add a class-level streaming flag.
- **`endpoint` is the trailing segment**: Setting `endpoint = "/{name}/v1"` mounts under the workspace prefix; you can't relocate the function to a different plugin's URL namespace.
- **Reserved CLI flag**: `submit --workspace` controls the URL path segment — don't put a `workspace` field in your spec model with conflicting semantics.

For the architectural framing alongside `NemoResource` and `NemoJob`, see the shared resources/jobs/functions design note.
