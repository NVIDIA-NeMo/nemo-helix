<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# nemo-platform-plugin — Agent Instructions

`nemo-platform-plugin` is the **only package plugin authors install** to build a fully-featured NeMo Platform plugin. It re-exports all abstract base classes, configuration, entity, schema, and CLI utilities needed to contribute services, CLI commands, jobs, controllers, and more to the NeMo Platform.

## Key Concepts

- **One install, everything included**: `nemo-platform-plugin` provides `get_entity_client` and bundles the `nemo_platform` SDK module. It does not pull in `nmp-common` or platform internals. A plugin needs no other NeMo dependency.
- **Four primary surfaces**: `NemoService` (HTTP routes), `NemoCLI` (CLI commands), `NemoJob` (schedulable jobs), `NemoController` (background reconcile loop).
- **Discovery via entry-points**: Plugins register surfaces under entry-point groups (`nemo.services`, `nemo.cli`, `nemo.jobs`, `nemo.controllers`) in `pyproject.toml`. The platform scans these at startup — no code registration needed.
- **Entities for persistent state**: Use `NemoEntity` + `NemoEntitiesClient` to store plugin data in the NeMo Platform entity store. Entity types are global — use plugin-scoped names (`"my_plugin_widget"` not `"widget"`).
- **Fault isolation**: A broken plugin (import error) logs a warning and is skipped at startup — the platform continues.
- **Route authorization is mandatory**: every plugin HTTP route must carry an authz rule — `@path_rule` on the handler, or `authz=` on the job / function route factory — or the auth service refuses to build the OPA bundle (`on_invalid_plugin=hard_fail`, the default) and the platform 502s. There is no silent fencing; an unruled route fails closed.

## Always load a skill first

Before writing any plugin code, load the relevant skill. Skills contain exact import paths, working code examples, and gotchas that prevent common mistakes.

- **`creating-a-plugin`** → starting a new plugin, setting up `pyproject.toml`, registering entry-points, or asking how plugins are discovered
- **`plugin-entities`** → defining `NemoEntity` subclasses, CRUD operations, optimistic locking, filter patterns
- **`plugin-config`** → adding `NemoConfig` fields, env var naming, test overrides
- **`plugin-job`** → adding `NemoJob` surfaces, the three-verb CLI (`run` / `submit` / `explain`), `spec_schema` / `input_spec_schema` / `to_spec` / `compile`, mounting routes with `add_job_routes`, container execution
- **`plugin-service`** → adding HTTP routes with `NemoService`, `RouterSpec`, response schemas, pagination
- **`plugin-authz`** → declaring HTTP authorization on plugin routes: `@path_rule`, `AuthzScope` / `PermissionSet`, caller-kind (`PRINCIPAL` vs `SERVICE_PRINCIPAL`), the `hard_fail` bundle build, migrating off `get_authz_contribution`
- **`plugin-controller`** → background reconcile loops with `NemoController`, `on_startup()` patterns, service-principal clients
- **`plugin-platform-services`** → calling platform services (jobs, files, secrets, models, inference gateway, auth) from a plugin
- **`plugin-testing`** → writing tests for any plugin surface — entity client mocking, service route tests, job tests, config overrides

## Python Conventions

- **No `__init__.py` files**: The plugin package directory does not require `__init__.py`. Do not add one.
- **Concrete type hints**: Prefer concrete type hints over string-based ones. Do not import types under `TYPE_CHECKING` when they are runtime-available in the same package — use regular imports instead.
- **Package path**: `src/nemo_<plugin_name>/` where `plugin_name` matches the CLI entry-point key (e.g., `src/nemo_my_plugin/` for plugin `"my-plugin"`).
- **Build system**: Always use hatchling with `packages = ["src/nemo_my_plugin"]` — use the exact package directory name.
- **Run tests**: `uv run pytest`

## Available Skills

- [`creating-a-plugin`](src/nemo_platform_plugin/.agents/skills/creating-a-plugin/SKILL.md) — Creates a new NeMo plugin from scratch. Use when starting plugin development, setting up a plugin package, registering surfaces via entry points, or asking how plugins are discovered by the platform. _Trigger keywords: create plugin, new plugin, plugin setup, entry-points, plugin structure, get started, plugin discovered, entry point._
- [`plugin-entities`](src/nemo_platform_plugin/.agents/skills/plugin-entities/SKILL.md) — Defines `NemoEntity` subclasses and uses `NemoEntitiesClient` for CRUD in the NeMo Platform entity store. Use when defining a new entity type, storing plugin data in the entity store, handling optimistic locking conflicts, listing entities with filters, or building entity clients for controllers. _Trigger keywords: entity, entity store, NemoEntity, entity_type, NemoEntitiesClient, entity client, store data, optimistic lock, EntityConflictError, EntityNotFoundError._
- [`plugin-config`](src/nemo_platform_plugin/.agents/skills/plugin-config/SKILL.md) — Creates plugin configuration using `NemoConfig` with environment variables and YAML file support. Use when adding plugin configuration fields, reading config values at runtime, setting up test config overrides, or understanding the env var naming formula. _Trigger keywords: config, configuration, NemoConfig, env var, environment variable, plugin_name, NMP_CONFIG, YAML config, config override, test config._
- [`plugin-job`](src/nemo_platform_plugin/.agents/skills/plugin-job/SKILL.md) — Creates schedulable `NemoJob` surfaces for NeMo Platform plugins. Use when adding a job, declaring `spec_schema` / `input_spec_schema` / `to_spec` / `compile`, mounting job routes with `add_job_routes`, understanding the three CLI verbs (`run` / `submit` / `explain`), or running jobs in containers. _Trigger keywords: job, NemoJob, spec_schema, input_spec_schema, to_spec, compile, add_job_routes, nemo_platform_plugin.jobs, three verbs, run, submit, explain, NemoJobScheduler._
- [`plugin-service`](src/nemo_platform_plugin/.agents/skills/plugin-service/SKILL.md) — Builds HTTP service surfaces for NeMo Platform plugins using `NemoService`, `RouterSpec`, `NemoListResponse`, and `NemoFilter`. Use when adding REST API routes to a plugin, implementing CRUD endpoints, handling pagination and filtering, or testing FastAPI routes. _Trigger keywords: HTTP routes, REST API, FastAPI, CRUD, endpoint, router, NemoService, pagination, filter, list endpoint, NemoListResponse, RouterSpec._
- [`plugin-authz`](src/nemo_platform_plugin/.agents/skills/plugin-authz/SKILL.md) — Declares HTTP authorization on plugin routes with `@path_rule`, `AuthzScope`, and `PermissionSet`. Use when attaching authz rules to route handlers, picking caller kinds (`PRINCIPAL` vs `SERVICE_PRINCIPAL`), passing `authz=` to `add_job_routes` / `add_function_routes`, granting permissions with no 1:1 route via `extra_permissions` / `extra_role_permissions`, or migrating a plugin off the removed `get_authz_contribution`. _Trigger keywords: authz, authorization, path_rule, AuthzScope, PermissionSet, perm, permission, caller kind, PRINCIPAL, SERVICE_PRINCIPAL, OPA bundle, hard_fail, on_invalid_plugin, extra_permissions, extra_role_permissions, get_authz_contribution._
- [`plugin-controller`](src/nemo_platform_plugin/.agents/skills/plugin-controller/SKILL.md) — Creates background reconcile-loop controllers using `NemoController`. Use when implementing state-machine reconciliation, running periodic background work, managing deployment lifecycle, building service-principal entity clients for background use, or understanding controller startup/shutdown sequence. _Trigger keywords: controller, NemoController, reconcile, background loop, reconcile_one, list_objects, on_startup, state machine, deployment lifecycle, service principal, interval_seconds._
- [`plugin-platform-services`](src/nemo_platform_plugin/.agents/skills/plugin-platform-services/SKILL.md) — Calls NeMo Platform services (entity store, jobs, files, secrets, models, inference gateway, auth) from a plugin. Use when a plugin needs to submit jobs, access files, read secrets, look up models, call the inference gateway, check permissions, or route calls between services. _Trigger keywords: jobs service, files service, secrets service, models service, inference gateway, auth client, NeMo SDK, platform SDK, service-to-service, inter-service call, job_route_factory, NMP_BASE_URL._
- [`plugin-testing`](src/nemo_platform_plugin/.agents/skills/plugin-testing/SKILL.md) — Tests NeMo Platform plugin surfaces without a running platform. Use when writing tests for entity CRUD routes, mocking the entity client, testing `NemoJob.run()` methods, setting up config overrides, or verifying FastAPI route error handling. _Trigger keywords: test, pytest, mock entity client, TestClient, dependency_overrides, AsyncMock, test job, test config, test service, test controller._
- [`plugin-inference-middleware`](src/nemo_platform_plugin/.agents/skills/plugin-inference-middleware/SKILL.md) — Implements `NemoInferenceMiddleware` plugins for in-process inference request/response interception in IGW. Use when building a middleware plugin, implementing `process_request` or `process_response`, handling `MiddlewareCall` config (inline or `config_id`), exposing config entity CRUD APIs, or wiring up the `nemo.inference_middleware` entry-point. _Trigger keywords: inference middleware, NemoInferenceMiddleware, process_request, process_response, MiddlewareCall, config_id, ImmediateResponse, VirtualModel middleware, middleware plugin._
