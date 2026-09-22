# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""High-level router wiring for ``NemoJob`` subclasses.

``add_job_routes(job_cls)`` is the one-line replacement for the verbose
``job_route_factory(...)`` call pattern in plugin services. It derives
``service_name`` / ``job_type`` / ``job_input`` / ``job_output`` /
``input_to_output`` / ``platform_job_config_compiler`` from the job
class's declared attributes and delegates to :func:`job_route_factory`
for the actual route generation.

Usage::

    from nemo_helix_plugin.jobs.routes import add_job_routes

    router = add_job_routes(MyJob)
    app.include_router(
        router,
        prefix="/v2/workspaces/{workspace}",
        tags=["My Plugin"],
    )

Compare to the legacy pattern this replaces::

    router = job_route_factory(
        service_name="my-plugin",
        job_type="MyJob",
        job_input=MyInputSchema,
        job_output=MyCanonicalSchema,
        input_to_output=my_to_spec,
        platform_job_config_compiler=my_compiler,
        generate_job_name=my_name_generator,
    )

``add_job_routes`` threads submitter-provided ``profile`` / ``options``
body fields through to ``NemoJob.compile``. It then applies
:func:`stamp_profile` to the compiled ``HelixJobSpec`` using the submitted
profile, or ``default_profile`` when the request did not choose one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter
from fastapi.routing import APIRoute
from nemo_helix_plugin.authz import AuthzScope
from nemo_helix_plugin.job import job_collection_path_for
from nemo_helix_plugin.jobs.api_factory import (
    JobRouteOption,
    HelixJobResultRoute,
    _submit_control_kwargs,
    job_route_factory,
)
from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError
from nemo_helix_plugin.jobs.profile import stamp_profile

if TYPE_CHECKING:
    from collections.abc import Callable

    from nemo_helix_plugin.job import NemoJob


def add_job_routes(
    job_cls: type["NemoJob"],
    *,
    service_name: str | None = None,
    route_options: list[JobRouteOption] | None = None,
    job_result_routes: list[HelixJobResultRoute] | None = None,
    generate_job_name: "Callable[..., str] | None" = None,
    default_profile: str = "default",
    authz: AuthzScope | None = None,
) -> APIRouter:
    """Mount submit/list/get/delete routes for *job_cls* on a fresh router.

    Returns an :class:`APIRouter` produced by :func:`job_route_factory`,
    rebased onto the job collection path. The caller mounts it with the
    service/workspace URL prefix (typically
    ``/apis/{api}/v2/workspaces/{workspace}``).

    Args:
        job_cls: :class:`~nemo_helix_plugin.job.NemoJob` subclass. Must declare
            ``spec_schema``; must override ``compile`` to be remote-capable.
            When ``input_spec_schema`` is also declared, :meth:`to_spec` is
            used to transform the submitted spec into the canonical shape
            that ``compile`` sees.
        service_name: Override the factory's ``service_name``. Defaults to
            the top-level module name of *job_cls*, converted from
            underscores to hyphens (e.g. ``my_plugin.jobs.widget`` →
            ``"my-plugin"``). The factory uses this in OpenAPI schema names
            and error messages.
        route_options: Which :class:`JobRouteOption` routes to enable.
            Defaults to ``[JobRouteOption.CORE]`` inside the factory.
        job_result_routes: Custom :class:`HelixJobResultRoute` entries
            for result download endpoints.
        generate_job_name: Callable invoked when the submitter didn't
            provide a ``name``. Passthrough to the factory.
        default_profile: Profile label stamped onto each step of the
            compiled ``HelixJobSpec`` when the plugin's ``compile``
            didn't set one explicitly and the submitter did not provide
            ``profile``.
        authz: The plugin's :class:`~nemo_helix_plugin.authz.AuthzScope`.
            When set, every generated route is stamped with a ``@path_rule``
            (callers :data:`~nemo_helix_plugin.authz.GENERATED_ROUTE_CALLERS`)
            and the matching read/write scope.

    Returns:
        An :class:`APIRouter` with the standard job endpoints mounted.

    Raises:
        At request time, ``compile()`` on a class that hasn't overridden
        the base marker raises :class:`NotImplementedError`; the wrapper
        catches it and re-raises as
        :class:`HelixJobCompilationError` so the factory surfaces a
        clean 422 response instead of a 500 traceback.
    """
    if job_cls.spec_schema is None:
        raise TypeError(
            f"{job_cls.__name__}.spec_schema is None; add_job_routes requires a declared spec_schema. "
            f"Set it to a Pydantic BaseModel subclass on the NemoJob class."
        )

    resolved_service_name = service_name or _derive_service_name(job_cls)
    job_type = _derive_job_type(job_cls)
    job_input = job_cls.input_spec_schema or job_cls.spec_schema
    # When input_spec_schema is declared, the stored shape is the canonical
    # spec_schema; otherwise input and stored shapes coincide.
    job_output = job_cls.spec_schema if job_cls.input_spec_schema is not None else None
    input_to_output = _adapt_to_spec(job_cls) if job_cls.input_spec_schema is not None else None

    router = job_route_factory(
        service_name=resolved_service_name,
        job_type=job_type,
        job_input=job_input,
        job_output=job_output,
        input_to_output=input_to_output,
        platform_job_config_compiler=_adapt_compile(job_cls, default_profile),
        route_options=route_options,
        job_result_routes=job_result_routes,
        generate_job_name=generate_job_name,
        authz=authz,
    )
    return _rebase_job_collection_routes(router, job_collection_path_for(job_cls))


def _rebase_job_collection_routes(router: APIRouter, collection_path: str) -> APIRouter:
    """Move factory routes from ``/jobs`` to *collection_path*.

    ``job_route_factory`` remains the low-level legacy primitive that emits a
    generic ``/jobs`` collection. ``add_job_routes`` is the NemoJob-aware
    helper, so it owns rebasing those routes onto the job type's collection
    path (``/jobs/{job_cls.name}`` by default).
    """
    rebased = APIRouter()
    for route in router.routes:
        if not isinstance(route, APIRoute):
            rebased.routes.append(route)
            continue

        path = route.path
        if path == "/jobs" or path.startswith("/jobs/"):
            path = f"{collection_path}{path[len('/jobs') :]}"

        rebased.add_api_route(
            path=path,
            endpoint=route.endpoint,
            methods=route.methods,
            name=route.name,
            response_model=route.response_model,
            status_code=route.status_code,
            tags=route.tags,
            dependencies=route.dependencies,
            summary=route.summary,
            description=route.description,
            response_description=route.response_description,
            responses=route.responses,
            deprecated=route.deprecated,
            operation_id=route.operation_id,
            response_model_include=route.response_model_include,
            response_model_exclude=route.response_model_exclude,
            response_model_by_alias=route.response_model_by_alias,
            response_model_exclude_unset=route.response_model_exclude_unset,
            response_model_exclude_defaults=route.response_model_exclude_defaults,
            response_model_exclude_none=route.response_model_exclude_none,
            include_in_schema=route.include_in_schema,
            response_class=route.response_class,
            openapi_extra=route.openapi_extra,
        )
    return rebased


# ---------------------------------------------------------------------------
# Derivation helpers
# ---------------------------------------------------------------------------


def _derive_service_name(job_cls: type["NemoJob"]) -> str:
    """Top-level package of *job_cls*, hyphen-normalised.

    Matches the convention used by
    :func:`nemo_helix_plugin.scheduler._api_segment_for`, so the factory's
    ``service_name`` and the submit URL's ``{api}`` segment stay in
    lockstep for a given plugin.
    """
    return job_cls.__module__.split(".")[0].replace("_", "-")


def _derive_job_type(job_cls: type["NemoJob"]) -> str:
    """PascalCase form of ``job_cls.name`` for OpenAPI schema names.

    ``"generate"`` → ``"Generate"``; ``"metric-eval"`` → ``"MetricEval"``;
    ``"raw_job"`` → ``"RawJob"``; ``"automodel.jobs"`` → ``"AutomodelJobs"``.

    Splits on ``.`` as well as ``-``/``_`` so a dot never reaches the generated
    schema class name (``{job_type}JobRequest``). Were a dot to survive, Pydantic
    would encode it as a ``__`` separator in the OpenAPI ref name, and schema-name
    normalization (which strips everything before the last ``__`` to drop module
    namespaces) would then collapse per-backend names like ``automodel.jobs`` and
    ``rl.jobs`` to a single ``jobsJobRequest``, making every backend's request
    body alias the first one. Folding the dot into the PascalCase name keeps the
    discriminator inside the final segment so each backend gets a distinct schema.
    """
    parts = job_cls.name.replace("_", "-").replace(".", "-").split("-")
    return "".join(part[:1].upper() + part[1:] for part in parts if part)


# ---------------------------------------------------------------------------
# Signature adapters
# ---------------------------------------------------------------------------


def _adapt_to_spec(job_cls: type["NemoJob"]) -> "Callable[..., Any]":
    """Bridge ``NemoJob.to_spec`` to the factory's ``input_to_output`` shape.

    The factory calls ``input_to_output(original, workspace, entity_client,
    job_name, sdk)`` with an :class:`AsyncNeMoHelix` in the ``sdk``
    slot. :meth:`NemoJob.to_spec` takes the same data but organises it as
    ``(input_spec, *, workspace, entity_client, async_sdk)`` (no
    ``job_name``); ``async_sdk`` matches the codebase-wide convention
    that this name carries an async client. The shim adapts and awaits
    the ``async classmethod``.
    """

    async def to_spec_adapter(
        original_spec: Any,
        workspace: str,
        entity_client: Any,
        job_name: str | None,
        sdk: Any,
    ) -> Any:
        del job_name  # NemoJob.to_spec doesn't use it (names belong to the Jobs service)
        return await job_cls.to_spec(
            original_spec,
            workspace=workspace,
            entity_client=entity_client,
            async_sdk=sdk,
            is_local=False,
        )

    return to_spec_adapter


def _adapt_compile(
    job_cls: type["NemoJob"],
    default_profile: str,
) -> "Callable[..., Any]":
    """Bridge ``NemoJob.compile`` to the factory's ``platform_job_config_compiler`` shape.

    The factory calls ``compiler(workspace, original_spec, transformed_spec,
    entity_client, job_name, sdk)`` and forwards submitter-provided
    ``profile`` / ``options`` to compilers that accept those kwargs.
    :meth:`NemoJob.compile` is an ``async classmethod`` that uses kwargs and
    accepts those submit controls. After ``compile`` returns, the adapter
    applies :func:`stamp_profile` with the submitted profile or
    ``default_profile``.

    Missing-override errors from the ``NemoJob.compile`` base marker
    become :class:`HelixJobCompilationError` so the factory's
    existing handler surfaces them as 422 instead of 500.
    """

    async def compile_adapter(
        workspace: str,
        original_spec: Any,
        transformed_spec: Any,
        entity_client: Any,
        job_name: str | None,
        sdk: Any,
        profile: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> Any:
        del original_spec  # NemoJob.compile only needs the canonical (transformed) spec
        try:
            submit_control_kwargs = cast(
                dict[str, Any],
                _submit_control_kwargs(job_cls.compile, profile=profile, options=options),
            )
            result = await job_cls.compile(
                workspace=workspace,
                spec=transformed_spec,
                entity_client=entity_client,
                job_name=job_name,
                async_sdk=sdk,
                **submit_control_kwargs,
            )
        except NotImplementedError as exc:
            raise HelixJobCompilationError(str(exc)) from exc

        stamp_profile(result, profile or default_profile)
        return result

    return compile_adapter
