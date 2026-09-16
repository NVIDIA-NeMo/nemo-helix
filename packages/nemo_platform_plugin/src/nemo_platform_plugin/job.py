# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin job interface — what plugin authors implement for schedulable jobs.

Plugin authors subclass :class:`NemoJob` and register the class under the
``nemo.jobs`` entry-point group. The platform (or SDK) instantiates each class
and invokes it through the :class:`~nemo_platform_plugin.scheduler.NemoJobScheduler`
for local execution, or POSTs it to the plugin service for remote submission.

Mental model — *Job = spec + profile + options*:

- **``spec``** — what the job should do (plugin-authored schema). The canonical
  shape ``run()`` and ``compile()`` see is :attr:`spec_schema`.
- **``profile``** — where to run it, operator-configured; a submitter picks it
  via ``--profile``.
- **``options``** — how to tune it for the resolved backend, as an opaque
  ``{"<backend>": {...}}`` bag forwarded to the Jobs service.

One class, mixed-colour methods. Each method's colour is fixed by where
it executes — plugin authors never make a class-level sync/async choice:

- :meth:`NemoJob.to_spec` and :meth:`NemoJob.compile` are
  ``async classmethod``s. They run in the API process, where async I/O
  is the right default.
- :meth:`NemoJob.run` and :meth:`NemoJob.report_progress` are sync
  ``def``. They run in the task container, where there is no event loop
  and most work calls into sync library protocols.

The scheduler runs the async lifecycle methods through a single
``asyncio.run`` at the top of :meth:`NemoJobScheduler.run_local`; the
sync ``run`` is invoked directly from the resulting canonical spec.

Example::

    # my_plugin/jobs/train.py
    from nemo_platform_plugin.job import NemoJob
    from pydantic import BaseModel

    class TrainSpec(BaseModel):
        model: str
        epochs: int = 1

    class TrainJob(NemoJob):
        name        = "train"
        description = "Fine-tune a model"
        container   = "gpu-tasks"
        spec_schema = TrainSpec

        def run(self, config: dict) -> dict:
            # config is the spec as a dict, validated server-side against spec_schema
            spec = TrainSpec.model_validate(config)
            ...
            return {"status": "completed", "model": spec.model}

    # pyproject.toml:
    # [project.entry-points."nemo.jobs"]
    # my-plugin.train = "my_plugin.jobs.train:TrainJob"

Entry-point key convention: ``<plugin-name>.<job-name>``, e.g.
``example.say-hello``. This lets
:func:`~nemo_platform_plugin.discovery.discover_jobs` resolve jobs unambiguously
across plugins; programmatic execution goes through
:meth:`nemo_platform_plugin.scheduler.NemoJobScheduler.run_local`.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any, ClassVar

from nemo_platform_plugin._base import _NamedPlugin
from nemo_platform_plugin.client.adapter import AsyncPlatformClient
from nemo_platform_plugin.job_context import JobContext
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class NemoJob(_NamedPlugin):
    """Abstract base class for plugin-contributed schedulable jobs.

    Subclasses declare their identity via class variables and implement
    :meth:`run`. The platform instantiates each class and invokes its
    lifecycle — plugin authors never manage submission themselves.

    Identity and container:

    .. attribute:: name
        :type: str

        Unique job name within the plugin (e.g. ``"train"``). Combined with
        the plugin name for the full entry-point key (``"my-plugin.train"``).

    .. attribute:: description
        :type: str

        Human-readable description shown in ``nemo plugins list`` and CLI
        help.

    .. attribute:: container
        :type: str

        Container image key for remote execution (e.g. ``"gpu-tasks"``,
        ``"cpu-tasks"``). Used by the Jobs service to pick the right
        container image for each step. Ignored for local execution.

    .. attribute:: execution_provider
        :type: str

        Default compute-workload *shape* for single-step execution. One of
        ``"cpu"``, ``"gpu"``, ``"gpu_distributed"``. Multi-step jobs that
        override :meth:`compile` decide providers per step and can ignore
        this default.

    Spec schemas:

    .. attribute:: spec_schema
        :type: type[pydantic.BaseModel] | None

        Canonical Pydantic model for the job's inputs. ``run()`` and
        ``compile()`` always see a ``spec_schema`` instance. Declaring
        this lets the scheduler validate incoming specs before
        invocation and lets ``nemo <plugin> <job> explain`` surface the
        schema over OpenAPI. ``None`` means the raw dict flows through
        unchanged — prefer a concrete model for new jobs.

    .. attribute:: input_spec_schema
        :type: type[pydantic.BaseModel] | None

        Optional submitter-facing shape when the input differs from the
        canonical spec (e.g. a user-typed dataset name that resolves to a
        URL). When present, the scheduler / plugin service parses the
        incoming spec against this schema, then calls :meth:`to_spec` to
        produce a ``spec_schema`` instance before ``run()`` / ``compile()``
        see it. When ``None``, ``spec_schema`` is used on both sides and
        :meth:`to_spec` is an identity function.

    API routes:

    .. attribute:: job_collection_path
        :type: str | None

        Optional override for the job collection path relative to
        ``/apis/{api}/v2/workspaces/{workspace}``. When omitted, the
        default is ``/jobs/{name}``. Both the plugin service route helper
        and generated CLI submission use this value, so the API route and
        CLI command stay in sync. Examples: ``None`` for ``/jobs/train``
        when ``name = "train"``; ``"/metric-jobs"`` for a legacy flat
        collection path.

    .. attribute:: generate_legacy_verbs
        :type: bool

        Temporary CLI compatibility knob. ``True`` keeps the generated
        ``<job> run|submit|explain`` command group. ``False`` registers
        ``<job>`` itself as the remote submit command and omits the legacy
        ``run``, ``submit``, and ``explain`` verbs.

    Plugin-owned options:

    .. attribute:: backend_options_schemas
        :type: dict[str, type[pydantic.BaseModel]]

        Per-backend Pydantic schemas for plugin-authored tuning knobs
        that live inside ``options.<backend>``. Currently inert —
        declared here as a stable surface so plugins can land their
        schemas before validation and OpenAPI surfacing come online.
    """

    # ------------------------------------------------------------------ #
    # Identity / container                                               #
    # ------------------------------------------------------------------ #

    name: ClassVar[str]
    description: ClassVar[str] = ""
    container: ClassVar[str] = "cpu-tasks"
    execution_provider: ClassVar[str] = "cpu"

    # ------------------------------------------------------------------ #
    # Spec schemas — canonical ``spec_schema``; optional ``input_spec_schema``
    # ------------------------------------------------------------------ #

    spec_schema: ClassVar[type[BaseModel] | None] = None
    input_spec_schema: ClassVar[type[BaseModel] | None] = None

    # ------------------------------------------------------------------ #
    # API routes                                                         #
    # ------------------------------------------------------------------ #

    job_collection_path: ClassVar[str | None] = None

    # ------------------------------------------------------------------ #
    # Temporary CLI compatibility                                        #
    # ------------------------------------------------------------------ #

    generate_legacy_verbs: ClassVar[bool] = True

    # ------------------------------------------------------------------ #
    # Plugin-owned options (inert; see class docstring)                  #
    # ------------------------------------------------------------------ #

    backend_options_schemas: ClassVar[dict[str, type[BaseModel]]] = {}

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Execute the job with the given config and return results.

        The abstract signature uses ``(*args, **kwargs) -> Any`` so concrete
        jobs can expose their precise framework-managed dependencies without
        tripping method-override checks. At runtime, the first positional
        argument is still the canonical spec dict and keyword-only parameters
        (``ctx: JobContext``, ``sdk``, ``async_sdk``, ``is_local``) are resolved
        by name via signature-based DI.

        Args:
            config: The validated spec serialised to a ``dict``. When
                :attr:`spec_schema` is declared the scheduler validates
                the submitter's input against it and passes
                ``spec.model_dump()``; otherwise the raw input flows
                through unchanged.

        Returns:
            A plain JSON-serialisable ``dict``.
        """

    @classmethod
    async def to_spec(
        cls,
        input_spec: BaseModel,
        *,
        workspace: str,
        entity_client: object,
        async_sdk: AsyncPlatformClient,
        is_local: bool,
    ) -> BaseModel:
        """Transform *input_spec* into a canonical :attr:`spec_schema` instance.

        ``async classmethod`` because it runs in the API process — where
        async I/O is the right default — and is a class-level
        transformation that doesn't depend on instance state.

        Runs on the submitter side before local ``run`` and on the plugin
        service before custom ``compile``. The default implementation is the
        identity function — valid only when :attr:`input_spec_schema` is
        ``None``, which means input and canonical shapes coincide.

        Plugin authors override this when they want the submitter to type a
        human-friendly shape (e.g. dataset name) that expands to the
        canonical form (e.g. dataset ID + URL) via SDK / entity-client
        lookups.

        Args:
            input_spec: Validated :attr:`input_spec_schema` instance.
            workspace: Workspace scope (used for entity-client scoping).
            entity_client: Entity client for resolving names to IDs.
            async_sdk: Async platform handle (``AsyncNeMoPlatform`` on the
                plugin service, ``AsyncNemoClient`` from the local CLI);
                adapt it with ``client_from_platform``. ``to_spec`` runs in
                the API process and is itself ``async``, so the framework
                only offers the async client here — the parameter name
                follows the codebase convention (``sdk`` is sync,
                ``async_sdk`` is async).
            is_local: Whether this transformation is running for local
                scheduler execution. The plugin-service route adapter passes
                ``False``.

        Returns:
            A :attr:`spec_schema` instance.
        """
        return input_spec

    @classmethod
    async def compile(
        cls,
        *,
        workspace: str,
        spec: BaseModel,
        entity_client: object,
        job_name: str | None,
        async_sdk: AsyncPlatformClient,
        profile: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> object:
        """Compile the canonical spec into a ``PlatformJobSpec``.

        ``async classmethod`` for the same reasons as :meth:`to_spec`:
        runs in the API process, no per-instance state.

        Every :class:`NemoJob` that participates in remote submission
        must override this method; the plugin service produces the
        ``PlatformJobSpec`` the Jobs service expects by invoking it.

        Args:
            workspace: Workspace scope.
            spec: Canonical :attr:`spec_schema` instance.
            entity_client: For resolving references (datasets, models, ...).
            job_name: Optional job name supplied by the submitter.
            async_sdk: Async platform handle. Same contract as
                :meth:`to_spec`: this runs in the API process so only
                the async client is offered.
            profile: Submitter-selected profile. The factory applies
                ``stamp_profile(spec, profile)`` after this method
                returns; per-step overrides set here take precedence.
            options: Opaque wire ``{"<backend>": {...}}`` bag; read keys
                defensively.

        Returns:
            A ``PlatformJobSpec`` ready for the Jobs service.

        Raises:
            NotImplementedError: Raised by the default impl so
                remote-incapable jobs fail loudly at submit time.
        """
        raise NotImplementedError(f"{cls.__name__} must override compile() to be remote-capable.")

    # ------------------------------------------------------------------ #
    # Progress reporting (virtual; default = log only)                   #
    # ------------------------------------------------------------------ #

    def report_progress(
        self,
        ctx: JobContext,
        *,
        work_done: int | None = None,
        work_total: int | None = None,
        status: str | None = None,
        details: dict[str, str] | None = None,
    ) -> None:
        """Report progress on the running job.

        The default implementation logs at INFO. Service-specific
        subclasses override to publish to the Jobs service, a callback
        URL, or another structured sink, and typically guard on injected
        ``is_local`` to no-op when there's no platform job record to
        report against.

        The four fields cover both "samples processed" / "work-unit
        counter" reporting and "status transition + free-form details"
        reporting; every field is optional so subclasses can use the
        subset that matches their sink.

        Args:
            ctx: The :class:`~nemo_platform_plugin.job_context.JobContext` for
                the running job.
            work_done: Cumulative count of work units completed.
            work_total: Total work units expected, if known.
            status: Free-form status label (``"running"``,
                ``"completed"``, ``"failed"``, ...).
            details: Free-form string-string metadata.
        """
        logger.info(
            "progress: %s/%s status=%s details=%s",
            work_done,
            work_total,
            status,
            details or {},
        )


def job_collection_path_for(job_cls: type[NemoJob]) -> str:
    """Return the API collection path for *job_cls*.

    The returned path is relative to ``/apis/{api}/v2/workspaces/{workspace}``
    and always starts with ``/``. By default this keeps the CLI command name
    and API route segment aligned: ``TrainJob.name == "train"`` maps to
    ``/jobs/train``.
    """
    path = job_cls.job_collection_path
    if path is None:
        path = f"/jobs/{job_cls.name}"
    if not isinstance(path, str) or not path.strip().strip("/"):
        raise TypeError(f"{job_cls.__name__}.job_collection_path must be a non-empty string or None")
    return f"/{path.strip().strip('/')}"


__all__ = ["NemoJob", "job_collection_path_for"]
