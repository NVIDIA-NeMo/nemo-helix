# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NemoJobScheduler — remote submit and explain support for plugin jobs."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from nemo_platform_plugin.job import NemoJob, job_collection_path_for

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class NemoJobScheduler:
    """Remote submit and explain helper for :class:`NemoJob` subclasses.

    Construction-cheap — instantiate per CLI invocation. All state is
    derived from the :class:`NemoJob` subclass and the submitter-provided
    spec.
    """

    def submit_remote(
        self,
        job_cls: type["NemoJob"],
        spec: dict,
        *,
        base_url: str | None = None,
        workspace: str = "default",
        profile: str | None = None,
        options: dict | None = None,
        metadata: dict | None = None,
        http_client: httpx.Client | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> dict:
        """POST the job to the plugin service's per-job submit route.

        Builds the submit URL from
        :attr:`~nemo_platform_plugin.job.NemoJob.job_collection_path` and POSTs a JSON body of
        ``{**metadata, "spec": ..., "profile": ..., "options": ...}``.

        Args:
            job_cls: The :class:`~nemo_platform_plugin.job.NemoJob` subclass.
            spec: Submitter input (validated server-side against
                ``input_spec_schema`` / ``spec_schema``).
            base_url: Final plugin-service base URL
                (e.g. ``"https://nmp.dev.example"``).
            workspace: Workspace scope for the submit URL.
            profile: Profile label forwarded to the server.
            options: Opaque wire ``{"<backend>": {...}}`` bag.
            metadata: Optional envelope fields (``name``, ``description``,
                ``project``, ``ownership``, ``custom_fields``).
            http_client: Optional injected :class:`httpx.Client`. Defaults
                to a short-lived client per call; tests supply a mock
                transport.
            headers: Optional per-request headers (e.g. ``Authorization`` from
                the CLI). Merged on each POST; not inferred from *http_client*.
            timeout: Request timeout in seconds.

        Returns:
            The decoded JSON response from the plugin service.

        Raises:
            ValueError: If no ``base_url`` is provided.
            httpx.HTTPStatusError: On 4xx / 5xx responses.
        """
        url = self._build_submit_url(job_cls, base_url=base_url, workspace=workspace)
        body = self._build_submit_body(spec, profile=profile, options=options, metadata=metadata)
        return self._post_submit(url, body, http_client=http_client, headers=headers, timeout=timeout)

    # ------------------------------------------------------------------ #
    # Schema discovery                                                   #
    # ------------------------------------------------------------------ #

    def explain(
        self,
        job_cls: type["NemoJob"],
        *,
        profile: str | None = None,
    ) -> dict:
        """Return a schema bundle for *job_cls* — no network.

        The CLI holds a live reference to ``job_cls``, so
        ``spec_schema`` / ``input_spec_schema`` JSON Schemas are read
        directly via :meth:`model_json_schema` without touching the
        plugin service.

        Args:
            job_cls: The :class:`~nemo_platform_plugin.job.NemoJob` subclass.
            profile: Profile label to annotate the bundle with.

        Returns:
            Dict with keys ``job_key``, ``endpoint``, ``spec_schema``,
            ``input_spec_schema``, ``profile``, ``profile_providers``,
            ``options``. ``endpoint`` is a URL path template with
            ``{workspace}`` left as a literal placeholder.
            ``profile_providers`` / ``options`` are currently empty —
            they're populated once the Jobs service endpoints for
            operator-configured profiles and backend options are wired.
        """
        spec_schema = job_cls.spec_schema.model_json_schema() if job_cls.spec_schema else None
        input_spec_schema = job_cls.input_spec_schema.model_json_schema() if job_cls.input_spec_schema else None

        return {
            "job_key": _job_key_for(job_cls),
            "endpoint": submit_path_for(job_cls, workspace="{workspace}"),
            "spec_schema": spec_schema,
            "input_spec_schema": input_spec_schema,
            "profile": profile,
            "profile_providers": [],
            "options": {},
        }

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    # ---- submit_remote helpers ------------------------------------- #

    def _build_submit_url(
        self,
        job_cls: type["NemoJob"],
        *,
        base_url: str | None,
        workspace: str,
    ) -> str:
        """Construct the full submit URL for *job_cls*.

        Uses ``job_cls.job_collection_path`` or the default
        ``/jobs/{job_cls.name}`` collection path.
        Base URL must already be resolved by the CLI layer.
        """
        endpoint = submit_path_for(job_cls, workspace=workspace)
        if not base_url:
            raise ValueError(
                "submit_remote requires base_url; the CLI should resolve cluster and env fallbacks before calling it."
            )
        return f"{base_url.rstrip('/')}{endpoint}"

    def _build_submit_body(
        self,
        spec: dict,
        *,
        profile: str | None,
        options: dict | None,
        metadata: dict | None,
    ) -> dict[str, Any]:
        """Assemble the POST body ``{**metadata, "spec", "profile", "options"}``."""
        body: dict[str, Any] = {}
        if metadata:
            body.update(metadata)
        body["spec"] = spec
        if profile is not None:
            body["profile"] = profile
        if options:
            body["options"] = options
        return body

    def _post_submit(
        self,
        url: str,
        body: dict[str, Any],
        *,
        http_client: httpx.Client | None,
        headers: dict[str, str] | None,
        timeout: float,
    ) -> dict:
        """POST *body* to *url* and return the decoded JSON response.

        Uses *http_client* when provided; otherwise opens a short-lived
        client per call.
        """
        request_headers = dict(headers) if headers else None
        logger.debug("submit_remote POST %s", url)
        if http_client is not None:
            response = http_client.post(url, json=body, headers=request_headers, timeout=timeout)
        else:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, json=body, headers=request_headers)
        response.raise_for_status()
        return response.json()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _job_key_for(job_cls: type["NemoJob"]) -> str:
    """Best-effort entry-point key for *job_cls*.

    Uses the API segment (top-level package, ``nemo_`` prefix stripped, hyphenated)
    as a fallback when no registered entry point is available — e.g. a job in
    ``nemo_data_designer.jobs.generate`` keys as ``data-designer.<job.name>``,
    matching the documented ``<plugin>.<job>`` convention.
    """
    return f"{_api_segment_for(job_cls)}.{job_cls.name}"


def submit_path_for(job_cls: type["NemoJob"], *, workspace: str) -> str:
    """The submit URL path (no host / scheme) for *job_cls*.

    Useful for OpenAPI path lookups and explain bundles; mirrors
    :meth:`NemoJobScheduler._build_submit_url` without the host.
    """
    return f"/apis/{_api_segment_for(job_cls)}/v2/workspaces/{workspace}{job_collection_path_for(job_cls)}"


def _api_segment_for(job_cls: type["NemoJob"]) -> str:
    """Derive the ``{api}`` segment of the submit URL for *job_cls*.

    Jobs register under the ``nemo.jobs`` entry-point group keyed as
    ``<plugin>.<job>``, and the platform mounts their routes under that
    ``<plugin>`` segment. The authoritative source of truth is therefore
    the entry-point key — not the Python module path. We resolve it via
    :func:`~nemo_platform_plugin.discovery.discover_jobs` so a plugin registered
    as ``agents.evaluate`` correctly maps to ``/apis/agents/...``, even
    though its package directory is ``nemo_agents_plugin/`` (which the
    module-name heuristic would have collapsed to ``agents-plugin`` and
    404'd against).

    When the job class isn't installed as an entry point — unit tests
    with inline classes, ad-hoc invocations from a checkout — fall back
    to deriving the segment from the top-level package name with the
    ``nemo_`` prefix stripped and underscores converted to dashes.
    The fallback does *not* strip a trailing ``_plugin``: a plugin whose
    actual key happens to be ``my-plugin`` would be silently rewritten
    to ``my``.
    """
    from nemo_platform_plugin.discovery import discover_jobs

    try:
        registered = discover_jobs()
    except Exception:
        registered = {}
    for key, registered_cls in registered.items():
        if registered_cls is job_cls and "." in key:
            return key.split(".", 1)[0]

    module = job_cls.__module__.split(".")[0]
    if module.startswith("nemo_"):
        module = module[len("nemo_") :]
    return module.replace("_", "-")
