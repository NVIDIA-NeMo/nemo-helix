# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared CLI completion renderer for customization ``submit``.

All three customization backends (``automodel``, ``unsloth``, ``rl``) submit
through the same :class:`~nmp.customization_common.contributor.base.BaseContributor`
CLI path, so they share one completion renderer: on a successful submit it always
prints CLI tracking instructions, and — when Studio is deployed on the same
platform — a deep link to the job's Studio customization details page
(``{base}/studio/workspaces/{workspace}/customizations/{jobName}``).

Studio availability is detected by calling the platform ``/status`` endpoint and
checking whether ``studio`` is among the ready services. This is the platform's
own service-status API (not an ad-hoc HTML probe of ``/studio/``), so it reports
disabled/undeployed Studio precisely. The link is omitted whenever Studio is not
ready, no base URL resolved, or the created job name can't be read from the
submit response — the CLI tracking instructions are always shown regardless.

The renderer only runs in TTY mode: the framework bypasses renderers entirely
under the global ``--output-format json`` flag, so automation keeps the stable
raw-JSON submit contract.
"""

from __future__ import annotations

import json
import shlex
from typing import Any
from urllib.parse import quote

import httpx
import typer
from nemo_platform_plugin.cli_renderer import CLIRenderer, RendererContext
from rich.console import Console

#: Path prefix Studio's SPA is mounted under on the platform origin (basename
#: ``/studio``, path-based router — no hash). See ``web/packages/studio/src/App.tsx``.
_STUDIO_PATH_PREFIX = "studio"
#: Studio's registered platform-service name (``platform_runner/registry.py``);
#: its presence in ``/status`` ``services.ready`` means Studio is deployed and up.
_STUDIO_SERVICE_NAME = "studio"
#: Timeout (seconds) for the ``/status`` availability probe. Kept short so a slow
#: or wedged platform never stalls submit-completion output for long.
_STATUS_PROBE_TIMEOUT = 5.0


def build_studio_customization_url(base_url: str | None, workspace: str, job_name: str) -> str | None:
    """Build the Studio customization job-details deep link, or ``None``.

    Returns ``None`` when *base_url* is missing/blank, so callers can cleanly
    omit the link. The ``workspace`` and ``job_name`` path segments are
    percent-encoded (``safe=""`` so ``/`` in a name is escaped too), mirroring
    ``nmp.studio.studio_links``.
    """
    trimmed = base_url.strip().rstrip("/") if base_url else ""
    if not trimmed:
        return None
    workspace_segment = quote(workspace, safe="")
    job_segment = quote(job_name, safe="")
    return f"{trimmed}/{_STUDIO_PATH_PREFIX}/workspaces/{workspace_segment}/customizations/{job_segment}"


def studio_is_available(base_url: str | None, *, timeout: float = _STATUS_PROBE_TIMEOUT) -> bool:
    """Return whether Studio is a ready service on the platform at *base_url*.

    Calls the platform ``/status`` endpoint and checks for ``studio`` in
    ``services.ready``. Any resolution/transport/parse failure (no base URL,
    unreachable platform, unexpected payload) is treated as "not available" so
    the deep link is simply omitted rather than raising during submit output.
    """
    trimmed = base_url.strip().rstrip("/") if base_url else ""
    if not trimmed:
        return False
    try:
        response = httpx.get(f"{trimmed}/status", timeout=timeout, follow_redirects=True)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, httpx.InvalidURL, ValueError):
        return False
    services = payload.get("services") if isinstance(payload, dict) else None
    ready = services.get("ready") if isinstance(services, dict) else None
    return isinstance(ready, list) and _STUDIO_SERVICE_NAME in ready


def _job_name_from_result(result: Any) -> str | None:
    """Read the created job name from a submit response dict.

    The customization create route returns a job envelope whose ``name`` is the
    resolved (user-provided or generated) job name — the identifier the Studio
    details route keys on. Returns ``None`` when the response isn't a dict or
    carries no usable name, so the link is omitted rather than pointing at a
    wrong/empty job.
    """
    if not isinstance(result, dict):
        return None
    name = result.get("name")
    if isinstance(name, str) and name.strip():
        return name
    return None


class CustomizationSubmitRenderer(CLIRenderer):
    """Completion renderer for ``nemo customization <backend> submit``.

    Captures the submit response (delivered once via :meth:`on_frame`), then on
    :meth:`on_complete` echoes the raw response as JSON to **stdout** (byte-for-byte
    the prior default submit output, so automation that parses ``json.load(stdout)``
    is unaffected), and writes human-facing guidance — CLI tracking commands and,
    when Studio is a ready service, the Studio deep link — to **stderr** so it never
    contaminates the stdout JSON.
    """

    def __init__(self) -> None:
        self._result: Any = None

    def on_frame(self, frame: Any, *, ctx: RendererContext) -> None:
        # Jobs are non-streaming: on_frame fires exactly once with the submit
        # response dict. Stash it for on_complete.
        self._result = frame

    def on_complete(self, *, ctx: RendererContext) -> None:
        result = self._result

        # stdout stays PURE JSON — identical to the prior default submit echo — so
        # documented automation (`json.load` on submit stdout to read the job name)
        # keeps working. Use typer.echo, not the Rich console, so no markup parsing
        # can mangle a response string that happens to contain '[...]'.
        if result is not None:
            typer.echo(json.dumps(result, indent=2))

        # All human guidance goes to STDERR (a separate Rich console) — never stdout.
        # markup=False so a dynamic value (workspace / job name / URL) that happens to
        # contain '[...]' is never interpreted as Rich markup (addresses the review nit).
        err = Console(stderr=True, markup=False)
        job_name = _job_name_from_result(result)

        workspace = ctx.cli_kwargs.get("workspace")
        if not isinstance(workspace, str) or not workspace:
            workspace = "default"
        # shell-quote for the copy-pasteable command hints (a workspace may contain
        # spaces/special chars); the URL path segment is percent-encoded separately.
        workspace_flag = f" --workspace {shlex.quote(workspace)}"

        err.print()
        err.print("Job submitted. Track it with:")
        err.print(f"  nemo jobs list{workspace_flag}")
        if job_name is not None:
            err.print(f"  nemo jobs get-status {shlex.quote(job_name)}{workspace_flag}")

        if job_name is None:
            return
        if not studio_is_available(ctx.base_url):
            return
        url = build_studio_customization_url(ctx.base_url, workspace, job_name)
        if url is None:
            return
        err.print()
        err.print(f"View in Studio: {url}")
