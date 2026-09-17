# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin-supplied renderers for streamed CLI output.

A plugin's :class:`~nemo_platform_plugin.cli.NemoCLI` can return a :class:`CLIRenderer`
subclass from :meth:`~nemo_platform_plugin.cli.NemoCLI.get_function_renderer` /
:meth:`~nemo_platform_plugin.cli.NemoCLI.get_job_renderer`. When supplied, the framework
drives the renderer's lifecycle around the verb's streamed output instead of
echoing each NDJSON frame as JSON to stdout.

Lifecycle:

- :meth:`CLIRenderer.on_start` — called once before iteration begins.
- :meth:`CLIRenderer.on_frame` — called once per frame as it arrives. For
  local ``run`` the frame is a :class:`~pydantic.BaseModel` (the value yielded
  by :meth:`~nemo_platform_plugin.function.NemoFunction.run`). For remote ``submit`` the
  frame is a ``dict`` (parsed JSON for one NDJSON line). Plugin renderers that
  want typed frames are responsible for parsing the dict themselves.
- :meth:`CLIRenderer.on_complete` — called once after the stream closes
  cleanly.
- :meth:`CLIRenderer.on_error` — called once when iteration raises (and the
  exception still propagates). Mutually exclusive with ``on_complete``.

Subclasses override only the methods they care about. Defaults are no-ops.

The framework bypasses the renderer entirely when the user has set the global
``--output-format json`` flag, falling back to the default per-frame JSON
echo. Renderers focus on TTY UX; automation gets a stable NDJSON contract.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from rich.console import Console


@dataclass
class RendererContext:
    """Per-invocation context passed to every :class:`CLIRenderer` method.

    Attributes:
        console: Shared Rich :class:`~rich.console.Console` for stdout writes
            so renderers don't fight for the terminal.
        cli_kwargs: The original CLI kwargs the verb was invoked with.
            Renderers can branch on flags here (e.g. ``--non-interactive``).
        verb: ``"run"`` (in-process) or ``"submit"`` (HTTP). Lets one renderer
            class drive both verbs while branching when the verbs need
            different output (e.g. show a request id only for submit).
        is_local: ``True`` for in-process invocation (``run`` from a CLI with
            local SDKs), ``False`` for HTTP ``submit``. Distinct from ``verb``
            because future plugin types could blur the in-process/remote line.
        base_url: The resolved platform/plugin-service base URL the verb ran
            against (``submit`` only), or ``None`` when no base URL applies
            (e.g. local ``run``) or none could be resolved. Renderers that
            build deep links (e.g. a Studio job-detail URL) read it from here;
            it is deliberately null-safe so existing renderers ignore it.
    """

    console: Console
    cli_kwargs: Mapping[str, Any]
    verb: Literal["run", "submit"]
    is_local: bool
    base_url: str | None = None


class CLIRenderer(ABC):
    """Plugin-supplied renderer for a function/job verb's streamed output.

    Subclasses override only the methods they need; defaults are no-ops, so a
    renderer that only cares about the post-stream summary just overrides
    :meth:`on_complete` and ignores the per-frame stream.
    """

    def on_start(self, *, ctx: RendererContext) -> None:
        """Called once before iteration begins. Default: no-op."""

    def on_frame(self, frame: Any, *, ctx: RendererContext) -> None:
        """Called once per frame as it arrives. Default: no-op.

        ``frame`` is a :class:`~pydantic.BaseModel` for local ``run`` and a
        ``dict`` for remote ``submit``. Renderers that want typed frames in
        both cases parse the dict themselves.
        """

    def on_complete(self, *, ctx: RendererContext) -> None:
        """Called once after the stream closes cleanly. Default: no-op."""

    def on_error(self, error: BaseException, *, ctx: RendererContext) -> None:
        """Called once when iteration raises. Default: no-op.

        The exception still propagates after this returns; ``on_error`` is for
        rendering, not for swallowing errors.
        """
