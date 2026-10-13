# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for plugin CLI commands that need access to the CLI's state object.

The top-level ``nemo`` CLI populates ``typer.Context.obj`` with a state object
that exposes per-invocation handles to the platform SDK clients (sync and
async). The auto-generated ``run`` / ``submit`` verbs in
:mod:`nemo_helix_plugin.commands` consume that state through these helpers,
and **plugin-authored** Typer commands (i.e. anything a plugin registers via
:meth:`~nemo_helix_plugin.cli.NemoCLI.get_cli` rather than the
auto-generated verbs) should use the same surface so they participate in the
same protocol.

The state object's contract is :class:`CLIState`. The ``nemo`` CLI's
``CLIContext`` implements it; plugin commands depend on the protocol, never on
that concrete class.

Example::

    import typer
    from nemo_helix_plugin.cli_options import ListOutputFormatOption, WorkspaceOption
    from nemo_helix_plugin.cli_state import cli_state, resolve_cli_workspace, resolve_output_format

    def list_widgets(
        typer_ctx: typer.Context,
        workspace: WorkspaceOption = None,
        output_format: ListOutputFormatOption = None,
    ) -> None:
        state = cli_state(typer_ctx)
        resolved_output_format = resolve_output_format(typer_ctx, output_format)
        client = state.typed_client(WidgetsClient)
        ...
"""

import logging
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Protocol, TypeVar, cast

import click
import httpx
import typer
from nemo_helix_plugin.cli_options import ListOutputFormat, TimestampFormat
from nemo_helix_plugin.cli_output import is_tty
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.entities import DEFAULT_WORKSPACE

logger = logging.getLogger(__name__)

TypedClientT = TypeVar("TypedClientT", bound=NemoClient)
AsyncTypedClientT = TypeVar("AsyncTypedClientT", bound=AsyncNemoClient)
DEFAULT_BASE_URL = "http://localhost:8080"


class CLIState(Protocol):
    """The per-invocation state the ``nemo`` CLI stores on ``typer.Context.obj``.

    Clients from :meth:`typed_client` share the CLI's auth, transport, and base
    URL (including the global ``--base-url``), so commands never build clients
    from their own flags or config.
    """

    def get_client(self, timeout: float | httpx.Timeout | None = ...) -> NemoClient: ...

    def get_async_client(self, timeout: float | httpx.Timeout | None = ...) -> AsyncNemoClient: ...

    def typed_client(
        self, client_cls: type[TypedClientT], timeout: float | httpx.Timeout | None = ...
    ) -> TypedClientT: ...

    def async_typed_client(
        self,
        client_cls: type[AsyncTypedClientT],
        timeout: float | httpx.Timeout | None = ...,
    ) -> AsyncTypedClientT: ...

    def get_workspace(self) -> str | None: ...

    def get_base_url(self, default: str | None = None) -> str | None: ...

    def get_output_format(
        self,
        override: ListOutputFormat | None = None,
        *,
        apply_non_tty_default: bool = True,
    ) -> ListOutputFormat: ...

    def get_timestamp_format(self, override: TimestampFormat | None = None) -> TimestampFormat: ...

    def get_no_truncate(self, override: bool | None = None) -> bool: ...


def cli_state(typer_ctx: typer.Context) -> CLIState:
    """Return the ``nemo`` CLI state for *typer_ctx*.

    Raises :class:`RuntimeError` when no state is set, which means the command
    is running outside ``nemo`` (a test driving the Typer app directly must set
    ``obj=`` on the runner).
    """
    state = typer_ctx.obj
    if state is None:
        raise RuntimeError("No NeMo Helix CLI state on this context; run the command through `nemo`.")
    return cast(CLIState, state)


def current_cli_state() -> CLIState | None:
    """Return the ambient ``nemo`` CLI state from Click's current context."""
    ctx = click.get_current_context(silent=True)
    if ctx is None or ctx.obj is None:
        return None
    return cast(CLIState, ctx.obj)


def base_url_from_context() -> str | None:
    """Return the base URL configured in the ambient CLI context, if any."""
    state = current_cli_state()
    if state is None:
        return None
    try:
        return state.get_base_url(default=None)
    except Exception:
        logger.debug("Failed to resolve base URL from CLI context", exc_info=True)
        return None


def resolve_base_url() -> str:
    """Resolve and announce the platform base URL for plugin commands.

    The shared CLI context owns the real resolution order (global
    ``nemo --base-url``, environment, config, defaults). The target is echoed
    to stderr so command stdout remains parseable.
    """
    resolved = base_url_from_context() or DEFAULT_BASE_URL
    click.echo(f"Targeting {resolved}", err=True)
    return resolved


def shared_cli_client(
    client_cls: type[TypedClientT],
    *,
    timeout: float | httpx.Timeout | None = None,
) -> TypedClientT:
    """Return *client_cls* built on the ambient CLI state's shared platform client."""
    state = current_cli_state()
    if state is None:
        raise RuntimeError("No NeMo Helix CLI state on this context; run the command through `nemo`.")
    return state.typed_client(client_cls, timeout=timeout)


@contextmanager
def shared_cli_client_context(
    client_cls: type[TypedClientT],
    *,
    timeout: float | httpx.Timeout | None = None,
) -> Iterator[TypedClientT]:
    """Yield a CLI-owned shared client without taking ownership of its transport."""
    yield shared_cli_client(client_cls, timeout=timeout)


def resolve_context_headers(
    headers: Mapping[str, str] | None = None,
    *,
    url: str | None = None,
) -> dict[str, str] | None:
    """Resolve per-request headers from the ambient CLI state's shared client."""
    return shared_cli_client(NemoClient).request_headers(headers, url=url)


def resolve_output_format(typer_ctx: typer.Context, explicit: ListOutputFormat | None = None) -> ListOutputFormat:
    """Resolve the output format a command should render with.

    Resolution order, the same for core and plugin commands:

    1. The command's ``--output-format`` flag.
    2. The global ``nemo --output-format`` flag, then the context preference.
    3. A ``table`` result becomes ``json`` when stdout is not a TTY, so piped
       output is parseable.

    Steps 2-3 belong to the CLI state. Without one (a plugin app driven outside
    ``nemo``), the fallback is ``table`` on a TTY and ``json`` otherwise.
    """
    if explicit is not None:
        return explicit
    state = typer_ctx.obj
    if state is not None:
        return cast(CLIState, state).get_output_format()
    return "table" if is_tty() else "json"


def resolve_local_cli_sdks(
    typer_ctx: typer.Context,
) -> tuple[object | None, object | None]:
    """Pull ``(sdk, async_sdk)`` out of the CLI state object on ``typer_ctx.obj``.

    Returns ``(None, None)`` when no state object is set (e.g. plugin tests
    that exercise a Typer app directly without populating ``ctx.obj``). When
    a state object is set but does not implement one of the getter methods,
    that side returns ``None`` while the other still resolves — letting
    callers decide which handle they actually need.
    """
    state = typer_ctx.obj
    if not state:
        return None, None
    sdk = state.get_client() if hasattr(state, "get_client") else None
    async_sdk = state.get_async_client() if hasattr(state, "get_async_client") else None
    return sdk, async_sdk


def _workspace_from_state(state: Any) -> str | None:
    """Read the active workspace off a CLI state object, if it exposes one."""
    if state is None or not hasattr(state, "get_workspace"):
        return None
    try:
        resolved = state.get_workspace()
    except Exception:
        logger.debug("Failed to resolve workspace from CLI state", exc_info=True)
        return None
    return cast(str, resolved) if resolved else None


def resolve_cli_workspace(typer_ctx: typer.Context, explicit: str | None = None) -> str:
    """Resolve the workspace a command should act on, given an explicit context.

    An explicit ``--workspace`` wins. Otherwise the active CLI context
    decides, and it applies ``$NHX_WORKSPACE`` over its own configured
    workspace. If there is no usable context — no config file yet, or a
    plugin CLI driven outside ``nemo`` — fall back to ``$NHX_WORKSPACE``,
    then ``"default"``.

    Net user-visible order: ``--workspace`` > ``$NHX_WORKSPACE`` > the
    context's configured workspace > ``"default"``.

    The environment is therefore read in two places. That is deliberate, not
    redundant: with no config file on disk ``get_workspace()`` returns
    ``None``, and the lookup below is the only thing that honors
    ``$NHX_WORKSPACE`` — the documented way to pick a workspace in a fresh
    container or CI job.
    """
    if explicit is not None:
        return explicit
    return _workspace_from_state(typer_ctx.obj) or os.environ.get("NHX_WORKSPACE") or DEFAULT_WORKSPACE
