# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Signature-based dependency helpers for ``NemoJob.run``.

Both local scheduler execution and platform-launched task containers use
these helpers to bind supported keyword-only parameters on job ``run``
methods. Keeping this surface separate avoids making lightweight task
entrypoints import scheduler submission machinery.
"""

from __future__ import annotations

import inspect
from typing import Any

from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.job_context import JobContext

_UNBOUND: Any = object()
"""Sentinel meaning "leave the parameter unbound" — the kwarg is omitted so
Python applies the run signature's own default."""


class LocalRunError(RuntimeError):
    """Raised when a required ``sdk`` / ``async_sdk`` parameter on
    :meth:`NemoJob.run` has no handle to bind."""


def resolve_run_kwargs(
    job_cls: type[NemoJob],
    run: Any,
    *,
    sdk: object | None,
    async_sdk: object | None,
    ctx: JobContext,
    is_local: bool,
) -> dict[str, object]:
    """Bind keyword-only parameters on *run* by name.

    Recognises ``ctx``, ``sdk``, ``async_sdk``, and ``is_local`` on the
    keyword-only portion of *run*'s signature; everything else is left
    unbound so Python's own default applies. ``is_local`` lets jobs adapt
    behaviour to the execution context — ``True`` from the local scheduler,
    ``False`` from the platform task dispatcher.

    Raises:
        LocalRunError: When *run* declares a required ``sdk`` /
            ``async_sdk`` parameter with no default and the corresponding
            handle is not supplied.
    """
    try:
        sig = inspect.signature(run)
    except (TypeError, ValueError):
        # Builtins / C-extension callables have no inspectable
        # signature; fall back to calling with just the config dict.
        return {}

    # Drop the first positional — the caller passes ``config`` by position.
    params = list(sig.parameters.values())
    if params and params[0].kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        params = params[1:]

    resolved: dict[str, object] = {}
    for param in params:
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        binding = _resolve_run_param(
            job_cls=job_cls,
            param=param,
            sdk=sdk,
            async_sdk=async_sdk,
            ctx=ctx,
            is_local=is_local,
        )
        if binding is _UNBOUND:
            continue
        resolved[param.name] = binding
    return resolved


def _resolve_run_param(
    *,
    job_cls: type[NemoJob],
    param: inspect.Parameter,
    sdk: object | None,
    async_sdk: object | None,
    ctx: JobContext,
    is_local: bool,
) -> object:
    """Resolve a single ``run`` parameter or return :data:`_UNBOUND`."""
    required = param.default is inspect.Parameter.empty

    if param.name == "ctx":
        return ctx

    if param.name == "is_local":
        return is_local

    if param.name == "sdk":
        if sdk is not None:
            return sdk
        if required:
            raise LocalRunError(
                f"{job_cls.__name__}.run requires a `sdk` argument; "
                f"pass it via NemoJobScheduler.run_local(sdk=...) or "
                f"nemo_helix_plugin.tasks.dispatcher.run_task(sdk=...)."
            )
        return _UNBOUND

    if param.name == "async_sdk":
        if async_sdk is not None:
            return async_sdk
        if required:
            raise LocalRunError(
                f"{job_cls.__name__}.run requires an `async_sdk` "
                f"argument; pass it via "
                f"NemoJobScheduler.run_local(async_sdk=...) or "
                f"nemo_helix_plugin.tasks.dispatcher.run_task(async_sdk=...)."
            )
        return _UNBOUND

    if required:
        # Surface as LocalRunError instead of a downstream TypeError.
        raise LocalRunError(
            f"{job_cls.__name__}.run declares unsupported required parameter "
            f"`{param.name}`; only `ctx`, `sdk`, `async_sdk`, and `is_local` "
            "are injected automatically."
        )

    return _UNBOUND


__all__ = ["LocalRunError", "resolve_run_kwargs"]
