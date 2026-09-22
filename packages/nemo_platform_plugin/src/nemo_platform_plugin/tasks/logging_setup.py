# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logging bootstrap for platform-spawned task processes.

A task container runs its entrypoint directly (``python -m <task module>``),
so nothing configures the root logger the way a service's startup does. With
no root handler Python falls back to :data:`logging.lastResort`: WARNING and
above reach stderr as bare messages with no timestamp, level or logger name,
and INFO/DEBUG are dropped outright. A job that fails without raising can
therefore produce completely empty logs.

:func:`configure_task_logging` closes that gap for every typed task entrypoint
dispatched through :mod:`nemo_platform_plugin.tasks.dispatcher`.

Lookup order for the provider
-----------------------------

Mirrors :mod:`nemo_platform_plugin.client_provider` and
:mod:`nemo_platform_plugin.sdk_provider`, for the same reason: it lets the
platform supply a richer implementation without ``nemo-platform-plugin`` ever
depending on ``nmp-common``.

1. **Explicit override** - set via :func:`set_task_logging_provider` (tests).
2. **Entry-point discovery** - scans the ``nemo.logging_provider`` group.
   When ``nmp-common`` is installed (every platform image), its provider is
   picked up and task output becomes the same structured stream the services
   emit, so log aggregation treats the two alike.
3. **Built-in default** - :class:`DefaultTaskLoggingProvider`, a plain stderr
   handler. Covers local development and any image without ``nmp-common``.

Unlike the client and SDK providers, a provider that fails to load here is
downgraded to the default rather than raised: logging is diagnostic
scaffolding, and failing a task because its logging setup broke would destroy
the very evidence needed to debug it.
"""

from __future__ import annotations

import logging
from typing import Literal, Protocol, runtime_checkable

from nemo_platform_plugin.discovery import discover_entry_points

#: The platform's configured logging vocabulary, mirroring
#: ``CommonServiceConfig`` so a value read from settings can be handed
#: straight to a provider without a cast.
LogLevel = Literal["DEBUG", "INFO", "WARN", "ERROR"]
LogFormat = Literal["json", "plain"]

#: Entry-point group a platform-side logging provider registers under.
TASK_LOGGING_PROVIDER_GROUP = "nemo.logging_provider"

#: Formatter used by the built-in default. Deliberately carries the fields
#: ``lastResort`` drops - timestamp, level and logger name - since that is what
#: makes a task log readable at all.
FALLBACK_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

#: Transport libraries that log a line per request at INFO. Clamped so their
#: output cannot drown the job's own messages.
_NOISY_LOGGERS = ("httpx", "httpcore", "asyncio")

logger = logging.getLogger(__name__)

_provider_override: TaskLoggingProvider | None = None


@runtime_checkable
class TaskLoggingProvider(Protocol):
    """Configures root logging for a task process."""

    def configure_logging(self, *, level: LogLevel, log_format: LogFormat) -> None: ...


class DefaultTaskLoggingProvider:
    """Built-in provider: a plain stderr handler on the root logger."""

    def configure_logging(self, *, level: LogLevel, log_format: LogFormat) -> None:
        # ``log_format`` is part of the provider contract but the default has
        # only one rendering; a platform provider is what honours "json".
        del log_format
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(FALLBACK_LOG_FORMAT))
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            root.setLevel(level)
        except ValueError:
            root.setLevel(logging.INFO)
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)


def set_task_logging_provider(provider: TaskLoggingProvider | None) -> None:
    """Override the provider, or restore discovery by passing ``None``."""
    global _provider_override
    _provider_override = provider


def configure_task_logging() -> None:
    """Configure root logging for a task process, if nothing else has.

    A no-op when the root logger already has handlers, so a caller that
    configured logging itself keeps ownership of it: a service that dispatches
    in-process, the ``nemo-platform run task`` entrypoint, or a test harness.
    Safe to call more than once.
    """
    if logging.getLogger().handlers:
        return

    level, log_format = _resolve_log_config()
    provider = _resolve_provider()
    try:
        provider.configure_logging(level=level, log_format=log_format)
    except Exception:
        # A provider that raised part-way through can leave a configuration that
        # looks present but does not work - a handler attached before it got to
        # setting the root level, say, which would strand the task at WARNING.
        # The early return above guarantees the root logger was bare when we
        # called it, so anything here now came from the failed call: discard it
        # rather than trust half of it.
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        DefaultTaskLoggingProvider().configure_logging(level=level, log_format=log_format)
        # Logged only once the fallback is in place. Reporting the failure
        # first would emit it into the broken configuration we are replacing.
        logger.warning("Task logging provider failed; using the default.", exc_info=True)


def _resolve_log_config() -> tuple[LogLevel, LogFormat]:
    """Read the level/format the platform is configured with.

    Going through ``Configuration`` merges the platform config file when one
    is reachable (local and subprocess runs, and any container where it is mounted),
    and ``EnvironmentFirstSettings`` still lets the environment win - which is what
    carries the setting into containers that have no config file, since the
    jobs backends inject the platform's resolved ``LOG_LEVEL``/``LOG_FORMAT``
    into every task.

    Falls back to the field defaults rather than failing: a task with misconfigured
    settings should still log.
    """
    try:
        from nemo_platform_plugin.config import CommonServiceConfig, Configuration

        service_config = Configuration.get_service_config(CommonServiceConfig)
    except Exception:
        return "INFO", "plain"
    return service_config.log_level, service_config.log_format


def _resolve_provider() -> TaskLoggingProvider:
    """Resolve the provider: explicit override -> entry-point -> default."""
    if _provider_override is not None:
        return _provider_override

    try:
        entry_points = discover_entry_points(TASK_LOGGING_PROVIDER_GROUP)
    except Exception:
        logger.warning("Failed to scan for task logging providers.", exc_info=True)
        return DefaultTaskLoggingProvider()

    # The nemo-platform bundle inherits nmp-common's entry points, so the same
    # provider legitimately appears under one name from two distributions; a
    # dict keyed by name already collapses that. More than one *name* means a
    # second package registered its own, which is not something to guess at.
    if len(entry_points) > 1:
        logger.warning(
            "Multiple task logging providers registered under %r: %s. Using the default.",
            TASK_LOGGING_PROVIDER_GROUP,
            ", ".join(sorted(entry_points)),
        )
        return DefaultTaskLoggingProvider()

    for entry_point in entry_points.values():
        try:
            provider = entry_point.load()
            if isinstance(provider, type):
                provider = provider()
        except Exception:
            logger.warning(
                "Failed to load task logging provider %r from %r; using the default.",
                entry_point.name,
                entry_point.value,
                exc_info=True,
            )
            break
        if not isinstance(provider, TaskLoggingProvider):
            logger.warning(
                "Task logging provider %r does not implement configure_logging; using the default.",
                entry_point.name,
            )
            break
        return provider

    return DefaultTaskLoggingProvider()


__all__ = [
    "FALLBACK_LOG_FORMAT",
    "TASK_LOGGING_PROVIDER_GROUP",
    "DefaultTaskLoggingProvider",
    "LogFormat",
    "LogLevel",
    "TaskLoggingProvider",
    "configure_task_logging",
    "set_task_logging_provider",
]
