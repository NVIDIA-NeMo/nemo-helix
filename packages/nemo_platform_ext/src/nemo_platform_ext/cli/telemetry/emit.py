# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fire-and-flush emission with three opt-out layers and the first-run notice."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from nemo_platform_ext.cli.telemetry.events import PlatformTelemetryEvent
from nemo_platform_ext.cli.telemetry.handler import TelemetryHandler, _telemetry_enabled
from nemo_platform_ext.cli.telemetry.session import get_session_id

logger = logging.getLogger(__name__)
_TELEMETRY_SUPPRESSED_EXCEPTIONS: tuple[type[BaseException], ...] = (
    Exception,
    KeyboardInterrupt,
    asyncio.CancelledError,
)

_invocation_opt_out = False

_NOTICE_TEXT = (
    "NeMo Platform CLI telemetry is on by default and sends anonymous usage data to improve the product. "
    "It does not send prompts, model inputs or outputs, datasets, secrets, file contents, or personal identifiers. "
    "Turn it off with NEMO_TELEMETRY_ENABLED=false or telemetry_enabled: false in the CLI config. "
    "Run nemo docs cli/configuration for details.\n"
)


def set_invocation_opt_out(value: bool) -> None:
    """Per-invocation opt-out (e.g. a --no-telemetry flag on the current command)."""
    global _invocation_opt_out
    _invocation_opt_out = value


def _config_opted_out() -> bool:
    """True when the persisted config file sets ``telemetry_enabled: false``."""
    try:
        from nemo_platform_ext.config.config import Config

        cfg = Config.load()
        return cfg.get_config_file().telemetry_enabled is False
    except Exception:
        # A privacy control must fail closed: if we cannot read the config to confirm
        # the user is opted in, treat them as opted out and do not send.
        logger.debug("Could not read telemetry opt-out config; failing closed (opted out)", exc_info=True)
        return True


def telemetry_opted_in() -> bool:
    """Opted in only when all three layers agree: per-invocation, env, and config."""
    if _invocation_opt_out:
        return False
    if not _telemetry_enabled():
        return False
    return not _config_opted_out()


def _client_version() -> str:
    try:
        from nemo_platform_ext.cli.version import UNKNOWN_VERSION, client_version

        resolved = client_version()
        return "undefined" if resolved == UNKNOWN_VERSION else resolved
    except Exception:
        logger.debug("Could not resolve client version for telemetry", exc_info=True)
        return "undefined"


def emit_event(event: PlatformTelemetryEvent) -> None:
    """Best effort. Telemetry must never break a user command."""
    try:
        if not telemetry_opted_in():
            return
        # No retries on the CLI exit path: a synchronous send blocks the user's command,
        # so cap the worst case at one bounded send (SEND_TIMEOUT_SECONDS) rather than
        # retrying against a slow or unreachable endpoint while the user waits.
        handler = TelemetryHandler(source_client_version=_client_version(), session_id=get_session_id(), max_retries=0)
        handler.enqueue(event)
        handler.stop()
    except _TELEMETRY_SUPPRESSED_EXCEPTIONS:  # noqa: BLE001
        logger.debug("Failed to emit telemetry event", exc_info=True)


def _notice_marker_path() -> Path:
    from nemo_platform_ext.config.config import Config

    return Config.get_default_config_path().parent / "telemetry-notice-shown"


def maybe_print_first_run_notice() -> None:
    """Print the first-run notice to stderr once. Stdout stays machine-clean."""
    try:
        if not telemetry_opted_in():
            return
        marker = _notice_marker_path()
        if marker.exists():
            return
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        sys.stderr.write(_NOTICE_TEXT)
    except Exception:
        logger.debug("Failed to print telemetry notice", exc_info=True)
