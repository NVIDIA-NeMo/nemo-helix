# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform task-logging provider.

Registered under the ``nemo.logging_provider`` entry-point group so
:func:`nemo_helix_plugin.tasks.logging_setup.configure_task_logging` picks
it up in any image where ``nhx-common`` is installed. This is the
:mod:`nhx.common.client_factory` pattern applied to logging: the plugin
package declares the seam and keeps no dependency on ``nhx-common``, while the
platform supplies the real implementation.

The effect is that a task container's output is the same structured stream a
service emits - same renderer, same level handling, same sensitive-data
filters - so log aggregation does not have to treat task logs as a special
case. Without this provider the plugin's built-in default gives a task plain
stderr lines, which is readable but unstructured.
"""

from __future__ import annotations

from typing import Literal

from nhx.common.observability.otel import settings
from nhx.common.observability.structured_logging import initialize_logging


class PlatformTaskLoggingProvider:
    """Configure a task process with the platform's structured logging."""

    def configure_logging(
        self,
        *,
        level: Literal["DEBUG", "INFO", "WARN", "ERROR"],
        log_format: Literal["json", "plain"],
    ) -> None:
        # Sync the OTEL settings before initializing, mirroring what the
        # ``nemo-helix run task`` entrypoint does, so a task honours the
        # deployment's configured level and format.
        settings.log_level = level
        settings.log_format = log_format
        # Only the logging half of observability. Tracing and metrics export
        # are a separate concern and are not something a logging bootstrap
        # should switch on behind the caller's back.
        initialize_logging()
